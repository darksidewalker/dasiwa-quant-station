import json
import math
import os
from collections import defaultdict
from contextlib import ExitStack
from typing import Any, Dict, Iterable, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from core.adapter_factorization import factorize_lora, factorize_lokr
from core.consensus_merge import merge_consensus_rows, resolve_consensus_preset
from utils.lora_inspector import discover_diff_patches, discover_lora_pairs, read_safetensors_manifest
from core.lora_merge_engine import _get_profile


MAX_EFFECTIVE_LORA_STRENGTH = 3.0


def _log(text: str) -> Dict[str, str]:
    return {"type": "log", "text": text}


def _status(status: str) -> Dict[str, str]:
    return {"type": "status", "status": status}


def _canonical_base(name: str) -> str:
    for prefix in ("base_model.model.", "model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    if not name.startswith("diffusion_model."):
        name = "diffusion_model." + name
    return name


def _canonical_candidates(candidates: Tuple[str, ...], fallback: str) -> str:
    normalized = []
    for candidate in candidates:
        value = candidate[:-len(".weight")] if candidate.endswith(".weight") else candidate
        canonical = _canonical_base(value)
        normalized.append(canonical)
        if not value.startswith("lora_unet_") and "base_model." not in value:
            return canonical
    return normalized[-1] if normalized else _canonical_base(fallback)


def _alpha_scale(handle: Any, key: str | None, rank: int, kind: str) -> float:
    if kind == "lokr" or not key:
        return 1.0
    value = float(handle.get_tensor(key).reshape(-1)[0].item())
    return value / max(rank, 1) if math.isfinite(value) else 1.0


def _full_delta(handle: Any, item: Dict[str, Any], device: str) -> torch.Tensor:
    if item["kind"] == "diff":
        return handle.get_tensor(item["diff_key"]).to(device=device, dtype=torch.float32) * item["scale"]
    first = handle.get_tensor(item["down_key"]).to(device=device, dtype=torch.float32)
    second = handle.get_tensor(item["up_key"]).to(device=device, dtype=torch.float32)
    if item["kind"] == "lokr":
        delta = torch.kron(first, second)
    else:
        delta = second @ first
    return delta * item["scale"] * _alpha_scale(handle, item.get("alpha_key"), item.get("rank", 1), item["kind"])


def _device(payload: Dict[str, Any], estimated_bytes: int = 0) -> str:
    requested = str(payload.get("merge_device") or "auto").lower()
    cuda = str(payload.get("cuda_device") or "cuda:0")
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("merge_device must be auto, cpu, or cuda")
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        try:
            torch.empty(0, device=cuda)
            free_bytes, _ = torch.cuda.mem_get_info(cuda)
            reserve = int(payload.get("vram_headroom_mb") or 1024) * 1024 * 1024
            if free_bytes >= estimated_bytes + reserve:
                return cuda
            if requested == "cuda":
                raise ValueError("insufficient CUDA VRAM for composition and requested headroom")
        except (RuntimeError, ValueError) as exc:
            if requested == "cuda":
                raise ValueError(f"CUDA device {cuda!r} is unavailable or lacks headroom") from exc
    if requested == "cuda":
        raise ValueError(f"CUDA device {cuda!r} is unavailable")
    return "cpu"


def run_lora_compose(payload: Dict[str, Any]) -> Iterable[Dict[str, str]]:
    specs = payload.get("loras") or []
    if len(specs) < 2:
        raise ValueError("LoRA composition requires at least two adapters")
    architecture = payload.get("architecture") or "LTX-2.3"
    settings = resolve_consensus_preset(payload.get("consensus_preset"), architecture)
    output_kind = str(payload.get("output_adapter") or "auto").lower()
    if output_kind not in {"auto", "lora", "lokr"}:
        raise ValueError("output_adapter must be auto, lora, or lokr")
    max_rank = int(payload.get("output_rank") or 0)
    energy = float(payload.get("frobenius_energy") or 0.99)
    if max_rank < 0 or not 0.0 < energy <= 1.0:
        raise ValueError("output_rank must be non-negative and frobenius_energy must be in (0, 1]")
    mismatch = str(payload.get("mismatch_mode") or "error").lower()
    if mismatch not in {"error", "skip"}:
        raise ValueError("mismatch_mode must be error or skip")
    dry_run = bool(payload.get("dry_run", False))
    output_path = os.path.realpath(os.path.expanduser(payload.get("output_path") or "merged_lora.safetensors"))
    if not output_path.endswith(".safetensors"):
        output_path += ".safetensors"

    layers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    manifests = {}
    _, classify_key, strategy_multiplier = _get_profile(architecture)
    for index, spec in enumerate(specs):
        path = os.path.realpath(os.path.expanduser(spec["path"]))
        strength = float(spec.get("strength", 1.0))
        global_strength = float(1.0 if payload.get("global_strength") is None else payload["global_strength"])
        effective = strength * global_strength
        if abs(effective) > MAX_EFFECTIVE_LORA_STRENGTH:
            raise ValueError(f"{os.path.basename(path)} effective strength {effective:g} exceeds safe limit ±3")
        manifest = read_safetensors_manifest(path)
        manifests[path] = manifest
        for pair in discover_lora_pairs(manifest):
            if pair.kind == "lokr_decomposed":
                if mismatch == "error":
                    raise ValueError(f"factorized LoKr input is unsupported: {pair.base_name}")
                continue
            logical = _canonical_candidates(pair.target_candidates, pair.base_name)
            layer_scale = effective * strategy_multiplier(spec.get("strategy") or ("All" if architecture == "LTX-2.3" else "Balanced"), classify_key(logical))
            shape = ((pair.down_shape[0] * pair.up_shape[0], pair.down_shape[1] * pair.up_shape[1])
                     if pair.kind == "lokr" else (pair.up_shape[0], pair.down_shape[1]))
            layers[logical].append({
                "path": path, "source": index, "kind": pair.kind, "down_key": pair.down_key,
                "up_key": pair.up_key, "alpha_key": pair.alpha_key, "rank": pair.rank,
                "shape": tuple(shape), "scale": layer_scale,
                "lokr_shapes": (tuple(pair.down_shape), tuple(pair.up_shape)) if pair.kind == "lokr" else None,
            })
        for patch in discover_diff_patches(manifest):
            logical = _canonical_candidates(patch.target_candidates, patch.diff_key[:-len(".diff")])
            layer_scale = effective
            layers[logical].append({"path": path, "source": index, "kind": "diff", "diff_key": patch.diff_key,
                                    "shape": tuple(patch.diff_shape), "scale": layer_scale, "lokr_shapes": None})

    if not layers:
        yield _log("No mergeable adapter layers found; no output written.\n")
        yield {"type": "done", "status": "no matches"}
        return

    invalid = []
    plans = []
    for logical, contributors in sorted(layers.items()):
        shapes = {item["shape"] for item in contributors}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2:
            invalid.append(logical)
            continue
        anchors = [item["lokr_shapes"] for item in contributors if item["lokr_shapes"]]
        actual = "lokr" if output_kind == "lokr" or (output_kind == "auto" and anchors) else "lora"
        if actual == "lokr" and not anchors:
            invalid.append(logical + " (missing LoKr anchor)")
            continue
        if anchors and any(anchor != anchors[0] for anchor in anchors):
            invalid.append(logical + " (incompatible LoKr anchors)")
            continue
        plans.append((logical, contributors, actual, anchors[0] if anchors else None))
    if invalid and mismatch == "error":
        detail = ", ".join(invalid[:5])
        if any("missing LoKr anchor" in item for item in invalid):
            raise ValueError(f"LoKr anchor required for forced LoKr output: {detail}")
        raise ValueError(f"incompatible adapter layers: {detail}")

    estimated_bytes = max(
        (contributors[0]["shape"][0] * contributors[0]["shape"][1] * 4 * (2 * len(contributors) + 2)
         for _, contributors, _, _ in plans),
        default=0,
    )
    device = _device(payload, estimated_bytes)

    actual_kinds = sorted({actual for _, _, actual, _ in plans})
    yield _log(
        f"LoRA composition init\nArchitecture: {architecture}\nPreset: {settings.name}\n"
        f"Output adapter request: {output_kind}\nWill write: {', '.join(actual_kinds)}\n"
        f"Layers: {len(plans)} skipped={len(invalid)}\nDevice: {device}\n"
    )
    if dry_run:
        yield _log(json.dumps({"layers": len(plans), "skipped": invalid, "output_adapter": output_kind,
                               "consensus_preset": settings.name}, indent=2) + "\n")
        yield _status("Dry run complete")
        yield {"type": "done", "status": "dry-run complete"}
        return

    output: Dict[str, torch.Tensor] = {}
    reports = []
    with ExitStack() as stack:
        handles = {path: stack.enter_context(safe_open(path, framework="pt", device="cpu")) for path in manifests}
        total_layers = len(plans)
        progress_every = max(1, total_layers // 100)
        for index, (logical, contributors, actual, anchor) in enumerate(plans, 1):
            try:
                deltas = [_full_delta(handles[item["path"]], item, device) for item in contributors]
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                deltas = [_full_delta(handles[item["path"]], item, "cpu") for item in contributors]
            shape = contributors[0]["shape"]
            rows = torch.stack([value.reshape(shape[0], -1) for value in deltas])
            merged, stats = merge_consensus_rows(rows, settings, return_stats=True)
            merged = merged.reshape(shape).cpu()
            if actual == "lokr":
                w1, w2, report = factorize_lokr(merged, anchor[0], anchor[1])
                output[logical + ".lokr_w1"] = w1.to(torch.bfloat16)
                output[logical + ".lokr_w2"] = w2.to(torch.bfloat16)
            else:
                input_ranks = [int(item.get("rank", 0)) for item in contributors if item["kind"] == "lora"]
                layer_rank_cap = max_rank or (max(input_ranks) if input_ranks else min(shape))
                down, up, report = factorize_lora(merged, max_rank=layer_rank_cap, energy=energy)
                output[logical + ".lora_A.weight"] = down.to(torch.bfloat16)
                output[logical + ".lora_B.weight"] = up.to(torch.bfloat16)
            reports.append({"layer": logical, "kind": actual, "providers": len(contributors),
                            "rank": report.rank, "retained_energy": report.retained_energy,
                            "relative_error": report.relative_error, "rejected": stats.rejected_contributors})
            del deltas, rows, merged
            if index % progress_every == 0 or index == total_layers:
                yield {"type": "progress", "text": f"Merge adapters: {index}/{total_layers} layers ({index * 100 // total_layers}%) · writing {actual.upper()}"}

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = output_path + ".tmp"
    metadata = {"modelspec.architecture": architecture, "dasiwa.adapter_merge": "consensus",
                "dasiwa.adapter_types": ",".join(sorted({item["kind"] for item in reports}))}
    try:
        save_file(output, tmp_path, metadata=metadata)
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    recipe = output_path.rsplit(".", 1)[0] + ".txt"
    with open(recipe, "w", encoding="utf-8") as handle:
        handle.write("DaSiWa LoRA Compose Recipe\n")
        handle.write(f"Output: {os.path.basename(output_path)}\nArchitecture: {architecture}\nConsensus preset: {settings.name}\nOutput adapter: {output_kind}\n")
        handle.write(f"Output rank: {max_rank}\nFrobenius energy: {energy}\n")
        for index, spec in enumerate(specs, 1):
            handle.write(f"{index}. {os.path.realpath(os.path.expanduser(spec['path']))}\n")
            handle.write(f"   Strength: {spec.get('strength', 1.0)}\n")
        handle.write("Layer report:\n" + json.dumps(reports, indent=2) + "\n")
    yield _log(f"Wrote composed adapter: {output_path}\nWrote recipe: {recipe}\n")
    yield _status(f"LoRA composition complete: {len(reports)} layers")
    yield {"type": "done", "status": "finished"}
