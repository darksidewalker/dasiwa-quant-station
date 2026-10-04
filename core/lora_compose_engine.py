import json
import math
import os
import statistics
from collections import defaultdict
from contextlib import ExitStack
from typing import Any, Dict, Iterable, List, Tuple

import torch
from safetensors import safe_open
from core.safetensors_stream import TensorSpool, destination

from core.adapter_factorization import factorize_lora, factorize_lokr, factorize_additive_lora, FactorizationReport
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


def _reconstruction_recap(reports: List[Dict[str, Any]], *, max_rank: int, energy: float) -> str:
    """Describe factorization loss, not the adapter's unmeasured inference quality."""
    errors = [item["relative_error"] for item in reports]
    worst = max(reports, key=lambda item: item["relative_error"])
    lora = [item for item in reports if item["kind"] == "lora"]
    lokr_count = len(reports) - len(lora)
    high_loss = sum(value > 0.5 for value in errors)
    signal = ("high reconstruction loss; try a higher rank and compare outputs" if high_loss else
              "material reconstruction loss; compare with a higher rank" if max(errors) > 0.1 else
              "low measured reconstruction loss")
    lines = [
        "Reconstruction recap (numerical estimate, not inference-verified):",
        f"Layers: {len(reports)} (LoRA: {len(lora)}, LoKr: {lokr_count}) | "
        f"Relative error median: {statistics.median(errors):.3f} | "
        f"Worst: {worst['relative_error']:.3f} ({worst['layer']})",
        f"Relative error > 0.5: {high_loss}/{len(reports)}",
    ]
    if lora:
        below = sum(item["retained_energy"] + 1e-6 < energy for item in lora)
        lines.append(f"Requested energy: {energy:.1%} | Retained energy below target: {below}/{len(lora)} "
                     f"| Lowest: {min(item['retained_energy'] for item in lora):.1%}")
        if max_rank:
            lines.append(f"At rank cap: {sum(item['rank'] == max_rank for item in lora)}/{len(lora)} "
                         f"(cap {max_rank})")
    lines.append(f"Result signal: {signal}. This is factorization loss, not a quality verdict. "
                 "Test the adapter in ComfyUI with matching prompt and seed; output is not inference-verified.")
    return "\n".join(lines) + "\n"


ROW_BATCH_SIZE = 256


def _delta_rows(handle, item, start, end, device):
    """Reconstruct only requested output rows, including direct Kronecker factors."""
    if item['kind'] == 'diff':
        return handle.get_slice(item['diff_key'])[start:end].to(device=device, dtype=torch.float32) * item['scale']
    from core.consensus_merge import factor_delta_rows
    rows = factor_delta_rows(handle, item['down_key'], item['up_key'], start, end, item['kind'], device)
    return rows * item['scale'] * _alpha_scale(handle, item.get('alpha_key'), item.get('rank', 1), item['kind'])


def _merged_row_batches(contributors, handles, device, algorithm, settings):
    from core.consensus_merge import ConsensusStats
    count = contributors[0]['shape'][0]
    for start in range(0, count, ROW_BATCH_SIZE):
        end = min(count, start + ROW_BATCH_SIZE)
        values = [_delta_rows(handles[item['path']], item, start, end, device).reshape(end-start, -1)
                  for item in contributors]
        if algorithm == 'additive':
            merged = values[0]
            for value in values[1:]:
                merged = merged + value
            stats = ConsensusStats()
        else:
            merged, stats = merge_consensus_rows(torch.stack(values), settings, return_stats=True)
        yield start, end, merged.cpu(), stats


def _serialized_report(contributors, handles, algorithm, settings, tensors, report):
    error = 0.; norm = 0.
    for start, end, source, _ in _merged_row_batches(contributors, handles, 'cpu', algorithm, settings):
        if report.kind == 'lora':
            rebuilt = tensors['.lora_B.weight'][start:end].float() @ tensors['.lora_A.weight'].float()
        elif report.kind == 'lokr':
            w1, w2 = tensors['.lokr_w1'].float(), tensors['.lokr_w2'].float()
            indices = torch.arange(start, end)
            rebuilt = (w1[indices // w2.shape[0], :, None] * w2[indices % w2.shape[0], None, :]).reshape(end-start, -1)
        else:
            rebuilt = tensors['.diff_b'][start:end].reshape(end-start, -1).float()
        error += float((source.double() - rebuilt.double()).square().sum())
        norm += float(source.double().square().sum())
    return FactorizationReport(report.kind, report.rank, report.retained_energy,
                               math.sqrt(error / max(norm, 1e-24)))


def _compose_layer(contributors, handles, device, algorithm, settings, actual, anchor, max_rank, energy):
    from core.consensus_merge import ConsensusStats
    shape = contributors[0]['shape']
    ranks = [int(item.get('rank', 0)) for item in contributors if item['kind'] == 'lora']
    cap = max_rank or (max(ranks) if ranks else min(shape))
    if algorithm == 'additive' and actual == 'lora' and all(item['kind'] == 'lora' for item in contributors):
        pairs = []
        for item in contributors:
            handle = handles[item['path']]
            a = handle.get_tensor(item['down_key']).to(device=device, dtype=torch.float32)
            b = handle.get_tensor(item['up_key']).to(device=device, dtype=torch.float32)
            scale = item['scale'] * _alpha_scale(handle, item.get('alpha_key'), item['rank'], 'lora')
            pairs.append((a, b, scale))
        down, up, report = factorize_additive_lora(pairs, max_rank=cap, energy=energy)
        tensors = {'.lora_A.weight': down.cpu().to(torch.bfloat16), '.lora_B.weight': up.cpu().to(torch.bfloat16)}
        return tensors, _serialized_report(contributors, handles, algorithm, settings, tensors, report), ConsensusStats()
    merged = torch.empty(shape, dtype=torch.float32)
    stats = ConsensusStats()
    for start, end, rows, batch_stats in _merged_row_batches(contributors, handles, device, algorithm, settings):
        merged[start:end] = rows.reshape(merged[start:end].shape)
        for field in vars(stats):
            setattr(stats, field, getattr(stats, field) + getattr(batch_stats, field))
    if actual == 'bias':
        tensors = {'.diff_b': merged}
        report = FactorizationReport('bias', 0, 1., 0.)
    elif actual == 'lokr':
        w1, w2, report = factorize_lokr(merged, anchor[0], anchor[1])
        tensors = {'.lokr_w1': w1.to(torch.bfloat16), '.lokr_w2': w2.to(torch.bfloat16)}
    else:
        down, up, report = factorize_lora(merged, max_rank=cap, energy=energy)
        tensors = {'.lora_A.weight': down.to(torch.bfloat16), '.lora_B.weight': up.to(torch.bfloat16)}
    return tensors, _serialized_report(contributors, handles, algorithm, settings, tensors, report), stats


def run_lora_compose(payload: Dict[str, Any]) -> Iterable[Dict[str, str]]:
    specs = payload.get("loras") or []
    if len(specs) < 2:
        raise ValueError("LoRA composition requires at least two adapters")
    architecture = payload.get("architecture") or "LTX-2.3"
    algorithm = str(payload.get('merge_algorithm') or 'consensus').lower()
    if algorithm not in {'consensus', 'additive'}:
        raise ValueError('merge_algorithm must be consensus or additive')
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
    unsupported_inputs = []
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
        pairs = discover_lora_pairs(manifest)
        patches = discover_diff_patches(manifest)
        supported_pairs = [pair for pair in pairs if pair.kind != "lokr_decomposed"]
        if not supported_pairs and not patches:
            raise ValueError(
                f"Unsupported standalone module/adapter input: {path}. No recognized LoRA A/B "
                "or down/up pairs, direct LoKr pairs, or .diff patches were found. "
                "Full fc1/fc2/fc3 weights and biases are not LoRA deltas and cannot be composed. "
                "Remove this file and load the standalone conditioning bridge/module separately "
                "with its compatible runtime loader; conversion requires an explicit supported format."
            )
        consumed = {key for pair in supported_pairs for key in
                    (pair.down_key, pair.up_key, pair.alpha_key) if key}
        consumed.update(patch.diff_key for patch in patches)
        unhandled = sorted(set(manifest) - consumed)
        if unhandled:
            report = {"path": path, "count": len(unhandled), "keys": unhandled}
            unsupported_inputs.append(report)
            yield _log("WARNING: unsupported/unhandled input tensors (not added as deltas): "
                       + json.dumps(report) + "\n")
        for pair in pairs:
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
        for patch in patches:
            if patch.target_kind == 'buffer':
                raise ValueError('table-changing patches cannot be composed')
            logical = _canonical_candidates(patch.target_candidates, patch.diff_key[:-len(".diff")])
            layer_scale = effective
            layers[logical].append({"path": path, "source": index, "kind": "diff", "diff_key": patch.diff_key,
                                    "shape": tuple(patch.diff_shape), "scale": layer_scale, "lokr_shapes": None})

    # Only AdaLN contributors require coordinates; ordinary adapters may be unbound.
    adaln_paths = {item['path'] for logical, items in layers.items()
                   if '.adaln_proj' in logical for item in items}
    gauges = set()
    for path in manifests if adaln_paths else ():
        with safe_open(path, framework='pt', device='cpu') as handle:
            gauge = (handle.metadata() or {}).get('adaln_coordinate_table_sha256')
            if gauge is not None or path in adaln_paths:
                gauges.add(gauge)
    if len(gauges) > 1:
        raise ValueError('incompatible AdaLN adapter gauge metadata (bound/unbound or conflicting hashes)')
    output_gauge = next(iter(gauges), None)

    if not layers:
        yield _log("No mergeable adapter layers found; no output written.\n")
        yield {"type": "done", "status": "no matches"}
        return

    invalid = []
    plans = []
    for logical, contributors in sorted(layers.items()):
        shapes = {item["shape"] for item in contributors}
        if len(shapes) == 1 and len(next(iter(shapes))) == 1 and logical.endswith('.bias'):
            plans.append((logical, contributors, 'bias', None))
            continue
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

    if not plans:
        yield _log("No compatible adapter layers remain; no output written.\n")
        yield {"type": "done", "status": "no matches"}
        return

    estimated_bytes = max(
        (math.prod(contributors[0]["shape"]) * 4 * (2 * len(contributors) + 2)
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
                               "consensus_preset": settings.name, "unsupported_inputs": unsupported_inputs}, indent=2) + "\n")
        yield _status("Dry run complete")
        yield {"type": "done", "status": "dry-run complete"}
        return

    output_path = destination(dict(payload, output_path=output_path), 'merged_lora.safetensors', manifests)
    reports = []
    with ExitStack() as stack:
        spool = stack.enter_context(TensorSpool(output_path))
        handles = {path: stack.enter_context(safe_open(path, framework="pt", device="cpu")) for path in manifests}
        total_layers = len(plans)
        progress_every = max(1, total_layers // 100)
        for index, (logical, contributors, actual, anchor) in enumerate(plans, 1):
            used_device, fallback_reason = device, None
            try:
                tensors, report, stats = _compose_layer(contributors, handles, device, algorithm, settings,
                                                         actual, anchor, max_rank, energy)
            except torch.cuda.OutOfMemoryError:
                if device == 'cpu':
                    raise
                torch.cuda.empty_cache()
                tensors, report, stats = _compose_layer(contributors, handles, 'cpu', algorithm, settings,
                                                         actual, anchor, max_rank, energy)
                used_device, fallback_reason = 'cpu', 'cuda_oom'
                yield _log(f'CUDA OOM: {logical}; CPU retry completed\n')
            for suffix, value in tensors.items():
                spool.tensor((logical[:-len('.bias')] if actual == 'bias' else logical) + suffix, value)
            ranks = [int(item.get('rank', 0)) for item in contributors if item['kind'] == 'lora']
            cap = max_rank or (max(ranks) if ranks else min(contributors[0]['shape']))
            reports.append({"layer": logical, "kind": actual, "providers": len(contributors),
                            "device": used_device, "fallback_reason": fallback_reason,
                            "rank": report.rank, "effective_rank_cap": cap if actual == 'lora' else 0,
                            "rank_cap_limited": actual == 'lora' and report.rank >= cap and report.retained_energy + 1e-6 < energy,
                            "storage_dtype": 'F32' if actual == 'bias' else 'BF16',
                            "error_basis": 'serialized_vs_original_merged_delta',
                            "retained_energy_basis": 'pre_storage_svd',
                            "retained_energy": report.retained_energy,
                            "relative_error": report.relative_error, "rejected": stats.rejected_contributors})
            del tensors
            if index % progress_every == 0 or index == total_layers:
                yield {"type": "progress", "text": f"Merge adapters: {index}/{total_layers} layers ({index * 100 // total_layers}%) · writing {actual.upper()}"}

        metadata = {"modelspec.architecture": architecture, "dasiwa.adapter_merge": algorithm,
                    "dasiwa.adapter_types": ",".join(sorted({item["kind"] for item in reports}))}
        if output_gauge:
            metadata['adaln_coordinate_table_sha256'] = output_gauge
        recap = _reconstruction_recap(reports, max_rank=max_rank, energy=energy)
        lines = ['DaSiWa LoRA Compose Recipe',
                 f'Output: {os.path.basename(output_path)}', f'Architecture: {architecture}',
                 f'Consensus preset: {settings.name}', f'Output adapter: {output_kind}',
                 f'Output rank: {max_rank}', f'Frobenius energy: {energy}',
                 f'Merge algorithm: {algorithm}',
                 'Output rank 0: legacy largest-input-rank auto cap (not unlimited)',
                 'Relative error: serialized BF16 factors versus original merged delta; bounded-row measurement',
                 'Retained energy: pre-storage SVD retained energy (not BF16 accuracy)']
        for index, spec in enumerate(specs, 1):
            lines.extend([f"{index}. {os.path.realpath(os.path.expanduser(spec['path']))}",
                          f"   Strength: {spec.get('strength', 1.0)}"])
        lines.extend(['Unsupported/unhandled input tensors (not composed):',
                      json.dumps(unsupported_inputs, indent=2), recap,
                      'Layer report:', json.dumps(reports, indent=2)])
        recipe = spool.publish(metadata, '\n'.join(lines) + '\n')
    yield _log(recap)
    yield _log(f"Wrote composed adapter: {output_path}\nWrote recipe: {recipe}\n")
    yield _status(f"LoRA composition complete: {len(reports)} layers")
    yield {"type": "done", "status": "finished"}
