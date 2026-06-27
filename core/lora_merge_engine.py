import json
import os
import struct
from contextlib import ExitStack
from typing import Any, Dict, Iterable, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from utils.lora_inspector import discover_lora_pairs, read_safetensors_manifest
from core.metadata_manager import merge_custom_metadata



def _get_profile(arch: str):
    """Return (is_preserved, classify, strategy_mult) for *arch*."""
    if arch == "Krea 2":
        from utils.krea2_layer_profiles import (
            classify_krea2_key as classify,
            is_krea2_preserved_key as is_preserved,
            strategy_multiplier as strat_mult,
        )
        return is_preserved, classify, strat_mult
    if arch == "WAN 2.2":
        from utils.wan22_layer_profiles import (
            classify_wan22_key as classify,
            is_wan22_preserved_key as is_preserved,
            strategy_multiplier as strat_mult,
        )
        return is_preserved, classify, strat_mult
    # Default to LTX-2.3 (original behaviour).
    from utils.ltx23_layer_profiles import (
        classify_ltx23_key as classify,
        is_ltx23_preserved_key as is_preserved,
        strategy_multiplier as strat_mult,
    )
    return is_preserved, classify, strat_mult


def run_lora_merge(payload: Dict[str, Any]) -> Iterable[Dict[str, str]]:
    base_path = os.path.realpath(os.path.expanduser(payload["base_path"]))
    output_path = os.path.realpath(os.path.expanduser(payload.get("output_path") or _default_output_path(payload)))
    loras = payload.get("loras") or []
    strategy = payload.get("strategy") or "Balanced"
    global_strength = float(payload.get("global_strength", 1.0))
    adaptive = bool(payload.get("adaptive", True))
    dry_run = bool(payload.get("dry_run", False))
    strict = bool(payload.get("strict_matching", True))
    architecture = payload.get("architecture") or "LTX-2.3"
    merge_device = _normalize_merge_device(payload.get("merge_device"))
    cuda_device = payload.get("cuda_device") or "cuda:0"
    vram_headroom_mb = int(payload.get("vram_headroom_mb") or 1024)

    is_preserved, classify_key, strat_mult = _get_profile(architecture)

    yield _log(f"LoRA merge init\nBase: {base_path}\nArchitecture: {architecture}\nStrategy: {strategy}\nAdaptive: {'yes' if adaptive else 'no'}\nDry run: {'yes' if dry_run else 'no'}\nMerge device: requested={merge_device} cuda_device={cuda_device} headroom={vram_headroom_mb}MB\n")
    base_manifest = read_safetensors_manifest(base_path)
    base_keys = set(base_manifest)
    yield _status(f"Inspected base: {len(base_manifest)} tensors")

    reports: List[Dict[str, Any]] = []
    matched_ops: List[Dict[str, Any]] = []
    skipped = 0
    unmatched = 0
    ambiguous = 0

    for lora_spec in loras:
        lora_path = os.path.realpath(os.path.expanduser(lora_spec["path"]))
        lora_strength = float(lora_spec.get("strength", 1.0))
        lora_strategy = lora_spec.get("strategy") or strategy
        lora_manifest = read_safetensors_manifest(lora_path)
        pairs = discover_lora_pairs(lora_manifest)
        yield _log(f"LoRA: {os.path.basename(lora_path)} | strategy={lora_strategy} | tensors={len(lora_manifest)} | pairs={len(pairs)}\n")
        with safe_open(lora_path, framework="pt", device="cpu") as lf:
            for pair in pairs:
                candidates = [c for c in pair.target_candidates if c in base_keys]
                if len(candidates) == 0:
                    unmatched += 1
                    reports.append(_report(pair.base_name, None, "unmatched", lora_path))
                    continue
                if len(candidates) > 1:
                    ambiguous += 1
                    reports.append(_report(pair.base_name, None, "ambiguous", lora_path, candidates=candidates))
                    if strict:
                        continue
                target_key = candidates[0]
                if is_preserved(target_key):
                    skipped += 1
                    reports.append(_report(pair.base_name, target_key, "skipped_preserve", lora_path))
                    continue
                base_shape = base_manifest[target_key].shape
                delta_shape = _delta_shape(pair.down_shape, pair.up_shape)
                if tuple(base_shape) != tuple(delta_shape):
                    unmatched += 1
                    reports.append(_report(pair.base_name, target_key, "shape_mismatch", lora_path, base_shape=base_shape, delta_shape=delta_shape))
                    continue
                category = classify_key(target_key)
                scale = global_strength * lora_strength * strat_mult(lora_strategy, category)
                matched_ops.append({
                    "lora_path": lora_path,
                    "strategy": lora_strategy,
                    "down_key": pair.down_key,
                    "up_key": pair.up_key,
                    "alpha_key": pair.alpha_key,
                    "rank": pair.rank,
                    "target_key": target_key,
                    "category": category,
                    "scale": scale,
                })
                reports.append(_report(pair.base_name, target_key, "matched", lora_path, category=category, strategy=lora_strategy, scale=scale, rank=pair.rank))

    yield _log(f"Dry-run report: matched={len(matched_ops)} skipped={skipped} unmatched={unmatched} ambiguous={ambiguous}\n")
    if dry_run:
        yield _log(json.dumps({"reports": reports[:200], "report_count": len(reports)}, indent=2) + "\n")
        yield _status("Dry run complete")
        yield {"type": "done", "status": "dry-run complete"}
        return

    if not matched_ops:
        yield _log("No matching LoRA tensors to merge; no output written.\n")
        yield {"type": "done", "status": "no matches"}
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ops_by_target: Dict[str, List[Dict[str, Any]]] = {}
    for op in matched_ops:
        ops_by_target.setdefault(op["target_key"], []).append(op)

    device_summary = {
        "requested": merge_device,
        "cuda_device": cuda_device,
        "cuda_tensors": 0,
        "cpu_tensors": 0,
        "cuda_unavailable": 0,
        "insufficient_vram": 0,
        "cuda_oom": 0,
    }
    merge_summary = {
        "matched_ops": len(matched_ops),
        "applied_ops": 0,
        "target_tensors": len(ops_by_target),
        "altered_targets": 0,
        "unaltered_targets": 0,
    }

    output_name = os.path.basename(output_path)
    meta = merge_custom_metadata(
        architecture,
        output_name.replace(".safetensors", ""),
        output_path,
        bits="BF16 merged",
        custom_meta=payload.get("custom_metadata"),
    )
    tmp_output_path = output_path + ".tmp"
    with open(tmp_output_path, "wb") as out_f, ExitStack() as stack:
        header = _build_safetensors_header(base_manifest, meta)
        out_f.write(struct.pack("<Q", len(header)))
        out_f.write(header)
        lora_handles = {
            path: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            for path in sorted({op["lora_path"] for op in matched_ops})
        }
        with safe_open(base_path, framework="pt", device="cpu") as bf:
            for key in bf.keys():
                base = bf.get_tensor(key)
                ops = ops_by_target.get(key)
                if not ops:
                    _write_tensor_bytes(out_f, base)
                    del base
                    continue
                tensor, used_device, fallback_reason = _merge_target_with_policy(
                    base, ops, lora_handles, adaptive, merge_device, cuda_device, vram_headroom_mb
                )
                merge_summary["applied_ops"] += len(ops)
                if _tensor_was_altered(base, tensor):
                    merge_summary["altered_targets"] += 1
                else:
                    merge_summary["unaltered_targets"] += 1
                _write_tensor_bytes(out_f, tensor)
                del base, tensor
                if used_device == "cuda":
                    device_summary["cuda_tensors"] += 1
                else:
                    device_summary["cpu_tensors"] += 1
                if fallback_reason:
                    device_summary[fallback_reason] += 1
    os.replace(tmp_output_path, output_path)

    yield _log(
        "LoRA merge device summary: "
        f"requested={device_summary['requested']} cuda={device_summary['cuda_tensors']} "
        f"cpu={device_summary['cpu_tensors']} cuda_unavailable={device_summary['cuda_unavailable']} "
        f"insufficient_vram={device_summary['insufficient_vram']} oom={device_summary['cuda_oom']}\n"
    )
    all_applied = merge_summary["applied_ops"] == merge_summary["matched_ops"]
    yield _log(
        "LoRA merge success summary: "
        f"all_matched_applied={'yes' if all_applied else 'no'} "
        f"altered_targets={merge_summary['altered_targets']}/{merge_summary['target_tensors']} "
        f"applied_ops={merge_summary['applied_ops']}/{merge_summary['matched_ops']}\n"
    )
    if merge_summary["unaltered_targets"]:
        yield _log(f"WARNING: {merge_summary['unaltered_targets']} matched target tensor(s) were not altered. Check zero-strength LoRAs, zero deltas, or strategy multipliers.\n")
    yield _status(f"LoRA merge complete: {merge_summary['altered_targets']}/{merge_summary['target_tensors']} targets altered")

    # Write merge recipe as .txt next to the checkpoint.
    # Do NOT embed LoRA details in safetensors metadata — keep it clean.
    recipe_path = _write_recipe(output_path, payload, loras, strategy,
                                global_strength, adaptive, matched_ops,
                                skipped, unmatched, ambiguous, device_summary, merge_summary)

    yield _log(f"Wrote merged checkpoint: {output_path}\n")
    yield _log(f"Wrote merge recipe: {recipe_path}\n")
    yield _status("LoRA merge complete")
    yield {"type": "done", "status": "finished"}


def _default_output_path(payload: Dict[str, Any]) -> str:
    models_dir = payload.get("models_dir") or os.path.dirname(payload["base_path"])
    output_name = payload.get("output_name") or "ltx23_lora_merged.safetensors"
    if not output_name.endswith(".safetensors"):
        output_name += ".safetensors"
    return os.path.join(models_dir, output_name)


def _log(text: str) -> Dict[str, str]:
    return {"type": "log", "text": text}


def _status(status: str) -> Dict[str, str]:
    return {"type": "status", "status": status}


def _report(base_name: str, target_key: str | None, status: str, lora_path: str, **extra: Any) -> Dict[str, Any]:
    obj = {"base_name": base_name, "target_key": target_key, "status": status, "lora": os.path.basename(lora_path)}
    obj.update(extra)
    return obj


_SAFETENSORS_DTYPE = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}


def _build_safetensors_header(manifest: Dict[str, Any], metadata: Dict[str, str]) -> bytes:
    header: Dict[str, Any] = {"__metadata__": {str(k): str(v) for k, v in metadata.items()}}
    offset = 0
    for key, info in manifest.items():
        dtype = _safetensors_dtype(info.dtype)
        size = _numel(tuple(info.shape)) * _dtype_size(dtype)
        header[key] = {"dtype": dtype, "shape": list(info.shape), "data_offsets": [offset, offset + size]}
        offset += size
    return json.dumps(header, separators=(",", ":")).encode("utf-8")


def _safetensors_dtype(dtype: str) -> str:
    normalized = dtype.replace("torch.", "").upper()
    aliases = {
        "FLOAT64": "F64",
        "FLOAT32": "F32",
        "FLOAT16": "F16",
        "BFLOAT16": "BF16",
        "INT64": "I64",
        "INT32": "I32",
        "INT16": "I16",
        "INT8": "I8",
        "UINT8": "U8",
        "BOOL": "BOOL",
    }
    if normalized in aliases:
        return aliases[normalized]
    if normalized in set(aliases.values()):
        return normalized
    raise ValueError(f"Unsupported safetensors dtype: {dtype}")


def _dtype_size(dtype: str) -> int:
    sizes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "I16": 2,
        "I32": 4,
        "I64": 8,
        "F16": 2,
        "BF16": 2,
        "F32": 4,
        "F64": 8,
    }
    return sizes[dtype]


def _write_tensor_bytes(out_f: Any, tensor: torch.Tensor) -> None:
    contiguous = tensor.detach().cpu().contiguous()
    contiguous.view(torch.uint8).numpy().tofile(out_f)


def _tensor_was_altered(before: torch.Tensor, after: torch.Tensor) -> bool:
    return not torch.equal(before.detach().cpu(), after.detach().cpu())


def _normalize_merge_device(value: str | None) -> str:
    device = (value or "auto").strip().lower()
    if device not in {"cpu", "auto", "cuda"}:
        raise ValueError(f"merge_device must be one of cpu, auto, cuda; got {value!r}")
    return device


def _cuda_available(device: str) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.device(device)
        return True
    except (RuntimeError, ValueError):
        return False


def _vram_free_total(device: str) -> Tuple[int, int]:
    return torch.cuda.mem_get_info(device)


def _estimate_lora_merge_peak_bytes(base_shape: Tuple[int, ...], down_shape: Tuple[int, ...], up_shape: Tuple[int, ...]) -> int:
    target_numel = _numel(base_shape)
    down_numel = _numel(down_shape)
    up_numel = _numel(up_shape)
    return target_numel * 4 * 4 + down_numel * 4 + up_numel * 4


def _numel(shape: Tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _has_cuda_headroom(device: str, estimated_bytes: int, headroom_mb: int) -> bool:
    free_bytes, _ = _vram_free_total(device)
    return free_bytes > estimated_bytes + max(headroom_mb, 0) * 1024 * 1024


def _delta_shape(down_shape: Tuple[int, ...], up_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(down_shape) == 2 and len(up_shape) == 2:
        return (up_shape[0], down_shape[1])
    return up_shape[:-1] + down_shape[1:]


def _compute_delta(down: torch.Tensor, up: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    if down.ndim == 2 and up.ndim == 2:
        delta = up @ down
    else:
        raise ValueError(f"Only 2D LoRA tensors are supported initially, got {tuple(down.shape)} and {tuple(up.shape)}")
    if tuple(delta.shape) != target_shape:
        raise ValueError(f"Delta shape {tuple(delta.shape)} does not match target {target_shape}")
    return delta


def _merge_target_cpu(base: torch.Tensor, ops: List[Dict[str, Any]], lora_handles: Dict[str, Any], adaptive: bool) -> torch.Tensor:
    original_dtype = base.dtype
    merged = base
    for op in ops:
        lf = lora_handles[op["lora_path"]]
        down = lf.get_tensor(op["down_key"]).to(torch.float32)
        up = lf.get_tensor(op["up_key"]).to(torch.float32)
        delta = _compute_delta(down, up, tuple(merged.shape))
        scale = op["scale"] * _alpha_scale(lf, op["alpha_key"], op["rank"])
        if adaptive:
            scale *= _adaptive_multiplier(merged, delta)
        merged = (merged.to(torch.float32) + delta * scale).to(original_dtype)
    return merged


def _merge_target_cuda(base: torch.Tensor, ops: List[Dict[str, Any]], lora_handles: Dict[str, Any], adaptive: bool, device: str) -> torch.Tensor:
    original_dtype = base.dtype
    target_shape = tuple(base.shape)
    with torch.no_grad():
        merged = base.to(device=device, dtype=torch.float32)
        for op in ops:
            lf = lora_handles[op["lora_path"]]
            down = lf.get_tensor(op["down_key"]).to(device=device, dtype=torch.float32)
            up = lf.get_tensor(op["up_key"]).to(device=device, dtype=torch.float32)
            delta = _compute_delta(down, up, target_shape)
            scale = op["scale"] * _alpha_scale(lf, op["alpha_key"], op["rank"])
            if adaptive:
                scale *= _adaptive_multiplier(merged, delta)
            merged = merged + delta * scale
            del down, up, delta
        result = merged.to(device="cpu", dtype=original_dtype)
        del merged
        return result


def _merge_target_with_policy(base: torch.Tensor, ops: List[Dict[str, Any]], lora_handles: Dict[str, Any], adaptive: bool, policy: str, device: str, headroom_mb: int) -> Tuple[torch.Tensor, str, str | None]:
    if policy == "cpu":
        return _merge_target_cpu(base, ops, lora_handles, adaptive), "cpu", None
    if not _cuda_available(device):
        return _merge_target_cpu(base, ops, lora_handles, adaptive), "cpu", "cuda_unavailable"

    estimated = 0
    for op in ops:
        lf = lora_handles[op["lora_path"]]
        down_shape = tuple(lf.get_slice(op["down_key"]).get_shape())
        up_shape = tuple(lf.get_slice(op["up_key"]).get_shape())
        estimated = max(estimated, _estimate_lora_merge_peak_bytes(tuple(base.shape), down_shape, up_shape))
    if not _has_cuda_headroom(device, estimated, headroom_mb):
        return _merge_target_cpu(base, ops, lora_handles, adaptive), "cpu", "insufficient_vram"

    try:
        return _merge_target_cuda(base, ops, lora_handles, adaptive, device), "cuda", None
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return _merge_target_cpu(base, ops, lora_handles, adaptive), "cpu", "cuda_oom"


def _alpha_scale(lf: safe_open, alpha_key: str | None, rank: int) -> float:
    if not alpha_key:
        return 1.0
    alpha = float(lf.get_tensor(alpha_key).reshape(-1)[0].item())
    return alpha / max(rank, 1)


def _adaptive_multiplier(base: torch.Tensor, delta: torch.Tensor) -> float:
    base_norm = float(torch.linalg.vector_norm(base.to(torch.float32)).item())
    delta_norm = float(torch.linalg.vector_norm(delta.to(torch.float32)).item())
    if base_norm <= 1e-8 or delta_norm <= 1e-8:
        return 1.0
    ratio = delta_norm / base_norm
    target = 0.08
    mult = target / max(min(ratio, 1.0), 1e-4)
    return max(0.25, min(mult, 2.0))


def _write_recipe(output_path: str, payload: Dict[str, Any],
                  loras: List[Dict[str, Any]], strategy: str,
                  global_strength: float, adaptive: bool,
                  matched_ops: List[Dict[str, Any]],
                  skipped: int, unmatched: int, ambiguous: int,
                  device_summary: Dict[str, Any] | None = None,
                  merge_summary: Dict[str, Any] | None = None) -> str:
    """Write a human-readable merge recipe .txt next to the checkpoint."""
    import datetime

    recipe_path = output_path.rsplit(".", 1)[0] + ".txt"
    base_name = os.path.basename(payload.get("base_path", "unknown"))
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 64,
        "  DaSiWa LoRA Merge Recipe",
        "=" * 64,
        "",
        f"Date:              {now}",
        f"Output:            {os.path.basename(output_path)}",
        f"Base checkpoint:   {base_name}",
        f"Base path:         {os.path.realpath(os.path.expanduser(payload['base_path']))}",
        f"Default strategy:  {strategy}",
        f"Global strength:   {global_strength}",
        f"Adaptive scaling:  {'yes' if adaptive else 'no'}",
        f"Requested device:  {(device_summary or {}).get('requested', payload.get('merge_device', 'auto'))}",
        f"CUDA device:       {(device_summary or {}).get('cuda_device', payload.get('cuda_device', 'cuda:0'))}",
        f"CUDA tensors:      {(device_summary or {}).get('cuda_tensors', 0)}",
        f"CPU tensors:       {(device_summary or {}).get('cpu_tensors', 0)}",
        f"CUDA OOM retries:  {(device_summary or {}).get('cuda_oom', 0)}",
        f"VRAM fallbacks:    {(device_summary or {}).get('insufficient_vram', 0)}",
        f"Applied ops:       {(merge_summary or {}).get('applied_ops', len(matched_ops))}/{(merge_summary or {}).get('matched_ops', len(matched_ops))}",
        f"Altered targets:   {(merge_summary or {}).get('altered_targets', 0)}/{(merge_summary or {}).get('target_tensors', 0)}",
        f"Unaltered targets: {(merge_summary or {}).get('unaltered_targets', 0)}",
        "",
        "-" * 64,
        "  LoRAs",
        "-" * 64,
        "",
    ]

    for i, spec in enumerate(loras, 1):
        lora_name = os.path.basename(os.path.realpath(os.path.expanduser(spec["path"])))
        lora_strength = spec.get("strength", "1.0")
        lora_strategy = spec.get("strategy") or strategy
        lines.append(f"  {i}. {lora_name}")
        lines.append(f"     Strength:  {lora_strength}")
        lines.append(f"     Strategy:  {lora_strategy}")
        lines.append("")

    lines.extend([
        "-" * 64,
        "  Merge Summary",
        "-" * 64,
        "",
        f"  Matched tensors:  {len(matched_ops)}",
        f"  Skipped (preserve): {skipped}",
        f"  Unmatched:        {unmatched}",
        f"  Ambiguous:        {ambiguous}",
        "",
    ])

    # Per-LoRA tensor breakdown
    by_lora: Dict[str, List[Dict[str, Any]]] = {}
    for op in matched_ops:
        by_lora.setdefault(os.path.basename(op["lora_path"]), []).append(op)

    if by_lora:
        lines.append("-" * 64)
        lines.append("  Per-LoRA Tensor Details")
        lines.append("-" * 64)
        lines.append("")

        for lora_name, ops in by_lora.items():
            lines.append(f"  {lora_name}  ({len(ops)} tensors merged)")
            # Group by category
            by_cat: Dict[str, int] = {}
            for op in ops:
                cat = op.get("category", "unknown")
                by_cat[cat] = by_cat.get(cat, 0) + 1
            for cat, count in sorted(by_cat.items()):
                lines.append(f"    {cat}: {count}")
            lines.append("")

    lines.extend([
        "-" * 64,
        "  Reproduce this merge",
        "-" * 64,
        "",
        "  1. Select the base checkpoint listed above.",
        "  2. Add each LoRA with the same strength and strategy.",
        "  3. Set global strength and adaptive scaling as shown.",
        "  4. Run the merge in DaSiWa Quant Station (LoRA Merge mode).",
        "",
        "=" * 64,
    ])

    with open(recipe_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return recipe_path
