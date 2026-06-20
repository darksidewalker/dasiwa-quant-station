import json
import os
from typing import Any, Dict, Iterable, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from utils.lora_inspector import discover_lora_pairs, read_safetensors_manifest
from core.metadata_manager import merge_custom_metadata


# Architecture-specific profile dispatch.
_ARCH_PROFILES = {
    "LTX-2.3": None,  # loaded lazily below
    "WAN 2.2": None,
}


def _get_profile(arch: str):
    """Return (is_preserved, classify, strategy_mult) for *arch*."""
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

    is_preserved, classify_key, strat_mult = _get_profile(architecture)

    yield _log(f"LoRA merge init\nBase: {base_path}\nArchitecture: {architecture}\nStrategy: {strategy}\nAdaptive: {'yes' if adaptive else 'no'}\nDry run: {'yes' if dry_run else 'no'}\n")
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
    merged: Dict[str, torch.Tensor] = {}
    with safe_open(base_path, framework="pt", device="cpu") as bf:
        for key in bf.keys():
            merged[key] = bf.get_tensor(key)

    by_lora: Dict[str, List[Dict[str, Any]]] = {}
    for op in matched_ops:
        by_lora.setdefault(op["lora_path"], []).append(op)

    for lora_path, ops in by_lora.items():
        with safe_open(lora_path, framework="pt", device="cpu") as lf:
            for op in ops:
                base = merged[op["target_key"]]
                down = lf.get_tensor(op["down_key"]).to(torch.float32)
                up = lf.get_tensor(op["up_key"]).to(torch.float32)
                delta = _compute_delta(down, up, tuple(base.shape))
                alpha_scale = _alpha_scale(lf, op["alpha_key"], op["rank"])
                scale = op["scale"] * alpha_scale
                if adaptive:
                    scale *= _adaptive_multiplier(base, delta)
                merged[op["target_key"]] = (base.to(torch.float32) + delta * scale).to(base.dtype)
        yield _log(f"Merged {len(ops)} tensors from {os.path.basename(lora_path)}\n")

    # Write merge recipe as .txt next to the checkpoint.
    # Do NOT embed LoRA details in safetensors metadata — keep it clean.
    recipe_path = _write_recipe(output_path, payload, loras, strategy,
                                global_strength, adaptive, matched_ops,
                                skipped, unmatched, ambiguous)

    # Build full metadata: merge_custom_metadata preserves required LTX 2.3
    # functional fields (architecture, resolution hints, license, etc.) while
    # allowing user-edited custom metadata to overlay.  No merge provenance
    # is embedded — that lives in the recipe file.
    output_name = os.path.basename(output_path)
    meta = merge_custom_metadata(
        architecture,
        output_name.replace(".safetensors", ""),
        output_path,
        bits="BF16 merged",
        custom_meta=payload.get("custom_metadata"),
    )
    save_file(merged, output_path, metadata=meta)
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
                  skipped: int, unmatched: int, ambiguous: int) -> str:
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
