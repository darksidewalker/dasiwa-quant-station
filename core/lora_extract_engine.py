"""Extract standard LoRAs from a base/modified checkpoint pair.

The generic recipe needs exactly two shape-compatible checkpoints and SVD-factorizes
changed 2-D tensors into standard LoRA pairs. MiniMax H3 additionally supports a
curve-pruned recipe. For that recipe, AdaLN deltas are rebased onto the exact
coordinate gauge stored by the selected pruned target:

    full(t) = delta_W @ (c + V @ q(t)) + delta_b
            = (delta_W @ V) @ q(t) + (delta_b + delta_W @ c)

The rebased AdaLN payload uses ``.diff`` and ``.diff_b`` because the constant
term cannot be represented by a conventional LoRA pair alone.
"""

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from utils.arch_detector import verify_architecture_match
from utils.lora_inspector import read_safetensors_manifest


_H3_TIME_KEYS = (
    "time_embedder.proj_in.weight",
    "time_embedder.proj_in.bias",
    "time_embedder.proj_out.weight",
    "time_embedder.proj_out.bias",
)
_ADALN_WEIGHT_SUFFIX = ".adaln_proj.linear.weight"
_ADALN_BIAS_SUFFIX = ".adaln_proj.linear.bias"


def _event(kind: str, text: str = "", status: str = "") -> Dict[str, str]:
    event: Dict[str, str] = {"type": kind}
    if text:
        event["text"] = text
    if status:
        event["status"] = status
    return event


def _resolve(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


def _require_file(path: str, label: str) -> str:
    resolved = _resolve(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if not resolved.lower().endswith(".safetensors"):
        raise ValueError(f"{label} must be a .safetensors file: {resolved}")
    return resolved


def _h3_prefix(manifest: Dict[str, Any], require_full: bool) -> str:
    matches = [key[:-len(_H3_TIME_KEYS[0])] for key in manifest if key.endswith(_H3_TIME_KEYS[0])]
    if require_full:
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one MiniMax H3 full time embedder, found {len(matches)}")
        prefix = matches[0]
        missing = [prefix + key for key in _H3_TIME_KEYS if prefix + key not in manifest]
        if missing:
            raise ValueError(f"Full MiniMax H3 checkpoint is missing time embedder tensors: {missing}")
        return prefix
    tables = [key[:-len("adaln_t_table")] for key in manifest if key.endswith("adaln_t_table")]
    if len(tables) != 1:
        raise ValueError(f"Expected exactly one MiniMax H3 pruned AdaLN table, found {len(tables)}")
    return tables[0]


def _time_curve(handle, prefix: str, grid: int, device: torch.device) -> torch.Tensor:
    w1 = handle.get_tensor(prefix + "time_embedder.proj_in.weight").to(device=device, dtype=torch.float64)
    b1 = handle.get_tensor(prefix + "time_embedder.proj_in.bias").to(device=device, dtype=torch.float64)
    w2 = handle.get_tensor(prefix + "time_embedder.proj_out.weight").to(device=device, dtype=torch.float64)
    b2 = handle.get_tensor(prefix + "time_embedder.proj_out.bias").to(device=device, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, grid, dtype=torch.float64, device=device)
    half = w1.shape[1] // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float64, device=device) / half)
    embedding = torch.cat((torch.cos(t[:, None] * freqs), torch.sin(t[:, None] * freqs)), dim=-1)
    return F.silu(F.linear(F.silu(F.linear(embedding, w1, b1)), w2, b2))


def _recover_target_gauge(full_handle, full_prefix: str, pruned_handle, pruned_prefix: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    table = pruned_handle.get_tensor(pruned_prefix + "adaln_t_table").to(dtype=torch.float64, device=device)
    if table.ndim != 2 or tuple(table.shape) != (1025, 8):
        raise ValueError(f"Target pruned AdaLN table must have shape [1025, 8], got {list(table.shape)}")
    curve = _time_curve(full_handle, full_prefix, table.shape[0], device)
    design = torch.cat((table, torch.ones((table.shape[0], 1), dtype=torch.float64, device=device)), dim=1)
    affine = torch.linalg.lstsq(design, curve).solution
    basis = affine[:8].T.contiguous()  # [2688, 8]
    center = affine[8].contiguous()    # [2688]
    residual = design @ affine - curve
    relative = float(torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(curve))
    table_hash = hashlib.sha256(table.to(dtype=torch.float32).contiguous().cpu().numpy().tobytes()).hexdigest()
    return basis, center, table, table_hash + f" (least-squares relative residual {relative:.3e})"


def _svd_factors(delta: torch.Tensor, energy: float, min_rank: int, max_rank: int) -> Tuple[torch.Tensor, torch.Tensor, int, float]:
    if delta.ndim != 2:
        raise ValueError(f"LoRA extraction supports only 2-D tensors, got {list(delta.shape)}")
    u, singular, vh = torch.linalg.svd(delta.to(dtype=torch.float32), full_matrices=False)
    total = singular.square().sum()
    if not torch.isfinite(total) or total <= 0:
        return torch.empty((0, delta.shape[1]), dtype=torch.bfloat16), torch.empty((delta.shape[0], 0), dtype=torch.bfloat16), 0, 1.0
    retained = torch.cumsum(singular.square(), dim=0) / total
    rank = int(torch.searchsorted(retained, torch.tensor(energy, device=retained.device)).item()) + 1
    rank = max(min_rank, rank)
    if max_rank > 0:
        rank = min(rank, max_rank)
    rank = min(rank, singular.numel())
    root = singular[:rank].sqrt()
    down = (root[:, None] * vh[:rank]).to(dtype=torch.bfloat16).contiguous()
    up = (u[:, :rank] * root[None, :]).to(dtype=torch.bfloat16).contiguous()
    actual = float(retained[rank - 1].item())
    return down, up, rank, actual


def _adapter_module(key: str) -> str:
    """Return a portable adapter module name accepted by the merge loader."""
    module = key[:-len(".weight")] if key.endswith(".weight") else key
    if module.startswith("model.diffusion_model."):
        return module[len("model."):]
    if module.startswith("diffusion_model."):
        return module
    return "diffusion_model." + module


def _write_recipe(output_path: str, payload: Dict[str, Any], table_hash: str, summary: Dict[str, Any]) -> str:
    recipe_path = output_path + ".txt"
    recipe = payload.get("recipe") or ("h3_pruned" if payload.get("output_mode", "pruned") == "pruned" else "h3_full")
    lines = [
        "DaSiWa Quant Station LoRA Extract Recipe",
        "",
        f"Recipe: {recipe}",
        f"Architecture: {payload.get('architecture', 'Not set')}",
        f"Base checkpoint: {payload['base_path']}",
        f"Modified checkpoint: {payload['merged_path']}",
        f"Pruned target: {payload.get('pruned_target_path') or 'none'}",
        f"Output mode: {payload.get('output_mode', 'pruned')}",
        f"Frobenius energy: {float(payload.get('frobenius_energy', 0.99)):.6f}",
        f"Minimum rank: {int(payload.get('min_rank', 1))}",
        f"Maximum rank: {int(payload.get('max_rank', 0))}",
        f"AdaLN table SHA256: {table_hash or 'not applicable'}",
        "",
        f"LoRA pairs: {summary['pairs']}",
        f"AdaLN direct patches: {summary['adaln']}",
        f"Skipped unchanged tensors: {summary['unchanged']}",
        f"Skipped unsupported tensors: {summary['unsupported']}",
    ]
    Path(recipe_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return recipe_path


def run_lora_extract(payload: Dict[str, Any]) -> Iterable[Dict[str, str]]:
    """Extract a standard or H3 curve-pruned adapter from two checkpoints."""
    architecture = payload.get("architecture") or "Not set"
    recipe = payload.get("recipe")
    if not recipe:
        legacy_mode = payload.get("output_mode") or "pruned"
        recipe = "h3_pruned" if legacy_mode == "pruned" else "h3_full"
    if recipe not in {"generic", "h3_full", "h3_pruned"}:
        raise ValueError("recipe must be 'generic', 'h3_full', or 'h3_pruned'")
    if recipe.startswith("h3_") and architecture != "MiniMax H3":
        raise ValueError("MiniMax H3 extraction recipes require the MiniMax H3 architecture")

    base_path = _require_file(payload["base_path"], "Base checkpoint")
    merged_path = _require_file(payload["merged_path"], "Modified checkpoint")
    output_mode = "pruned" if recipe == "h3_pruned" else "full"
    payload = dict(payload)
    payload["recipe"] = recipe
    payload["output_mode"] = output_mode
    default_name = "minimax_h3_extracted_lora.safetensors" if recipe.startswith("h3_") else "checkpoint_delta_lora.safetensors"
    output_path = _resolve(payload.get("output_path") or os.path.join(payload.get("output_dir") or os.path.dirname(merged_path), payload.get("output_name") or default_name))
    if not output_path.endswith(".safetensors"):
        output_path += ".safetensors"
    if os.path.exists(output_path):
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    energy = float(payload.get("frobenius_energy", 0.99))
    min_rank = int(payload.get("min_rank", 1))
    max_rank = int(payload.get("max_rank", 0))
    dry_run = bool(payload.get("dry_run", False))
    if not 0 < energy <= 1:
        raise ValueError("frobenius_energy must be in (0, 1]")
    if min_rank < 1 or max_rank < 0:
        raise ValueError("min_rank must be >= 1 and max_rank must be >= 0")

    ok, message = verify_architecture_match(base_path, architecture)
    if not ok:
        raise ValueError(message)
    ok, message = verify_architecture_match(merged_path, architecture)
    if not ok:
        raise ValueError(message)
    yield _event("log", f"LoRA Extract: {architecture} ({recipe})\nBase: {base_path}\nModified: {merged_path}\nFrobenius energy: {energy:.4f}\n")

    base_manifest = read_safetensors_manifest(base_path)
    merged_manifest = read_safetensors_manifest(merged_path)
    if set(base_manifest) != set(merged_manifest):
        only_base = sorted(set(base_manifest) - set(merged_manifest))[:3]
        only_merged = sorted(set(merged_manifest) - set(base_manifest))[:3]
        raise ValueError(f"Full base and merged checkpoints have different key sets (base-only={only_base}, merged-only={only_merged})")
    shape_mismatches = [key for key in base_manifest if base_manifest[key].shape != merged_manifest[key].shape]
    if shape_mismatches:
        raise ValueError(f"Full base and merged checkpoints have shape mismatches: {shape_mismatches[:3]}")
    base_prefix = ""
    if recipe == "h3_pruned":
        base_prefix = _h3_prefix(base_manifest, require_full=True)
        merged_prefix = _h3_prefix(merged_manifest, require_full=True)
        if base_prefix != merged_prefix:
            raise ValueError(f"Full base and merged prefixes differ: {base_prefix!r} vs {merged_prefix!r}")

    pruned_path = ""
    table_hash = ""
    device = torch.device("cpu")
    basis = center = None
    if output_mode == "pruned":
        pruned_path = _require_file(payload.get("pruned_target_path", ""), "Target pruned checkpoint")
        pruned_manifest = read_safetensors_manifest(pruned_path)
        ok, message = verify_architecture_match(pruned_path, "MiniMax H3")
        if not ok:
            raise ValueError(message)
        pruned_prefix = _h3_prefix(pruned_manifest, require_full=False)
        with safe_open(base_path, framework="pt", device="cpu") as base_handle, safe_open(pruned_path, framework="pt", device="cpu") as pruned_handle:
            basis, center, _table, table_detail = _recover_target_gauge(base_handle, base_prefix, pruned_handle, pruned_prefix, device)
        table_hash = table_detail.split(" ")[0]
        yield _event("log", f"Recovered target AdaLN gauge: {table_detail}\n")

    output: Dict[str, torch.Tensor] = {}
    summary = {"pairs": 0, "adaln": 0, "unchanged": 0, "unsupported": 0}
    ranks: List[int] = []
    with safe_open(base_path, framework="pt", device="cpu") as base_handle, safe_open(merged_path, framework="pt", device="cpu") as merged_handle:
        keys = list(base_handle.keys())
        total = len(keys)
        for index, key in enumerate(keys, start=1):
            if key.endswith(_ADALN_BIAS_SUFFIX) and output_mode == "pruned":
                continue
            base = base_handle.get_tensor(key)
            merged = merged_handle.get_tensor(key)
            if (
                base.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or base.ndim != 2
                or (recipe == "generic" and not key.endswith(".weight"))
            ):
                summary["unsupported"] += 1
                continue
            delta = (merged.to(dtype=torch.float32) - base.to(dtype=torch.float32))
            if not torch.any(delta):
                summary["unchanged"] += 1
                continue
            module = _adapter_module(key)
            if output_mode == "pruned" and key.endswith(_ADALN_WEIGHT_SUFFIX):
                bias_key = key[:-len("weight")] + "bias"
                if bias_key not in base_manifest:
                    raise ValueError(f"AdaLN weight has no matching bias: {key}")
                delta_b = merged_handle.get_tensor(bias_key).to(dtype=torch.float64) - base_handle.get_tensor(bias_key).to(dtype=torch.float64)
                rebased_w = (delta.to(dtype=torch.float64) @ basis).to(dtype=torch.float32).contiguous()
                rebased_b = (delta_b + delta.to(dtype=torch.float64) @ center).to(dtype=torch.float32).contiguous()
                if not dry_run:
                    output[f"{module}.diff"] = rebased_w
                    output[f"{module}.diff_b"] = rebased_b
                summary["adaln"] += 1
            else:
                down, up, rank, actual = _svd_factors(delta, energy, min_rank, max_rank)
                if rank == 0:
                    summary["unchanged"] += 1
                    continue
                if not dry_run:
                    output[f"{module}.lora_A.weight"] = down
                    output[f"{module}.lora_B.weight"] = up
                    output[f"{module}.alpha"] = torch.tensor(float(rank), dtype=torch.float32)
                summary["pairs"] += 1
                ranks.append(rank)
            if index % 10 == 0 or index == total:
                yield _event("progress", f"Extracting tensors: {index}/{total}")

    if summary["pairs"] == 0 and summary["adaln"] == 0:
        raise ValueError("No changed 2-D tensors found between full base and merged checkpoints")
    rank_note = f" ranks={min(ranks)}..{max(ranks)}" if ranks else ""
    yield _event("log", f"Extraction plan: LoRA pairs={summary['pairs']}, AdaLN rebased patches={summary['adaln']},{rank_note}\n")
    if dry_run:
        yield _event("done", status="dry-run complete")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metadata = {
        "format": "dasiwa_checkpoint_delta_lora" if recipe == "generic" else "dasiwa_minimax_h3_extracted_lora",
        "architecture": architecture,
        "recipe": recipe,
        "output_mode": output_mode,
        "source_base": os.path.basename(base_path),
        "source_merged": os.path.basename(merged_path),
        "frobenius_energy": f"{energy:.6f}",
        "min_rank": str(min_rank),
        "max_rank": str(max_rank),
    }
    if output_mode == "pruned":
        metadata.update({
            "adaln_patch_format": "diff+diff_b",
            "adaln_coordinate_table_sha256": table_hash,
            "adaln_source_width": "2688",
            "adaln_target_width": "8",
            "adaln_target": os.path.basename(pruned_path),
        })
    save_file(output, output_path, metadata=metadata)
    recipe_path = _write_recipe(output_path, payload, table_hash, summary)
    yield _event("log", f"Wrote extracted adapter: {output_path}\nWrote recipe: {recipe_path}\n")
    yield _event("done", status="finished")
