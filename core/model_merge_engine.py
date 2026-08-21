"""
Model-level merge engine (NOT LoRA math).

Currently one recipe:

  h3_hybrid
    Base   : fl2va  (all tensors)
    Overlay: ref2va (blocks.{N}.adaln_proj.linear.{bias,weight,weight_scale}
             where N >= len(blocks)//2)
    Order-agnostic: filenames are inspected to auto-detect which is which.
    Metadata: base __metadata__ + minimax_h3_hybrid=baked
              + base_model/overlay_model filenames.

All merges stream tensor-by-tensor (no full model in memory).
"""

import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from utils.lora_inspector import read_safetensors_manifest
from core.metadata_manager import merge_custom_metadata
from core.watermark import watermark_for

# ---------------------------------------------------------------------------
# Recipe registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Recipe:
    name: str
    architecture: str
    # regex matched against tensor key; group(1) must be the block index
    overlay_key_pattern: str
    # metadata keys to add (values filled in at merge time)
    extra_meta_static: Dict[str, str]
    # human-readable label for the UI
    label: str

_H3_ADALN_RE = re.compile(
    r"^blocks\.(\d+)\.adaln_proj\.linear\.(bias|weight|weight_scale)$"
)

RECIPES: Dict[str, _Recipe] = {
    "h3_hybrid": _Recipe(
        name="h3_hybrid",
        architecture="MiniMax H3",
        overlay_key_pattern=r"^blocks\.(\d+)\.adaln_proj\.linear\.(bias|weight|weight_scale)$",
        extra_meta_static={
            "minimax_h3_hybrid": "baked",
        },
        label="Hybrid MiniMax H3",
    ),
}

# ---------------------------------------------------------------------------
# File-type detection (order-agnostic)
# ---------------------------------------------------------------------------

_FL2VA_MARKER = "fl2va"
_REF2VA_MARKER = "ref2va"


def _classify_h3_file(path: str) -> Optional[str]:
    """Return 'fl2va', 'ref2va', or None for a checkpoint path."""
    lower = os.path.basename(path).lower()
    if _FL2VA_MARKER in lower:
        return "fl2va"
    if _REF2VA_MARKER in lower:
        return "ref2va"
    return None


def _resolve_roles(base_path: str, overlay_path: str) -> Tuple[str, str]:
    """
    Return (fl2va_path, ref2va_path) regardless of which file the user
    designated as base/overlay.

    Raises ValueError when either file cannot be classified.
    """
    a = _classify_h3_file(base_path)
    b = _classify_h3_file(overlay_path)
    if a is None or b is None:
        raise ValueError(
            f"Cannot classify both files as fl2va/ref2va "
            f"({a!r}, {b!r}). Filenames must contain 'fl2va' or 'ref2va'."
        )
    if a == b:
        raise ValueError(
            f"Both files are classified as {a!r} — need one fl2va and one ref2va."
        )
    fl2va = base_path if a == "fl2va" else overlay_path
    ref2va = overlay_path if a == "fl2va" else base_path
    return fl2va, ref2va


# ---------------------------------------------------------------------------
# Overlay key set
# ---------------------------------------------------------------------------

def _compute_overlay_keys(
    base_manifest: Dict[str, Any],
    overlay_key_re: re.Pattern,
) -> set:
    """
    Determine which tensor keys should come from the overlay (ref2va) file.

    For h3_hybrid: all ``blocks.{i}.adaln_proj.linear.{bias,weight,weight_scale}``
    where ``i >= len(blocks) // 2``.
    """
    # Collect block indices from keys that match the pattern
    block_set: set = set()
    for key in base_manifest:
        m = overlay_key_re.match(key)
        if m:
            block_set.add(int(m.group(1)))
    if not block_set:
        return set()
    threshold = len(block_set) // 2
    overlay_keys: set = set()
    for key in base_manifest:
        m = overlay_key_re.match(key)
        if m and int(m.group(1)) >= threshold:
            overlay_keys.add(key)
    return overlay_keys


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_model_merge(payload: Dict[str, Any]) -> Iterable[Dict[str, str]]:
    """
    Stream model-level merge events.

    Required payload keys:
      base_path     : str   (one of the two H3 checkpoints)
      overlay_path  : str   (the other)
      architecture  : str   ("MiniMax H3")
      recipe        : str   ("h3_hybrid")

    Optional:
      output_path / output_name / output_dir / models_dir
      dry_run       : bool
      watermark     : bool (default True)
      custom_metadata : dict
    """
    base_path = os.path.realpath(os.path.expanduser(payload["base_path"]))
    overlay_path = os.path.realpath(os.path.expanduser(payload["overlay_path"]))
    architecture = payload.get("architecture") or "MiniMax H3"
    recipe_name = payload.get("recipe") or "h3_hybrid"
    dry_run = bool(payload.get("dry_run", False))
    do_watermark = payload.get("watermark", True)

    recipe = RECIPES.get(recipe_name)
    if recipe is None:
        yield _log(f"Unknown recipe: {recipe_name!r}\n")
        yield {"type": "done", "status": "failed"}
        return

    # Resolve fl2va/ref2va roles (order-agnostic)
    try:
        fl2va_path, ref2va_path = _resolve_roles(base_path, overlay_path)
    except ValueError as exc:
        yield _log(f"ERROR: {exc}\n")
        yield {"type": "done", "status": "failed"}
        return

    overlay_re = re.compile(recipe.overlay_key_pattern)

    # Default output name
    output_path = payload.get("output_path")
    if not output_path:
        output_dir = payload.get("output_dir") or payload.get("models_dir") or os.path.dirname(base_path)
        output_name = payload.get("output_name") or "minimax_h3_hybrid_merged.safetensors"
        if not output_name.endswith(".safetensors"):
            output_name += ".safetensors"
        output_path = os.path.join(output_dir, output_name)
    output_path = os.path.realpath(os.path.expanduser(output_path))

    yield _log(
        f"Model merge init\n"
        f"Recipe: {recipe.label} ({recipe_name})\n"
        f"fl2va  (base): {fl2va_path}\n"
        f"ref2va (overlay): {ref2va_path}\n"
        f"Output: {output_path}\n"
        f"Dry run: {'yes' if dry_run else 'no'}\n"
        f"Watermark: {'on' if do_watermark else 'off'}\n"
    )

    # Read manifests (header-only, cheap)
    base_manifest = read_safetensors_manifest(fl2va_path)
    overlay_manifest = read_safetensors_manifest(ref2va_path)

    base_keys = set(base_manifest)
    overlay_keys_available = set(overlay_manifest)

    yield _status(f"Inspected checkpoints: base {len(base_keys)} tensors, overlay {len(overlay_keys_available)} tensors")

    # Compute which keys come from the overlay
    overlay_set = _compute_overlay_keys(base_manifest, overlay_re)

    if not overlay_set:
        yield _log("No overlay keys matched — nothing to merge.\n")
        yield {"type": "done", "status": "failed"}
        return

    # The output key set is the BASE key set (streamed in base order). The
    # overlay file only needs to supply the overlay keys; extra overlay-only
    # keys are discarded. This keeps quant-layout marker tensors (e.g.
    # comfy_quant, weight_scale) that one parent quant carries but the other
    # does not from breaking the merge.
    missing_in_overlay = overlay_set - overlay_keys_available
    if missing_in_overlay:
        yield _log(
            f"ERROR: overlay is missing {len(missing_in_overlay)} overlay key(s): "
            f"{sorted(missing_in_overlay)[:5]}\n"
        )
        yield {"type": "done", "status": "failed"}
        return

    # Overlay tensors must match the base's shape/dtype for the same key,
    # otherwise the base-built header would not describe the written bytes.
    overlay_mismatches = [
        k for k in sorted(overlay_set)
        if (base_manifest[k].shape, base_manifest[k].dtype) != (
            overlay_manifest[k].shape,
            overlay_manifest[k].dtype,
        )
    ]
    if overlay_mismatches:
        k = overlay_mismatches[0]
        yield _log(
            f"ERROR: overlay key shape/dtype mismatch for {len(overlay_mismatches)} key(s), "
            f"e.g. {k!r}: base {base_manifest[k].dtype} {list(base_manifest[k].shape)} "
            f"vs overlay {overlay_manifest[k].dtype} {list(overlay_manifest[k].shape)}\n"
        )
        yield {"type": "done", "status": "failed"}
        return

    extra_in_overlay = overlay_keys_available - base_keys
    if extra_in_overlay:
        yield _log(
            f"Overlay carries {len(extra_in_overlay)} key(s) not present in the base "
            f"(sample: {sorted(extra_in_overlay)[:5]}); they are discarded — "
            f"output keeps the base key set.\n"
        )

    overlay_set_list = sorted(overlay_set)
    block_idxs = sorted({int(overlay_re.match(k).group(1)) for k in overlay_set_list})
    yield _log(
        f"Overlay keys: {len(overlay_set)} tensors "
        f"(blocks {block_idxs[0]}..{block_idxs[-1]}, "
        f"{len(block_idxs)} of the overlay set)\n"
    )

    # Build metadata: base __metadata__ + recipe static fields + provenance
    base_meta = _read_base_metadata(fl2va_path)
    meta: Dict[str, str] = dict(base_meta)
    meta.update(recipe.extra_meta_static)
    meta["base_model"] = os.path.basename(fl2va_path)
    meta["overlay_model"] = os.path.basename(ref2va_path)

    if do_watermark:
        wm = watermark_for(
            architecture,
            os.path.basename(output_path).replace(".safetensors", ""),
            output_path,
            bits="model-merge",
        )
        if wm:
            meta.update(wm)

    # Ensure spacer for future in-place edits
    if "__spacer" not in meta:
        meta["__spacer"] = " " * 2048

    if dry_run:
        yield _log(
            f"Dry run summary: {len(base_keys)} tensors total, "
            f"{len(overlay_set)} from overlay (ref2va), "
            f"{len(base_keys) - len(overlay_set)} from base (fl2va).\n"
            f"Metadata fields: {list(meta.keys())}\n"
        )
        yield _status("Dry run complete")
        yield {"type": "done", "status": "dry-run complete"}
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_output_path = output_path + ".tmp"

    # Stream: base order, overlay keys read from ref2va
    overlay_set_frozenset = frozenset(overlay_set)
    with open(tmp_output_path, "wb") as out_f:
        header = _build_safetensors_header(base_manifest, meta)
        out_f.write(struct.pack("<Q", len(header)))
        out_f.write(header)

        with safe_open(fl2va_path, framework="pt", device="cpu") as bf, \
             safe_open(ref2va_path, framework="pt", device="cpu") as of:
            for key in base_manifest:
                if key in overlay_set_frozenset:
                    tensor = of.get_tensor(key)
                else:
                    tensor = bf.get_tensor(key)
                _write_tensor_bytes(out_f, tensor)
                del tensor

    os.replace(tmp_output_path, output_path)

    yield _log(f"Wrote merged checkpoint: {output_path}\n")
    yield _log(
        f"Model merge complete: {len(base_keys)} tensors "
        f"({len(overlay_set)} overlay + {len(base_keys) - len(overlay_set)} base)\n"
    )
    yield _status("Model merge complete")
    yield {"type": "done", "status": "finished"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_base_metadata(file_path: str) -> Dict[str, str]:
    """Read __metadata__ from a safetensors file (header-only)."""
    with open(file_path, "rb") as f:
        (n8,) = struct.unpack("<Q", f.read(8))
        f.seek(8)
        header = json.loads(f.read(n8))
    md = header.get("__metadata__", {})
    return {str(k): str(v) for k, v in md.items()}


def _build_safetensors_header(
    manifest: Dict[str, Any],
    metadata: Dict[str, str],
) -> bytes:
    """Build a valid safetensors header JSON (tensor specs + __metadata__)."""
    from core.lora_merge_engine import _safetensors_dtype, _numel, _dtype_size

    header: Dict[str, Any] = {
        "__metadata__": {str(k): str(v) for k, v in metadata.items()}
    }
    offset = 0
    for key, info in manifest.items():
        dtype = _safetensors_dtype(info.dtype)
        size = _numel(tuple(info.shape)) * _dtype_size(dtype)
        header[key] = {
            "dtype": dtype,
            "shape": list(info.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    return json.dumps(header, separators=(",", ":")).encode("utf-8")


def _write_tensor_bytes(out_f: Any, tensor: torch.Tensor) -> None:
    contiguous = tensor.detach().cpu().contiguous()
    contiguous.view(torch.uint8).numpy().tofile(out_f)


def _log(text: str) -> Dict[str, str]:
    return {"type": "log", "text": text}


def _status(status: str) -> Dict[str, str]:
    return {"type": "status", "status": status}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_recipes() -> List[Dict[str, str]]:
    """Return recipe descriptors for the UI."""
    return [
        {
            "id": r.name,
            "label": r.label,
            "architecture": r.architecture,
        }
        for r in RECIPES.values()
    ]
