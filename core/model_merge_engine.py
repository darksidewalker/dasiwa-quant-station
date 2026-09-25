"""
Model-level merge engine (NOT LoRA math).

Recipes:

  h3_hybrid
    Base   : fl2va  (all tensors)
    Overlay: ref2va (blocks.{N}.adaln_proj.linear.{bias,weight,weight_scale}
             where N >= len(blocks)//2)
    Order-agnostic: filenames are inspected to auto-detect which is which.
    Metadata: base __metadata__ + minimax_h3_hybrid=baked
              + base_model/overlay_model filenames.

  h3_delta
    Base   : fl2va (all tensors)
    Overlay: ref2va — the full ref2va − fl2va weight delta, fused back
             into the fl2va base so one partition can serve both
             keyframe (fl2va) and reference (ref2va) conditioning.
    Two modes (payload `rank`):
      rank = 0 (default) — exact delta: W = fl + strength * (ref − fl),
             every shared tensor. Output ≈ ref2va-equivalent on the
             reference path, pristine fl2va where the delta is zero.
      rank = N  — SVD-rank-N (diffusers-modular / ethanfel style):
             randomized SVD of the delta on SVD-eligible 2-D matrices,
             exact (non-SVD) application for the non-compressible
             families — biases, RMSNorm weights, and the timestep
             conditioning table (pruned `adaln_t_table` / full-model
             `time_embedder.*`). W = fl + strength * Δ', with
             Δ' = rank-N reconstruction + exact families.
             Single streaming pass: the SVD energy report is accumulated
             during the write pass; the header (a size-bound all-spaces
             placeholder) is finalized in place afterwards, so each
             source file is read exactly once.
    Strength s: uniform scale on the delta (s=1.0 = full delta;
    s<1.0 blends back toward fl2va). s=0 is treated as unset → 1.0.
    Works on both pruned (adaln_t_table, 532 keys) and full
    (time_embedder MLP, 535 keys) key sets — detection is automatic
    from the base manifest; no pruned/full choice.
    Metadata: base __metadata__ + minimax_h3_delta=baked +
              h3_delta_{mode,rank,strength} + h3_delta_energy
              (per-family captured-energy JSON) + base/overlay names.

All merges stream tensor-by-tensor (no full model in memory).
"""

import json
import os
import re
import struct
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from utils.lora_inspector import read_safetensors_manifest
from core.metadata_manager import _LOADER_CONFIG_KEYS, merge_custom_metadata
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
    # runner name — "splice" (tensor copy) or "delta" (ref − fl math)
    kind: str = "splice"

_H3_ADALN_RE = re.compile(
    r"^blocks\.(\d+)\.adaln_proj\.linear\.(bias|weight|weight_scale)$"
)

# Default block range: 30-49 (last 20 of 50 blocks). Quality-first default
# based on community testing — FL2VA quality is preserved while still
# providing functional reference conditioning. Lower start values increase
# reference fidelity at the cost of raw output quality.
_H3_DEFAULT_BLOCK_START = 30
_H3_DEFAULT_BLOCK_END = 49

RECIPES: Dict[str, _Recipe] = {
    "h3_hybrid": _Recipe(
        name="h3_hybrid",
        architecture="MiniMax H3",
        overlay_key_pattern=r"^blocks\.(\d+)\.adaln_proj\.linear\.(bias|weight|weight_scale)$",
        extra_meta_static={
            "minimax_h3_hybrid": "baked",
        },
        label="Hybrid MiniMax H3",
        kind="splice",
    ),
    "h3_delta": _Recipe(
        name="h3_delta",
        architecture="MiniMax H3",
        # Delta uses every shared key; the pattern is only used for
        # validation, and ".*" matches all keys in the base.
        overlay_key_pattern=r".*",
        extra_meta_static={
            "minimax_h3_delta": "baked",
        },
        label="Delta-fused MiniMax H3 (fl2va + ref2va)",
        kind="delta",
    ),
}

# ---------------------------------------------------------------------------
# H3 tensor family classification (delta recipe)
#
# Families mirror the NVFP4 preserve-table philosophy: the
# non-compressible conditioning families are applied exactly (or via
# rank-0 SVD, which is exact) even in SVD mode; the heavy 2-D trunk
# matrices are where the rank cap actually compresses.
# ---------------------------------------------------------------------------

_H3_EXACT_FAMILIES = {
    # timestep conditioning: pruned adaln_t_table or full time_embedder
    "timestep_table", "time_embedder",
    # RMS / layer norms
    "norm",
    # biases (1-D, incompressible by 2-D SVD)
    "bias",
    # structural buffers
    "rope",
}


def _classify_h3_family(key: str) -> str:
    """
    Classify an H3 tensor key into a merge family.

    Family sets:
      exact (incompressible / must stay exact):
        timestep_table — pruned adaln_t_table (8-wide curve)
        time_embedder — full-model timestep MLP
        norm         — layer/RMS norm scales (norm1/norm2, q/k_norm,
                       final norms); all 1-D, SVD cannot compress
        bias         — 1-D biases
        rope         — inv_freq buffer
      SVD-eligible (rank cap applies):
        ada  — adaln_proj.linear.weight (2-D; tiny rank in practice)
        attn — qkv_proj / out_proj (q/k_norm lands in 'norm')
        mlp  — fc1 / fc2
        out  — final_layer.video_out / audio_out
        proj — video_patch_proj / audio_patch_proj / condition_proj

    Prefix/structural checks first, then norm detection, then
    adaln_proj (so its bias stays in 'ada', not 'bias'), then the
    leaf-name fallbacks.
    """
    if "adaln_t_table" in key:
        return "timestep_table"
    if key.startswith("time_embedder.") or ".time_embedder." in key:
        return "time_embedder"
    if key == "rope.inv_freq" or key.startswith("rope.") or ".rope." in key:
        return "rope"
    # Norm scales: block norms, final norms, q/k norms, final_layer.norm
    if (
        re.search(r"(^|\.)norm[12]\.", key)
        or re.search(r"(^|\.)final_norm\.", key)
        or re.search(r"(^|\.)attn\.[qk]_norm\.", key)
        or re.search(r"(^|\.)norm\.(weight|bias)$", key)
    ):
        return "norm"
    # AdaLN modulation — before the bias fallback so adaln_proj biases
    # report under 'ada' instead of 'bias' (both are exact families;
    # only the energy-report bucket differs).
    if "adaln_proj" in key:
        return "ada"
    if key.endswith(".bias"):
        return "bias"
    if re.search(r"(^|\.)final_layer\.(video_out|audio_out)\.", key):
        return "out"
    if ".mlp." in key or key.startswith("mlp."):
        return "mlp"
    if ".attn." in key or key.startswith("attn."):
        return "attn"
    if re.match(r"^(video|audio)_patch_proj\.", key) or re.match(r"^condition_proj\.", key):
        return "proj"
    return "attn"


def _is_svd_eligible(key: str, shape: Tuple[int, ...]) -> bool:
    """
    2-D matrices that a rank cap can actually compress.
    """
    if len(shape) != 2:
        return False
    fam = _classify_h3_family(key)
    return fam not in _H3_EXACT_FAMILIES

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
    block_start: int = _H3_DEFAULT_BLOCK_START,
    block_end: int = _H3_DEFAULT_BLOCK_END,
    final_adaln_from_overlay: bool = False,
    overlay_blocks: Optional[set] = None,
) -> Tuple[set, int, int]:
    """
    Determine which tensor keys should come from the overlay (ref2va) file.

    For h3_hybrid: all ``blocks.{i}.adaln_proj.linear.{bias,weight,weight_scale}``
    in the inclusive range, or exactly the indices in ``overlay_blocks`` when
    given. The optional final AdaLN overlay remains independent of this choice.
    """
    # Collect block indices from keys that match the pattern
    block_set: set = set()
    for key in base_manifest:
        m = overlay_key_re.match(key)
        if m:
            block_set.add(int(m.group(1)))
    if not block_set:
        return set(), 0, 0

    # Validate range against actual block count
    total_blocks = len(block_set)
    max_block = total_blocks - 1
    start = min(max(block_start, 0), max_block)
    end = min(max(block_end, start), max_block)

    overlay_keys: set = set()
    for key in base_manifest:
        m = overlay_key_re.match(key)
        if m and (int(m.group(1)) in overlay_blocks if overlay_blocks is not None
                  else start <= int(m.group(1)) <= end):
            overlay_keys.add(key)

    # Optional final_layer.adaln_proj overlay (max-reference mode)
    if final_adaln_from_overlay:
        final_adaln_re = re.compile(r"^final_layer\.adaln_proj\.linear\.(bias|weight)$")
        for key in base_manifest:
            if final_adaln_re.match(key):
                overlay_keys.add(key)

    return overlay_keys, start, end


def _compute_blend_weights(
    base_manifest: Dict[str, Any],
    overlay_key_re: re.Pattern,
    block_start: int,
    block_end: int,
    blend_mode: str,
    final_adaln_from_overlay: bool = False,
) -> Tuple[Dict[str, Optional[float]], int, int]:
    """
    Compute per-key blend weights for hybrid merge.

    Returns (weights_dict, actual_start, actual_end).

    Modes:
      binary: weight 0.0 or 1.0 (original behavior)
      larv:   LARV (Layer-wise Adaptive Rescaling Veneer) per arXiv:2602.09413.
             Computes per-layer information richness and conflict diagnostics
             to scale overlay contribution adaptively. Returns None weights to
             signal "compute at merge time" for all matching keys.

    Note: larv requires reading tensor data, so it's computed during the
    streaming pass, not here. This function returns None weights for those
    keys to signal "compute at merge time".
    """
    # Collect block indices
    block_set: set = set()
    for key in base_manifest:
        m = overlay_key_re.match(key)
        if m:
            block_set.add(int(m.group(1)))
    if not block_set:
        return {}, 0, 0

    total_blocks = len(block_set)
    max_block = total_blocks - 1
    start = min(max(block_start, 0), max_block)
    end = min(max(block_end, start), max_block)

    weights: Dict[str, Optional[float]] = {}

    if blend_mode == "larv":
        # LARV: signal "compute at merge time" for all matching keys
        # Scale computed per-block from weight diagnostics (effective rank + commutator conflict)
        for key in base_manifest:
            m = overlay_key_re.match(key)
            if not m:
                continue
            block_idx = int(m.group(1))
            if start <= block_idx <= end:
                weights[key] = None  # compute during streaming pass
            else:
                weights[key] = 0.0 if block_idx < start else 1.0

    else:  # binary (default)
        for key in base_manifest:
            m = overlay_key_re.match(key)
            if m and start <= int(m.group(1)) <= end:
                weights[key] = 1.0

    if final_adaln_from_overlay:
        final_adaln_re = re.compile(r"^final_layer\.adaln_proj\.linear\.(bias|weight)$")
        for key in base_manifest:
            if final_adaln_re.match(key):
                weights[key] = 1.0

    return weights, start, end


# ---------------------------------------------------------------------------
# Delta runner (h3_delta)
# ---------------------------------------------------------------------------

def _h3_variant(base_manifest: Dict[str, Any]) -> str:
    """'pruned' (adaln_t_table) or 'full' (time_embedder) H3 key set."""
    keys = set(base_manifest)
    if any(k == "adaln_t_table" or k.endswith(".adaln_t_table") for k in keys):
        return "pruned"
    if any(k.startswith("time_embedder.") for k in keys):
        return "full"
    return "unknown"


def _randomized_svd_cap(
    mat: torch.Tensor, rank: int
) -> Tuple[torch.Tensor, float, int]:
    """
    Randomized SVD reconstruction of a 2-D matrix, capped at `rank`.

    Returns (approx, captured_energy_fraction, achieved_rank).
    Deterministic per input (fixed seed) so re-runs are reproducible.
    """
    r = min(rank, min(mat.shape))
    # Oversampled power-iteration SVD: sketch against a fixed random
    # Gaussian, iterate a few powers, QR-reduce, and SVD the small
    # projected matrix. Rank <= r on either dimension.
    mat32 = mat.to(torch.float32)
    if r <= 0:
        return mat32.new_zeros(mat.shape), 0.0, 0
    if r >= min(mat.shape):
        # Cap does not bind: full-rank reconstruction is the matrix itself,
        # so the approximation is exact and all energy is captured.
        return mat32.clone(), 1.0, r
    k = min(r + 5, min(mat.shape))
    gen = torch.Generator(device=mat.device).manual_seed(42)
    omega = torch.randn(mat.shape[1], k, generator=gen,
                        dtype=torch.float32, device=mat.device)
    y = mat32 @ omega
    for _ in range(3):  # power iteration
        y = mat32.t() @ y
        y = mat32 @ y
    q, _ = torch.linalg.qr(y)
    b = q.t() @ mat32
    ub, S, vh = torch.linalg.svd(b, full_matrices=False)
    S = S[:r]
    U = q @ ub[:, :r]
    approx = (U * S) @ vh[:r]
    full_energy = float((mat32.pow(2).sum()))
    cap_energy = float((S.pow(2).sum()))
    return approx, (cap_energy / full_energy) if full_energy > 0 else 1.0, r


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    a_flat = a.flatten().to(torch.float32)
    b_flat = b.flatten().to(torch.float32)
    dot = torch.dot(a_flat, b_flat)
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float((dot / (norm_a * norm_b)).clamp(-1.0, 1.0))


def _effective_rank(matrix: torch.Tensor, k: int = 32) -> float:
    """Effective rank from the small Gram matrix, avoiding a full SVD."""
    if matrix.ndim != 2 or matrix.numel() == 0:
        return 1.0
    m32 = matrix.to(torch.float32)
    # AdaLN projections are extremely tall (e.g. 96,768 × 8). Their singular
    # values equal sqrt(eigenvalues(MᵀM)), so this keeps the largest temporary
    # bounded by the feature dimension rather than allocating a tall QR basis.
    gram = m32.t() @ m32
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min_(0)
    singular_values = eigenvalues.sqrt()
    energy = singular_values.pow(2)
    total_energy = energy.sum()
    if total_energy <= 1e-10:
        return 1.0
    probabilities = (energy / total_energy).clamp_min_(1e-10)
    entropy = -(probabilities * probabilities.log()).sum()
    return float(torch.exp(entropy).item())


def _commutator_conflict(base: torch.Tensor, delta: torch.Tensor) -> float:
    """Compute normalized commutator conflict coefficient.

    Measures non-commutativity between base operator and update.
    Higher values indicate more potential interference.
    """
    b32 = base.to(torch.float32)
    d32 = delta.to(torch.float32)
    norm_b = torch.norm(b32).item()
    norm_d = torch.norm(d32).item()
    if norm_b < 1e-10 or norm_d < 1e-10:
        return 0.0

    # For rectangular matrices, the paper averages left and right Gram
    # commutators. The left term normally has rows² entries; its squared
    # Frobenius norm is exactly recoverable from small feature-space Grams:
    # ||AΔᵀ - ΔAᵀ||² = 2 tr((AᵀA)(ΔᵀΔ)) - 2 tr((AᵀΔ)²).
    base_gram = b32.t() @ b32
    delta_gram = d32.t() @ d32
    cross_gram = b32.t() @ d32
    left_sq = 2.0 * torch.trace(base_gram @ delta_gram) - 2.0 * torch.trace(cross_gram @ cross_gram)
    left_norm = torch.sqrt(left_sq.clamp_min(0.0)) / (norm_b * norm_d + 1e-10)
    right_norm = torch.norm(b32.t() @ d32 - d32.t() @ b32) / (norm_b * norm_d + 1e-10)
    return float(0.5 * (left_norm + right_norm))


def _larv_tier_scales(scores: Dict[int, float]) -> Dict[int, float]:
    """H3-safe LARV terciles; never extrapolate the direct ref2va checkpoint."""
    if not scores:
        return {}
    values = torch.tensor(list(scores.values()), dtype=torch.float32)
    lower, upper = torch.quantile(values, torch.tensor([1 / 3, 2 / 3]))
    return {
        block: 0.5 if score <= lower else 0.75 if score <= upper else 1.0
        for block, score in scores.items()
    }


def _larv_block_metrics(base_tensor: torch.Tensor, delta_tensor: torch.Tensor) -> Tuple[float, float]:
    """Return paper-defined (effective-rank contrast, commutator conflict)."""
    if torch.norm(delta_tensor.to(torch.float32)) < 1e-10:
        return 0.0, 0.0
    effective_rank_contrast = _effective_rank(base_tensor) / (_effective_rank(delta_tensor) + 1e-8)
    return effective_rank_contrast, _commutator_conflict(base_tensor, delta_tensor)


def _larv_block_scales(
    metrics: Dict[int, Tuple[float, float]],
    total_layers: int,
    gate: str,
    eta: float = 1.0,
    rho: float = 0.5,
    gamma: float = 3.0,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Apply LARV Eq. 8 across all selected H3 blocks, then gate its scores."""
    if not metrics:
        return {}, {}
    blocks = sorted(metrics)
    conflicts = torch.tensor([metrics[block][1] for block in blocks], dtype=torch.float32)
    std = conflicts.std(correction=0)
    standardized_conflicts = (conflicts - conflicts.mean()) / (std if std > 1e-8 else 1.0)
    scores: Dict[int, float] = {}
    for block, standardized_conflict in zip(blocks, standardized_conflicts.tolist()):
        effective_rank_contrast = metrics[block][0]
        depth_prior = (block + 1) / max(total_layers, 1)
        score = (effective_rank_contrast ** eta) * (depth_prior ** rho)
        score /= float(torch.nn.functional.softplus(torch.tensor(standardized_conflict)).item()) + 1.0
        scores[block] = score
    if gate == "tiered":
        return _larv_tier_scales(scores), scores
    # H3 hybrid has a direct, viable ref2va endpoint. Unlike the paper's
    # multi-task task-vector setting, scaling past 1.0 extrapolates beyond that
    # checkpoint and can corrupt AdaLN modulation (black video). Keep LARV as
    # a non-extrapolating interpolation for this architecture.
    scales = {
        block: min(1.0, 1.0 + 0.5 * float(torch.tanh(gamma * (torch.tensor(score) - 1.0)).item()))
        for block, score in scores.items()
    }
    return scales, scores


def _analyze_h3_larv_blocks(
    fl2va_path: str,
    ref2va_path: str,
    block_indices: Iterable[int],
    gate: str,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, Tuple[float, float]]]:
    """Read one AdaLN weight per selected block and return bounded LARV scales."""
    metrics: Dict[int, Tuple[float, float]] = {}
    with safe_open(fl2va_path, framework="pt", device="cpu") as bf, \
         safe_open(ref2va_path, framework="pt", device="cpu") as of:
        for block in sorted(set(block_indices)):
            key = f"blocks.{block}.adaln_proj.linear.weight"
            base = bf.get_tensor(key).to(torch.float32)
            delta = of.get_tensor(key).to(torch.float32) - base
            metrics[block] = _larv_block_metrics(base, delta)
            del base, delta
    scales, scores = _larv_block_scales(metrics, total_layers=50, gate=gate)
    return scales, scores, metrics


def _run_h3_delta(
    fl2va_path: str,
    ref2va_path: str,
    base_manifest: Dict[str, Any],
    output_path: str,
    rank: int,
    strength: float,
    dry_run: bool,
    meta: Dict[str, str],
) -> Iterable[Dict[str, str]]:
    """
    Compute W = fl + s * Δ (Δ = ref − fl, optionally rank-capped) for
    every base key, streamed tensor-by-tensor into ``output_path``.

    rank = 0  → exact delta, every tensor (single streaming pass).
    rank  = N → SVD rank-N on SVD-eligible 2-D matrices; exact
                application for norm/bias/timestep/rope families.
                The energy report is accumulated DURING the write pass
                (single read of both sources); the header is written
                first as a size-bound all-spaces placeholder, then
                finalized in place so the file carries
                ``h3_delta_energy`` without a second source read.
    The output key/shape/dtype set is the base's: deltas never change
    shape, and each result is cast back to the base tensor dtype.
    """
    keys = list(base_manifest.keys())
    rank = max(0, int(rank))
    strength = float(strength)
    mode = "exact" if rank == 0 else f"svd-r{rank}"
    total_keys = len(keys)

    if dry_run:
        yield _log(
            f"h3_delta dry run: {total_keys} tensors, mode={mode}, "
            f"strength={strength}, variant={_h3_variant(base_manifest)}\n"
        )
        yield {"type": "done", "status": "dry-run complete"}
        return

    # ------------------------------------------------------------------
    # Header + single streaming pass
    #
    # SVD mode: the energy report is accumulated DURING this pass (no
    # separate source re-read). The header is written first as a
    # size-reserving placeholder (a 4 KiB spacer), then re-serialized
    # and rewritten in place after the pass, so the file carries
    # h3_delta_energy without a second full source read — the same
    # spacer-padding trick as metadata_manager._try_inplace_metadata_rewrite.
    # Exact mode (rank=0): header is final and written directly.
    # ------------------------------------------------------------------
    meta["h3_delta_mode"] = mode
    meta["h3_delta_rank"] = str(rank)
    meta["h3_delta_strength"] = f"{strength:.6f}"
    meta["h3_delta_variant"] = _h3_variant(base_manifest)

    svd_counts = 0
    zero_delta_keys = 0
    exact_family_counts: Dict[str, int] = {}
    energy_scratch: Dict[str, Dict[str, float]] = {}

    if rank > 0:
        # Reserve header room for the post-pass energy report.
        meta["__spacer"] = " " * 4096
    else:
        if "__spacer" not in meta:
            meta["__spacer"] = " " * 2048

    header_bytes = _build_safetensors_header(base_manifest, meta)
    header_size = len(header_bytes)

    tmp_output_path = output_path + ".tmp"
    os.makedirs(os.path.dirname(tmp_output_path) or ".", exist_ok=True)

    progress_every = max(1, total_keys // 100)
    t0 = time.monotonic()

    with open(tmp_output_path, "wb") as out_f:
        out_f.write(struct.pack("<Q", header_size))
        out_f.write(header_bytes)
        with safe_open(fl2va_path, framework="pt", device="cpu") as bf, \
             safe_open(ref2va_path, framework="pt", device="cpu") as of:
            for i, key in enumerate(keys):
                base_t = bf.get_tensor(key)
                ref_t = of.get_tensor(key)
                # fp32 delta for numerical stability; cast back to base dtype.
                base32 = base_t.to(torch.float32)
                ref32 = ref_t.to(torch.float32)
                delta = ref32 - base32

                if rank > 0 and _is_svd_eligible(key, tuple(base_t.shape)):
                    approx, cap_frac, _ach = _randomized_svd_cap(delta, rank)
                    delta = approx
                    svd_counts += 1
                    fam = _classify_h3_family(key)
                    rec = energy_scratch.setdefault(fam, {"cap": 0.0, "n": 0})
                    rec["cap"] += cap_frac
                    rec["n"] += 1
                elif rank > 0:
                    exact_family_counts[_classify_h3_family(key)] = \
                        exact_family_counts.get(_classify_h3_family(key), 0) + 1

                if torch.equal(ref32, base32):
                    zero_delta_keys += 1

                out = (base32 + strength * delta).to(base_t.dtype)
                _write_tensor_bytes(out_f, out)
                del out, base_t, ref_t, base32, ref32, delta

                done = i + 1
                if done % progress_every == 0 or done == total_keys:
                    elapsed = time.monotonic() - t0
                    eta = elapsed * (total_keys - done) / done
                    progress_msg = (
                        f"h3_delta {done}/{total_keys} tensors "
                        f"(svd={svd_counts}, zero-delta={zero_delta_keys}, "
                        f"elapsed {_format_duration(elapsed)}, "
                        f"eta {_format_duration(eta)})"
                    )
                    # Self-updating single line in the console window; no
                    # status-bar output, nothing appended to the scrollback.
                    yield _progress(progress_msg)

    if rank > 0:
        # Finalize the header in place: add the energy report, adjust the
        # spacer so the final JSON lands exactly at the reserved size.
        energy_report = {
            fam: {"avg_captured": round(rec["cap"] / rec["n"], 6), "tensors": rec["n"]}
            for fam, rec in energy_scratch.items()
        }
        meta["h3_delta_energy"] = json.dumps(energy_report, sort_keys=True)
        final_header = _build_safetensors_header(base_manifest, meta)
        for _ in range(2):  # spacer length changes serialize 1:1; one adjust suffices
            align = header_size - len(final_header)
            if align == 0:
                break
            spacer = meta["__spacer"]
            if align > 0:
                spacer = spacer + " " * align
            else:
                if len(spacer) + align < 0:
                    raise RuntimeError("Could not align h3_delta header size via spacer")
                spacer = spacer[: len(spacer) + align]
            meta["__spacer"] = spacer
            final_header = _build_safetensors_header(base_manifest, meta)
        if len(final_header) != header_size:
            raise RuntimeError(
                f"h3_delta header finalization misaligned "
                f"({len(final_header)} vs {header_size})"
            )
        with open(tmp_output_path, "r+b") as f:
            f.seek(0)
            f.write(struct.pack("<Q", header_size))
            f.write(final_header)

    os.replace(tmp_output_path, output_path)

    elapsed_total = time.monotonic() - t0
    yield _log(
        f"h3_delta complete: {total_keys} tensors "
        f"(svd-approx={svd_counts}, exact-families="
        f"{sum(exact_family_counts.values()) if rank > 0 else total_keys}, "
        f"zero-delta={zero_delta_keys}), mode={mode}, strength={strength}, "
        f"elapsed {_format_duration(elapsed_total)}, output={output_path}\n"
    )
    yield {"type": "done", "status": "finished"}


def _write_model_merge_recipe(
    output_path: str,
    recipe_name: str,
    architecture: str,
    fl2va_path: str,
    ref2va_path: str,
    preserve_loader_metadata: bool,
    watermark: bool,
    details: List[str],
) -> str:
    """Write the parameters actually used, next to a completed checkpoint."""
    recipe_path = os.path.splitext(output_path)[0] + ".txt"
    lines = [
        "DaSiWa Quant Station Model Merge Recipe",
        "",
        f"Recipe: {recipe_name}",
        f"Architecture: {architecture}",
        f"Base checkpoint: {fl2va_path}",
        f"Overlay checkpoint: {ref2va_path}",
        f"Output checkpoint: {output_path}",
        *details,
        f"Preserve loader metadata: {'yes' if preserve_loader_metadata else 'no'}",
        f"Watermark: {'on (requested)' if watermark else 'off'}",
    ]
    tmp_path = recipe_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    os.replace(tmp_path, recipe_path)
    return recipe_path

def _splice_h3_adaln_modalities(base: torch.Tensor, overlay: torch.Tensor,
                                modalities: Tuple[int, ...]) -> torch.Tensor:
    """Copy complete video/text/audio AdaLN banks without interpolating weights."""
    rows = base.shape[0] // 3
    result = base.clone().reshape(3, rows, *base.shape[1:])
    source = overlay.reshape(3, rows, *overlay.shape[1:])
    for modality in modalities:
        result[modality].copy_(source[modality])
    return result.reshape(base.shape)


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
    preserve_loader_metadata = bool(payload.get("preserve_loader_metadata", True))

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

    # ------------------------------------------------------------------
    # Delta recipe (h3_delta): W = fl + s * (ref - fl) over every tensor.
    # Both files must carry the same key set (shape + dtype) because the
    # delta is computed for all base tensors; no overlay-key subset.
    # ------------------------------------------------------------------
    if recipe.kind == "delta":
        missing = base_keys - overlay_keys_available
        if missing:
            yield _log(
                f"ERROR: delta recipe needs every base tensor in the overlay; "
                f"overlay is missing {len(missing)} key(s): "
                f"{sorted(missing)[:5]}\n"
            )
            yield {"type": "done", "status": "failed"}
            return
        mismatched = [
            k for k in sorted(base_keys)
            if (base_manifest[k].shape, base_manifest[k].dtype) != (
                overlay_manifest[k].shape, overlay_manifest[k].dtype,
            )
        ]
        if mismatched:
            k = mismatched[0]
            yield _log(
                f"ERROR: shape/dtype mismatch on {len(mismatched)} key(s), "
                f"e.g. {k!r}: base {base_manifest[k].dtype} {list(base_manifest[k].shape)} "
                f"vs overlay {overlay_manifest[k].dtype} {list(overlay_manifest[k].shape)}\n"
            )
            yield {"type": "done", "status": "failed"}
            return

        rank_raw = payload.get("rank", 0)
        rank = int(rank_raw) if str(rank_raw).lstrip("-").isdigit() else 0
        strength_raw = payload.get("strength", 1.0)
        try:
            strength = float(strength_raw)
        except (TypeError, ValueError):
            strength = 1.0
        if strength == 0.0:  # 0 is treated as unset → full delta
            strength = 1.0

        # Build metadata before the header (delta runner writes it in).
        base_meta = _read_base_metadata(fl2va_path, preserve_loader_metadata)
        delta_meta: Dict[str, str] = dict(base_meta)
        delta_meta.update(recipe.extra_meta_static)
        delta_meta["base_model"] = os.path.basename(fl2va_path)
        delta_meta["overlay_model"] = os.path.basename(ref2va_path)
        if do_watermark:
            wm = watermark_for(
                architecture,
                os.path.basename(output_path).replace(".safetensors", ""),
                output_path,
                bits="model-merge",
            )
            if wm:
                delta_meta.update(wm)
        if "__spacer" not in delta_meta:
            delta_meta["__spacer"] = " " * 2048

        yield _log(
            f"h3_delta init: mode=rank {rank} "
            f"({'exact' if rank == 0 else 'SVD-capped'}), strength={strength}, "
            f"variant auto-detect from base\n"
        )
        for event in _run_h3_delta(
            fl2va_path, ref2va_path, base_manifest,
            output_path, rank, strength, dry_run, delta_meta,
        ):
            if event.get("type") == "done" and event.get("status") == "finished":
                try:
                    recipe_path = _write_model_merge_recipe(
                        output_path, recipe_name, architecture, fl2va_path, ref2va_path,
                        preserve_loader_metadata, do_watermark,
                        [f"Delta rank: {max(0, rank)}",
                         f"Delta mode: {'exact' if rank <= 0 else f'svd-r{rank}'}",
                         f"Delta strength: {strength}",
                         f"Model variant: {_h3_variant(base_manifest)}",
                         f"Output tensors: {len(base_manifest)}"],
                    )
                except OSError as exc:
                    yield _log(f"ERROR: checkpoint created but could not write merge recipe: {exc}\n")
                    yield {"type": "done", "status": "failed"}
                    return
                yield _log(f"Wrote model merge recipe: {recipe_path}\n")
            yield event
        return

    # Compute which keys come from the overlay (splice recipe)
    block_start = int(payload.get("block_range_start", _H3_DEFAULT_BLOCK_START))
    block_end = int(payload.get("block_range_end", _H3_DEFAULT_BLOCK_END))
    final_adaln_from_overlay = bool(payload.get("final_adaln_from_overlay", False))
    blend_mode = payload.get("blend_mode", "binary")
    if blend_mode != "binary":
        yield _log(
            "ERROR: LARV is disabled for MiniMax H3. Even non-extrapolating "
            "LARV interpolation produced black videos because H3 AdaLN weights "
            "are not validated task vectors. Use Binary hard switch.\n"
        )
        yield {"type": "done", "status": "failed"}
        return

    # Exact row-bank selection is experimental and distinct from the legacy
    # whole-tensor switch. The grid always selects video banks in this mode.
    modality_mode = payload.get("modality_mode", "whole")
    audio_blocks = payload.get("audio_blocks", "all")
    text_blocks = payload.get("text_blocks", "all")
    audio_out_from_overlay = payload.get("audio_out_from_overlay", False)
    if (modality_mode not in ("whole", "split") or
            audio_blocks not in ("all", "selected", "base") or
            text_blocks not in ("all", "selected", "base") or
            type(audio_out_from_overlay) is not bool):
        yield _log("ERROR: invalid H3 modality selection.\n")
        yield {"type": "done", "status": "failed"}
        return
    if modality_mode == "whole" and (audio_out_from_overlay or
                                     audio_blocks != "all" or text_blocks != "all"):
        yield _log("ERROR: audio/text bank options require modality split mode.\n")
        yield {"type": "done", "status": "failed"}
        return

    # Explicit block lists override the legacy contiguous range. Validate against
    # actual checkpoint indices: never silently clamp/drop a requested block.
    custom_blocks = None
    if "overlay_blocks" in payload:
        selected = payload["overlay_blocks"]
        available = {int(match.group(1)) for key in base_manifest if (match := overlay_re.match(key))}
        if (not isinstance(selected, list) or not selected
                or any(type(index) is not int for index in selected)
                or len(selected) != len(set(selected))
                or not set(selected) <= available):
            yield _log("ERROR: overlay_blocks must be a non-empty list of unique available block indices.\n")
            yield {"type": "done", "status": "failed"}
            return
        custom_blocks = set(selected)

    overlay_set, actual_start, actual_end = _compute_overlay_keys(
        base_manifest, overlay_re, block_start, block_end, final_adaln_from_overlay,
        custom_blocks,
    )
    video_idxs = {int(match.group(1)) for key in overlay_set
                  if (match := overlay_re.match(key))}
    available_idxs = {int(match.group(1)) for key in base_manifest
                      if (match := overlay_re.match(key))}
    if not video_idxs:
        yield _log("ERROR: no requested video AdaLN blocks found in base checkpoint.\n")
        yield {"type": "done", "status": "failed"}
        return
    audio_idxs = (available_idxs if audio_blocks == "all" else
                  video_idxs if audio_blocks == "selected" else set()) if modality_mode == "split" else set()
    text_idxs = (available_idxs if text_blocks == "all" else
                 video_idxs if text_blocks == "selected" else set()) if modality_mode == "split" else set()
    modality_keys = {}
    if modality_mode == "split":
        touched = video_idxs | audio_idxs | text_idxs
        for block in sorted(touched):
            sources = tuple(modality for modality, selected in
                            ((0, video_idxs), (1, text_idxs), (2, audio_idxs)) if block in selected)
            for suffix in ("weight", "bias"):
                key = f"blocks.{block}.adaln_proj.linear.{suffix}"
                if key not in base_manifest or key not in overlay_manifest:
                    yield _log(f"ERROR: modality split needs matching AdaLN weight/bias: {key}\n")
                    yield {"type": "done", "status": "failed"}
                    return
                spec = base_manifest[key]
                other = overlay_manifest[key]
                if (spec.shape != other.shape or spec.dtype != other.dtype or
                        len(spec.shape) != (2 if suffix == "weight" else 1) or
                        spec.shape[0] == 0 or spec.shape[0] % 18 or
                        spec.dtype not in ("F16", "BF16", "F32")):
                    yield _log(f"ERROR: unsupported modality-split AdaLN layout: {key}\n")
                    yield {"type": "done", "status": "failed"}
                    return
                modality_keys[key] = sources
            weight_shape = base_manifest[f"blocks.{block}.adaln_proj.linear.weight"].shape
            bias_shape = base_manifest[f"blocks.{block}.adaln_proj.linear.bias"].shape
            if weight_shape[0] != bias_shape[0]:
                yield _log(f"ERROR: AdaLN weight/bias row mismatch in block {block}\n")
                yield {"type": "done", "status": "failed"}
                return
            for suffix in ("weight_scale",):
                key = f"blocks.{block}.adaln_proj.linear.{suffix}"
                if key in base_manifest or key in overlay_manifest:
                    yield _log(f"ERROR: quantized AdaLN cannot be split by modality: {key}\n")
                    yield {"type": "done", "status": "failed"}
                    return
        overlay_set = (overlay_set - {key for key in overlay_set if overlay_re.match(key)}) | set(modality_keys)
        if audio_out_from_overlay:
            for suffix in ("weight", "bias"):
                key = f"final_layer.audio_out.{suffix}"
                if key not in base_manifest or key not in overlay_manifest:
                    yield _log(f"ERROR: audio output head missing: {key}\n")
                    yield {"type": "done", "status": "failed"}
                    return
                overlay_set.add(key)
    weights = {k: 1.0 for k in overlay_set}
    is_larv = False  # Kept for the shared streaming branch; unsafe LARV is rejected above.

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
    block_match_keys = [k for k in overlay_set_list if overlay_re.match(k)]
    block_idxs = (sorted(video_idxs) if modality_mode == "split" else
                  sorted({int(match.group(1)) for key in block_match_keys
                          if (match := overlay_re.match(key))}))
    final_included = any(k.startswith("final_layer.adaln_proj") for k in overlay_set_list)
    block_description = (",".join(map(str, block_idxs)) if custom_blocks is not None
                         else f"{block_idxs[0]}..{block_idxs[-1]}")
    yield _log(
        f"Overlay keys: {len(overlay_set)} tensors "
        f"(blocks {block_description}, "
        f"{len(block_idxs)} blocks"
        + (", final_layer.adaln_proj included" if final_included else "")
        + ")\n"
    )

    if modality_mode == "split":
        yield _log(f"Experimental modality split: video={block_idxs}, "
                   f"text={sorted(text_idxs)}, audio={sorted(audio_idxs)}, "
                   f"ref2va audio_out={audio_out_from_overlay}. "
                   "This has not been validated for audio fidelity or visual quality.\n")

    larv_scales: Dict[int, float] = {}
    larv_scores: Dict[int, float] = {}
    if is_larv:
        gate = "tiered" if blend_mode == "larv_tiered" else "continuous"
        yield _log(f"LARV {gate} analysis: {len(block_idxs)} AdaLN block weights (paper metrics)\n")
        larv_scales, larv_scores, _ = _analyze_h3_larv_blocks(
            fl2va_path, ref2va_path, block_idxs, gate,
        )
        tier_counts = {scale: sum(value == scale for value in larv_scales.values()) for scale in (0.5, 0.75, 1.0)}
        if gate == "tiered":
            yield _log(
                "LARV H3-safe tiered scales: "
                f"0.5×={tier_counts[0.5]}, 0.75×={tier_counts[0.75]}, 1.0×={tier_counts[1.0]}\n"
            )
        else:
            yield _log(
                f"LARV continuous scales: min={min(larv_scales.values()):.3f}, "
                f"max={max(larv_scales.values()):.3f}\n"
            )

    # Build metadata: base __metadata__ + recipe static fields + provenance
    base_meta = _read_base_metadata(fl2va_path, preserve_loader_metadata)
    meta: Dict[str, str] = dict(base_meta)
    meta.update(recipe.extra_meta_static)
    meta["base_model"] = os.path.basename(fl2va_path)
    meta["overlay_model"] = os.path.basename(ref2va_path)
    if recipe.kind == "splice":
        meta["h3_hybrid_block_start"] = str(actual_start)
        meta["h3_hybrid_block_end"] = str(actual_end)
        meta["h3_hybrid_block_selection"] = "custom" if custom_blocks is not None else "range"
        meta["h3_hybrid_overlay_blocks"] = json.dumps(block_idxs)
        meta["h3_hybrid_blend_mode"] = blend_mode
        if modality_mode == "split":
            meta["h3_hybrid_modality_mode"] = "split"
            meta["h3_hybrid_video_blocks"] = json.dumps(block_idxs)
            meta["h3_hybrid_audio_blocks"] = json.dumps(sorted(audio_idxs))
            meta["h3_hybrid_text_blocks"] = json.dumps(sorted(text_idxs))
            meta["h3_hybrid_audio_out"] = "ref2va" if audio_out_from_overlay else "fl2va"
        if is_larv:
            meta["h3_hybrid_larv_gate"] = "tiered" if blend_mode == "larv_tiered" else "continuous"
            meta["h3_hybrid_larv_scales"] = json.dumps(larv_scales, sort_keys=True)
            meta["h3_hybrid_larv_scores"] = json.dumps(larv_scores, sort_keys=True)

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
            f"{len(modality_keys)} modality-sliced, "
            f"{len(overlay_set) - len(modality_keys)} copied whole from ref2va, "
            f"{len(base_keys) - len(overlay_set)} copied whole from fl2va.\n"
            f"Metadata fields: {list(meta.keys())}\n"
        )
        yield _status("Dry run complete")
        yield {"type": "done", "status": "dry-run complete"}
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_output_path = output_path + ".tmp"

    # Stream: base order, overlay keys read from ref2va (or blended)
    overlay_set_frozenset = frozenset(overlay_set)
    total_keys = len(base_manifest)
    progress_every = max(1, total_keys // 100)
    t0 = time.monotonic()

    with open(tmp_output_path, "wb") as out_f:
        header = _build_safetensors_header(base_manifest, meta)
        out_f.write(struct.pack("<Q", len(header)))
        out_f.write(header)

        with safe_open(fl2va_path, framework="pt", device="cpu") as bf, \
             safe_open(ref2va_path, framework="pt", device="cpu") as of:
            for i, key in enumerate(base_manifest):
                if blend_mode == "binary":
                    if key in modality_keys:
                        tensor = _splice_h3_adaln_modalities(
                            bf.get_tensor(key), of.get_tensor(key), modality_keys[key])
                    elif key in overlay_set_frozenset:
                        tensor = of.get_tensor(key)
                    else:
                        tensor = bf.get_tensor(key)
                elif is_larv:
                    # LARV: Layer-wise Adaptive Rescaling Veneer (arXiv:2602.09413)
                    # Compute per-layer scale from weight diagnostics, apply as:
                    # result = base + s_l * (ref - base) where s_l is the LARV scale
                    m = overlay_re.match(key)
                    if not m:
                        tensor = of.get_tensor(key) if key in overlay_set_frozenset else bf.get_tensor(key)
                    else:
                        block_idx = int(m.group(1))
                        w = weights.get(key)
                        if w is not None:
                            # Outside blend range → binary selection
                            if w >= 1.0:
                                tensor = of.get_tensor(key)
                            else:
                                tensor = bf.get_tensor(key)
                        else:
                            # Metrics were precomputed across the complete selected block set;
                            # biases share their weight tensor's one scalar LARV scale.
                            scale_l = larv_scales[block_idx]
                            fl_t = bf.get_tensor(key).to(torch.float32)
                            ref_t = of.get_tensor(key).to(torch.float32)
                            tensor = (fl_t + scale_l * (ref_t - fl_t)).to(fl_t.dtype)
                            del fl_t, ref_t
                else:
                    # Fallback to binary
                    if key in overlay_set_frozenset:
                        tensor = of.get_tensor(key)
                    else:
                        tensor = bf.get_tensor(key)

                _write_tensor_bytes(out_f, tensor)
                del tensor

                done = i + 1
                if done % progress_every == 0 or done == total_keys:
                    elapsed = time.monotonic() - t0
                    eta = elapsed * (total_keys - done) / done
                    progress_msg = (
                        f"h3_hybrid {done}/{total_keys} tensors "
                        f"(elapsed {_format_duration(elapsed)}, "
                        f"eta {_format_duration(eta)})"
                    )
                    # Self-updating single line in the console window.
                    yield _progress(progress_msg)

    os.replace(tmp_output_path, output_path)

    try:
        recipe_path = _write_model_merge_recipe(
            output_path, recipe_name, architecture, fl2va_path, ref2va_path,
            preserve_loader_metadata, do_watermark,
            ["Blend mode: binary",
             f"Block selection: {'custom' if custom_blocks is not None else 'range'}",
             f"Block range: {actual_start}–{actual_end}" if custom_blocks is None
             else "Block range: not used (custom selection)",
             f"Selected blocks: {', '.join(map(str, block_idxs))}",
             *(["Modality selection: experimental split",
                f"Video blocks: {', '.join(map(str, block_idxs))}",
                f"Text blocks: {', '.join(map(str, sorted(text_idxs))) or 'none'}",
                f"Audio blocks: {', '.join(map(str, sorted(audio_idxs))) or 'none'}",
                f"Audio output head from overlay: {'yes' if audio_out_from_overlay else 'no'}",
                f"Sliced AdaLN tensors: {len(modality_keys)}"] if modality_mode == "split" else []),
             f"Final AdaLN requested: {'yes' if final_adaln_from_overlay else 'no'}",
             f"Final AdaLN from overlay: {'yes' if final_included else 'no'}",
             f"Overlay tensors: {len(overlay_set) - len(modality_keys)}",
             f"Sliced tensors: {len(modality_keys)}",
             f"Base tensors: {len(base_keys) - len(overlay_set)}"],
        )
    except OSError as exc:
        yield _log(f"ERROR: checkpoint created but could not write merge recipe: {exc}\n")
        yield {"type": "done", "status": "failed"}
        return
    yield _log(f"Wrote model merge recipe: {recipe_path}\n")

    elapsed_total = time.monotonic() - t0
    yield _log(f"Wrote merged checkpoint: {output_path}\n")
    yield _log(
        f"Model merge complete: {len(base_keys)} tensors "
        f"({len(overlay_set) - len(modality_keys)} overlay + "
        f"{len(modality_keys)} sliced + {len(base_keys) - len(overlay_set)} base), "
        f"elapsed {_format_duration(elapsed_total)}\n"
    )
    yield _status("Model merge complete")
    yield {"type": "done", "status": "finished"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Metadata keys inherited from a base checkpoint describe that checkpoint,
# not the merged output, so they must not be carried into merge results.
_INHERITED_HASH_METADATA_PREFIX = "civitai.hash."
_INHERITED_HASH_METADATA_KEYS = {"modelspec.hash_sha256"}


def _read_base_metadata(file_path: str, preserve_loader_metadata: bool = True) -> Dict[str, str]:
    """Read __metadata__ from a safetensors file (header-only).

    Hash metadata is dropped: it describes the source checkpoint, not the
    merged output, and may be a stale/placeholder value from an upstream
    tool. Merge outputs never re-carry base hash keys.

    With ``preserve_loader_metadata`` disabled, only the loader-critical
    runtime keys (config/format/architecture) are inherited; everything else
    stays with the source checkpoint.
    """
    with open(file_path, "rb") as f:
        (n8,) = struct.unpack("<Q", f.read(8))
        f.seek(8)
        header = json.loads(f.read(n8))
    md = header.get("__metadata__", {})
    meta: Dict[str, str] = {}
    for k, v in md.items():
        k = str(k)
        if k.startswith(_INHERITED_HASH_METADATA_PREFIX) or k in _INHERITED_HASH_METADATA_KEYS:
            continue
        if not preserve_loader_metadata and k not in _LOADER_CONFIG_KEYS:
            continue
        meta[k] = str(v)
    return meta


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


def _format_duration(seconds: float) -> str:
    """Compact duration for progress events: 90s -> '1m 30s', 3661s -> '1h 1m'."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _status(status: str) -> Dict[str, str]:
    return {"type": "status", "status": status}


def _progress(text: str) -> Dict[str, str]:
    """One progress tick for the UI's self-updating console line.

    ``type: "progress"`` events carry only ``text``: the frontend renders
    them into a dedicated single line inside the console card, overwriting
    the line in place. No ``status`` field, so the topbar status line and
    the job summary status are untouched; and unlike ``type: "log"``
    events, nothing is appended to the console scrollback, so the rest of
    the console window stays intact while the line keeps updating.
    """
    return {"type": "progress", "text": text}


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
