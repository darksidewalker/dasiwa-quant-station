import re
from typing import Optional

from core.layer_config_builder import PRESERVE_PATTERNS, BAKED_VAE_PATTERNS


_PRESERVE_RX = [re.compile(p) for p in PRESERVE_PATTERNS["WAN 2.2"] + BAKED_VAE_PATTERNS]


def is_wan22_preserved_key(key: str) -> bool:
    if any(rx.search(key) for rx in _PRESERVE_RX):
        return True
    # 1D norm/shift/scale tensors that are structural, not learnable weight matrices
    # (e.g. modulation output norms, final head norm)
    if re.search(r"\.(weight|bias)$", key) is None:
        return False
    return False




def classify_wan22_key(key: str) -> str:
    """Classify a WAN 2.2 tensor key into a merge strategy category.

    WAN 2.2 uses split q/k/v/o projections in self_attn and cross_attn,
    and ffn.0/ffn.2 for the feed-forward network. No audio components.
    """
    if is_wan22_preserved_key(key):
        return "preserve"

    # Self-attention Q/K/V
    if re.search(r"\.self_attn\.(q|k|v)\.", key) or re.search(r"\.self_attn\.(q|k|v)$", key):
        return "self_attn_qkv"

    # Self-attention output projection
    if re.search(r"\.self_attn\.proj", key):
        return "self_attn_out"

    # Cross-attention Q/K/V
    if re.search(r"\.cross_attn\.(q|k|v)\.", key) or re.search(r"\.cross_attn\.(q|k|v)$", key):
        return "cross_attn_qkv"

    # Cross-attention output projection
    if re.search(r"\.cross_attn\.proj", key):
        return "cross_attn_out"

    # FFN up projection (ffn.0 or ffn.net.0)
    if re.search(r"\.ffn\.0", key) or re.search(r"\.ffn\.net\.0", key):
        return "ffn_in"

    # FFN down projection (ffn.2 or ffn.net.2)
    if re.search(r"\.ffn\.2", key) or re.search(r"\.ffn\.net\.2", key):
        return "ffn_out"

    # Modulation linear (per-block adaln-style modulation)
    if "modulation" in key:
        return "modulation"

    # Caption / text embedding projection
    if "caption" in key or "text_embedding" in key:
        return "caption_projection"

    # Patch embedding / output head
    if "patch_embedding" in key or "head" in key or "proj_out" in key:
        return "patchify_or_output"

    # Norm layers
    if key.endswith("_norm.weight") or ".norm" in key:
        return "norm"

    return "other"


# Filter-based strategy multipliers.
# Each preset selects which tensor categories receive LoRA modifications (1.0)
# and which are excluded from merging (0.0). No boosting — only selection.
_STRATEGY_MULTIPLIERS = {
    # Balanced: apply all non-preserved tensors uniformly.
    "Balanced": {
        "self_attn_qkv": 1.0,
        "self_attn_out": 1.0,
        "cross_attn_qkv": 1.0,
        "cross_attn_out": 1.0,
        "ffn_in": 1.0,
        "ffn_out": 1.0,
        "modulation": 1.0,
        "caption_projection": 1.0,
        "patchify_or_output": 1.0,
        "norm": 1.0,
        "other": 1.0,
    },
    # Motion-only: only attention modules (self + cross QKV/out) get merged;
    # FFN and structural tensors excluded — focuses on motion dynamics in text-to-video.
    "Motion": {
        "self_attn_qkv": 1.0,
        "self_attn_out": 1.0,
        "cross_attn_qkv": 1.0,
        "cross_attn_out": 1.0,
        # Non-attention components excluded from merge.
    },
    # Visuals-only: only FFN, caption_projection and output-side tensors get merged;
    # attention modules excluded — focuses on visual quality over motion dynamics.
    "Visuals": {
        "ffn_in": 1.0,
        "ffn_out": 1.0,
        "caption_projection": 1.0,
        "patchify_or_output": 1.0,
        # Attention modules excluded from merge.
    },
}


def strategy_multiplier(strategy: str, category: str) -> float:
    table = _STRATEGY_MULTIPLIERS.get(strategy) or _STRATEGY_MULTIPLIERS["Balanced"]
    return table.get(category, table.get("other", 0.0))
