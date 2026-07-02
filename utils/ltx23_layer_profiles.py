import re
from typing import Optional

from core.layer_config_builder import PRESERVE_PATTERNS, BAKED_VAE_PATTERNS


_PRESERVE_RX = [re.compile(p) for p in PRESERVE_PATTERNS["LTX-2.3"] + BAKED_VAE_PATTERNS]


def is_ltx23_preserved_key(key: str) -> bool:
    if any(rx.search(key) for rx in _PRESERVE_RX):
        return True
    # Quantization preserve table sees module names such as `.to_gate_logits`,
    # while LoRA merge targets the concrete `.to_gate_logits.weight` tensor.
    return bool(re.search(r"\.to_gate_logits(\.weight|\.bias)?$", key))




def classify_ltx23_key(key: str) -> str:
    if is_ltx23_preserved_key(key):
        return "preserve"
    if re.search(r"\.audio_to_video_attn\.", key):
        return "audio_to_video_attn"
    if re.search(r"\.video_to_audio_attn\.", key):
        return "video_to_audio_attn"
    if re.search(r"\.audio_attn\d+\..*to_[qkv]", key):
        return "audio_attn"
    if re.search(r"\.audio_attn\d+\..*to_out\.0", key):
        return "audio_attn_out"
    if re.search(r"\.attn\d+\..*to_[qkv]", key):
        return "attn_qkv"
    if re.search(r"\.attn\d+\..*to_out\.0", key):
        return "attn_out"
    if re.search(r"\.audio_ff\.net\.0", key):
        return "audio_ff_in"
    if re.search(r"\.audio_ff\.net\.2", key):
        return "audio_ff_out"
    if re.search(r"\.ff\.net\.0", key):
        return "ff_in"
    if re.search(r"\.ff\.net\.2", key):
        return "ff_out"
    if "caption_projection" in key:
        return "caption_projection"
    if "patchify_proj" in key or "proj_out" in key:
        return "patchify_or_output"
    if key.endswith("_norm.weight") or ".norm" in key:
        return "norm"
    return "other"


# Filter-based strategy multipliers.
# Each preset selects which tensor categories receive LoRA modifications (1.0)
# and which are excluded from merging (0.0). No boosting — only selection.
_STRATEGY_MULTIPLIERS = {
    # Balanced: apply all non-preserved tensors uniformly.
    "Balanced": {
        "attn_qkv": 1.0,
        "attn_out": 1.0,
        "ff_in": 1.0,
        "ff_out": 1.0,
        "audio_attn": 1.0,
        "audio_attn_out": 1.0,
        "audio_to_video_attn": 1.0,
        "video_to_audio_attn": 1.0,
        "audio_ff_in": 1.0,
        "audio_ff_out": 1.0,
        "caption_projection": 1.0,
        "patchify_or_output": 1.0,
        "norm": 1.0,
        "other": 1.0,
    },
    # Audio-only: only audio-specific tensors get merged; everything else is excluded.
    "Audio": {
        "audio_attn": 1.0,
        "audio_attn_out": 1.0,
        "audio_ff_in": 1.0,
        "audio_ff_out": 1.0,
        # Cross-modal bridges stay neutral (both directions) — LoRA won't modify them.
        "audio_to_video_attn": 0.0,
        "video_to_audio_attn": 0.0,
    },
    # Video-only: all non-audio tensors get merged; audio-specific ones excluded.
    "Video": {
        "attn_qkv": 1.0,
        "attn_out": 1.0,
        "ff_in": 1.0,
        "ff_out": 1.0,
        # Cross-modal bridges — video side gets full LoRA application.
        "audio_to_video_attn": 1.0,
        "video_to_audio_attn": 1.0,
        "caption_projection": 1.0,
        "patchify_or_output": 1.0,
        "norm": 1.0,
        # Audio tensors explicitly excluded — prevent fallback to 'other: 1.0'.
        "audio_attn": 0.0,
        "audio_attn_out": 0.0,
        "audio_ff_in": 0.0,
        "audio_ff_out": 0.0,
        "other": 1.0,
    },
}


def strategy_multiplier(strategy: str, category: str) -> float:
    table = _STRATEGY_MULTIPLIERS.get(strategy) or _STRATEGY_MULTIPLIERS["Balanced"]
    return table.get(category, table.get("other", 0.0))
