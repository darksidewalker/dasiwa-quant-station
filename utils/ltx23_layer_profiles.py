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


def extract_block_index(key: str) -> Optional[int]:
    m = re.search(r"transformer_blocks\.(\d+)\.", key)
    return int(m.group(1)) if m else None


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


_STRATEGY_MULTIPLIERS = {
    "Balanced": {
        "attn_qkv": 1.0,
        "attn_out": 1.0,
        "ff_in": 0.9,
        "ff_out": 0.9,
        "audio_attn": 0.8,
        "audio_attn_out": 0.8,
        "audio_to_video_attn": 1.0,
        "video_to_audio_attn": 1.0,
        "audio_ff_in": 0.7,
        "audio_ff_out": 0.7,
        "other": 0.5,
    },
    "Motion": {
        "attn_qkv": 0.85,
        "attn_out": 0.85,
        "ff_in": 0.65,
        "ff_out": 0.65,
        "audio_attn": 1.1,
        "audio_attn_out": 1.1,
        "audio_to_video_attn": 1.25,
        "video_to_audio_attn": 1.25,
        "audio_ff_in": 0.9,
        "audio_ff_out": 0.9,
        "other": 0.35,
    },
    "Visuals": {
        "attn_qkv": 1.05,
        "attn_out": 1.05,
        "ff_in": 1.15,
        "ff_out": 1.15,
        "audio_attn": 0.45,
        "audio_attn_out": 0.45,
        "audio_to_video_attn": 0.55,
        "video_to_audio_attn": 0.55,
        "audio_ff_in": 0.35,
        "audio_ff_out": 0.35,
        "other": 0.45,
    },
    "Audio": {
        "audio_attn": 1.2,
        "audio_attn_out": 1.2,
        "audio_to_video_attn": 0.9,
        "video_to_audio_attn": 0.9,
        "audio_ff_in": 1.1,
        "audio_ff_out": 1.1,
        "other": 0.0,
    },
}


def strategy_multiplier(strategy: str, category: str) -> float:
    table = _STRATEGY_MULTIPLIERS.get(strategy) or _STRATEGY_MULTIPLIERS["Balanced"]
    return table.get(category, table.get("other", 0.5))
