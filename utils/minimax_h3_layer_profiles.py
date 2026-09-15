import re

from core.layer_config_builder import BAKED_VAE_PATTERNS, PRESERVE_PATTERNS


_PRESERVE_RX = [re.compile(pattern) for pattern in PRESERVE_PATTERNS["MiniMax H3"] + BAKED_VAE_PATTERNS]


def is_minimax_h3_preserved_key(key: str) -> bool:
    return any(pattern.search(key) for pattern in _PRESERVE_RX) or bool(
        re.search(r"(?:^|\.)(?:adaln|modulation|norm|rope)(?:\.|$)", key)
    )


def classify_minimax_h3_key(key: str) -> str:
    if is_minimax_h3_preserved_key(key):
        return "preserve"
    if key.startswith("token_refiner.") or ".token_refiner." in key:
        return "token_refiner"
    if re.search(r"\.attn\.qkv_proj\.weight$", key):
        return "attn_qkv"
    if re.search(r"\.attn\.out_proj\.weight$", key):
        return "attn_out"
    if re.search(r"\.mlp\.fc1\.weight$", key):
        return "ff_in"
    if re.search(r"\.mlp\.fc2\.weight$", key):
        return "ff_out"
    return "other"


_STRATEGIES = {
    "Balanced": {"attn_qkv": 1.0, "attn_out": 1.0, "ff_in": 1.0, "ff_out": 1.0, "token_refiner": 1.0, "other": 1.0},
    "Motion": {"attn_qkv": 1.0, "attn_out": 1.0},
    "Visuals": {"attn_qkv": 1.0, "attn_out": 1.0, "ff_in": 1.0, "ff_out": 1.0},
}


def strategy_multiplier(strategy: str, category: str) -> float:
    table = _STRATEGIES.get(strategy) or _STRATEGIES["Balanced"]
    return table.get(category, 0.0 if strategy in {"Motion", "Visuals"} else 1.0)
