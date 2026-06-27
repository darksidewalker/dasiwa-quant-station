import re


_PRESERVE_PATTERNS = [
    # Modulation / timestep / input-output / norm tensors are structural and
    # should not be changed by LoRA merges. Observed in krea2_raw.safetensors.
    re.compile(r"(^|\.)mod(ulation)?\.lin$"),
    re.compile(r"(^|\.)tproj\."),
    re.compile(r"(^|\.)tmlp\."),
    re.compile(r"(^|\.)txtmlp\."),
    re.compile(r"(^|\.)first\."),
    re.compile(r"(^|\.)last\."),
    re.compile(r"(^|\.)txtfusion\.projector\."),
    re.compile(r"(^|\.)(pre|post)?norm\.scale$"),
    re.compile(r"\.qknorm\.[qk]norm\.scale$"),
]


def is_krea2_preserved_key(key: str) -> bool:
    return any(rx.search(key) for rx in _PRESERVE_PATTERNS)


def classify_krea2_key(key: str) -> str:
    if is_krea2_preserved_key(key):
        return "structural"
    if ".attn.wq.weight" in key or ".attn.wk.weight" in key or ".attn.wv.weight" in key:
        return "attn_qkv"
    if ".attn.wo.weight" in key:
        return "attn_out"
    if ".attn.gate.weight" in key:
        return "attn_gate"
    if ".mlp.gate.weight" in key or ".mlp.up.weight" in key:
        return "ff_in"
    if ".mlp.down.weight" in key:
        return "ff_out"
    if key.startswith("txtfusion."):
        return "text_fusion"
    return "other"


_STRATEGY_MULTIPLIERS = {
    "Balanced": {
        "attn_qkv": 1.00,
        "attn_out": 1.00,
        "attn_gate": 0.90,
        "ff_in": 1.00,
        "ff_out": 1.00,
        "text_fusion": 0.80,
        "structural": 0.0,
        "other": 0.80,
    },
    "Visuals": {
        "attn_qkv": 1.10,
        "attn_out": 1.10,
        "attn_gate": 1.00,
        "ff_in": 1.05,
        "ff_out": 1.05,
        "text_fusion": 0.85,
        "structural": 0.0,
        "other": 0.85,
    },
    "Motion": {
        # Krea 2 is an image model; keep Motion as a compatibility alias rather
        # than exposing special video behavior.
        "attn_qkv": 1.00,
        "attn_out": 1.00,
        "attn_gate": 0.90,
        "ff_in": 1.00,
        "ff_out": 1.00,
        "text_fusion": 0.80,
        "structural": 0.0,
        "other": 0.80,
    },
    "Audio": {
        # Krea 2 has no audio layers; treated like Balanced if stale UI state
        # submits it. The frontend filters Audio out for Krea 2.
        "attn_qkv": 1.00,
        "attn_out": 1.00,
        "attn_gate": 0.90,
        "ff_in": 1.00,
        "ff_out": 1.00,
        "text_fusion": 0.80,
        "structural": 0.0,
        "other": 0.80,
    },
}


def strategy_multiplier(strategy: str, category: str) -> float:
    table = _STRATEGY_MULTIPLIERS.get(strategy) or _STRATEGY_MULTIPLIERS["Balanced"]
    return table.get(category, table["other"])
