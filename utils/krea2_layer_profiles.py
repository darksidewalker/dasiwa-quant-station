import re


_PRESERVE_PATTERNS = [
    # Modulation / timestep / input-output / norm tensors are structural and
    # should not be changed by LoRA merges. Observed in krea2_raw.safetensors.
    re.compile(r"(^|\.)mod(ulation)?\.lin$"),
    re.compile(r"^last\."),
    re.compile(r"^first\."),
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


# Filter-based strategy multipliers.
# Each preset selects which tensor categories receive LoRA modifications (1.0)
# and which are excluded from merging (0.0). No boosting — only selection.
_STRATEGY_MULTIPLIERS = {
    # Balanced: apply all non-structural tensors uniformly.
    "Balanced": {
        "attn_qkv": 1.0,
        "attn_out": 1.0,
        "attn_gate": 1.0,
        "ff_in": 1.0,
        "ff_out": 1.0,
        "text_fusion": 1.0,
        "structural": 0.0,
        "other": 1.0,
    },
    # Style: only attention-related tensors get merged; FFN and text pathways excluded.
    # Use this when the LoRA primarily affects visual style/aesthetics via attention routing.
    "Style": {
        "attn_qkv": 1.0,
        "attn_out": 1.0,
        "attn_gate": 1.0,
        "structural": 0.0,
    },
    # Content: only feed-forward networks get merged; attention and text pathways excluded.
    # Use this when the LoRA primarily affects subject/content via FFN pathway modifications.
    "Content": {
        "ff_in": 1.0,
        "ff_out": 1.0,
        "structural": 0.0,
    },
}


def strategy_multiplier(strategy: str, category: str) -> float:
    table = _STRATEGY_MULTIPLIERS.get(strategy) or _STRATEGY_MULTIPLIERS["Balanced"]
    return table.get(category, table.get("other", 0.0))
