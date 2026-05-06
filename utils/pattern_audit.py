# utils/pattern_audit.py
"""
Pattern audit: tells you what the layer-config builder would do with a
specific model BEFORE you quantize it. Detects layers that look structurally
important but aren't covered by any pattern (e.g. when a new model variant
renames a component the heuristic was calibrated against).

Categorization per layer:
  - SKIP      : matched by ALWAYS_SKIP_PATTERNS for this architecture
  - KEEP_HIGH : matched by KEEP_HIGHER_PRECISION_PATTERNS
  - DEFAULT   : neither matched (will get the base format)
  - SUSPICIOUS: DEFAULT but the layer name contains transformer-component
                keywords AND is a 2D weight tensor. These are the layers
                most likely to be miscategorized for a new model variant.

Note: this audit reports our config's intent. The actual --wan / --ltxv2
preset inside convert_to_quant skips additional layers independently;
those won't show as SKIP here but will still be skipped at runtime.
"""
import os
import re
from collections import Counter, defaultdict

from core.layer_config_builder import (
    ALWAYS_SKIP_PATTERNS,
    KEEP_HIGHER_PRECISION_PATTERNS,
)

# Component name fragments that suggest a layer is a SENSITIVE one we
# should be matching but might have missed (e.g. due to a rename in a
# new model variant). The audit only flags layers matching these AND
# not matched by any existing pattern.
#
# Patterns are anchored at component boundaries (\.X\b or X$ etc.) to
# avoid substring false-positives like "to_v" matching inside the module
# name "audio_to_video_attn".
SUSPICIOUS_PATTERNS = [
    # Value projection variants (sensitive - keep_higher targets these)
    r"\.to_v$",              # diffusers attention naming
    r"\.v_proj$",            # HuggingFace transformers naming
    r"\.v$",                 # WAN-style naming
    # FFN down projection variants (sensitive)
    r"\.down_proj$",         # HuggingFace MLP naming
    r"\.ffn\.2$",            # WAN-style FFN
    r"\.ff\.net\.2$",        # diffusers MLP
    r"\.w2$",                # LLaMA-style MLP
    r"\.fc2$",               # generic
    r"\.layer2$",            # generic alt
    # Connector / bridge variants
    r"connector",            # any connector substring (broad on purpose)
    r"bridge",               # any bridge substring
    # Modulation tables
    r"modulation",
    r"scale_shift",
    r"adaln",
]


def _read_layers(safetensors_path):
    """Return list of (key, shape, dtype, n_params) without loading weights."""
    from safetensors import safe_open
    out = []
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_slice(k)
            shape = tuple(t.get_shape())
            n_params = 1
            for d in shape:
                n_params *= d
            out.append((k, shape, t.get_dtype(), n_params))
    return out


def _classify_layer(key, skip_rxs, keep_rxs, suspicious_rx, shape):
    """Return the category for one layer.
    
    Note: convert_to_quant matches regex against the layer name WITHOUT
    the .weight suffix (verified against template format). We strip it
    here so the audit reflects what actually happens at quantization time.
    """
    match_key = key[:-len(".weight")] if key.endswith(".weight") else key
    if any(rx.search(match_key) for rx in skip_rxs):
        return "SKIP"
    if any(rx.search(match_key) for rx in keep_rxs):
        return "KEEP_HIGH"
    # Default category - check if suspicious
    is_2d_weight = (len(shape) == 2 and key.endswith(".weight"))
    if is_2d_weight and suspicious_rx.search(match_key):
        return "SUSPICIOUS"
    return "DEFAULT"


def _collapse_stem(key):
    """blocks.0.attn.q -> blocks.N.attn.q (for grouping in output)."""
    s = re.sub(r"\.\d+\.", ".N.", key)
    s = re.sub(r"\.\d+$", ".N", s)
    return s


def audit_patterns(safetensors_path, model_type):
    """
    Audit a safetensors file against the layer-config patterns.
    Returns a formatted text report ready to display.
    """
    if not os.path.exists(safetensors_path):
        return f"❌ Error: File not found at {safetensors_path}"

    if model_type not in ALWAYS_SKIP_PATTERNS:
        return (
            f"❌ Error: No patterns defined for architecture '{model_type}'. "
            f"Known: {list(ALWAYS_SKIP_PATTERNS)}"
        )

    skip_rxs = [re.compile(p) for p in ALWAYS_SKIP_PATTERNS[model_type]]
    keep_rxs = [re.compile(p) for p in KEEP_HIGHER_PRECISION_PATTERNS[model_type]]
    suspicious_rx = re.compile("|".join(SUSPICIOUS_PATTERNS))

    try:
        layers = _read_layers(safetensors_path)
    except Exception as e:
        return f"🔥 Read error: {e}"

    if not layers:
        return f"❌ Empty layer list in {os.path.basename(safetensors_path)}"

    # Categorize every layer
    categories = Counter()
    suspicious_examples = []
    keep_examples = []
    skip_examples = []

    # Group suspicious layers by stem so output is compact
    suspicious_stems = defaultdict(lambda: {"count": 0, "sample": None, "shape": None})

    for key, shape, dtype, n_params in layers:
        cat = _classify_layer(key, skip_rxs, keep_rxs, suspicious_rx, shape)
        categories[cat] += 1
        if cat == "SUSPICIOUS":
            stem = _collapse_stem(key)
            entry = suspicious_stems[stem]
            entry["count"] += 1
            if entry["sample"] is None:
                entry["sample"] = key
                entry["shape"] = shape
        elif cat == "KEEP_HIGH" and len(keep_examples) < 4:
            keep_examples.append(key)
        elif cat == "SKIP" and len(skip_examples) < 4:
            skip_examples.append(key)

    # Build report
    out = []
    out.append(f"🔍 Pattern Audit: {os.path.basename(safetensors_path)}")
    out.append(f"   Architecture: {model_type}")
    out.append("-" * 60)
    out.append(f"Total layers in file: {len(layers)}")
    out.append("")
    out.append(f"  SKIP (structural)   : {categories['SKIP']:>5}")
    out.append(f"  KEEP_HIGH (sensitive): {categories['KEEP_HIGH']:>5}")
    out.append(f"  DEFAULT (base format): {categories['DEFAULT']:>5}")
    out.append(f"  SUSPICIOUS          : {categories['SUSPICIOUS']:>5}  "
               f"{'⚠️  REVIEW NEEDED' if categories['SUSPICIOUS'] > 0 else ''}")
    out.append("")

    if skip_examples:
        out.append("✅ Sample SKIP layers (preserved at FP16/BF16):")
        for k in skip_examples:
            out.append(f"   {k}")
        out.append("")

    if keep_examples:
        out.append("✅ Sample KEEP_HIGH layers (sensitive, get bumped/skipped):")
        for k in keep_examples:
            out.append(f"   {k}")
        out.append("")

    if suspicious_stems:
        out.append("⚠️  SUSPICIOUS layers (look structural but unmatched):")
        out.append("   These will get the base format. If that's wrong, add a")
        out.append("   pattern in core/layer_config_builder.py to cover them.")
        out.append("")
        # Sort by count descending
        sorted_stems = sorted(
            suspicious_stems.items(),
            key=lambda kv: -kv[1]["count"]
        )
        for stem, info in sorted_stems:
            out.append(f"   [{info['count']:>3}x] {stem}")
            out.append(f"          example: {info['sample']}")
            out.append(f"          shape:   {info['shape']}")
        out.append("")
        out.append("   How to add a pattern: open core/layer_config_builder.py,")
        out.append("   find the relevant arch in KEEP_HIGHER_PRECISION_PATTERNS")
        out.append("   or ALWAYS_SKIP_PATTERNS, and add a regex matching the")
        out.append("   layer name (without the .weight suffix).")

    if categories["SUSPICIOUS"] == 0:
        out.append("✅ No suspicious unmatched layers. Patterns cover this model.")

    return "\n".join(out)
