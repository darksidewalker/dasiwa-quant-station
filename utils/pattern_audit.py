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
    BAKED_VAE_PATTERNS,
)
from utils.arch_detector import verify_architecture_match

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
    r"\.(audio_)?ff\.net\.2$",        # diffusers MLP
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


def _classify_layer(key, skip_pats, keep_pats, suspicious_rx, shape):
    """Return (category, matching_pattern_or_None) for one layer.
    
    Pattern lists are passed as (pattern_string, compiled_regex) tuples
    so we can report which specific rule fired.
    
    Note: convert_to_quant matches regex against the layer name WITHOUT
    the .weight suffix (verified against template format). We strip it
    here so the audit reflects what actually happens at quantization time.
    """
    match_key = key[:-len(".weight")] if key.endswith(".weight") else key
    for pat_str, rx in skip_pats:
        if rx.search(match_key):
            return "SKIP", pat_str
    for pat_str, rx in keep_pats:
        if rx.search(match_key):
            return "KEEP_HIGH", pat_str
    is_2d_weight = (len(shape) == 2 and key.endswith(".weight"))
    if is_2d_weight and suspicious_rx.search(match_key):
        return "SUSPICIOUS", None
    return "DEFAULT", None


def _collapse_stem(key):
    """blocks.0.attn.q -> blocks.N.attn.q (for grouping in output)."""
    s = re.sub(r"\.\d+\.", ".N.", key)
    s = re.sub(r"\.\d+$", ".N", s)
    return s


def audit_patterns(safetensors_path, model_type):
    """
    Audit a safetensors file against the layer-config patterns.
    Returns a formatted text report ready to display.
    
    Report tells you:
      - Top-line verdict: are patterns sufficient for this model?
      - Per-pattern coverage: which rules fired and how many layers each caught
      - Which layers fall to base format (default behavior)
      - Suspicious unmatched layers that look structurally important
    """
    if not os.path.exists(safetensors_path):
        return f"❌ Error: File not found at {safetensors_path}"

    if model_type not in ALWAYS_SKIP_PATTERNS:
        return (
            f"❌ Error: No patterns defined for architecture '{model_type}'. "
            f"Known: {list(ALWAYS_SKIP_PATTERNS)}"
        )

    # Verify the file actually matches the declared architecture.
    # Without this, an LTX file audited under WAN patterns produces a
    # confusing "every layer is suspicious" report, when the real issue
    # is that the user has the wrong architecture radio selected.
    arch_ok, arch_msg = verify_architecture_match(safetensors_path, model_type)
    if not arch_ok:
        return (
            f"🔍 Pattern Audit: {os.path.basename(safetensors_path)}\n"
            f"   Architecture: {model_type}\n"
            + "=" * 60 + "\n"
            + arch_msg
        )

    # Build (pattern_string, compiled_regex) tuples so we can report
    # which specific rule matched each layer.
    # Include BAKED_VAE_PATTERNS so VAE/vocoder layers (if baked in) show
    # as SKIP rather than SUSPICIOUS.
    skip_pattern_strs = list(ALWAYS_SKIP_PATTERNS[model_type]) + list(BAKED_VAE_PATTERNS)
    skip_pats = [(p, re.compile(p)) for p in skip_pattern_strs]
    keep_pats = [(p, re.compile(p)) for p in KEEP_HIGHER_PRECISION_PATTERNS[model_type]]
    suspicious_rx = re.compile("|".join(SUSPICIOUS_PATTERNS))

    try:
        layers = _read_layers(safetensors_path)
    except Exception as e:
        return f"🔥 Read error: {e}"

    if not layers:
        return f"❌ Empty layer list in {os.path.basename(safetensors_path)}"

    # Track everything per-pattern + per-category
    categories = Counter()
    skip_pattern_hits = Counter()    # pattern_string -> count
    keep_pattern_hits = Counter()
    suspicious_stems = defaultdict(lambda: {"count": 0, "sample": None, "shape": None})

    for key, shape, dtype, n_params in layers:
        cat, matched_pattern = _classify_layer(
            key, skip_pats, keep_pats, suspicious_rx, shape
        )
        categories[cat] += 1
        if cat == "SKIP":
            skip_pattern_hits[matched_pattern] += 1
        elif cat == "KEEP_HIGH":
            keep_pattern_hits[matched_pattern] += 1
        elif cat == "SUSPICIOUS":
            stem = _collapse_stem(key)
            entry = suspicious_stems[stem]
            entry["count"] += 1
            if entry["sample"] is None:
                entry["sample"] = key
                entry["shape"] = shape

    total = len(layers)
    out = []

    # === HEADER ===
    out.append(f"🔍 Pattern Audit: {os.path.basename(safetensors_path)}")
    out.append(f"   Architecture: {model_type}")
    out.append("=" * 60)

    # === TOP-LINE VERDICT ===
    n_suspicious = categories["SUSPICIOUS"]
    if n_suspicious == 0:
        out.append("✅ VERDICT: Patterns fully cover this model.")
        out.append("   Every structural and sensitive layer is matched by an")
        out.append("   active rule. No changes needed for this architecture.")
    else:
        n_families = len(suspicious_stems)
        out.append(f"⚠️  VERDICT: {n_families} unmatched layer "
                   f"{'family' if n_families == 1 else 'families'} found "
                   f"({n_suspicious} layers).")
        out.append("   These look structurally important but no pattern matches.")
        out.append("   Review the SUSPICIOUS section below.")
    out.append("")

    # === SUMMARY ===
    out.append(f"Total layers in file: {total}")
    out.append("")
    out.append(f"  SKIP (matched ALWAYS_SKIP)         : {categories['SKIP']:>5}")
    out.append(f"  KEEP_HIGH (matched sensitive list) : {categories['KEEP_HIGH']:>5}")
    out.append(f"  DEFAULT (intentionally at base fmt): {categories['DEFAULT']:>5}")
    if n_suspicious:
        out.append(f"  SUSPICIOUS (review needed)         : {n_suspicious:>5}")
    out.append("")

    # === ALWAYS_SKIP COVERAGE ===
    out.append("─" * 60)
    out.append("ALWAYS_SKIP patterns (structural layers preserved at FP16):")
    out.append("─" * 60)
    for pat_str, _ in skip_pats:
        hits = skip_pattern_hits.get(pat_str, 0)
        marker = "✓" if hits > 0 else "·"
        out.append(f"  {marker} [{hits:>4}x] {pat_str}")
    unused_skip = sum(1 for p, _ in skip_pats if skip_pattern_hits.get(p, 0) == 0)
    if unused_skip:
        out.append(f"  Note: {unused_skip} pattern(s) matched 0 layers in this model.")
    out.append("")

    # === KEEP_HIGHER COVERAGE ===
    out.append("─" * 60)
    out.append("KEEP_HIGHER_PRECISION patterns (sensitive layers):")
    out.append("  On FP8 base: these stay at FP16/BF16 (skip mode)")
    out.append("  On NVFP4/INT8 base: bumped to FP8")
    out.append("─" * 60)
    for pat_str, _ in keep_pats:
        hits = keep_pattern_hits.get(pat_str, 0)
        marker = "✓" if hits > 0 else "·"
        out.append(f"  {marker} [{hits:>4}x] {pat_str}")
    unused_keep = sum(1 for p, _ in keep_pats if keep_pattern_hits.get(p, 0) == 0)
    if unused_keep:
        out.append(f"  Note: {unused_keep} pattern(s) matched 0 layers in this model.")
    out.append("")

    # === SUSPICIOUS DETAILS (only if any) ===
    if suspicious_stems:
        out.append("─" * 60)
        out.append("⚠️  SUSPICIOUS layers (look structural but unmatched):")
        out.append("─" * 60)
        out.append("These will get the base format. If that's wrong, add a")
        out.append("pattern in core/layer_config_builder.py to cover them.")
        out.append("")
        for stem, info in sorted(suspicious_stems.items(), key=lambda kv: -kv[1]["count"]):
            out.append(f"  [{info['count']:>3}x] {stem}")
            out.append(f"          example: {info['sample']}")
            out.append(f"          shape:   {info['shape']}")
        out.append("")
        out.append("How to add a pattern:")
        out.append("  1. Open core/layer_config_builder.py")
        out.append("  2. Find ALWAYS_SKIP_PATTERNS or KEEP_HIGHER_PRECISION_PATTERNS")
        out.append(f"  3. Find the '{model_type}' entry")
        out.append("  4. Add a regex that matches the layer stems above")
        out.append("     (without the trailing .weight suffix)")

    return "\n".join(out)
