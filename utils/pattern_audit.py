# utils/pattern_audit.py
"""
Pattern audit: tells you what the layer-config builder would do with a
specific model BEFORE you quantize it. Detects layers that look structurally
important but aren't covered by any pattern (e.g. when a new model variant
renames a component the heuristic was calibrated against).

Categorization per layer:
  - PRESERVE  : matched by PRESERVE_PATTERNS for this architecture
  - RESCUE    : matched by RESCUE_PATTERNS
  - DEFAULT   : neither matched (will get the base format)
  - SUSPICIOUS: DEFAULT but the layer name contains transformer-component
                keywords AND is a 2D weight tensor. These are the layers
                most likely to be miscategorized for a new model variant.
  - MIXED-KEPT: a heavy linear kept at source precision by a RECOGNIZED
                mixed quantization profile. Intentional retention, not a
                pattern miss — never contributes to the verdict.

Quant profile detection (MiniMax H3):
  Community H3 NVFP4 quants ship distinct, comment-proofed profiles.
  Auditing a quantized H3 file classifies which profile it uses:
    - nvfp4_pure             : all 200 main-matrix heavy linears U8-packed
    - nvfp4_hq_mixed        : 170 packed + 30 kept BF16 per the known HQ
                              plan (27 attn.out_proj + 3 mlp.fc2, DmitryDB
                              NVFP4-HQ) -> recognized variant
    - nvfp4_fp8_adaln_mixed : packed heavy linears + FP8 adaln_proj tier
                              (Abiray mixed) -> recognized variant
    - nvfp4_mixed_unknown   : kept layers outside any known plan -> soft
                              review note, not a pattern error
  Recognized mixed profiles never produce SUSPICIOUS flags for their
  retained layers ("no false flagging").

Note: this audit reports our config's intent. The actual --wan / --ltxv2
preset inside convert_to_quant skips additional layers independently;
those won't show as SKIP here but will still be skipped at runtime.
"""
import os
import re
from collections import Counter, defaultdict

from core.layer_config_builder import (
    BAKED_VAE_PATTERNS,
    PRESERVE_PATTERNS,
    RESCUE_PATTERNS,
    H3_NVFP4_HQ_OUTPROJ_BLOCKS,
    H3_NVFP4_HQ_FC2_BLOCKS,
    H3_NVFP4_HQ_LAYER_PLAN,
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
    # Value projection variants (sensitive - rescue targets these)
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

# ---------------------------------------------------------------------------
# MiniMax H3: intentional base-format heavy linears.
#
# H3's policy packs exactly these four per-block linears (plus the
# token_refiner equivalents) to the base format; they are DEFAULT by
# design, not pattern misses. The generic SUSPICIOUS keyword `\.fc2$`
# (LLaMA-style) would flag H3's fc2 layers as "review needed" on every
# H3 file — a false positive. When auditing MiniMax H3, these stems are
# excluded from the suspicious check.
_H3_HEAVY_STEMS = re.compile(
    r"^.*blocks\.\d+\.attn\.(qkv_proj|out_proj)$"
    r"|^.*blocks\.\d+\.mlp\.(fc1|fc2)$"
    r"|^.*token_refiner\.blocks\.\d+\.attn\.(qkv_proj|out_proj)$"
    r"|^.*token_refiner\.blocks\.\d+\.mlp\.(fc1|fc2)$"
)

# --- H3 NVFP4 quant-profile detection ------------------------------------
# Recognized community profiles (comment-proofed, see
# references/minimax-h3-nvfp4-community-reference-audit.md in the
# dasiwa-quant-station skill):
#   nvfp4_pure             all 200 main-matrix heavy linears U8-packed
#   nvfp4_hq_mixed         170 packed + 30 kept per the HQ plan
#                          (27 attn.out_proj + 3 mlp.fc2, DmitryDB)
#   nvfp4_fp8_adaln_mixed  all 200 packed + FP8 adaln_proj tier (Abiray)
#   nvfp4_mixed_unknown    kept layers outside any known plan (soft note)

_H3_MAIN_MATRIX_RX = re.compile(
    r"^blocks\.(\d+)\.(attn\.(qkv_proj|out_proj)|mlp\.(fc1|fc2))\.weight$"
)
_H3_ADALN_WEIGHT_RX = re.compile(r"^blocks\.\d+\.adaln_proj\.linear\.weight$")


def detect_h3_quant_profile(layers):
    """Classify the NVFP4 quant profile of a MiniMax H3 file from dtypes.

    layers: list of (key, shape, dtype, n_params) from _read_layers().
    Returns None for unquantized H3 source files (no U8-packed main
    matrix), otherwise a dict:
      profile:   one of the four profile ids above
      packed:    count of U8-packed main-matrix heavy linears
      kept:      [(block, kind, dtype, key)] non-packed main-matrix layers
      fp8_adaln: True when per-block adaln projections are F8_E4M3
    """
    packed = []
    kept = []
    fp8_adaln = False
    for key, _shape, dtype, _n_params in layers:
        m = _H3_MAIN_MATRIX_RX.match(key)
        if m:
            block, kind = int(m.group(1)), m.group(2)
            if dtype == "U8":
                packed.append((block, kind))
            else:
                kept.append((block, kind, dtype, key))
        elif _H3_ADALN_WEIGHT_RX.match(key) and dtype == "F8_E4M3":
            fp8_adaln = True
    if not packed:
        return None  # unquantized source file: nothing to detect
    plan = {(b, k) for b, k in H3_NVFP4_HQ_LAYER_PLAN}
    kept_set = {(b, k) for b, k, _d, _key in kept}
    if not kept and not fp8_adaln:
        profile = "nvfp4_pure"
    elif not fp8_adaln and kept_set == plan:
        profile = "nvfp4_hq_mixed"
    elif not kept and fp8_adaln:
        profile = "nvfp4_fp8_adaln_mixed"
    else:
        profile = "nvfp4_mixed_unknown"
    return {
        "profile": profile,
        "packed": len(packed),
        "kept": kept,
        "fp8_adaln": fp8_adaln,
    }


def _block_range_text(blocks):
    """[0, 1, 2, 17, 19, 27] -> '0-2,17,19,27'"""
    blocks = sorted(set(blocks))
    if not blocks:
        return ""
    parts = []
    start = prev = blocks[0]
    for b in blocks[1:]:
        if b == prev + 1:
            prev = b
        else:
            parts.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = b
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


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


def _classify_layer(key, skip_pats, keep_pats, suspicious_rx, shape,
                   suspicious_exclude_rx=None):
    """Return (category, matching_pattern_or_None) for one layer.
    
    Pattern lists are passed as (pattern_string, compiled_regex) tuples
    so we can report which specific rule fired.
    
    Note: convert_to_quant matches regex against the layer name WITHOUT
    the .weight suffix (verified against template format). We strip it
    here so the audit reflects what actually happens at quantization time.
    
    suspicious_exclude_rx (architecture-specific): stems that are
    INTENTIONALLY at the base format for this architecture (e.g. MiniMax
    H3's heavy linears, which the policy packs by design). These are
    reported as DEFAULT but never as SUSPICIOUS, so known policies and
    recognized mixed quant profiles do not trigger review flags.
    """
    match_key = key[:-len(".weight")] if key.endswith(".weight") else key
    for pat_str, rx in skip_pats:
        if rx.search(match_key):
            return "PRESERVE", pat_str
    for pat_str, rx in keep_pats:
        if rx.search(match_key):
            return "RESCUE", pat_str

    # Structural check: 2D weights or any large tensor (like embeddings)
    # that look structural but aren't matched by skip/keep rules.
    n_params = 1
    for d in shape: n_params *= d
    is_weight = key.endswith(".weight")
    is_large = n_params > 1000000  # Tensors > 1M params are usually structural
    if suspicious_exclude_rx is not None and suspicious_exclude_rx.match(match_key):
        return "DEFAULT", None
    if (is_weight or is_large) and suspicious_rx.search(match_key):
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

    if model_type not in PRESERVE_PATTERNS:
        return (
            f"❌ Error: No patterns defined for architecture '{model_type}'. "
            f"Known: {list(PRESERVE_PATTERNS)}"
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
    skip_pattern_strs = list(PRESERVE_PATTERNS[model_type]) + list(BAKED_VAE_PATTERNS)
    skip_pats = [(p, re.compile(p)) for p in skip_pattern_strs]
    keep_pats = [(p, re.compile(p)) for p in RESCUE_PATTERNS[model_type]]
    suspicious_rx = re.compile("|".join(SUSPICIOUS_PATTERNS))

    # MiniMax H3: the four per-block heavy linears (attn.qkv_proj,
    # attn.out_proj, mlp.fc1, mlp.fc2, plus token_refiner equivalents) are
    # base-format (or skip) by design. The generic SUSPICIOUS keyword
    # `\.fc2$` would flag H3's fc2 layers as "review needed" on every H3
    # file. Exclude them from the suspicious check; recognized mixed
    # profiles that intentionally keep some of these at source precision
    # are reported as MIXED-KEPT below, never as suspicious pattern misses.
    suspicious_exclude_rx = _H3_HEAVY_STEMS if model_type == "MiniMax H3" else None

    try:
        layers = _read_layers(safetensors_path)
    except Exception as e:
        return f"🔥 Read error: {e}"

    if not layers:
        return f"❌ Empty layer list in {os.path.basename(safetensors_path)}"

    # Detect which H3 NVFP4 quant profile this file uses (pure / HQ-mixed /
    # FP8-adaln-mixed / unknown-mixed). Unquantized H3 source files return
    # None and skip profile reporting.
    h3_profile = None
    if model_type == "MiniMax H3":
        h3_profile = detect_h3_quant_profile(layers)

    # Track everything per-pattern + per-category
    categories = Counter()
    skip_pattern_hits = Counter()    # pattern_string -> count
    keep_pattern_hits = Counter()
    suspicious_stems = defaultdict(lambda: {"count": 0, "sample": None, "shape": None})

    for key, shape, dtype, n_params in layers:
        cat, matched_pattern = _classify_layer(
            key, skip_pats, keep_pats, suspicious_rx, shape,
            suspicious_exclude_rx=suspicious_exclude_rx,
        )
        categories[cat] += 1
        if cat == "PRESERVE":
            skip_pattern_hits[matched_pattern] += 1
        elif cat == "RESCUE":
            keep_pattern_hits[matched_pattern] += 1
        elif cat == "SUSPICIOUS":
            stem = _collapse_stem(key)
            entry = suspicious_stems[stem]
            entry["count"] += 1
            if entry["sample"] is None:
                entry["sample"] = key
                entry["shape"] = shape

    # Mixed-profile recognition: heavy H3 linears kept at source precision
    # by a RECOGNIZED quant profile are intentional retention, not pattern
    # misses. Group them by (kind, dtype) with block ranges for the report;
    # they never count against the SUSPICIOUS verdict.
    mixed_kept_families = Counter()
    mixed_kept_samples = {}
    if h3_profile is not None and h3_profile["profile"] in (
        "nvfp4_hq_mixed", "nvfp4_fp8_adaln_mixed", "nvfp4_mixed_unknown"
    ):
        by_kind = {}
        for block, kind, dtype, key in h3_profile["kept"]:
            entry = by_kind.setdefault((kind, dtype), {"blocks": [], "sample": key})
            entry["blocks"].append(block)
        for (kind, dtype), info in sorted(by_kind.items()):
            family = f"{kind} (blocks {_block_range_text(info['blocks'])}, {dtype})"
            mixed_kept_families[family] = len(info["blocks"])
            mixed_kept_samples[family] = info["sample"]
        # FP8 adaln tier is a whole-family retention: summarize once.
        if h3_profile["fp8_adaln"]:
            n_adaln = sum(
                1 for k, _s, d, _n in layers if d == "F8_E4M3"
                and re.match(_H3_ADALN_WEIGHT_RX, k)
            )
            family = "adaln_proj.linear (F8_E4M3 tier)"
            mixed_kept_families[family] = n_adaln
            mixed_kept_samples[family] = "blocks.0.adaln_proj.linear.weight"

    total = len(layers)
    out = []

    # === HEADER ===
    out.append(f"🔍 Pattern Audit: {os.path.basename(safetensors_path)}")
    out.append(f"   Architecture: {model_type}")

    # === QUANT PROFILE (MiniMax H3 quantized files) ===
    if h3_profile is not None:
        profile = h3_profile["profile"]
        profile_names = {
            "nvfp4_pure": "NVFP4 pure (all heavy linears packed)",
            "nvfp4_hq_mixed": "NVFP4 HQ mixed (per-block BF16 retention)",
            "nvfp4_fp8_adaln_mixed": "NVFP4 + FP8 adaln tier",
            "nvfp4_mixed_unknown": "NVFP4 mixed (unrecognized plan)",
        }
        out.append(f"   Quant profile : {profile_names[profile]}")
        out.append(f"   Packed U8     : {h3_profile['packed']} main-matrix heavy linears")
        if h3_profile["kept"]:
            out.append(
                f"   Kept BF16/F16 : {len(h3_profile['kept'])} layers "
                "(intentional, see MIXED-KEPT section)"
            )
        if profile == "nvfp4_hq_mixed":
            out.append(
                "   Recognized variant: DmitryDB NVFP4-HQ plan "
                f"({len(H3_NVFP4_HQ_LAYER_PLAN)} layers: "
                f"{len(H3_NVFP4_HQ_OUTPROJ_BLOCKS)}x out_proj + "
                f"{len(H3_NVFP4_HQ_FC2_BLOCKS)}x fc2)"
            )
        elif profile == "nvfp4_fp8_adaln_mixed":
            out.append("   Recognized variant: FP8 rescue tier on adaln_proj")
        elif profile == "nvfp4_mixed_unknown":
            out.append("   Kept layers do not match any known community plan;")
            out.append("   review the MIXED-KEPT section before re-quantizing.")
    out.append("=" * 60)

    # === TOP-LINE VERDICT ===
    n_suspicious = categories["SUSPICIOUS"]
    if n_suspicious == 0:
        out.append("✅ VERDICT: Patterns fully cover this model.")
        out.append("   Every structural and sensitive layer is matched by an")
        out.append("   active rule. No changes needed for this architecture.")
        if h3_profile is not None and h3_profile["profile"] != "nvfp4_pure":
            out.append("   Mixed-profile retention layers are recognized variants,")
            out.append("   not pattern misses.")
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
    out.append(f"  PRESERVE (source precision)       : {categories['PRESERVE']:>5}")
    out.append(f"  RESCUE (FP8 on lower-bit bases)   : {categories['RESCUE']:>5}")
    out.append(f"  DEFAULT (intentionally at base fmt): {categories['DEFAULT']:>5}")
    if h3_profile is not None and h3_profile["profile"] != "nvfp4_pure":
        out.append(f"  MIXED-KEPT (recognized profile)   : {sum(mixed_kept_families.values()):>5}")
    if n_suspicious:
        out.append(f"  SUSPICIOUS (review needed)         : {n_suspicious:>5}")
    out.append("")

    # === PRESERVE COVERAGE ===
    out.append("─" * 60)
    out.append("PRESERVE patterns (source precision on every base):")
    out.append("─" * 60)
    for pat_str, _ in skip_pats:
        hits = skip_pattern_hits.get(pat_str, 0)
        marker = "✓" if hits > 0 else "·"
        out.append(f"  {marker} [{hits:>4}x] {pat_str}")
    unused_skip = sum(1 for p, _ in skip_pats if skip_pattern_hits.get(p, 0) == 0)
    if unused_skip:
        out.append(f"  Note: {unused_skip} pattern(s) matched 0 layers in this model.")
    out.append("")

    # === RESCUE COVERAGE ===
    out.append("─" * 60)
    out.append("RESCUE patterns (sensitive under lower-bit bases):")
    out.append("  On FP8 base: use the FP8 base format")
    out.append("  On NVFP4/INT8 base: rescue to FP8")
    out.append("─" * 60)
    for pat_str, _ in keep_pats:
        hits = keep_pattern_hits.get(pat_str, 0)
        marker = "✓" if hits > 0 else "·"
        out.append(f"  {marker} [{hits:>4}x] {pat_str}")
    unused_keep = sum(1 for p, _ in keep_pats if keep_pattern_hits.get(p, 0) == 0)
    if unused_keep:
        out.append(f"  Note: {unused_keep} pattern(s) matched 0 layers in this model.")
    out.append("")

    # === MIXED-KEPT DETAIL (recognized profiles only) ===
    if mixed_kept_families:
        out.append("─" * 60)
        out.append("MIXED-KEPT layers (recognized quant profile, intentional):")
        out.append("─" * 60)
        out.append("These heavy linears are kept at source precision by a known")
        out.append("mixed quant profile, NOT a pattern miss. No action needed.")
        out.append("")
        for family in sorted(mixed_kept_families):
            out.append(f"  [{mixed_kept_families[family]:>3}x] {family}")
            out.append(f"          example: {mixed_kept_samples[family]}")
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
        out.append("  2. Find PRESERVE_PATTERNS or RESCUE_PATTERNS")
        out.append(f"  3. Find the '{model_type}' entry")
        out.append("  4. Add a regex that matches the layer stems above")
        out.append("     (without the trailing .weight suffix)")

    return "\n".join(out)
