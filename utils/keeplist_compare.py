# utils/keeplist_compare.py
"""
Compares our pattern-based keep-list against an author's reference FP8 file.

The author's FP8 file is ground truth from people who built the model: any
tensor stored at FP16/BF16 in their file is one they considered too sensitive
to quantize. Comparing against our patterns reveals:

  - DISAGREEMENT_AUTHOR_KEEPS: author preserves, we quantize
        -> review and likely add to our patterns
  - DISAGREEMENT_WE_KEEP: we preserve, author quantizes
        -> we may be over-conservative (less critical)
  - AGREEMENT_BOTH_KEEP: both preserve
        -> confirms our pattern is correct
  - AGREEMENT_BOTH_QUANT: both quantize
        -> confirms our default is correct

The comparison ignores tensors that exist in only one file (e.g. scale
tensors added during quantization that don't exist in the FP16 source).
"""
import os
import re
from collections import defaultdict, Counter

from core.layer_config_builder import (
    ALWAYS_SKIP_PATTERNS,
    KEEP_HIGHER_PRECISION_PATTERNS,
    BAKED_VAE_PATTERNS,
)


# Dtypes we consider "preserved at high precision" in a quantized file.
# Anything else (float8_*, int8) is a quantized tensor.
HIGH_PRECISION_DTYPES = {"F16", "BF16", "F32", "F64"}


def _read_dtype_map(safetensors_path):
    """Return {tensor_name: dtype_str} without loading weights."""
    from safetensors import safe_open
    out = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_slice(k)
            out[k] = str(t.get_dtype())
    return out


def _is_quantizable_weight(name, shape_or_none):
    """Filter to weight tensors that would be quantized (2D matmuls).

    We compare only on these because:
      - 1D tensors (norms, biases) are never quantized regardless of mode
      - Non-.weight tensors (scales, zeros) are quantization artifacts
    """
    # Include .weight OR large structural names often used for embeddings
    structural_keywords = ["embedding", "img_emb", "pos_emb", "modulation"]
    return name.endswith(".weight") or any(k in name for k in structural_keywords)


def _our_classification(layer_key, skip_rxs, keep_rxs):
    """Apply our patterns to a layer key. Returns 'preserve' or 'quantize'.
    
    Mirrors convert_to_quant: matches against name without .weight suffix.
    """
    match_key = layer_key[:-len(".weight")] if layer_key.endswith(".weight") else layer_key
    if any(rx.search(match_key) for rx in skip_rxs):
        return "preserve"
    if any(rx.search(match_key) for rx in keep_rxs):
        return "preserve"
    return "quantize"


def _author_classification(dtype_str):
    """Classify a tensor in the author's file as preserved or quantized."""
    # safetensors dtype strings: "F16", "BF16", "F8_E4M3", etc.
    # The high-precision check handles BF16/F16/F32 explicitly; everything
    # else (F8, I8, U8) counts as quantized.
    return "preserve" if dtype_str in HIGH_PRECISION_DTYPES else "quantize"


def _collapse_stem(key):
    """blocks.0.attn.q -> blocks.N.attn.q for grouping output."""
    s = re.sub(r"\.\d+\.", ".N.", key)
    s = re.sub(r"\.\d+$", ".N", s)
    return s


def compare_to_reference(reference_fp8_path, model_type):
    """
    Compare our pattern decisions against the author's FP8 reference.
    Returns a formatted text report.
    """
    if not os.path.exists(reference_fp8_path):
        return f"❌ Reference file not found: {reference_fp8_path}"

    if model_type not in ALWAYS_SKIP_PATTERNS:
        return (
            f"❌ No patterns defined for '{model_type}'. "
            f"Known: {list(ALWAYS_SKIP_PATTERNS)}"
        )

    skip_pattern_strs = list(ALWAYS_SKIP_PATTERNS[model_type]) + list(BAKED_VAE_PATTERNS)
    skip_rxs = [re.compile(p) for p in skip_pattern_strs]
    keep_rxs = [re.compile(p) for p in KEEP_HIGHER_PRECISION_PATTERNS[model_type]]

    try:
        ref_dtypes = _read_dtype_map(reference_fp8_path)
    except Exception as e:
        return f"🔥 Read error on reference: {e}"

    # Filter to tensors that are weights OR matched by our patterns.
    # This ensures we catch large embeddings/modulations that don't end in .weight.
    weight_tensors = {
        k: v for k, v in ref_dtypes.items() 
        if k.endswith(".weight") or any(rx.search(k) for rx in skip_rxs + keep_rxs)
    }
    if not weight_tensors:
        return f"❌ No .weight tensors found in {os.path.basename(reference_fp8_path)}"

    # Categorize every weight tensor
    categories = Counter()
    disagreement_author = []  # (key, our_classification, author_dtype)
    disagreement_we = []
    
    for name, dtype in weight_tensors.items():
        ours = _our_classification(name, skip_rxs, keep_rxs)
        theirs = _author_classification(dtype)
        
        if ours == "preserve" and theirs == "preserve":
            categories["BOTH_KEEP"] += 1
        elif ours == "quantize" and theirs == "quantize":
            categories["BOTH_QUANT"] += 1
        elif ours == "quantize" and theirs == "preserve":
            categories["AUTHOR_KEEPS"] += 1
            disagreement_author.append((name, dtype))
        else:  # ours == "preserve", theirs == "quantize"
            categories["WE_KEEP"] += 1
            disagreement_we.append((name, dtype))

    total = sum(categories.values())
    
    out = []
    out.append(f"🔬 Keep-List Comparison vs Reference")
    out.append(f"   Reference: {os.path.basename(reference_fp8_path)}")
    out.append(f"   Architecture: {model_type}")
    out.append("-" * 60)
    out.append(f"Total weight tensors compared: {total}")
    out.append("")
    out.append("AGREEMENT:")
    out.append(f"  Both preserve at FP16/BF16: {categories['BOTH_KEEP']:>5}")
    out.append(f"  Both quantize             : {categories['BOTH_QUANT']:>5}")
    pct_agree = (categories['BOTH_KEEP'] + categories['BOTH_QUANT']) / total * 100
    out.append(f"  Agreement rate            : {pct_agree:>5.1f}%")
    out.append("")
    out.append("DISAGREEMENTS:")
    out.append(f"  Author keeps, we quantize : {categories['AUTHOR_KEEPS']:>5}  "
               f"{'⚠️  REVIEW' if categories['AUTHOR_KEEPS'] > 0 else ''}")
    out.append(f"  We keep, author quantizes : {categories['WE_KEEP']:>5}  "
               f"{'(over-conservative)' if categories['WE_KEEP'] > 0 else ''}")
    out.append("")

    # Detail the high-priority disagreements: layers the author preserves
    # that we don't. Group by stem for compact display.
    if disagreement_author:
        out.append("=" * 60)
        out.append("⚠️  AUTHOR KEEPS, WE QUANTIZE (high priority)")
        out.append("    These are layers the model author chose to preserve.")
        out.append("    Add patterns in core/layer_config_builder.py to match.")
        out.append("")
        stems = defaultdict(lambda: {"count": 0, "sample": None, "dtype": None})
        for name, dtype in disagreement_author:
            stem = _collapse_stem(name)
            entry = stems[stem]
            entry["count"] += 1
            if entry["sample"] is None:
                entry["sample"] = name
                entry["dtype"] = dtype
        for stem, info in sorted(stems.items(), key=lambda kv: -kv[1]["count"]):
            out.append(f"  [{info['count']:>4}x] {stem}")
            out.append(f"           example: {info['sample']}")
            out.append(f"           author stores at: {info['dtype']}")
        out.append("")

    # Lower priority: where we're over-conservative
    if disagreement_we:
        out.append("=" * 60)
        out.append("ℹ️  WE KEEP, AUTHOR QUANTIZES (over-conservative)")
        out.append("    Lower priority: we preserve more than the author.")
        out.append("    Output is correct but slightly larger than necessary.")
        out.append("")
        stems = defaultdict(lambda: {"count": 0, "sample": None})
        for name, dtype in disagreement_we:
            stem = _collapse_stem(name)
            entry = stems[stem]
            entry["count"] += 1
            if entry["sample"] is None:
                entry["sample"] = name
        # Show only top 10 most common to avoid spam
        sorted_stems = sorted(stems.items(), key=lambda kv: -kv[1]["count"])[:10]
        for stem, info in sorted_stems:
            out.append(f"  [{info['count']:>4}x] {stem}")
            out.append(f"           example: {info['sample']}")
        if len(stems) > 10:
            out.append(f"  ... and {len(stems) - 10} more stems")
        out.append("")

    if not disagreement_author and not disagreement_we:
        out.append("✅ Perfect agreement with author's keep-list.")
    elif not disagreement_author:
        out.append("✅ No layers missed - we cover everything the author preserves.")

    # === SUGGESTED PATTERNS BLOCK ===
    # Generate regex patterns from the AUTHOR_KEEPS disagreements that the
    # user can paste directly into core/layer_config_builder.py. Patterns
    # are derived from the collapsed stems (block indices already
    # generalized to .N during _collapse_stem).
    if disagreement_author:
        out.append("")
        out.append("=" * 60)
        out.append("📋 SUGGESTED PATTERNS")
        out.append("=" * 60)
        out.append("Copy these regex patterns into core/layer_config_builder.py")
        out.append(f"under ALWAYS_SKIP_PATTERNS['{model_type}'] to match the")
        out.append("author's preservation decisions:")
        out.append("")
        # Build patterns from stems, dedup
        suggested = set()
        stems = defaultdict(lambda: {"count": 0, "sample": None})
        for name, dtype in disagreement_author:
            stem = _collapse_stem(name)
            stems[stem]["count"] += 1
            if stems[stem]["sample"] is None:
                stems[stem]["sample"] = name
        for stem in stems:
            # Strip trailing .weight, escape dots, replace .N with \.\d+
            pat_source = stem
            if pat_source.endswith(".weight"):
                pat_source = pat_source[:-len(".weight")]
            # Strip leading model.diffusion_model. prefix for portability
            if pat_source.startswith("model.diffusion_model."):
                pat_source = pat_source[len("model.diffusion_model."):]
                anchor_prefix = r"(^|\.)"
            else:
                # Already prefix-free (e.g. WAN keys) - anchor at start
                anchor_prefix = r"^"
            # Escape dots, generalize block indices
            escaped = pat_source.replace(".N", "<<BLOCKIDX>>").replace(".", r"\.").replace("<<BLOCKIDX>>", r"\.\d+")
            pattern = f"{anchor_prefix}{escaped}$"
            suggested.add(pattern)

        # Sort by length (more specific first) for stable output
        for pat in sorted(suggested):
            out.append(f'    r"{pat}",')
        out.append("")
        out.append(f"Total: {len(suggested)} unique patterns covering "
                   f"{categories['AUTHOR_KEEPS']} layers.")
        out.append("")
        out.append("⚠️  Review before pasting:")
        out.append("    - Some 'AUTHOR_KEEPS' may be out of scope for your source")
        out.append("      file (e.g. VAE/vocoder layers if source is transformer-only)")
        out.append("    - The author may preserve different blocks selectively;")
        out.append("      our generated regex matches ALL block indices (\\d+)")
        out.append("    - Test with the Audit Patterns button after adding")

    return "\n".join(out)
