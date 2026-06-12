# utils/exact_config.py
"""
Builds a layer config that exactly mirrors a reference file's preservation
decisions, bypassing the regex-based heuristic.

Use case: you have an author-provided FP8 reference for a base model, and
you're quantizing a fine-tune of that exact base. The author already
decided which layers to preserve based on testing the model; you can
inherit those decisions directly per-tensor instead of approximating them
with regex.

Trade-off vs pattern-based config:
  - More precise (no regex approximation)
  - Larger config file (one entry per quantizable tensor)
  - Tied to the specific layer naming of the reference: if your source has
    additional layers (or different names), they get the base format
  - Won't generalize to other architectures or model variants

Schema produced matches the regex-based builder: _default at the chosen
base format, plus per-layer overrides for everything the reference
preserves at FP16/BF16.
"""
import os
import json

# Dtypes treated as "preserved at high precision" in the reference file.
HIGH_PRECISION_DTYPES = {"F16", "BF16", "F32", "F64"}

# Map UI format label -> _default format string (must match the values
# convert_to_quant's layer config loader accepts).
BASE_FORMAT_FOR_CONFIG = {
    "FP8": "float8_e4m3fn",
    "INT8 Row-wise ConvRot": "int8_tensorwise",
    "NVFP4": "nvfp4",
}


def _read_dtype_map(safetensors_path):
    """{tensor_name: dtype_str} from header only, no weight loading."""
    from safetensors import safe_open
    out = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_slice(k)
            out[k] = str(t.get_dtype())
    return out


def build_exact_config(reference_fp8_path, base_format_ui_label,
                       source_keys=None):
    """
    Build a layer config that mirrors the reference file's per-tensor
    preservation decisions.

    Args:
        reference_fp8_path: path to the author's reference FP8 safetensors
        base_format_ui_label: "FP8", "NVFP4", or "INT8 Row-wise ConvRot"
        source_keys: optional set of layer names from the source file.
            When provided, the config only includes entries for layers
            present in the source. Without this, the config tries to set
            entries for layers that may not exist in the source, which
            convert_to_quant may complain about or silently ignore.

    Returns:
        (config_dict, summary) on success
        Raises ValueError on bad input.
    """
    base_fmt = BASE_FORMAT_FOR_CONFIG.get(base_format_ui_label)
    if base_fmt is None:
        raise ValueError(
            f"Unknown base format '{base_format_ui_label}'. "
            f"Valid: {list(BASE_FORMAT_FOR_CONFIG)}"
        )

    if not os.path.exists(reference_fp8_path):
        raise ValueError(f"Reference file not found: {reference_fp8_path}")

    ref_dtypes = _read_dtype_map(reference_fp8_path)

    config = {
        "_default": {"format": base_fmt},
        "_exclusions": [],
    }

    # Filter to weight tensors only (consistent with how the regex
    # builder behaves). Scales, biases, etc. are 1D and auto-handled.
    # NOTE: We include large non-suffix tensors like embeddings/modulations
    # which are structural and often several gigabytes.
    from safetensors import safe_open

    weight_tensors = {}
    with safe_open(reference_fp8_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            shape = f.get_slice(k).get_shape()
            n_params = 1
            for d in shape: n_params *= d
            # Capture .weight OR any tensor with > 1M params (embeddings)
            if k.endswith(".weight") or n_params > 1000000:
                weight_tensors[k] = ref_dtypes[k]

    preserved_in_ref = 0
    preserved_in_scope = 0
    out_of_scope = 0

    for name, dtype in weight_tensors.items():
        if dtype not in HIGH_PRECISION_DTYPES:
            continue  # author quantized this; default base_fmt covers it

        preserved_in_ref += 1

        # If source_keys provided, skip layers that don't exist in source
        if source_keys is not None and name not in source_keys:
            out_of_scope += 1
            continue

        # Layer config keys exclude the .weight suffix per convert_to_quant
        # convention (verified earlier in this project)
        key_no_weight = name[:-len(".weight")]
        # On FP8 base, "preserve" = skip:true (stay at source FP16/BF16)
        # On lower-bit bases, "preserve" = bump to FP8
        if base_fmt == "float8_e4m3fn":
            config[key_no_weight] = {"skip": True}
        else:
            config[key_no_weight] = {
                "format": "float8_e4m3fn",
                "scaling_mode": "tensor",
            }

    summary = {
        "base_format": base_fmt,
        "preserved_in_reference": preserved_in_ref,
        "preserved_applied": preserved_in_ref - out_of_scope,
        "out_of_scope": out_of_scope,
        "total_entries": len(config) - 2,  # minus _default + _exclusions
    }
    return config, summary


def write_exact_config(reference_fp8_path, base_format_ui_label,
                       out_dir, source_keys=None):
    """
    Build the config and write it to disk under out_dir.
    
    Returns (config_path, log_lines).
    """
    log = []
    try:
        config, summary = build_exact_config(
            reference_fp8_path, base_format_ui_label, source_keys
        )
    except ValueError as e:
        log.append(f"[exact-config] {e}")
        return None, log
    except Exception as e:
        log.append(f"[exact-config] Failed reading reference: {e}")
        return None, log

    os.makedirs(out_dir, exist_ok=True)
    ref_base = os.path.splitext(os.path.basename(reference_fp8_path))[0]
    fmt_slug = base_format_ui_label.replace(" ", "_").lower()
    config_path = os.path.join(out_dir, f"_exact_{ref_base}_{fmt_slug}.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    log.append(
        f"[exact-config] Base: {summary['base_format']} | "
        f"Reference preserves: {summary['preserved_in_reference']} layers | "
        f"Applied: {summary['preserved_applied']} | "
        f"Out of scope: {summary['out_of_scope']}"
    )
    if summary["out_of_scope"] > 0:
        log.append(
            f"[exact-config] {summary['out_of_scope']} layers exist in "
            f"reference but not in source - skipped."
        )
    log.append(f"[exact-config] Written: {os.path.basename(config_path)}")
    return config_path, log


def export_preview(reference_fp8_path, base_format_ui_label, source_keys=None):
    """
    Build the config but return a summary report instead of writing to disk.
    Used by the UI's preview button so the user can see what would be
    generated before committing to a run.
    """
    try:
        config, summary = build_exact_config(
            reference_fp8_path, base_format_ui_label, source_keys
        )
    except ValueError as e:
        return f"❌ {e}"
    except Exception as e:
        return f"🔥 Read error: {e}"

    out = []
    out.append(f"🎯 Exact Config Preview")
    out.append(f"   Reference: {os.path.basename(reference_fp8_path)}")
    out.append(f"   Base format: {summary['base_format']}")
    out.append("-" * 60)
    out.append(f"Reference preserves: {summary['preserved_in_reference']} layers at high precision")
    if source_keys is not None:
        out.append(f"In source scope    : {summary['preserved_applied']}")
        out.append(f"Out of scope       : {summary['out_of_scope']} (skipped)")
    out.append(f"Total config entries: {summary['total_entries']}")
    out.append("")
    if summary["out_of_scope"] > 0:
        out.append(f"ℹ️  {summary['out_of_scope']} reference layers don't exist in your source.")
        out.append("   This is expected when the reference is a full checkpoint")
        out.append("   (with VAE, vocoder, etc.) and your source is transformer-only.")
        out.append("")
    out.append("Each high-precision tensor in the reference becomes a config")
    out.append("entry that either skips (FP8 base) or bumps to FP8 (lower bases).")
    out.append("")
    out.append("To use this config, click 'Use Reference Config' in the UI")
    out.append("before starting quantization. The exact config takes precedence")
    out.append("over the regex-based auto layer config.")
    return "\n".join(out)
