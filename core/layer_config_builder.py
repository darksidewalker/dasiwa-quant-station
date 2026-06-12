# core/layer_config_builder.py
"""
Builds a layer config for convert_to_quant in-memory each run.

Verified facts (May 2026, against convert_to_quant/config/layer_config.py):
  - Layer config keys are interpreted as REGEX patterns ("regex mode")
  - Every entry must have a real format value; empty "" is rejected
  - _default also requires a real format
  - Valid formats: float8_e4m3fn, float8_e4m3fn_blockwise,
    float8_e4m3fn_rowwise, float8_e4m3fn_block3d, int8_blockwise,
    int8_tensorwise, mxfp8, nvfp4, hybrid_mxfp8
  - {"skip": true} on an entry excludes the layer from quantization
    (kept at source precision, e.g. FP16/BF16)

Why in-memory and not on-disk:
  - Patterns are arch-specific, not model-specific (regex covers every layer
    in a family with one entry), so per-model files would be redundant
  - Config gets written to a temp file just before subprocess launch
  - The patterns in this file are the single source of truth

Design:
  Two pattern groups per architecture:

  PRESERVE_PATTERNS:
    Structural / routing / IO layers we never want quantized regardless of
    base format. These stay at source precision via {"skip": true}.

  RESCUE_PATTERNS:
    Layers that are sensitive under lower-bit formats but do not need to be
    BF16/FP16 when the base format is already FP8.
      - Base = float8_e4m3fn (FP8): no override, use the FP8 base
      - Base = nvfp4 / int8_*  : mark as {"format":"float8_e4m3fn"} -> bump to FP8
"""
import os
import json
import tempfile


# Map UI format choice -> _default format string in the layer config
BASE_FORMAT_FOR_CONFIG = {
    "FP8": "float8_e4m3fn",
    "INT8 Tensor-wise": "int8_tensorwise",
    "INT8 Row-wise ConvRot Runtime": "int8_tensorwise",
    # Backward-compatible alias for configs/runs created before ConvRot was
    # removed from the default INT8 path.
    "INT8 Row-wise ConvRot": "int8_tensorwise",
    "NVFP4": "nvfp4",
}


# Patterns for baked-in companion modules: VAE, audio VAE, vocoder, text
# encoders, audio encoders, and text-embedding projection layers. These are
# present when a model ships a full inference pipeline rather than a
# transformer-only checkpoint.
#
# They are decode/conditioning-quality-critical and should not be quantized
# by transformer quantization presets. Dedicated text-encoder quantization
# remains available through the separate text architecture presets.
#
# Applied to all architectures unconditionally. Patterns are anchored with ^
# because these prefixes live at the top level of the layer name (no
# model.diffusion_model. prefix).
BAKED_VAE_PATTERNS = [
    r"^vae\.",
    r"^audio_vae\.",
    r"^vocoder\.",
    r"^text_encoder($|\.)",
    r"^text_encoder_\d+\.",
    r"^text_encoders\.",
    r"^t5\.",
    r"^clip($|\.)",
    r"^clip_text($|\.)",
    r"^clip_vision($|\.)",
    r"^gemma($|\.)",
    r"^llm($|\.)",
    r"^language_model\.",
    r"^audio_encoder\.",
    r"^audio_text_encoder\.",
    r"^text_embedding_projection\.",
    r"^encoder\.",
    r"^decoder\.",
]


# Layers that should never be quantized for a given architecture.
# These are structural/routing/IO tensors where quantization damage tends to
# cause hard failures or obvious output corruption rather than graceful quality
# loss. They are skipped for FP8, NVFP4, INT8, and GGUF sensitivity maps.
PRESERVE_PATTERNS = {
    "LTX-2.3": [
        # All adaln_single variants (timestep modulation tables).
        # Catches: adaln_single, audio_adaln_single, audio_prompt_adaln_single,
        # prompt_adaln_single, and all av_ca_*_adaln_single (gate/scale_shift
        # variants for the audio-video cross-attention modules).
        r"adaln_single\.",
        # All connector blocks (audio + video)
        r"(audio|video)_embeddings_connector\.",
        # Caption / patchify / proj_out / scale_shift_table per lcpp.patch
        r"(^|\.)caption_projection\.",
        r"(^|\.)patchify_proj($|\.)",
        r"(^|\.)proj_out($|\.)",
        # Audio-specific patchify and proj_out (verified via author's
        # reference FP8 file - they preserve these at BF16).
        r"(^|\.)audio_patchify_proj($|\.)",
        r"(^|\.)audio_proj_out($|\.)",
        r"scale_shift_table",
        # Gate logits for gated attention (apply_gated_attention=true in
        # LTX23 config). Small tables that determine attention routing -
        # corrupting these changes which tokens get attended to.
        r"\.to_gate_logits$",
        # RMS norm scales on Q and K projections (qk_norm: rms_norm in
        # LTX23 config). 1D tensors so already auto-skipped by --ltxv2,
        # but the pattern makes audit/comparison reports accurate.
        r"\.[qk]_norm$",
    ],
    "WAN 2.2": [
    # Patterns use (^|\.) to match both naked keys and keys still carrying
    # the model.diffusion_model. prefix, depending on what convert_to_quant
    # passes to the layer config matcher.
    r"(^|\.)modulation($|\.)",
    r"(^|\.)patch_embedding\.",
    r"(^|\.)text_embedding\.",
    r"(^|\.)time_projection\.",
    r"(^|\.)time_embedding\.",
    r"(^|\.)img_emb\.",
    r"(^|\.)head\.",
    ],
}


# Sensitive layers: use FP8 when the base format is lower than FP8.
# On FP8 base these are intentionally left at the base format, not kept
# BF16/FP16; official LTX-2 FP8 cast policy quantizes these transformer
# linears rather than preserving them wholesale.
RESCUE_PATTERNS = {
    "LTX-2.3": [
        # Official LTX-2 FP8 cast targets these transformer linears. For
        # lower-bit bases, rescue them to FP8 instead of BF16.
        r"^(?!.*_embeddings_connector).*\.transformer_blocks\.\d+\..*\.to_[qkv]$",
        r"^(?!.*_embeddings_connector).*\.transformer_blocks\.\d+\..*\.to_out\.\d+$",
        r"^(?!.*_embeddings_connector).*\.transformer_blocks\.\d+\.(audio_)?ff\.net\.0(\.proj)?$",
        r"^(?!.*_embeddings_connector).*\.transformer_blocks\.\d+\.(audio_)?ff\.net\.2$",
    ],
    "WAN 2.2": [
        # WAN uses split q/k/v/o (never fused). to_v in self + cross attn.
        # WAN's preserve table doesn't overlap with these patterns.
        r"\.self_attn\.v$",
        r"\.cross_attn\.v$",
        # FFN second linear (down projection)
        r"\.ffn\.2$",
    ],
}

def build_layer_config_dict(model_type, base_format_ui_label):
    """
    Build the layer config as a Python dict.

    Args:
        model_type: UI architecture label, e.g. "WAN 2.2" or "LTX-2.3"
        base_format_ui_label: UI format label, e.g. "FP8", "NVFP4", "INT8 Tensor-wise"

    Returns:
        (config_dict, summary_dict) where summary contains pattern counts
        for logging.

    Raises:
        ValueError if architecture or base format is unknown.
    """
    base_fmt = BASE_FORMAT_FOR_CONFIG.get(base_format_ui_label)
    if base_fmt is None:
        raise ValueError(
            f"Unknown base format '{base_format_ui_label}'. "
            f"Valid: {list(BASE_FORMAT_FOR_CONFIG)}"
        )

    preserve_patterns = PRESERVE_PATTERNS.get(model_type)
    rescue_patterns = RESCUE_PATTERNS.get(model_type)
    if preserve_patterns is None or rescue_patterns is None:
        raise ValueError(
            f"No patterns defined for architecture '{model_type}'. "
            f"Valid: {list(PRESERVE_PATTERNS)}"
        )

    # Merge baked-VAE patterns into the skip list. These apply regardless
    # of architecture - any VAE/vocoder/text_projection layers present in
    # the source must stay at source precision. Harmless if not present
    # in the source (zero matches, zero cost).
    preserve_patterns = list(preserve_patterns) + list(BAKED_VAE_PATTERNS)

    config = {
        "_default": {"format": base_fmt},
        "_exclusions": [],
    }

    # Rescue-layer behavior depends on base format.
    if base_fmt == "float8_e4m3fn":
        rescue_action = "base FP8 (no extra override)"
    else:
        # Lower-bit base (nvfp4, int8_*): rescue sensitive layers to FP8.
        for pat in rescue_patterns:
            config[pat] = {"format": "float8_e4m3fn", "scaling_mode": "tensor"}
        rescue_action = "rescue to float8_e4m3fn"

    # Preserve patterns get {"skip": true}. Added LAST so they override
    # rescue patterns in lower-bit modes (BF16 skip > FP8 rescue).
    for pat in preserve_patterns:
        config[pat] = {"skip": True}

    summary = {
        "base_format": base_fmt,
        "preserve_count": len(preserve_patterns),
        "rescue_count": len(rescue_patterns),
        "rescue_action": rescue_action,
        # Compatibility keys for existing log formatting.
        "always_skip_count": len(preserve_patterns),
        "keep_higher_count": len(rescue_patterns),
        "keep_action": rescue_action,
    }
    return config, summary


def write_layer_config(model_type, base_format_ui_label, out_dir=None):
    """
    Build the layer config and write it to a temp file.

    Args:
        model_type: UI architecture label
        base_format_ui_label: UI format label
        out_dir: directory for the temp file (defaults to system tmp).
                 Pass FILTERS_DIR to keep generated configs visible to
                 the user for debugging.

    Returns:
        (config_path, log_lines) on success.
        (None, log_lines) on failure.
    """
    log = []
    try:
        config, summary = build_layer_config_dict(model_type, base_format_ui_label)
    except ValueError as e:
        log.append(f"[layer-config] {e}")
        return None, log

    # Use a temp file but in a predictable location for debugging.
    # Filename includes arch + base format so concurrent runs don't collide.
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        arch_slug = model_type.replace(" ", "").replace(".", "").replace("-", "").lower()
        fmt_slug = base_format_ui_label.replace(" ", "_").lower()
        config_path = os.path.join(out_dir, f"_runtime_{arch_slug}_{fmt_slug}.json")
    else:
        fd, config_path = tempfile.mkstemp(suffix=".json", prefix="dasiwa_lc_")
        os.close(fd)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    log.append(
        f"[layer-config] {model_type} | base={summary['base_format']} | "
        f"preserve patterns={summary['preserve_count']} | "
        f"rescue patterns={summary['rescue_count']} | "
        f"rescue action: {summary['rescue_action']}"
    )
    log.append(f"[layer-config] Written: {os.path.basename(config_path)}")
    return config_path, log


# Backwards-compat shim for callers that still use the old name.
# Old signature: build_layer_config(source_path, model_type, filters_dir)
# Source path is no longer needed (regex covers every model of an arch).
def build_layer_config(source_path, model_type, filters_dir):
    """
    Deprecated entry point. Defaults to NVFP4 base for the legacy mixed mode.
    Prefer write_layer_config() with an explicit base format.
    """
    return write_layer_config(model_type, "NVFP4", out_dir=filters_dir)
