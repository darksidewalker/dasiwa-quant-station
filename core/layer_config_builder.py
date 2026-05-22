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

  ALWAYS_SKIP_PATTERNS:
    Layers we never want quantized regardless of base format. These mirror
    what --wan / --ltxv2 already skip via the arch preset. Listed here as
    defense-in-depth: even if the preset's interaction with --layer-config
    changes in a future convert_to_quant version, these layers stay safe.

  KEEP_HIGHER_PRECISION_PATTERNS:
    Sensitive layers (to_v, ffn_down, connectors). Behavior depends on base:
      - Base = float8_e4m3fn (FP8): mark as {"skip": true} -> stay at FP16
      - Base = nvfp4 / int8_*  : mark as {"format":"float8_e4m3fn"} -> bump to FP8
"""
import os
import json
import tempfile


# Map UI format choice -> _default format string in the layer config
BASE_FORMAT_FOR_CONFIG = {
    "FP8": "float8_e4m3fn",
    "INT8 Block-wise": "int8_blockwise",
    "NVFP4": "nvfp4",
}


# Patterns for baked-in VAE, audio VAE, vocoder, and text-embedding projection
# layers. These are present when a model ships with its decoder/vocoder baked
# into the checkpoint (full inference pipeline) rather than transformer-only.
# They are decode-quality-critical: even the author's FP8 releases keep them
# at BF16 (verified via Compare to Reference against author's sulphur_dev FP8).
#
# Applied to all architectures unconditionally. Patterns are anchored with ^
# because these prefixes live at the top level of the layer name (no
# model.diffusion_model. prefix).
BAKED_VAE_PATTERNS = [
    r"^vae\.",
    r"^audio_vae\.",
    r"^vocoder\.",
    r"^text_embedding_projection\.",
    r"^encoder\.",
    r"^decoder\.",
]


# Layers that should never be quantized for a given architecture.
# Mirrors --wan / --ltxv2 preset skips. Source: TEST FP8 Simple log
# ("ltxv2 skip" lines) for LTX-2.3; lcpp.patch quantize-skip rules for both.
ALWAYS_SKIP_PATTERNS = {
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
        # Author preserves first 2 and last 2 blocks for these sensitive projections.
        # Using (0|1|4[67]) specifically targets the author's preservation logic.
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.attn1\.to_k$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.attn1\.to_out\.\d+$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.attn1\.to_q$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.attn1\.to_v$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.attn2\.to_k$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.attn2\.to_out\.\d+$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.attn2\.to_q$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.attn2\.to_v$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.audio_attn1\.to_k$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.audio_attn1\.to_out\.\d+$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.audio_attn1\.to_q$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.audio_attn1\.to_v$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.audio_attn2\.to_k$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.audio_attn2\.to_out\.\d+$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.audio_attn2\.to_q$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.audio_attn2\.to_v$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.(audio_)?ff\.net\.\d+$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.(audio_)?ff\.net\.\d+\.proj$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.audio_to_video_attn\.to_k$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.audio_to_video_attn\.to_out\.\d+$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.audio_to_video_attn\.to_q$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.audio_to_video_attn\.to_v$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.video_to_audio_attn\.to_k$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.video_to_audio_attn\.to_out\.\d+$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.video_to_audio_attn\.to_q$",
        r"(^|\.)transformer_blocks\.(0|1|4[67])\.video_to_audio_attn\.to_v$",
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


# Sensitive layers: kept at higher precision than the base format.
# Same heuristic as before (City96 keys_hiprec + lcpp.patch bump rules).
# Patterns verified against real WAN and LTX-2.3 templates.
KEEP_HIGHER_PRECISION_PATTERNS = {
    "LTX-2.3": [
        # to_v across every attention variant (excluding connector blocks
        # which are already covered by ALWAYS_SKIP). Negative lookahead
        # prevents double-matching the same layer with two different rules.
        r"^(?!.*_embeddings_connector).*\.transformer_blocks\.\d+\..*\.to_v$",
        # FFN down projection (ff.net.2 and audio_ff.net.2), excluding
        # connector layers for the same reason.
        r"^(?!.*_embeddings_connector).*\.transformer_blocks\.\d+\.(audio_)?ff\.net\.2$",
    ],
    "WAN 2.2": [
        # WAN uses split q/k/v/o (never fused). to_v in self + cross attn.
        # WAN's ALWAYS_SKIP doesn't overlap with these patterns.
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
        base_format_ui_label: UI format label, e.g. "FP8", "NVFP4", "INT8 Block-wise"

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

    skip_patterns = ALWAYS_SKIP_PATTERNS.get(model_type)
    keep_patterns = KEEP_HIGHER_PRECISION_PATTERNS.get(model_type)
    if skip_patterns is None or keep_patterns is None:
        raise ValueError(
            f"No patterns defined for architecture '{model_type}'. "
            f"Valid: {list(ALWAYS_SKIP_PATTERNS)}"
        )

    # Merge baked-VAE patterns into the skip list. These apply regardless
    # of architecture - any VAE/vocoder/text_projection layers present in
    # the source must stay at source precision. Harmless if not present
    # in the source (zero matches, zero cost).
    skip_patterns = list(skip_patterns) + list(BAKED_VAE_PATTERNS)

    config = {
        "_default": {"format": base_fmt},
        "_exclusions": [],
    }

    # Sensitive-layer behavior depends on base format
    if base_fmt == "float8_e4m3fn":
        # FP8 base: nothing higher to bump to within the format enum.
        # Keep sensitive layers at source precision (FP16/BF16) via skip.
        for pat in keep_patterns:
            config[pat] = {"skip": True}
        keep_action = "skip (stay at source FP16/BF16)"
    else:
        # Lower-bit base (nvfp4, int8_*): bump sensitive layers to FP8.
        for pat in keep_patterns:
            config[pat] = {"format": "float8_e4m3fn"}
        keep_action = "bump to float8_e4m3fn"

    # Always-skip patterns get {"skip": true}. Added LAST so they override
    # keep_higher patterns in lower-bit modes (BF16 skip > FP8 bump).
    for pat in skip_patterns:
        config[pat] = {"skip": True}

    summary = {
        "base_format": base_fmt,
        "always_skip_count": len(skip_patterns),
        "keep_higher_count": len(keep_patterns),
        "keep_action": keep_action,
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
        f"skip patterns={summary['always_skip_count']} | "
        f"sensitive layer action: {summary['keep_action']}"
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
