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
      - Base = nvfp4: for architectures with rescue patterns, bump to FP8
      - Base = int8_*: no rescue (all transformer weights stay INT8)
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
    # Mixed-profile NVFP4 (MiniMax H3 only): same NVFP4 base format, but the
    # layer config additionally keeps a per-block subset of heavy linears at
    # source precision (see H3_NVFP4_HQ_PRESERVE_PATTERNS).
    "NVFP4 HQ": "nvfp4",
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
        r"^.*(^|\.)[^.]*adaln_single\..*",
        # All connector blocks (audio + video)
        r"^.*(^|\.)(audio|video)_embeddings_connector\..*",
        # Caption / patchify / proj_out / scale_shift_table per lcpp.patch
        r"^.*(^|\.)caption_projection\..*",
        r"^.*(^|\.)patchify_proj($|\..*)",
        r"^.*(^|\.)proj_out($|\..*)",
        # Audio-specific patchify and proj_out (verified via author's
        # reference FP8 file - they preserve these at BF16).
        r"^.*(^|\.)audio_patchify_proj($|\..*)",
        r"^.*(^|\.)audio_proj_out($|\..*)",
        r"^.*scale_shift_table.*",
        # Gate logits for gated attention (apply_gated_attention=true in
        # LTX23 config). Small tables that determine attention routing -
        # corrupting these changes which tokens get attended to.
        r"^.*\.to_gate_logits($|\..*)",
        # RMS norm scales on Q and K projections (qk_norm: rms_norm in
        # LTX23 config). 1D tensors so already auto-skipped by --ltxv2,
        # but the pattern makes audit/comparison reports accurate.
        r"^.*\.[qk]_norm($|\..*)",
    ],
    "WAN 2.2": [
    # Patterns use (^|\.) to match both naked keys and keys still carrying
    # the model.diffusion_model. prefix, depending on what convert_to_quant
    # passes to the layer config matcher.
    r"^.*(^|\.)modulation($|\..*)",
    r"^.*(^|\.)patch_embedding($|\..*)",
    r"^.*(^|\.)text_embedding($|\..*)",
    r"^.*(^|\.)time_projection($|\..*)",
    r"^.*(^|\.)time_embedding($|\..*)",
    r"^.*(^|\.)img_emb($|\..*)",
    r"^.*(^|\.)head($|\..*)",
    ],
    "Krea 2": [
        # Empirically verified against bf16/fp8/int8/nvfp4 ComfyUI exports.
        # These patterns are intentionally fullmatch-safe: convert_to_quant's
        # layer-config matcher may require the regex to cover the whole key.
        # Keep suffix-tolerant forms so both raw keys (first.weight) and audit
        # stems (first) match.
        # Match both bare Krea 2 keys and keys that still carry a wrapper
        # prefix (model.diffusion_model.*). Some conversion paths pass raw
        # safetensors keys through unchanged; missing the prefixed form packs
        # structural tensors and produces broken NVFP4 outputs.
        r"^(.*\.)?first($|\..*)",                    # 2 layers: first.bias, first.weight
        r"^(.*\.)?last\.linear($|\..*)",             # 2 layers: last.linear.bias/weight
        r"^(.*\.)?tproj($|\..*)",                    # 2 layers: tproj.1.bias/weight
        r"^(.*\.)?tmlp($|\..*)",                     # 4 layers: tmlp.0/2.bias/weight
        r"^(.*\.)?txtmlp($|\..*)",                   # 4 layers: txtmlp.1/3.bias/weight
        r"^(.*\.)?txtfusion\.projector($|\..*)",     # 1 layer: txtfusion.projector.weight
        r"^.*(^|\.)qknorm\.[qk]norm\.scale$", # attention Q/K norm scales
        r"^.*(^|\.)(pre|post)norm\.scale$",   # transformer norm scales
    ],
    "MiniMax H3": [
        # Verified against the H3 reference quants (pruned 932 keys, full 1035
        # keys, hybrid key-identical to full Ref2VA). The reference quantizer
        # keeps the structural / modulation / norm layers at source precision
        # (BF16/F32/F16) and packs ONLY the four heavy linears
        # (attn.qkv_proj, attn.out_proj, mlp.fc1, mlp.fc2) to the low-bit
        # format. We preserve everything else.
        #
        # Patterns are fullmatch-safe and prefix-tolerant: they match both
        # naked keys (blocks.0.attn.q_norm.weight) and keys carrying a
        # wrapper prefix (model.diffusion_model.blocks.0.attn.q_norm.weight),
        # matching the same behaviour the Krea 2 / LTX-2.3 entries rely on.

        # --- per-block (blocks.N.*) structural layers ---
        r"^.*(^|\.)adaln_proj\.linear($|\..*)",  # timestep modulation (96768 x 8)
        r"^.*(^|\.)attn\.[qk]_norm($|\..*)",     # attention Q/K RMS scales (128,)
        r"^.*(^|\.)norm[12]($|\..*)",           # per-block layer norms (5376,)

        # --- top-level structural layers ---
        r"^(.*\.)?adaln_t_table($|\..*)",        # global timestep adaln (pruned variant)
        r"^(.*\.)?time_embedder($|\..*)",        # timestep embedder (full variant)
        r"^(.*\.)?final_layer($|\..*)",         # final video_out/audio_out/adaln/norm
        r"^(.*\.)?token_refiner($|\..*)",       # conditioning refinement blocks (BF16)
        r"^(.*\.)?video_patch_proj($|\..*)",    # video patchify embedding
        r"^(.*\.)?audio_patch_proj($|\..*)",    # audio patchify embedding
        r"^(.*\.)?condition_proj($|\..*)",      # conditioning projection
        r"^(.*\.)?rope($|\..*)",                # rope.inv_freq (16,)
    ],
}


# Sensitive layers: use FP8 when the base format is lower than FP8.
# On FP8 base these are intentionally left at the base format, not kept
# BF16/FP16; official LTX-2 FP8 cast policy quantizes these transformer
# linears rather than preserving them wholesale.
RESCUE_PATTERNS = {
    # Lightricks' official LTX-2.3 NVFP4 checkpoint is QAD-style, not an
    # FP8-rescue mixed quant: transformer blocks 2..45 are packed NVFP4 while
    # blocks 0, 1, 46, and 47 remain BF16. Keep LTX rescue empty and add the
    # official block-range preserve below only for the NVFP4 base format.
    "LTX-2.3": [],
    "WAN 2.2": [
        # WAN 2.2 I2V/T2V 14B MoE high/low checkpoints use split q/k/v/o
        # projections. Public NVFP4-mixed 14B I2V checkpoints keep attn.v and
        # ffn.2 at FP8 to reduce ghosting/quality loss while leaving q/k/o and
        # ffn.0 packed as NVFP4. The optional suffix keeps these patterns valid
        # in both quantization (raw keys ending .weight/.bias) and GGUF/audit
        # contexts that strip the .weight suffix before matching.
        r"(^|.*\.)self_attn\.v(\.(weight|bias))?$",
        r"(^|.*\.)cross_attn\.v(\.(weight|bias))?$",
        r"(^|.*\.)ffn\.2(\.(weight|bias))?$",
    ],
    "Krea 2": [],
    # H3 reference quants show a clean split: structural layers stay at source
    # precision, heavy linears go to the low-bit base format. No FP8 rescue
    # middle tier needed — for NVFP4, heavy linears go straight to NVFP4.
    "MiniMax H3": [],
}


# Format-specific preserves for LTX-2.3 NVFP4. These mirror the public
# Lightricks/LTX-2.3-nvfp4 header: first two and last two transformer blocks
# are BF16, while the middle blocks are NVFP4/U8. Kept separate from
# PRESERVE_PATTERNS so FP8/INT8/GGUF policies are not changed accidentally.
LTX23_NVFP4_OFFICIAL_PRESERVE_PATTERNS = [
    r"^.*\.transformer_blocks\.(0|1|46|47)($|\..*)",
]

# ---------------------------------------------------------------------------
# MiniMax H3 "NVFP4 HQ" mixed profile (per-block source-precision preserves)
#
# Verified 2026-08-23 against DmitryDB/MiniMax-H3-ComfyUI-Quants
# (FL2VA + Ref2VA NVFP4-HQ, both 1042 tensors, identical plan, Blackwell
# loadtest JSONs shipped in-repo). Of the 200 main-matrix heavy linears,
# 170 are NVFP4-packed (U8 + F8_E4M3 block scale + F32 scalar) and 30 stay
# at source precision (BF16):
#
#   attn.out_proj  -> blocks 0-15, 17, 19, 20, 27, 38, 43-47, 49   (27 layers)
#   mlp.fc2        -> blocks 39, 45, 49                             (3 layers)
#
# Source: reports/layer_policy.json / layer_policy_ref2va.json
# ("bf16_main_layers", profile "quality21"). token_refiner and adaln layers
# already stay at source precision through the standard H3 preserve patterns,
# so the HQ plan is expressed purely as additional per-block preserves.
#
# This is a known, comment-proofed mixed profile — audits must recognize it
# as a valid variant instead of flagging the 30 kept layers as suspicious
# (see utils/pattern_audit.py profile detection).
H3_NVFP4_HQ_PRESERVE_PATTERNS = [
    # attn.out_proj kept at source precision (27 blocks).
    # Prefix-tolerant like the other H3 patterns: H3 keys are naked
    # ("blocks.3.attn.out_proj.weight"), so no mandatory leading prefix.
    r"^.*(^|\.)blocks\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|17|19|20|27|38|43|44|45|46|47|49)\.attn\.out_proj($|\..*)",
    # mlp.fc2 kept at source precision (3 blocks)
    r"^.*(^|\.)blocks\.(39|45|49)\.mlp\.fc2($|\..*)",
]

# Exported layer plan (single source of truth) for audits and tests:
# H3_NVFP4_HQ_LAYER_PLAN maps (block_index, matrix_kind) -> True for every
# heavy linear that the NVFP4-HQ profile keeps at source precision.
H3_NVFP4_HQ_OUTPROJ_BLOCKS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                              14, 15, 17, 19, 20, 27, 38, 43, 44, 45, 46,
                              47, 49)
H3_NVFP4_HQ_FC2_BLOCKS = (39, 45, 49)
H3_NVFP4_HQ_LAYER_PLAN = (
    [(b, "attn.out_proj") for b in H3_NVFP4_HQ_OUTPROJ_BLOCKS]
    + [(b, "mlp.fc2") for b in H3_NVFP4_HQ_FC2_BLOCKS]
)

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
    # NVFP4: rescue sensitive layers to FP8 when an architecture defines
    # rescue patterns. Krea 2 intentionally has no rescue list: the working
    # NVFP4 recipe quantizes transformer attn/MLP weights to NVFP4 and only
    # preserves structural/normalization tensors.
    # INT8 (tensor-wise and ConvRot): do NOT rescue. Winnougan's working
    # int8tensormixed model keeps ALL transformer weights (attention + FFN)
    # as INT8 — rescuing to FP8 breaks compatibility with ComfyUI-INT8-Fast.
    if base_fmt == "float8_e4m3fn":
        rescue_action = "base FP8 (no extra override)"
    elif base_fmt == "nvfp4":
        if model_type == "LTX-2.3":
            preserve_patterns.extend(LTX23_NVFP4_OFFICIAL_PRESERVE_PATTERNS)
        # MiniMax H3 "NVFP4 HQ" mixed profile: on top of the standard H3
        # structural preserves, keep a per-block subset of heavy linears at
        # source precision (27 attn.out_proj + 3 mlp.fc2, verified against
        # DmitryDB's comment-proofed NVFP4-HQ quants). Plain "NVFP4" keeps
        # the pure-NVFP4 policy (all 200 heavy linears packed).
        if model_type == "MiniMax H3" and base_format_ui_label == "NVFP4 HQ":
            preserve_patterns.extend(H3_NVFP4_HQ_PRESERVE_PATTERNS)
        for pat in rescue_patterns:
            config[pat] = {"format": "float8_e4m3fn", "scaling_mode": "tensor"}
        if model_type == "MiniMax H3" and base_format_ui_label == "NVFP4 HQ":
            rescue_action = (
                "H3 mixed profile: keep "
                f"{len(H3_NVFP4_HQ_LAYER_PLAN)} heavy linears at source "
                "precision, rest NVFP4"
            )
        else:
            rescue_action = "rescue to float8_e4m3fn" if rescue_patterns else "no rescue (all eligible weights NVFP4)"
    elif base_fmt == "int8_tensorwise":
        # INT8 tensor-wise: no rescue, all transformer weights stay INT8.
        rescue_action = "no rescue (all INT8)"
        # ConvRot Runtime needs row-wise scaling in the layer config so
        # the per-layer converter applies convrot.  Detect from the UI label.
        if "ConvRot" in base_format_ui_label:
            config["_default"]["scaling_mode"] = "row"
            rescue_action = "no rescue (all INT8 row-wise ConvRot)"
    else:
        rescue_action = "no rescue"

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

