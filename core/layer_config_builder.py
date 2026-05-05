# core/layer_config_builder.py
"""
Auto-generates a layer config for mixed NVFP4 + FP8 quantization.

Strategy:
  1. Run `convert_to_quant --dry-run create-template` to get the
     authoritative list of layers that will be quantized for this
     architecture. This list reflects the per-arch flag (--ltxv2 / --wan)
     so structural layers (norms, embeddings, time/text projections) are
     already excluded.
  2. Mutate the template: set `format: "fp8"` on layers matching the
     keep-FP8 heuristic; leave others as `""` (meaning: use base format
     from --nvfp4).
  3. Save and pass via `--layer-config`.

Schema (verified against real templates, May 2026):
  {
    "_default": {"format": ""},
    "_exclusions": [],
    "<layer.path.no.weight.suffix>": {
        "format": "fp8" | "nvfp4" | "" (= base),
        "_shape": [out, in]
    },
    ...
  }
  Layer keys include the full prefix (e.g. "model.diffusion_model....")
  and do NOT include the trailing ".weight".
"""
import os
import re
import json
import subprocess


# Heuristic: layers to keep at FP8 when base = NVFP4.
# Source: City96 keys_hiprec list in convert.py + the img_tensor_get_type
# bump rules in lcpp.patch (to_v, fused qkv, ffn_down get bumped one tier).
#
# Regexes match against the layer key WITHOUT the .weight suffix
# but WITH the model.diffusion_model. prefix.
KEEP_FP8_PATTERNS = {
    "LTX-2.3": [
        # to_v across every attention variant: attn1, attn2, audio_attn1/2,
        # audio_to_video_attn, video_to_audio_attn, and the connector attns.
        r"\.to_v$",
        # FFN down projection (second linear). Matches both ff.net.2 and
        # audio_ff.net.2.
        r"\.(audio_)?ff\.net\.2$",
        # Connector blocks have their own attention dims (per LTX23 metadata
        # config: connector_attention_head_dim=128, audio_connector=64) and
        # are sensitive to aggressive quant. Keep all connector linears FP8.
        r"(audio|video)_embeddings_connector\.",
    ],
    "WAN 2.2": [
        # Verified May 2026 against wan2_2_i2v_14B high+low templates.
        # WAN keys are naked (no model.diffusion_model. prefix) and use
        # split q/k/v/o (never fused qkv). Both noise checkpoints share
        # this layout exactly.
        #
        # to_v across both attention types (40 blocks each = 80 layers)
        r"\.self_attn\.v$",
        r"\.cross_attn\.v$",
        # FFN down projection (ffn.2 = second linear, 40 layers)
        r"\.ffn\.2$",
        # Structural layers that --wan keeps in the quantize universe but
        # are too sensitive for NVFP4 per City96 lcpp.patch (5 + 1 layers)
        r"^(text_embedding|time_embedding|time_projection)\.",
        r"^head\.",
    ],
}


def _arch_slug(model_type):
    """'WAN 2.2' -> 'wan22', 'LTX-2.3' -> 'ltx23'."""
    return model_type.replace(" ", "").replace(".", "").replace("-", "").lower()


def _arch_flag(model_type):
    return {"WAN 2.2": "--wan", "LTX-2.3": "--ltxv2"}.get(model_type)


def _run_template(source_path, model_type, template_path):
    """Run convert_to_quant --dry-run create-template. Returns (ok, msg)."""
    flag = _arch_flag(model_type)
    if not flag:
        return False, f"Unknown architecture: {model_type}"

    cmd = [
        "convert_to_quant",
        "-i", source_path,
        "--nvfp4",
        flag,
        "--dry-run", "create-template",
        "-o", template_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        return False, "convert_to_quant not on PATH"
    except subprocess.TimeoutExpired:
        return False, "Template generation timed out"
    except Exception as e:
        return False, f"Template subprocess error: {e}"

    if result.returncode != 0:
        return False, f"rc={result.returncode}: {result.stderr.strip()[:400]}"
    if not os.path.exists(template_path):
        return False, "Template flag succeeded but no file written"
    return True, "ok"


def _apply_fp8_overrides(template, regexes):
    """Mutate template: set format='fp8' on matching layers. Returns counts."""
    fp8 = 0
    base = 0
    for key, entry in template.items():
        if key.startswith("_") or not isinstance(entry, dict):
            continue
        if any(rx.search(key) for rx in regexes):
            entry["format"] = "fp8"
            fp8 += 1
        else:
            # Leave format="" so it inherits base type from --nvfp4 flag
            base += 1
    return fp8, base


def build_layer_config(source_path, model_type, filters_dir):
    """
    Build (or reuse) a layer config for mixed NVFP4+FP8 quantization.

    Returns (config_path: str|None, log_lines: list[str]).
    Cache key: source basename + arch slug. Regenerates only when missing.
    """
    log = []
    os.makedirs(filters_dir, exist_ok=True)

    arch = _arch_slug(model_type)
    base = os.path.splitext(os.path.basename(source_path))[0]
    config_path = os.path.join(filters_dir, f"{arch}_{base}_layer_config.json")
    template_path = os.path.join(filters_dir, f"{arch}_{base}_template.json")

    if os.path.exists(config_path):
        log.append(f"[layer-config] Reusing cached: {os.path.basename(config_path)}")
        return config_path, log

    log.append(f"[layer-config] Generating template for {model_type}...")
    ok, msg = _run_template(source_path, model_type, template_path)
    if not ok:
        log.append(f"[layer-config] create-template failed: {msg}")
        return None, log

    try:
        with open(template_path) as f:
            template = json.load(f)
    except Exception as e:
        log.append(f"[layer-config] Template JSON load failed: {e}")
        return None, log

    layer_count = sum(1 for k in template if not k.startswith("_"))
    log.append(f"[layer-config] Template lists {layer_count} quantizable layers")

    patterns = KEEP_FP8_PATTERNS.get(model_type)
    if model_type not in KEEP_FP8_PATTERNS:
        log.append(f"[layer-config] No FP8 patterns defined for {model_type}")
        return None, log

    regexes = [re.compile(p) for p in patterns]
    fp8_count, base_count = _apply_fp8_overrides(template, regexes)

    if fp8_count == 0:
        log.append(
            "[layer-config] ABORT: 0 layers matched FP8 patterns. "
            "Patterns are wrong for this model's naming."
        )
        return None, log

    with open(config_path, "w") as f:
        json.dump(template, f, indent=2)

    log.append(
        f"[layer-config] FP8: {fp8_count} layers | "
        f"NVFP4 (base): {base_count} layers"
    )
    log.append(f"[layer-config] Saved: {os.path.basename(config_path)}")
    return config_path, log
