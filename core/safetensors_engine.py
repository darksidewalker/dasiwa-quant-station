# core/safetensors_engine.py
import os, subprocess, sys
from core.metadata_manager import inject_metadata, get_current_meta, get_specialized_meta, merge_custom_metadata
from core.layer_config_builder import write_layer_config
from utils.arch_detector import verify_architecture_match
from config import CONVERT_PY, ROOT_DIR
from utils.file_ops import save_log

FILTERS_DIR = os.path.join(ROOT_DIR, "filters")


def run_safe_conversion(MODELS_DIR, source_path, formats, model_name, model_type,
                        optimizer_choice, options, log_acc, low_vram=False, actcal=False,
                        is_full_checkpoint=False, custom_metadata=None):

    # Mapping UI selection to CLI flags
    FLAG_MAP = {
        "FP8": ["--comfy_quant"],
        "INT8 Tensor-wise": [
            "--int8",
            "--scaling_mode", "tensor",
            "--comfy_quant",
        ],
        "INT8 Row-wise ConvRot Runtime": [
            "--int8",
            "--scaling_mode", "row",
            "--convrot",
            "--convrot-group-size", "256",
            "--comfy_quant",
        ],
        # Backward-compatible alias for stale UI/session values. This now
        # intentionally maps to the safe non-ConvRot path.
        "INT8 Row-wise ConvRot": [
            "--int8",
            "--scaling_mode", "tensor",
            "--comfy_quant",
        ],
        "NVFP4": ["--nvfp4", "--comfy_quant"],
    }

    # Base formats where the automatic layer config is meaningful.
    # Preserve patterns always stay at source precision; rescue patterns use
    # the FP8 base as-is and are bumped to FP8 for NVFP4 / INT8.
    LAYER_CONFIG_ELIGIBLE = {
        "FP8",
        "NVFP4",
        "INT8 Tensor-wise",
        "INT8 Row-wise ConvRot Runtime",
        "INT8 Row-wise ConvRot",
    }

    # Architecture registry. Single source of truth for:
    #   - the CLI flag passed to convert_to_quant
    #   - the optimizer-driven learned-rounding params for that arch
    # An entry with flag=None means "Not set": no preset is applied,
    # convert_to_quant runs with its own defaults. In that mode we also
    # bypass layer-config building and architecture verification, since
    # neither is meaningful without a declared architecture.
    #
    # Flag list verified against convert_to_quant --help output
    # (silveroxides/convert_to_quant, May 2026).
    #
    # Optimizer params track current convert_to_quant defaults explicitly:
    # Prodigy, 4000 iters, 3072 calibration samples, LR 1.0, plateau
    # schedule, and stall 2000. Keeping them explicit makes command logs
    # reproducible while avoiding stale local quality tuning.
    _OPTIMIZER_DEFAULT = [
        "--num_iter", "4000",
        "--calib_samples", "3072",
        "--lr", "1.0",
        "--lr_schedule", "plateau",
        "--early-stop-stall", "2000",
    ]

    ARCH_REGISTRY = {
        "Not set":           {"flag": None,                  "optimizer": _OPTIMIZER_DEFAULT},
        # Verified-pattern archs (have entries in layer_config_builder,
        # arch_detector, and assets.MODEL_METADATA_CONFIGS).
        "WAN 2.2":           {"flag": "--wan",               "optimizer": _OPTIMIZER_DEFAULT},
        "LTX-2.3":           {"flag": "--ltxv2",             "optimizer": _OPTIMIZER_DEFAULT},
        # Other convert_to_quant presets. No verified layer-name patterns
        # in this project yet, so layer-config is skipped and we rely on
        # the convert_to_quant preset's own skip rules.
        "Flux.2":            {"flag": "--flux2",             "optimizer": _OPTIMIZER_DEFAULT},
        "Hunyuan Video":     {"flag": "--hunyuan",           "optimizer": _OPTIMIZER_DEFAULT},
        "Qwen Image":        {"flag": "--qwen",              "optimizer": _OPTIMIZER_DEFAULT},
        "Z-Image":           {"flag": "--zimage",            "optimizer": _OPTIMIZER_DEFAULT},
        "Z-Image Refiner":   {"flag": "--zimage_refiner",    "optimizer": _OPTIMIZER_DEFAULT},
        "Anima":             {"flag": "--anima",             "optimizer": _OPTIMIZER_DEFAULT},
        "Radiance":          {"flag": "--radiance",          "optimizer": _OPTIMIZER_DEFAULT},
        "Distillation Large":{"flag": "--distillation_large","optimizer": _OPTIMIZER_DEFAULT},
        "Distillation Small":{"flag": "--distillation_small","optimizer": _OPTIMIZER_DEFAULT},
        "NeRF Large":        {"flag": "--nerf_large",        "optimizer": _OPTIMIZER_DEFAULT},
        "NeRF Small":        {"flag": "--nerf_small",        "optimizer": _OPTIMIZER_DEFAULT},
        # Text-encoder / non-diffusion presets. Kept available for power
        # users who want to quantize companion models.
        "T5-XXL":            {"flag": "--t5xxl",             "optimizer": _OPTIMIZER_DEFAULT},
        "Qwen 3.5":          {"flag": "--qwen35",            "optimizer": _OPTIMIZER_DEFAULT},
        "Mistral":           {"flag": "--mistral",           "optimizer": _OPTIMIZER_DEFAULT},
        "Visual":            {"flag": "--visual",            "optimizer": _OPTIMIZER_DEFAULT},
        "Generic Text":      {"flag": "--generic_text",      "optimizer": _OPTIMIZER_DEFAULT},
    }

    # Backwards-compat views derived from the registry. Other code in the
    # codebase reads these directly (e.g. command guard below).
    ARCH_FLAGS = {k: v["flag"] for k, v in ARCH_REGISTRY.items()
                  if v["flag"] is not None}

    # === ARCHITECTURE VERIFICATION ===
    # Before doing anything else, verify the source file matches the user's
    # declared architecture. Mismatch (e.g. LTX file with WAN selected) causes
    # convert_to_quant to apply the wrong preset, quantizing structural layers
    # that should be preserved and producing damaged output.
    #
    # Skipped when model_type is "Not set" (no preset is declared, so there
    # is nothing to verify against) or when the arch has no registered
    # markers in arch_detector (unverified preset; user takes responsibility).
    if model_type == "Not set":
        log_acc += "ℹ️  Architecture verification skipped (Not set).\n"
        yield log_acc, "Architecture: not set"
    else:
        log_acc += "🔎 Verifying source architecture...\n"
        yield log_acc, "Verifying architecture"
        arch_ok, arch_msg = verify_architecture_match(source_path, model_type)
        log_acc += f"{arch_msg}\n"
        if not arch_ok:
            yield log_acc, "Aborted: architecture mismatch"
            return
        yield log_acc, "Architecture verified"

    for fmt in formats:
        suffix = fmt.replace(" ", "_").lower()
        # Output path is finalized AFTER layer config resolution so the
        # _mixed suffix only appears when a layer config actually attaches.
        # If the builder fails or no manual config exists, we degrade to
        # the plain filename - the file accurately reflects what was produced.
        
        # --- 1. RESOLVE LAYER CONFIG (deferred path decision) ---
        layer_config_path = None
        layer_config_log = []
        is_exact_config = False

        # "Not set" disables every layer-config path: we explicitly chose
        # to let convert_to_quant run with its own defaults, with no
        # per-layer overrides from us. Exact configs are also bypassed
        # because they're keyed to a specific reference architecture.
        layer_config_enabled = (model_type != "Not set")

        # Exact configs (built from a reference FP8) take precedence over
        # auto/manual regex configs. They live at filters/_exact_*.json
        # and are produced by utils/exact_config.py from the UI.
        # We look for one matching the current base format.
        if layer_config_enabled and fmt in LAYER_CONFIG_ELIGIBLE:
            fmt_slug = fmt.replace(" ", "_").lower()
            # Glob for any _exact_*_{fmt_slug}.json file in filters/
            if os.path.isdir(FILTERS_DIR):
                for fn in os.listdir(FILTERS_DIR):
                    if fn.startswith("_exact_") and fn.endswith(f"_{fmt_slug}.json"):
                        layer_config_path = os.path.join(FILTERS_DIR, fn)
                        is_exact_config = True
                        layer_config_log.append(
                            f"[layer-config] EXACT mode: using {fn}"
                        )
                        layer_config_log.append(
                            f"[layer-config] (Regex-based auto config is bypassed "
                            f"because an exact config exists.)"
                        )
                        break

        if layer_config_enabled and not is_exact_config and fmt in LAYER_CONFIG_ELIGIBLE:
            layer_config_path, build_log = write_layer_config(
                model_type, fmt, out_dir=FILTERS_DIR
            )
            layer_config_log.extend(build_log)
        elif not layer_config_enabled and fmt in LAYER_CONFIG_ELIGIBLE:
            layer_config_log.append(
                "[layer-config] Skipped: architecture is 'Not set'. "
                "Running convert_to_quant with no layer overrides."
            )
        
        # Apply suffix based on which config mode is in use, so the user
        # can distinguish output files: _exact for reference-derived,
        # _mixed for regex-based, plain for no-config.
        if is_exact_config:
            suffix = f"{suffix}_exact"
        elif layer_config_path:
            suffix = f"{suffix}_mixed"
        
        final_path = os.path.join(MODELS_DIR, f"{model_name}_{suffix}.safetensors")

        # Ensure target directory exists (handles cases where model_name contains subfolders)
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        
        # --- 2. BUILD COMMAND ---
        cmd = ["convert_to_quant", "-i", source_path, "-o", final_path, "--save-quant-metadata"]
        
        if low_vram:
            cmd.append("--low-memory")
        
        if fmt in FLAG_MAP:
            cmd.extend(FLAG_MAP[fmt])

        if fmt == "INT8 Row-wise ConvRot Runtime":
            log_acc += (
                "NOTE: INT8 Row-wise ConvRot requires a runtime that reads "
                ".comfy_quant convrot metadata and rotates activations.\n"
            )
        
        # Flush layer config log and attach the flag if resolved
        for line in layer_config_log:
            log_acc += line + "\n"
        if layer_config_path:
            cmd.extend(["--layer-config", layer_config_path])
            yield log_acc, "Building layer config..."
        elif fmt in LAYER_CONFIG_ELIGIBLE and model_type != "Not set":
            # No layer config attached. Could be either:
            #   - arch has no patterns registered (expected for unverified
            #     archs - convert_to_quant's preset handles skips), or
            #   - the builder actually failed (rare, logged above).
            # The detail is in layer_config_log; the headline note here
            # stays neutral so unverified archs don't look broken.
            log_acc += f"NOTE: no layer config attached; running plain {fmt}\n"
            yield log_acc, "Plain quantization (no layer config)"
        
        # --- 3. ARCHITECTURE FLAG ---
        # Resolved from the registry. "Not set" maps to flag=None, in which
        # case no architecture flag is appended and convert_to_quant uses
        # its own defaults.
        arch_entry = ARCH_REGISTRY.get(model_type)
        if arch_entry is None:
            log_acc += f"❌ FATAL: Unknown architecture '{model_type}'. Aborting batch.\n"
            yield log_acc, "Aborted: unknown architecture"
            return
        arch_flag = arch_entry["flag"]
        if arch_flag is not None:
            cmd.append(arch_flag)

        # --- 4. STRATEGY FLAGS (independent of architecture) ---
        if options == "Simple":
            cmd.append("--simple")
        elif options == "Optimizer-driven":
            opt_params = arch_entry["optimizer"]
            cmd.extend(["--optimizer", optimizer_choice])
            cmd.extend(opt_params)
        else:
            log_acc += f"❌ FATAL: Unknown strategy '{options}'. Aborting batch.\n"
            yield log_acc, "Aborted: unknown strategy"
            return

        # --- 5. SAFETY GUARD ---
        # Validate the assembled command before launching the subprocess.
        # The guard catches the class of bug where a refactor silently drops
        # the architecture flag or the strategy preset (which causes
        # convert_to_quant to fall back to its own defaults and quantize
        # tensors the architecture preset would have skipped, producing
        # damaged output that LOOKS valid but is multiple GB lighter).
        #
        # "Not set" is an intentional, user-acknowledged version of this
        # behavior, so the guard allows zero architecture flags in that
        # mode only.
        guard_errors = []
        all_arch_flags = list(ARCH_FLAGS.values())
        present_arch = [f for f in all_arch_flags if f in cmd]
        if arch_flag is None:
            # "Not set": zero arch flags is expected; multiple would be a bug.
            if len(present_arch) > 0:
                guard_errors.append(
                    f"unexpected architecture flag(s) with 'Not set': {present_arch}"
                )
        else:
            if len(present_arch) == 0:
                guard_errors.append(f"missing architecture flag (expected one of {all_arch_flags})")
            elif len(present_arch) > 1:
                guard_errors.append(f"multiple architecture flags present: {present_arch}")
            elif cmd.count(arch_flag) != 1:
                guard_errors.append(f"{arch_flag} appears {cmd.count(arch_flag)} times (expected 1)")

        strategy_flags = ["--simple", "--optimizer"]
        present_strategy = [f for f in strategy_flags if f in cmd]
        if len(present_strategy) == 0:
            guard_errors.append("missing strategy flag (--simple/--optimizer)")
        if "--optimizer" in cmd and "--simple" in cmd:
            guard_errors.append("--optimizer combined with --simple")

        if guard_errors:
            log_acc += f"❌ FATAL command guard for {fmt}:\n"
            for err in guard_errors:
                log_acc += f"   - {err}\n"
            log_acc += f"   Command would have been: {' '.join(cmd)}\n"
            log_acc += f"   Skipping {fmt} to prevent damaged output.\n"
            yield log_acc, f"Aborted {fmt}: command validation failed"
            continue

        log_acc += f"\n🛠️ CONFIG: {model_type} | FMT: {fmt} | TWEAK: {options}\n"
        log_acc += f"▶️ COMMAND: {' '.join(cmd)}\n"
        yield log_acc, f"Quantizing {fmt}..."

        # Subprocess execution
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, bufsize=1, universal_newlines=True
        )

        current_line = ""
        has_finished = False # Flag to prevent multiple 100% lines
        
        while True:
            char = process.stdout.read(1)
            if not char and process.poll() is not None: 
                break
            
            if char in ['\n', '\r']:
                clean_line = current_line.strip()
                
                # Identify if this is a spammy optimization line
                is_progress_spam = any(x in clean_line.lower() for x in ["optimizing", "step", "worse_count", "%|"])
                
                # Case 1: Standard logs (Errors, initialization, etc.)
                if clean_line and not is_progress_spam:
                    log_acc += clean_line + "\n"
                    yield log_acc, f"Quantizing {fmt}..."
                
                # Case 2: The very first 100% line we encounter
                elif "100%" in clean_line and not has_finished:
                    log_acc += clean_line + "\n"
                    yield log_acc, f"Quantization of {fmt} Complete."
                    has_finished = True # Lock it so no more 100% lines pass through

                current_line = ""
            else:
                current_line += char

        process.wait()

        # --- 4. FINALIZATION & METADATA ---
        if process.returncode == 0 and os.path.exists(final_path):
            # Build metadata: merge_custom_metadata preserves required LTX 2.3
            # functional fields while allowing user-edited custom metadata to
            # overlay on top.
            meta = merge_custom_metadata(
                model_type, model_name, final_path,
                bits=fmt,
                custom_meta=custom_metadata,
                is_full=is_full_checkpoint,
            )
            
            # Inject the resulting dictionary into the safetensor
            success, msg = inject_metadata(final_path, meta)
            
            if success:
                log_acc += f"📝 Meta Injected [{model_type}]: {os.path.basename(final_path)}\n"
            else:
                log_acc += f"⚠️ Metadata injection failed: {msg}\n"
        else:
            log_acc += f"❌ Quantization Failed. Return code: {process.returncode}\n"

    save_log(model_name, log_acc)       
    yield log_acc, "Finished Batch"
