# core/safetensors_engine.py
import os, subprocess, sys
from core.metadata_manager import inject_metadata, get_current_meta, get_specialized_meta
from core.layer_config_builder import build_layer_config
from config import CONVERT_PY, ROOT_DIR
from utils.file_ops import save_log

FILTERS_DIR = os.path.join(ROOT_DIR, "filters")


def run_safe_conversion(MODELS_DIR, source_path, formats, model_name, model_type,
                        optimizer_choice, options, log_acc, low_vram=False, actcal=False,
                        auto_layer_config=True):

    # Mapping UI selection to CLI flags
    FLAG_MAP = {
        "FP8": ["--comfy_quant"],
        "INT8 Block-wise": ["--int8", "--scaling_mode", "block", "--comfy_quant"],
        "NVFP4": ["--nvfp4", "--comfy_quant"],
    }

    # Base formats where bumping sensitive layers to FP8 is meaningful.
    # FP8 is excluded: nothing higher to bump to within convert_to_quant.
    MIXED_ELIGIBLE = {"NVFP4", "INT8 Block-wise"}

    # Architecture flag mapping. Decoupled from strategy so it cannot be
    # accidentally lost when strategy logic changes.
    ARCH_FLAGS = {
        "WAN 2.2": "--wan",
        "LTX-2.3": "--ltxv2",
    }

    # Per-architecture optimizer parameters for Ultra-Quality strategy.
    # Stored as data, not inlined into branches, so adding a new arch
    # cannot silently fail to set these.
    ULTRA_OPTIMIZER_PARAMS = {
        "WAN 2.2": [
            "--num_iter", "9000",
            "--calib_samples", "10000",
            "--lr", "9e-3",
            "--lr_schedule", "plateau",
            "--early-stop-stall", "20000",
        ],
        "LTX-2.3": [
            "--num_iter", "9000",
            "--calib_samples", "4096",
            "--lr", "1.0",
            "--lr_schedule", "adaptive",
            "--lr_adaptive_mode", "simple-reset",
            "--early-stop-stall", "2000",
        ],
    }

    for fmt in formats:
        suffix = fmt.replace(" ", "_").lower()
        # Output path is finalized AFTER layer config resolution so the
        # _mixed suffix only appears when a layer config actually attaches.
        # If the builder fails or no manual config exists, we degrade to
        # the plain filename - the file accurately reflects what was produced.
        
        # --- 1. RESOLVE LAYER CONFIG (deferred path decision) ---
        layer_config_path = None
        layer_config_log = []
        
        if fmt in MIXED_ELIGIBLE and auto_layer_config:
            layer_config_path, build_log = build_layer_config(
                source_path, model_type, FILTERS_DIR
            )
            layer_config_log.extend(build_log)
        elif fmt in MIXED_ELIGIBLE and not auto_layer_config:
            arch_slug = model_type.replace(" ", "").replace(".", "").replace("-", "").lower()
            base = os.path.splitext(os.path.basename(source_path))[0]
            manual_cfg = os.path.join(FILTERS_DIR, f"{arch_slug}_{base}_layer_config.json")
            if os.path.exists(manual_cfg):
                layer_config_path = manual_cfg
                layer_config_log.append(f"[layer-config] Using manual config: {os.path.basename(manual_cfg)}")
        # FP8 + auto=True is a deliberate no-op (nothing higher to bump to)
        
        # Apply _mixed suffix only if we actually have a config to attach
        if layer_config_path:
            suffix = f"{suffix}_mixed"
        
        final_path = os.path.join(MODELS_DIR, f"{model_name}_{suffix}.safetensors")
        
        # --- 2. BUILD COMMAND ---
        cmd = ["convert_to_quant", "-i", source_path, "-o", final_path, "--save-quant-metadata"]
        
        if low_vram:
            cmd.append("--low-memory")
        
        if fmt in FLAG_MAP:
            cmd.extend(FLAG_MAP[fmt])
        
        # Flush layer config log and attach the flag if resolved
        for line in layer_config_log:
            log_acc += line + "\n"
        if layer_config_path:
            cmd.extend(["--layer-config", layer_config_path])
            log_acc += f"[layer-config] Mixed precision: {fmt} base + FP8 keep-list\n"
            yield log_acc, "Building layer config..."
        elif fmt in MIXED_ELIGIBLE and auto_layer_config:
            log_acc += f"WARN: layer config build failed; running pure {fmt}\n"
            yield log_acc, "Layer config failed"
        
        # --- 3. ARCHITECTURE FLAG (unconditional, exactly once) ---
        arch_flag = ARCH_FLAGS.get(model_type)
        if arch_flag is None:
            log_acc += f"❌ FATAL: Unknown architecture '{model_type}'. Aborting batch.\n"
            yield log_acc, "Aborted: unknown architecture"
            return
        cmd.append(arch_flag)

        # --- 4. STRATEGY FLAGS (independent of architecture) ---
        if options == "Simple":
            cmd.append("--simple")
        elif options == "Auto-Quality (Heur)":
            cmd.append("--heur")
        elif options == "Ultra-Quality (Optimizer)":
            opt_params = ULTRA_OPTIMIZER_PARAMS.get(model_type)
            if opt_params is None:
                log_acc += f"❌ FATAL: No Ultra params defined for '{model_type}'. Aborting.\n"
                yield log_acc, "Aborted: missing Ultra params"
                return
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
        guard_errors = []
        all_arch_flags = list(ARCH_FLAGS.values())
        present_arch = [f for f in all_arch_flags if f in cmd]
        if len(present_arch) == 0:
            guard_errors.append(f"missing architecture flag (expected one of {all_arch_flags})")
        elif len(present_arch) > 1:
            guard_errors.append(f"multiple architecture flags present: {present_arch}")
        elif cmd.count(arch_flag) != 1:
            guard_errors.append(f"{arch_flag} appears {cmd.count(arch_flag)} times (expected 1)")

        strategy_flags = ["--simple", "--heur", "--optimizer"]
        present_strategy = [f for f in strategy_flags if f in cmd]
        if len(present_strategy) == 0:
            guard_errors.append("missing strategy flag (--simple/--heur/--optimizer)")
        if "--simple" in cmd and "--heur" in cmd:
            guard_errors.append("--simple and --heur both present")
        if "--optimizer" in cmd and ("--simple" in cmd or "--heur" in cmd):
            guard_errors.append("--optimizer combined with --simple/--heur")

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
            # This calls the new logic that merges your LTX23_metadata.json
            meta = get_specialized_meta(model_type, model_name, final_path, fmt)
            
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