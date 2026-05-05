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
        "NVFP4+FP8 Mixed": ["--nvfp4", "--comfy_quant"],
    }

    for fmt in formats:
        # Define output path
        suffix = fmt.replace(" ", "_").lower()
        final_path = os.path.join(MODELS_DIR, f"{model_name}_{suffix}.safetensors")
        
        # Base Command
        cmd = ["convert_to_quant", "-i", source_path, "-o", final_path, "--save-quant-metadata"]
        
        # --- 1. HARDWARE & CALIBRATION FLAGS ---
        if low_vram:
            cmd.append("--low-memory")
        
        # --- 2. FORMAT SPECIFIC FLAGS ---
        if fmt in FLAG_MAP:
            cmd.extend(FLAG_MAP[fmt])

        # --- 2b. AUTO LAYER CONFIG (Mixed precision only) ---
        if fmt == "NVFP4+FP8 Mixed" and auto_layer_config:
            config_path, build_log = build_layer_config(
                source_path, model_type, FILTERS_DIR
            )
            for line in build_log:
                log_acc += line + "\n"
            yield log_acc, "Building layer config..."
            if config_path:
                cmd.extend(["--layer-config", config_path])
            else:
                log_acc += "WARN: layer config build failed; running pure NVFP4\n"
                yield log_acc, "Layer config failed"
        elif fmt == "NVFP4+FP8 Mixed":
            # Auto-build disabled: look for hand-edited config
            arch_slug = model_type.replace(" ", "").replace(".", "").replace("-", "").lower()
            base = os.path.splitext(os.path.basename(source_path))[0]
            manual_cfg = os.path.join(FILTERS_DIR, f"{arch_slug}_{base}_layer_config.json")
            if os.path.exists(manual_cfg):
                cmd.extend(["--layer-config", manual_cfg])
                log_acc += f"[layer-config] Using manual config: {os.path.basename(manual_cfg)}\n"
            else:
                log_acc += f"[layer-config] No manual config at {manual_cfg}; running pure NVFP4\n"
        
        # --- 3. ARCHITECTURE & TWEAK LOGIC ---
        if options == "Simple":
            cmd.append("--simple")
            if model_type == "WAN 2.2": cmd.append("--wan")
            elif model_type == "LTX-2.3": cmd.append("--ltxv2")
            
        elif options == "Auto-Quality (Heur)":
            cmd.append("--heur")
            if model_type == "WAN 2.2": cmd.append("--wan")
            elif model_type == "LTX-2.3": cmd.append("--ltxv2")

        else: # Ultra-Quality (Optimizer)
            if model_type == "WAN 2.2":
                cmd.extend([
                    "--wan", 
                    "--optimizer", optimizer_choice,
                    "--num_iter", "9000", 
                    "--calib_samples", "10000",
                    "--lr", "9e-3",
                    "--lr_schedule", "plateau",
                    "--early-stop-stall", "20000"
                ])
            elif model_type == "LTX-2.3":
                cmd.extend([
                    "--ltxv2", 
                    "--optimizer", optimizer_choice,
                    "--num_iter", "9000",
                    "--calib_samples", "4096",
                    "--lr", "1.0",
                    "--lr_schedule", "adaptive", 
                    "--lr_adaptive_mode", "simple-reset",
                    "--early-stop-stall", "2000"
                ])

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