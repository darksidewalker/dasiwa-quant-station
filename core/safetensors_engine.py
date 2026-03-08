# core/safetensors_engine.py
import os, subprocess
from core.metadata_manager import inject_metadata, get_current_meta
from config import CONVERT_PY 
from utils.file_ops import save_log

def run_safe_conversion(MODELS_DIR, source_path, formats, model_name, options, log_acc):
    FLAG_MAP = {
        "FP8": ["--comfy_quant"],
        "INT8 Block-wise": ["--int8", "--block_size", "128", "--comfy_quant"],
        "NVFP4": ["--nvfp4", "--comfy_quant"],
        "Auto-Quality (Heur)": ["--heur"]
    }
    
    ULTRA_ARGS = [
        "--comfy_quant", "--save-quant-metadata", "--wan", 
        "--lr_schedule", "plateau", "--lr_patience", "2", "--lr_factor", "0.96", 
        "--lr_min", "9e-9", "--lr_cooldown", "0", "--lr_threshold", "1e-11", 
        "--num_iter", "9000", "--calib_samples", "45000", 
        "--lr", "9.916700000002915715e-3", "--top_p", "0.05", 
        "--min_k", "64", "--max_k", "256", "--early-stop-stall", "20000", 
        "--early-stop-lr", "1e-8", "--early-stop-loss", "9e-8", "--lr-shape-influence", "3.5"
    ]

    for fmt in formats:
        suffix = fmt.lower().replace(" ", "_")
        final_path = source_path.replace(".safetensors", f"_{suffix}.safetensors")
        
        cmd = ["python", CONVERT_PY, "-i", source_path, "-o", final_path]
        
        if options == "Ultra-Quality (Optimizer)":
            log_acc += f"💎 MODE: Ultra-Quality Optimizer (9000 Iters) for {fmt}\n"
            cmd.extend(ULTRA_ARGS)
        elif options == "Auto-Quality (Heur)":
            # If using Auto-Heuristic, ensure the flag is added
            if "--heur" not in cmd:
                cmd.append("--heur")
            if fmt in FLAG_MAP: 
                cmd.extend([f for f in FLAG_MAP[fmt] if f != "--heur"])
        else:
            # Fast Mode (Simple) - Just use the basic format flags
            if fmt in FLAG_MAP:
                cmd.extend([f for f in FLAG_MAP[fmt] if f != "--heur"])

        # Ensure --wan is always present for Wan models regardless of tweak
        if "wan" in source_path.lower() and "--wan" not in cmd: 
            cmd.append("--wan")

        yield log_acc, f"Safetensors: {fmt}"
        
        subprocess.run(cmd)

        if os.path.exists(final_path):
            meta = get_current_meta(model_name, fmt)
            inject_metadata(final_path, meta)
            log_acc += f"📝 Meta Injected: {os.path.basename(final_path)}\n"

    save_log(model_name, log_acc)        
    yield log_acc, "Finished Safetensors"