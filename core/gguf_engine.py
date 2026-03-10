# core/gguf_engine.py
import os, subprocess, sys
import hashlib
from core.metadata_manager import write_gguf_meta
from config import CONVERT_PY, FIX_5D_PY, LLAMA_BIN, ROOT_DIR, FIX_5D_DATA
from utils.file_ops import save_log

def run_gguf_conversion(MODELS_DIR, source_path, formats, model_name, log_acc):
    base_name = os.path.splitext(os.path.basename(source_path))[0]
    master_gguf = os.path.join(MODELS_DIR, f"{base_name}.gguf")

    # 1. Base GGUF Conversion
    if not os.path.exists(master_gguf):
        log_acc += f"📦 Base GGUF missing. Converting {base_name}.gguf...\n"
        yield log_acc, "GGUF Base Prep"
        subprocess.run([sys.executable, CONVERT_PY, "--src", source_path, "--dst", master_gguf], cwd=ROOT_DIR)

    q_map = {
        "GGUF_Q8_0": "Q8_0", "GGUF_Q6_K": "Q6_K", "GGUF_Q5_K_M": "Q5_K_M", 
        "GGUF_Q4_K_M": "Q4_K_M", "GGUF_Q3_K_S": "Q3_K_S", "GGUF_Q2_K": "Q2_K"
    }

    for fmt in formats:
        q_flag = q_map.get(fmt, "Q8_0")
        out_q = os.path.join(MODELS_DIR, f"{base_name}_{q_flag}.gguf")
        out_qf = os.path.join(MODELS_DIR, f"{base_name}_{q_flag}-fix.gguf")
        
        if os.path.exists(out_qf): 
            log_acc += f"ℹ️ Skipping {q_flag} (exists)\n"
            continue

        # 2. Quantization
        log_acc += f"🔨 Quantizing {q_flag}...\n"
        yield log_acc, f"Quantizing {q_flag}"
        result = subprocess.run([os.path.abspath(LLAMA_BIN), master_gguf, out_q, q_flag], cwd=ROOT_DIR, capture_output=True, text=True)
        
        if result.returncode != 0:
            log_acc += f"❌ Quantization Failed: {result.stderr}\n"
            yield log_acc, "Error"
            continue

        # 5D tensor fix
        file_hash = hashlib.md5(os.path.basename(source_path).encode()).hexdigest()[:8]
        dynamic_fix_data = os.path.join(ROOT_DIR, f"fix_5d_tensors_wan_{file_hash}.safetensors")
        
        log_acc += f"🔧 Fixing 5D Tensors using: {os.path.basename(dynamic_fix_data)}\n"
        yield log_acc, f"Fixing {q_flag}"
        
        subprocess.run([
            sys.executable, 
            FIX_5D_PY, 
            "--src", out_q, 
            "--dst", out_qf, 
            "--fix", dynamic_fix_data
        ], cwd=ROOT_DIR)

        # 4. Metadata Injection
        if os.path.exists(out_qf):
            if os.path.exists(out_q):
                os.remove(out_q)
            
            success, msg = write_gguf_meta(out_qf, model_name, fmt)
            if success:
                log_acc += f"📝 GGUF Meta Injected: {os.path.basename(out_qf)}\n"
            else:
                log_acc += f"⚠️ Meta Injection Failed: {msg}\n"
            
            log_acc += f"✅ GGUF Done: {os.path.basename(out_qf)}\n"
        else:
            log_acc += f"❌ Error: {q_flag}-fix.gguf was not created.\n"

    save_log(model_name, log_acc)        
    yield log_acc, "Finished GGUF"