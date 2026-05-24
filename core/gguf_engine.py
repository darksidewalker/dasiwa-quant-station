# core/gguf_engine.py
import os, subprocess, sys
import hashlib
from core.metadata_manager import write_gguf_meta
from config import CONVERT_PY, FIX_5D_PY, LLAMA_BIN, ROOT_DIR, FIX_5D_DATA
from utils.file_ops import save_log


# Map UI architecture names to convert.py's internal arch strings.
# These are used to locate the per-source 5D fix file written by convert.py
# (named ./fix_5d_tensors_{arch}_{hash}.safetensors), and to decide whether
# the post-quant 5D fix step is needed at all.
_UI_ARCH_TO_CONVERT_ARCH = {
    "WAN 2.2": "wan",
    "LTX-2.3": "ltxv",
}

# Architectures that produce 5D tensors during convert.py and therefore need
# fix_5d_tensors.py to run after llama-quantize. LTX-2.3 VAE and connectors
# are now also protected via the sidecar strategy.
_ARCHS_NEEDING_5D_FIX = {"wan", "hyvid", "ltxv"}


def run_gguf_conversion(MODELS_DIR, source_path, formats, model_name, log_acc,
                        model_type="WAN 2.2", is_full=False):
    base_name = os.path.splitext(os.path.basename(source_path))[0]
    master_gguf = os.path.join(MODELS_DIR, f"{base_name}.gguf")

    convert_arch = _UI_ARCH_TO_CONVERT_ARCH.get(model_type, "wan")
    needs_5d_fix = convert_arch in _ARCHS_NEEDING_5D_FIX

    # 1. Base GGUF Conversion
    if not os.path.exists(master_gguf):
        log_acc += f"📦 Base GGUF missing. Converting {base_name}.gguf...\n"
        log_acc += f"   Architecture: {model_type} (convert.py arch: {convert_arch})\n"
        yield log_acc, "GGUF Base Prep"
        # Added --arch flag and captured output for better error reporting
        conv_res = subprocess.run([sys.executable, CONVERT_PY, "--src", source_path, "--dst", master_gguf, "--arch", convert_arch], cwd=ROOT_DIR, capture_output=True, text=True)
        if conv_res.returncode != 0 or not os.path.exists(master_gguf):
            log_acc += f"❌ Base GGUF Conversion Failed: {conv_res.stderr}\n"
            yield log_acc, "Error"
            return

    q_map = {
        "GGUF_Q8_0": "Q8_0", "GGUF_Q6_K": "Q6_K", "GGUF_Q5_K_M": "Q5_K_M",
        "GGUF_Q4_K_M": "Q4_K_M", "GGUF_Q3_K_S": "Q3_K_S", "GGUF_Q2_K": "Q2_K"
    }

    for fmt in formats:
        q_flag = q_map.get(fmt, "Q8_0")
        out_q = os.path.join(MODELS_DIR, f"{base_name}_{q_flag}.gguf")
        out_qf = os.path.join(MODELS_DIR, f"{base_name}_{q_flag}-fix.gguf")

        # The final user-facing file: "-fix.gguf" for 5D archs, "_Q*.gguf" for non-5D
        final_path = out_qf if needs_5d_fix else out_q

        if os.path.exists(final_path):
            log_acc += f"ℹ️ Skipping {q_flag} (exists: {os.path.basename(final_path)})\n"
            continue

        # 2. Quantization
        log_acc += f"🔨 Quantizing {q_flag}...\n"
        yield log_acc, f"Quantizing {q_flag}"
        result = subprocess.run(
            [os.path.abspath(LLAMA_BIN), master_gguf, out_q, q_flag],
            cwd=ROOT_DIR, capture_output=True, text=True
        )

        if result.returncode != 0:
            log_acc += f"❌ Quantization Failed: {result.stderr}\n"
            yield log_acc, "Error"
            continue

        # 3. 5D tensor fix (only for architectures that need it)
        if needs_5d_fix:
            file_hash = hashlib.md5(os.path.basename(source_path).encode()).hexdigest()[:8]
            dynamic_fix_data = os.path.join(
                ROOT_DIR, f"fix_5d_tensors_{convert_arch}_{file_hash}.safetensors"
            )

            if not os.path.exists(dynamic_fix_data):
                log_acc += (
                    f"❌ 5D fix data missing: {os.path.basename(dynamic_fix_data)}\n"
                    f"   The base conversion step did not produce a 5D fix file. "
                    f"Re-run the conversion or verify the source model.\n"
                )
                yield log_acc, "Error"
                continue

            log_acc += f"🔧 Fixing 5D Tensors using: {os.path.basename(dynamic_fix_data)}\n"
            yield log_acc, f"Fixing {q_flag}"

            fix_result = subprocess.run([
                sys.executable,
                FIX_5D_PY,
                "--src", out_q,
                "--dst", out_qf,
                "--fix", dynamic_fix_data
            ], cwd=ROOT_DIR, capture_output=True, text=True)

            if fix_result.returncode != 0 or not os.path.exists(out_qf):
                log_acc += f"❌ 5D Fix Failed: {fix_result.stderr}\n"
                yield log_acc, "Error"
                continue

            # Remove the intermediate post-quant file; keep only the fixed one
            if os.path.exists(out_q):
                os.remove(out_q)
        else:
            log_acc += f"ℹ️ {model_type} has no 5D tensors; skipping fix step.\n"

        # 4. Metadata Injection (operates on whichever file is the final one)
        if not os.path.exists(final_path):
            log_acc += f"❌ Error: expected output not found: {os.path.basename(final_path)}\n"
            continue

        success, msg = write_gguf_meta(final_path, model_name, model_type, bits=q_flag, is_full=is_full)
        if success:
            log_acc += f"📝 GGUF Meta Injected: {os.path.basename(final_path)} ({msg})\n"
        else:
            log_acc += f"⚠️ Meta Injection Failed: {msg}\n"

        log_acc += f"✅ GGUF Done: {os.path.basename(final_path)}\n"

    save_log(model_name, log_acc)
    yield log_acc, "Finished GGUF"