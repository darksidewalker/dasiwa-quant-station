# core/gguf_engine.py
import os, subprocess, sys, json, re, hashlib
from core.metadata_manager import write_gguf_meta
from config import ROOT_DIR
from safetensors import safe_open
from core.layer_config_builder import ALWAYS_SKIP_PATTERNS, KEEP_HIGHER_PRECISION_PATTERNS, BAKED_VAE_PATTERNS
from utils.file_ops import save_log

GGUFY_BIN = os.path.realpath(os.path.join(ROOT_DIR, "bin", "ggufy"))

# Map UI architecture names to canonical ggufy metadata arch strings.
# These values are used for the ggufy --arch free-form metadata field
# and also for sensitivity / 5D fix file naming.
_UI_ARCH_TO_CONVERT_ARCH = {
    "WAN 2.2": "wan",
    "LTX-2.3": "ltxv",
}

def _sanitize_model_name(model_name):
    if not model_name or not isinstance(model_name, str):
        return None
    candidate = model_name.strip()
    if not candidate:
        return None
    # Reject navigational path components and preserve a safe flat filename
    if ".." in candidate:
        return None
    candidate = candidate.replace("\\", "_").replace("/", "_")
    candidate = candidate.strip("._ ")
    return candidate if candidate else None


def _generate_ggufy_sensitivities(source_path, model_type, convert_arch, is_full):
    """Builds a temporary JSON sensitivity file for ggufy based on local patterns."""
    sensitivities = {}
    
    # Combine patterns for this architecture
    skip_list = list(ALWAYS_SKIP_PATTERNS.get(model_type, []))
    keep_high = list(KEEP_HIGHER_PRECISION_PATTERNS.get(model_type, []))
    
    if is_full:
        skip_list.extend(BAKED_VAE_PATTERNS)

    with safe_open(source_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            # Strip .weight for matching logic consistency with patterns
            match_name = key[:-7] if key.endswith(".weight") else key

            # Critical layers (95+ score in ggufy) are always kept at source precision
            if any(re.search(pat, match_name) for pat in skip_list):
                sensitivities[key] = 100
                continue
            
            # High precision layers (80-94 score) are preserved or bumped to Q8_0
            if any(re.search(pat, match_name) for pat in keep_high):
                sensitivities[key] = 90

    sens_path = os.path.abspath(os.path.join(ROOT_DIR, f"ggufy_sens_{convert_arch}_{hashlib.md5(source_path.encode()).hexdigest()[:8]}.json"))
    with open(sens_path, "w") as j:
        json.dump(sensitivities, j)
    return sens_path

def run_gguf_conversion(MODELS_DIR, source_path, formats, model_name, log_acc,
                        model_type="WAN 2.2", is_full=False):
    base_name = os.path.splitext(os.path.basename(source_path))[0]
    convert_arch = _UI_ARCH_TO_CONVERT_ARCH.get(model_type, "wan")

    q_map = {
        "GGUF_F32": "f32",
        "GGUF_BF16": "bf16",
        "GGUF_F16": "f16",
        "GGUF_Q8_0": "q8_0",
        "GGUF_Q6_K": "q6_k",
        "GGUF_Q5_K": "q5_k",
        "GGUF_Q4_K": "q4_k",
        "GGUF_Q3_K": "q3_k",
        "GGUF_Q2_K": "q2_k",
        "GGUF_Q1_0": "q1_0",
    }

    # 1. Sanitize and Normalize All Paths
    sanitized_model_name = _sanitize_model_name(model_name)
    if not sanitized_model_name:
        log_acc += f"❌ Error: Invalid model name provided. Avoid path separators or '..'. Received: {repr(model_name)}\n"
        yield log_acc, "Error"
        return

    # Normalize and resolve symlinks for paths to avoid filesystem weirdness
    MODELS_DIR = os.path.realpath(os.path.expanduser(MODELS_DIR)).rstrip(os.sep)
    source_path = os.path.realpath(os.path.expanduser(source_path))

    if not os.path.isdir(MODELS_DIR):
        log_acc += f"❌ Error: Model Directory is invalid: {MODELS_DIR}\n"
        yield log_acc, "Error"
        return

    if not os.path.isfile(source_path):
        log_acc += f"❌ Error: Source file does not exist: {source_path}\n"
        yield log_acc, "Error"
        return

    if not os.path.isfile(GGUFY_BIN) or not os.access(GGUFY_BIN, os.X_OK):
        log_acc += f"❌ Error: GGUFY binary missing or not executable: {GGUFY_BIN}\n"
        log_acc += "   Tip: Run start-linux.sh to install or repair the GGUFY binary.\n"
        yield log_acc, "Error"
        return

    for fmt in formats:
        q_flag = q_map.get(fmt, "q8_0")
        out_filename = f"{sanitized_model_name}_{q_flag.upper()}"
        final_path = os.path.join(MODELS_DIR, f"{out_filename}.gguf")

        if os.path.exists(final_path):
            log_acc += f"ℹ️ Skipping {q_flag} (exists: {os.path.basename(final_path)})\n"
            continue

        # 🚀 Direct GGUFY Conversion
        log_acc += f"🔨 GGUFY Converting {base_name} to {q_flag}...\n"
        yield log_acc, f"Quantizing {q_flag}"
        
        sens_path = None
        try:
            # 1. Generate sensitivity map to respect our layer_config_builder patterns
            sens_path = _generate_ggufy_sensitivities(source_path, model_type, convert_arch, is_full)

            # 2. Setup output directory
            target_dir = os.path.dirname(final_path)
            try:
                os.makedirs(target_dir, exist_ok=True)
            except OSError as e:
                log_acc += f"❌ Error creating output directory: {target_dir}\n"
                log_acc += f"   {str(e)}\n"
                yield log_acc, "Error"
                continue
            pure_output_name = os.path.splitext(os.path.basename(final_path))[0]

            # 3. Build Command: ggufy convert <input-file>
            real_target_dir = os.path.realpath(target_dir)
            real_source_path = os.path.realpath(source_path)
            real_sens_path = os.path.realpath(sens_path)
            cmd = [
                GGUFY_BIN, "convert",
                "--datatype", q_flag,
                "--output-dir", real_target_dir,
                "--output-name", pure_output_name,
            ]
            if model_type and model_type != "Not set":
                gguf_arch = _UI_ARCH_TO_CONVERT_ARCH.get(model_type, model_type)
                cmd.extend(["--arch", gguf_arch])
            cmd.extend([
                "--sensitivities", real_sens_path,
                real_source_path
            ])
            log_acc += f"   CMD: {' '.join(cmd)}\n"

            # Use Popen to stream logs to the UI terminal in real-time
            process = subprocess.Popen(
                cmd, cwd=ROOT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True
            )
            for line in process.stdout:
                log_acc += line
                yield log_acc, f"Quantizing {q_flag}..."
            process.wait()

            if process.returncode != 0:
                log_acc += f"❌ GGUFY Failed with return code {process.returncode}\n"
                yield log_acc, "Error"
                continue

        except OSError as e:
            log_acc += f"❌ Execution Error: {str(e)}\n"
            if e.errno == 8:
                log_acc += "   Tip: The GGUFY binary at bin/ggufy is invalid or corrupted. Run start-linux.sh again.\n"
            yield log_acc, "Error"
            continue
        finally:
            if sens_path and os.path.exists(sens_path):
                os.remove(sens_path)

        # 4. Final Meta Injection
        if os.path.exists(final_path):
            success, msg = write_gguf_meta(final_path, model_name, model_type, bits=q_flag.upper(), is_full=is_full)
            if success:
                log_acc += f"📝 GGUF Meta Injected: {os.path.basename(final_path)} ({msg})\n"
            else:
                log_acc += f"⚠️ Meta Injection Failed: {msg}\n"
        else:
            log_acc += "❌ Final GGUF file missing after conversion.\n"
        log_acc += f"✅ GGUF Done: {os.path.basename(final_path)}\n"

    save_log(sanitized_model_name, log_acc)
    yield log_acc, "Finished GGUF"