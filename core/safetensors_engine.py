# core/safetensors_engine.py
import os, subprocess, sys
import threading
from collections import deque
from core.metadata_manager import inject_metadata, get_current_meta, get_specialized_meta
from config import CONVERT_PY 
from utils.file_ops import save_log


# Patterns that indicate a memory/process-kill failure mode worth flagging to the user
_OOM_HINTS = (
    "out of memory",
    "cuda out of memory",
    "cuda error: out of memory",
    "cudaerrormemoryallocation",
    "killed",
    "memoryerror",
    "torch.cuda.outofmemoryerror",
)


def _stderr_reader(stream, sink):
    """Drain a subprocess stream into a deque (keeps last N lines)."""
    try:
        for line in iter(stream.readline, ''):
            if line:
                sink.append(line.rstrip('\r\n'))
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _diagnose_failure(stderr_lines, returncode):
    """Build a diagnostic message from collected stderr."""
    joined = "\n".join(stderr_lines)
    lower = joined.lower()
    hints = []
    for pat in _OOM_HINTS:
        if pat in lower:
            hints.append(f"  • Detected pattern: '{pat}'")
            break  # one hint is enough
    diag = [f"❌ Quantization Failed. Return code: {returncode}"]
    if returncode in (-9, 137):
        diag.append("  • Process was SIGKILLed (likely OOM-killer). Check `dmesg | tail`.")
    if hints:
        diag.append("  • Looks like an out-of-memory failure.")
        diag.extend(hints)
        diag.append("  • Try: enable Low VRAM Mode, or reduce calib_samples / num_iter.")
    if stderr_lines:
        diag.append("---- stderr (last lines) ----")
        diag.extend(f"  {ln}" for ln in stderr_lines)
        diag.append("-----------------------------")
    return "\n".join(diag) + "\n"

def run_safe_conversion(MODELS_DIR, source_path, formats, model_name, model_type, 
                        optimizer_choice, options, log_acc, low_vram=False, actcal=False):

    # Mapping UI selection to CLI flags
    FLAG_MAP = {
        "FP8": ["--comfy_quant"],
        "INT8 Block-wise": ["--int8", "--scaling_mode", "block", "--comfy_quant"],
        "NVFP4": ["--nvfp4", "--comfy_quant"],
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
        
        # --- 3. ARCHITECTURE & TWEAK LOGIC ---
        if options == "Simple":
            cmd.append("--simple")
            if model_type == "WAN 2.2": cmd.append("--wan")
            elif model_type == "LTX-2": cmd.append("--ltxv2")
            
        elif options == "Auto-Quality (Heur)":
            cmd.append("--heur")
            if model_type == "WAN 2.2": cmd.append("--wan")
            elif model_type == "LTX-2": cmd.append("--ltxv2")

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
            elif model_type == "LTX-2":
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

        # Subprocess execution: split stderr so Python tracebacks / CUDA OOM
        # messages come through their own channel and aren't lost in stdout buffering.
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Background drain of stderr (keep last 60 lines for diagnostics)
        stderr_tail = deque(maxlen=60)
        stderr_thread = threading.Thread(
            target=_stderr_reader,
            args=(process.stderr, stderr_tail),
            daemon=True,
        )
        stderr_thread.start()

        has_finished = False  # Flag to prevent multiple 100% lines

        # Line-buffered stdout read. readline() returns '' only at EOF, so we
        # don't lose the final flush before a crash the way char-by-char did.
        for line in iter(process.stdout.readline, ''):
            clean_line = line.rstrip('\r\n').strip()
            if not clean_line:
                continue

            # Identify if this is a spammy optimization line
            is_progress_spam = any(
                x in clean_line.lower()
                for x in ["optimizing", "step", "worse_count", "%|"]
            )

            # Case 1: Standard logs (Errors, initialization, etc.)
            if not is_progress_spam:
                log_acc += clean_line + "\n"
                yield log_acc, f"Quantizing {fmt}..."

            # Case 2: The very first 100% line we encounter
            elif "100%" in clean_line and not has_finished:
                log_acc += clean_line + "\n"
                yield log_acc, f"Quantization of {fmt} Complete."
                has_finished = True  # Lock it so no more 100% lines pass through

        try:
            process.stdout.close()
        except Exception:
            pass

        process.wait()
        stderr_thread.join(timeout=5)

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
            log_acc += _diagnose_failure(list(stderr_tail), process.returncode)
            yield log_acc, f"Failed: {fmt}"

    save_log(model_name, log_acc)       
    yield log_acc, "Finished Batch"