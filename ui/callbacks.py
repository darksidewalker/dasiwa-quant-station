# ui/callbacks.py
import gradio as gr
from core.safetensors_engine import run_safe_conversion
from core.gguf_engine import run_gguf_conversion
from core.metadata_manager import update_metadata_preview, read_any_metadata
from utils.file_ops import list_files, get_full_path
from config import MODELS_DIR
from utils.scanner_5d import scan_5d_tensors
from utils.file_ops import get_full_path

def setup_callbacks(
    base_dd, friendly_name, refresh_btn, run_btn, stop_btn, 
    q_format, pipeline_status, extra_flags, terminal_box, 
    metadata_input, inject_btn, read_btn
):
    # --- 1. Model List Management ---
    refresh_btn.click(fn=list_files, outputs=[base_dd])

    # --- 2. Dynamic Metadata Preview ---
    friendly_name.change(
        fn=update_metadata_preview, 
        inputs=[friendly_name], 
        outputs=[metadata_input]
    )

    def handle_scan(file_name):
        if not file_name:
            return "❌ No model selected for scanning."
        # Resolve the full path using MODELS_DIR
        full_path = get_full_path(file_name)
        return scan_5d_tensors(full_path)

    # --- 3. The Main Conversion Logic ---
    def start_process(file_name, model_name, formats, options):
        if not file_name or not model_name:
            yield "❌ Error: Select a file and enter a model name.", "Error"
            return

        source_path = get_full_path(file_name)
        log_acc = f"🚀 Starting conversion for: {model_name}\n"
        
        # Separate formats into Safetensors and GGUF groups
        safe_fmts = [f for f in formats if f in ["FP8", "INT8 Block-wise", "NVFP4"]]
        gguf_fmts = [f for f in formats if f.startswith("GGUF_")]

        # Process Safetensors
        if safe_fmts:
            for log, status in run_safe_conversion(MODELS_DIR, source_path, safe_fmts, model_name, options, log_acc):
                log_acc = log
                yield log_acc, status

        # Process GGUF
        if gguf_fmts:
            for log, status in run_gguf_conversion(MODELS_DIR, source_path, gguf_fmts, model_name, log_acc):
                log_acc = log
                yield log_acc, status

    # Wire the Start Button
    run_event = run_btn.click(
        fn=start_process,
        inputs=[base_dd, friendly_name, q_format, extra_flags],
        outputs=[terminal_box, pipeline_status]
    )

    # Stop Button functionality
    stop_btn.click(fn=None, cancels=[run_event])

    # --- 4. Metadata Tools ---
    read_btn.click(
        fn=read_any_metadata,
        inputs=[gr.State(MODELS_DIR), base_dd],
        outputs=[terminal_box]
    )

    scan_btn.click(
        fn=handle_scan,
        inputs=[base_dd],
        outputs=[terminal_box]
    )