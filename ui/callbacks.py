# ui/callbacks.py
import gradio as gr
import json
from core.safetensors_engine import run_safe_conversion
from core.gguf_engine import run_gguf_conversion
from core.metadata_manager import (
    update_metadata_preview, 
    read_any_metadata, 
    inject_metadata, 
    calculate_sha256
)
from utils.file_ops import list_files, get_full_path
from config import MODELS_DIR
from utils.scanner_5d import scan_5d_tensors
import os

def setup_callbacks(base_dd, friendly_name, refresh_btn, run_btn, stop_btn, 
                   q_format, pipeline_status, extra_flags, terminal_box, 
                   metadata_input, inject_btn, read_btn, scan_btn,
                   model_type, optimizer_choice, low_vram):
    
    # --- 1. MODEL LIST MANAGEMENT ---
    refresh_btn.click(fn=list_files, outputs=[base_dd])

    # --- 2. THE MAIN CONVERSION LOGIC ---
    # This function is triggered by START BATCH. 
    # It must accept exactly 8 arguments to match the UI inputs.
    def start_process(file_name, model_name, formats, options, m_type, opt_choice, lv):
        if not file_name or not model_name:
            yield "❌ Error: Select a source file and enter a model name.", "Error"
            return

        source_path = get_full_path(file_name)
        log_acc = f"🚀 Initializing Pipeline for: {model_name}\n"
        log_acc += f"📦 Target Architecture: {m_type}\n"
        log_acc += "-"*40 + "\n"
        
        # Filter selected formats
        safe_fmts = [f for f in formats if f in ["FP8", "INT8 Block-wise", "NVFP4"]]
        gguf_fmts = [f for f in formats if f.startswith("GGUF_")]

        # Execute Safetensors Quantization
        if safe_fmts:
            # Note: Ensure run_safe_conversion doesn't expect 'ac' either!
            for log, status in run_safe_conversion(
                MODELS_DIR, source_path, safe_fmts, model_name, 
                m_type, opt_choice, options, log_acc,
                low_vram=lv
            ):
                log_acc = log
                yield log_acc, status

        # Execute GGUF Quantization
        if gguf_fmts:
            for log, status in run_gguf_conversion(
                MODELS_DIR, source_path, gguf_fmts, model_name, log_acc
            ):
                log_acc = log
                yield log_acc, status

    # Wire the START button to the process
    run_event = run_btn.click(
        fn=start_process,
        inputs=[
            base_dd,           # 1
            friendly_name,     # 2
            q_format,          # 3
            extra_flags,       # 4
            model_type,        # 5
            optimizer_choice,  # 6
            low_vram           # 7
        ],
        outputs=[terminal_box, pipeline_status]
    )

    # Wire the STOP button to cancel the running thread
    stop_btn.click(fn=None, cancels=[run_event])

    # --- 3. METADATA TOOLS & UTILITIES ---

    def handle_metadata_injection(file_name, manual_json_str):
        """Manually injects metadata into a selected source file."""
        if not file_name:
            return "❌ No file selected."
        
        full_path = get_full_path(file_name)
        
        try:
            # We parse the JSON currently visible in the UI box
            meta_dict = json.loads(manual_json_str)

            from core.metadata_manager import calculate_sha256
            meta_dict["modelspec.hash_sha256"] = calculate_sha256(full_path)
            
            success, msg = inject_metadata(full_path, meta_dict)
            return f"✅ {msg}" if success else f"❌ {msg}"
        except Exception as e:
            return f"🔥 Injection Error: {str(e)}"

    def handle_scan(file_name):
        """Triggers the 5D Tensor Scanner for WAN models."""
        if not file_name:
            return "❌ No model selected for scanning."
        full_path = get_full_path(file_name)
        return scan_5d_tensors(full_path)

    # Metadata Action Buttons
    inject_btn.click(
        fn=handle_metadata_injection,
        inputs=[base_dd, metadata_input],
        outputs=[terminal_box]
    )

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

    # --- 4. DYNAMIC UI REFRESH ---
    # Update the metadata preview automatically when the name or architecture changes
    def update_json_on_ui_change(name, architecture):
        return update_metadata_preview(name, architecture)

    # These triggers ensure the JSON editor reflects your LTX-2 or WAN choices instantly
    model_type.change(
        fn=update_json_on_ui_change,
        inputs=[friendly_name, model_type],
        outputs=[metadata_input]
    )
    
    friendly_name.change(
        fn=update_json_on_ui_change,
        inputs=[friendly_name, model_type],
        outputs=[metadata_input]
    )