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
from utils.pattern_audit import audit_patterns
from utils.keeplist_compare import compare_to_reference
from utils.exact_config import write_exact_config
import os

def setup_callbacks(base_dd, friendly_name, refresh_btn, run_btn, stop_btn, 
                   q_format, pipeline_status, extra_flags, terminal_box, 
                   metadata_input, inject_btn, read_btn, scan_btn,
                   model_type, optimizer_choice, low_vram, auto_layer_config,
                   audit_btn, reference_dd, compare_btn,
                   build_exact_btn, clear_exact_btn):
    
    # --- 1. MODEL LIST MANAGEMENT ---
    def refresh_both():
        update = list_files()
        return update, update
    refresh_btn.click(fn=refresh_both, outputs=[base_dd, reference_dd])

    # --- 2. THE MAIN CONVERSION LOGIC ---
    # This function is triggered by START BATCH. 
    def start_process(file_name, model_name, formats, options, m_type, opt_choice, lv, auto_lc):
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
            for log, status in run_safe_conversion(
                MODELS_DIR, source_path, safe_fmts, model_name, 
                m_type, opt_choice, options, log_acc,
                low_vram=lv,
                auto_layer_config=auto_lc
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
            low_vram,          # 7
            auto_layer_config  # 8
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

    def handle_audit(file_name, m_type):
        """Audits the selected model against the layer-config patterns."""
        if not file_name:
            return "❌ No model selected for audit."
        if not file_name.endswith(".safetensors"):
            return "❌ Pattern audit only works on .safetensors source files."
        full_path = get_full_path(file_name)
        return audit_patterns(full_path, m_type)

    def handle_compare(reference_name, m_type):
        """Compares our pattern decisions against an author's reference FP8."""
        if not reference_name:
            return "❌ No reference file selected. Place author's FP8 in models/ and refresh."
        if not reference_name.endswith(".safetensors"):
            return "❌ Reference must be a .safetensors file (the author's quantized FP8)."
        full_path = get_full_path(reference_name)
        return compare_to_reference(full_path, m_type)

    def handle_build_exact(source_name, reference_name, formats):
        """
        Builds an exact-mode layer config from the reference FP8 for each
        eligible format the user has selected. Once written, the next
        quantization run picks these up automatically (engine prefers
        _exact_*.json over auto regex configs).
        """
        import os as _os
        from utils.exact_config import _read_dtype_map

        if not reference_name:
            return "❌ Pick a Reference FP8 first."
        if not source_name:
            return "❌ Pick the source file too (for scope filtering)."
        if not reference_name.endswith(".safetensors"):
            return "❌ Reference must be a .safetensors file."

        eligible = [f for f in formats if f in ("FP8", "NVFP4", "INT8 Block-wise")]
        if not eligible:
            return ("❌ No layer-config-eligible format selected. "
                    "Tick FP8, NVFP4, or INT8 Block-wise in Target Formats first.")

        reference_path = get_full_path(reference_name)
        source_path = get_full_path(source_name)
        filters_dir = _os.path.join(_os.path.dirname(MODELS_DIR), "filters")

        # Read source keys for scope filtering
        try:
            source_dtypes = _read_dtype_map(source_path)
            source_keys = set(source_dtypes.keys())
        except Exception as e:
            return f"🔥 Could not read source file: {e}"

        lines = [f"🎯 Building exact configs from: {reference_name}"]
        lines.append(f"   Source scope: {source_name} ({len(source_keys)} tensors)")
        lines.append("")
        for fmt in eligible:
            cfg_path, build_log = write_exact_config(
                reference_path, fmt, filters_dir, source_keys=source_keys
            )
            lines.append(f"--- {fmt} ---")
            lines.extend(build_log)
            lines.append("")
        lines.append("Done. The engine will use these configs on the next")
        lines.append("quantization run (output filenames will have _exact suffix).")
        lines.append("Click 'Clear Exact Config' to revert to regex auto mode.")
        return "\n".join(lines)

    def handle_clear_exact():
        """Removes all _exact_*.json files in filters/ so the next run reverts
        to regex-based auto config."""
        import os as _os
        filters_dir = _os.path.join(_os.path.dirname(MODELS_DIR), "filters")
        if not _os.path.isdir(filters_dir):
            return "ℹ️  No filters/ directory exists yet. Nothing to clear."
        removed = []
        for fn in _os.listdir(filters_dir):
            if fn.startswith("_exact_") and fn.endswith(".json"):
                try:
                    _os.remove(_os.path.join(filters_dir, fn))
                    removed.append(fn)
                except Exception as e:
                    return f"🔥 Failed to remove {fn}: {e}"
        if not removed:
            return "ℹ️  No exact configs found. Already in regex auto mode."
        return ("✅ Cleared exact configs:\n" +
                "\n".join(f"   {f}" for f in removed) +
                "\nNext run will use regex-based auto config.")

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

    audit_btn.click(
        fn=handle_audit,
        inputs=[base_dd, model_type],
        outputs=[terminal_box]
    )

    compare_btn.click(
        fn=handle_compare,
        inputs=[reference_dd, model_type],
        outputs=[terminal_box]
    )

    build_exact_btn.click(
        fn=handle_build_exact,
        inputs=[base_dd, reference_dd, q_format],
        outputs=[terminal_box]
    )

    clear_exact_btn.click(
        fn=handle_clear_exact,
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