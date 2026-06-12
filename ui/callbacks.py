# ui/callbacks.py
import gradio as gr
from config import MODELS_DIR
import json
from core.safetensors_engine import run_safe_conversion
from core.gguf_engine import run_gguf_conversion
from core.metadata_manager import (
    update_metadata_preview, 
    read_any_metadata, 
    inject_metadata, 
    calculate_sha256
)
from utils.file_listing import get_model_list
from utils.scanner_5d import scan_5d_tensors
from utils.pattern_audit import audit_patterns
from utils.keeplist_compare import compare_to_reference
from utils.exact_config import write_exact_config
from utils.arch_detector import inspect_checkpoint
import os

def _resolve_model_path(m_path):
    if isinstance(m_path, list):
        if not m_path:
            return ""
        m_path = m_path[0]
    if isinstance(m_path, dict):
        return m_path.get("name") or m_path.get("tmp_path") or ""
    # Normalize to a filesystem path string
    s = str(m_path) if m_path is not None else ""
    s = os.path.expanduser(s)
    # If the component returned a file path, use its containing directory
    try:
        if os.path.isfile(s):
            return os.path.dirname(s)
        return s
    except Exception:
        return s


def setup_callbacks(models_dir_dd, base_dd, friendly_name, refresh_btn, run_btn, stop_btn, 
                   q_format, pipeline_status, extra_flags, terminal_box, 
                   metadata_input, inject_btn, read_btn, scan_btn,
                   model_type, optimizer_choice, low_vram,
                   audit_btn, reference_dd, compare_btn,
                   build_exact_btn, clear_exact_btn,
                   full_checkpoint):
    
    # --- 1. MODEL LIST MANAGEMENT ---
    def refresh_both(m_path):
        # m_path is the selected directory string from the dropdown
        m_path = _resolve_model_path(m_path)
        update = get_model_list(m_path)
        return gr.update(choices=update), gr.update(choices=update)
    refresh_btn.click(fn=refresh_both, inputs=[models_dir_dd], outputs=[base_dd, reference_dd])
    models_dir_dd.change(fn=refresh_both, inputs=[models_dir_dd], outputs=[base_dd, reference_dd])

    def handle_source_selection(m_path, file_name, current_arch, model_name):
        """Inspect selected safetensors header and prefill safe defaults."""
        no_change = gr.update()
        if not file_name:
            return no_change, no_change, "", no_change
        if not str(file_name).endswith(".safetensors"):
            return no_change, no_change, "Source inspection skipped: select a .safetensors file.", no_change

        m_path = _resolve_model_path(m_path)
        m_path = os.path.realpath(os.path.expanduser(m_path))
        source_path = os.path.realpath(os.path.join(m_path, file_name))
        if not os.path.isfile(source_path):
            return no_change, no_change, f"Source inspection failed: file not found:\n{source_path}", no_change

        try:
            detected_arch, is_full, inspect_log = inspect_checkpoint(source_path)
        except Exception as e:
            return no_change, no_change, f"Source inspection failed: {e}", no_change

        arch_update = no_change
        arch_line = f"Architecture: {detected_arch}"
        if detected_arch in ("WAN 2.2", "LTX-2.3"):
            arch_update = gr.update(value=detected_arch)
            arch_line += " (auto-selected)"
            effective_arch = detected_arch
        elif detected_arch == "UNKNOWN":
            arch_line += f" (kept current selection: {current_arch})"
            effective_arch = current_arch
        else:
            arch_line += f" (ambiguous; kept current selection: {current_arch})"
            effective_arch = current_arch

        full_update = gr.update(value=is_full)
        full_line = "Full checkpoint: yes" if is_full else "Full checkpoint: no"
        display_name = model_name or "TreasureChest"
        metadata_json = update_metadata_preview(display_name, effective_arch, is_full=is_full)

        msg = [
            f"Source inspected: {file_name}",
            arch_line,
            full_line,
            "",
            *inspect_log,
            "",
            "You can still override Architecture or Full Checkpoint before starting.",
        ]
        return arch_update, full_update, "\n".join(msg), metadata_json

    base_dd.change(
        fn=handle_source_selection,
        inputs=[models_dir_dd, base_dd, model_type, friendly_name],
        outputs=[model_type, full_checkpoint, terminal_box, metadata_input],
    )

    # --- 2. THE MAIN CONVERSION LOGIC ---
    # This function is triggered by START BATCH. 
    def start_process(m_path, file_name, model_name, formats, options, m_type, opt_choice, lv, is_full):
        if not file_name or not model_name:
            yield "❌ Error: Select a source file and enter a model name.", "Error"
            return

        m_path = _resolve_model_path(m_path)
        m_path = os.path.realpath(os.path.expanduser(m_path))
        source_path = os.path.realpath(os.path.join(m_path, file_name))
        if not os.path.isfile(source_path):
            yield f"❌ Error: Selected source file does not exist: {source_path}\n", "Error"
            return

        if not formats:
            yield "❌ Error: No target format selected. Pick at least one target format before starting.", "Error"
            return

        log_acc = f"🚀 Initializing Pipeline for: {model_name}\n"
        log_acc += f"📦 Target Architecture: {m_type}\n"
        log_acc += f"📦 Full Checkpoint: {'Yes' if is_full else 'No'}\n"
        log_acc += "-"*40 + "\n"
        
        # Filter selected formats
        safe_fmts = [f for f in formats if f in ["FP8", "INT8 Row-wise ConvRot", "NVFP4"]]
        gguf_fmts = [f for f in formats if f.startswith("GGUF_")]

        # Execute Safetensors Quantization
        if safe_fmts:
            for log, status in run_safe_conversion(
                m_path, source_path, safe_fmts, model_name, 
                m_type, opt_choice, options, log_acc,
                low_vram=lv,
                is_full_checkpoint=is_full
            ):
                log_acc = log
                yield log_acc, status

        # Execute GGUF Quantization
        if gguf_fmts:
            for log, status in run_gguf_conversion(
                m_path, source_path, gguf_fmts, model_name, log_acc,
                model_type=m_type,
                is_full=is_full
            ):
                log_acc = log
                yield log_acc, status

    # Wire the START button to the process
    run_event = run_btn.click(
        fn=start_process,
        inputs=[
            models_dir_dd,     # 0
            base_dd,           # 1
            friendly_name,     # 2
            q_format,          # 3
            extra_flags,       # 4
            model_type,        # 5
            optimizer_choice,  # 6
            low_vram,          # 7
            full_checkpoint    # 8
        ],
        outputs=[terminal_box, pipeline_status]
    )

    # Wire the STOP button to cancel the running thread
    stop_btn.click(fn=None, cancels=[run_event])

    # --- 3. METADATA TOOLS & UTILITIES ---

    def handle_metadata_injection(m_path, file_name, manual_json_str):
        """Manually injects metadata into a selected source file."""
        if not file_name:
            return "❌ No file selected."
        
        m_path = _resolve_model_path(m_path)
        m_path = os.path.expanduser(m_path)
        full_path = os.path.join(m_path, file_name)
        
        try:
            # We parse the JSON currently visible in the UI box
            meta_dict = json.loads(manual_json_str)
            meta_dict["modelspec.hash_sha256"] = calculate_sha256(full_path)
            
            success, msg = inject_metadata(full_path, meta_dict)
            return f"✅ {msg}" if success else f"❌ {msg}"
        except Exception as e:
            return f"🔥 Injection Error: {str(e)}"

    def handle_scan(m_path, file_name):
        """Triggers the 5D Tensor Scanner for WAN models."""
        if not file_name:
            return "❌ No model selected for scanning."
        m_path = _resolve_model_path(m_path)
        m_path = os.path.expanduser(m_path)
        full_path = os.path.join(m_path, file_name)
        return scan_5d_tensors(full_path)

    def handle_audit(m_path, file_name, m_type):
        """Audits the selected model against the layer-config patterns."""
        if not file_name:
            return "❌ No model selected for audit."
        if not file_name.endswith(".safetensors"):
            return "❌ Pattern audit only works on .safetensors source files."
        m_path = _resolve_model_path(m_path)
        m_path = os.path.expanduser(m_path)
        full_path = os.path.join(m_path, file_name)
        return audit_patterns(full_path, m_type)

    def handle_compare(m_path, reference_name, m_type):
        """Compares our pattern decisions against an author's reference FP8."""
        if not reference_name:
            return "❌ No reference file selected. Place author's FP8 in models/ and refresh."
        if not reference_name.endswith(".safetensors"):
            return "❌ Reference must be a .safetensors file (the author's quantized FP8)."
        m_path = _resolve_model_path(m_path)
        m_path = os.path.expanduser(m_path)
        full_path = os.path.join(m_path, reference_name)
        return compare_to_reference(full_path, m_type)

    def handle_build_exact(m_path, source_name, reference_name, formats):
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

        eligible = [f for f in formats if f in ("FP8", "NVFP4", "INT8 Row-wise ConvRot")]
        if not eligible:
            return ("❌ No layer-config-eligible format selected. "
                    "Tick FP8, NVFP4, or INT8 Row-wise ConvRot in Target Formats first.")

        m_path = _resolve_model_path(m_path)
        m_path = os.path.expanduser(m_path)
        reference_path = os.path.join(m_path, reference_name)
        source_path = os.path.join(m_path, source_name)
        filters_dir = _os.path.join(os.path.dirname(m_path), "filters")

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

    def handle_clear_exact(m_path):
        """Removes all _exact_*.json files in filters/ so the next run reverts
        to regex-based auto config."""
        import os as _os
        m_path = _resolve_model_path(m_path)
        m_path = os.path.expanduser(m_path)
        filters_dir = _os.path.join(_os.path.dirname(m_path), "filters")
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
        inputs=[models_dir_dd, base_dd, metadata_input],
        outputs=[terminal_box]
    )

    read_btn.click(
        fn=read_any_metadata, 
        inputs=[models_dir_dd, base_dd], 
        outputs=[terminal_box]
    )
    
    scan_btn.click(
        fn=handle_scan, 
        inputs=[models_dir_dd, base_dd], 
        outputs=[terminal_box]
    )

    audit_btn.click(
        fn=handle_audit,
        inputs=[models_dir_dd, base_dd, model_type],
        outputs=[terminal_box]
    )

    compare_btn.click(
        fn=handle_compare,
        inputs=[models_dir_dd, reference_dd, model_type],
        outputs=[terminal_box]
    )

    build_exact_btn.click(
        fn=handle_build_exact,
        inputs=[models_dir_dd, base_dd, reference_dd, q_format],
        outputs=[terminal_box]
    )

    clear_exact_btn.click(
        fn=handle_clear_exact,
        inputs=[models_dir_dd],
        outputs=[terminal_box]
    )
