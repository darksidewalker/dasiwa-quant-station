import gradio as gr
import torch, os, gc, subprocess, shutil, datetime, re, json
from config import *
from utils import get_sys_info, inject_metadata, get_metadata

# Global handle for the background process
active_process = None
ensure_dirs()

# --- INTERNAL METADATA TEMPLATE ---
# This replaces the need for recipes/metadata.json
METADATA_TEMPLATE = {
    "modelspec.title": "DaSiWa WAN 2.2 I2V 14B {model_name}",
    "modelspec.author": "Darksidewalker",
    "modelspec.description": "Multi-Expert Image-to-Video diffusion model quantized via DaSiWa Station.",
    "modelspec.date": "{date}",
    "modelspec.architecture": "wan_2.2_14b_i2v",
    "modelspec.implementation": "https://github.com/Wan-Video/Wan2.2",
    "modelspec.tags": "image-to-video, moe, diffusion, wan2.2, DaSiWa",
    "modelspec.license": "apache-2.0 and Custom License Addendum Distribution Restriction",
    "quantization.tool": "https://github.com/darksidewalker/dasiwa-wan2.2-master",
    "quantization.version": "1.0.0",
    "quantization.bits": "{bits}"
}

# --- HELPER FUNCTIONS ---

def list_files():
    """Lists available models in the models directory."""
    m = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith('.safetensors')])
    return gr.update(choices=m)

def stop_pipeline():
    """Kills any active quantization process."""
    global active_process
    if active_process:
        active_process.kill() 
        active_process = None
    torch.cuda.empty_cache()
    gc.collect()
    return "🛑 CONVERSION TERMINATED\n" + "-"*60, "Idle"

def update_metadata_preview(name):
    """Updates the JSON Code box in real-time as the user types."""
    curr_date = datetime.datetime.now().strftime("%Y-%m-%d")
    preview = {}
    for k, v in METADATA_TEMPLATE.items():
        preview[k] = v.replace("{model_name}", name).replace("{date}", curr_date).replace("{bits}", "SELECTED_QUANT")
    return json.dumps(preview, indent=4)

def handle_injection(file_name, json_str):
    """Manually injects the contents of the UI Code box into a file."""
    if not file_name: return "❌ Error: Select a model first."
    try:
        custom_meta = json.loads(json_str)
        path = os.path.join(MODELS_DIR, file_name)
        success, msg = inject_metadata(path, custom_meta)
        return f"✅ {msg}" if success else f"❌ {msg}"
    except Exception as e: return f"❌ JSON Formatting Error: {str(e)}"

def read_selected_metadata(file_name):
    """Reads the current header of a selected safetensors file."""
    if not file_name: return "❌ Error: Select a model first."
    path = os.path.join(MODELS_DIR, file_name)
    metadata, err = get_metadata(path)
    if err: return f"❌ Error: {err}"
    return f"🔍 METADATA FOR: {file_name}\n" + "-"*40 + "\n" + json.dumps(metadata, indent=4)

# --- CORE CONVERSION ENGINE ---

def run_conversion(base_model, q_formats, model_name):
    global active_process
    
    if not q_formats:
        yield "❌ ERROR: No export formats selected.", "", "Idle"
        return
    
    if not model_name or model_name.strip() == "":
        yield "❌ ERROR: Model Display Name is required for metadata injection.", "", "Idle"
        return

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_acc = f"[{timestamp}] 📦 Q-QUANT STATION ACTIVE\n" + "="*60 + "\n"
    
    source_path = os.path.join(MODELS_DIR, base_model)
    ROOT_DIR = os.getcwd()
    LLAMA_BIN = os.path.join(ROOT_DIR, "llama.cpp", "build", "bin", "llama-quantize")
    CONVERT_PY = os.path.join(ROOT_DIR, "convert.py")
    FIX_5D_PY = os.path.join(ROOT_DIR, "fix_5d_tensors.py")
    
    try:
        log_acc += f"Source Model: {base_model}\nDisplay Name: {model_name}\n\n"
        yield log_acc, "", "Processing..."

        for idx, fmt in enumerate(q_formats):
            batch_status = f"Exporting {fmt} ({idx+1}/{len(q_formats)})"
            
            # --- GGUF Q-QUANT PIPELINE ---
            if "GGUF_" in fmt:
                q_type = fmt.replace("GGUF_", "")
                final_path = source_path.replace(".safetensors", f"-{q_type}.gguf")
                bf16_gguf = source_path.replace(".safetensors", "-BF16.gguf")
                
                steps = [
                    (f"📦 Step 1: Converting to BF16 GGUF...", ["python", CONVERT_PY, "--src", source_path]),
                    (f"🔨 Step 2: Llama-Quantize to {q_type}...", [LLAMA_BIN, bf16_gguf, final_path, q_type]),
                    (f"🔧 Step 3: Applying 5D Expert Tensor Fix...", ["python", FIX_5D_PY, "--src", final_path, "--dst", final_path])
                ]
                
                for step_msg, cmd in steps:
                    log_acc += f"{step_msg}\n"
                    yield log_acc, "", batch_status
                    active_process = subprocess.Popen(cmd)
                    active_process.wait()
                    if active_process.returncode != 0:
                        log_acc += f"❌ FAILED at {step_msg}\n"
                        break
                
                if os.path.exists(bf16_gguf): os.remove(bf16_gguf)

            # --- SAFETENSORS QUANTIZATION (FP8/INT8/NVFP4) ---
            else:
                suffix = fmt.lower().replace(" ", "_").split("(")[0].strip()
                final_path = source_path.replace(".safetensors", f"_{suffix}.safetensors")
                
                cmd = ["convert_to_quant", "-i", source_path, "-o", final_path, "--wan"]
                if "int8" in fmt.lower(): cmd += ["--int8", "--block_size", "128"]
                elif "nvfp4" in fmt.lower(): cmd += ["--nvfp4"]
                
                log_acc += f"🚀 Running {fmt} export...\n"
                yield log_acc, "", batch_status
                active_process = subprocess.Popen(cmd)
                active_process.wait()

            # --- DYNAMIC METADATA INJECTION (FOR SAFETENSORS) ---
            if os.path.exists(final_path) and final_path.endswith('.safetensors'):
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                
                # Build run-specific metadata from internal template
                run_metadata = {}
                for k, v in METADATA_TEMPLATE.items():
                    run_metadata[k] = v.replace("{model_name}", model_name) \
                                       .replace("{date}", current_date) \
                                       .replace("{bits}", fmt)

                success, msg = inject_metadata(final_path, run_metadata)
                if success:
                    log_acc += f"📝 Metadata Injected: {fmt}\n"
                else:
                    log_acc += f"⚠️ Metadata Error: {msg}\n"

            log_acc += f"✅ FINISHED: {fmt}\n"
            yield log_acc, final_path, batch_status

        log_acc += "\n✨ ALL QUANTIZATIONS COMPLETE."
        yield log_acc, source_path, "Idle"

    except Exception as e:
        yield log_acc + f"\n🔥 ERROR: {str(e)}", "", "Error"
    finally:
        active_process = None

# --- UI LAYOUT ---
with gr.Blocks(title="Conversion Station") as demo:
    # Row 1: Header and Vitals
    with gr.Row():
        with gr.Column(scale=4): 
            gr.Markdown("# 📦 DaSiWa Quant Station\n**Direct GGUF & Safetensors Quantization for Wan 2.2**")
        with gr.Column(scale=3):
            vitals_box = gr.Textbox(label="Hardware Vitals", value=get_sys_info(), lines=3, interactive=False, elem_classes=["vitals-card"])
            gr.Timer(2).tick(get_sys_info, outputs=vitals_box)
        with gr.Column(scale=3):
            pipeline_status = gr.Label(label="Process State", value="Idle")

    # Row 2: The Main Workspace
    with gr.Row():
        # LEFT COLUMN (Controls)
        with gr.Column(scale=3):
            with gr.Group():
                base_dd = gr.Dropdown(label="Select Source Safetensors", allow_custom_value=True)
                friendly_name = gr.Textbox(
                    label="Model Display Name (Required)", 
                    placeholder="e.g. Cinema-Mix-V1",
                    interactive=True
                )
                refresh_btn = gr.Button("🔄 Refresh Models", size="sm")
            
            with gr.Group():
                gr.Markdown("### ⚖️ Select Formats")
                q_format = gr.CheckboxGroup(
                    choices=[
                        "FP8 (SVD)", "INT8 (Block-wise)", "NVFP4",
                        "GGUF_Q8_0", "GGUF_Q6_K", "GGUF_Q5_K_M", "GGUF_Q5_0",
                        "GGUF_Q4_K_M", "GGUF_Q4_0", "GGUF_Q3_K_M", "GGUF_Q2_K"
                    ],
                    label="Available Quants",
                    value=["FP8 (SVD)"]
                )

            with gr.Row():
                run_btn = gr.Button("🧩 START BATCH", variant="primary", elem_classes=["primary-button"])
                stop_btn = gr.Button("🛑 STOP", variant="stop", elem_classes=["stop-button"])
            
            last_path_state = gr.State("")

        # RIGHT COLUMN (Terminal + Metadata Injector stacked)
        with gr.Column(scale=7):
            terminal_box = gr.Textbox(lines=20, interactive=False, show_label=False, elem_id="terminal")
            
            with gr.Group():
                gr.Markdown("### 📝 Metadata Injector & Live Preview")
                metadata_input = gr.Code(
                    language="json",
                    label="Current Metadata Header (Live Preview)",
                    interactive=True
                )
                with gr.Row():
                    inject_btn = gr.Button("💉 Inject to Source")
                    read_btn = gr.Button("🔍 Read Current Header")

    # --- BINDINGS ---
    # Initialization
    demo.load(list_files, outputs=[base_dd])
    refresh_btn.click(list_files, outputs=[base_dd])
    
    # Real-time JSON Preview as you type the model name
    friendly_name.change(fn=update_metadata_preview, inputs=[friendly_name], outputs=[metadata_input])
    
    # Main Conversion Run
    run_btn.click(
        fn=run_conversion,
        inputs=[base_dd, q_format, friendly_name],
        outputs=[terminal_box, last_path_state, pipeline_status]
    )
    
    # Process Control
    stop_btn.click(fn=stop_pipeline, outputs=[terminal_box, pipeline_status])
    
    # Metadata Tools
    inject_btn.click(
        fn=handle_injection,
        inputs=[base_dd, metadata_input],
        outputs=[terminal_box]
    )
    
    read_btn.click(
        fn=read_selected_metadata,
        inputs=[base_dd],
        outputs=[terminal_box]
    )

    # Auto-scroll terminal
    terminal_box.change(fn=None, js=JS_AUTO_SCROLL, inputs=[terminal_box])

if __name__ == "__main__":
    demo.launch(css=CSS_STYLE)