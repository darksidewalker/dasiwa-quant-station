import gradio as gr
import torch, os, gc, subprocess, shutil, datetime, re, json
from config import *
from utils import get_sys_info

# Global handle for the background process
active_process = None
ensure_dirs()

def list_files():
    # Only show safetensors as source for conversion
    m = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith('.safetensors')])
    return gr.update(choices=m)

def stop_pipeline():
    global active_process
    if active_process:
        active_process.kill() 
        active_process = None
    torch.cuda.empty_cache()
    gc.collect()
    return "🛑 CONVERSION TERMINATED\n" + "-"*60, "Idle"

def run_conversion(base_model, q_formats, model_name):
    global active_process
    
    # Validation: Ensure a name is provided
    if not q_formats:
        yield "❌ ERROR: No export formats selected.", "", "Idle"
        return
    
    if not model_name or model_name.strip() == "":
        yield "❌ ERROR: Model Display Name is required for metadata injection.", "", "Idle"
        return

    # Load the JSON recipe template
    try:
        with open("recipes/metadata.json", "r") as f:
            meta_template = f.read()
    except Exception as e:
        yield f"❌ ERROR: Could not find recipes/metadata.json: {str(e)}", "", "Idle"
        return

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_acc = f"[{timestamp}] 📦 Q-QUANT STATION ACTIVE\n" + "="*60 + "\n"
    
    source_path = os.path.join(MODELS_DIR, base_model)
    ROOT_DIR = os.getcwd()
    LLAMA_BIN = os.path.join(ROOT_DIR, "llama.cpp", "build", "bin", "llama-quantize")
    CONVERT_PY = os.path.join(ROOT_DIR, "convert.py")
    FIX_5D_PY = os.path.join(ROOT_DIR, "fix_5d_tensors.py")
    
    try:
        log_acc += f"Source Model: {base_model}\n"
        log_acc += f"Display Name: {model_name}\n\n"
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

            # --- METADATA INJECTION (Post-Quantization) ---
            if os.path.exists(final_path) and final_path.endswith('.safetensors'):
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                try:
                    # Replace placeholders with current run values
                    formatted_json = meta_template.replace("{model_name}", model_name).replace("{date}", current_date).replace("{bits}", fmt)
                    meta_dict = json.loads(formatted_json)
                    
                    # Call the utility to rewrite header
                    success, msg = inject_metadata(final_path, meta_dict)
                    if success:
                        log_acc += f"📝 Metadata Injected: {fmt}\n"
                    else:
                        log_acc += f"⚠️ Metadata Error: {msg}\n"
                except Exception as e:
                    log_acc += f"⚠️ JSON Processing Error: {str(e)}\n"

            log_acc += f"✅ FINISHED: {fmt}\n"
            yield log_acc, final_path, batch_status

        log_acc += "\n✨ ALL QUANTIZATIONS COMPLETE."
        yield log_acc, source_path, "Idle"

    except Exception as e:
        yield log_acc + f"\n🔥 ERROR: {str(e)}", "", "Error"
    finally:
        active_process = None

def handle_injection(file_name, json_str):
    if not file_name:
        return "❌ Error: No file selected."
    try:
        meta = json.loads(json_str)
        path = os.path.join(MODELS_DIR, file_name)
        success, msg = inject_metadata(path, meta)
        return f"✅ {msg}" if success else f"❌ {msg}"
    except Exception as e:
        return f"❌ JSON Error: {str(e)}"

# --- UI LAYOUT ---
with gr.Blocks(title="Conversion Station", css=CSS_STYLE) as demo:
    with gr.Row():
        with gr.Column(scale=4): 
            gr.Markdown("# 📦 DaSiWa Quant Station\n**Direct GGUF & Safetensors Quantization**")
        with gr.Column(scale=3):
            vitals_box = gr.Textbox(label="Hardware Vitals", value=get_sys_info(), lines=3, interactive=False, elem_classes=["vitals-card"])
            gr.Timer(2).tick(get_sys_info, outputs=vitals_box)
        with gr.Column(scale=3):
            pipeline_status = gr.Label(label="Process State", value="Idle")

    with gr.Row():
        with gr.Column(scale=3):
            base_dd = gr.Dropdown(label="Select Source Safetensors", allow_custom_value=True)
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
            with gr.Column(scale=3):
                base_dd = gr.Dropdown(label="Select Source Safetensors", allow_custom_value=True)
                # NEW: Mandatory Model Name field
                friendly_name = gr.Textbox(label="Model Display Name (Required)", placeholder="e.g. Cinema-Mix-V1")

            with gr.Group():
                gr.Markdown("### 📝 Metadata Injector")
                metadata_input = gr.Code(
                    value='{"author": "Darksidewalker", "model_type": "Wan 2.2 14B I2V"}',
                    language="json",
                    label="Metadata JSON"
                )
                inject_btn = gr.Button("💉 Inject Metadata to Source")
            
            with gr.Row():
                run_btn = gr.Button("🧩 START BATCH", variant="primary", elem_classes=["primary-button"])
                stop_btn = gr.Button("🛑 STOP", variant="stop", elem_classes=["stop-button"])
            
            last_path_state = gr.State("")

        with gr.Column(scale=7):
            terminal_box = gr.Textbox(lines=26, interactive=False, show_label=False, elem_id="terminal")

    # --- BINDINGS ---
    demo.load(list_files, outputs=[base_dd])
    refresh_btn.click(list_files, outputs=[base_dd])
    
    run_btn.click(
        fn=run_conversion,
        inputs=[base_dd, q_format],
        outputs=[terminal_box, last_path_state, pipeline_status]
    )
    stop_btn.click(fn=stop_pipeline, outputs=[terminal_box, pipeline_status])
    terminal_box.change(fn=None, js=JS_AUTO_SCROLL, inputs=[terminal_box])

    inject_btn.click(
    fn=handle_injection,
    inputs=[base_dd, metadata_input],
    outputs=[terminal_box]
    )

if __name__ == "__main__":
    demo.launch()