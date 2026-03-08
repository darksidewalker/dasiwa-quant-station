import gradio as gr
import torch, os, gc, subprocess, shutil, datetime, re, json
from config import *
from utils import get_sys_info, inject_metadata, get_metadata

# Global handle for the background process
active_process = None
ensure_dirs()

# --- INTERNAL METADATA TEMPLATE ---
METADATA_TEMPLATE = {
    "modelspec.title": "DaSiWa WAN 2.2 I2V {model_name}",
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

# --- BACKEND 1: GGUF ENGINE (Mirroring Bash Script) ---

def run_gguf_conversion(source_path, formats, model_name, log_acc):
    global active_process
    ROOT_DIR = os.getcwd()
    LLAMA_BIN = os.path.join(ROOT_DIR, "llama.cpp", "build", "bin", "llama-quantize")
    CONVERT_PY = "convert.py"
    FIX_PY = "fix_5d_tensors.py"
    
    base_name = os.path.splitext(os.path.basename(source_path))[0]
    master_gguf = os.path.join(MODELS_DIR, f"{base_name}.gguf")

    if os.path.exists(master_gguf):
        log_acc += f"ℹ️ Base GGUF exists: {os.path.basename(master_gguf)} — skipping convert.\n"
    else:
        log_acc += f"📦 Converting {os.path.basename(source_path)} -> {os.path.basename(master_gguf)}\n"
        yield log_acc, "GGUF Base Prep"
        active_process = subprocess.Popen(["python", CONVERT_PY, "--src", source_path, "--dst", master_gguf])
        active_process.wait()

    q_map = {
        "GGUF_Q8_0": "Q8_0", "GGUF_Q6_K": "Q6_K", "GGUF_Q5_K_M": "Q5_K_M",
        "GGUF_Q4_K_M": "Q4_K_M", "GGUF_Q3_K_S": "Q3_K_S", "GGUF_Q2_K": "Q2_K"
    }

    for fmt in formats:
        q_flag = q_map.get(fmt, "Q8_0")
        out_q = os.path.join(MODELS_DIR, f"{base_name}_{q_flag}.gguf")
        out_qf = os.path.join(MODELS_DIR, f"{base_name}_{q_flag}-fix.gguf")

        if os.path.exists(out_qf):
            log_acc += f"ℹ️ Fixed file exists: {os.path.basename(out_qf)} — skipping.\n"
            continue

        log_acc += f"🔨 Quantizing {q_flag}...\n"
        yield log_acc, f"Quantizing {q_flag}"
        active_process = subprocess.Popen([LLAMA_BIN, master_gguf, out_q, q_flag])
        active_process.wait()

        log_acc += f"🔧 Fixing 5D tensors -> {os.path.basename(out_qf)}\n"
        yield log_acc, f"Fixing {q_flag}"
        active_process = subprocess.Popen(["python", FIX_PY, "--src", out_q, "--dst", out_qf])
        active_process.wait()

        if os.path.exists(out_qf) and os.path.exists(out_q):
            os.remove(out_q)
            # Metadata Injection using gguf library
            success, msg = write_gguf_meta(out_qf, model_name, fmt)
            log_acc += f"📝 GGUF Meta: {msg}\n✅ Finished {q_flag}\n"
            
    return log_acc

# --- BACKEND 2: SAFETENSORS ENGINE ---

def run_safe_conversion(source_path, formats, model_name, options, log_acc):
    global active_process
    FLAG_MAP = {
        "Standard FP8 (ComfyUI)": ["--comfy_quant"],
        "INT8 Block-wise": ["--int8", "--block_size", "128", "--comfy_quant"],
        "Blackwell NVFP4": ["--nvfp4", "--comfy_quant"],
        "Auto-Quality (Heur)": ["--heur"]
    }

    ULTRA_PRESET = [
        "--comfy_quant", "--save-quant-metadata", "--wan", 
        "--lr_schedule", "plateau", "--lr_patience", "2", "--lr_factor", "0.96", 
        "--lr_min", "9e-9", "--lr_cooldown", "0", "--lr_threshold", "1e-11", 
        "--num_iter", "9000", "--calib_samples", "45000", 
        "--lr", "9.916700000002915715e-3", "--top_p", "0.05", 
        "--min_k", "64", "--max_k", "256", "--early-stop-stall", "20000", 
        "--early-stop-lr", "1e-8", "--early-stop-loss", "9e-8", "--lr-shape-influence", "3.5"
    ]

    for fmt in formats:
        suffix = fmt.lower().replace(" ", "_").split("(")[0].strip()
        final_path = source_path.replace(".safetensors", f"_{suffix}.safetensors")
        cmd = ["convert_to_quant", "-i", source_path, "-o", final_path]

        if "Ultra-Quality (Optimizer)" in options:
            cmd.extend(ULTRA_PRESET)
        else:
            if fmt in FLAG_MAP: cmd.extend(FLAG_MAP[fmt])
            for opt in options:
                if opt in FLAG_MAP: cmd.extend(FLAG_MAP[opt])
            if "wan" in source_path.lower(): cmd.append("--wan")

        yield log_acc, f"Safetensors: {fmt}"
        active_process = subprocess.Popen(cmd)
        active_process.wait()

        if os.path.exists(final_path) and final_path.endswith('.safetensors'):
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            meta = {k: v.replace("{model_name}", model_name).replace("{date}", current_date).replace("{bits}", fmt) for k, v in METADATA_TEMPLATE.items()}
            success, msg = inject_metadata(final_path, meta)
            log_acc += f"📝 Metadata: {msg}\n"
            
    return log_acc

# --- READ METADATA (Safetensors + GGUF) ---

def read_selected_metadata(file_name):
    if not file_name: return "❌ Select a model first."
    path = os.path.join(MODELS_DIR, file_name)
    
    if file_name.endswith('.gguf'):
        if not gguf: return "❌ gguf package not installed. Cannot read KV pairs."
        try:
            reader = gguf.GGUFReader(path)
            kv_data = "🔍 GGUF KV PAIRS:\n" + "-"*40 + "\n"
            for key in reader.fields:
                val = reader.fields[key]
                kv_data += f"{key}: {val.parts[val.data[0]] if val.data else 'None'}\n"
            return kv_data
        except Exception as e: return f"🔥 GGUF Read Error: {str(e)}"

    meta, err = get_metadata(path)
    if err: return f"❌ Error: {err}"
    return f"🔍 SAFETENSORS METADATA:\n" + "-"*40 + "\n" + json.dumps(meta, indent=4)

# --- TRAFFIC CONTROLLER ---

def run_conversion(base_model, q_formats, model_name, extra_options):
    if not q_formats: yield "❌ No formats.", "", "Idle"; return
    log_acc = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 🚀 BATCH START\n" + "="*60 + "\n"
    source_path = os.path.join(MODELS_DIR, base_model)
    gguf_list = [f for f in q_formats if "GGUF" in f]
    safe_list = [f for f in q_formats if "GGUF" not in f]

    if gguf_list:
        for log, status in run_gguf_conversion(source_path, gguf_list, model_name, log_acc):
            log_acc = log; yield log_acc, "", status
    if safe_list:
        for log, status in run_safe_conversion(source_path, safe_list, model_name, extra_options, log_acc):
            log_acc = log; yield log_acc, "", status
    yield log_acc + "\n✨ DONE.", source_path, "Idle"

# --- UI LAYOUT ---

with gr.Blocks(title="Conversion Station") as demo:
    # Row 1: App Header
    with gr.Row():
        gr.Markdown("# 📦 DaSiWa Quant Station\n**Direct GGUF & Safetensors Quantization for Wan 2.2**")

    # Row 2: Control Panel (3-Column Grid)
    with gr.Row():
        # Column 1: Source Files
        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("### 📥 Source Settings")
                base_dd = gr.Dropdown(label="Select Source Safetensors", allow_custom_value=True)
                friendly_name = gr.Textbox(label="Model Display Name", placeholder="e.g. Cinema-Mix-V1")
                refresh_btn = gr.Button("🔄 Refresh Models", size="sm")
            
            with gr.Row():
                run_btn = gr.Button("🧩 START BATCH", variant="primary", elem_classes=["primary-button"])
                stop_btn = gr.Button("🛑 STOP", variant="stop", elem_classes=["stop-button"])
            
            last_path_state = gr.State("")

        # Column 2: Quantization Formats (Full List)
        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("### ⚖️ Select Formats")
                q_format = gr.CheckboxGroup(
                    choices=[
                        "FP8", "INT8 Block-wise", "NVFP4",
                        "GGUF_Q8_0", "GGUF_Q6_K", "GGUF_Q5_K_M", "GGUF_Q5_0",
                        "GGUF_Q4_K_M", "GGUF_Q4_0", "GGUF_Q3_K_M", "GGUF_Q2_K"
                    ],
                    label="Target Format",
                    info="Choose precision level. GGUF requires a multi-step build.",
                    value=["FP8"]
                )

        # Column 3: Optimizations & Vitals
        with gr.Column(scale=4):
            with gr.Group():
                gr.Markdown("### 🛠️ Optimization Flags")
                extra_flags = gr.CheckboxGroup(
                    choices=["Ultra-Quality (Optimizer)", "Auto-Quality (Heur)", "Fast Mode (Simple)", "Low Memory Mode"],
                    label="Quantization Tweaks",
                    info="For FP-Quants - Ultra: 9k iterations (Pro Quality). Heur: Fixes black screens.",
                    value=["Auto-Quality (Heur)"]
                )
            
            vitals_box = gr.Textbox(label="Hardware Vitals", value=get_sys_info(), lines=2, interactive=False, elem_classes=["vitals-card"])
            gr.Timer(2).tick(get_sys_info, outputs=vitals_box)
            pipeline_status = gr.Label(label="Process State", value="Idle")

    # Row 3: Output Terminal & Metadata Tools
    with gr.Row():
        with gr.Column(scale=6):
            terminal_box = gr.Textbox(lines=22, interactive=False, show_label=False, elem_id="terminal")
        
        with gr.Column(scale=4):
            with gr.Group():
                gr.Markdown("### 📝 Metadata Injector")
                metadata_input = gr.Code(
                    value=update_metadata_preview("Enter Name..."), 
                    language="json",
                    label="Header Preview",
                    interactive=True
                )
                with gr.Row():
                    inject_btn = gr.Button("💉 Inject to Source")
                    read_btn = gr.Button("🔍 Read Current Header")

    # --- BINDINGS ---
    demo.load(list_files, outputs=[base_dd])
    friendly_name.change(fn=update_metadata_preview, inputs=[friendly_name], outputs=[metadata_input])
    
    run_btn.click(
        fn=run_conversion,
        inputs=[base_dd, q_format, friendly_name, extra_flags],
        outputs=[terminal_box, last_path_state, pipeline_status]
    )
    
    stop_btn.click(fn=stop_pipeline, outputs=[terminal_box, pipeline_status])
    refresh_btn.click(list_files, outputs=[base_dd])
    inject_btn.click(fn=handle_injection, inputs=[base_dd, metadata_input], outputs=[terminal_box])
    read_btn.click(fn=read_selected_metadata, inputs=[base_dd], outputs=[terminal_box])

    # Auto-scroll terminal
    terminal_box.change(fn=None, js=JS_AUTO_SCROLL, inputs=[terminal_box])

if __name__ == "__main__":
    demo.launch(css=CSS_STYLE)