# ui/layout.py
import gradio as gr
from ui.assets import CSS_STYLE
from core.metadata_manager import update_metadata_preview
from utils.system import get_sys_info
from ui.callbacks import setup_callbacks

def create_ui():
    with gr.Blocks(title="DaSiWa Quant Station") as demo:
        gr.Markdown("# 📦 DaSiWa Quant Station\nWan 2.2")

        with gr.Row():
            # Column 1: Source Settings
            with gr.Column(scale=3):
                with gr.Group():
                    gr.Markdown("### 📥 Source Settings")
                    base_dd = gr.Dropdown(
                        label="Select Source Safetensors", 
                        interactive=True,
                        allow_custom_value=True 
                    )
                    friendly_name = gr.Textbox(label="Model Display Name (Required)", placeholder="e.g. Cinema-Mix-V1")
                    refresh_btn = gr.Button("🔄 Refresh Models", size="sm")
                run_btn = gr.Button("🧩 START BATCH", variant="primary")
                stop_btn = gr.Button("🛑 STOP", variant="secondary")

            # Column 2: Formats & Optimizations (Now Next to Each Other)
            with gr.Column(scale=4):
                with gr.Group():
                    gr.Markdown("### ⚖️ Select Formats & 🛠️ Flags")
                    with gr.Row():
                        with gr.Column(scale=1):
                            q_format = gr.CheckboxGroup(
                                choices=[
                                    "FP8", "INT8 Block-wise", "NVFP4", 
                                    "GGUF_Q8_0", "GGUF_Q6_K", "GGUF_Q5_K_M", 
                                    "GGUF_Q4_K_M", "GGUF_Q3_K_S", "GGUF_Q2_K"
                                ],
                                label="Target Formats", 
                                value=["FP8"]
                            )
                        with gr.Column(scale=1):
                            extra_flags = gr.Radio( # Changed from CheckboxGroup to Radio
                                choices=[
                                    "Ultra-Quality (Optimizer)", 
                                    "Auto-Quality (Heur)", 
                                    "Fast Mode (Simple)"
                                ],
                                label="Quantization Tweaks", 
                                value="Auto-Quality (Heur)" # Removed brackets, now a single string
                            )

            # Column 3: Vitals & Status
            with gr.Column(scale=3):
                vitals_box = gr.Textbox(label="Hardware Vitals", value=get_sys_info(), lines=2, interactive=False)
                gr.Timer(2).tick(get_sys_info, outputs=vitals_box)
                pipeline_status = gr.Label(label="Process State", value="Idle")

        # Terminal and Metadata
        terminal_box = gr.Textbox(lines=22, interactive=False, show_label=False, elem_id="terminal", placeholder="System logs...")
        
        with gr.Row():
            with gr.Column(scale=6):
                metadata_input = gr.Code(value=update_metadata_preview("Enter Name..."), language="json", interactive=True)
            with gr.Column(scale=4):
                inject_btn = gr.Button("💉 Inject to Source")
                read_btn = gr.Button("🔍 Read Current Header")

        # Logic Wiring
        setup_callbacks(
            base_dd, friendly_name, refresh_btn, run_btn, stop_btn, 
            q_format, pipeline_status, extra_flags, terminal_box, 
            metadata_input, inject_btn, read_btn
        )

        # 🚀 INITIALIZATION: Scan folder on startup
        from utils.file_ops import list_files
        demo.load(fn=list_files, outputs=[base_dd])


    return demo