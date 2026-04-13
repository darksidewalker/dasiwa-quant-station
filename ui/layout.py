# ui/layout.py
import gradio as gr
from core.metadata_manager import update_metadata_preview
from utils.system import get_sys_info
from ui.callbacks import setup_callbacks

def create_ui():
    with gr.Blocks(title="DaSiWa Quant Station") as demo:
        gr.Markdown("# 📦 DaSiWa Quant Station\n*Advanced Video Model Quantization Workstation*")

        with gr.Row():
            # --- LEFT SIDE: CONFIGURATION PANEL (Scale 4) ---
            with gr.Column(scale=4):
                with gr.Group():
                    gr.Markdown("### 📥 1. Source & Architecture")
                    base_dd = gr.Dropdown(
                        label="Source Safetensors", 
                        interactive=True,
                        allow_custom_value=True 
                    )
                    refresh_btn = gr.Button("🔄 Refresh Folder", size="sm")
                    
                    model_type = gr.Radio(
                        choices=["WAN 2.2", "LTX-2.3"],
                        value="WAN 2.2", 
                        label="Architecture Selection"
                    )
                    friendly_name = gr.Textbox(
                        label="Display Name (Trigger Metadata Update)", 
                        placeholder="e.g. TreasureChest-v1"
                    )
                
                with gr.Group():
                    gr.Markdown("### ⚙️ 2. Engine & Optimization")
                    optimizer_choice = gr.Dropdown(
                        choices=["prodigy", "adamw", "radam", "original"], 
                        value="prodigy", 
                        label="Optimizer"
                    )
                    extra_flags = gr.Radio(
                        choices=["Ultra-Quality (Optimizer)", "Auto-Quality (Heur)", "Simple"],
                        label="Quantization Strategy", 
                        value="Ultra-Quality (Optimizer)"
                    )
                    tweak_hint = gr.Markdown("*Mode: Manual Optimizer active (9000 iters)*")

                with gr.Group():
                    gr.Markdown("### ⚖️ 3. Quantization Targets")
                    q_format = gr.CheckboxGroup(
                        choices=[
                            "FP8", "INT8 Block-wise", "NVFP4", 
                            "GGUF_Q8_0", "GGUF_Q6_K", "GGUF_Q5_K_M", 
                            "GGUF_Q4_K_M", "GGUF_Q3_K_S", "GGUF_Q2_K"
                        ],
                        label="Target Formats", 
                        value=["FP8"]
                    )
                    with gr.Row():
                        low_vram = gr.Checkbox(label="Low VRAM Mode", value=False)

                with gr.Row():
                    run_btn = gr.Button("🧩 START BATCH", variant="primary", scale=2)
                    stop_btn = gr.Button("🛑 STOP", variant="secondary", scale=1)

            # --- RIGHT SIDE: WORKSPACE & STATUS (Scale 6) ---
            with gr.Column(scale=6):
                # Row: Telemetry & Vitals
                with gr.Row():
                    with gr.Column(scale=1):
                        pipeline_status = gr.Label(label="Process State", value="Idle")
                    with gr.Column(scale=2):
                        vitals_box = gr.Textbox(
                            label="Hardware Vitals", 
                            value=get_sys_info(), 
                            lines=2, 
                            interactive=False
                        )
                        gr.Timer(2).tick(get_sys_info, outputs=vitals_box)

                # Terminal Window (CLI Output)
                terminal_box = gr.Textbox(
                    label="CLI Console Output",
                    lines=16, 
                    interactive=False, 
                    elem_id="terminal", 
                    placeholder="Execution logs will stream here..."
                )

                # Metadata Suite (Dynamic JSON Editor)
                with gr.Group():
                    gr.Markdown("### 📝 Metadata & Header Injection")
                    metadata_input = gr.Code(
                        value=update_metadata_preview("TreasureChest", "WAN 2.2"), 
                        language="json", 
                        interactive=True,
                        elem_id="meta_editor"
                    )
                    with gr.Row():
                        read_btn = gr.Button("🔍 Read Current Header")
                        inject_btn = gr.Button("💉 Inject to Source", variant="primary")
                        scan_btn = gr.Button("🔎 Scan 5D Tensors", variant="secondary")

        # --- RE-ACTIVE UI LOGIC ---

        # --- RE-ACTIVE UI LOGIC ---

        def on_settings_change(m_type, name, selection):
            """
            Synchronizes the UI state when Architecture, Name, or Strategy changes.
            """
            # 1. Logic for Strategy Selection
            if selection == "Ultra-Quality (Optimizer)":
                hint = "*Mode: Manual Optimizer active (9000 iters)*"
                opt_update = gr.update(interactive=True)
            elif selection == "Auto-Quality (Heur)":
                hint = "*Mode: Heuristics active (Engine-controlled)*"
                opt_update = gr.update(interactive=False, value="prodigy")
            else: # Simple
                hint = "*Mode: Fast Simple Quant (Optimization Disabled)*"
                opt_update = gr.update(interactive=False)

            # 2. Update Metadata JSON Layout
            new_json = update_metadata_preview(name, m_type)
            
            # RETURN exactly 3 values to match the 3 components in 'outputs'
            return opt_update, hint, new_json

        # --- Trigger Grouping ---
        # Ensure these lists match the function signature and the UI components
        settings_trigger_inputs = [model_type, friendly_name, extra_flags]
        settings_trigger_outputs = [optimizer_choice, tweak_hint, metadata_input]

        # Connect the changes
        for component in settings_trigger_inputs:
            component.change(
                fn=on_settings_change, 
                inputs=settings_trigger_inputs, 
                outputs=settings_trigger_outputs
            )

        # Each of these triggers must send ALL 3 inputs to satisfy the function signature
        model_type.change(
            fn=on_settings_change, 
            inputs=settings_trigger_inputs, 
            outputs=settings_trigger_outputs
        )

        friendly_name.change(
            fn=on_settings_change, 
            inputs=settings_trigger_inputs, 
            outputs=settings_trigger_outputs
        )

        extra_flags.change(
            fn=on_settings_change, 
            inputs=settings_trigger_inputs, 
            outputs=settings_trigger_outputs
        )

        # --- Logic Wiring ---
        # Connects all 17 UI components to the callback handlers in ui/callbacks.py
        setup_callbacks(
            base_dd, friendly_name, refresh_btn, run_btn, stop_btn, 
            q_format, pipeline_status, extra_flags, terminal_box, 
            metadata_input, inject_btn, read_btn, 
            scan_btn, model_type, optimizer_choice,
            low_vram
        )

        # Initial folder scan on startup
        from utils.file_ops import list_files
        demo.load(fn=list_files, outputs=[base_dd])

    return demo