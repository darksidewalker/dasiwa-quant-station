# ui/layout.py
"""
Layout structure:

  ┌─ Sidebar (lean) ──┐  ┌─ Main area ─────────────────────────┐
  │  Source picker    │  │  Status strip                       │
  │  Architecture     │  │                                     │
  │  Display name     │  │  Terminal output (always visible)   │
  │                   │  │                                     │
  │                   │  │  ▾ Quantize         (open default)  │
  │                   │  │  ▸ Tools            (collapsed)     │
  │                   │  │  ▸ Metadata         (collapsed)     │
  └───────────────────┘  └─────────────────────────────────────┘

Accordions instead of tabs because Gradio 6's tab-switch reconciliation
freezes the browser with non-trivial UIs (gradio issue #12943). Accordions
expand in place, so the terminal stays visible without any tab-switch
machinery. All component variable names are preserved.
"""
import gradio as gr
from core.metadata_manager import update_metadata_preview
from config import MODELS_DIR
from utils.system import get_sys_info
from utils.file_listing import get_model_list
from utils.file_ops import list_dirs
from ui.callbacks import setup_callbacks


def create_ui():
    with gr.Blocks(title="DaSiWa Quant Station") as demo:

        # =========================================================
        # SIDEBAR: only set-once-per-session settings
        # =========================================================
        with gr.Sidebar(position="left", width=280, open=True):
            gr.Markdown("## 📦 DaSiWa")
            gr.Markdown(
                "<span style='color:#8b949e; font-size:0.85em;'>"
                "Video model quantization workstation</span>"
            )

            gr.Markdown("### Source", elem_classes=["section-heading"])
            models_dir_dd = gr.Dropdown(
                label="Model Directory",
                choices=[],
                value=MODELS_DIR,
                interactive=True,
            )
            base_dd = gr.Dropdown(
                label="Safetensors file",
                interactive=True,
                allow_custom_value=True,
            )
            refresh_btn = gr.Button("↻ Refresh folder", size="sm")

            model_type = gr.Dropdown(
                # Grouped by category in the visible label; the actual value
                # is the unprefixed name used as the key in
                # core.safetensors_engine.ARCH_REGISTRY.
                choices=[
                    ("— None —    Not set",                "Not set"),
                    ("— Video —   WAN 2.2",                "WAN 2.2"),
                    ("— Video —   LTX-2.3",                "LTX-2.3"),
                    ("— Video —   Hunyuan Video",          "Hunyuan Video"),
                    ("— Image —   Flux.2",                 "Flux.2"),
                    ("— Image —   Qwen Image",             "Qwen Image"),
                    ("— Image —   Z-Image",                "Z-Image"),
                    ("— Image —   Z-Image Refiner",        "Z-Image Refiner"),
                    ("— Image —   Anima",                  "Anima"),
                    ("— Other —   Radiance",               "Radiance"),
                    ("— Other —   Distillation Large",     "Distillation Large"),
                    ("— Other —   Distillation Small",     "Distillation Small"),
                    ("— Other —   NeRF Large",             "NeRF Large"),
                    ("— Other —   NeRF Small",             "NeRF Small"),
                    ("— Text —    T5-XXL",                 "T5-XXL"),
                    ("— Text —    Qwen 3.5",               "Qwen 3.5"),
                    ("— Text —    Mistral",                "Mistral"),
                    ("— Text —    Visual",                 "Visual"),
                    ("— Text —    Generic Text",           "Generic Text"),
                ],
                value="WAN 2.2",
                label="Architecture",
            )
            friendly_name = gr.Textbox(
                label="Display name",
                placeholder="e.g. TreasureChest-v1",
            )
            full_checkpoint = gr.Checkbox(
                label="Full Checkpoint (Inc. VAE)",
                value=False,
                info="Check if the source file includes VAE/Audio VAE weights (LTX-2.3 only)."
            )

        # =========================================================
        # MAIN AREA: header + status + pinned terminal + accordions
        # =========================================================
        gr.Markdown(
            "# DaSiWa Quant Station\n"
            "<span style='color:#8b949e;'>Advanced video model quantization</span>"
        )

        # --- Status strip ---
        with gr.Row(equal_height=True):
            pipeline_status = gr.Label(
                label="Status",
                value="Idle",
                scale=1,
            )
            vitals_box = gr.Textbox(
                label="Hardware vitals",
                value=get_sys_info(),
                lines=3,
                interactive=False,
                elem_id="vitals",
                scale=2,
            )
            # 5s instead of 2s - reduces background reactive churn under
            # Gradio 6 Svelte 5. Hardware vitals don't change that fast.
            gr.Timer(5).tick(get_sys_info, outputs=vitals_box)

        # --- Terminal (always visible above accordions) ---
        terminal_box = gr.Textbox(
            label="Console",
            lines=18,
            interactive=False,
            elem_id="terminal",
            placeholder=(
                "Configure your run in the Quantize section below, then click Start.\n"
                "Output streams here. The console stays visible while you use any tool."
            ),
            buttons=["copy"],
        )

        # ------------------------------------------------
        # ACCORDION 1: QUANTIZE (open by default)
        # ------------------------------------------------
        with gr.Accordion("▶  Quantize", open=True):
            with gr.Row():
                # Left: strategy + optimizer
                with gr.Column(scale=1):
                    gr.Markdown("### Strategy", elem_classes=["section-heading"])
                    extra_flags = gr.Radio(
                        choices=[
                            "Ultra-Quality (Optimizer)",
                            "Auto-Quality (Heur)",
                            "Simple",
                        ],
                        label="Quantization strategy",
                        value="Ultra-Quality (Optimizer)",
                    )
                    optimizer_choice = gr.Dropdown(
                        choices=["prodigy", "adamw", "radam", "original"],
                        value="prodigy",
                        label="Optimizer (Ultra only)",
                    )
                    tweak_hint = gr.Markdown(
                        "<span style='color:#8b949e; font-size:0.85em;'>"
                        "Manual optimizer active (9000 iters)</span>"
                    )

                # Middle: target formats
                with gr.Column(scale=1):
                    gr.Markdown("### Formats", elem_classes=["section-heading"])
                    q_format = gr.CheckboxGroup(
                        choices=[
                            "FP8", "INT8 Block-wise", "NVFP4",
                            "GGUF_Q8_0", "GGUF_Q6_K", "GGUF_Q5_K_M",
                            "GGUF_Q4_K_M", "GGUF_Q3_K_S", "GGUF_Q2_K"
                        ],
                        label="Target formats",
                        value=["FP8"],
                    )

                # Right: options + run buttons
                with gr.Column(scale=1):
                    gr.Markdown("### Options", elem_classes=["section-heading"])
                    auto_layer_config = gr.Checkbox(
                        label="Keep sensitive layers high-precision",
                        value=True,
                        info="FP8 base: stays at FP16. NVFP4/INT8: bumped to FP8.",
                    )
                    low_vram = gr.Checkbox(
                        label="Low VRAM mode",
                        value=False,
                    )

                    gr.Markdown("### Run", elem_classes=["section-heading"])
                    with gr.Row():
                        run_btn = gr.Button(
                            "▶  Start Batch",
                            variant="primary",
                            scale=2,
                            elem_classes=["primary-action"],
                        )
                        stop_btn = gr.Button(
                            "■  Stop",
                            scale=1,
                            elem_classes=["danger-action"],
                        )

        # ------------------------------------------------
        # ACCORDION 2: TOOLS (collapsed by default)
        # ------------------------------------------------
        with gr.Accordion("🔍  Tools", open=False):
            gr.Markdown(
                "<span style='color:#8b949e;'>"
                "Inspection and configuration. Output appears in the console above."
                "</span>"
            )
            with gr.Row():
                # Left: source-only tools
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["tool-card"]):
                        gr.Markdown(
                            "**Pattern Audit**  \n"
                            "<span style='color:#8b949e; font-size:0.9em;'>"
                            "Verify our patterns cover the source file. "
                            "Suspicious layers indicate uncovered families."
                            "</span>"
                        )
                        audit_btn = gr.Button("🩺  Audit Patterns", size="sm")

                    with gr.Group(elem_classes=["tool-card"]):
                        gr.Markdown(
                            "**5D Tensor Scan**  \n"
                            "<span style='color:#8b949e; font-size:0.9em;'>"
                            "List tensors with >4 dimensions (WAN 5D self-"
                            "healing diagnostic)."
                            "</span>"
                        )
                        scan_btn = gr.Button("🔎  Scan 5D Tensors", size="sm")

                # Right: reference-based tools
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["tool-card"]):
                        gr.Markdown(
                            "**Compare to Reference FP8**  \n"
                            "<span style='color:#8b949e; font-size:0.9em;'>"
                            "Diff against an author's FP8 file. Reports "
                            "disagreements and generates suggested patterns."
                            "</span>"
                        )
                        reference_dd = gr.Dropdown(
                            label="Reference FP8 file",
                            interactive=True,
                            allow_custom_value=True,
                            info="Place file in models/ then click Refresh in sidebar.",
                        )
                        compare_btn = gr.Button("🔬  Compare to Reference", size="sm")

                    with gr.Group(elem_classes=["tool-card"]):
                        gr.Markdown(
                            "**Exact Config from Reference**  \n"
                            "<span style='color:#8b949e; font-size:0.9em;'>"
                            "Mirror the reference's per-tensor preservation "
                            "choices exactly. Next run uses _exact suffix."
                            "</span>"
                        )
                        with gr.Row():
                            build_exact_btn = gr.Button(
                                "🎯  Build Exact", size="sm", scale=2
                            )
                            clear_exact_btn = gr.Button(
                                "🧹  Clear", size="sm", scale=1
                            )

        # ------------------------------------------------
        # ACCORDION 3: METADATA (collapsed by default)
        # ------------------------------------------------
        with gr.Accordion("📝  Metadata", open=False):
            gr.Markdown(
                "<span style='color:#8b949e;'>"
                "Preview updates live based on architecture and display name. "
                "Edit the JSON to customize, then use Inject."
                "</span>"
            )
            with gr.Row():
                with gr.Column(scale=3):
                    metadata_input = gr.Code(
                        value=update_metadata_preview("TreasureChest", "WAN 2.2"),
                        language="json",
                        interactive=True,
                        elem_id="meta_editor",
                        lines=18,
                    )
                with gr.Column(scale=1):
                    gr.Markdown(
                        "<span style='color:#8b949e; font-size:0.85em;'>"
                        "**Read** displays the current header of the file "
                        "selected in the sidebar.<br><br>"
                        "**Inject** writes the JSON above into that same "
                        "file. Use with care - overwrites existing metadata."
                        "</span>"
                    )
                    read_btn = gr.Button("🔍  Read Header", size="sm")
                    inject_btn = gr.Button(
                        "💉  Inject to Source",
                        variant="primary",
                        size="sm",
                    )

        # =========================================================
        # REACTIVE LOGIC
        # =========================================================
        def on_settings_change(m_type, name, selection, is_full):
            if selection == "Ultra-Quality (Optimizer)":
                hint = ("<span style='color:#8b949e; font-size:0.85em;'>"
                        "Manual optimizer active (9000 iters)</span>")
                opt_update = gr.update(interactive=True)
            elif selection == "Auto-Quality (Heur)":
                hint = ("<span style='color:#8b949e; font-size:0.85em;'>"
                        "Heuristics active (engine-controlled)</span>")
                opt_update = gr.update(interactive=False, value="prodigy")
            else:  # Simple
                hint = ("<span style='color:#8b949e; font-size:0.85em;'>"
                        "Fast simple quant (no optimization)</span>")
                opt_update = gr.update(interactive=False)
            new_json = update_metadata_preview(name, m_type, is_full=is_full)
            return opt_update, hint, new_json

        settings_inputs = [model_type, friendly_name, extra_flags, full_checkpoint]
        settings_outputs = [optimizer_choice, tweak_hint, metadata_input]
        for component in settings_inputs:
            component.change(
                fn=on_settings_change,
                inputs=settings_inputs,
                outputs=settings_outputs,
            )

        # Wire all callbacks
        setup_callbacks(
            models_dir_dd,
            base_dd, friendly_name, refresh_btn, run_btn, stop_btn,
            q_format, pipeline_status, extra_flags, terminal_box,
            metadata_input, inject_btn, read_btn,
            scan_btn, model_type, optimizer_choice,
            low_vram, auto_layer_config, audit_btn,
            reference_dd, compare_btn,
            build_exact_btn, clear_exact_btn,
            full_checkpoint,
        )

        # Initial population: directories dropdown, then file dropdowns
        demo.load(fn=list_dirs, inputs=[], outputs=[models_dir_dd])

        def refresh_both_init(m_path):
            update = get_model_list(m_path)
            return gr.update(choices=update), gr.update(choices=update)
        demo.load(fn=refresh_both_init, inputs=[models_dir_dd], outputs=[base_dd, reference_dd])

        # Ensure refresh button also refreshes available directories
        refresh_btn.click(fn=list_dirs, inputs=[], outputs=[models_dir_dd])

    return demo
