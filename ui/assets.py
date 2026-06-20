# ui/assets.py

# Dictionary mapping architectures to their specific metadata fields.
# Used by core/metadata_manager.py - do not rename or remove.
#
# Keys must match the labels in core.safetensors_engine.ARCH_REGISTRY and
# ui/layout.py's Architecture dropdown. Archs without an entry here fall
# back to the WAN 2.2 template via get_current_meta() in metadata_manager.
MODEL_METADATA_CONFIGS = {
    "Not set": {
        # Generic template used when the user declines to declare an
        # architecture. modelspec.architecture is left as a marker so
        # downstream tools can tell the field was deliberately unset.
        "modelspec.title": "{model_name}",
        "modelspec.author": "Darksidewalker",
        "modelspec.description": "Model quantized via DaSiWa Station with no architecture preset.",
        "modelspec.architecture": "unspecified",
        "modelspec.implementation": "",
        "modelspec.license": "Custom License Addendum Distribution Restriction",
        "modelspec.tags": "diffusion, DaSiWa",
    },
    "WAN 2.2": {
        "modelspec.title": "{model_name}",
        "modelspec.author": "Darksidewalker",
        "modelspec.description": "Multi-Expert Image-to-Video diffusion model quantized via DaSiWa Station.",
        "modelspec.architecture": "wan_2.2_14b_i2v",
        "modelspec.implementation": "https://github.com/Wan-Video/Wan2.2",
        "modelspec.license": "apache-2.0 and Custom License Addendum Distribution Restriction",
        "modelspec.tags": "image-to-video, moe, diffusion, wan2.2, DaSiWa",
        "modelspec.resolution": "832x480",
        "modelspec.resolution_hints": "480p, 720p",
        "modelspec.resolution_native": "832x480",
        "modelspec.resolution_aspect": "16:9",
    },
    "LTX-2.3": {
        "modelspec.title": "{model_name}",
        "modelspec.author": "Darksidewalker",
        "modelspec.description": "High-fidelity Image-to-Video diffusion model quantized via DaSiWa Station.",
        "modelspec.architecture": "ltx2.3_22b_ti2v",
        "modelspec.implementation": "https://github.com/Lightricks/LTX-2",
        "modelspec.license": "LTX-2 Community License Agreement and Custom License Addendum Distribution Restriction",
        "modelspec.tags": "image-to-video, text-to-video, video-to-video, audio, ltx2, diffusion, DaSiWa",
        "modelspec.resolution": "1280x720",
        "modelspec.resolution_hints": "720p, 1080p, 480p",
        "modelspec.resolution_native": "1280x720",
        "modelspec.resolution_aspect": "16:9",
    },
    "Flux.2": {
        "modelspec.title": "{model_name}",
        "modelspec.author": "Darksidewalker",
        "modelspec.description": "Flux.2 diffusion model quantized via DaSiWa Station.",
        "modelspec.architecture": "flux2",
        "modelspec.implementation": "https://github.com/black-forest-labs/flux2",
        "modelspec.license": "flux-2-dev-non-commercial-license and Custom License Addendum Distribution Restriction",
        "modelspec.tags": "text-to-image, diffusion, flux2, DaSiWa",
        "modelspec.resolution": "1024x1024",
        "modelspec.resolution_hints": "1MP, 2MP, 4MP",
        "modelspec.resolution_native": "1024x1024",
        "modelspec.resolution_aspect": "1:1",
    },
    "Hunyuan Video": {
        "modelspec.title": "{model_name}",
        "modelspec.author": "Darksidewalker",
        "modelspec.description": "Hunyuan Video diffusion model quantized via DaSiWa Station.",
        "modelspec.architecture": "hunyuan_video",
        "modelspec.implementation": "https://github.com/Tencent/HunyuanVideo",
        "modelspec.license": "tencent-hunyuan-community and Custom License Addendum Distribution Restriction",
        "modelspec.tags": "text-to-video, diffusion, hunyuan, DaSiWa",
        "modelspec.resolution": "1280x720",
        "modelspec.resolution_hints": "480p, 720p",
        "modelspec.resolution_native": "1280x720",
        "modelspec.resolution_aspect": "16:9",
    },
    "Qwen Image": {
        "modelspec.title": "{model_name}",
        "modelspec.author": "Darksidewalker",
        "modelspec.description": "Qwen Image diffusion model quantized via DaSiWa Station.",
        "modelspec.architecture": "qwen_image",
        "modelspec.implementation": "https://github.com/QwenLM/Qwen-Image",
        "modelspec.license": "apache-2.0 and Custom License Addendum Distribution Restriction",
        "modelspec.tags": "text-to-image, diffusion, qwen, DaSiWa",
        "modelspec.resolution": "1328x1328",
        "modelspec.resolution_hints": "1024x1024, 1328x1328, 4MP",
        "modelspec.resolution_native": "1328x1328",
        "modelspec.resolution_aspect": "1:1",
    },
    "Z-Image": {
        "modelspec.title": "{model_name}",
        "modelspec.author": "Darksidewalker",
        "modelspec.description": "Z-Image diffusion model quantized via DaSiWa Station.",
        "modelspec.architecture": "z_image",
        "modelspec.implementation": "",
        "modelspec.license": "apache-2.0 and Custom License Addendum Distribution Restriction",
        "modelspec.tags": "text-to-image, diffusion, z-image, DaSiWa",
        "modelspec.resolution": "1024x1024",
        "modelspec.resolution_hints": "1024x1024, 1440x1440, 1920x1088",
        "modelspec.resolution_native": "1024x1024",
        "modelspec.resolution_aspect": "1:1",
    },
    "Z-Image Refiner": {
        "modelspec.title": "{model_name}",
        "modelspec.author": "Darksidewalker",
        "modelspec.description": "Z-Image refiner model quantized via DaSiWa Station.",
        "modelspec.architecture": "z_image_refiner",
        "modelspec.implementation": "",
        "modelspec.license": "apache-2.0 and Custom License Addendum Distribution Restriction",
        "modelspec.tags": "text-to-image, refiner, diffusion, z-image, DaSiWa",
        "modelspec.resolution": "1024x1024",
        "modelspec.resolution_hints": "1024x1024, 1440x1440, 1920x1088",
        "modelspec.resolution_native": "1024x1024",
        "modelspec.resolution_aspect": "1:1",
    },
    "Anima": {
        "modelspec.title": "{model_name}",
        "modelspec.author": "Darksidewalker",
        "modelspec.description": "Anima model quantized via DaSiWa Station.",
        "modelspec.architecture": "anima",
        "modelspec.implementation": "",
        "modelspec.license": "CircleStone Labs Non-Commercial License",
        "modelspec.tags": "diffusion, anima, DaSiWa, anime, art, illustration",
        "modelspec.resolution": "1024x1024",
    },
    "Radiance": {
        "modelspec.title": "{model_name}",
        "modelspec.author": "Darksidewalker",
        "modelspec.description": "Radiance model quantized via DaSiWa Station.",
        "modelspec.architecture": "radiance",
        "modelspec.implementation": "",
        "modelspec.license": "Custom License Addendum Distribution Restriction",
        "modelspec.tags": "radiance, DaSiWa",
        "modelspec.resolution": "1024x1024",
        "modelspec.resolution_hints": "512x512, 1024x1024",
        "modelspec.resolution_native": "1024x1024",
        "modelspec.resolution_aspect": "1:1",
    },
    "Distillation Large": {
        "modelspec.title": "{model_name}",
        "modelspec.author": "Darksidewalker",
        "modelspec.description": "Distilled (large) model quantized via DaSiWa Station.",
        "modelspec.architecture": "distillation_large",
        "modelspec.implementation": "",
        "modelspec.license": "Custom License Addendum Distribution Restriction",
        "modelspec.tags": "distillation, DaSiWa",
        "modelspec.resolution": "1024x1024",
        "modelspec.resolution_hints": "512x512, 1024x1024",
        "modelspec.resolution_native": "1024x1024",
        "modelspec.resolution_aspect": "1:1",
    },
    "Distillation Small": {
        "modelspec.title": "{model_name}",
        "modelspec.author": "Darksidewalker",
        "modelspec.description": "Distilled (small) model quantized via DaSiWa Station.",
        "modelspec.architecture": "distillation_small",
        "modelspec.implementation": "",
        "modelspec.license": "Custom License Addendum Distribution Restriction",
        "modelspec.tags": "distillation, DaSiWa",
        "modelspec.resolution": "1024x1024",
        "modelspec.resolution_hints": "512x512, 1024x1024",
        "modelspec.resolution_native": "1024x1024",
        "modelspec.resolution_aspect": "1:1",
    },
    "NeRF Large": {
        "modelspec.title": "{model_name}",
        "modelspec.author": "Darksidewalker",
        "modelspec.description": "NeRF (large) model quantized via DaSiWa Station.",
        "modelspec.architecture": "nerf_large",
        "modelspec.implementation": "",
        "modelspec.license": "Custom License Addendum Distribution Restriction",
        "modelspec.tags": "nerf, DaSiWa",
        "modelspec.resolution": "512x512",
        "modelspec.resolution_hints": "256x256, 512x512",
        "modelspec.resolution_native": "512x512",
        "modelspec.resolution_aspect": "1:1",
    },
    "NeRF Small": {
        "modelspec.title": "{model_name}",
        "modelspec.author": "Darksidewalker",
        "modelspec.description": "NeRF (small) model quantized via DaSiWa Station.",
        "modelspec.architecture": "nerf_small",
        "modelspec.implementation": "",
        "modelspec.license": "Custom License Addendum Distribution Restriction",
        "modelspec.tags": "nerf, DaSiWa",
        "modelspec.resolution": "512x512",
        "modelspec.resolution_hints": "256x256, 512x512",
        "modelspec.resolution_native": "512x512",
        "modelspec.resolution_aspect": "1:1",
    },
    "T5-XXL": {
        "modelspec.title": "{model_name}",
        "modelspec.author": "Darksidewalker",
        "modelspec.description": "T5-XXL text encoder quantized via DaSiWa Station.",
        "modelspec.architecture": "t5xxl",
        "modelspec.implementation": "",
        "modelspec.license": "apache-2.0 and Custom License Addendum Distribution Restriction",
        "modelspec.tags": "text-encoder, t5, DaSiWa",
    },
    "Qwen 3.5": {
        "modelspec.title": "{model_name}",
        "modelspec.author": "Darksidewalker",
        "modelspec.description": "Qwen 3.5 text encoder quantized via DaSiWa Station.",
        "modelspec.architecture": "qwen35",
        "modelspec.implementation": "",
        "modelspec.license": "apache-2.0 and Custom License Addendum Distribution Restriction",
        "modelspec.tags": "text-encoder, qwen, DaSiWa",
    },
    "Mistral": {
        "modelspec.title": "{model_name}",
        "modelspec.author": "Darksidewalker",
        "modelspec.description": "Mistral text encoder quantized via DaSiWa Station.",
        "modelspec.architecture": "mistral",
        "modelspec.implementation": "",
        "modelspec.license": "apache-2.0 and Custom License Addendum Distribution Restriction",
        "modelspec.tags": "text-encoder, mistral, DaSiWa",
    },
    "Visual": {
        "modelspec.title": "{model_name}",
        "modelspec.author": "Darksidewalker",
        "modelspec.description": "Visual encoder quantized via DaSiWa Station.",
        "modelspec.architecture": "visual",
        "modelspec.implementation": "",
        "modelspec.license": "Custom License Addendum Distribution Restriction",
        "modelspec.tags": "visual-encoder, DaSiWa",
    },
    "Generic Text": {
        "modelspec.title": "{model_name}",
        "modelspec.author": "Darksidewalker",
        "modelspec.description": "Generic text encoder quantized via DaSiWa Station.",
        "modelspec.architecture": "generic_text",
        "modelspec.implementation": "",
        "modelspec.license": "Custom License Addendum Distribution Restriction",
        "modelspec.tags": "text-encoder, DaSiWa",
    },
}

COMMON_METADATA = {
    "modelspec.date": "{date}",
    "quantization.tool": "https://github.com/darksidewalker/dasiwa-quant-station",
    "quantization.bits": "{bits}"
}


# Softer dark palette. Terminal uses a muted teal instead of pure neon green
# (less eye strain on long quant runs). Borders, backgrounds and accents use
# GitHub-dark-inspired neutrals.
CSS_STYLE = """
/* Terminal output: monospace, soft teal on near-black */
#terminal textarea {
    background-color: #0d1117 !important;
    color: #7dd3c0 !important;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
    border: 1px solid #21262d !important;
    border-radius: 6px !important;
}

/* Hardware vitals: same monospace family, dim accent */
#vitals textarea {
    background-color: #0d1117 !important;
    color: #8b949e !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 12px !important;
    border: 1px solid #21262d !important;
    border-radius: 6px !important;
}

/* JSON editor: lighter than terminal so it reads as editable */
#meta_editor {
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}

/* Card-style groups: subtle separation without heavy borders */
.tool-card {
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    padding: 12px !important;
    background: #0d1117 !important;
    margin-bottom: 8px !important;
}

/* Compact ggufy/safetensors target format selector */
#q_format fieldset {
    display: grid !important;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 4px !important;
}

#q_format label {
    font-size: 0.78rem !important;
    line-height: 1.2 !important;
    padding: 5px 7px !important;
}

#q_format input[type="checkbox"] {
    transform: scale(0.88);
    margin-right: 5px !important;
}

#q_format .gradio-checkbox {
    padding: 3px 5px !important;
}

/* Status banner */
#status-banner {
    border-left: 3px solid #7dd3c0 !important;
    padding: 8px 12px !important;
    background: rgba(125, 211, 192, 0.05) !important;
    border-radius: 4px !important;
}

/* Section headings - smaller, less shouty than Gradio defaults */
.section-heading h3 {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #8b949e !important;
    margin: 0 0 8px 0 !important;
}

/* Primary action: muted teal instead of default Gradio orange */
.primary-action {
    background: linear-gradient(180deg, #2ea043 0%, #238636 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border: 1px solid rgba(240, 246, 252, 0.1) !important;
}

/* Destructive action: muted red for STOP */
.danger-action {
    background: #21262d !important;
    color: #f85149 !important;
    border: 1px solid #f85149 !important;
}
.danger-action:hover {
    background: #f85149 !important;
    color: white !important;
}
"""


def get_theme():
    """Returns the configured Gradio theme.
    Built lazily because gradio import is heavy and not all entry points
    that touch assets.py need the theme (e.g. metadata utilities)."""
    import gradio as gr
    return gr.themes.Default(
        primary_hue="emerald",
        secondary_hue="slate",
        neutral_hue="slate",
    ).set(
        body_background_fill_dark="#010409",
        background_fill_primary_dark="#0d1117",
        background_fill_secondary_dark="#161b22",
        border_color_primary_dark="#30363d",
    )
