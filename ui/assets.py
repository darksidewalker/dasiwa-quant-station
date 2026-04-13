# ui/assets.py

# Dictionary mapping architectures to their specific metadata fields
MODEL_METADATA_CONFIGS = {
    "WAN 2.2": {
        "modelspec.title": "{model_name}",
        "modelspec.author": "Darksidewalker",
        "modelspec.description": "Multi-Expert Image-to-Video diffusion model quantized via DaSiWa Station.",
        "modelspec.architecture": "wan_2.2_14b_i2v",
        "modelspec.implementation": "https://github.com/Wan-Video/Wan2.2",
        "modelspec.license": "apache-2.0 and Custom License Addendum Distribution Restriction",
        "modelspec.tags": "image-to-video, moe, diffusion, wan2.2, DaSiWa",
    },
    "LTX-2.3": {
        "modelspec.title": "{model_name}",
        "modelspec.author": "Darksidewalker",
        "modelspec.description": "High-fidelity Image-to-Video diffusion model quantized via DaSiWa Station.",
        "modelspec.architecture": "ltx2.3_22b_ti2v",
        "modelspec.implementation": "https://github.com/Lightricks/LTX-2",
        "modelspec.license": "LTX-2 Community License Agreement and Custom License Addendum Distribution Restriction",
        "modelspec.tags": "image-to-video, text-to-video, video-to-video, audio, ltx2, diffusion, DaSiWa",
    }
}

COMMON_METADATA = {
    "modelspec.date": "{date}",
    "quantization.tool": "https://github.com/darksidewalker/dasiwa-wan2.2-master",
    "quantization.bits": "{bits}"
}

CSS_STYLE = """
#terminal textarea { 
    background-color: #0d1117 !important; 
    color: #00ff41 !important; 
    font-family: 'Fira Code', monospace !important; 
    font-size: 13px !important;
}
.vitals-card { border: 1px solid #30363d; padding: 15px; border-radius: 8px; background: #0d1117; }
"""