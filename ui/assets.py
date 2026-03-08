# ui/assets.py

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

CSS_STYLE = """
#terminal textarea { 
    background-color: #0d1117 !important; 
    color: #00ff41 !important; 
    font-family: 'Fira Code', monospace !important; 
    font-size: 13px !important;
}
.vitals-card { border: 1px solid #30363d; padding: 15px; border-radius: 8px; background: #0d1117; }
.primary-button {
    background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%) !important;
    color: white !important;
    font-weight: bold !important;
}
.stop-button { background: #8b0000 !important; color: white !important; }
"""

JS_AUTO_SCROLL = """
(x) => {
    const el = document.getElementById('terminal');
    if (el) {
        const textarea = el.querySelector('textarea');
        if (textarea) textarea.scrollTop = textarea.scrollHeight;
    }
}
"""