# config.py
import os

# --- DIRECTORIES ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts") 

# --- 5D FIX DATA (In Root) ---
FIX_5D_DATA = os.path.join(ROOT_DIR, "fix_5d_tensors_wan.safetensors")

# --- BINARIES & TOOLS ---
# Ensure this path is correct for your Linux setup
LLAMA_BIN = os.path.join(ROOT_DIR, "llama.cpp", "build", "bin", "llama-quantize")

# Scripts inside the /scripts folder
CONVERT_PY = os.path.join(SCRIPTS_DIR, "convert.py")
FIX_5D_PY = os.path.join(ROOT_DIR, "fix_5d_tensors.py")

# --- UI ASSETS (Centralized) ---
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

def ensure_dirs():
    """Initializes the project folder structure."""
    dirs = [MODELS_DIR, LOGS_DIR]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)