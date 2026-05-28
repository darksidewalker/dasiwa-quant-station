# utils/file_ops.py
import os
import gradio as gr
from config import MODELS_DIR, LOGS_DIR
import datetime

def ensure_dirs():
    for d in [MODELS_DIR, LOGS_DIR]:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            print(f"📁 Created directory: {d}")

def list_files():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)
    
    files = sorted([
        f for f in os.listdir(MODELS_DIR) 
        if f.endswith(('.safetensors', '.gguf'))
    ])

    return gr.update(choices=files, value=files[0] if files else None)

def list_dirs(base_dir=None):
    """Return a gr.update for subdirectories under base_dir.

    If base_dir is None, uses MODELS_DIR from config when available.
    """
    from config import MODELS_DIR
    if base_dir is None:
        base_dir = MODELS_DIR

    if not os.path.exists(base_dir):
        return gr.update(choices=[], value=None)

    dirs = sorted([
        os.path.join(base_dir, d) for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])

    # include the base_dir itself as an option
    if base_dir not in dirs:
        dirs.insert(0, base_dir)

    return gr.update(choices=dirs, value=dirs[0] if dirs else base_dir)

def get_full_path(file_name):
    return os.path.join(MODELS_DIR, file_name)

def clean_temp_files(directory):
    for f in os.listdir(directory):
        if f.endswith(".tmp") or f.endswith(".partial"):
            os.remove(os.path.join(directory, f))

def save_log(model_name, content):
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"quant_{model_name}_{timestamp}.log"
    log_path = os.path.join(LOGS_DIR, log_filename)
    
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(content)
    return log_path