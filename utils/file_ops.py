# utils/file_ops.py
import os
from config import MODELS_DIR, LOGS_DIR
import datetime

def ensure_dirs():
    for d in [MODELS_DIR, LOGS_DIR]:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            print(f"📁 Created directory: {d}")

def save_log(model_name, content):
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"quant_{model_name}_{timestamp}.log"
    log_path = os.path.join(LOGS_DIR, log_filename)
    
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(content)
    return log_path
