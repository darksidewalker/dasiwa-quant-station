import psutil, torch, os, subprocess
from safetensors.torch import load_file, save_file
import json
from safetensors import safe_open

def get_sys_info():
    ram = psutil.virtual_memory().percent
    cpu = psutil.cpu_percent()
    gpu_load, vram_info = "0%", "0.0/0.0GB"
    if torch.cuda.is_available():
        try:
            v_used = torch.cuda.memory_reserved() / 1e9
            v_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram_info = f"{v_used:.1f}/{v_total:.1f}GB"
            res = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], encoding='utf-8')
            gpu_load = f"{res.strip()}%"
        except: gpu_load = "ERR%"
    return f"🖥️ CPU: {cpu:>3}% | RAM: {ram:>3}%\n📟 GPU: {gpu_load:>3} | VRAM: {vram_info}"
    
def inject_metadata(file_path, metadata_dict):
    """
    Overwrites the safetensors file with new metadata.
    """
    try:
        # Load existing tensors (as pointers, not loading whole file into RAM)
        tensors = load_file(file_path)
        
        # Ensure all values in metadata are strings (Safetensors requirement)
        clean_metadata = {k: str(v) for k, v in metadata_dict.items()}
        
        # Save back to the same path
        save_file(tensors, file_path, metadata=clean_metadata)
        return True, f"Successfully injected metadata into {os.path.basename(file_path)}"
    except Exception as e:
        return False, str(e)

def get_metadata(file_path):
    """Reads and returns the metadata header from a safetensors file."""
    if not os.path.exists(file_path):
        return None, "File not found."
    try:
        with safe_open(file_path, framework="pt") as f:
            return f.metadata(), None
    except Exception as e:
        return None, str(e)
