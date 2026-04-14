# utils/system.py
import psutil
import torch
import platform

def get_sys_info():
    # 1. Physical RAM & CPU
    ram = psutil.virtual_memory().percent
    cpu = psutil.cpu_percent()
    
    # 2. OS Name (Avoid "Lin" by using the full name or a cleaner map)
    os_name = platform.system()
    if os_name == "Windows": os_name = "Win"
    elif os_name == "Linux": os_name = "Linux"
    elif os_name == "Darwin": os_name = "Mac"

    # 3. Swap / Pagefile Logic
    swap = psutil.swap_memory()
    swap_used_gb = swap.used / (1024**3)
    swap_total_gb = swap.total / (1024**3)
    swap_percent = swap.percent
    
    gpu_load = "0%"
    vram_info = "0.0/0.0GB"
    
    # 4. GPU Logic
    if torch.cuda.is_available():
        try:
            # Memory Info
            free_b, total_b = torch.cuda.mem_get_info()
            used_gb = (total_b - free_b) / (1024**3)
            total_gb = total_b / (1024**3)
            vram_info = f"{used_gb:.1f}/{total_gb:.1f}GB"
            
            load_val = torch.cuda.utilization()
            gpu_load = f"{load_val}%"
        except Exception:
            gpu_load = "Active"

    return (
        f"🖥️ CPU: {cpu:>3.0f}% | RAM: {ram:>3.0f}% | OS: {os_name}\n"
        f"🔄 SWAP: {swap_percent:>2.0f}% ({swap_used_gb:.1f}/{swap_total_gb:.1f}GB)\n"
        f"📟 GPU: {gpu_load:>4} | VRAM: {vram_info}"
    )