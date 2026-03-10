# utils/5d_scanner.py
import os
import torch
from safetensors.torch import load_file

def scan_5d_tensors(file_path):
    """
    Scans a safetensors file for tensors with > 4 dimensions.
    Returns a formatted string for the terminal.
    """
    if not os.path.exists(file_path):
        return f"❌ Error: File not found at {file_path}"

    try:
        state_dict = load_file(file_path)
        file_name = os.path.basename(file_path)
        
        output = [f"🔍 Scanning 5D+ Tensors in: {file_name}", "-" * 50]
        found_any = False
        total_5d_bytes = 0

        for key, tensor in state_dict.items():
            if len(tensor.shape) > 4:
                num_elements = tensor.numel()
                bytes_per_elem = tensor.element_size()
                
                size_mb = (num_elements * bytes_per_elem) / (1024 * 1024)
                total_5d_bytes += (num_elements * bytes_per_elem)

                output.append(f"🎯 KEY: {key}")
                output.append(f"   Shape: {list(tensor.shape)}")
                output.append(f"   Dtype: {tensor.dtype}")
                output.append(f"   Size:  {size_mb:.2f} MB")
                output.append("-" * 25)
                found_any = True

        if not found_any:
            output.append("✅ No 5D tensors found in this model.")
        else:
            total_mb = total_5d_bytes / (1024 * 1024)
            output.append(f"📊 TOTAL 5D STORAGE REQUIRED: {total_mb:.2f} MB")

        return "\n".join(output)

    except Exception as e:
        return f"🔥 Scanning Error: {str(e)}"