# core/metadata_manager.py
import os
import json
import datetime
from ui.assets import MODEL_METADATA_CONFIGS, COMMON_METADATA
from safetensors.torch import load_file, save_file
from safetensors import safe_open

try:
    import gguf
except ImportError:
    gguf = None

def get_current_meta(model_name, architecture, bits="FP8"):
    """
    Assembles the metadata dictionary based on architecture and model name.
    Replaces {model_name}, {date}, and {bits} tokens.
    """
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Get architecture-specific fields (fallback to WAN 2.2 if not found)
    base_config = MODEL_METADATA_CONFIGS.get(architecture, MODEL_METADATA_CONFIGS["WAN 2.2"])
    
    # Merge with common fields (Date, Tool, Bits)
    full_template = {**base_config, **COMMON_METADATA}
    
    # Perform string replacement for all template tokens
    final_meta = {}
    for k, v in full_template.items():
        if isinstance(v, str):
            final_meta[k] = (v.replace("{model_name}", model_name)
                              .replace("{date}", date_str)
                              .replace("{bits}", bits))
        else:
            final_meta[k] = v
            
    return final_meta

def update_metadata_preview(name, architecture="WAN 2.2"):
    """
    Generates a formatted JSON string for the UI preview.
    FIXED: Now accepts two arguments to match layout.py
    """
    # Handle the default UI state or empty strings
    if not name or name in ["Enter Name...", "Model-Name", ""]: 
        name = "TreasureChest"
        
    preview_dict = get_current_meta(name, architecture, bits="SELECTED_QUANT")
    return json.dumps(preview_dict, indent=4)

def inject_metadata(file_path, metadata_dict):
    """Writes the metadata header into a Safetensors file while preserving tensors."""
    try:
        tensors = load_file(file_path)
        # Safetensors metadata must be a dictionary of strings
        clean_metadata = {k: str(v) for k, v in metadata_dict.items()}
        save_file(tensors, file_path, metadata=clean_metadata)
        return True, f"Successfully injected into {os.path.basename(file_path)}"
    except Exception as e:
        return False, f"Injection Error: {str(e)}"

def write_gguf_meta(file_path, model_name, architecture, bits):
    """Appends metadata to a GGUF file using the gguf-py library."""
    if not gguf: 
        return False, "❌ 'gguf' package missing."
    try:
        reader = gguf.GGUFReader(file_path)
        writer = gguf.GGUFWriter(file_path, reader.arch, mode="r+")
        meta = get_current_meta(model_name, architecture, bits)
        
        writer.add_string("general.name", model_name)
        for k, v in meta.items():
            writer.add_string(k, str(v))
            
        writer.write_header_only()
        writer.close()
        return True, "GGUF Meta Injected"
    except Exception as e:
        return False, f"GGUF Error: {str(e)}"

def read_any_metadata(MODELS_DIR, file_name):
    """Reads and returns the header metadata from either Safetensors or GGUF."""
    if not file_name: 
        return "❌ No file selected."
    
    path = os.path.join(MODELS_DIR, file_name)
    if not os.path.exists(path):
        return f"❌ File not found at: {path}"

    if file_name.lower().endswith(".gguf"):
        if not gguf: return "❌ 'gguf' library missing."
        try:
            reader = gguf.GGUFReader(path)
            kv_pairs = [f"{key}: {field.parts[field.data[0]]}" for key, field in reader.fields.items()]
            return f"🔍 GGUF HEADER: {file_name}\n" + "-"*40 + "\n" + "\n".join(kv_pairs)
        except Exception as e:
            return f"🔥 GGUF Read Error: {str(e)}"

    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            meta = f.metadata()
            if meta:
                return f"🔍 SAFETENSORS HEADER: {file_name}\n" + "-"*40 + f"\n{json.dumps(meta, indent=4)}"
            return f"ℹ️ {file_name} has no metadata header."
    except Exception as e:
        return f"🔥 Safetensors Read Error: {str(e)}"