# core/metadata_manager.py
import os, json, datetime
from ui.assets import METADATA_TEMPLATE
from safetensors.torch import load_file, save_file
from safetensors import safe_open

try:
    import gguf
except ImportError:
    gguf = None

def get_current_meta(model_name, bits):
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    return {k: v.replace("{model_name}", model_name).replace("{date}", date).replace("{bits}" , bits) 
            for k, v in METADATA_TEMPLATE.items()}

def write_gguf_meta(file_path, model_name, bits):
    if not gguf: return False, "gguf package missing"
    try:
        reader = gguf.GGUFReader(file_path)
        writer = gguf.GGUFWriter(file_path, reader.arch, mode="r+")
        meta = get_current_meta(model_name, bits)
        
        writer.add_string("general.name", model_name)
        for k, v in meta.items():
            writer.add_string(k, str(v))
            
        writer.write_header_only()
        writer.close()
        return True, "GGUF Meta Injected"
    except Exception as e:
        return False, str(e)

def read_any_metadata(MODELS_DIR, file_name):
    if not file_name: 
        return "❌ No file selected. Please pick a model from the dropdown."
    
    path = os.path.join(MODELS_DIR, file_name)
    if not os.path.exists(path):
        return f"❌ File not found at: {path}"

    if file_name.lower().endswith(".gguf"):
        if not gguf: 
            return "❌ 'gguf' library missing."
        try:
            reader = gguf.GGUFReader(path)
            kv_pairs = []
            for key, field in reader.fields.items():
                try:
                    if field.data:
                        raw_val = field.parts[field.data[0]]
                        val = raw_val.decode('utf-8', errors='ignore') if isinstance(raw_val, bytes) else str(raw_val)
                        kv_pairs.append(f"{key}: {val}")
                except Exception:
                    continue
            
            return f"🔍 GGUF HEADER: {file_name}\n" + "-"*40 + "\n" + "\n".join(kv_pairs)
        except Exception as e:
            return f"🔥 GGUF Read Error: {str(e)}"

    # --- 2. HANDLE SAFETENSORS FILES ---
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            meta = f.metadata()
            if meta:
                return f"🔍 SAFETENSORS HEADER: {file_name}\n" + "-"*40 + f"\n{json.dumps(meta, indent=4)}"
            else:
                return f"ℹ️ {file_name} has no metadata header (it is 'clean')."
    except Exception as e:
        return f"🔥 Safetensors Read Error: {str(e)}"
    
    return "❌ Unsupported file format for metadata reading."

def update_metadata_preview(name):
    if not name or name == "Enter Name...": name = "Model-Name"
    curr_date = datetime.datetime.now().strftime("%Y-%m-%d")
    preview = {
        k: v.replace("{model_name}", name).replace("{date}", curr_date).replace("{bits}", "SELECTED_QUANT") 
        for k, v in METADATA_TEMPLATE.items()
    }
    return json.dumps(preview, indent=4)

def inject_metadata(file_path, metadata_dict):
    try:
        tensors = load_file(file_path)
        clean_metadata = {k: str(v) for k, v in metadata_dict.items()}
        save_file(tensors, file_path, metadata=clean_metadata)
        return True, f"Successfully injected into {os.path.basename(file_path)}"
    except Exception as e:
        return False, str(e)