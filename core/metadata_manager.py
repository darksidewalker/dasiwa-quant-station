import os
import json
import datetime
import hashlib
from ui.assets import MODEL_METADATA_CONFIGS, COMMON_METADATA
from safetensors.torch import load_file, save_file
from safetensors import safe_open

try:
    import gguf
except ImportError:
    gguf = None

def calculate_sha256(file_path):
    """Calculates a clean 0x-prefixed SHA256 hash of the target file."""
    if not os.path.exists(file_path) or file_path == "PREVIEW_MODE":
        return "0x[HASH_WILL_BE_CALCULATED_ON_SAVE]"
    
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in 64kb chunks for memory efficiency
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return f"0x{sha256_hash.hexdigest()}"

def get_current_meta(model_name, architecture, bits="FP8"):
    """
    Standard asset-based metadata template (Fallback logic).
    Used if no specialized JSON dump is found in the core folder.
    """
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Get architecture-specific fields (fallback to WAN 2.2 if not found)
    base_config = MODEL_METADATA_CONFIGS.get(architecture, MODEL_METADATA_CONFIGS.get("WAN 2.2"))
    
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

def get_specialized_meta(architecture, model_name, final_file_path, bits="FP8"):
    """
    PRIORITY 1: Loads the FULL content of {Arch}_metadata.json from /core.
    PRIORITY 2: Falls back to asset-based template.
    """
    core_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Cleaning logic: "LTX-2.3" -> "LTX23", "WAN 2.2" -> "WAN22"
    clean_arch = architecture.replace("-", "").replace(".", "").replace(" ", "")
    seed_filename = f"{clean_arch}_metadata.json"
    seed_path = os.path.join(core_dir, seed_filename)

    if os.path.exists(seed_path):
        try:
            with open(seed_path, 'r', encoding='utf-8') as f:
                # 1. Load the FULL dump (License, Architecture, Resolution, etc.)
                meta = json.load(f)
            
            # 2. OVERWRITE ONLY our specific UI/Session fields
            meta["modelspec.title"] = model_name 
            meta["modelspec.hash_sha256"] = calculate_sha256(final_file_path)
            meta["modelspec.date"] = datetime.datetime.now().strftime("%Y-%m-%d")
            meta["quantization.bits"] = bits
            meta["quantization.tool"] = "https://github.com/darksidewalker/dasiwa-quant-station"
            
            # Ensure the spacer exists for future header edits
            if "__spacer" not in meta:
                meta["__spacer"] = " " * 2048
            
            return meta
        except Exception as e:
            print(f"❌ Error merging {seed_filename}: {e}")

    # Fallback to standard assets.py config if JSON seed is missing
    return get_current_meta(model_name, architecture, bits)

def update_metadata_preview(name, architecture="WAN 2.2"):
    """Called by UI to generate the preview for the Gradio JSON box."""
    # We pass a placeholder path so SHA256 returns the "WILL BE CALCULATED" string
    meta = get_specialized_meta(architecture, name, "PREVIEW_MODE")
    return json.dumps(meta, indent=4)

def inject_metadata(file_path, meta_dict):
    """Writes metadata dictionary into a Safetensors file header."""
    try:
        tensors = load_file(file_path)
        save_file(tensors, file_path, metadata=meta_dict)
        return True, "Metadata Injected Successfully"
    except Exception as e:
        return False, str(e)

def write_gguf_meta(file_path, model_name, architecture, bits="FP8"):
    """
    Handles metadata for GGUF files.
    Matches the function name required by core/gguf_engine.py.
    """
    if not gguf:
        return False, "gguf library not installed"
    try:
        writer = gguf.GGUFWriter(file_path, architecture)
        meta = get_specialized_meta(architecture, model_name, file_path, bits)
        
        writer.add_string("general.name", model_name)
        for k, v in meta.items():
            writer.add_string(k, str(v))
            
        writer.write_header_only()
        writer.close()
        return True, "GGUF Meta Injected"
    except Exception as e:
        return False, f"GGUF Error: {str(e)}"

def read_any_metadata(MODELS_DIR, file_name):
    """Reads and returns the header metadata from Safetensors or GGUF for the terminal."""
    if not file_name: 
        return "❌ No file selected."
    
    path = os.path.join(MODELS_DIR, file_name)
    if not os.path.exists(path):
        return f"❌ File not found at: {path}"

    # GGUF Reading Logic
    if file_name.lower().endswith(".gguf"):
        if not gguf: return "❌ 'gguf' library missing."
        try:
            reader = gguf.GGUFReader(path)
            kv_pairs = [f"{key}: {field.parts[field.data[0]]}" for key, field in reader.fields.items()]
            return f"🔍 GGUF HEADER: {file_name}\n" + "-"*40 + "\n" + "\n".join(kv_pairs)
        except Exception as e:
            return f"🔥 GGUF Read Error: {str(e)}"

    # Safetensors Reading Logic
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            meta = f.metadata()
            if not meta:
                return f"🔍 HEADER: {file_name}\n" + "-"*40 + "\nEmpty or no metadata found."
            return f"🔍 SAFETENSORS HEADER: {file_name}\n" + "-"*40 + "\n" + json.dumps(meta, indent=4)
    except Exception as e:
        return f"🔥 Read Error: {str(e)}"