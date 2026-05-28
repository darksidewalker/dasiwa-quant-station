import os

def get_model_list(models_dir, extensions=None):
    """
    Recursively scans models_dir for files with specific extensions.
    Returns a sorted list of relative paths.

    Args:
        models_dir: Base directory to scan.
        extensions: Set of file extensions to include (default: safetensors, ckpt, pt, bin, gguf).
    """
    if extensions is None:
        extensions = {".safetensors", ".ckpt", ".pt", ".bin", ".gguf"}
    
    found_files = []
    if not models_dir or not os.path.isdir(models_dir):
        return found_files

    for root, _, files in os.walk(models_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                rel_path = os.path.relpath(os.path.join(root, file), models_dir)
                found_files.append(rel_path)
    
    return sorted(found_files)