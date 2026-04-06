# AGENTS.md - Developer & AI Agent Guide

## 🌀 Project Purpose
**DaSiWa WAN 2.2 Master** is a specialized quantization toolkit for Video Models. It manages complex GGUF conversions, Safetensors quantization, and critical 5D tensor "self-healing" to prevent video corruption.

## 🏗 System Architecture & File Map
- **`app.py`**: Entry point. Initializes folders and launches the Gradio UI.
- **`config.py`**: Centralized path management (`MODELS_DIR`, `LOGS_DIR`, `LLAMA_BIN`).
- **`/core`**: 
    - `gguf_engine.py`: Orchestrates `llama-quantize` and 5D tensor fixing.
    - `safetensors_engine.py`: Interfaces with `convert_to_quant` for safetensors conversion.
    - `metadata_manager.py`: Handles `modelspec` header injection for both GGUF and Safetensors.
- **`/ui`**: 
    - `layout.py`: Visual structure.
    - `callbacks.py`: Event handling and process threading.
    - `assets.py`: CSS styling and metadata templates.
- **`/utils`**:
    - `scanner_5d.py`: Validation tool to verify tensor dimensions.
    - `system.py`: Real-time hardware monitoring (VRAM/CPU).
- **`lcpp.patch`**: A mandatory patch for `llama.cpp` to support Wan 2.2's specific architecture.

## 🛠 Environmental Requirements
- **Package Manager:** Use `uv` for all dependency syncs.
- **Python:** 3.12+
- **Hardware Target:** Optimizations are tailored for NVIDIA Ada (40-series) and Blackwell (50-series).
- **Virtual Env:** Default location is `.venv/` in the project root.

## 🚦 Critical Rules for AI Agents
1. **The 5D Tensor Fix:** In GGUF workflows, `llama-quantize` often flattens tensors. You **must** ensure the `fix_5d_tensors.py` script is called after any GGUF quantization to restore the model's 5D structure.
2. **Metadata Injection:** Never consider a quantization "complete" without calling `metadata_manager.py`. The `modelspec` tags are required for compatibility with downstream tools.
3. **Subprocess Handling:** Use `subprocess.Popen` with `yield` for long-running tasks to keep the Gradio terminal updated. Do not use blocking `subprocess.run` for the main quantization loop.
4. **Path Safety:** Always reference directories via `config.py`. Do not assume the agent's working directory is the root; use absolute paths derived from `ROOT_DIR`.
5. **Patching:** If modifying the `llama.cpp` integration, refer to `lcpp.patch`. Any changes to the build process must be reflected in `start-linux.sh`.

## 🔄 Common Workflows
- **Adding a Quantization Format:** Update the `choices` in `ui/layout.py` and map the flag in the corresponding engine (`gguf_engine.py` or `safetensors_engine.py`).
- **Debugging:** Check `logs/` for session-specific `.log` files.
- **UI Tweaks:** Custom styles are located in `ui/assets.py` under `CSS_STYLE`.

## 🎯 Verification Checklist
- [ ] Do not prune code unless it is faulty.