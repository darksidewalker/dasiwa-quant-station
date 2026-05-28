# AGENTS.md - Developer & AI Agent Guide

## 🌀 Project Purpose
**DaSiWa WAN 2.2 Master** is a specialized quantization toolkit for Video Models. It manages complex GGUF conversions, Safetensors quantization, and critical 5D tensor "self-healing" to prevent video corruption.

## 🏗 System Architecture & File Map
- **`app.py`**: Entry point. Initializes folders and launches the Gradio UI.
- **`config.py`**: Centralized path management (`MODELS_DIR`, `LOGS_DIR`, `LLAMA_BIN`).
- **`/core`**: 
    - `gguf_engine.py`: Orchestrates `ggufy` conversion. Handles metadata architecture mapping and custom sensitivity map generation for video-tensor preservation.
    - `safetensors_engine.py`: Interfaces with `convert_to_quant` for safetensors conversion. Owns the `ARCH_REGISTRY` (single source of truth for the architecture dropdown: CLI flag + ultra-quality optimizer params per arch, including the `"Not set"` no-preset entry).
    - `layer_config_builder.py`: Per-arch regex patterns for sensitive-layer preservation. Only WAN 2.2 and LTX-2.3 have verified patterns; other archs in the registry fall through cleanly and rely on the upstream `convert_to_quant` preset's own skip rules.
    - `metadata_manager.py`: Handles `modelspec` header injection for both GGUF and Safetensors.
- **`/ui`**: 
    - `layout.py`: Visual structure. The Architecture dropdown values must exactly match `ARCH_REGISTRY` keys in `safetensors_engine.py`. The `Model Directory` control is now a directory-dropdown that lists `MODELS_DIR` and its subfolders (see `utils/file_ops.list_dirs`).
    - `callbacks.py`: Event handling and process threading; callbacks now expect a directory-dropdown (`models_dir_dd`) and refresh file lists accordingly.
    - `assets.py`: CSS styling and `MODEL_METADATA_CONFIGS` (per-arch modelspec templates; keys must match `ARCH_REGISTRY`).
- **`/utils`**:
    - `arch_detector.py`: Source-file architecture verification. Only WAN 2.2 and LTX-2.3 have marker patterns; other archs fall through as UNKNOWN with a warning (the engine still runs).
    - `scanner_5d.py`: Validation tool to verify tensor dimensions.
    - `pattern_audit.py`, `keeplist_compare.py`, `exact_config.py`: WAN-2.2/LTX-2.3-only diagnostic tools (they hard-error on archs without patterns).
    - `system.py`: Real-time hardware monitoring (VRAM/CPU).
- **`lcpp.patch`**: A mandatory patch for `llama.cpp` to support Wan 2.2's specific architecture.

## 🛠 Environmental Requirements
- **Package Manager:** Use `uv` for all dependency syncs.
- **Python:** 3.12+
- **Hardware Target:** Optimizations are tailored for NVIDIA Ada (40-series) and Blackwell (50-series).
- **Virtual Env:** Default location is `.venv/` in the project root.

## 🚦 Critical Rules for AI Agents
1. **GGUFY Conversion:** GGUF quantization is handled via the `ggufy` binary. It preserves 5D tensor structures by using generated sensitivity maps that assign high scores (100) to structural layers.
2. **Metadata Injection:** Never consider a quantization "complete" without calling `metadata_manager.py`. The `modelspec` tags are required for compatibility with downstream tools.
3. **Subprocess Handling:** Use `subprocess.Popen` with `yield` for long-running tasks to keep the Gradio terminal updated. Do not use blocking `subprocess.run` for the main quantization loop.
4. **Path Safety:** Always reference directories via `config.py`. Do not assume the agent's working directory is the root; use absolute paths derived from `ROOT_DIR`.
5. **GGUFY Binary:** The engine expects `ggufy` at `bin/ggufy`. Ensure `start-linux.sh` is used to maintain the correct binary for the system architecture.

## 🔄 Common Workflows
- **Adding a Quantization Format:** Update the `choices` in `ui/layout.py` (`q_format` CheckboxGroup) and map the flag in the `FLAG_MAP` dict at the top of `core/safetensors_engine.py` (for safetensors formats) or in `gguf_engine.py` (for GGUF).
- **Adding an Architecture (safetensors path):** Add one entry to `ARCH_REGISTRY` in `core/safetensors_engine.py` (`{"flag": "--your_flag", "ultra": _ULTRA_DEFAULT}`), one matching entry in the `model_type` Dropdown `choices` in `ui/layout.py`, and one matching entry in `MODEL_METADATA_CONFIGS` in `ui/assets.py`. The three sets of keys must agree exactly. Layer-config patterns (`layer_config_builder.py`) and arch markers (`arch_detector.py`) are optional - omit them and the engine falls through cleanly, relying on the upstream `convert_to_quant` preset.
- **"Not set" semantics:** Selecting `Not set` in the Architecture dropdown skips three things: architecture-flag append, layer-config building (regex and exact), and source-file architecture verification. The command guard accepts zero arch flags only in this mode.
- **Debugging:** Check `logs/` for session-specific `.log` files.
- **UI Tweaks:** Custom styles are located in `ui/assets.py` under `CSS_STYLE`.

## 🎯 Verification Checklist
- [ ] Do not prune code unless it is faulty.