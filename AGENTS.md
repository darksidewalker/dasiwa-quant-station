# AGENTS.md - Developer & AI Agent Guide

## Project Purpose
**DaSiWa Quant Station** is a specialized quantization toolkit for video diffusion models. It manages GGUF conversion, safetensors quantization, metadata injection, automatic architecture/full-checkpoint inspection, and critical preservation of video/companion tensors to prevent corrupted outputs.

Primary targets are WAN 2.2 and LTX-2.3 style video models on NVIDIA Ada/Blackwell systems, while the safetensors path also exposes the broader upstream `convert_to_quant` preset set.

## Current Architecture
- **`cmd/dasiwa/main.go`**: Go entry point. Starts the local web server at `http://127.0.0.1:7878`.
- **`internal/app/server.go`**: Go HTTP server, static frontend host, job manager, SSE log streaming, hardware endpoint, filesystem browser, metadata endpoints, quantization/update orchestration.
- **`web/`**: Modern HTML/CSS/JS frontend. This is the primary UI.
- **`scripts/go_bridge.py`**: Python bridge used by the Go server. Calls the existing Python quantization, metadata, inspection, scan, and audit code.
- **`app.py`**: Legacy Gradio entry point. Still available through `./start-linux.sh --gradio`, but no longer the primary UI.
- **`config.py`**: Centralized Python path management (`MODELS_DIR`, `LOGS_DIR`, `LLAMA_BIN`, `ROOT_DIR`).
- **`core/gguf_engine.py`**: Orchestrates `ggufy` conversion and sensitivity-map generation for video tensor preservation.
- **`core/safetensors_engine.py`**: Orchestrates `convert_to_quant` conversion. Owns the Python `ARCH_REGISTRY`, format flag mapping, optimizer/simple strategy selection, architecture verification, layer-config attachment, and metadata injection.
- **`core/layer_config_builder.py`**: Single source of truth for verified preserve/rescue regex tables. WAN 2.2 and LTX-2.3 have local verified patterns; other archs intentionally fall through to upstream `convert_to_quant` preset skip logic.
- **`core/metadata_manager.py`**: Handles modelspec metadata preview/read/injection for safetensors and GGUF where supported.
- **`utils/arch_detector.py`**: Header-only source inspection. Detects known WAN 2.2/LTX-2.3 markers and whether a safetensors source appears to be a full checkpoint with companion modules.
- **`utils/scanner_5d.py`**: 5D tensor validation tool.
- **`utils/pattern_audit.py`, `utils/keeplist_compare.py`, `utils/exact_config.py`**: WAN 2.2/LTX-2.3 diagnostics and exact/reference config tooling.
- **`start-linux.sh`**: Preferred launcher. Installs/refreshes system tools where possible, installs `uv`, syncs Python dependencies, refreshes `convert_to_quant`, installs `comfy-kitchen[cublas]`, ensures `bin/ggufy`, builds the Go binary, and launches the Go UI.
- **`lcpp.patch`**: Mandatory llama.cpp patch for Wan 2.2 GGUF support.

## Launch And Update Semantics
- `./start-linux.sh` is the normal start path. It performs setup/update work before launching the Go UI.
- `./start-linux.sh --setup-only` performs dependency/binary setup and exits.
- `./start-linux.sh --gradio` launches the legacy Gradio UI.
- `./quantstation` starts the already-built Go binary directly. It does **not** sync dependencies, update `convert_to_quant`, refresh `ggufy`, or rebuild itself.
- `./build.sh` rebuilds the Go binary to `quantstation` without running setup.
- The Go UI has an **Update & Restart** button. It runs `bash start-linux.sh --setup-only`, rebuilds to `quantstation.next`, replaces `quantstation`, starts a new copy of the binary, then exits the old process.
- The in-app updater may call `sudo` through the startup script if system packages are missing. It streams logs, but browser UIs cannot safely answer password prompts; launch from a terminal or pre-cache sudo when needed.

## Environmental Requirements
- **Package manager:** Use `uv` for all Python dependency syncs.
- **Python:** 3.12+.
- **Go:** Required for the primary UI build.
- **Virtual env:** `.venv/` in the project root.
- **GGUFY binary:** Expected at `bin/ggufy`; maintained by `start-linux.sh`.
- **Hardware target:** NVIDIA Ada (40-series) and Blackwell (50-series), with CUDA available for the relevant quantizers.

## Quantization Rules
1. **GGUF path:** GGUF quantization is handled through `bin/ggufy`. Video structural tensors are protected by generated sensitivity maps that assign high scores to preserve critical layers.
2. **Safetensors path:** Safetensors quantization is handled through upstream `convert_to_quant`. The app adds architecture flags, optional optimizer-driven flags, and local layer configs only when appropriate.
3. **Metadata is required:** Do not consider a quantization complete without modelspec metadata. The Go UI exposes read/inject tools, and conversion flows should call `metadata_manager.py`.
4. **Architecture selection matters:** Selecting the wrong preset can quantize structural layers that should be preserved. The UI attempts header-only detection and allows manual override.
5. **Full checkpoints:** If a safetensors source contains VAE/text/audio companion modules, the detector marks it as a full checkpoint. Baked companion patterns are skipped/preserved by `layer_config_builder.py` when local layer configs are active.
6. **Unverified archs:** For architectures without local preserve/rescue patterns, do not invent tables casually. Let upstream `convert_to_quant` preset logic handle skip rules unless verified against real tensor names and output behavior.

## INT8 And ConvRot Guidance
- **Default INT8:** `INT8 Tensor-wise` is the safe Comfy-compatible INT8 path. It maps to `--int8 --scaling_mode tensor --comfy_quant`.
- **ConvRot INT8:** `INT8 Row-wise ConvRot (runtime)` maps to `--int8 --scaling_mode row --convrot --convrot-group-size 256 --comfy_quant`. Use only when the target runtime reads `.comfy_quant` ConvRot metadata and rotates activations during inference.
- If a ConvRot-quantized model is loaded in a runtime that only sees rotated weights and ignores activation rotation metadata, expect severe artifacts such as pixel clutter.
- The stale value `INT8 Row-wise ConvRot` is intentionally treated as a compatibility alias for the safe tensor-wise path.

## Layer Preservation Model
- `PRESERVE_PATTERNS`: Structural/routing/IO layers that stay at source precision via `{"skip": true}`.
- `RESCUE_PATTERNS`: Layers that stay FP8 when the base format is lower-bit (`NVFP4` or `INT8`). On FP8 base they remain normal FP8, not BF16/FP16.
- `BAKED_VAE_PATTERNS`: Unconditional companion-module skip patterns for VAE, audio VAE, vocoder, text encoders, audio encoders, projection layers, and similar full-checkpoint components.
- Verified local pattern tables currently exist for **WAN 2.2** and **LTX-2.3**.
- The 5D scanner and pattern audit tools are diagnostic aids; do not widen their architecture scope without verified patterns.

## Common Workflows
- **Adding a safetensors format:** Update the Go API format list in `internal/app/server.go`, frontend grouping in `web/app.js`/`web/index.html` if needed, and `FLAG_MAP` plus config mapping in `core/safetensors_engine.py`/`core/layer_config_builder.py`.
- **Adding a GGUF format:** Update the Go API format list and the GGUF mapping in `core/gguf_engine.py`.
- **Adding an architecture:** Keep names synchronized between `internal/app/server.go`, `core/safetensors_engine.py`, and metadata templates in `ui/assets.py`/metadata generation code. Add `arch_detector.py` markers and `layer_config_builder.py` patterns only after real header/tensor verification.
- **Changing startup behavior:** Edit `start-linux.sh` and the `/api/update` path in `internal/app/server.go` together so CLI launch and in-app update stay equivalent.
- **UI tweaks:** Primary UI styles live in `web/styles.css`; UI behavior lives in `web/app.js`. Legacy Gradio styles live in `ui/assets.py`.
- **Debugging:** Check `logs/`, browser network/SSE output, and the Go server terminal.

## Agent Rules
- Use `rg`/`rg --files` for search.
- Use `uv` for dependency syncs.
- Use `apply_patch` for manual edits.
- Do not revert user changes or unrelated dirty worktree files.
- Do not prune code unless it is faulty.
- Keep Go UI and Python bridge behavior aligned; the Go app is a shell over the proven Python quantization logic, not a rewrite of the quantizers.
- For long-running quantization or update jobs, preserve streamed output behavior. Go should stream via SSE; Python/legacy Gradio should use subprocess streaming/yields.
