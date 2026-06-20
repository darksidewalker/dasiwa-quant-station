# AGENTS.md - Developer & AI Agent Guide

## Project Purpose
**DaSiWa Quant Station** is a specialized quantization and LoRA merge toolkit for video diffusion models. It manages GGUF conversion, safetensors quantization, metadata injection, automatic architecture/full-checkpoint inspection, LoRA merging with architecture-specific tensor classification, and critical preservation of video/companion tensors to prevent corrupted outputs.

Primary targets are WAN 2.2 and LTX-2.3 style video models on NVIDIA Ada/Blackwell systems, while the safetensors path also exposes the broader upstream `convert_to_quant` preset set.

## Current Architecture

### Go Server
- **`cmd/dasiwa/main.go`**: Go entry point. Starts the local web server at `http://127.0.0.1:7878`. Auto-opens browser. Supports `DASIWA_NO_BROWSER` env var.
- **`internal/app/server.go`**: Go HTTP server with:
  - Static frontend hosting (`web/` directory)
  - Job manager with async goroutines, context cancellation, SSE event channels (buffer 512)
  - Idle shutdown tracker (3-minute grace period after zero browser connections, then `os.Exit(0)`)
  - Hardware monitoring endpoint (`/api/system`) - CPU%, RAM, GPU%, VRAM
  - Filesystem browser (`/api/browse`, `/api/files`) - recursive model file discovery
  - Metadata endpoints (preview, read, inject)
  - Quantization and LoRA merge orchestration
  - Memory cleanup endpoint (`/api/memory/clean`) - Go GC, Python GC, malloc_trim, PyTorch/CuPy cache
  - Diagnostic tools (5D scan, pattern audit)
  - Update & restart endpoint

### Frontend
- **`web/`**: Modern HTML/CSS/JS frontend. Vanilla JS, no frameworks.
  - `index.html` - Single-page app with CSS grid layout (320px sidebar + 1fr main)
  - `app.js` - State management, API calls, SSE log streaming, system polling (5s interval)
  - `styles.css` - Dark theme (`--bg: #101214`), responsive breakpoints at 980px

### Python Bridge
- **`scripts/go_bridge.py`**: Python CLI bridge used by the Go server. Dispatches to core engines:
  - `inspect` - architecture detection via `utils/arch_detector.py`
  - `metadata` - modelspec preview generation
  - `read-metadata-path` - read metadata from safetensors/GGUF
  - `inject-metadata-path` - inject metadata into safetensors
  - `quantize` - orchestrates safetensors or GGUF conversion
  - `lora-merge` - architecture-aware LoRA merge pipeline
  - `scan` - 5D tensor validation
  - `audit` - pattern coverage audit
  - `clean-memory` - multi-layer memory release

### Core Engines
- **`config.py`**: Centralized Python path management (`MODELS_DIR`, `LOGS_DIR`, `LLAMA_BIN`, `ROOT_DIR`).
- **`core/gguf_engine.py`**: Orchestrates `ggufy` conversion and sensitivity-map generation for video tensor preservation.
- **`core/safetensors_engine.py`**: Orchestrates `convert_to_quant` conversion. Owns the Python `ARCH_REGISTRY`, format flag mapping, optimizer/simple strategy selection, architecture verification, layer-config attachment, and metadata injection.
- **`core/lora_merge_engine.py`**: Architecture-aware LoRA merge pipeline:
  - Loads base checkpoint and LoRA safetensors files
  - Discovers LoRA A/B pairs via `utils/lora_inspector.py`
  - Matches LoRA tensors to base checkpoint candidates (handles prefix variations: `diffusion_model.` vs `model.diffusion_model.`)
  - Dispatches to architecture-specific profiles (LTX-2.3, WAN 2.2) for tensor classification
  - Applies per-LoRA strategy multipliers with global strength scaling
  - Supports dry-run mode (reports recipe without writing output)
  - Strict matching mode (rejects unmatched LoRA tensors)
  - Preserves structural layers per architecture preserve tables
  - Generates merge recipe for reproducibility
- **`core/layer_config_builder.py`**: Single source of truth for verified preserve/rescue regex tables. WAN 2.2 and LTX-2.3 have local verified patterns; other archs intentionally fall through to upstream `convert_to_quant` preset skip logic.
- **`core/metadata_manager.py`**: Handles modelspec metadata preview/read/injection for safetensors and GGUF where supported.

### Utilities
- **`utils/arch_detector.py`**: Header-only source inspection. Detects known WAN 2.2/LTX-2.3 markers and whether a safetensors source appears to be a full checkpoint with companion modules.
- **`utils/lora_inspector.py`**: LoRA pair discovery. Reads safetensors manifests, identifies A/B weight pairs, infers rank, generates target candidate keys with prefix normalization.
- **`utils/ltx23_layer_profiles.py`**: LTX-2.3 tensor classification into categories (attn_qkv, attn_out, ff_in/out, audio_attn, audio_ff, audio/video connectors, caption_projection, patchify_or_output, norm, other). Strategy multipliers for Balanced/Motion/Visuals/Audio. Preserves adaln, gate logits, baked VAE/text/audio modules.
- **`utils/wan22_layer_profiles.py`**: WAN 2.2 tensor classification into categories (self_attn_qkv/out, cross_attn_qkv/out, ffn_in/out, modulation, caption_projection, patchify_or_output, norm, other). Strategy multipliers for Balanced/Motion/Visuals. No Audio strategy. Preserves modulation.lin, patch_embedding, baked companions. Norm layers always 0.0 multiplier.
- **`utils/scanner_5d.py`**: 5D tensor validation tool.
- **`utils/pattern_audit.py`**: Pattern coverage audit against checkpoint manifest.
- **`utils/system.py`**: CPU/RAM/GPU/VRAM monitoring via nvidia-smi and psutil.
- **`utils/file_ops.py`**: Filesystem utilities (size formatting, path operations).

### Build/Startup
- **`start-linux.sh`**: Preferred launcher. Installs/refreshes system tools, installs `uv`, syncs Python dependencies, refreshes `convert_to_quant`, installs `comfy-kitchen[cublas]`, ensures `bin/ggufy`, builds the Go binary, and launches the Go UI.
- **`build.sh`**: Rebuilds Go binary to `quantstation` without setup.
- **`lcpp.patch`**: Mandatory llama.cpp patch for Wan 2.2 GGUF support.

## Launch And Update Semantics
- `./start-linux.sh` is the normal start path. Performs setup/update work before launching the Go UI.
- `./start-linux.sh --setup-only` performs dependency/binary setup and exits.
- `./quantstation` starts the already-built Go binary directly. Does **not** sync dependencies, update `convert_to_quant`, refresh `ggufy`, or rebuild itself.
- `./build.sh` rebuilds the Go binary to `quantstation` without running setup.
- The Go UI has an **Update & Restart** button. Runs `bash start-linux.sh --setup-only`, rebuilds to `quantstation.next`, replaces `quantstation`, starts a new copy of the binary, then exits the old process.
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
- **Default INT8:** `INT8 Tensor-wise` is the safe Comfy-compatible INT8 path. Maps to `--int8 --scaling_mode tensor --comfy_quant`.
- **ConvRot INT8:** `INT8 Row-wise ConvRot (runtime)` maps to `--int8 --scaling_mode row --convrot --convrot-group-size 256 --comfy_quant`. Use only when the target runtime reads `.comfy_quant` ConvRot metadata and rotates activations during inference.
- If a ConvRot-quantized model is loaded in a runtime that only sees rotated weights and ignores activation rotation metadata, expect severe artifacts such as pixel clutter.
- The stale value `INT8 Row-wise ConvRot` is intentionally treated as a compatibility alias for the safe tensor-wise path.

## Layer Preservation Model
- `PRESERVE_PATTERNS`: Structural/routing/IO layers that stay at source precision via `{"skip": true}`.
- `RESCUE_PATTERNS`: Layers that stay FP8 when the base format is lower-bit (`NVFP4` or `INT8`). On FP8 base they remain normal FP8, not BF16/FP16.
- `BAKED_VAE_PATTERNS`: Unconditional companion-module skip patterns for VAE, audio VAE, vocoder, text encoders, audio encoders, projection layers, and similar full-checkpoint components.
- Verified local pattern tables currently exist for **WAN 2.2** and **LTX-2.3**.
- The 5D scanner and pattern audit tools are diagnostic aids; do not widen their architecture scope without verified patterns.

## LoRA Merge Rules
- Each LoRA can choose its own strategy (Balanced/Motion/Visuals/Audio) independently within a single merge.
- Global strength scales all LoRA contributions uniformly.
- Adaptive mode adjusts per-tensor multipliers based on tensor magnitude.
- Strict matching rejects the merge if any LoRA tensor has no base candidate.
- Dry run mode reports the complete merge recipe (matched/skipped/total counts, per-LoRA strategy, tensor categories) without writing output.
- Preserved keys are never modified, regardless of LoRA content.
- LTX-2.3 has an Audio strategy; WAN 2.2 does not (no audio components).
- Norm layers in WAN 2.2 always get 0.0 multiplier (untouched).

## Common Workflows

### Adding a safetensors format
Update the Go API format list in `internal/app/server.go`, frontend grouping in `web/app.js`/`web/index.html` if needed, and `FLAG_MAP` plus config mapping in `core/safetensors_engine.py`/`core/layer_config_builder.py`.

### Adding a GGUF format
Update the Go API format list and the GGUF mapping in `core/gguf_engine.py`.

### Adding an architecture
Keep names synchronized between `internal/app/server.go`, `core/safetensors_engine.py`, and metadata templates in `core/metadata_configs.py`. Add `arch_detector.py` markers and `layer_config_builder.py` patterns only after real header/tensor verification. Add LoRA merge profiles in `utils/<arch>_layer_profiles.py` with tensor classification and strategy multipliers.

### Changing startup behavior
Edit `start-linux.sh` and the `/api/update` path in `internal/app/server.go` together so CLI launch and in-app update stay equivalent.

### UI tweaks
Primary UI styles live in `web/styles.css`; UI behavior lives in `web/app.js`.

### Debugging
Check `logs/`, browser network/SSE output, and the Go server terminal.

## Testing
- `tests/test_lora_merge_engine.py`: 9 test cases covering LoRA pair discovery, LTX-2.3 profiles, WAN 2.2 profiles, dry run, merge with scaled deltas, preserved key skipping, per-LoRA strategy selection.
- Run with: `python3 -m pytest tests/ -v` or `python3 tests/test_lora_merge_engine.py`

## Agent Rules
- Use `rg`/`rg --files` for search.
- Use `uv` for dependency syncs.
- Use `patch` for manual edits (not `apply_patch`).
- Do not revert user changes or unrelated dirty worktree files.
- Do not prune code unless it is faulty.
- Keep Go UI and Python bridge behavior aligned; the Go app is a shell over the proven Python quantization logic, not a rewrite of the quantizers.
- For long-running quantization or update jobs, preserve streamed output behavior. Go should stream via SSE; Python bridge uses subprocess streaming/yields.
- When adding LoRA merge strategies, update both the profile file and the frontend strategy list in `web/app.js` (`renderLoras()`).
- When modifying preserve patterns, run the pattern audit tool against a real checkpoint to verify coverage before committing.
