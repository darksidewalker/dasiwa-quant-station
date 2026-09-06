# DaSiWa Quant Station

DaSiWa Quant Station is a local quantization and LoRA merge workbench for video and image diffusion models. It combines a modern Go-powered web UI with proven Python quantization engines for GGUF and safetensors workflows, plus architecture-aware LoRA merging with per-LoRA strategy control.

Built around the practical pain points of WAN 2.2, LTX-2.3, Krea 2, and MiniMax H3 diffusion checkpoints: preserving 5D/video-critical tensors, keeping full-checkpoint companion modules safe, injecting modelspec metadata, avoiding INT8 paths that produce corrupted video, and merging LoRAs with architecture-specific tensor classification. Quant outputs carry an EC-based, only-you-can-decode `modelspec.watermark` provenance token.

![Quant Station Preview](assets/DaSiWa-QuantStation.webp)

---

## Table of Contents

- [At a Glance](#at-a-glance)
- [Quick Start](#quick-start)
- [Choosing Formats](#choosing-formats) (incl. [Support Matrix](#support-matrix))
- [Quantization Workflow](#quantization-workflow)
- [MiniMax H3 NVFP4 HQ mixed profile](#minimax-h3-nvfp4-hq-mixed-profile)
- [LoRA Merge Workflow](#lora-merge-workflow)
- [Model Merge (model-level)](#model-merge-model-level)
- [Architecture And Preservation](#architecture-and-preservation)
- [Diagnostic Tools](#diagnostic-tools)
- [UI Features](#ui-features)
- [Update & Restart](#update--restart)
- [Provenance Watermark](#provenance-watermark)
- [Project Layout](#project-layout)
- [API Endpoints](#api-endpoints)
- [Credits](#credits)

Detailed reference material lives in the [doc/](doc/) folder — linked from each section below.

---

## At a Glance

| Area | What it does |
|------|-------------|
| Safetensors quantization | FP8, NVFP4, **NVFP4 HQ** (H3 per-block mixed profile), MXFP8, Hybrid MXFP8, INT8 Tensor-wise, INT8 Row-wise ConvRot Runtime, **INT4 ConvRot** and **W4A8 (asym_w4a8_int8)** via `silveroxides/convert_to_quant` and `comfy-kitchen` |
| GGUF conversion | F32/BF16/F16/Q8_0/Q6_K/Q5_K/Q4_K/Q3_K/Q2_K via `ggufy` with sensitivity maps for video tensor preservation |
| LoRA merge | Architecture-aware merging for WAN 2.2, LTX-2.3, and Krea 2 with per-LoRA strength, global scaling, dry-run, strict matching, adaptive mode, `.diff` format support, and Krea 2 unchain |
| Layer safety | Verified preserve/rescue tables for WAN 2.2, LTX-2.3, and Krea 2. Baked VAE/text/audio companion preservation for full checkpoints |
| Metadata tools | Preview, read, inject modelspec metadata. In-place header rewrite (avoids GB-scale full rewrites). EC-based `modelspec.watermark` provenance (only-you-can-decode). Stale source hashes (`civitai.hash.*`, `modelspec.hash_sha256`) are dropped from generated outputs — they describe the source checkpoint, not the quant/merge result. **Loader metadata preservation** — per-run checkbox (on by default) that keeps loader-critical `__metadata__` (config, architecture, implementation, runtime quant layout) from the source checkpoint through quantization, LoRA merge, and model merge; the generated layout always wins over stale source layout, and unsupported quantized merge sources fail closed |
| Diagnostics | 5D tensor scanner, pattern audit, LoRA shape-mismatch detection with ratio analysis |
| Hardware monitor | Real-time CPU%, RAM, GPU%, VRAM bars with 5-second polling |
| Memory cleanup | One-button RAM/VRAM cache release (Go GC, Python GC, malloc_trim, PyTorch/CuPy cache) |
| Convenience | User-selectable output folder, file browser search, settings persistence, recipe reload, backend status indicator, Quit button |

---

## Quick Start

```bash
chmod +x start-linux.sh
./start-linux.sh
```

Then open:

```text
http://127.0.0.1:7878
```

The startup script handles everything: it installs/refreshes build tools, Python dependencies (`uv` + `.venv/`), `convert_to_quant`, `comfy-kitchen[cublas]` (required for INT4 ConvRot and W4A8), and the `bin/ggufy` binary, then builds and starts the Go UI.

### Launch Modes

```bash
./start-linux.sh              # Full setup/update, build, launch
./start-linux.sh --setup-only # Setup only, no launch
./quantstation                # Run already-built binary directly
./build.sh                    # Rebuild Go binary without setup
```

Starting `./quantstation` directly skips dependency updates and rebuilds — use when you just want to launch fast.

Models are loaded from `$DASIWA_MODELS_DIR` (if set), `~/models`, or `<project-root>/models`.

---

## Choosing Formats

| Format | Best Use |
|--------|----------|
| FP8 | RTX 40/50-series quality baseline |
| NVFP4 | Blackwell VRAM savings |
| NVFP4 HQ | MiniMax H3 VRAM + quality balance (mixed NVFP4 profile) |
| MXFP8 | Blackwell microscaling (SM >= 10.0 required) |
| Hybrid MXFP8 | Ada + Blackwell compatibility |
| INT4 ConvRot | Maximum compression (LTX-2.3, WAN 2.2, Krea 2, MiniMax H3) |
| W4A8 | MiniMax H3 reference low-bit (asym_w4a8_int8) |
| INT8 Tensor-wise | Safer INT8 path for broad ComfyUI compatibility |
| INT8 Row-wise ConvRot Runtime | Runtime-specific INT8 (requires matching activation rotation) |
| GGUF Q formats | llama.cpp-style deployment |

Full per-format details (notes, strategy/source requirements) are in [doc/choosing-formats.md](doc/choosing-formats.md).

### Support Matrix

An architecture marked with 🔒 has locally verified preserve/rescue tables. Others rely on upstream `convert_to_quant` preset skip rules. GGUF always applies sensitivity maps for 🔒 architectures; Q1_0 is disabled.

| Architecture | FP8 | NVFP4 | NVFP4 HQ | MXFP8 | Hybrid MXFP8 | INT4 ConvRot | W4A8 | INT8 Tensor-wise | INT8 ConvRot RT | GGUF |
|-------------|:---:|:-----:|:--------:|:-----:|:------------:|:------------:|:----:|:----------------:|:---------------:|:----:|
| WAN 2.2 🔒 | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| LTX-2.3 🔒 | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| Krea 2 🔒 | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| MiniMax H3 🔒 | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; |
| Flux.2 | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x274C; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| Hunyuan Video | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x274C; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| Qwen Image | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x274C; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| Z-Image | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x274C; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| Z-Image Refiner | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x274C; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| Anima | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x274C; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| Radiance | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x274C; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| Distillation Large | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x274C; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| Distillation Small | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x274C; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| NeRF Large | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x274C; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| NeRF Small | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x274C; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |

If an INT8 model produces pixel clutter, first try **INT8 Tensor-wise**. W4A8 and NVFP4 HQ are MiniMax H3 only.

---

## Quantization Workflow

1. Pick a model folder or file with the file browser buttons
2. Select a safetensors checkpoint or GGUF source
3. The app inspects the source automatically; adjust architecture/full-checkpoint if needed
4. Choose formats from the grouped safetensors/GGUF chips (hover for explanations)
5. Pick **Optimizer-driven** for per-layer learned-rounding defaults, or **Simple** for uniform quantization
6. Toggle **Low VRAM** if GPU memory is constrained
7. Start the batch and watch the streamed log
8. Use Metadata tools to read/inject modelspec metadata when needed

---

### MiniMax H3 NVFP4 HQ mixed profile

NVFP4 HQ is a quality-boosted NVFP4 variant, available for MiniMax H3 only. It runs the same dedicated command (`--nvfp4 --comfy_quant`); the entire quality difference comes from the layer config, which keeps a verified per-block subset of the heavy linears at source precision instead of packing all 200 main-matrix layers:

- 27× `attn.out_proj` kept BF16 at blocks 0-15, 17, 19, 20, 27, 38, 43-47, 49
- 3× `mlp.fc2` kept BF16 at blocks 39, 45, 49

The 30-layer plan is a single source of truth in `core/layer_config_builder.py` (`H3_NVFP4_HQ_LAYER_PLAN` and related exports). It was derived from a deep analysis of working community NVFP4 quants of MiniMax H3 — see the upstream analysis repos listed under [Credits](#credits).

The pattern audit detects which profile an H3 NVFP4 file actually uses — `nvfp4_pure`, `nvfp4_hq_mixed`, `nvfp4_fp8_adaln_mixed`, or `nvfp4_mixed_unknown` — and reports mixed retention in a dedicated **MIXED-KEPT** section as a recognized variant, not a pattern miss. H3's heavy linears (incl. `fc2`) are excluded from the suspicious check, so intentionally-BF16 layers are never falsely flagged.

---

## LoRA Merge Workflow

1. Switch to **LoRA Merge** mode in the topbar mode switch (Quantize | LoRA Merge | Model Merge)
2. Select a base checkpoint in the Source panel
3. Add one or more LoRAs (multi-select, drag-and-drop supported). Each LoRA gets its own strategy type and independent strength multiplier
4. Set global strength scaling, toggle adaptive mode or strict matching
5. For Krea 2: toggle **Unchain** to negate `txtfusion.projector.weight` positions 8-10
6. Use **Dry run** to preview the merge recipe without writing output
7. Shape-mismatch diagnostics automatically detect LoRAs trained on different hidden dimensions and warn before merge
8. The **Display & Output Name** field in the Source panel sets the merged output filename (shared across all merge modes). Start the merge from the sidebar **Start Merge** button

Per-architecture strategy presets (LTX-2.3 All/Video/Audio, WAN 2.2 Balanced/Motion/Visuals, Krea 2 Balanced/Style/Content/Detail), supported LoRA formats, and recipe reload are documented in [doc/lora-merge-strategies.md](doc/lora-merge-strategies.md).

---

## Model Merge (model-level)

A model-level merge — not LoRA math. The current recipe is the **Hybrid MiniMax H3** merge (fl2va base + ref2va overlay, roles auto-detected). Full details: [doc/model-merge.md](doc/model-merge.md).

---

## Architecture And Preservation

The architecture selection controls the `convert_to_quant` preset and, for verified models, DaSiWa's local preservation table (WAN 2.2, LTX-2.3, Krea 2, MiniMax H3). Unverified architectures fall back to upstream preset skip rules. The layer preservation model — **PRESERVE_PATTERNS** (skip), **RESCUE_PATTERNS** (FP8 bump on lower-bit bases), **BAKED_VAE_PATTERNS** (unconditional companion-module skip) — is explained in [doc/architecture-and-preservation.md](doc/architecture-and-preservation.md).

---

## Diagnostic Tools

- **5D Scanner:** Validate 5D tensor shapes in safetensors files. Detects video structural tensors and reports dimension anomalies
- **Pattern Audit:** Compare preserve/rescue pattern coverage against a checkpoint's tensor manifest. Shows matched/missed categories. For MiniMax H3 NVFP4 files it also detects the quant profile (`nvfp4_pure`, `nvfp4_hq_mixed`, `nvfp4_fp8_adaln_mixed`, `nvfp4_mixed_unknown`) and reports intentional mixed retention in a MIXED-KEPT section without false-flagging the kept heavy linears
- **Memory Cleanup:** One-button release of Go runtime memory, Python GC, libc malloc_trim, PyTorch CUDA cache, and CuPy cache
- **LoRA Shape Diagnostics:** Automatic detection of systematic shape mismatches (e.g., LoRA trained on different hidden dimensions) with ratio analysis and actionable warnings

---

## UI Features

- Dark theme, responsive layout, no heavy frameworks
- Backend status indicator showing service health
- File browser with recursive search
- Hover tooltips on format chips explaining each quantization type
- Settings persistence (remembers last choices across sessions via browser cookies)
- Last-used checkpoint folder remembered between runs
- Multi-select LoRA browser with drag-and-drop
- "Preserve loader metadata" checkbox (on by default) in Quantize, LoRA Merge, and Model Merge — keeps the loader-critical metadata structure (config, architecture, runtime quant layout) from the source checkpoint in every output; unchecking restores a clean metadata set with only the loader-critical keys
- Custom output directory selector
- Quit button for clean server shutdown
- Tab-aware keepalive (server stays alive while browser tab is open)

---

## Update & Restart

In-app **Update & Restart**: pulls latest source from origin/main, refreshes dependencies, rebuilds the Go binary, and cleanly restarts the server. If your system needs `sudo` for missing packages, run from a terminal — the browser cannot securely answer password prompts.

---

## Provenance Watermark

Every quantized and LoRA-merged output carries an EC-based provenance token in the `modelspec.watermark` field: ephemeral X25519 (ECIES) wrapping AES-256-GCM, derived from your local passphrase — unique per output, decodable only by you. Without a configured secret the field is simply not written.

- Scheme, payload, decode semantics: [doc/watermark.md](doc/watermark.md)
- Secret resolution: `DASIWA_WATERMARK_PASSPHRASE` → `DASIWA_WATERMARK_KEY` → `~/.dasiwa/watermark.key` (0600, outside the repo)
- UI: **Watermark outputs** checkbox (on by default) with live key-status hint; per-job kill switch when unchecked
- Status endpoint: `GET /api/watermark` (never returns the secret value)

```bash
python scripts/go_bridge.py watermark-key --passphrase "your-passphrase"  # persist key (0600)
python scripts/go_bridge.py watermark path/to/output.safetensors         # decode token
python scripts/go_bridge.py watermark-status                             # key available?
```

---

## Project Layout

The full annotated file tree is in [doc/project-layout.md](doc/project-layout.md). Short version: `cmd/quantstation` (Go entry point), `internal/app` (HTTP server + SSE jobs), `web/` (frontend), `scripts/go_bridge.py` (Go→Python bridge), `core/` (quantization, LoRA/model merge, metadata, watermark engines), `utils/` (detection, audit, scanning, monitoring).

---

## API Endpoints

The full endpoint table (config, system, browse/search, metadata, quantize, merges, update, jobs/SSE, tools, watermark) is in [doc/api-endpoints.md](doc/api-endpoints.md).

---

## Credits

- [silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant) — safetensors quantization engine
- [qskousen/ggufy](https://github.com/qskousen/ggufy) — GGUF conversion
- [Starnodes2024/comfyui-starnodes-modelconverter](https://github.com/Starnodes2024/comfyui-starnodes-modelconverter) — INT4 quantization code basis
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGUF format reference
- [City96 ComfyUI-GGUF tools](https://github.com/city96/ComfyUI-GGUF/tree/main/tools) — GGUF tooling
- [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) — TensorCore ConvRot W4A4 layout for INT4 ConvRot, and the AsymW4A8Int8 layout for W4A8 (MiniMax H3)
- Upstream MiniMax H3 NVFP4 analysis (deep structural + mixed-profile analysis, incl. the NVFP4 HQ layer plan): [lilcheaty/MiniMax-H3-NVFP4](https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4), [coolthor/MiniMax-H3-pruned-NVFP4](https://huggingface.co/coolthor/MiniMax-H3-pruned-NVFP4), [Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot), [DmitryDB/MiniMax-H3-ComfyUI-Quants](https://huggingface.co/DmitryDB/MiniMax-H3-ComfyUI-Quants)
