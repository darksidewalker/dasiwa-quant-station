# DaSiWa Quant Station

DaSiWa Quant Station is a local quantization and LoRA merge workbench for video and image diffusion models. It combines a modern Go-powered web UI with proven Python quantization engines for GGUF and safetensors workflows, plus architecture-aware LoRA merging with per-LoRA strategy control.

Built around the practical pain points of WAN 2.2, LTX-2.3, Krea 2, and MiniMax H3 diffusion checkpoints: preserving 5D/video-critical tensors, keeping full-checkpoint companion modules safe, injecting modelspec metadata, avoiding INT8 paths that produce corrupted video, and merging LoRAs with architecture-specific tensor classification. Quant outputs carry an EC-based, only-you-can-decode `modelspec.watermark` provenance token.

![Quant Station Preview](assets/DaSiWa-QuantStation.webp)

---

## At a Glance

| Area | What it does |
|------|-------------|
| Safetensors quantization | FP8, NVFP4, MXFP8, Hybrid MXFP8, INT8 Tensor-wise, INT8 Row-wise ConvRot Runtime, **INT4 ConvRot** via `silveroxides/convert_to_quant` and `comfy-kitchen` |
| GGUF conversion | F32/BF16/F16/Q8_0/Q6_K/Q5_K/Q4_K/Q3_K/Q2_K via `ggufy` with sensitivity maps for video tensor preservation |
| LoRA merge | Architecture-aware merging for WAN 2.2, LTX-2.3, and Krea 2 with per-LoRA strength, global scaling, dry-run, strict matching, adaptive mode, `.diff` format support, and Krea 2 unchain |
| Layer safety | Verified preserve/rescue tables for WAN 2.2, LTX-2.3, and Krea 2. Baked VAE/text/audio companion preservation for full checkpoints |
| Metadata tools | Preview, read, inject modelspec metadata. SHA256/AutoV1/V2/V3/CRC32 hash calculation. In-place header rewrite (avoids GB-scale full rewrites). EC-based `modelspec.watermark` provenance (only-you-can-decode) |
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

The startup script handles everything:

- Installs/checks system build tools where supported
- Installs `uv` if missing, creates `.venv/`, installs Python dependencies
- Refreshes `convert_to_quant` from GitHub
- Installs `comfy-kitchen[cublas]` (required for INT4 ConvRot)
- Downloads/repairs `bin/ggufy`
- Builds and starts the Go UI

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

| Format | Best Use | Notes |
|--------|----------|-------|
| FP8 | RTX 40/50-series quality baseline | Good default for video model compression |
| NVFP4 | Blackwell VRAM savings | More aggressive; sensitive layers rescued to FP8 where local tables exist |
| MXFP8 | Blackwell microscaling | Pure MXFP8; requires SM >= 10.0 (Blackwell). Use Hybrid for Ada compatibility |
| Hybrid MXFP8 | Ada + Blackwell compatibility | Two-pass: MXFP8 quantize then hybrid conversion with tensorwise FP8 fallback |
| INT4 ConvRot | Maximum compression | w4a4 ConvRot via comfy-kitchen TensorCore layout. Supports LTX-2.3, WAN 2.2, Krea 2, MiniMax H3. Requires BF16/FP16 source (refuses lossy sources) |
| INT8 Tensor-wise | Safer INT8 path | Recommended INT8 choice for broad ComfyUI compatibility |
| INT8 Row-wise ConvRot Runtime | Experimental/runtime-specific INT8 | Requires inference code that reads ConvRot metadata and rotates activations |
| GGUF Q formats | llama.cpp-style deployment | Uses `ggufy` plus sensitivity maps for verified video tensors |

If an INT8 model produces pixel clutter, first try **INT8 Tensor-wise**. ConvRot row-wise INT8 only works correctly when the runtime implements the matching activation rotation.

### Support Matrix

An architecture marked with 🔒 has locally verified preserve/rescue tables. Others rely on upstream `convert_to_quant` preset skip rules. GGUF always applies sensitivity maps for 🔒 architectures; Q1_0 is disabled.

| Architecture | FP8 | NVFP4 | MXFP8 | Hybrid MXFP8 | INT4 ConvRot | INT8 Tensor-wise | INT8 ConvRot RT | GGUF |
|-------------|:---:|:-----:|:-----:|:------------:|:------------:|:----------------:|:---------------:|:----:|
| WAN 2.2 🔒 | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; |
| LTX-2.3 🔒 | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; |
| Krea 2 🔒 | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; |
| MiniMax H3 🔒 | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; |
| Flux.2 | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| Hunyuan Video | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| Qwen Image | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| Z-Image | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| Z-Image Refiner | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| Anima | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| Radiance | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| Distillation Large | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| Distillation Small | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| NeRF Large | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |
| NeRF Small | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x274C; | &#x2705; | &#x2705; | &#x2705; |

INT4 ConvRot requires Simple strategy, BF16/FP16 source, and comfy-kitchen[cublas]. MXFP8 requires SM >= 10.0 (Blackwell); use Hybrid MXFP8 for Ada compatibility.

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

## LoRA Merge Workflow

1. Switch to **LoRA** mode in the workflow selector
2. Select a base checkpoint
3. Add one or more LoRAs (multi-select, drag-and-drop supported). Each LoRA gets its own strategy type and independent strength multiplier
4. Set global strength scaling, toggle adaptive mode or strict matching
5. For Krea 2: toggle **Unchain** to negate `txtfusion.projector.weight` positions 8-10
6. Use **Dry run** to preview the merge recipe without writing output
7. Shape-mismatch diagnostics automatically detect LoRAs trained on different hidden dimensions and warn before merge
8. Enter an output name and start the merge

### LoRA Merge Strategies

Each architecture applies its own filter-based preset to tensor categories:

**LTX-2.3 types** (All, Video, Audio):
- Classifies tensors into: attn_qkv, attn_out, ff_in, ff_out, audio_attn, audio_attn_out, audio_to_video_attn, video_to_audio_attn, audio_ff_in, audio_ff_out, caption_projection, patchify_or_output, norm, other
- Preserves structural layers (adaln, gate logits, baked VAE/text/audio modules)
- All merges all non-preserved weights (normal ComfyUI LoRA load behavior)
- Video merges only weights without `audio` in their key. Audio merges every weight with `audio` in its key, including cross-modal bridge weights

**WAN 2.2 strategies** (Balanced, Motion, Visuals):
- Classifies tensors into: self_attn_qkv, self_attn_out, cross_attn_qkv, cross_attn_out, ffn_in, ffn_out, modulation, caption_projection, patchify_or_output, norm, other
- No Audio strategy (WAN 2.2 has no audio components)
- Preserves modulation.lin, patch_embedding, and baked companion modules
- Norm layers always get 0.0 multiplier (untouched)

**Krea 2 strategies** (Balanced, Style, Content, Detail):
- Classifies tensors into: attn_qkv, attn_out, attn_gate, ff_in, ff_out, text_fusion, structural, other
- Style boosts attention (qkv/out/gate), reduces text_fusion — for aesthetic/style LoRAs
- Content boosts feed-forward, moderates attention — for subject/content LoRAs
- Detail applies mild global boost — for quality/detail LoRAs
- Preserves modulation.lin, tproj, tmlp, txtmlp, first/last layers, txtfusion.projector, norm.scale, qknorm

Strength limit: effective strength (`global x per_lora`) capped at +/-3.0 to prevent black images on Krea 2 gate tensors. Merge device: CPU/CUDA/auto with VRAM headroom check and OOM fallback.

### Supported LoRA Formats

Standard `.safetensors` LoRAs and ComfyUI `.diff` format are both supported.

### Recipe Reload

Every quantization and LoRA merge writes a human-readable `.txt` recipe alongside the output. Click **Load Recipe** in the UI to reload a previous run's exact settings (source, output name, all LoRAs, formats, strategy, strength) — useful for reproducing results or iterating on parameters.

---

## Model Merge (model-level)

A model-level merge — not LoRA math. Currently one recipe:

### Hybrid MiniMax H3 (`h3_hybrid`)

Switch to **Model Merge** mode, pick two MiniMax H3 checkpoints:

- **Base** = fl2va checkpoint (all tensors)
- **Overlay** = ref2va checkpoint (`blocks.{25..49}.adaln_proj.linear.{bias,weight,weight_scale}`)

Selection order doesn't matter — the engine auto-detects roles from filenames (fl2va/ref2va markers). Works for both pruned (932 keys) and full (1035 keys) variants. Output carries `minimax_h3_hybrid=baked` + `base_model`/`overlay_model` provenance.

---

## Architecture And Preservation

The architecture selection controls the `convert_to_quant` preset and, for verified models, DaSiWa's local preservation table.

| Architecture | Local Preserve/Rescue Table | Detection | Notes |
|-------------|--------------------------|-----------|-------|
| WAN 2.2 | Yes | Yes | Verified local table for structural/video-sensitive layers |
| LTX-2.3 | Yes | Yes | Verified local table, including audio/video connector and gate-sensitive patterns |
| Krea 2 | Yes | Yes | Verified local table for image diffusion transformer. No convert_to_quant flag; uses generic quantization with local layer config |
| MiniMax H3 | Yes | Yes | Omni-modal (video+audio) DiT. No upstream convert_to_quant preset (flag=None); the local layer config carries all quality. 2K native (2560x1440). Covers FL2VA and Ref2VA |
| Hunyuan Video, Flux.2, Qwen Image, Z-Image, Z-Image Refiner, Anima, Radiance, Distillation, NeRF, text presets | No | Limited/none | Uses upstream `convert_to_quant` preset skip logic |
| Not set | No | Skipped | Runs without architecture flag, local layer config, or architecture verification |

Full checkpoints are detected from the source header when possible. When local layer configs are active, baked companion modules such as VAE, audio VAE, vocoder, text encoders, audio encoders, and text projection layers are preserved instead of being quantized as transformer weights.

### Layer Preservation Model

- **PRESERVE_PATTERNS:** Structural/routing/I/O layers that stay at source precision via `{"skip": true}`
- **RESCUE_PATTERNS:** Layers bumped to FP8 when the base format is lower-bit (NVFP4, INT8, MXFP8, Hybrid MXFP8). On FP8 base they remain normal FP8, not BF16/FP16
- **BAKED_VAE_PATTERNS:** Unconditional companion-module skip patterns for VAE, audio VAE, vocoder, text encoders, audio encoders, projection layers, and similar full-checkpoint components

---

## Diagnostic Tools

- **5D Scanner:** Validate 5D tensor shapes in safetensors files. Detects video structural tensors and reports dimension anomalies
- **Pattern Audit:** Compare preserve/rescue pattern coverage against a checkpoint's tensor manifest. Shows matched/missed categories
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
- Custom output directory selector
- Quit button for clean server shutdown
- Tab-aware keepalive (server stays alive while browser tab is open)

---

## Update & Restart

In-app **Update & Restart**: pulls latest source from origin/main, refreshes dependencies, rebuilds the Go binary, and cleanly restarts the server. If your system needs `sudo` for missing packages, run from a terminal — the browser cannot securely answer password prompts.

---

## Provenance Watermark

Every quantized and LoRA-merged output carries an EC-based provenance token in the `modelspec.watermark` field. No plaintext author string is written, and the rest of the custom metadata is left untouched — only `modelspec.watermark` is added.

- **Scheme:** ephemeral X25519 (ECIES) wrapping an AES-256-GCM ciphertext. The static key is derived from your passphrase via PBKDF2-HMAC-SHA256 (clamped to a valid X25519 scalar). A fresh ephemeral key is generated per output, so every token is unique and only you can decode it.
- **Payload:** tool, architecture, model name, bit width, timestamp, a random nonce, and the SHA-256 of the output (when the file exists).
- **Decode:** with the correct passphrase the token decodes to the provenance payload; a wrong or tampered token fails (GCM authentication). Without any configured secret the field is simply not written (clean no-op).
- **Secret resolution (first hit wins):**
  1. `DASIWA_WATERMARK_PASSPHRASE` (environment)
  2. `DASIWA_WATERMARK_KEY` (environment; 64-hex pre-derived key or a passphrase)
  3. `~/.dasiwa/watermark.key` (0600, written by `go_bridge.py watermark-key`)
- **Passphrase location:** kept in your environment / a 0600 key file **outside the repository** — never committed to Gitea or GitHub.
- **UI:** a **Watermark outputs** checkbox (on by default, shared by Quantize and LoRA modes) toggles watermarking per run; a live hint below it tells you whether a key is available, whether no key is configured (no token written), or that watermarking is off for this run. Unchecking it sets a per-job kill switch so that run's outputs skip `modelspec.watermark`.
- **Status:** `GET /api/watermark` (and `go_bridge.py watermark-status`) reports whether a secret is resolvable — without ever returning the secret value.

```bash
# Persist the passphrase (0600, outside the repo)
python scripts/go_bridge.py watermark-key --passphrase "your-passphrase"

# Decode the watermark in a quant output (safetensors or GGUF)
python scripts/go_bridge.py watermark path/to/output.safetensors

# Check whether a watermark key is currently configured (for the UI)
python scripts/go_bridge.py watermark-status
```

---

## Project Layout

```
cmd/quantstation/main.go       Go entry point - web server at :7878
internal/app/server.go         Go HTTP server, API routes, SSE job manager
internal/pathbrowser/          Pure Go directory browser
web/                           Frontend (HTML/CSS/JS, dark theme)
scripts/go_bridge.py           Bridge from Go API to Python engines
core/                          Quantization, metadata, and LoRA merge engines
  gguf_engine.py               GGUF conversion + sensitivity maps
  int4_convrot_engine.py       INT4 ConvRot streaming conversion (w4a4)
  safetensors_engine.py        Safetensors quantization via convert_to_quant
  lora_merge_engine.py         Architecture-aware LoRA merge pipeline
  model_merge_engine.py        Model-level merge (H3 hybrid fl2va/ref2va)
  layer_config_builder.py      Verified preserve/rescue regex tables
  metadata_manager.py          Modelspec metadata preview/read/injection
  watermark.py                 EC X25519 provenance watermark (modelspec.watermark)
utils/                         Detection, scan, audit, system helpers
  arch_detector.py             Header-only architecture detection
  lora_inspector.py            LoRA pair discovery, tensor manifest reading
  ltx23_layer_profiles.py      LTX-2.3 tensor classification + strategy multipliers
  wan22_layer_profiles.py      WAN 2.2 tensor classification + strategy multipliers
  krea2_layer_profiles.py      Krea 2 tensor classification + strategy multipliers
  scanner_5d.py                5D tensor validation
  pattern_audit.py             Pattern coverage audit
  system.py                    CPU/RAM/GPU/VRAM monitoring
  file_ops.py                  Filesystem utilities
tests/                         Unit tests
models/                        Local model tree
logs/                          Conversion logs
bin/ggufy                      GGUFY binary maintained by start-linux.sh
start-linux.sh                 Preferred launcher
build.sh                       Go binary rebuild
lcpp.patch                     llama.cpp patch for Wan 2.2 GGUF support
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/config` | Architectures, formats, root/models directories, output_dir |
| GET | `/api/system` | CPU%, RAM, GPU%, VRAM metrics |
| GET | `/api/browse` | Directory browser (models) |
| GET | `/api/search` | Recursive file search (models) |
| GET | `/api/files` | Recursive model file listing |
| GET | `/api/inspect` | Header-only architecture detection |
| GET | `/api/metadata-preview` | Generate modelspec metadata preview |
| POST | `/api/metadata/read` | Read metadata from safetensors/GGUF |
| POST | `/api/metadata/inject` | Inject metadata into safetensors |
| POST | `/api/quantize` | Start quantization job |
| POST | `/api/lora/merge` | Start LoRA merge job |
| POST | `/api/model-merge` | Start model-level merge job (e.g. H3 hybrid) |
| POST | `/api/update` | Pull source, update dependencies, and restart |
| POST | `/api/memory/clean` | Release RAM/VRAM caches |
| POST | `/api/shutdown` | Graceful server shutdown |
| POST | `/api/tools/scan` | 5D tensor scan |
| POST | `/api/tools/audit` | Pattern coverage audit |
| GET | `/api/jobs/{id}/events` | SSE job log stream |
| POST | `/api/jobs/{id}/stop` | Cancel running job |
| GET | `/api/watermark` | Report if a watermark key is configured (no secret returned) |

---

## Credits

- [silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant) — safetensors quantization engine
- [qskousen/ggufy](https://github.com/qskousen/ggufy) — GGUF conversion
- [Starnodes2024/comfyui-starnodes-modelconverter](https://github.com/Starnodes2024/comfyui-starnodes-modelconverter) — INT4 quantization code basis
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGUF format reference
- [City96 ComfyUI-GGUF tools](https://github.com/city96/ComfyUI-GGUF/tree/main/tools) — GGUF tooling
- [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) — TensorCore ConvRot W4A4 layout for INT4 ConvRot
