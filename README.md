# DaSiWa Quant Station

DaSiWa Quant Station is a local quantization workbench for video diffusion models. It combines a modern Go-powered web UI with the existing Python quantization engines for GGUF and safetensors workflows.

It is built around the practical pain points of WAN 2.2 and LTX-2.3 style video checkpoints: preserving 5D/video-critical tensors, keeping full-checkpoint companion modules safe, injecting modelspec metadata, and avoiding INT8 paths that look valid but produce corrupted video.

![Quant Station Preview](assets/DaSiWa-QuantStation.webp)

## Highlights

- **Modern local UI:** Go server with HTML/CSS/JS frontend at `http://127.0.0.1:7878`.
- **Safetensors quantization:** FP8, NVFP4, INT8 Tensor-wise, and explicit INT8 ConvRot-runtime mode through `silveroxides/convert_to_quant`.
- **GGUF conversion:** GGUF F32/BF16/F16/Q8/Q6/Q5/Q4/Q3/Q2 through `ggufy`.
- **Layer safety:** Verified WAN 2.2 and LTX-2.3 preserve/rescue tables, plus baked VAE/text/audio companion preservation for full checkpoints.
- **Automatic inspection:** Header-only architecture and full-checkpoint detection when selecting a safetensors file.
- **Metadata tools:** Preview, read, inject, and hash modelspec metadata from the UI.
- **Hardware monitor:** Compact CPU/RAM/GPU/VRAM bars with MIG-friendly labels.
- **Update button:** The UI can refresh dependencies, rebuild the Go binary, and restart itself.

## Quick Start

```bash
chmod +x start-linux.sh
./start-linux.sh
```

Then open:

```text
http://127.0.0.1:7878
```

The startup script syncs the environment before launch:

- installs/checks system build tools where supported
- installs `uv` if missing
- creates `.venv/`
- installs Python dependencies
- refreshes `convert_to_quant` from GitHub
- installs `comfy-kitchen[cublas]`
- downloads/repairs `bin/ggufy`
- builds and starts the Go UI

## Launch Modes

```bash
./start-linux.sh              # setup/update, build Go UI, launch
./start-linux.sh --setup-only # setup/update only
./start-linux.sh --gradio     # launch the legacy Gradio UI
./dasiwa                      # launch the already-built Go binary only
```

Starting `./dasiwa` directly is fast, but it does not update dependencies or rebuild the app. Use `./start-linux.sh` when you want the startup-script maintenance behavior.

The Go UI also has **Update & Restart**. It runs the setup-only path, rebuilds `dasiwa`, starts a fresh copy, and exits the old server. If your system needs `sudo` for missing packages, run from a terminal or make sure sudo is already available, because the browser log cannot securely answer password prompts.

## Choosing Formats

| Format | Best Use | Notes |
|---|---|---|
| FP8 | RTX 40/50-series quality baseline | Good default for video model compression. |
| NVFP4 | Blackwell VRAM savings | More aggressive; sensitive layers are rescued to FP8 where local tables exist. |
| INT8 Tensor-wise | Safer INT8 path | Recommended INT8 choice for broad Comfy compatibility. |
| INT8 ConvRot runtime | Experimental/runtime-specific INT8 | Requires inference code that reads ConvRot metadata and rotates activations. |
| GGUF Q formats | GGUF/llama.cpp-style deployment | Uses `ggufy` plus sensitivity maps for verified video tensors. |

If an INT8 model produces pixel clutter, first try **INT8 Tensor-wise**. ConvRot row-wise INT8 only works correctly when the runtime implements the matching activation rotation; otherwise the weights and activations disagree.

## Architecture And Preservation

The architecture selection controls the `convert_to_quant` preset and, for verified models, DaSiWa's local preservation table.

| Architecture | Local Preserve/Rescue Table | Detection | Notes |
|---|---:|---:|---|
| WAN 2.2 | Yes | Yes | Verified local table for structural/video-sensitive layers. |
| LTX-2.3 | Yes | Yes | Verified local table, including audio/video connector and gate-sensitive patterns. |
| Hunyuan Video, Flux.2, Qwen Image, Z-Image, Anima, Radiance, Distillation, NeRF, text presets | No | Limited/none | Uses upstream `convert_to_quant` preset skip logic. |
| Not set | No | Skipped | Runs without architecture flag, local layer config, or architecture verification. |

Full checkpoints are detected from the source header when possible. When local layer configs are active, baked companion modules such as VAE, audio VAE, vocoder, text encoders, audio encoders, and text projection layers are preserved instead of being treated like transformer weights.

## UI Workflow

1. Pick a model folder or file with the file buttons.
2. Select a safetensors checkpoint or GGUF source.
3. Let the app inspect the source; adjust architecture/full-checkpoint if needed.
4. Choose formats from the grouped safetensors/GGUF chips.
5. Pick **Optimizer-driven** for the full `convert_to_quant` learned-rounding defaults, or **Simple** for the simple path.
6. Start the batch and watch the streamed log.
7. Use Metadata tools to read/inject modelspec metadata when needed.

## Project Layout

```text
cmd/dasiwa/              Go entry point
internal/app/            Go server, API routes, SSE job manager
web/                     Primary frontend
scripts/go_bridge.py     Bridge from Go API to Python engines
core/                    Quantization and metadata engines
utils/                   Detection, scan, audit, system helpers
models/                  Local model tree
logs/                    Conversion logs
bin/ggufy                GGUFY binary maintained by start-linux.sh
app.py                   Legacy Gradio UI
```

## Credits

- [silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant)
- [qskousen/ggufy](https://github.com/qskousen/ggufy)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [City96 ComfyUI-GGUF tools](https://github.com/city96/ComfyUI-GGUF/tree/main/tools)
- [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen)
