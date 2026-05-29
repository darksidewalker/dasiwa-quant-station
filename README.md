# 🌀 DaSiWa Quant Station

DaSiWa Quant Station is a high-performance toolkit designed for quantizing Video Models. Specifically engineered for systems with 64GB RAM and NVIDIA Ada (40-series) or Blackwell (50-series) GPUs.

📦 GGUF Expert: Native Wan 2.2 and LTX-2.3 GGUF quantization via `ggufy`, utilizing sensitivity-aware quantization to preserve 5D video tensors and prevent "gray-screen" or corrupted outputs.

💎 Next-Gen FP Quants: Native support for NVFP4 (Blackwell) and FP8 E4M3 (Ada) via optimized convert_to_quant integration.

🛡️ 64GB Safety Logic: Intelligent memory flushing and subprocess monitoring to prevent OOM (Out of Memory) crashes during model handling.

## 🚀 Quick Start

### Prerequisites

Ensure you have uv installed for high-speed dependency management:
Bash
```
curl -LsSf https://astral.sh/uv/install.sh | sh
``` 

### Installation & Launch

The included start-linux.sh environment syncing, and build requirements automatically.
Bash
```
chmod +x start-linux.sh
./start-linux.sh
```

#### 🛠️ Quantization Guide

|Format|Target|Hardware|
|-----:|-----:|-------:|
| GGUF (Q2-Q8) | Universal / CPU | Best for VRAM-constrained systems (8GB - 12GB).|
| FP8 (E4M3) | RTX 40-Series | Native Ada acceleration; best quality/speed balance.|
| NVFP4 | RTX 50-Series | Blackwell native 4-bit; extreme VRAM savings for 14B models.|
| MXFP8 | RTX 50-Series | Microscaled 8-bit; near-lossless video quality.|

#### 🧬 Architecture Support

The Architecture dropdown selects which `convert_to_quant` preset is applied to the safetensors quantization pass. All 18 upstream presets are exposed, grouped by category. A 19th option, **Not set**, bypasses the preset entirely and runs `convert_to_quant` with no architecture flag, no layer config, and no source-file architecture verification.

| Category | Choices | Status |
|---|---|---|
| Video | WAN 2.2, LTX-2.3, Hunyuan Video | WAN 2.2 and LTX-2.3 have hand-tuned layer-config patterns, GGUFY sensitivity maps, and source-file verification. |
| Image | Flux.2, Qwen Image, Z-Image, Z-Image Refiner, Anima | Upstream preset only (no per-arch layer config in this project yet). |
| Other | Radiance, Distillation Large/Small, NeRF Large/Small | Upstream preset only. |
| Text | T5-XXL, Qwen 3.5, Mistral, Visual, Generic Text | Upstream preset only. Companion-model use. |
| None | Not set | No preset, no layer config, no verification. |

For archs without per-arch layer-config patterns, sensitive-layer preservation is handled entirely by the upstream `convert_to_quant` preset (the same skip rules the author ships). The 5D Tensor Scan, Pattern Audit, and Compare-to-Reference tools currently only operate on WAN 2.2 and LTX-2.3 files; selecting another arch in the dropdown when using those tools returns a clear "no patterns defined" error.

### 📂 Directory Structure & UI

    models/: Place your .safetensors base models and LoRAs here.

    logs/: Automated session logs for debugging merge weights.

UI changes (current):
- The left-hand Model Directory control is a dropdown that lists the internal `MODELS_DIR` and its immediate subfolders. This replaces earlier free-text or upload-style folder pickers.
- Select a folder from the dropdown to populate the `Safetensors file` dropdown automatically. Use the `↻ Refresh folder` button to refresh the available directories and file lists.
- The app operates on file paths inside the host's `models/` tree; you do not upload files through the UI.

### 🤝 Credits

Quantization: 
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant)
- [City96](https://github.com/city96/ComfyUI-GGUF/tree/main/tools)
- [ggufy](https://github.com/qskousen/ggufy)

Utilities and UI:
- Gradio (UI library) — UI components and patterns used in the app UI.
- [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) for Blackwell/NVFP4 support.

