# Architecture And Preservation

Back to [README](../README.md)

The architecture selection controls the `convert_to_quant` preset and, for
verified models, DaSiWa's local preservation table.

| Architecture | Local Preserve/Rescue Table | Detection | Notes |
|-------------|--------------------------|-----------|-------|
| WAN 2.2 | Yes | Yes | Verified local table for structural/video-sensitive layers |
| LTX-2.3 | Yes | Yes | Verified local table, including audio/video connector and gate-sensitive patterns |
| Krea 2 | Yes | Yes | Verified local table for image diffusion transformer. No convert_to_quant flag; uses generic quantization with local layer config |
| MiniMax H3 | Yes | Yes | Omni-modal (video+audio) DiT. No upstream convert_to_quant preset (flag=None); the local layer config carries all quality. 2K native (2560x1440). Covers FL2VA and Ref2VA |
| Hunyuan Video, Flux.2, Qwen Image, Z-Image, Z-Image Refiner, Anima, Radiance, Distillation, NeRF, text presets | No | Limited/none | Uses upstream `convert_to_quant` preset skip logic |
| Not set | No | Skipped | Runs without architecture flag, local layer config, or architecture verification |

An architecture marked with 🔒 in the main README support matrix has locally
verified preserve/rescue tables; the others rely on upstream
`convert_to_quant` preset skip rules. GGUF always applies sensitivity maps
for 🔒 architectures; Q1_0 is disabled.

Full checkpoints are detected from the source header when possible. When
local layer configs are active, baked companion modules such as VAE, audio
VAE, vocoder, text encoders, audio encoders, and text projection layers are
preserved instead of being quantized as transformer weights.

## Layer Preservation Model

- **PRESERVE_PATTERNS:** Structural/routing/I/O layers that stay at source precision via `{"skip": true}`
- **RESCUE_PATTERNS:** Layers bumped to FP8 when the base format is lower-bit (NVFP4, INT8, MXFP8, Hybrid MXFP8). On FP8 base they remain normal FP8, not BF16/FP16
- **BAKED_VAE_PATTERNS:** Unconditional companion-module skip patterns for VAE, audio VAE, vocoder, text encoders, audio encoders, projection layers, and similar full-checkpoint components
