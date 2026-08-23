# Choosing Formats — Full Reference

Back to [README](../README.md)

Ten format families are available. The safetensors formats run through
`convert_to_quant` / `comfy-kitchen`; the GGUF formats run through `ggufy`.

| Format | Best Use | Notes |
|--------|----------|-------|
| FP8 | RTX 40/50-series quality baseline | Good default for video model compression |
| NVFP4 | Blackwell VRAM savings | More aggressive; sensitive layers rescued to FP8 where local tables exist |
| NVFP4 HQ | MiniMax H3 VRAM + quality balance | Mixed NVFP4 profile: same `--nvfp4 --comfy_quant` command as NVFP4, but a verified 30-layer plan (27× `attn.out_proj` + 3× `mlp.fc2` at specific blocks) stays at source BF16 while the other 170 main-matrix layers stay NVFP4-packed. MiniMax H3 only |
| MXFP8 | Blackwell microscaling | Pure MXFP8; requires SM >= 10.0 (Blackwell). Use Hybrid for Ada compatibility |
| Hybrid MXFP8 | Ada + Blackwell compatibility | Two-pass: MXFP8 quantize then hybrid conversion with tensorwise FP8 fallback |
| INT4 ConvRot | Maximum compression | w4a4 ConvRot via comfy-kitchen TensorCore layout. Supports LTX-2.3, WAN 2.2, Krea 2, MiniMax H3. Requires BF16/FP16 source (refuses lossy sources) |
| W4A8 | MiniMax H3 reference low-bit | asym_w4a8_int8 ConvRot via comfy-kitchen AsymW4A8Int8Layout (packed INT8 + 16-value codebook + FP8 group scales, ConvRot group 256). MiniMax H3 only; packs the heavy linears, preserves structural layers. Requires BF16/FP16 source, Simple strategy |
| INT8 Tensor-wise | Safer INT8 path | Recommended INT8 choice for broad ComfyUI compatibility |
| INT8 Row-wise ConvRot Runtime | Experimental/runtime-specific INT8 | Requires inference code that reads ConvRot metadata and rotates activations |
| GGUF Q formats | llama.cpp-style deployment | Uses `ggufy` plus sensitivity maps for verified video tensors |

## Practical notes

- If an INT8 model produces pixel clutter, first try **INT8 Tensor-wise**.
  ConvRot row-wise INT8 only works correctly when the runtime implements the
  matching activation rotation.
- INT4 ConvRot and W4A8 require Simple strategy, BF16/FP16 source, and
  comfy-kitchen[cublas] (W4A8 needs the AsymW4A8Int8Layout build, installed via
  the unpinned default-branch `comfy-kitchen`).
- MXFP8 requires SM >= 10.0 (Blackwell); use Hybrid MXFP8 for Ada
  compatibility.
- W4A8 and NVFP4 HQ are MiniMax H3 only. NVFP4 HQ is a quality variant of
  NVFP4 — the same packed NVFP4 layout plus a verified per-block BF16
  retention plan (30 heavy linears kept at source precision).
