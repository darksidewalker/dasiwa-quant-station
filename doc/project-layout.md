# Project Layout

Back to [README](../README.md)

```
cmd/quantstation/main.go       Go entry point - web server at :7878
internal/app/server.go         Go HTTP server, API routes, SSE job manager
internal/pathbrowser/          Pure Go directory browser
web/                           Frontend (HTML/CSS/JS, dark theme)
scripts/go_bridge.py           Bridge from Go API to Python engines
core/                          Quantization, metadata, and LoRA merge engines
  gguf_engine.py               GGUF conversion + sensitivity maps
  int4_convrot_engine.py       INT4 ConvRot streaming conversion (w4a4)
  w4a8_engine.py               W4A8 (asym_w4a8_int8) streaming conversion (MiniMax H3)
  safetensors_engine.py        Safetensors quantization via convert_to_quant
  lora_merge_engine.py         Architecture-aware LoRA merge pipeline
  model_merge_engine.py        Model-level merge (H3 hybrid fl2va/ref2va)
  layer_config_builder.py      Verified preserve/rescue regex tables + H3 NVFP4-HQ layer plan
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
