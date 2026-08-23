# API Endpoints

Back to [README](../README.md)

All endpoints go through the idle-shutdown tracker; job endpoints return SSE
streams.

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
