# API Endpoints

Back to [README](../README.md)

All endpoints go through the idle-shutdown tracker; job endpoints return SSE
streams.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/config` | Architectures, formats, `quant_capabilities` (format → architectures/strategies), root/models directories, output_dir |
| GET | `/api/system` | CPU%, RAM, GPU%, VRAM metrics |
| GET | `/api/browse` | Directory browser (models); items include `size` and `modified_at` for sorting |
| GET | `/api/search` | Recursive file search (models); results include `size` and `modified_at` |
| GET | `/api/files` | Recursive model file listing |
| GET | `/api/inspect` | Header-only architecture detection; `h3` variant and coordinate-table summary for H3 |
| GET | `/api/metadata-preview` | Generate modelspec metadata preview |
| POST | `/api/metadata/read` | Read metadata from safetensors/GGUF |
| POST | `/api/metadata/inject` | Inject metadata into safetensors |
| POST | `/api/quantize` | Start quantization job |
| POST | `/api/lora/merge` | Start checkpoint baking; `h3_turbo_complete` is default-OFF and requires additive H3 application |
| POST | `/api/lora/compose` | Compose adapters; `merge_algorithm` is `consensus` (default) or `additive` |
| POST | `/api/lora/extract` | Extract weight factors and bias patches from compatible checkpoints |
| POST | `/api/h3/prune` | Fold a full floating H3 checkpoint using reference coordinates or independent SVD |
| POST | `/api/h3/adapter-convert` | Convert a full-trained H3 adapter into the exact target pruned coordinates |
| POST | `/api/model-merge` | Start model-level merge job (e.g. H3 hybrid) |
| POST | `/api/update` | Pull source, update dependencies, and restart |
| POST | `/api/memory/clean` | Release RAM/VRAM caches |
| POST | `/api/shutdown` | Graceful server shutdown |
| POST | `/api/tools/scan` | 5D tensor scan |
| POST | `/api/tools/audit` | Pattern coverage audit |
| GET | `/api/jobs/{id}/events` | SSE job log stream |
| POST | `/api/jobs/{id}/stop` | Cancel running job |
| GET | `/api/watermark` | Report if a watermark key is configured (no secret returned) |

## H3 request contract

Both H3 jobs accept `base_path`, `architecture: "MiniMax H3"`, `output_path` or `output_dir`/`output_name`, `merge_device: "auto"|"cpu"|"cuda"`, `cuda_device`, `vram_headroom_mb`, and `dry_run`. Destinations must not overwrite an input, existing output or its exact `output.safetensors.txt` sidecar. Successful jobs return a `job_id`; consume the existing SSE endpoint for preparation, progress and final status. Stop cancels the job and cleans its owned staging files, not unrelated directories.

Pruning defaults to `fold_mode: "reference"` and requires `reference_path`. `fold_mode: "independent"` derives new SVD coordinates and does not promise compatibility with existing pruned adapters. Conversion requires `adapter_path` and `pruned_target_path`, plus the matching original full `base_path`.

`GET /api/config` includes `h3_ctq.supported` and `h3_ctq.detail` from an installed-converter capability check. This is converter availability, not inference validation. Quantization accepts `h3_quant_policy: "preserve_structural"` (default) or `"upstream_int8_convrot"`; the latter is restricted to H3, Simple strategy and `INT8 Row-wise ConvRot Runtime`. `verbose_level` accepts `DEBUG`, `VERBOSE`, `NORMAL` or `MINIMAL`.

See [H3 pruning and adapters](h3-pruning-and-adapters.md) for coordinate binding, affine bias offsets and verified execution scope.
