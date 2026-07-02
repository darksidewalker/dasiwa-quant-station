# AGENTS.md — DaSiWa Quant Station

## Project Purpose
Quantization & LoRA Merge toolkit for Diffusion Models (WAN 2.2, LTX-2.3, Krea 2 + more) on NVIDIA Ada/Blackwell GPUs. Go-based HTTP server with SSE job streaming as UI frontend wrapping proven Python bridge via `scripts/go_bridge.py`.

## Architecture Overview
```
cmd/quantstation/main.go     → Go entry point, port 7878, auto-browser (DASIWA_NO_BROWSER=1)
internal/app/server.go       → HTTP Server: SSE Jobs (Event {type,text,status}), JobStore, nvidia-smi system monitoring, /api/shutdown (graceful + os.Exit(0))
internal/pathbrowser/        → File browser helper for UI file picker
core/safetensors_engine.py   → convert_to_quant orchestration: ARCH_REGISTRY (18 archs), FLAG_MAP, Layer Config Builder, Command Guard Safety Net
core/gguf_engine.py          → bin/ggufy conversion with sensitivity maps from PRESERVE/RESCUE patterns (GGUF_Q1_0 disabled)
core/lora_merge_engine.py    → [687 lines] Arch-specific LoRA merge pipeline, CPU/CUDA/auto device policy with VRAM headroom check + OOM fallback, Krea 2 unchain builtin, strength limit ±3.0
core/layer_config_builder.py → Single source of truth: PRESERVE_PATTERNS, RESCUE_PATTERNS, BAKED_VAE_PATTERNS (in-memory build)
core/metadata_manager.py     → In-place header rewriting via spacer-padding (saves GB I/O), GGUF metadata handling with version abstraction
scripts/go_bridge.py         → Python CLI dispatcher for all quantization/LoRA operations
utils/arch_detector.py       → Header-only arch detection + verify_architecture_match() pre-flight
filters/                     → Runtime layer configs (_runtime_*.json) and exact configs (_exact_*_{fmt}.json)
web/index.html               → Vanilla JS frontend, two modes: Quantize / LoRA Merge, hardware metering (CPU/RAM/GPU/VRAM)
```

## Startup Semantics
| Command | Behavior |
|---|---|
| `./start-linux.sh` | Full setup (uv, convert_to_quant, ggufy) → Go build to `quantstation` → launch browser UI |
| `./start-linux.sh --setup-only` | Setup only, then exit |
| `./build.sh` | Only Go rebuild to `quantstation`, no checks/launches |
| `./quantstation` | Run binary directly (no setup) — reads `$DASIWA_MODELS_DIR`, $HOME or `<root>/models` |

In-App **Update & Restart** (`POST /api/update`): runs `start-linux.sh --setup-only` → builds to `quantstation.next` → replaces binary → execs new instance → exits old one.

## Quantization Rules
1. **Safetensors**: Call through `convert_to_quant`. ARCH_REGISTRY determines preset flag (`--wan`, `--ltxv2`, etc.). Layer configs: exact configs (filters/_exact_*) override runtime regex builds. Output suffix `_mixed` for runtime config, `_exact` for reference-derived. Command guard prevents incorrect flag combinations before subprocess launch.
2. **GGUF**: `bin/ggufy convert --datatype <fmt>`. Sensitivity maps: preserve-patterns completely removed from candidate list (not high score), rescue→score 25, standard 2D linear→50. GGUF_Q1_0 is disabled (ggufy binary unsupported).
3. **Metadata**: After each quantization: inject_metadata + recipe `.txt` alongside output (AutoV1/V2/V3/SHA256/CRC32 hashes). `core/metadata_configs.py` contains templates per architecture.

## Layer Preservation Model
- **PRESERVE_PATTERNS** (`{"skip": true}`): Remain at source precision. Locally verified for WAN 2.2, LTX-2.3, Krea 2 + more architectures.
- **RESCUE_PATTERNS**: Bump to FP8 on NVFP4 bases; no rescue on INT8 (ComfyUI compatibility with Winnougan's int8tensormixed). ConvRot Runtime: row-wise scaling in config default.
- **BAKED_VAE_PATTERNS** (arch-independent): VAE, audio_vae, vocoder, text_encoder*, clip/gemma/llm*, language_model, audio_encoder, embedding_projection — always `{skip:true}`, anchored with `^`.

## LoRA Merge Rules
- Arch profiles in `utils/<arch>_layer_profiles.py`: classify_key(), is_*_preserved_key(), strategy_multiplier(). Dispatch via `_get_profile()` in lora_merge_engine.py. Default=LTX23.
- Strategies: LTX23 (Balanced/Motion/Visuals/Audio), WAN 2.2 (Balanced/Motion, **no Audio**), Krea 2 (Balanced/Style/Content/Detail). Norm layers on WAN 2.2 always multiplier 0.0.
- `scale = global_strength × per_lora_strength × strategy_multiplier(category)`
- Strength limit: effective strength (`global × per_lora`) max ±3.0 — prevents black images on Krea 2 gate tensors.
- Built-in Krea 2 unchain: negates `txtfusion.projector.weight` positions 8–10 (shape `(1,12)`).
- Merge device: CPU/CUDA/auto with VRAM headroom check + fallback to CPU on OOM. Summary logs device distribution.
- Strict matching = reject when LoRA tensor has no base candidate; dry run = report only, no output write.

## Python Bridge Commands (go_bridge.py)
```
inspect <path>                              → arch_detector.inspect_checkpoint()
metadata --name X --architecture Y [--full] → metadata_manager.update_metadata_preview()
read-metadata-path <path>                   → read_any_metadata(path)
inject-metadata-path --json '{"path":"...","metadata":"..."}'  → inject_metadata (adds SHA256 hash)
quantize --json '{source,formats,arch,strategy,...}'           → run_safe_conversion and/or run_gguf_conversion (streaming JSON events)
lora-merge --json '{base,loras,strategy,strength,adaptive,dry_run,...}'  → run_lora_merge()
scan <path>                                 → scanner_5d.scan_5d_tensors()
audit <path> --architecture X               → pattern_audit.audit_patterns()
clean-memory                                → GC + malloc_trim + torch.cuda.empty_cache() + cupy pool flush + nvidia-smi VRAM holders listing
```

## Event Format (SSE)
All jobs stream JSON events to Go server:
`{"type":"log","text":"..."}`  — Log message
`{"type":"status","status":"running"}`  — Job state for UI progress
`{"type":"done","status":"finished"}|stopped|failed|dry-run complete|no matches|Aborted:*`  — Completion

## Go API Endpoints (server.go)
### Health & Config
- `GET /api/ping` → `{status: "ok"}`
- `GET /api/config` → Version, root/models dir, architecture list (18 archs), format list (FP8/NVFP4/MXFP8/INT8/GGUF*)

### System & Filesystem
- `GET /api/system` → CPU/RAM/GPU/VRAM stats (nvidia-smi + /proc/meminfo)
- `GET /api/browse?path=X` → Directory contents (.safetensors/.gguf/.ckpt etc. only)
- `GET /api/search?q=X&path=Y` → File search results

### Inspection & Metadata
- `GET /api/inspect?path=X` → Architecture detection + full checkpoint flag
- `GET /api/metadata-preview?name=X&architecture=Y&full=true/false` → Preview JSON
- `POST /api/metadata/read` → `{path: "..."}` → Read current metadata
- `POST /api/metadata/inject` → `{path: "...", metadata: "{...}"}` → Inject metadata

### Quantization (Async Jobs)
- `POST /api/quantize` → `{source_path, model_name, formats[], architecture, strategy, optimizer, low_vram, full_checkpoint}` → `{job_id: "..."}`
  - Job events via SSE: `GET /api/jobs/{id}/events`
  - Stop: `POST /api/jobs/{id}/stop`

### LoRA Merge (Async Jobs)
- `POST /api/lora/merge` → `{base_path, loras[{path, strength}], strategy, architecture, global_strength, adaptive, dry_run, strict_matching}` → `{job_id: "..."}`
  - Job events via SSE: `GET /api/jobs/{id}/events`

### Tools (Sync)
- `POST /api/tools/scan` → `{path: "..."}` → Scan 5D tensors
- `POST /api/tools/audit` → `{path, architecture}` → Pattern audit report

### System Management
- `POST /api/shutdown` → Graceful shutdown + os.Exit(0)
- `POST /api/memory/clean` → Go heap GC + Python gc.collect() + libc.malloc_trim + torch.cuda.empty_cache() + cupy pool flush + TF session clear
- `POST /api/update` → Update & restart pipeline

## Agent Rules (important!)
- **Search**: Use `rg` / `read_file`. Not grep/cat/head/tail in terminal.
- **Edit**: `patch` with fuzzy match. If patch fails 2× on same region → use `write_file`.
- **Dependencies**: `uv` for everything Python. `.venv/` in project root.
- **Binary path**: `cmd/quantstation/main.go` → built to `./quantstation`. Not `cmd/dasiwa/`.
- **No drive-by changes**: No refactors, renames or formatting without explicit request. Only change what the task needs.
- **Go = shell over Python**: Go is UI frontend for proven Python quantization; never reimplement quantizer in Go.
- **Preserve streams**: Job events (log/status/done) via SSE to frontend. Do not refactor to polling or buffered completion.

## Code Quality & Safety
- Command guard safety net: Before every subprocess launch, command line is validated for correct flag combinations (architecture flag present + exactly once, strategy flag present, no conflicting flags).
- Strength limit ±3.0 on LoRA merge prevents destructive effective strengths.
- In-place metadata rewrite avoids GB-scale full rewrites when possible.
- Architecture verification pre-flight warns about wrong arch selection (e.g., LTX file with WAN selected).

## Test Suite
```bash
# pytest for Python modules
pytest tests/test_lora_merge_engine.py -v  # LoRA merge tests
pytest tests/test_recipe_load.py -v         # Recipe loading tests