# LTX-2.3 Adaptive LoRA Merge Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a DaSiWa UI/API feature for merging LoRAs into LTX-2.3 safetensors checkpoints using verified LTX-2.3 layer naming, multi-trainer LoRA key normalization, and adaptive layer-scaled strategy profiles for I2V motion, visuals, audio, and balanced merges.

**Architecture:** Build a Python merge engine first, verified with synthetic safetensors and real header-only inspection when checkpoints are available. Expose it through `scripts/go_bridge.py`, then add a Go `/api/lora/merge` job endpoint and a frontend LoRA Merge panel/tab that streams job logs through the existing job/SSE system.

**Tech Stack:** Python 3.12, `torch`, `safetensors`, Go HTTP server, vanilla JS frontend, existing DaSiWa job/SSE infrastructure.

---

## Current Status Checklist

- [x] Confirmed repo currently has no LoRA merge feature or LoRA-specific code.
- [x] Confirmed existing LTX-2.3 layer knowledge lives in `core/layer_config_builder.py` and `utils/arch_detector.py`.
- [x] Confirmed existing LTX-2.3 preserved/sensitive names include `model.diffusion_model.transformer_blocks.*`, `adaln_single`, `*_embeddings_connector`, `caption_projection`, `patchify_proj`, `proj_out`, `audio_patchify_proj`, `audio_proj_out`, `to_gate_logits`, q/k norms, attention `to_q/to_k/to_v/to_out`, and `(audio_)?ff` linears.
- [x] Confirmed the primary UI is `web/index.html`, `web/app.js`, `web/styles.css` and server is `internal/app/server.go`.
- [x] User provided real full LTX-2.3 checkpoint: `/home/darksidewalker/GitHub/AI-Ressources/models/unet/LTX2/10Eros_v1.2_bf16.safetensors`.
- [x] User provided real LoRA directory: `/home/darksidewalker/GitHub/AI-Ressources/models/loras/LTX/`.
- [x] Header-inspected the provided full checkpoint: 42.971 GiB, 5,947 tensors, BF16, full checkpoint with top-level `model`, `vocoder`, `vae`, `audio_vae`, and `text_embedding_projection` families.
- [x] Confirmed provided LTX-2.3 checkpoint has 48 DiT transformer blocks: `model.diffusion_model.transformer_blocks.0` through `.47`.
- [x] Confirmed real DiT merge target families include `attn1`, `attn2`, `audio_attn1`, `audio_attn2`, `audio_to_video_attn`, `video_to_audio_attn`, `ff`, and `audio_ff` linears.
- [x] Header-inspected 28 real LoRA files from `/home/darksidewalker/GitHub/AI-Ressources/models/loras/LTX/`.
- [x] Confirmed real LoRA naming variants include `diffusion_model.*.lora_A/lora_B`, `diffusion_model.*.lora_down/lora_up` plus `.alpha`, and `base_model.model.transformer_blocks.*.lora_A/lora_B`.
- [x] Confirmed observed LoRA ranks vary, including 32, 64, 128, mixed 32/64, and one variable-rank LoRA with rank range 1-72.
- [ ] Needs review with generated I2V samples before declaring strategy defaults final.

## Real Artifact Inspection Notes

### Base checkpoint

Path:

```text
/home/darksidewalker/GitHub/AI-Ressources/models/unet/LTX2/10Eros_v1.2_bf16.safetensors
```

Observed with header/slice inspection using the project venv via `uv run python`:

- Size: 42.971 GiB
- Tensor count: 5,947
- Top-level families:
  - `model`: 4,444 tensors
  - `vocoder`: 1,227 tensors
  - `vae`: 170 tensors
  - `audio_vae`: 102 tensors
  - `text_embedding_projection`: 4 tensors
- Transformer blocks: 48 blocks, indices 0-47
- `model.diffusion_model.transformer_blocks.*`: 4,128 tensors
- Attention q/k/v merge targets: 864 tensors
- Attention output merge targets: 288 tensors
- FF/audio-FF targets: 384 tensors
- Audio-related tensors: 2,266 tensors
- Connector tensors: 258 tensors
- Caption/patch/output/scale-shift-ish tensors: 298 tensors

Representative real target keys and shapes:

```text
model.diffusion_model.transformer_blocks.0.attn1.to_k.weight                    [4096, 4096]
model.diffusion_model.transformer_blocks.0.attn1.to_out.0.weight               [4096, 4096]
model.diffusion_model.transformer_blocks.0.attn2.to_q.weight                    [4096, 4096]
model.diffusion_model.transformer_blocks.0.audio_attn1.to_k.weight             [2048, 2048]
model.diffusion_model.transformer_blocks.0.audio_attn2.to_out.0.weight         [2048, 2048]
model.diffusion_model.transformer_blocks.0.audio_to_video_attn.to_q.weight     [2048, 4096]
model.diffusion_model.transformer_blocks.0.audio_to_video_attn.to_out.0.weight [4096, 2048]
model.diffusion_model.transformer_blocks.0.video_to_audio_attn.to_k.weight     [2048, 4096]
model.diffusion_model.transformer_blocks.0.video_to_audio_attn.to_v.weight     [2048, 4096]
model.diffusion_model.transformer_blocks.0.video_to_audio_attn.to_out.0.weight [2048, 2048]
```

Important implication: key normalization must map common LoRA prefix `diffusion_model.` to base prefix `model.diffusion_model.`. It must also handle `base_model.model.transformer_blocks.` style by mapping to `model.diffusion_model.transformer_blocks.`.

### Real LoRA directory

Path:

```text
/home/darksidewalker/GitHub/AI-Ressources/models/loras/LTX/
```

Observed 28 `.safetensors` files. Key styles found:

1. Common Diffusers-style:

```text
diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight
diffusion_model.transformer_blocks.0.attn1.to_k.lora_B.weight
```

2. Down/up plus alpha style:

```text
diffusion_model.transformer_blocks.0.attn1.to_gate_logits.alpha
diffusion_model.transformer_blocks.0.attn1.to_gate_logits.lora_down.weight
diffusion_model.transformer_blocks.0.attn1.to_gate_logits.lora_up.weight
```

3. `base_model.model` prefix style:

```text
base_model.model.transformer_blocks.0.attn1.to_k.lora_A.weight
base_model.model.transformer_blocks.0.attn1.to_k.lora_B.weight
```

4. Audio-specific LoRA targets exist, e.g.:

```text
diffusion_model.transformer_blocks.0.audio_attn1.to_k.lora_A.weight
diffusion_model.transformer_blocks.0.audio_attn1.to_k.lora_B.weight
```

Observed rank patterns:

- Many LoRAs are fixed rank 32.
- Many LoRAs are fixed rank 64.
- Several LoRAs are fixed rank 128.
- Several LoRAs have mixed rank 32/64.
- `ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors` has variable rank range 1-72 and targets preserved `adaln_single` keys; preserved-key skip handling is mandatory.

Real LoRA adapter requirements from inspection:

- Pair `lora_A`/`lora_B` and `lora_down`/`lora_up`.
- Read `.alpha` tensors when present.
- Derive rank per target from tensor shape, not file-global metadata.
- Allow per-layer rank variation inside a single LoRA.
- Skip preserved structural keys such as `adaln_single` and `to_gate_logits` by default unless an expert override is explicitly enabled.
- Produce a dry-run match report before writing merged weights.

## Do We Need the Actual Checkpoint?

Not strictly for coding the feature, because the merge engine can be built and tested against synthetic safetensors and header-only key dumps.

But yes, for making the LTX-2.3 strategy reliable, we should examine at least the real checkpoint header. The full multi-GB weights are not necessary for initial layer verification; a tensor key/shape manifest is enough.

Preferred safe artifact:

```bash
uv run python - <<'PY'
from safetensors import safe_open
import json, sys
path = sys.argv[1]
with safe_open(path, framework='pt', device='cpu') as f:
    rows = {k: {'shape': list(f.get_tensor(k).shape), 'dtype': str(f.get_tensor(k).dtype)} for k in f.keys()}
print(json.dumps(rows, indent=2))
PY /path/to/ltx23.safetensors > ltx23-header-manifest.json
```

However, avoid that exact implementation for very large files during normal app use because `get_tensor()` loads tensors. The implemented app inspector must use `f.get_slice(k).get_shape()` and `f.get_slice(k).get_dtype()` or equivalent header/slice APIs where available. If that API is unavailable in the installed safetensors version, use `safe_open(...).keys()` for header-only key names and only inspect selected tensors on demand.

For actual merge validation, a real base checkpoint plus one or more real LoRAs is eventually needed because tensor dimensionality and trainer naming variants determine whether key matching is correct.

---

## Design Principles

1. Do not invent LTX-2.3 patterns without verification.
2. Never merge into structural/preserved tensors by default.
3. Always produce a dry-run match report before mutating weights.
4. Support multiple LoRA trainer naming schemes through explicit adapters.
5. Handle arbitrary LoRA ranks by deriving rank from tensor shapes, not config assumptions.
6. Fail closed on ambiguous or shape-mismatched keys.
7. Store merge metadata in output safetensors.
8. Keep adaptive scaling heuristic and transparent, not falsely “learned”.

---

## Proposed Merge Strategies

### Balanced I2V

Default profile. Moderate LoRA contribution on DiT attention/MLP tensors. Lower contribution near sensitive input/output/projection paths.

### I2V Motion

Higher contribution on temporal/audio-video cross-attention-like blocks and later transformer blocks, lower on visual projection/detail-heavy blocks. Intended to transfer motion dynamics without overbaking appearance.

### I2V Visuals

Higher contribution on spatial/detail MLP and visual attention layers, lower on motion/audio-video routing. Intended to transfer style, characters, textures, and visual composition.

### Audio

Only applies high contribution to keys clearly mapped to audio modules such as `audio_*`, `audio_vae`, vocoder, audio prompt paths, or audio-specific transformer submodules. If no audio keys match, warn and skip/abort depending on user choice.

### Custom

User supplies per-category multipliers in the UI.

### Adaptive Layer Scaling

Adaptive means heuristic/statistical scaling, not training. Suggested formula per matched target tensor:

```text
raw_delta_ratio = ||lora_delta|| / max(||base_weight||, eps)
scale_adjustment = target_ratio / clamp(raw_delta_ratio, min_ratio, max_ratio)
final_scale = strategy_scale * global_strength * per_lora_strength * clamp(scale_adjustment, min_scale, max_scale)
```

Then smooth scales by block index/category to avoid noisy per-layer jumps. Report all scales in a merge manifest.

---

## Files Likely To Change

- Create: `core/lora_merge_engine.py`
- Create: `utils/lora_inspector.py`
- Create: `utils/ltx23_layer_profiles.py` or add profile helpers near `core/lora_merge_engine.py`
- Modify: `scripts/go_bridge.py`
- Modify: `internal/app/server.go`
- Modify: `web/index.html`
- Modify: `web/app.js`
- Modify: `web/styles.css`
- Modify: `core/metadata_manager.py` if a reusable merge metadata helper is needed
- Optional create: `scripts/synthetic_lora_fixture.py`

---

## Task 1: Add Header-Only Tensor Manifest Helpers

**Objective:** Create a safe reusable helper to inspect safetensors keys, shapes, and dtypes without loading whole tensors.

**Files:**
- Create: `utils/lora_inspector.py`

**Implementation notes:**
- Provide `read_safetensors_manifest(path) -> dict[str, TensorInfo]`.
- Use `safe_open(path, framework='pt', device='cpu')`.
- Prefer `f.get_slice(key).get_shape()` and `f.get_slice(key).get_dtype()` if available.
- Fall back to key-only mode with clear warning if shape/dtype cannot be retrieved without loading.
- Include `summarize_ltx23_layers(manifest)` to count categories:
  - transformer blocks
  - attention q/k/v/out
  - MLP/FF
  - audio-prefixed modules
  - structural preserve matches
  - unknown/other

**Verification:**

```bash
uv run python -m py_compile utils/lora_inspector.py
```

Expected: no errors.

**Done:** [ ]

**Needs review later:** [ ] Verify against real LTX-2.3 checkpoint header.

---

## Task 2: Centralize LTX-2.3 Merge Categories

**Objective:** Create one profile source that maps verified LTX-2.3 tensor names to merge categories and preserved/sensitive behavior.

**Files:**
- Create: `utils/ltx23_layer_profiles.py`
- Read/reference: `core/layer_config_builder.py`

**Implementation notes:**
- Reuse or import LTX-2.3 preserve patterns from `core/layer_config_builder.py` instead of duplicating silently.
- Add functions:
  - `is_ltx23_preserved_key(key) -> bool`
  - `classify_ltx23_key(key) -> LayerCategory`
  - `extract_block_index(key) -> int | None`
- Categories should include:
  - `preserve`
  - `attn_qkv`
  - `attn_out`
  - `ff_in`
  - `ff_out`
  - `audio_attn`
  - `audio_ff`
  - `audio_io`
  - `caption_projection`
  - `patchify_or_output`
  - `norm_or_gate`
  - `other`

**Verification:**

```bash
uv run python -m py_compile utils/ltx23_layer_profiles.py
```

Expected: no errors.

**Done:** [ ]

**Needs review later:** [ ] Compare classification report against real LTX-2.3 manifest.

---

## Task 3: Implement LoRA Key Normalization Adapters

**Objective:** Support LoRAs from different trainers and naming styles.

**Files:**
- Create/modify: `utils/lora_inspector.py`

**Trainer/key styles to support initially:**
- Diffusers-like `...lora_A.weight` / `...lora_B.weight`
- Kohya-like `lora_unet_...lora_down.weight` / `...lora_up.weight`
- LyCORIS-ish up/down forms where dimensions are compatible
- Direct diff/delta tensors if a trainer emits full-size deltas
- Optional `.alpha` tensor or metadata alpha

**Implementation notes:**
- Build `discover_lora_pairs(manifest) -> list[LoraPair]`.
- Normalize keys to candidate base keys using explicit rules.
- Generate multiple candidate target names, including:
  - raw normalized name
  - `model.diffusion_model.` prefix added/removed
  - dot/underscore conversions for common trainer encodings
  - `.weight` suffix handling
- Do not guess too aggressively. Rank candidates and require exactly one shape-compatible target unless user enables permissive mode.
- Rank must be shape-derived:
  - down shape `[rank, in]` and up shape `[out, rank]`
  - or transposed variants if detected safely
- Record every unmatched, ambiguous, and shape-mismatched pair in dry-run output.

**Verification:**
- Create tiny in-memory/synthetic manifests in a small Python test script or doctest-style assertions.
- Validate adapter pairs produce expected target candidates.

```bash
uv run python -m py_compile utils/lora_inspector.py
```

Expected: no errors.

**Done:** [ ]

**Needs review later:** [ ] Add real trainer LoRA key examples from user’s files.

---

## Task 4: Build Backend Merge Engine With Dry-Run Mode

**Objective:** Implement deterministic merge math and reporting without UI/server changes.

**Files:**
- Create: `core/lora_merge_engine.py`

**Implementation notes:**
- Public API:
  - `run_lora_merge(payload) -> generator[(log_text, status)]`
  - or similar to `run_safe_conversion` style.
- Inputs:
  - base path
  - list of LoRA paths with per-LoRA strength
  - output path/name
  - strategy
  - global strength
  - adaptive enabled
  - dry-run flag
  - strict/permissive matching flag
- For each LoRA pair:
  - map to target base key
  - load only required tensors
  - compute delta using dtype-safe torch matmul on CPU/GPU option later
  - validate delta shape equals base tensor shape
  - classify target key
  - compute strategy scale
  - compute adaptive adjustment if enabled
  - accumulate merged tensor
- Preserve non-target base tensors unchanged.
- Skip preserved LTX-2.3 keys by default even if LoRA targets them; report as skipped.
- Write safetensors output only when not dry-run.
- Save metadata manifest in safetensors metadata.

**Verification:**
- Use a synthetic base with a few 2D tensors and synthetic LoRA up/down tensors.
- Check exact math for one pair.
- Check skipped preserved key remains identical.
- Check dry-run creates no output.

Command:

```bash
uv run python -m py_compile core/lora_merge_engine.py
```

Expected: no errors.

**Done:** [ ]

**Needs review later:** [ ] Validate memory behavior on large real checkpoints.

---

## Task 5: Add Adaptive Scaling Heuristic

**Objective:** Implement transparent adaptive scale computation and reporting.

**Files:**
- Modify: `core/lora_merge_engine.py`

**Implementation notes:**
- Add strategy config values:
  - target delta/base ratio
  - min/max layer scale
  - category multiplier
  - block-index curve
- Adaptive algorithm:
  - compute base norm and delta norm
  - compute delta ratio
  - adjust scale toward category target ratio
  - clamp scale
  - smooth nearby block scales after dry-run analysis if possible
- Include per-layer report fields:
  - base key
  - category
  - block index
  - original strategy scale
  - delta/base ratio
  - adaptive multiplier
  - final scale
  - reason if skipped

**Verification:**
- Synthetic tensors with intentionally huge and tiny deltas.
- Huge delta should clamp down.
- Tiny delta should boost only within max clamp.

**Done:** [ ]

**Needs review later:** [ ] Tune target ratios using real I2V outputs.

---

## Task 6: Add CLI Bridge Command

**Objective:** Expose the merge engine through `scripts/go_bridge.py`.

**Files:**
- Modify: `scripts/go_bridge.py`

**Implementation notes:**
- Import `run_lora_merge` from `core.lora_merge_engine`.
- Add subcommand:

```text
lora-merge --json <payload>
```

- Stream JSON events like quantization:
  - `{type: "log", text: "..."}`
  - `{type: "status", status: "..."}`
  - `{type: "done", status: "Finished"}`
  - `{type: "error", error: "..."}`

**Verification:**

```bash
uv run python scripts/go_bridge.py lora-merge --json '{...dry-run synthetic payload...}'
```

Expected: JSON log/status events and no output file in dry-run.

**Done:** [ ]

**Needs review later:** [ ] Validate payload schema with frontend.

---

## Task 7: Add Go API Endpoint And Job Wiring

**Objective:** Add a server endpoint that launches LoRA merge as a streamed job.

**Files:**
- Modify: `internal/app/server.go`

**Implementation notes:**
- Add route:

```go
mux.HandleFunc("POST /api/lora/merge", s.handleLoraMerge)
```

- Define request struct:
  - `base_path`
  - `loras: [{path, strength}]`
  - `models_dir`
  - `output_name`
  - `strategy`
  - `global_strength`
  - `adaptive`
  - `dry_run`
  - `strict_matching`
- Reuse existing job infrastructure and Python command launch pattern from quantization/update handlers.
- Ensure paths are cleaned/expanded similarly to other handlers.

**Verification:**

```bash
go test ./...
go build -o /tmp/dasiwa-test ./cmd/dasiwa
```

Expected: pass/build succeeds.

**Done:** [ ]

**Needs review later:** [ ] API ergonomics in UI.

---

## Task 8: Add Frontend LoRA Merge Panel/Tab

**Objective:** Add UI controls for selecting base model, LoRAs, strategy, adaptive settings, dry-run, and starting merge jobs.

**Files:**
- Modify: `web/index.html`
- Modify: `web/app.js`
- Modify: `web/styles.css`

**Implementation notes:**
- Add a new main workspace section or tab labelled `LoRA Merge`.
- Inputs:
  - Base checkpoint picker
  - Add LoRA button with file picker
  - Per-LoRA strength input
  - Output name
  - Strategy select: Balanced I2V, I2V Motion, I2V Visuals, Audio, Custom
  - Global strength
  - Adaptive checkbox default on
  - Dry-run checkbox default on for first run
  - Strict matching checkbox default on
- Start button calls `/api/lora/merge`.
- Reuse existing console/status/SSE flow.
- Show dry-run report in console.

**Verification:**
- Browser opens without JS errors.
- Dry-run job starts and streams logs.
- Existing quantization UI still works.

**Done:** [ ]

**Needs review later:** [ ] UX pass after first real merge attempt.

---

## Task 9: Add Synthetic Fixture Validation Script

**Objective:** Add a repeatable validation command that proves merge math and key matching work without real checkpoints.

**Files:**
- Create: `scripts/synthetic_lora_fixture.py`

**Implementation notes:**
- Generate tiny base safetensors with keys shaped like LTX-2.3 linears.
- Generate LoRA safetensors in at least two naming styles.
- Run dry-run and actual merge through bridge or direct engine.
- Verify output tensor equals expected base + scaled delta.
- Verify preserved key is unchanged.
- Print a pass/fail summary.

**Verification:**

```bash
uv run python scripts/synthetic_lora_fixture.py
```

Expected: all checks pass.

**Done:** [ ]

**Needs review later:** [ ] Expand fixture with every real LoRA naming style encountered.

---

## Task 10: Real-Model Header Verification

**Objective:** Verify actual LTX-2.3 layer naming and dimensions before claiming production readiness.

**Files:**
- No source changes required unless mismatches are found.
- Possible modifications: `utils/ltx23_layer_profiles.py`, `utils/lora_inspector.py`

**Steps:**
1. Inspect base checkpoint header only.
2. Save summary report under `logs/` or `filters/`.
3. Confirm known categories cover target DiT layers.
4. Confirm preserve patterns do not accidentally classify mergable linear weights.
5. Inspect one or more real LoRA headers.
6. Confirm LoRA pair detection maps to base keys with shape compatibility.

**Verification command:**

```bash
uv run python scripts/go_bridge.py lora-merge --json '{"dry_run": true, ...}'
```

Expected: high match rate for intended target tensors, zero unexpected preserved-layer merges, clear report for unmatched keys.

**Done:** [ ]

**Needs review later:** [ ] Required before moving feature out of experimental status.

---

## Task 11: Full Build/Test Verification

**Objective:** Confirm the whole app still builds and the new feature path works.

**Commands:**

```bash
uv run python -m py_compile scripts/go_bridge.py core/lora_merge_engine.py utils/lora_inspector.py utils/ltx23_layer_profiles.py
go test ./...
go build -o /tmp/dasiwa-test ./cmd/dasiwa
uv run python scripts/synthetic_lora_fixture.py
```

Expected: all pass.

**Done:** [ ]

**Needs review later:** [ ] Run a manual browser test at `http://127.0.0.1:7878`.

---

## Risks And Mitigations

### Risk: LoRA trainer key names vary too much

Mitigation: start strict, generate dry-run reports, add adapters only from observed real examples.

### Risk: Shape-compatible but semantically wrong target match

Mitigation: require strong normalized name match, not just shape match. Report ambiguous matches instead of merging.

### Risk: Adaptive scaling overcorrects

Mitigation: clamp scales, expose dry-run table, keep deterministic presets available.

### Risk: Audio strategy is underverified

Mitigation: mark audio strategy experimental until real audio LoRA/base key manifests are reviewed.

### Risk: Full checkpoint companion modules get accidentally merged

Mitigation: reuse preserve/baked companion patterns and skip non-DiT modules unless explicitly enabled.

---

## Open Questions For Later Sessions

- Which LTX-2.3 base checkpoint variant is the primary target: transformer-only or full checkpoint?
- Which trainers generate the LoRAs to support first?
- Are the target LoRAs classic LoRA only, or also LoHa/LoKr/DoRA variants?
- Should the first implementation support multiple LoRAs in one run, or one LoRA first for safety?
- Should merged output stay BF16/FP16, or optionally quantize immediately afterward through existing DaSiWa quantization?
- Should strategy defaults be exposed as editable JSON profiles in `filters/`?

---

## Recommended First Execution Slice

1. Implement `utils/lora_inspector.py` and `utils/ltx23_layer_profiles.py`.
2. Implement backend-only dry-run merge report.
3. Validate with synthetic fixtures.
4. Inspect real base and LoRA headers.
5. Only then wire the UI.

This avoids building a nice UI around merge logic that has not yet proven it can map real trainer LoRA keys safely.
