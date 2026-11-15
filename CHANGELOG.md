# Changelog

## 2026.11.15-hyperflow-meta (unreleased, uncommitted)

- **HyperFlow H3 Flash checkpoints survive quantization.** The H3 layer-config preserves now keep the loader-contract tensors at source precision: `endpoint_time_embedder.*` (the step-endpoint embedder of the two-time `(t, r)` recipe) and every `.down`/`.up` low-rank factor, which `HyperFlowLoader` installs as plain bf16 ops. Dense `qkv_proj/out_proj/fc1/fc2` still quantize as before. `hyperflow_gate`/`hyperflow_sigmas` always passed through as non-`.weight` tensors; patterns now also cover them in audits.
- **`hyperflow` metadata is preserved and re-authored.** The loader-contract JSON entry from the source already survived the source-metadata merge; quantization now honestly rewrites its `weights` field to the output layout and stamps `requantized_by` so the contract matches the file.
- Tests: `tests/test_hyperflow_preservation.py` (3): keeps across all H3 base formats, metadata rewrite, absent-stays-absent.

## 2026.11.13-diffusers-lora (unreleased, uncommitted)

- New **Diffusers → H3** mode: converts a Diffusers-layout MiniMax H3 LoRA (any `transformer.`/PEFT prefix, `lora_A/B` or `lora_down/up`, with or without `.alpha`) into a ComfyUI-native H3 LoRA. Fuses q/k/v onto `qkv_proj` (`[q;k;v]`), swaps the `mlp.fc1` row halves, maps `token_refiner`, and applies curve branches only on `adaln_t_table` checkpoints. Unknown/unpaired keys and shape mismatches are reported in the dry run and abort without writing; already-native input is rejected.
- Optional **SVD re-rank** output (energy target, max rank) with per-module rank/energy/relative error plus median/worst recap in console and recipe; exact output is bit-identical to the DEAS converter. BF16, streamed module-by-module, no overwrite.
- Real runs on `Open-H3-Regenerate-2K` (258 modules, fused rank 1728/576): exact estimate 8.77 GiB; energy 0.999 → 2.35 GiB, median 3.15 % / worst 3.16 % relative error (94 s CPU); energy 0.999 + rank cap 512 → 2.24 GiB, worst 5.92 %, 2 modules over 5 %, 20 at the cap. Both SVD files load in ComfyUI (meta model from the graviton header) with 258/258 targets patched and 0 unloaded keys. Inference/render quality not verified.
- Fix: `core/adapter_factorization.py` clamped no longer; `energy=1.0` could pick one rank past the spectrum and crash.
- Cross-stack: engine `core/h3_diffusers_lora_engine.py`, bridge command, Go route `POST /api/h3/diffusers-lora` (H3Request gains `checkpoint_path`, `curve_path`, `representation`, `energy`, `max_rank`; `validateH3Request` now takes the command name), UI mode with persisted controls and recipe reload, README + `doc/api-endpoints.md`.
- Tests: `tests/test_h3_diffusers_lora_engine.py` (12), `tests/web_h3_diffusers.test.js` (6), `tests/test_h3_bridge_ui.py`, `internal/app/h3_api_test.go`; plus a headless Playwright check of the mode, Start payload and reload persistence.
- Not committed: the tree also contains a concurrent GRAVITON workstream; split or land the two separately.

## 2026.11.13-graviton.1

- Add opt-in GRAVITON to the existing checkpoint-bake and reusable-adapter composition workflows, API and persisted algorithm selectors.
- Bake reconstructed signed delta sums into floating checkpoints across all existing architecture presets; retain dedicated preservation profiles and non-matrix additive updates.
- Compose adapters with explicitly labelled base-free spectral GRAVITON, followed by the existing LoRA/LoKr factorization and numerical error reporting. No checkpoint polar geometry is claimed without a base.
- Record semantics and reloadable settings in recipes; fail closed on unsupported/unmatched tensors, non-finite arithmetic and incompatible baking controls.
- Bind converted H3 curve adapters to the checkpoint AdaLN table using canonical F32 SHA-256; reject unequal q/k/v output rows even when their fused shape matches.
- Allow nonwriting H3 dry runs from restored recipes with existing output/sidecar, retaining write-time no-overwrite and input validation. Clarify all-preset GRAVITON support versus four dedicated profiles, and exact fusion versus BF16 storage rounding.
- Keep existing additive/consensus defaults, per-layer CPU/CUDA fallback, streamed SSE progress, staged atomic artifacts and cancellation ownership.
- Add CPU numerical/architecture/atomicity regressions, isolated real HTTP→bridge→compose→bake tests, and UI/settings/recipe tests. No inference-quality guarantee.

The application uses the rebuilt executable's SHA-256 prefix as its runtime version; this release marker tracks the feature contract without replacing that binary identity scheme.
