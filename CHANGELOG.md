# Changelog

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
