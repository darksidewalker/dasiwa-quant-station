# Model Merge (model-level)

Back to [README](../README.md)

Model-level recipes — not LoRA math. Currently two, both MiniMax H3.

## Hybrid MiniMax H3 (`h3_hybrid`)

Switch to **Model Merge** mode. The base (fl2va) checkpoint is the one picked
in the Source panel; the sidebar Model Merge section holds the recipe selector
and the second-checkpoint (ref2va) picker. Output name = the **Display &
Output Name** field in the Source panel.

- **Base** = fl2va checkpoint (all tensors)
- **Overlay** = ref2va checkpoint. In the default **Whole blocks** mode,
  blocks 30–49 contribute their entire AdaLN weight and bias; change the
  inclusive range or select any set of blocks in the 50-block grid.
- **Modality rows (experimental)** uses that same range/grid for *video* AdaLN
  rows. Independently choose Ref2VA *audio* and *text* rows from every block,
  only the selected blocks, or no blocks (FL2VA). The initial split settings
  keep Ref2VA audio and text rows in every block. Optionally copy the Ref2VA
  `final_layer.audio_out` weight and bias together; the separate final AdaLN
  checkbox affects **both** output streams, not just audio. Unlike whole-block
  selection, row selection requires matching, floating-point, unquantized
  AdaLN weight and bias tensors in both checkpoints. Unsupported layouts fail
  before output is written. No AdaLN values are interpolated.

Selection order doesn't matter — the engine auto-detects roles from filenames
(fl2va/ref2va markers). Pruned and full H3 variants are supported when the two
inputs have matching AdaLN shapes. The output carries `minimax_h3_hybrid=baked`,
source names, the exact selected video/audio/text blocks in metadata, and a
companion `.txt` containing the effective recipe. Dry runs report the selection
without writing files. **These modality settings are research ablations, not
verified full Ref2VA audio-preservation presets.** Compare renders with the
unmodified Ref2VA and FL2VA under identical reference conditioning, connected
video/audio VAEs, prompt, seed, and sampler; score voice/audio adherence,
visual-reference adherence, and output quality separately.

## Delta-fused MiniMax H3 (`h3_delta`)

Fuses the full `ref2va − fl2va` weight delta back into the fl2va base so one
partition can serve both keyframe (fl2va) and reference (ref2va)
conditioning — the same recipe as
`diffusers-modular/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024`, computed
locally instead of downloaded. Works on **both** pruned (adaln_t_table) and
full (time_embedder) key sets; the variant is auto-detected from the base.

W = fl2va + strength · Δ, with Δ = ref2va − fl2va:

- **Delta rank `0` (exact)** — every tensor gets the full delta. Output is
  ≈ ref2va-equivalent on the reference path and pristine fl2va where the
  delta is zero (keyframe path untouched). Largest output (~38 GB).
- **Delta rank `N` (SVD-capped)** — randomized SVD of the delta on the
  2-D trunk matrices (attention qkv/out projections, MLP fc1/fc2, adaln
  weights, final out projections, token_refiner), capped at rank N;
  incompressible families are applied **exactly** and are never
  rank-limited: RMS/layer norms (210), biases, the timestep conditioning
  table (`adaln_t_table` pruned / `time_embedder.*` full), and
  `rope.inv_freq`. The output header stores a per-family captured-energy
  report (`h3_delta_energy`), e.g. at rank 1024 the trunk attention
  deltas keep ~0.8 of their energy — the load-bearing bulk, with the
  exact families carrying the rest (matches the upstream r1024 recipe).

**Strength** scales the delta (1.0 = full delta; 0.5 blends halfway back
to the base; 0 is treated as unset → 1.0). Cap it at ±3.0 like LoRA
merges.

Both files must share the same key set (same shape + dtype per key); the
delta recipe fails closed otherwise. Output keeps the base key/shape/dtype
set and carries `minimax_h3_delta=baked` + `h3_delta_{mode,rank,strength,variant}`
+ provenance. All runs stream tensor-by-tensor (SVD mode does one
deterministic energy pre-pass first so the header is complete before any
bytes are written).
