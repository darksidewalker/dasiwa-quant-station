# GRAVITON: Geodesic Rank-Adaptive Variance-Isolated Tensor Orthogonal Normalization

**A data-free weight-space fusion architecture for diffusion / DiT checkpoints**

Consolidated micro-whitepaper. Every statement below matches the
`h3_graviton` implementation in `core/model_merge_engine.py` (commit
`69412c0` or later). Lineage: reimplemented and extended from the
community PRISM recipe (`OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM`
HF card, no published paper); the pre-geodesic mediation order, the
Gavish-Donoho auto rank, the quantization floor and the float-output
decision are this implementation's, not the card's.

## Why GRAVITON?

Merging step-distilled turbo models, diverse finetunes, or quantized
checkpoints into base models frequently triggers variance collapse,
high-frequency distillation noise, and destructive weight interference.
GRAVITON resolves this by:

- **Full-spectrum RMT filtering** — separates deterministic structural
  updates from diffuse Marchenko-Pastur bulk noise via exact Gram-matrix
  eigendecomposition and Gavish-Donoho optimal hard thresholding
  (replacing arbitrary fixed-rank cutoffs; fixed-rank mode remains
  available as an explicit opt-out).
- **Pre-geodesic sign mediation** — resolves salient opposite-sign
  conflicts between the base and the denoised target **before** the
  spherical rotation, so the geodesic stays final.
- **Polar geodesic fusion + norm re-anchor** — decouples row-wise
  directional SLERP from log-space geometric magnitude interpolation,
  strictly bounded (`t ≤ 1.0`, no extrapolation), then re-anchors each
  row to the geometric norm of the **original** pair.
- **Comfy-Kitchen W6A8 input bridge** — natively ingests and dequantizes
  QuantStation W6A8-packed checkpoints (`weight_s_rel` / `weight_s_channel`
  companions, ConvRot handled by `comfy_kitchen` itself) behind a
  fail-closed opt-in (`allow_quantized_inputs`), with an analytical
  Marchenko-Pastur quantization floor (`tau_quant`) and forced GD-auto
  rank selection for those layers.
- **Decoupled float output** — merged artifacts are emitted as clean
  floating-point weights and handed to the pre-tested downstream W6A8
  conversion job instead of risking inline packer corruption.

## 1. Abstract & problem statement

Weight-space merging across heterogeneous training regimes — multi-step
bases fused with step-distilled accelerators (8-step Turbo/DMD/LCM), or
group-quantized checkpoints — shows four failure modes:

1. **Spectral tail contamination.** Step distillation concentrates
   structural trajectory changes into the top singular vectors of the
   weight delta while leaving discretization artifacts in the spectral
   tail. Fixed-rank truncation (e.g. rank-64) ignores aspect ratios: it
   under-fits wide layers and leaks noise from narrow projections.
2. **Post-geodesic trajectory warping.** Element-wise sign-conflict
   mediation applied *after* SLERP flips components off the
   hypersphere and destroys the interpolated angle (the upstream PRISM
   card orders its steps this way; GRAVITON does not).
3. **Timestep conditioning instability.** Scaling weight deltas by
   `alpha > 1.0` inside adaptive layer norm (`adaln`) and timestep
   conditioning causes variance drift across denoising steps —
   empirically saturated or black-frame output.
4. **Quantized manifold mismatch.** Group-quantized checkpoints carry
   Hadamard-rotated (ConvRot), bit-packed coordinates whose
   discretization noise inflates the delta spectrum; naive merging in
   packed space corrupts silently.

GRAVITON is a strictly ordered, calibration-free pipeline operating
tensor-by-tensor in canonical fp32, streaming (never holding two full
models in RAM).

**Scope / family policy (not optional math):** the spectral + polar path
applies to SVD-eligible trunk 2-D matrices only (attention/MLP/projection).
Anti-aligned AdaLN modulation weights get spectral denoise plus a purely
linear delta (per-row SLERP on them reproduces the black-video failure);
norms, biases, timestep tables, rope frequencies and all 1-D tensors stay
exact linear fusion `W = A + t(B − A)`. The family classification is
MiniMax-H3-shaped; porting the kernel to other architectures requires
re-deriving that policy.

## 2. Mathematical pipeline

Given per-key weights `A, B ∈ R^{m×n}` (`strength = t ∈ (0,1]` is the
weight of B; `t > 1` is clamped, extrapolation is refused).

### Stage 0 — W6A8 ingestion & dequantization bridge (opt-in)

Fail-closed by default: only F16/BF16/F32 are accepted unless
`allow_quantized_inputs` is set. Then QuantStation's exact W6A8 contract
is recognized header-side — I8 packed `.weight`, F8E4M3 `.weight_s_rel`,
F32 `.weight_s_channel`, per-layer U8 JSON `.comfy_quant` — and dequantized
by comfy-kitchen's own `AsymW4A8Int8Layout.dequantize`, which also undoes
the ConvRot block-Hadamard rotation (`Q = Q^T = H_b/√b`, verified roundtrip
relL2 ≈ 0.02). Any other quantized format (GPTQ/qzeros, generic int8
rowwise, foreign 6-bit bitstreams) stays fail-closed. From the scale
tensors the per-element quantization noise variance is recorded
analytically (uniform step noise):

```
sigma_q^2 = mean(S^2) / 12      (S = s_channel * s_rel per group)
```

### Stage 1 — Full-spectrum RMT split with analytical quantization floor

The delta `Δ = B − A` is split by the **exact** spectrum of the smaller
Gram matrix (`G = ΔᵀΔ = VΛVᵀ`, `σ_i = √max(λ_i, 0)`) — no randomized
sketch in auto mode, because the median estimator needs the noise bulk:

```
omega(beta) = 0.56 b^3 − 0.95 b^2 + 1.82 b + 1.43,  beta = min(m,n)/max(m,n)
tau_GD      = omega(beta) * sigma_med               (Gavish-Donoho 2014, unknown-noise)
tau_quant   = (sqrt(m) + sqrt(n)) * sqrt(sigma_qA^2 + sigma_qB^2)   (MP bulk edge)
tau         = max(tau_GD, tau_quant)                (tau_quant = 0 for float inputs)
V_keep      = { v_i : sigma_i > tau }
L           = (Delta V_keep) V_keepᵀ ,   R = Delta − L
```

Quantized layers force this GD-auto path (`rank = -1` semantics; the MP
floor needs the full spectrum, fixed rank cannot express it). Fixed rank N
(randomized SVD, Halko et al. 2011) remains available for float pairs.

The denoised target is

```
B_raw = A + L + gamma_halo * R ,   gamma_halo = 1 − 0.08 * (L energy capture)
```

with `gamma_halo ∈ [0.92, 1.0]` derived per tensor from the captured
energy — **not a user dial** — and additionally capped at **0.15**
whenever dequantization noise is present, so grid staircases in R never
survive into the merge.

### Stage 2 — Pre-geodesic sign & interference mediation

On `B_raw` (before any rotation): salient elements (`|B_raw − A| ≥` row
median of `|B_raw − A|`) whose sign conflicts with A take the
magnitude-weighted winner (`t·|B_raw|` vs `(1−t)·|A|`; TIES/EMR logic).
Because mediation precedes the SLERP, no post-rotation flip can warp the
geodesic.

### Stage 3 — Polar geodesic interpolation (per output row)

```
A_hat = A/|A|, B'_hat = B'/|B'|, theta = arccos(clamp(<A_hat,B'_hat>, −1+eps, 1−eps))
u     = [sin((1−t)theta) A_hat + sin(t theta) B'_hat] / sin(theta)     (LERP fallback for theta < 1e-4; negated A for the short path when cos < 0)
W_i   = u_i * exp((1−t) ln|A_i| + t ln|B'_i|) / |u_i|
```

### Stage 4 — Magnitude re-anchor, then float export

Each row is rescaled to the geometric norm target of the **original**
pair, `|A_i|^(1−t) · |B_i|^t` with B's pre-denoise row norms
(RegMean-style norm preservation — this step is PRISM's fourth and is
part of GRAVITON). Output is always floating point (BF16 for
dequantized inputs); a completed run instructs a follow-up W6A8
conversion job for packed targets. Companion tensors never leak into the
output, and header/offsets are rebuilt from the unpacked logical layout.
`h3_graviton_quant_diag` records per run how many singular components GD
alone kept versus the floor-adjusted effective keep.

## 3. Supported merge regimes & operational matrix

`t` is a user setting (UI/API, default 0.5, clamped to (0,1]).
`gamma_halo` is auto-derived — the ranges below are what the energy
heuristic yields, not presets to tune.

| Pairing | Anchor (A) | Target (B) | Recommended t | Behavior |
| --- | --- | --- | --- | --- |
| Base + Turbo (balanced) | base bf16 | 8-step turbo | 0.45–0.55 | base diversity + faster convergence |
| Turbo + Base (8-step lock) | 8-step turbo | base bf16 | 0.20–0.30 | keeps turbo schedule, injects base textures |
| Base + Base (finetunes) | finetune 1 | finetune 2 | 0.30–0.70 | RMT strips uncorrelated overfit |
| Turbo + Turbo | turbo 1 | turbo 2 | 0.40–0.60 | no AdaLN variance drift (alpha ≤ 1 enforced) |
| Float + W6A8 | bf16 | w6a8_g32 | 0.35–0.50 | MP floor blocks grid artifacts from L; output float |

**Honesty clause:** GRAVITON is a research recipe. There is no published
ablation; the recommended ranges are guidance, not measured optima. The
acceptance test for any merge is A/B renders against **both parents**.

## References (grounded analogs)

Gavish & Donoho, *The Optimal Hard Threshold for Singular Values is 4/√3*,
arXiv:1305.5870 (unknown-noise form, ω(β) polynomial); Marchenko-Pastur
bulk edge `(√m+√n)σ_q`; Halko et al. 2011 (randomized SVD, fixed-rank
mode); DARE arXiv:2311.03099 (soft rescale analogue behind gamma);
TIES arXiv:2306.01708 / EMR arXiv:2405.17461 (sign mediation, weighted
winner); mergekit SLERP arXiv:2403.13257 / 2406.16768; RegMean
(Jin et al. 2023, norm preservation); upstream recipe: OliviaRossi PRISM
HF model card (four described steps, no code, no paper).
