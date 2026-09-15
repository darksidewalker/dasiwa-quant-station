# LoRA Merge Strategies

Back to [README](../README.md)

Quant Station has two distinct adapter workflows:

- **LoRA Merge** bakes adapters into a selected base checkpoint. `Additive` is conventional weighted addition. `Consensus` reconstructs effective deltas, blends contributors targeting the same tensor, then adds one consensus delta to the base.
- **Merge LoRAs** needs no base checkpoint. It consensus-merges two or more adapters and factorizes the result into a reusable standard LoRA, direct LoKr, or Auto mixed adapter.

## Consensus-Weighted Blending

For every row, CWB computes a median consensus direction, cosine similarity for every contributor, rejects similarities below the preset threshold, raises accepted similarity to `power_alpha`, optionally applies the diversity term, normalizes weights, blends, and optionally restores the contributors' average row norm. If every contributor is rejected, it falls back to equal weights. A layer supplied by one adapter is kept as a singleton; missing layers are not treated as zero contributors.

| Preset | Threshold | Alpha | Diversity beta | Norm rescale | Comfort bandpass |
|---|---:|---:|---:|:---:|:---:|
| Balanced | 0.00 | 2.00 | 4.00 | yes | yes |
| Conservative | 0.35 | 1.25 | 0.00 | no | no |
| Neutral | 0.00 | 2.00 | 0.00 | yes | no |

Useful starting defaults, not universal quality claims:

| Architecture | Consensus | Layer strategy | Adapter strength | Global | Strict | Dry-run |
|---|---|---|---:|---:|:---:|:---:|
| MiniMax H3 | Balanced | Balanced | 1.0 | 1.0 | yes | yes |
| LTX-2.3 | Conservative | All | 1.0 | 1.0 | yes | yes |

Alpha normalization, per-adapter strength, global strength, and architecture strategy are applied before consensus. Effective strength is limited to ±3. Adaptive scaling and Krea 2 unchain are intentionally unavailable with checkpoint consensus because they alter the same delta semantics.

## Architecture Strategies

**MiniMax H3 strategies** (Balanced, Motion, Visuals): classify `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, `mlp.fc2`, token-refiner, structural/preserved, and other keys. Structural AdaLN/modulation/norm/rope and baked companion tensors remain untouched.

**LTX-2.3 types** (All, Video, Audio): classify video, audio, cross-modal bridge, caption, patchify/output, norm, and other tensors. All follows normal live-LoRA behavior; Video excludes audio keys; Audio includes audio and A/V bridge keys.

**WAN 2.2 strategies** (Balanced, Motion, Visuals): preserve modulation, patch embedding, norms, and baked companions. WAN has no Audio strategy.

**Krea 2 strategies** (Balanced, Style, Content, Detail): Style favors attention; Content favors feed-forward; Detail applies a mild global boost. Structural and normalization tensors remain preserved.

## Adapter Input and Output

Inputs may be standard safetensors LoRA (`lora_A/B` or `lora_down/up`), direct LoKr (`lokr_w1/w2`), or ComfyUI `.diff`. Direct LoKr uses `delta = kron(w1, w2)` and ignores stored alpha for ComfyUI parity. Factorized LoKr (`lokr_w1_a/b`, `lokr_w2_a/b`, optional `lokr_t2`) is reported but unsupported.

Standard LoRA output writes canonical `.lora_A.weight` and `.lora_B.weight` tensors. Scale is absorbed into the factors, so no alpha is written. SVD chooses the smallest rank meeting the requested Frobenius-energy target, bounded by Max Rank when nonzero; the cap can prevent reaching the target.

Direct LoKr output writes `.lokr_w1` and `.lokr_w2` with no alpha. A general consensus delta is not necessarily one Kronecker product, so Quant Station computes the nearest single Kronecker product for an explicit factor layout and reports relative reconstruction error. The layout comes from a compatible direct-LoKr input. Forced LoKr fails rather than guessing when no anchor exists. Auto uses direct LoKr for anchored layers and standard LoRA otherwise; verify mixed-layout loading in the target ComfyUI version before distribution.

All composition output is written atomically. Dry-run writes no output and reports layer counts, selected output kinds, skipped mismatches, retained energy, and reconstruction error. Detailed local paths and settings are stored in the adjacent recipe, not embedded in model metadata.

## Recipe Reload

Every quantization, checkpoint merge, and adapter composition writes a human-readable `.txt` recipe alongside its output. Use **Load Recipe** to restore a previous run. Composition recipes start with `DaSiWa LoRA Compose Recipe` and record all source adapters, strengths, consensus settings, output policy, and per-layer factorization report.

## References

The consensus design is based on silveroxides' Consensus-Weighted Blending implementation and informed by TIES-Merging and DARE. Musubi and Kohya are conventional additive-merge parity references. Quant Station reconstructs fixed-coordinate deltas before consensus so standard LoRA, direct LoKr, and `.diff` inputs can interoperate; its output is therefore not byte-equivalent to silveroxides' latent-factor LoRA output.
