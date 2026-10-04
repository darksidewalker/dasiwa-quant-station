# MiniMax H3 pruning and adapters

Back to [README](../README.md)

Pruning, quantization, Turbo distillation and adapter extraction are different operations. Pruning replaces the full-width time-conditioning projections with a small shared curve coordinate system. Quantization changes the storage/compute representation of eligible weights. A Turbo LoRA is a trained update for a few-step sampling workflow. Extraction calculates an adapter from compatible checkpoints; it does not train a Turbo model.

## Coordinate compatibility

A pruned H3 checkpoint contains `adaln_t_table`. Its coordinate convention is part of its adapter interface: two tables can describe equally accurate curve approximations while using different signs or rotations. Matching tensor shapes alone does not establish LoRA compatibility.

For a reference-compatible fold, use the original full checkpoint and the exact corresponding pruned reference. Do not mix FL2VA and Ref2VA references or assume a hybrid shares the same time-conditioning source merely because its filename mentions H3. Validate the recovered curve and folded projections against the actual reference.

An independently derived SVD fold has its own convention. Existing pruned-trained adapters are not automatically portable to it. Preserve the exact generated table and identify the target when converting adapters.

## Affine offsets are required

If the full time curve satisfies `x(t) ≈ c + V q(t)`, the folded linear is:

```text
W_pruned = W V
b_pruned = b + W c
```

A full-trained LoRA `B A` therefore contributes both a projected factor update and a constant bias update:

```text
A_pruned = A V
B_pruned = B
bias_delta = B A c
```

Adapter alpha/rank and user strengths apply exactly once. Dropping the constant does not preserve the original adapter. A reusable converted adapter needs a representation that carries both terms, such as standard factors plus `.diff_b`, or explicit `.diff` plus `.diff_b`.

Changes to the time embedder require a different contract. They can affect every AdaLN projection, including projections absent from the input adapter. A table-changing delta must not be advertised as an ordinary freely stackable LoRA: applying a folded checkpoint difference at a different strength does not generally reproduce folding the corresponding full-space interpolation.

## EMA600 is a trained checkpoint

`minimax_h3_turbo_v4_step600_ema.safetensors` is a Turbo adapter checkpoint, not a new merge or extraction algorithm. The name identifies a training checkpoint with EMA weights. Its full-width AdaLN factors need conversion before use with a pruned base unless the runtime loader already implements that conversion.

Use additive application as the baseline for reproducing a trained Turbo update. Consensus blends contributors and can alter that update. Baking also introduces storage-rounding effects; compare it against runtime application rather than assuming equivalence from matched-key counts.

Sampler and audio/video schedule handling remain runtime requirements. A baked checkpoint does not serialize the workflow's sampler configuration. Consult the [upstream Turbo loader](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo) and [adapter model card](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) for the applicable workflow.

## INT8 ConvRot reference commands

The supplied silveroxides commands for a full checkpoint and a pruned checkpoint are:

```text
ctq -i ./minimax_h3_bf16.safetensors -o ./minimax_h3_int8_convrot.safetensors --comfy_quant --save-quant-metadata  --verbose VERBOSE --low-memory --minimaxh3 --int8 --scaling_mode row --convrot --convrot-group-size 256 --simple
```

```text
ctq -i ./minimax_h3_pruned_bf16.safetensors -o ./minimax_h3_pruned_int8_convrot.safetensors --comfy_quant --save-quant-metadata  --verbose VERBOSE --low-memory --minimaxh3 --int8 --scaling_mode row --convrot --convrot-group-size 256 --simple --exclude_layers "(adaln_t_table|adaln_proj)"
```

`ctq` and `convert_to_quant` are executable names; the generated argument semantics matter. Pruned coordinates and AdaLN projections must remain protected. Quant Station's conservative structural policy also preserves full-model AdaLN, whereas the full reference command has no additional AdaLN exclusion. Keep those policies distinguishable when comparing file sizes and quality.

The native converter also finalizes the dtype of unquantized 2D weights. A layer-config `skip` or `--exclude_layers` does not by itself preserve an F32 time-embedding weight: native Full H3 output can cast it to BF16. The structural Full H3 INT8 policy therefore adds `--preserve-layers "(time_embedder)"` to retain its source dtype. This is not added to the opt-in upstream recipe.

Tiny CPU conversions exercised all three policies with the installed converter: Full upstream quantized eligible attention and AdaLN matrices to INT8 ConvRot; Pruned upstream kept the table and AdaLN tensor payloads unchanged; structural Full kept AdaLN and F32 time weights unchanged while attention was INT8 ConvRot. These checks establish exclusion, dtype and metadata behavior on fixtures, not full-model quality.

Bake into an original floating checkpoint before quantizing when checkpoint baking is required. Do not add unrotated LoRA deltas directly to stored ConvRot weights; their coordinates differ. Dequantization alone cannot recover the original pre-quantized floating model.

## Available workflows and limits

| Operation | Required input | Output / compatibility |
|---|---|---|
| Reference pruning | Original floating full H3 plus its exact matching pruned reference | Folded checkpoint bound to the reference table; curve and sampled projection consistency checked |
| Independent pruning | Original floating full H3 | New SVD table/basis; no automatic compatibility with existing pruned adapters |
| Full adapter conversion | Original full H3, exact pruned target, standard 2D LoRA or direct patches | Projected factors or `.diff`, plus required F32 `.diff_b`; table hash retained |
| Full checkpoint extraction | Compatible base and modified floating checkpoints | BF16 SVD factors and F32 bias patches; retained energy and saved-factor rounding reported |
| Pruned H3 extraction | Original full base, modified full checkpoint, exact pruned target | AdaLN `.diff`/`.diff_b` in target coordinates; changed time embedders rejected |
| Adapter composition | Compatible supported adapters | Additive or consensus output; bound AdaLN contributors must share a table hash |
| Complete H3 Turbo baking | Floating H3 base plus compatible adapter; additive and adaptive scaling OFF | Explicit completeness audit; protected refiner omissions remain partial |

Gauge validation and SVD stay on CPU in double precision where needed. Pruning and conversion project bounded row chunks on the selected CUDA device when headroom permits; unavailable CUDA or allocation failure falls back to CPU and is counted in the recipe. Untouched checkpoint tensors are byte-copied, and output tensors are spooled instead of retaining a complete checkpoint dictionary. Exact SVD still requires a layer-sized matrix and workspace.

Recipes are stored beside the artifact as `output.safetensors.txt` and restore the effective workflow settings and paths. Cancellation cleanup applies to Go-owned jobs. An externally SIGKILLed standalone Python invocation cannot run its context-manager cleanup.

Unsupported adapter tensors, inconsistent shapes, conflicting coordinates and nonfinite updates are errors or explicit omissions, never evidence of a complete Turbo application. Standalone conditioning-bridge weights are not automatically LoRA deltas. Quantization Error-Correction LoRA remains unexposed until runtime-coordinate consumption is verified.

## Verification and limits

- Compare source and folded modulation functions at grid endpoints and off-grid timesteps; include the final AdaLN and bias-only changes.
- Check unchanged tensor payloads byte-for-byte and preserve loader-critical configuration.
- Check that the runtime consumes every exported factor and bias patch. A numerical exporter test is not a runtime-loader test.
- For inference, use the same checkpoint ancestry, prompt, seed, sampler, schedule, steps, resolution, frame count and audio/video components.
- Compare visual fidelity, audio quality and reference fidelity separately. A low matrix reconstruction error is not a visual-quality guarantee.
- Record conversion time, peak memory, output size, precision and reconstruction error. Do not call an independent fold ecosystem-compatible or a quantized bake lossless without evidence.

Quantization Error-Correction LoRA is a separate optional workflow. Its residual must be expressed in the coordinate convention consumed by the runtime, including activation rotation. Availability of an upstream extraction flag alone is insufficient to establish this compatibility.
