# Model Merge (model-level) — h3_hybrid

Back to [README](../README.md)

A model-level merge — not LoRA math. Currently one recipe: the Hybrid
MiniMax H3 recipe.

## Hybrid MiniMax H3 (`h3_hybrid`)

Switch to **Model Merge** mode. The base (fl2va) checkpoint is the one picked
in the Source panel; the sidebar Model Merge section holds the recipe selector
and the overlay (ref2va) picker. Output name = the **Display & Output Name**
field in the Source panel.

- **Base** = fl2va checkpoint (all tensors)
- **Overlay** = ref2va checkpoint (`blocks.{25..49}.adaln_proj.linear.{bias,weight,weight_scale}`)

Selection order doesn't matter — the engine auto-detects roles from filenames
(fl2va/ref2va markers). Works for both pruned (932 keys) and full (1035 keys)
variants. Output carries `minimax_h3_hybrid=baked` +
`base_model`/`overlay_model` provenance.
