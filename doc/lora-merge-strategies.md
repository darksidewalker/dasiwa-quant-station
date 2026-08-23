# LoRA Merge Strategies

Back to [README](../README.md)

Each architecture applies its own filter-based preset to tensor categories:

**LTX-2.3 types** (All, Video, Audio):
- Classifies tensors into: attn_qkv, attn_out, ff_in, ff_out, audio_attn, audio_attn_out, audio_to_video_attn, video_to_audio_attn, audio_ff_in, audio_ff_out, caption_projection, patchify_or_output, norm, other
- Preserves structural layers (adaln, gate logits, baked VAE/text/audio modules)
- All merges all non-preserved weights (normal ComfyUI LoRA load behavior)
- Video merges only weights without `audio` in their key. Audio merges every weight with `audio` in its key, including cross-modal bridge weights

**WAN 2.2 strategies** (Balanced, Motion, Visuals):
- Classifies tensors into: self_attn_qkv, self_attn_out, cross_attn_qkv, cross_attn_out, ffn_in, ffn_out, modulation, caption_projection, patchify_or_output, norm, other
- No Audio strategy (WAN 2.2 has no audio components)
- Preserves modulation.lin, patch_embedding, and baked companion modules
- Norm layers always get 0.0 multiplier (untouched)

**Krea 2 strategies** (Balanced, Style, Content, Detail):
- Classifies tensors into: attn_qkv, attn_out, attn_gate, ff_in, ff_out, text_fusion, structural, other
- Style boosts attention (qkv/out/gate), reduces text_fusion — for aesthetic/style LoRAs
- Content boosts feed-forward, moderates attention — for subject/content LoRAs
- Detail applies mild global boost — for quality/detail LoRAs
- Preserves modulation.lin, tproj, tmlp, txtmlp, first/last layers, txtfusion.projector, norm.scale, qknorm

Strength limit: effective strength (`global x per_lora`) capped at +/-3.0 to
prevent black images on Krea 2 gate tensors. Merge device: CPU/CUDA/auto with
VRAM headroom check and OOM fallback.

## Supported LoRA Formats

Standard `.safetensors` LoRAs and ComfyUI `.diff` format are both supported.

## Recipe Reload

Every quantization and LoRA merge writes a human-readable `.txt` recipe
alongside the output. Click **Load Recipe** in the UI to reload a previous
run's exact settings (source, output name, all LoRAs, formats, strategy,
strength) — useful for reproducing results or iterating on parameters.
