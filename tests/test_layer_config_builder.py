import re
import unittest

from core.layer_config_builder import build_layer_config_dict


class LayerConfigBuilderTests(unittest.TestCase):
    def _action_for(self, config, key):
        for pattern, action in config.items():
            if pattern.startswith("_"):
                continue
            if re.fullmatch(pattern, key):
                return action
        return config["_default"]

    def test_ltx23_preserve_patterns_match_full_raw_and_prefixed_keys(self):
        config, _ = build_layer_config_dict("LTX-2.3", "FP8")

        keys = (
            "adaln_single.linear.weight",
            "audio_adaln_single.linear.weight",
            "audio_prompt_adaln_single.linear.weight",
            "audio_embeddings_connector.0.weight",
            "video_embeddings_connector.0.weight",
            "caption_projection.linear.weight",
            "patchify_proj.weight",
            "proj_out.weight",
            "audio_patchify_proj.weight",
            "audio_proj_out.weight",
            "transformer_blocks.0.scale_shift_table",
            "transformer_blocks.0.attn1.to_gate_logits.weight",
            "transformer_blocks.0.attn1.q_norm.weight",
            "transformer_blocks.0.attn1.k_norm.weight",
        )
        for key in keys:
            self.assertEqual(self._action_for(config, key), {"skip": True}, key)
            prefixed = f"model.diffusion_model.{key}"
            self.assertEqual(self._action_for(config, prefixed), {"skip": True}, prefixed)

    def test_wan22_preserve_patterns_match_full_raw_and_prefixed_keys(self):
        config, _ = build_layer_config_dict("WAN 2.2", "FP8")

        keys = (
            "blocks.0.modulation.weight",
            "patch_embedding.weight",
            "text_embedding.0.weight",
            "time_projection.weight",
            "time_embedding.0.weight",
            "img_emb.weight",
            "head.weight",
        )
        for key in keys:
            self.assertEqual(self._action_for(config, key), {"skip": True}, key)
            prefixed = f"model.diffusion_model.{key}"
            self.assertEqual(self._action_for(config, prefixed), {"skip": True}, prefixed)

    def test_ltx23_nvfp4_matches_official_block_preserve_policy(self):
        config, _ = build_layer_config_dict("LTX-2.3", "NVFP4")

        def action_for(key):
            for pattern, action in config.items():
                if pattern.startswith("_"):
                    continue
                if re.fullmatch(pattern, key):
                    return action
            return config["_default"]

        # Official Lightricks/LTX-2.3-nvfp4 keeps first two and last two
        # transformer blocks BF16, and packs the middle blocks as NVFP4/U8.
        for block in (0, 1, 46, 47):
            for suffix in (
                "attn1.to_q.weight",
                "attn1.to_v.weight",
                "attn1.to_out.0.weight",
                "ff.net.0.proj.weight",
                "ff.net.2.weight",
                "audio_attn1.to_q.weight",
                "audio_ff.net.2.weight",
                "attn1.to_gate_logits.weight",
            ):
                key = f"model.diffusion_model.transformer_blocks.{block}.{suffix}"
                self.assertEqual(action_for(key), {"skip": True}, key)

        for block in (2, 10, 45):
            for suffix in (
                "attn1.to_q.weight",
                "attn1.to_v.weight",
                "attn1.to_out.0.weight",
                "ff.net.0.proj.weight",
                "ff.net.2.weight",
                "audio_attn1.to_q.weight",
                "audio_ff.net.2.weight",
            ):
                key = f"model.diffusion_model.transformer_blocks.{block}.{suffix}"
                self.assertEqual(action_for(key), {"format": "nvfp4"}, key)

    def test_ltx23_fp8_does_not_apply_nvfp4_official_block_preserve(self):
        config, _ = build_layer_config_dict("LTX-2.3", "FP8")

        def action_for(key):
            for pattern, action in config.items():
                if pattern.startswith("_"):
                    continue
                if re.fullmatch(pattern, key):
                    return action
            return config["_default"]

        key = "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"
        self.assertEqual(action_for(key), {"format": "float8_e4m3fn"})

    def test_wan22_nvfp4_rescue_patterns_match_14b_moe_high_low_keys(self):
        config, _ = build_layer_config_dict("WAN 2.2", "NVFP4")

        def action_for(key):
            for pattern, action in config.items():
                if pattern.startswith("_"):
                    continue
                if re.fullmatch(pattern, key):
                    return action
            return config["_default"]

        fp8_rescue = {"format": "float8_e4m3fn", "scaling_mode": "tensor"}

        # WAN 2.2 I2V/T2V 14B MoE high and low checkpoints use 40 transformer
        # blocks (0..39). These are the NVFP4-mixed rescue targets observed in
        # public 14B I2V high/low checkpoints; this is intentionally not based
        # on the smaller TI2V 5B checkpoint layout.
        for key in (
            "blocks.0.self_attn.v.weight",
            "blocks.39.self_attn.v.weight",
            "blocks.0.cross_attn.v.weight",
            "blocks.39.cross_attn.v.weight",
            "blocks.0.ffn.2.weight",
            "blocks.39.ffn.2.weight",
            "model.diffusion_model.blocks.0.self_attn.v.weight",
            "model.diffusion_model.blocks.39.cross_attn.v.weight",
            "model.diffusion_model.blocks.39.ffn.2.weight",
        ):
            self.assertEqual(action_for(key), fp8_rescue, key)

        for key in (
            "blocks.0.self_attn.q.weight",
            "blocks.39.self_attn.k.weight",
            "blocks.0.self_attn.o.weight",
            "blocks.39.cross_attn.q.weight",
            "blocks.0.cross_attn.k.weight",
            "blocks.39.cross_attn.o.weight",
            "blocks.0.ffn.0.weight",
            "blocks.39.ffn.0.weight",
        ):
            self.assertEqual(action_for(key), {"format": "nvfp4"}, key)

        # GGUF sensitivity code strips .weight before applying the same pattern
        # list, so the rescue regex must also match stripped stems.
        stripped_stems = (
            "blocks.0.self_attn.v",
            "blocks.39.cross_attn.v",
            "blocks.39.ffn.2",
        )
        rescue_patterns = [
            pattern
            for pattern, action in config.items()
            if action == fp8_rescue
        ]
        for stem in stripped_stems:
            self.assertTrue(any(re.search(pattern, stem) for pattern in rescue_patterns), stem)

    def test_krea2_nvfp4_patterns_match_raw_tensor_keys(self):
        config, _ = build_layer_config_dict("Krea 2", "NVFP4")

        def action_for(key):
            for pattern, action in config.items():
                if pattern.startswith("_"):
                    continue
                if re.fullmatch(pattern, key):
                    return action
            return config["_default"]

        for key in (
            "first.weight",
            "first.bias",
            "last.linear.weight",
            "last.linear.bias",
            "tproj.1.weight",
            "tmlp.0.weight",
            "txtmlp.1.weight",
            "txtfusion.projector.weight",
            "blocks.0.attn.qknorm.qnorm.scale",
            "blocks.0.prenorm.scale",
            "txtfusion.layerwise_blocks.0.attn.qknorm.qnorm.scale",
            "txtfusion.layerwise_blocks.0.prenorm.scale",
            "model.diffusion_model.first.weight",
            "model.diffusion_model.last.linear.weight",
            "model.diffusion_model.tproj.1.weight",
            "model.diffusion_model.tmlp.0.weight",
            "model.diffusion_model.txtmlp.1.weight",
            "model.diffusion_model.txtfusion.projector.weight",
        ):
            self.assertEqual(action_for(key), {"skip": True}, key)

        for key in (
            "blocks.0.attn.wq.weight",
            "blocks.0.attn.wk.weight",
            "blocks.0.attn.wv.weight",
            "blocks.0.attn.wo.weight",
            "blocks.0.attn.gate.weight",
            "blocks.0.mlp.gate.weight",
            "blocks.0.mlp.up.weight",
            "blocks.0.mlp.down.weight",
            "txtfusion.layerwise_blocks.0.attn.wq.weight",
            "txtfusion.layerwise_blocks.0.mlp.up.weight",
            "txtfusion.refiner_blocks.0.attn.wq.weight",
            "model.diffusion_model.blocks.0.attn.wq.weight",
            "model.diffusion_model.blocks.0.mlp.up.weight",
            "model.diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.weight",
        ):
            self.assertEqual(action_for(key), {"format": "nvfp4"}, key)


if __name__ == "__main__":
    unittest.main()
