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

    def test_minimax_h3_preserve_patterns_match_full_raw_and_prefixed_keys(self):
        config, _ = build_layer_config_dict("MiniMax H3", "FP8")

        # Structural / modulation / norm layers -> skip (source precision).
        keys = (
            "blocks.0.adaln_proj.linear.weight",
            "blocks.49.adaln_proj.linear.bias",
            "blocks.0.attn.q_norm.weight",
            "blocks.0.attn.k_norm.weight",
            "blocks.0.norm1.weight",
            "blocks.0.norm2.weight",
            "adaln_t_table",
            "time_embedder.proj_in.weight",
            "time_embedder.proj_out.weight",
            "final_layer.video_out.weight",
            "final_layer.audio_out.weight",
            "final_layer.norm.weight",
            "token_refiner.blocks.0.attn.qkv_proj.weight",
            "token_refiner.final_norm.weight",
            "video_patch_proj.weight",
            "audio_patch_proj.weight",
            "condition_proj.weight",
            "rope.inv_freq",
        )
        for key in keys:
            self.assertEqual(self._action_for(config, key), {"skip": True}, key)
            self.assertEqual(self._action_for(config, f"model.diffusion_model.{key}"),
                             {"skip": True}, f"prefixed {key}")

        # The four heavy linears must NOT be skipped (base format applies).
        heavy = (
            "blocks.0.attn.qkv_proj.weight",
            "blocks.0.attn.out_proj.weight",
            "blocks.0.mlp.fc1.weight",
            "blocks.0.mlp.fc2.weight",
        )
        for key in heavy:
            self.assertNotEqual(self._action_for(config, key), {"skip": True}, key)

    def test_minimax_h3_rescue_is_empty(self):
        config, summary = build_layer_config_dict("MiniMax H3", "NVFP4")
        self.assertEqual(summary["rescue_count"], 0)

    # --- MiniMax H3 NVFP4 HQ mixed profile -------------------------------
    # Verified plan (DmitryDB NVFP4-HQ, FL2VA+Ref2VA identical): of the 200
    # main-matrix heavy linears, 170 are NVFP4-packed and 30 stay at source
    # precision: attn.out_proj in 27 blocks + mlp.fc2 in 3 blocks.

    HQ_OUTPROJ = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                  17, 19, 20, 27, 38, 43, 44, 45, 46, 47, 49)
    HQ_FC2 = (39, 45, 49)

    def test_minimax_h3_nvfp4_hq_keeps_planned_heavy_linears_at_source_precision(self):
        config, summary = build_layer_config_dict("MiniMax H3", "NVFP4 HQ")
        self.assertEqual(summary["base_format"], "nvfp4")
        self.assertIn("30 heavy linears", summary["keep_action"])

        # Planned out_proj blocks -> skip
        for b in self.HQ_OUTPROJ:
            key = f"blocks.{b}.attn.out_proj.weight"
            self.assertEqual(self._action_for(config, key), {"skip": True}, key)
            self.assertEqual(
                self._action_for(config, f"model.diffusion_model.{key}"),
                {"skip": True}, f"prefixed {key}",
            )
        # Planned fc2 blocks -> skip
        for b in self.HQ_FC2:
            key = f"blocks.{b}.mlp.fc2.weight"
            self.assertEqual(self._action_for(config, key), {"skip": True}, key)

    def test_minimax_h3_nvfp4_hq_packs_unplanned_heavy_linears(self):
        config, _ = build_layer_config_dict("MiniMax H3", "NVFP4 HQ")

        def packed(key):
            return self._action_for(config, key) == {"format": "nvfp4"}

        # All 50 qkv_proj + all 50 fc1 stay packed.
        for b in range(50):
            self.assertTrue(packed(f"blocks.{b}.attn.qkv_proj.weight"))
            self.assertTrue(packed(f"blocks.{b}.mlp.fc1.weight"))
        # out_proj: packed outside the 27-block plan.
        for b in range(50):
            if b not in self.HQ_OUTPROJ:
                self.assertTrue(packed(f"blocks.{b}.attn.out_proj.weight"), b)
        # fc2: packed outside the 3-block plan.
        for b in range(50):
            if b not in self.HQ_FC2:
                self.assertTrue(packed(f"blocks.{b}.mlp.fc2.weight"), b)
        # Structural layers still skipped.
        self.assertEqual(
            self._action_for(config, "blocks.0.adaln_proj.linear.weight"),
            {"skip": True},
        )

    def test_minimax_h3_plain_nvfp4_stays_pure_all_200_packed(self):
        config, _ = build_layer_config_dict("MiniMax H3", "NVFP4")
        for b in self.HQ_OUTPROJ:
            key = f"blocks.{b}.attn.out_proj.weight"
            self.assertNotEqual(self._action_for(config, key), {"skip": True}, key)
        for b in self.HQ_FC2:
            key = f"blocks.{b}.mlp.fc2.weight"
            self.assertNotEqual(self._action_for(config, key), {"skip": True}, key)
        # 200 packed heavy linears = default base format applies to all.

    def test_minimax_h3_nvfp4_hq_preserve_count_matches_plan(self):
        _, plain_summary = build_layer_config_dict("MiniMax H3", "NVFP4")
        _, hq_summary = build_layer_config_dict("MiniMax H3", "NVFP4 HQ")
        # HQ adds exactly 2 pattern entries (out_proj plan + fc2 plan).
        self.assertEqual(
            hq_summary["preserve_count"], plain_summary["preserve_count"] + 2
        )
        from core.layer_config_builder import H3_NVFP4_HQ_LAYER_PLAN
        self.assertEqual(len(H3_NVFP4_HQ_LAYER_PLAN), 30)


if __name__ == "__main__":
    unittest.main()
