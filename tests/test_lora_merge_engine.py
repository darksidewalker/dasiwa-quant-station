import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from core.lora_merge_engine import run_lora_merge
from utils.lora_inspector import discover_lora_pairs, read_safetensors_manifest
from utils.ltx23_layer_profiles import classify_ltx23_key, is_ltx23_preserved_key
from utils.wan22_layer_profiles import classify_wan22_key, is_wan22_preserved_key


class LoraMergeEngineTests(unittest.TestCase):
    def test_lora_a_b_pair_maps_diffusion_model_prefix_to_base_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora = Path(tmp) / "style.safetensors"
            save_file({
                "diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight": torch.ones(2, 4),
                "diffusion_model.transformer_blocks.0.attn1.to_k.lora_B.weight": torch.ones(3, 2),
            }, str(lora))

            manifest = read_safetensors_manifest(str(lora))
            pairs = discover_lora_pairs(manifest)

            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].base_name, "diffusion_model.transformer_blocks.0.attn1.to_k")
            self.assertIn(
                "model.diffusion_model.transformer_blocks.0.attn1.to_k.weight",
                pairs[0].target_candidates,
            )
            self.assertEqual(pairs[0].rank, 2)

    def test_profiles_classify_audio_and_preserve_keys(self):
        self.assertEqual(
            classify_ltx23_key("model.diffusion_model.transformer_blocks.0.audio_attn1.to_k.weight"),
            "audio_attn",
        )
        self.assertTrue(is_ltx23_preserved_key("model.diffusion_model.adaln_single.linear.weight"))
        self.assertTrue(is_ltx23_preserved_key("model.diffusion_model.transformer_blocks.0.attn1.to_gate_logits.weight"))

    def test_dry_run_reports_match_without_writing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            lora = tmp / "motion.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "model.diffusion_model.transformer_blocks.0.attn1.to_k.weight": torch.zeros(3, 4),
            }, str(base))
            save_file({
                "diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight": torch.ones(2, 4),
                "diffusion_model.transformer_blocks.0.attn1.to_k.lora_B.weight": torch.ones(3, 2),
            }, str(lora))

            events = list(run_lora_merge({
                "base_path": str(base),
                "loras": [{"path": str(lora), "strength": 0.5}],
                "output_path": str(out),
                "strategy": "Balanced",
                "architecture": "LTX-2.3",
                "global_strength": 1.0,
                "adaptive": False,
                "dry_run": True,
                "strict_matching": True,
            }))

            text = "".join(e.get("text", "") for e in events)
            self.assertIn("matched=1", text)
            self.assertFalse(out.exists())

    def test_merge_applies_scaled_delta_and_skips_preserved_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            lora = tmp / "style.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "model.diffusion_model.transformer_blocks.0.attn1.to_k.weight": torch.zeros(3, 4),
                "model.diffusion_model.adaln_single.linear.weight": torch.zeros(3, 4),
            }, str(base))
            save_file({
                "diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight": torch.ones(2, 4),
                "diffusion_model.transformer_blocks.0.attn1.to_k.lora_B.weight": torch.ones(3, 2),
                "diffusion_model.adaln_single.linear.lora_A.weight": torch.ones(2, 4),
                "diffusion_model.adaln_single.linear.lora_B.weight": torch.ones(3, 2),
            }, str(lora))

            list(run_lora_merge({
                "base_path": str(base),
                "loras": [{"path": str(lora), "strength": 0.5}],
                "output_path": str(out),
                "strategy": "Balanced",
                "architecture": "LTX-2.3",
                "global_strength": 1.0,
                "adaptive": False,
                "dry_run": False,
                "strict_matching": True,
            }))

            merged = load_file(str(out))
            self.assertTrue(torch.allclose(
                merged["model.diffusion_model.transformer_blocks.0.attn1.to_k.weight"],
                torch.ones(3, 4),
            ))
            self.assertTrue(torch.allclose(
                merged["model.diffusion_model.adaln_single.linear.weight"],
                torch.zeros(3, 4),
            ))

    def test_each_lora_can_choose_its_own_strategy_in_one_merge_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            motion = tmp / "motion.safetensors"
            audio = tmp / "audio.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "model.diffusion_model.transformer_blocks.0.attn1.to_k.weight": torch.zeros(3, 4),
                "model.diffusion_model.transformer_blocks.0.audio_attn1.to_k.weight": torch.zeros(3, 4),
            }, str(base))
            save_file({
                "diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight": torch.ones(2, 4),
                "diffusion_model.transformer_blocks.0.attn1.to_k.lora_B.weight": torch.ones(3, 2),
            }, str(motion))
            save_file({
                "diffusion_model.transformer_blocks.0.audio_attn1.to_k.lora_A.weight": torch.ones(2, 4),
                "diffusion_model.transformer_blocks.0.audio_attn1.to_k.lora_B.weight": torch.ones(3, 2),
            }, str(audio))

            events = list(run_lora_merge({
                "base_path": str(base),
                "loras": [
                    {"path": str(motion), "strength": 0.5, "strategy": "Visuals"},
                    {"path": str(audio), "strength": 0.5, "strategy": "Audio"},
                ],
                "output_path": str(out),
                "strategy": "Balanced",
                "architecture": "LTX-2.3",
                "global_strength": 1.0,
                "adaptive": False,
                "dry_run": False,
                "strict_matching": True,
            }))

            text = "".join(e.get("text", "") for e in events)
            self.assertIn("strategy=Visuals", text)
            self.assertIn("strategy=Audio", text)
            merged = load_file(str(out))
            self.assertTrue(torch.allclose(
                merged["model.diffusion_model.transformer_blocks.0.attn1.to_k.weight"],
                torch.full((3, 4), 1.05),
            ))
            self.assertTrue(torch.allclose(
                merged["model.diffusion_model.transformer_blocks.0.audio_attn1.to_k.weight"],
                torch.full((3, 4), 1.2),
            ))

    # ---- WAN 2.2 tests ----

    def test_profiles_classify_wan22_keys(self):
        # WAN 2.2 uses self_attn / cross_attn with split q/k/v
        self.assertEqual(
            classify_wan22_key("model.diffusion_model.transformer_blocks.0.self_attn.q.weight"),
            "self_attn_qkv",
        )
        self.assertEqual(
            classify_wan22_key("model.diffusion_model.transformer_blocks.0.cross_attn.v.weight"),
            "cross_attn_qkv",
        )
        self.assertEqual(
            classify_wan22_key("model.diffusion_model.transformer_blocks.0.ffn.0.weight"),
            "ffn_in",
        )
        self.assertEqual(
            classify_wan22_key("model.diffusion_model.transformer_blocks.0.ffn.2.weight"),
            "ffn_out",
        )
        self.assertTrue(
            is_wan22_preserved_key("model.diffusion_model.modulation.lin.weight")
        )
        self.assertTrue(
            is_wan22_preserved_key("model.diffusion_model.patch_embedding.weight")
        )

    def test_wan22_dry_run_reports_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            lora = tmp / "style.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "model.diffusion_model.transformer_blocks.0.self_attn.q.weight": torch.zeros(3, 4),
            }, str(base))
            save_file({
                "diffusion_model.transformer_blocks.0.self_attn.q.lora_A.weight": torch.ones(2, 4),
                "diffusion_model.transformer_blocks.0.self_attn.q.lora_B.weight": torch.ones(3, 2),
            }, str(lora))

            events = list(run_lora_merge({
                "base_path": str(base),
                "loras": [{"path": str(lora), "strength": 0.5}],
                "output_path": str(out),
                "strategy": "Balanced",
                "architecture": "WAN 2.2",
                "global_strength": 1.0,
                "adaptive": False,
                "dry_run": True,
                "strict_matching": True,
            }))

            text = "".join(e.get("text", "") for e in events)
            self.assertIn("matched=1", text)
            self.assertFalse(out.exists())

    def test_wan22_merge_applies_scaled_delta_and_skips_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            lora = tmp / "style.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "model.diffusion_model.transformer_blocks.0.self_attn.q.weight": torch.zeros(3, 4),
                "model.diffusion_model.modulation.lin.weight": torch.zeros(3, 4),
            }, str(base))
            save_file({
                "diffusion_model.transformer_blocks.0.self_attn.q.lora_A.weight": torch.ones(2, 4),
                "diffusion_model.transformer_blocks.0.self_attn.q.lora_B.weight": torch.ones(3, 2),
                "diffusion_model.modulation.lin.lora_A.weight": torch.ones(2, 4),
                "diffusion_model.modulation.lin.lora_B.weight": torch.ones(3, 2),
            }, str(lora))

            list(run_lora_merge({
                "base_path": str(base),
                "loras": [{"path": str(lora), "strength": 0.5}],
                "output_path": str(out),
                "strategy": "Balanced",
                "architecture": "WAN 2.2",
                "global_strength": 1.0,
                "adaptive": False,
                "dry_run": False,
                "strict_matching": True,
            }))

            merged = load_file(str(out))
            # self_attn.q should be merged (Balanced multiplier 1.0)
            self.assertTrue(torch.allclose(
                merged["model.diffusion_model.transformer_blocks.0.self_attn.q.weight"],
                torch.ones(3, 4),
            ))
            # modulation should be preserved (skipped)
            self.assertTrue(torch.allclose(
                merged["model.diffusion_model.modulation.lin.weight"],
                torch.zeros(3, 4),
            ))


if __name__ == "__main__":
    unittest.main()
