import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from core import lora_merge_engine
from core.lora_merge_engine import run_lora_merge
from utils.lora_inspector import discover_lora_pairs, read_safetensors_manifest
from utils.krea2_layer_profiles import classify_krea2_key, is_krea2_preserved_key
from utils.ltx23_layer_profiles import classify_ltx23_key, is_ltx23_preserved_key
from utils.wan22_layer_profiles import classify_wan22_key, is_wan22_preserved_key


class LoraMergeEngineTests(unittest.TestCase):
    def test_merge_device_policy_helpers_validate_and_estimate_vram(self):
        self.assertEqual(lora_merge_engine._normalize_merge_device(None), "auto")
        self.assertEqual(lora_merge_engine._normalize_merge_device("cpu"), "cpu")
        self.assertEqual(lora_merge_engine._normalize_merge_device("AUTO"), "auto")
        with self.assertRaises(ValueError):
            lora_merge_engine._normalize_merge_device("tpu")

        small = lora_merge_engine._estimate_lora_merge_peak_bytes((3, 4), (2, 4), (3, 2))
        large = lora_merge_engine._estimate_lora_merge_peak_bytes((30, 40), (20, 40), (30, 20))
        self.assertGreater(small, 0)
        self.assertGreater(large, small)

    def test_cuda_oom_falls_back_to_cpu_and_writes_correct_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            lora = tmp / "style.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "model.diffusion_model.transformer_blocks.0.attn1.to_k.weight": torch.zeros(3, 4),
            }, str(base))
            save_file({
                "diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight": torch.ones(2, 4),
                "diffusion_model.transformer_blocks.0.attn1.to_k.lora_B.weight": torch.ones(3, 2),
            }, str(lora))

            with mock.patch.object(lora_merge_engine, "_cuda_available", return_value=True), \
                 mock.patch.object(lora_merge_engine, "_has_cuda_headroom", return_value=True), \
                 mock.patch.object(lora_merge_engine, "_merge_target_cuda", side_effect=torch.cuda.OutOfMemoryError("boom")), \
                 mock.patch.object(torch.cuda, "empty_cache"):
                events = list(run_lora_merge({
                    "base_path": str(base),
                    "loras": [{"path": str(lora), "strength": 0.5}],
                    "output_path": str(out),
                    "strategy": "Balanced",
                    "architecture": "LTX-2.3",
                    "global_strength": 1.0,
                    "adaptive": False,
                    "dry_run": False,
                    "strict_matching": True,
                    "merge_device": "cuda",
                    "cuda_device": "cuda:0",
                    "vram_headroom_mb": 1024,
                }))

            text = "".join(e.get("text", "") for e in events)
            self.assertIn("oom=1", text)
            merged = load_file(str(out))
            self.assertTrue(torch.allclose(
                merged["model.diffusion_model.transformer_blocks.0.attn1.to_k.weight"],
                torch.ones(3, 4),
            ))

    def test_real_merge_streams_safetensors_without_save_file_dict_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            lora = tmp / "style.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "model.diffusion_model.transformer_blocks.0.attn1.to_k.weight": torch.zeros(3, 4),
                "model.diffusion_model.transformer_blocks.0.attn1.to_v.weight": torch.full((3, 4), 7.0),
                "bf16.weight": torch.ones(2, 2, dtype=torch.bfloat16),
            }, str(base))
            save_file({
                "diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight": torch.ones(2, 4),
                "diffusion_model.transformer_blocks.0.attn1.to_k.lora_B.weight": torch.ones(3, 2),
            }, str(lora))

            with mock.patch.object(lora_merge_engine, "save_file", side_effect=AssertionError("save_file should not be used for merge output")):
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
                    "merge_device": "cpu",
                }))

            merged = load_file(str(out))
            self.assertTrue(torch.allclose(
                merged["model.diffusion_model.transformer_blocks.0.attn1.to_k.weight"],
                torch.ones(3, 4),
            ))
            self.assertTrue(torch.allclose(
                merged["model.diffusion_model.transformer_blocks.0.attn1.to_v.weight"],
                torch.full((3, 4), 7.0),
            ))
            self.assertEqual(merged["bf16.weight"].dtype, torch.bfloat16)

    def test_real_merge_logs_success_summary_when_all_matched_tensors_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            lora = tmp / "style.safetensors"
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
                "dry_run": False,
                "strict_matching": True,
                "merge_device": "cpu",
            }))

            text = "".join(e.get("text", "") for e in events)
            self.assertIn("LoRA merge success summary: all_matched_applied=yes altered_targets=1/1", text)
            statuses = [e.get("status") for e in events]
            self.assertIn("LoRA merge complete: 1/1 targets altered", statuses)

    def test_real_merge_warns_when_matched_tensor_is_not_altered(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            lora = tmp / "zero.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "model.diffusion_model.transformer_blocks.0.attn1.to_k.weight": torch.zeros(3, 4),
            }, str(base))
            save_file({
                "diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight": torch.zeros(2, 4),
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
                "dry_run": False,
                "strict_matching": True,
                "merge_device": "cpu",
            }))

            text = "".join(e.get("text", "") for e in events)
            self.assertIn("LoRA merge success summary: all_matched_applied=yes altered_targets=0/1", text)
            self.assertIn("WARNING: 1 matched target tensor(s) were not altered", text)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_merge_matches_cpu_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = torch.zeros(3, 4)
            lora_path = tmp / "style.safetensors"
            save_file({
                "down": torch.ones(2, 4),
                "up": torch.ones(3, 2),
            }, str(lora_path))
            op = {"lora_path": str(lora_path), "down_key": "down", "up_key": "up", "alpha_key": None, "rank": 2, "scale": 0.5}
            with safe_open(str(lora_path), framework="pt", device="cpu") as lf:
                handles = {str(lora_path): lf}
                cpu = lora_merge_engine._merge_target_cpu(base, [op], handles, adaptive=False)
                cuda = lora_merge_engine._merge_target_cuda(base, [op], handles, adaptive=False, device="cuda:0")
            self.assertTrue(torch.allclose(cpu, cuda))

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

    def test_krea2_lora_unet_underscore_keys_map_to_base_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora = Path(tmp) / "krea.safetensors"
            save_file({
                "lora_unet_blocks_0_attn_wq.lora_down.weight": torch.ones(2, 4),
                "lora_unet_blocks_0_attn_wq.lora_up.weight": torch.ones(3, 2),
            }, str(lora))

            manifest = read_safetensors_manifest(str(lora))
            pairs = discover_lora_pairs(manifest)

            self.assertEqual(len(pairs), 1)
            self.assertIn("blocks.0.attn.wq.weight", pairs[0].target_candidates)

    def test_profiles_classify_krea2_keys(self):
        self.assertEqual(classify_krea2_key("blocks.0.attn.wq.weight"), "attn_qkv")
        self.assertEqual(classify_krea2_key("blocks.0.mlp.down.weight"), "ff_out")
        self.assertTrue(is_krea2_preserved_key("blocks.0.mod.lin"))
        self.assertTrue(is_krea2_preserved_key("last.linear.weight"))

    def test_krea2_dry_run_reports_match_for_real_key_style(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            lora = tmp / "style.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "blocks.0.attn.wq.weight": torch.zeros(3, 4),
            }, str(base))
            save_file({
                "lora_unet_blocks_0_attn_wq.lora_down.weight": torch.ones(2, 4),
                "lora_unet_blocks_0_attn_wq.lora_up.weight": torch.ones(3, 2),
            }, str(lora))

            events = list(run_lora_merge({
                "base_path": str(base),
                "loras": [{"path": str(lora), "strength": 0.5}],
                "output_path": str(out),
                "strategy": "Balanced",
                "architecture": "Krea 2",
                "global_strength": 1.0,
                "adaptive": False,
                "dry_run": True,
                "strict_matching": True,
            }))

            text = "".join(e.get("text", "") for e in events)
            self.assertIn("matched=1", text)
            self.assertFalse(out.exists())

    def test_krea2_strategies_apply_correct_multipliers(self):
        from utils.krea2_layer_profiles import strategy_multiplier
        # Balanced: neutral baseline
        self.assertEqual(strategy_multiplier("Balanced", "attn_qkv"), 1.00)
        self.assertEqual(strategy_multiplier("Balanced", "ff_in"), 1.00)
        self.assertEqual(strategy_multiplier("Balanced", "text_fusion"), 0.80)
        # Style: boost attention, reduce text_fusion
        self.assertEqual(strategy_multiplier("Style", "attn_qkv"), 1.15)
        self.assertEqual(strategy_multiplier("Style", "ff_in"), 1.00)
        self.assertEqual(strategy_multiplier("Style", "text_fusion"), 0.70)
        # Content: boost FF, reduce attention
        self.assertEqual(strategy_multiplier("Content", "attn_qkv"), 0.90)
        self.assertEqual(strategy_multiplier("Content", "ff_in"), 1.15)
        self.assertEqual(strategy_multiplier("Content", "text_fusion"), 0.85)
        # Detail: mild global boost
        self.assertEqual(strategy_multiplier("Detail", "attn_qkv"), 1.05)
        self.assertEqual(strategy_multiplier("Detail", "ff_in"), 1.05)
        self.assertEqual(strategy_multiplier("Detail", "text_fusion"), 0.85)
        # Structural always 0.0
        for strat in ["Balanced", "Style", "Content", "Detail"]:
            self.assertEqual(strategy_multiplier(strat, "structural"), 0.0)

    def test_krea2_style_merge_applies_attention_boost(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            lora = tmp / "style.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "blocks.0.attn.wq.weight": torch.zeros(3, 4),
            }, str(base))
            save_file({
                "lora_unet_blocks_0_attn_wq.lora_down.weight": torch.ones(2, 4),
                "lora_unet_blocks_0_attn_wq.lora_up.weight": torch.ones(3, 2),
            }, str(lora))

            list(run_lora_merge({
                "base_path": str(base),
                "loras": [{"path": str(lora), "strength": 0.5, "strategy": "Style"}],
                "output_path": str(out),
                "strategy": "Balanced",
                "architecture": "Krea 2",
                "global_strength": 1.0,
                "adaptive": False,
                "dry_run": False,
                "strict_matching": True,
            }))

            merged = load_file(str(out))
            # Style: attn_qkv multiplier 1.15, strength 0.5, global 1.0
            # delta = up @ down = ones(3,2) @ ones(2,4) = full(3,4, 2.0)
            # result = 0 + 1.0 * 0.5 * 1.15 * 2.0 = 1.15
            self.assertTrue(torch.allclose(
                merged["blocks.0.attn.wq.weight"],
                torch.full((3, 4), 1.15),
            ))

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
