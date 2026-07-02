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
from utils.lora_inspector import discover_lora_pairs, discover_diff_patches, read_safetensors_manifest
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

    def test_rejects_unsafe_effective_lora_strength_before_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            lora = tmp / "too_hot.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "blocks.0.attn.wq.weight": torch.zeros(3, 4),
            }, str(base))
            save_file({
                "lora_unet_blocks_0_attn_wq.lora_down.weight": torch.ones(2, 4),
                "lora_unet_blocks_0_attn_wq.lora_up.weight": torch.ones(3, 2),
            }, str(lora))

            with self.assertRaisesRegex(ValueError, "effective strength 25 exceeds safe limit"):
                list(run_lora_merge({
                    "base_path": str(base),
                    "loras": [{"path": str(lora), "strength": 25}],
                    "output_path": str(out),
                    "strategy": "Balanced",
                    "architecture": "Krea 2",
                    "global_strength": 1.0,
                    "adaptive": False,
                    "dry_run": False,
                    "strict_matching": True,
                    "merge_device": "cpu",
                }))

            self.assertFalse(out.exists())

    def test_allows_negative_lora_strength_within_safe_limit(self):
        lora = {"path": "/tmp/negative.safetensors", "strength": -3.0}
        lora_merge_engine._validate_lora_strengths([lora], global_strength=1.0)

    def test_rejects_strength_just_over_safe_limit(self):
        lora = {"path": "/tmp/too_hot.safetensors", "strength": 3.05}
        with self.assertRaisesRegex(ValueError, "exceeds safe limit"):
            lora_merge_engine._validate_lora_strengths([lora], global_strength=1.0)

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
        # Balanced: all non-structural tensors pass through (factor=1.0).
        self.assertEqual(strategy_multiplier("Balanced", "attn_qkv"), 1.00)
        self.assertEqual(strategy_multiplier("Balanced", "ff_in"), 1.00)
        self.assertEqual(strategy_multiplier("Balanced", "text_fusion"), 1.00)
        self.assertEqual(strategy_multiplier("Balanced", "attn_gate"), 1.00)
        self.assertEqual(strategy_multiplier("Balanced", "other"), 1.00)
        # Style: only attention tensors pass through; FFN and text excluded (0.0).
        self.assertEqual(strategy_multiplier("Style", "attn_qkv"), 1.00)
        self.assertEqual(strategy_multiplier("Style", "ff_in"), 0.00)
        self.assertEqual(strategy_multiplier("Style", "text_fusion"), 0.00)
        # Content: only FFN tensors pass through; attention and text excluded (0.0).
        self.assertEqual(strategy_multiplier("Content", "attn_qkv"), 0.00)
        self.assertEqual(strategy_multiplier("Content", "ff_in"), 1.00)
        self.assertEqual(strategy_multiplier("Content", "text_fusion"), 0.00)
        # Structural always excluded (0.0).
        for strat in ["Balanced", "Style", "Content"]:
            self.assertEqual(strategy_multiplier(strat, "structural"), 0.0)

    def test_missing_adaptive_payload_defaults_to_comfyui_parity(self):
        """Omitting adaptive must not silently amplify LoRA strength."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            lora = tmp / "style.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "blocks.0.attn.wq.weight": torch.full((3, 4), 100.0),
            }, str(base))
            save_file({
                "lora_unet_blocks_0_attn_wq.lora_down.weight": torch.ones(2, 4),
                "lora_unet_blocks_0_attn_wq.lora_up.weight": torch.ones(3, 2),
            }, str(lora))

            list(run_lora_merge({
                "base_path": str(base),
                "loras": [{"path": str(lora), "strength": 0.5}],
                "output_path": str(out),
                "strategy": "Balanced",
                "architecture": "Krea 2",
                "global_strength": 1.0,
                "dry_run": False,
                "strict_matching": True,
                "merge_device": "cpu",
            }))

            merged = load_file(str(out))
            # ComfyUI/live LoRA math: base + (up @ down) * strength = 100 + 2 * 0.5.
            self.assertTrue(torch.allclose(
                merged["blocks.0.attn.wq.weight"],
                torch.full((3, 4), 101.0),
            ))

    def test_krea2_style_merge_applies_attention_only(self):
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
            # Style: attn_qkv filter=1.0 (apply), ff_in/text_fusion excluded (0.0) — pure selection, no boost.
            # delta = up @ down = ones(3,2) @ ones(2,4) = full(3,4, 2.0)
            # result = base + global_strength * lora_strength * filter_multiplier * delta
            #        = 0 + 1.0 * 0.5 * 1.0 * 2.0 = 1.0
            self.assertTrue(torch.allclose(
                merged["blocks.0.attn.wq.weight"],
                torch.full((3, 4), 1.0),
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
            video = tmp / "video.safetensors"
            audio = tmp / "audio.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "model.diffusion_model.transformer_blocks.0.attn1.to_k.weight": torch.zeros(3, 4),
                "model.diffusion_model.transformer_blocks.0.audio_attn1.to_k.weight": torch.zeros(3, 4),
            }, str(base))
            save_file({
                # Video LoRA: targets attn_qkv (non-audio tensor) → pass with Audio strategy? No — Video only applies non-audio tensors.
                "diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight": torch.ones(2, 4),
                "diffusion_model.transformer_blocks.0.attn1.to_k.lora_B.weight": torch.ones(3, 2),
            }, str(video))
            save_file({
                # Audio LoRA: targets audio_attn → pass with Audio strategy only.
                "diffusion_model.transformer_blocks.0.audio_attn1.to_k.lora_A.weight": torch.ones(2, 4),
                "diffusion_model.transformer_blocks.0.audio_attn1.to_k.lora_B.weight": torch.ones(3, 2),
            }, str(audio))

            events = list(run_lora_merge({
                "base_path": str(base),
                "loras": [
                    # Video: attn_qkv=1.0 (included in Video filter) → delta applied with full strength.
                    {"path": str(video), "strength": 0.5, "strategy": "Video"},
                    # Audio: audio_attn=1.0 (included in Audio filter), other=0.0 → only audio tensors get LoRA.
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
            self.assertIn("strategy=Video", text)
            self.assertIn("strategy=Audio", text)
            merged = load_file(str(out))
            # Video: attn_qkv filter=1.0 → delta * 0.5 * 1.0 = full(3,4,2)*0.5*1.0 = full(3,4,1.0), base + 1.0 = 1.0
            self.assertTrue(torch.allclose(
                merged["model.diffusion_model.transformer_blocks.0.attn1.to_k.weight"],
                torch.full((3, 4), 1.0),
            ))
            # Audio: audio_attn filter=1.0 → delta * 0.5 * 1.0 = full(3,4,2)*0.5*1.0 = full(3,4,1.0), base + 1.0 = 1.0
            self.assertTrue(torch.allclose(
                merged["model.diffusion_model.transformer_blocks.0.audio_attn1.to_k.weight"],
                torch.full((3, 4), 1.0),
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

    # ---- .diff patch tests (ComfyUI direct-weight-patch format) ----

    def test_discover_diff_patches_finds_diff_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora = Path(tmp) / "diff.safetensors"
            save_file({
                "diffusion_model.txtfusion.projector.diff": torch.tensor([[1.0, 2.0, 3.0]]),
            }, str(lora))

            manifest = read_safetensors_manifest(str(lora))
            patches = discover_diff_patches(manifest)

            self.assertEqual(len(patches), 1)
            self.assertEqual(patches[0].diff_key, "diffusion_model.txtfusion.projector.diff")
            self.assertEqual(patches[0].diff_shape, (1, 3))
            # Should produce candidates with and without diffusion_model. prefix
            self.assertIn("diffusion_model.txtfusion.projector.weight", patches[0].target_candidates)
            self.assertIn("txtfusion.projector.weight", patches[0].target_candidates)

    def test_discover_diff_patches_ignores_non_diff_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora = Path(tmp) / "mixed.safetensors"
            save_file({
                "diffusion_model.txtfusion.projector.diff": torch.tensor([[1.0, 2.0, 3.0]]),
                "some_other.bias": torch.tensor([0.5]),
                "diffusion_model.blocks.0.attn.wq.lora_A.weight": torch.ones(2, 4),
                "diffusion_model.blocks.0.attn.wq.lora_B.weight": torch.ones(3, 2),
            }, str(lora))

            manifest = read_safetensors_manifest(str(lora))
            patches = discover_diff_patches(manifest)

            self.assertEqual(len(patches), 1)
            self.assertEqual(patches[0].diff_key, "diffusion_model.txtfusion.projector.diff")

    def test_diff_patch_prefix_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora = Path(tmp) / "diff.safetensors"
            save_file({
                "model.diffusion_model.blocks.0.attn.wq.diff": torch.ones(3, 4),
            }, str(lora))

            manifest = read_safetensors_manifest(str(lora))
            patches = discover_diff_patches(manifest)

            self.assertEqual(len(patches), 1)
            candidates = patches[0].target_candidates
            self.assertIn("model.diffusion_model.blocks.0.attn.wq.weight", candidates)
            self.assertIn("diffusion_model.blocks.0.attn.wq.weight", candidates)

    def test_diff_patch_merge_applied_as_direct_add(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            diff_lora = tmp / "filterbypass.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "blocks.0.attn.wq.weight": torch.tensor([[-0.5, -0.8, -0.6, 0.0]]),
            }, str(base))
            save_file({
                "diffusion_model.blocks.0.attn.wq.diff": torch.tensor([[-0.5, -0.8, -0.6, 0.0]]),
            }, str(diff_lora))

            list(run_lora_merge({
                "base_path": str(base),
                "loras": [{"path": str(diff_lora), "strength": 1.0}],
                "output_path": str(out),
                "strategy": "Balanced",
                "architecture": "Krea 2",
                "global_strength": 1.0,
                "adaptive": False,
                "dry_run": False,
                "strict_matching": True,
            }))

            merged = load_file(str(out))
            # base + diff = [-1.0, -1.6, -1.2, 0.0]
            self.assertTrue(torch.allclose(
                merged["blocks.0.attn.wq.weight"],
                torch.tensor([[-1.0, -1.6, -1.2, 0.0]]),
            ))

    def test_diff_patch_dry_run_reports_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            diff_lora = tmp / "diff.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "blocks.0.attn.wq.weight": torch.tensor([[-0.5, -0.8, -0.6, 0.0]]),
            }, str(base))
            save_file({
                "diffusion_model.blocks.0.attn.wq.diff": torch.tensor([[-0.5, -0.8, -0.6, 0.0]]),
            }, str(diff_lora))

            events = list(run_lora_merge({
                "base_path": str(base),
                "loras": [{"path": str(diff_lora), "strength": 1.0}],
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
            self.assertIn("diff_patches=1", text)
            self.assertFalse(out.exists())

    def test_diff_patch_with_strength_scaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            diff_lora = tmp / "diff.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "blocks.0.attn.wq.weight": torch.tensor([[1.0, 2.0, 3.0]]),
            }, str(base))
            save_file({
                "diffusion_model.blocks.0.attn.wq.diff": torch.tensor([[0.5, 0.5, 0.5]]),
            }, str(diff_lora))

            list(run_lora_merge({
                "base_path": str(base),
                "loras": [{"path": str(diff_lora), "strength": 0.5}],
                "output_path": str(out),
                "strategy": "Balanced",
                "architecture": "Krea 2",
                "global_strength": 2.0,
                "adaptive": False,
                "dry_run": False,
                "strict_matching": True,
            }))

            merged = load_file(str(out))
            # base + diff * global_strength * lora_strength = [1,2,3] + [0.5,0.5,0.5] * 2.0 * 0.5
            # = [1,2,3] + [0.5,0.5,0.5] = [1.5, 2.5, 3.5]
            self.assertTrue(torch.allclose(
                merged["blocks.0.attn.wq.weight"],
                torch.tensor([[1.5, 2.5, 3.5]]),
            ))

    def test_diff_patch_skipped_when_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            diff_lora = tmp / "diff.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "blocks.0.mod.lin": torch.tensor([[1.0, 2.0]]),
            }, str(base))
            save_file({
                "diffusion_model.blocks.0.mod.lin.diff": torch.tensor([[0.5, 0.5]]),
            }, str(diff_lora))

            events = list(run_lora_merge({
                "base_path": str(base),
                "loras": [{"path": str(diff_lora), "strength": 1.0}],
                "output_path": str(out),
                "strategy": "Balanced",
                "architecture": "Krea 2",
                "global_strength": 1.0,
                "adaptive": False,
                "dry_run": True,
                "strict_matching": True,
            }))

            text = "".join(e.get("text", "") for e in events)
            self.assertIn("skipped_preserve=1 skipped_strategy=0", text)

    def test_diff_patch_unmatched_when_no_base_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            diff_lora = tmp / "diff.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "blocks.0.attn.wq.weight": torch.tensor([[1.0]]),
            }, str(base))
            save_file({
                "diffusion_model.nonexistent.layer.diff": torch.tensor([[0.5]]),
            }, str(diff_lora))

            events = list(run_lora_merge({
                "base_path": str(base),
                "loras": [{"path": str(diff_lora), "strength": 1.0}],
                "output_path": str(out),
                "strategy": "Balanced",
                "architecture": "Krea 2",
                "global_strength": 1.0,
                "adaptive": False,
                "dry_run": True,
                "strict_matching": True,
            }))

            text = "".join(e.get("text", "") for e in events)
            self.assertIn("unmatched=1", text)

    def test_diff_patch_cumulative_when_applied_twice(self):
        """Apply same diff to original base twice via two separate LoRA entries.
        Confirms diff patches are cumulative (not idempotent)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            diff_lora = tmp / "diff.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "blocks.0.attn.wq.weight": torch.tensor([[1.0]]),
            }, str(base))
            save_file({
                "diffusion_model.blocks.0.attn.wq.diff": torch.tensor([[0.5]]),
            }, str(diff_lora))

            list(run_lora_merge({
                "base_path": str(base),
                "loras": [
                    {"path": str(diff_lora), "strength": 1.0},
                    {"path": str(diff_lora), "strength": 1.0},
                ],
                "output_path": str(out),
                "strategy": "Balanced",
                "architecture": "Krea 2",
                "global_strength": 1.0,
                "adaptive": False,
                "dry_run": False,
                "strict_matching": True,
            }))

            merged = load_file(str(out))
            # 1.0 + 0.5 + 0.5 = 2.0 (cumulative)
            self.assertTrue(torch.allclose(
                merged["blocks.0.attn.wq.weight"],
                torch.tensor([[2.0]]),
            ))

    def test_diff_and_lora_pairs_in_same_file(self):
        """A file can contain both traditional LoRA pairs and .diff patches."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            mixed = tmp / "mixed.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "blocks.0.attn.wq.weight": torch.zeros(3, 4),
                "blocks.0.attn.wk.weight": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            }, str(base))
            save_file({
                "lora_unet_blocks_0_attn_wq.lora_down.weight": torch.ones(2, 4),
                "lora_unet_blocks_0_attn_wq.lora_up.weight": torch.ones(3, 2),
                "diffusion_model.blocks.0.attn.wk.diff": torch.tensor([[-0.5, -0.5, -0.5, -0.5]]),
            }, str(mixed))

            events = list(run_lora_merge({
                "base_path": str(base),
                "loras": [{"path": str(mixed), "strength": 1.0}],
                "output_path": str(out),
                "strategy": "Balanced",
                "architecture": "Krea 2",
                "global_strength": 1.0,
                "adaptive": False,
                "dry_run": False,
                "strict_matching": True,
            }))

            text = "".join(e.get("text", "") for e in events)
            self.assertIn("pairs=1", text)
            self.assertIn("diff_patches=1", text)
            self.assertIn("matched=2", text)

            merged = load_file(str(out))
            # LoRA: up @ down = ones(3,2) @ ones(2,4) = full(3,4, 2.0); scale=1.0*1.0*1.0=1.0
            self.assertTrue(torch.allclose(
                merged["blocks.0.attn.wq.weight"],
                torch.full((3, 4), 2.0),
            ))
            # Diff: [1,2,3,4] + [-0.5,-0.5,-0.5,-0.5] * 1.0 = [0.5, 1.5, 2.5, 3.5]
            self.assertTrue(torch.allclose(
                merged["blocks.0.attn.wk.weight"],
                torch.tensor([[0.5, 1.5, 2.5, 3.5]]),
            ))

    def test_builtin_unchain_applied_correctly(self):
        """Builtin unchain multiplies positions 8-10 by (1 + scale)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "txtfusion.projector.weight": torch.tensor([
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
                ]),
            }, str(base))

            list(run_lora_merge({
                "base_path": str(base),
                "loras": [],
                "output_path": str(out),
                "strategy": "Balanced",
                "architecture": "Krea 2",
                "global_strength": 1.0,
                "adaptive": False,
                "dry_run": False,
                "strict_matching": True,
                "krea2_unchain": True,
            }))

            merged = load_file(str(out))
            # Positions 0-7 and 11 unchanged
            self.assertTrue(torch.allclose(
                merged["txtfusion.projector.weight"][:, :8],
                torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]),
            ))
            self.assertTrue(torch.allclose(
                merged["txtfusion.projector.weight"][:, 11:],
                torch.tensor([[12.0]]),
            ))
            # Positions 8-10 multiplied by (1 + 1.05) = 2.05
            self.assertTrue(torch.allclose(
                merged["txtfusion.projector.weight"][:, 8:11],
                torch.tensor([[9.0 * 2.05, 10.0 * 2.05, 11.0 * 2.05]]),
                atol=1e-5,
            ))

    def test_builtin_unchain_dry_run_reports(self):
        """Dry run with unchain reports the builtin patch."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "txtfusion.projector.weight": torch.ones(1, 12),
            }, str(base))

            events = list(run_lora_merge({
                "base_path": str(base),
                "loras": [],
                "output_path": str(out),
                "strategy": "Balanced",
                "architecture": "Krea 2",
                "global_strength": 1.0,
                "adaptive": False,
                "dry_run": True,
                "strict_matching": True,
                "krea2_unchain": True,
            }))

            text = "".join(e.get("text", "") for e in events)
            self.assertIn("builtin unchain", text.lower())
            self.assertIn("matched=1", text)
            self.assertFalse(out.exists())

    def test_builtin_unchain_skipped_wrong_arch(self):
        """Unchain is only applied for Krea 2 architecture."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "txtfusion.projector.weight": torch.ones(1, 12),
            }, str(base))

            events = list(run_lora_merge({
                "base_path": str(base),
                "loras": [],
                "output_path": str(out),
                "strategy": "Balanced",
                "architecture": "LTX-2.3",  # Not Krea 2
                "global_strength": 1.0,
                "adaptive": False,
                "dry_run": True,
                "strict_matching": True,
                "krea2_unchain": True,
            }))

            text = "".join(e.get("text", "") for e in events)
            self.assertNotIn("builtin unchain", text)
            self.assertIn("matched=0", text)

    def test_builtin_unchain_skipped_wrong_shape(self):
        """Unchain is skipped if txtfusion.projector.weight shape != (1, 12)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "txtfusion.projector.weight": torch.ones(2, 12),  # Wrong shape
            }, str(base))

            events = list(run_lora_merge({
                "base_path": str(base),
                "loras": [],
                "output_path": str(out),
                "strategy": "Balanced",
                "architecture": "Krea 2",
                "global_strength": 1.0,
                "adaptive": False,
                "dry_run": True,
                "strict_matching": True,
                "krea2_unchain": True,
            }))

            text = "".join(e.get("text", "") for e in events)
            self.assertIn("Skipping unchain", text)

    def test_builtin_unchain_with_lora_and_global_strength(self):
        """Unchain respects global_strength scaling alongside LoRA pairs."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "base.safetensors"
            lora = tmp / "lora.safetensors"
            out = tmp / "merged.safetensors"
            save_file({
                "blocks.0.attn.wq.weight": torch.zeros(3, 4),
                "txtfusion.projector.weight": torch.tensor([
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
                ]),
            }, str(base))
            save_file({
                "lora_unet_blocks_0_attn_wq.lora_down.weight": torch.ones(2, 4),
                "lora_unet_blocks_0_attn_wq.lora_up.weight": torch.ones(3, 2),
            }, str(lora))

            list(run_lora_merge({
                "base_path": str(base),
                "loras": [{"path": str(lora), "strength": 1.0}],
                "output_path": str(out),
                "strategy": "Balanced",
                "architecture": "Krea 2",
                "global_strength": 0.5,  # scale = 0.5 * 1.05 = 0.525
                "adaptive": False,
                "dry_run": False,
                "strict_matching": True,
                "krea2_unchain": True,
            }))

            merged = load_file(str(out))
            # LoRA: up @ down = ones(3,2) @ ones(2,4) = full(3,4, 2.0); scale = 0.5*1.0*1.0 = 0.5; merged = 0 + 2.0*0.5 = 1.0
            self.assertTrue(torch.allclose(
                merged["blocks.0.attn.wq.weight"],
                torch.full((3, 4), 1.0),
            ))
            # Unchain: multiplier = 1 + (0.5 * 1.05) = 1.525
            self.assertTrue(torch.allclose(
                merged["txtfusion.projector.weight"][:, 8:11],
                torch.tensor([[9.0 * 1.525, 10.0 * 1.525, 11.0 * 1.525]]),
                atol=1e-5,
            ))


if __name__ == "__main__":
    unittest.main()
