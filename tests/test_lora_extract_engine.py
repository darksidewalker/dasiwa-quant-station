import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from core.lora_extract_engine import run_lora_extract


class TestGenericCheckpointLoraExtract(unittest.TestCase):
    @mock.patch("core.lora_extract_engine.verify_architecture_match", return_value=(True, "ok"))
    def test_generic_recipe_extracts_two_checkpoint_delta_as_standard_lora(self, _verify):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.safetensors"
            modified = root / "modified.safetensors"
            output = root / "adapter.safetensors"
            base_weight = torch.zeros((3, 4), dtype=torch.float32)
            delta = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]],
                dtype=torch.float32,
            )
            save_file({"blocks.0.self_attn.q.weight": base_weight}, str(base))
            save_file({"blocks.0.self_attn.q.weight": base_weight + delta}, str(modified))

            events = list(run_lora_extract({
                "recipe": "generic",
                "architecture": "WAN 2.2",
                "base_path": str(base),
                "merged_path": str(modified),
                "output_path": str(output),
                "frobenius_energy": 1.0,
            }))

            self.assertEqual(events[-1]["status"], "finished")
            with safe_open(str(output), framework="pt", device="cpu") as handle:
                down = handle.get_tensor("diffusion_model.blocks.0.self_attn.q.lora_A.weight")
                up = handle.get_tensor("diffusion_model.blocks.0.self_attn.q.lora_B.weight")
                torch.testing.assert_close((up @ down).float(), delta, rtol=0.01, atol=0.01)
                self.assertEqual(handle.metadata()["format"], "dasiwa_checkpoint_delta_lora")
                self.assertEqual(handle.metadata()["architecture"], "WAN 2.2")
                self.assertEqual(handle.metadata()["recipe"], "generic")


class TestMiniMaxH3LoraExtract(unittest.TestCase):
    def _full(self, path: Path, changed: bool = False):
        tensors = {
            "time_embedder.proj_in.weight": torch.tensor([[0.2, -0.1], [0.1, 0.3], [-0.2, 0.2], [0.4, 0.1]], dtype=torch.float32),
            "time_embedder.proj_in.bias": torch.tensor([0.1, -0.1, 0.05, 0.2], dtype=torch.float32),
            "time_embedder.proj_out.weight": torch.eye(4, dtype=torch.float32),
            "time_embedder.proj_out.bias": torch.zeros(4, dtype=torch.float32),
            "blocks.0.adaln_proj.linear.weight": torch.zeros((3, 4), dtype=torch.float32),
            "blocks.0.adaln_proj.linear.bias": torch.zeros(3, dtype=torch.float32),
            "blocks.0.attn.qkv_proj.weight": torch.zeros((3, 4), dtype=torch.float32),
        }
        if changed:
            tensors["blocks.0.adaln_proj.linear.weight"][0, 0] = 2.0
            tensors["blocks.0.adaln_proj.linear.bias"][0] = 0.5
            tensors["blocks.0.attn.qkv_proj.weight"][1, 2] = 3.0
        save_file(tensors, str(path))

    def _pruned(self, path: Path):
        table = torch.stack([torch.linspace(-1.0, 1.0, 1025) ** (i + 1) for i in range(8)], dim=1)
        save_file({
            "adaln_t_table": table,
            "blocks.0.adaln_proj.linear.weight": torch.zeros((3, 8), dtype=torch.float32),
            "blocks.0.adaln_proj.linear.bias": torch.zeros(3, dtype=torch.float32),
        }, str(path))

    @mock.patch("core.lora_extract_engine.verify_architecture_match", return_value=(True, "ok"))
    def test_pruned_extract_rebases_adaln_and_factors_other_matrices(self, _verify):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, merged, target, output = (root / "base.safetensors", root / "merged.safetensors", root / "target.safetensors", root / "output.safetensors")
            self._full(base)
            self._full(merged, changed=True)
            self._pruned(target)
            events = list(run_lora_extract({
                "architecture": "MiniMax H3",
                "base_path": str(base),
                "merged_path": str(merged),
                "pruned_target_path": str(target),
                "output_path": str(output),
                "output_mode": "pruned",
                "frobenius_energy": 1.0,
            }))
            self.assertEqual(events[-1]["status"], "finished")
            with safe_open(str(output), framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                self.assertIn("diffusion_model.blocks.0.adaln_proj.linear.diff", keys)
                self.assertIn("diffusion_model.blocks.0.adaln_proj.linear.diff_b", keys)
                self.assertNotIn("diffusion_model.blocks.0.adaln_proj.linear.lora_A.weight", keys)
                self.assertIn("diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight", keys)
                self.assertIn("diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight", keys)
                self.assertEqual(tuple(handle.get_tensor("diffusion_model.blocks.0.adaln_proj.linear.diff").shape), (3, 8))
                self.assertEqual(tuple(handle.get_tensor("diffusion_model.blocks.0.adaln_proj.linear.diff_b").shape), (3,))
                self.assertEqual(handle.metadata()["adaln_patch_format"], "diff+diff_b")

    @mock.patch("core.lora_extract_engine.verify_architecture_match", return_value=(True, "ok"))
    def test_full_extract_emits_standard_lora_for_adaln(self, _verify):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, merged, output = root / "base.safetensors", root / "merged.safetensors", root / "output.safetensors"
            self._full(base)
            self._full(merged, changed=True)
            list(run_lora_extract({
                "architecture": "MiniMax H3", "base_path": str(base), "merged_path": str(merged),
                "output_path": str(output), "output_mode": "full", "frobenius_energy": 1.0,
            }))
            with safe_open(str(output), framework="pt", device="cpu") as handle:
                self.assertIn("diffusion_model.blocks.0.adaln_proj.linear.lora_A.weight", handle.keys())
                self.assertNotIn("diffusion_model.blocks.0.adaln_proj.linear.diff", handle.keys())


if __name__ == "__main__":
    unittest.main()
