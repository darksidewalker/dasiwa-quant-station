import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from core.adapter_factorization import reconstruct_lokr
from core.lora_compose_engine import run_lora_compose


class LoraComposeEngineTests(unittest.TestCase):
    def _events(self, payload):
        return list(run_lora_compose(payload))

    def test_two_loras_compose_to_standard_lora(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b, out = root / "a.safetensors", root / "b.safetensors", root / "out.safetensors"
            save_file({
                "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": torch.tensor([[1.0, 0.0]]),
                "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": torch.tensor([[1.0], [0.0]]),
            }, str(a))
            save_file({
                "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": torch.tensor([[0.0, 1.0]]),
                "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": torch.tensor([[0.0], [1.0]]),
            }, str(b))
            events = self._events({
                "loras": [{"path": str(a), "strength": 1.0}, {"path": str(b), "strength": 1.0}],
                "architecture": "MiniMax H3", "output_path": str(out),
                "output_adapter": "lora", "output_rank": 2, "frobenius_energy": 1.0,
                "consensus_preset": "neutral", "dry_run": False,
            })
            tensors = load_file(str(out))
            down = tensors["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"]
            up = tensors["diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"]
            self.assertEqual(tuple((up @ down).shape), (2, 2))
            self.assertTrue(torch.isfinite(up @ down).all())
            self.assertEqual(events[-1]["status"], "finished")
            self.assertTrue(out.with_suffix(".txt").exists())

    def test_direct_lokr_output_has_no_alpha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b, out = root / "a.safetensors", root / "b.safetensors", root / "out.safetensors"
            source = {
                "diffusion_model.blocks.0.attn.qkv_proj.lokr_w1": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "diffusion_model.blocks.0.attn.qkv_proj.lokr_w2": torch.tensor([[0.5, -1.0], [2.0, 0.0]]),
            }
            save_file(source, str(a)); save_file(source, str(b))
            self._events({
                "loras": [{"path": str(a)}, {"path": str(b)}], "architecture": "MiniMax H3",
                "output_path": str(out), "output_adapter": "lokr",
                "consensus_preset": "balanced", "dry_run": False,
            })
            tensors = load_file(str(out))
            self.assertIn("diffusion_model.blocks.0.attn.qkv_proj.lokr_w1", tensors)
            self.assertIn("diffusion_model.blocks.0.attn.qkv_proj.lokr_w2", tensors)
            self.assertFalse(any(key.endswith(".alpha") for key in tensors))
            rebuilt = reconstruct_lokr(
                tensors["diffusion_model.blocks.0.attn.qkv_proj.lokr_w1"],
                tensors["diffusion_model.blocks.0.attn.qkv_proj.lokr_w2"],
            )
            self.assertTrue(torch.isfinite(rebuilt).all())

    def test_forced_lokr_requires_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for name in ("a", "b"):
                path = root / f"{name}.safetensors"
                save_file({
                    "diffusion_model.x.lora_A.weight": torch.ones(1, 2),
                    "diffusion_model.x.lora_B.weight": torch.ones(2, 1),
                }, str(path)); paths.append(path)
            out = root / "out.safetensors"
            with self.assertRaisesRegex(ValueError, "LoKr anchor"):
                self._events({"loras": [{"path": str(p)} for p in paths], "output_path": str(out), "output_adapter": "lokr"})
            self.assertFalse(out.exists())

    def test_zero_strength_is_preserved_as_disabled_contributor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for name in ("a", "b"):
                path = root / f"{name}.safetensors"
                save_file({
                    "diffusion_model.x.lora_A.weight": torch.ones(1, 2),
                    "diffusion_model.x.lora_B.weight": torch.ones(2, 1),
                }, str(path)); paths.append(path)
            out = root / "out.safetensors"
            self._events({"loras": [{"path": str(path), "strength": 0.0} for path in paths],
                          "global_strength": 1.0, "output_path": str(out), "output_adapter": "lora"})
            tensors = load_file(str(out))
            rebuilt = tensors["diffusion_model.x.lora_B.weight"] @ tensors["diffusion_model.x.lora_A.weight"]
            torch.testing.assert_close(rebuilt, torch.zeros_like(rebuilt))

    def test_h3_underscore_keys_are_written_as_dotted_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for name in ("a", "b"):
                path = root / f"{name}.safetensors"
                save_file({
                    "lora_unet_blocks_0_attn_qkv_proj.lora_A.weight": torch.ones(1, 2),
                    "lora_unet_blocks_0_attn_qkv_proj.lora_B.weight": torch.ones(2, 1),
                }, str(path)); paths.append(path)
            out = root / "out.safetensors"
            self._events({"loras": [{"path": str(path)} for path in paths], "architecture": "MiniMax H3",
                          "output_path": str(out), "output_adapter": "lora"})
            tensors = load_file(str(out))
            self.assertIn("diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight", tensors)
            self.assertFalse(any("lora_unet_" in key for key in tensors))

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for name in ("a", "b"):
                path = root / f"{name}.safetensors"
                save_file({
                    "diffusion_model.x.lora_A.weight": torch.ones(1, 2),
                    "diffusion_model.x.lora_B.weight": torch.ones(2, 1),
                }, str(path)); paths.append(path)
            out = root / "out.safetensors"
            events = self._events({"loras": [{"path": str(p)} for p in paths], "output_path": str(out), "dry_run": True})
            self.assertFalse(out.exists())
            self.assertEqual(events[-1]["status"], "dry-run complete")


if __name__ == "__main__":
    unittest.main()
