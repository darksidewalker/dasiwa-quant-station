import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from core.lora_merge_engine import run_lora_merge


class ConsensusCheckpointMergeTests(unittest.TestCase):
    def test_consensus_bakes_group_once_and_rejects_adaptive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, a, b, out = (root / name for name in ("base.safetensors", "a.safetensors", "b.safetensors", "out.safetensors"))
            key = "model.diffusion_model.transformer_blocks.0.attn.to_q.weight"
            save_file({key: torch.zeros(2, 2)}, str(base))
            for path, value in ((a, 1.0), (b, 0.5)):
                save_file({
                    "base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight": torch.eye(2),
                    "base_model.model.transformer_blocks.0.attn.to_q.lora_B.weight": torch.eye(2) * value,
                }, str(path))
            events = list(run_lora_merge({
                "base_path": str(base), "output_path": str(out), "architecture": "LTX-2.3",
                "strategy": "All", "loras": [{"path": str(a), "strength": 1.0}, {"path": str(b), "strength": 1.0}],
                "global_strength": 1.0, "merge_algorithm": "consensus", "consensus_preset": "neutral",
                "merge_device": "cpu", "dry_run": False, "strict_matching": True,
            }))
            merged = load_file(str(out))[key]
            self.assertTrue(torch.isfinite(merged).all())
            self.assertGreater(float(merged.diagonal().min()), 0.0)
            self.assertEqual(events[-1]["status"], "finished")
            with self.assertRaisesRegex(ValueError, "adaptive"):
                list(run_lora_merge({
                    "base_path": str(base), "output_path": str(out), "architecture": "LTX-2.3",
                    "loras": [{"path": str(a)}, {"path": str(b)}], "merge_algorithm": "consensus",
                    "adaptive": True,
                }))


if __name__ == "__main__":
    unittest.main()
