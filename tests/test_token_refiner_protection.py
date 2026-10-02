"""Checkpoint-only protection and adapter no-silent-drop regression fixtures."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from core.lora_merge_engine import is_token_refiner_key, run_lora_merge
from core.lora_compose_engine import run_lora_compose


class TokenRefinerProtectionTests(unittest.TestCase):
    def fixtures(self, root, prefix=""):
        ref = prefix + "token_refiner.blocks.0.attn.qkv_proj"
        bias = prefix + "token_refiner.blocks.1.mlp.fc1.bias"
        main = prefix + "blocks.0.attn.qkv_proj"
        base = root / "base.safetensors"
        tensors = {ref + ".weight": torch.ones(2, 2), bias: torch.ones(2),
                   main + ".weight": torch.ones(2, 2)}
        save_file(tensors, str(base))
        paths = []
        for name in ("a", "b"):
            path = root / (name + ".safetensors")
            save_file({ref + ".lora_A.weight": torch.ones(1, 2),
                       ref + ".lora_B.weight": torch.ones(2, 1),
                       main + ".lora_A.weight": torch.ones(1, 2),
                       main + ".lora_B.weight": torch.ones(2, 1),
                       bias + ".diff": torch.ones(2)}, str(path))
            paths.append(path)
        return {"base_path": str(base), "loras": [{"path": str(p)} for p in paths],
                "architecture": "MiniMax H3", "merge_device": "cpu", "watermark": False,
                "output_path": str(root / "out.safetensors")}, tensors

    def test_exact_architecture_aware_names(self):
        for prefix in ("", "diffusion_model.", "model.diffusion_model.", "base_model.model."):
            self.assertTrue(is_token_refiner_key(prefix + "token_refiner.blocks.0.weight", "MiniMax H3"))
            self.assertTrue(is_token_refiner_key(prefix + "txtfusion.refiner_blocks.0.weight", "Krea 2"))
        for key in ("tokenizer.weight", "text_encoder.weight", "mytoken_refiner.x", "token_refiner_extra.x", "blocks.0.weight"):
            self.assertFalse(is_token_refiner_key(key, "MiniMax H3"))
        self.assertFalse(is_token_refiner_key("token_refiner.x", "WAN 2.2"))
        self.assertFalse(is_token_refiner_key("txtfusion.layerwise_blocks.0.weight", "Krea 2"))

    def test_checkpoint_pair_and_diff_protected_in_both_algorithms(self):
        for algorithm in ("additive", "consensus"):
            for prefix in ("", "model.diffusion_model.", "base_model.model."):
                for enabled in (None, False, True):
                    with self.subTest(algorithm=algorithm, prefix=prefix, enabled=enabled), tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp)
                        payload, base = self.fixtures(root, prefix)
                        payload["merge_algorithm"] = algorithm
                        if enabled is not None:
                            payload["protect_token_refiner"] = enabled
                        events = list(run_lora_merge(payload))
                        self.assertEqual(events[-1]["status"], "finished")
                        output = load_file(payload["output_path"])
                        for key, value in base.items():
                            if "token_refiner" in key and enabled:
                                torch.testing.assert_close(output[key], value, rtol=0, atol=0)
                            else:
                                self.assertTrue(torch.all(output[key] > value), key)
                        recipe = root.joinpath("out.txt").read_text()
                        self.assertIn("Protect Token Refiner: " + ("yes" if enabled else "no"), recipe)
                        self.assertIn("Skipped (Token Refiner): " + ("4" if enabled else "0"), recipe)

    def test_krea_known_refiner_only_and_quant_preservation_unchanged(self):
        from utils.minimax_h3_layer_profiles import is_minimax_h3_preserved_key
        from core.layer_config_builder import PRESERVE_PATTERNS
        import re
        self.assertFalse(is_minimax_h3_preserved_key("token_refiner.blocks.0.attn.qkv_proj.weight"))
        self.assertTrue(is_minimax_h3_preserved_key("token_refiner.blocks.0.attn.q_norm.weight"))
        self.assertTrue(any(re.search(p, "token_refiner.blocks.0.attn.qkv_proj.weight")
                            for p in PRESERVE_PATTERNS["MiniMax H3"]))
        for enabled in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                base, adapter, out = (root / name for name in ("base.safetensors", "adapter.safetensors", "out.safetensors"))
                ref, main = "txtfusion.refiner_blocks.0.attn.wq", "txtfusion.layerwise_blocks.0.attn.wq"
                save_file({ref + ".weight": torch.ones(2, 2), main + ".weight": torch.ones(2, 2)}, str(base))
                save_file({key + ".diff": torch.ones(2, 2) for key in (ref, main)}, str(adapter))
                list(run_lora_merge({"base_path": str(base), "loras": [{"path": str(adapter)}],
                    "architecture": "Krea 2", "protect_token_refiner": enabled, "merge_device": "cpu",
                    "watermark": False, "output_path": str(out)}))
                output = load_file(str(out))
                torch.testing.assert_close(output[ref + ".weight"], torch.full((2, 2), 1.0 if enabled else 2.0))
                torch.testing.assert_close(output[main + ".weight"], torch.full((2, 2), 2.0))

    def test_dry_run_reports_skips_without_writing(self):
        for algorithm in ("additive", "consensus"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                payload, _ = self.fixtures(root)
                payload.update(protect_token_refiner=True, dry_run=True, merge_algorithm=algorithm)
                events = list(run_lora_merge(payload))
                report = next(json.loads(e["text"]) for e in events if e.get("text", "").startswith('{\n  "per_lora_summary"'))
                for summary in report["per_lora_summary"].values():
                    self.assertEqual(summary["skipped_breakdown"]["token_refiner"], 2)
                    self.assertEqual(summary["matched"], 1)
                    self.assertEqual(summary["unmatched"], 0)
                self.assertFalse(root.joinpath("out.safetensors").exists())
                self.assertFalse(root.joinpath("out.txt").exists())

    def test_real_bridge_forwards_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload, _ = self.fixtures(Path(tmp))
            payload.update(protect_token_refiner=True, dry_run=True)
            result = subprocess.run([sys.executable, "scripts/go_bridge.py", "lora-merge", "--json", json.dumps(payload)],
                                    cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            events = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertTrue(any("skipped_token_refiner=4" in e.get("text", "") for e in events))
            self.assertEqual(events[-1]["status"], "dry-run complete")

    def test_standalone_bridge_rejected_even_in_skip_mode_before_artifacts(self):
        for dry_run in (True, False):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                payload, _ = self.fixtures(root)
                bridge = root / "bunnyH3ConditioningBridge_v10.safetensors"
                save_file({f"fc{i}.{kind}": torch.ones((2, 3) if kind == "weight" else (2,))
                           for i in (1, 2, 3) for kind in ("weight", "bias")}, str(bridge))
                payload["loras"][1]["path"] = str(bridge)
                payload.update(dry_run=dry_run, mismatch_mode="skip")
                with self.assertRaisesRegex(ValueError, "Unsupported standalone module") as raised:
                    list(run_lora_compose(payload))
                self.assertIn(str(bridge), str(raised.exception))
                self.assertIn("load the standalone conditioning bridge/module separately", str(raised.exception))
                self.assertFalse(root.joinpath("out.safetensors").exists())
                self.assertFalse(root.joinpath("out.txt").exists())
                self.assertFalse(root.joinpath("out.safetensors.tmp").exists())

    def test_mixed_extra_weights_reported_and_compose_does_not_protect_refiner(self):
        for dry_run in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                paths = []
                for name in ("a", "b"):
                    path = root / (name + ".safetensors")
                    save_file({"token_refiner.blocks.0.attn.qkv_proj.lora_A.weight": torch.ones(1, 2),
                               "token_refiner.blocks.0.attn.qkv_proj.lora_B.weight": torch.ones(2, 1),
                               "fc1.weight": torch.ones(2, 3), "fc1.bias": torch.ones(2)}, str(path))
                    paths.append(path)
                out = root / "out.safetensors"
                events = list(run_lora_compose({"loras": [{"path": str(p)} for p in paths],
                    "architecture": "MiniMax H3", "protect_token_refiner": True,
                    "merge_device": "cpu", "dry_run": dry_run, "output_path": str(out)}))
                text = "".join(e.get("text", "") for e in events)
                self.assertIn("unsupported/unhandled", text)
                self.assertIn('"fc1.bias"', text)
                self.assertIn('"fc1.weight"', text)
                if not dry_run:
                    output = load_file(str(out))
                    self.assertEqual(len(output), 2)
                    self.assertTrue(all("token_refiner" in k for k in output))
                    recipe = out.with_suffix(".txt").read_text()
                    self.assertIn("fc1.weight", recipe)
                    self.assertIn("Unsupported/unhandled input tensors", recipe)
                else:
                    self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
