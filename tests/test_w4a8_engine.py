import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from core.w4a8_engine import (
    W4A8_FORMAT,
    W4A8_QUANT_GROUP_SIZE,
    W4A8_CONVROT_GROUP_SIZE,
    build_w4a8_layer_metadata,
    validate_w4a8_request,
    is_preserved_key,
    validate_quantizable_tensor,
)


class W4A8EngineTests(unittest.TestCase):
    def test_minimax_h3_simple_request_is_accepted(self):
        self.assertIsNone(validate_w4a8_request("MiniMax H3", "Simple"))

    def test_non_h3_architectures_are_rejected(self):
        for arch in ("LTX-2.3", "WAN 2.2", "Krea 2", "Not set"):
            self.assertIn("MiniMax H3", validate_w4a8_request(arch, "Simple"))

    def test_only_simple_strategy_is_accepted(self):
        self.assertIn("Simple", validate_w4a8_request("MiniMax H3", "Optimizer-driven"))
        self.assertIn("Simple", validate_w4a8_request("MiniMax H3", "Balanced"))

    def test_minimax_h3_preserve_policy(self):
        # Structural / modulation / norm layers stay at source precision.
        self.assertTrue(is_preserved_key("MiniMax H3", "blocks.0.adaln_proj.linear.weight"))
        self.assertTrue(is_preserved_key("MiniMax H3", "blocks.0.attn.q_norm.weight"))
        self.assertTrue(is_preserved_key("MiniMax H3", "blocks.0.norm1.weight"))
        self.assertTrue(is_preserved_key("MiniMax H3", "adaln_t_table"))
        self.assertTrue(is_preserved_key("MiniMax H3", "time_embedder.proj_in.weight"))
        self.assertTrue(is_preserved_key("MiniMax H3", "token_refiner.final_norm.weight"))
        # The four heavy linears are NOT preserved (packed by W4A8).
        self.assertFalse(is_preserved_key("MiniMax H3", "blocks.0.attn.qkv_proj.weight"))
        self.assertFalse(is_preserved_key("MiniMax H3", "blocks.0.attn.out_proj.weight"))
        self.assertFalse(is_preserved_key("MiniMax H3", "blocks.0.mlp.fc1.weight"))
        self.assertFalse(is_preserved_key("MiniMax H3", "blocks.0.mlp.fc2.weight"))

    def test_layer_metadata_matches_reference_quants(self):
        self.assertEqual(
            build_w4a8_layer_metadata(),
            {
                "format": W4A8_FORMAT,
                "group_size": W4A8_QUANT_GROUP_SIZE,
                "convrot": True,
                "convrot_groupsize": W4A8_CONVROT_GROUP_SIZE,
            },
        )
        self.assertEqual(W4A8_FORMAT, "asym_w4a8_int8")

    def test_rejects_incompatible_weight_shapes(self):
        self.assertIn("2D", validate_quantizable_tensor("layer.weight", torch.ones(256)))
        self.assertIn("16", validate_quantizable_tensor("layer.weight", torch.ones(64, 248)))
        self.assertIn("256", validate_quantizable_tensor("layer.weight", torch.ones(64, 240)))
        self.assertIsNone(validate_quantizable_tensor("layer.weight", torch.ones(64, 512, dtype=torch.bfloat16)))
        self.assertIn(".weight", validate_quantizable_tensor("layer.bias", torch.ones(64)))

    def test_quantize_weight_emits_reference_companion_set(self):
        from core.w4a8_engine import quantize_weight

        weight = torch.randn(64, 512, dtype=torch.bfloat16) * 0.02
        companions = quantize_weight(weight)
        self.assertEqual(companions[""].dtype, torch.int8)
        self.assertEqual(companions[""].shape, (64, 256))  # packed K/2
        self.assertEqual(companions["_s_rel"].shape, (64, 32))  # K/group_size
        self.assertEqual(companions["_s_rel"].dtype, torch.float8_e4m3fn)
        self.assertEqual(companions["_s_channel"].shape, (64,))
        self.assertEqual(companions["_codebook"].shape, (16,))
        self.assertEqual(companions["_codebook"].dtype, torch.float32)

    def test_streaming_conversion_writes_reference_layout(self):
        from core.w4a8_engine import run_w4a8_conversion

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.safetensors"
            save_file({
                "blocks.0.attn.out_proj.weight": torch.ones(64, 512, dtype=torch.bfloat16),
                "blocks.0.adaln_proj.linear.weight": torch.ones(8, 8, dtype=torch.bfloat16),
                "rope.inv_freq": torch.ones(16, dtype=torch.float32),
            }, str(source))
            with mock.patch("core.w4a8_engine.verify_architecture_match", return_value=(True, "ok")):
                events = list(run_w4a8_conversion(tmp, str(source), "output", "MiniMax H3", "Simple", False))

            self.assertEqual(events[-1][1], "W4A8 complete")
            self.assertIn("1 quantized", events[-1][0])
            self.assertIn("2 preserved", events[-1][0])
            output = Path(tmp) / "output_w4a8.safetensors"
            self.assertTrue(output.exists())
            with safe_open(output, framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                # Packed companions under the reference key names.
                for name in ("weight", "weight_s_rel", "weight_s_channel", "weight_codebook"):
                    self.assertIn(f"blocks.0.attn.out_proj.{name}", keys)
                # Structural tensors preserved verbatim.
                self.assertIn("blocks.0.adaln_proj.linear.weight", keys)
                self.assertIn("rope.inv_freq", keys)
                self.assertNotIn("blocks.0.adaln_proj.linear.weight_s_rel", keys)
                self.assertEqual(handle.get_tensor("blocks.0.attn.out_proj.weight").shape, (64, 256))
                self.assertEqual(handle.get_tensor("blocks.0.attn.out_proj.weight").dtype, torch.int8)
                self.assertEqual(
                    handle.get_tensor("blocks.0.attn.out_proj.weight_s_rel").dtype,
                    torch.float8_e4m3fn,
                )
                layers = json.loads(handle.metadata()["_quantization_metadata"])["layers"]
                self.assertEqual(
                    layers["blocks.0.attn.out_proj"],
                    {"format": "asym_w4a8_int8", "group_size": 16, "convrot": True, "convrot_groupsize": 256},
                )
                self.assertNotIn("blocks.0.adaln_proj.linear", layers)
                self.assertNotIn("civitai.hash.SHA256", handle.metadata())

    def test_lossy_source_is_rejected(self):
        from core.w4a8_engine import run_w4a8_conversion

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.safetensors"
            save_file(
                {"blocks.0.attn.out_proj.weight": torch.ones(64, 512, dtype=torch.bfloat16)},
                str(source),
                metadata={
                    "_quantization_metadata": json.dumps(
                        {"layers": {"blocks.0.attn.out_proj": {"format": "asym_w4a8_int8", "convrot": True}}}
                    )
                },
            )
            with mock.patch("core.w4a8_engine.verify_architecture_match", return_value=(True, "ok")):
                events = list(run_w4a8_conversion(tmp, str(source), "output", "MiniMax H3", "Simple", False))
            self.assertEqual(events[0][1], "Aborted: lossy source")


if __name__ == "__main__":
    unittest.main()
