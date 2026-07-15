import unittest
import tempfile
from pathlib import Path
from unittest import mock

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from core.int4_convrot_engine import (
    INT4_CONVROT_FORMAT,
    INT4_QUANT_GROUP_SIZE,
    CONVROT_GROUP_SIZE,
    build_quant_layer_metadata,
    validate_int4_convrot_request,
)


class Int4ConvRotEngineTests(unittest.TestCase):
    def test_ltx23_simple_request_is_accepted(self):
        self.assertEqual(
            validate_int4_convrot_request("LTX-2.3", "Simple"),
            None,
        )

    def test_only_ltx23_simple_request_is_accepted_in_phase_one(self):
        self.assertIn("LTX-2.3", validate_int4_convrot_request("WAN 2.2", "Simple"))
        self.assertIn("Simple", validate_int4_convrot_request("LTX-2.3", "Optimizer-driven"))

    def test_quant_metadata_describes_the_comfyui_convrot_layout(self):
        self.assertEqual(
            build_quant_layer_metadata(),
            {
                "format": INT4_CONVROT_FORMAT,
                "convrot_groupsize": CONVROT_GROUP_SIZE,
                "quant_group_size": INT4_QUANT_GROUP_SIZE,
            },
        )

    def test_rejects_non_2d_and_incompatible_weight_shapes(self):
        from core.int4_convrot_engine import validate_quantizable_tensor

        self.assertIn("2D", validate_quantizable_tensor("layer.weight", torch.ones(256)))
        self.assertIn("256", validate_quantizable_tensor("layer.weight", torch.ones(64, 128)))
        self.assertIn("256", validate_quantizable_tensor("layer.weight", torch.ones(64, 320)))

    def test_quantizes_weight_to_packed_payload_and_scale(self):
        from core.int4_convrot_engine import quantize_weight

        payload, scale = quantize_weight(torch.ones(64, 256, dtype=torch.bfloat16))

        self.assertEqual(payload.dtype, torch.int8)
        self.assertEqual(payload.shape, (64, 128))
        self.assertGreater(scale.numel(), 0)

    def test_streaming_conversion_writes_convrot_metadata_and_preserves_structural_tensor(self):
        from core.int4_convrot_engine import run_int4_convrot_conversion

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.safetensors"
            save_file({
                "model.diffusion_model.transformer_blocks.2.attn1.to_q.weight": torch.ones(64, 256, dtype=torch.bfloat16),
                "model.diffusion_model.adaln_single.linear.weight": torch.ones(4, 4, dtype=torch.bfloat16),
            }, str(source))
            with mock.patch("core.int4_convrot_engine.verify_architecture_match", return_value=(True, "ok")):
                events = list(run_int4_convrot_conversion(tmp, str(source), "output", "LTX-2.3", "Simple", False))

            self.assertEqual(events[-1][1], "INT4 ConvRot complete")
            self.assertGreater(len(events), 2)
            output = Path(tmp) / "output_int4_convrot.safetensors"
            with safe_open(output, framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
                layers = __import__("json").loads(metadata["_quantization_metadata"])["layers"]
                self.assertIn("model.diffusion_model.transformer_blocks.2.attn1.to_q", layers)
                self.assertNotIn("HASH_WILL_BE_CALCULATED_ON_SAVE", metadata["civitai.hash.SHA256"])
                self.assertEqual(len(metadata["civitai.hash.SHA256"]), 64)
                self.assertEqual(handle.get_tensor("model.diffusion_model.transformer_blocks.2.attn1.to_q.weight").shape, (64, 128))
                self.assertEqual(handle.get_tensor("model.diffusion_model.adaln_single.linear.weight").shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
