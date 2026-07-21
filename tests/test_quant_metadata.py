import tempfile
import unittest
from pathlib import Path

import json
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from core.metadata_manager import (
    calculate_civitai_hashes,
    inject_metadata,
    merge_custom_metadata,
    normalize_quantization_bits,
    update_metadata_preview,
)
from core.safetensors_engine import write_quant_recipe


class QuantMetadataTests(unittest.TestCase):
    def test_civitai_hashes_are_added_to_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            path.write_bytes(b"DaSiWa quant hash test")

            hashes = calculate_civitai_hashes(str(path))
            self.assertEqual(set(hashes), {"AutoV1", "AutoV2", "AutoV3", "SHA256", "CRC32"})
            self.assertEqual(len(hashes["AutoV1"]), 8)
            self.assertEqual(len(hashes["AutoV2"]), 10)
            self.assertEqual(len(hashes["AutoV3"]), 12)
            self.assertEqual(len(hashes["SHA256"]), 64)
            self.assertEqual(len(hashes["CRC32"]), 8)

            meta = merge_custom_metadata("WAN 2.2", "hash-test", str(path), bits="FP8")
            self.assertEqual(meta["civitai.hash.AutoV1"], hashes["AutoV1"])
            self.assertEqual(meta["civitai.hash.AutoV2"], hashes["AutoV2"])
            self.assertEqual(meta["civitai.hash.AutoV3"], hashes["AutoV3"])
            self.assertEqual(meta["civitai.hash.SHA256"], hashes["SHA256"])
            self.assertEqual(meta["civitai.hash.CRC32"], hashes["CRC32"])

    def test_quant_recipe_writes_summary_txt_next_to_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "video_fp8.safetensors"
            out.write_bytes(b"fake quant output")
            recipe = write_quant_recipe(
                output_path=str(out),
                source_path="/models/source.safetensors",
                model_name="video",
                architecture="WAN 2.2",
                fmt="FP8",
                strategy="Simple",
                optimizer_choice="prodigy",
                low_vram=True,
                actcal=False,
                is_full_checkpoint=True,
                layer_config_path="/tmp/layer.json",
                command=["convert_to_quant", "-i", "/models/source.safetensors"],
                metadata_injected=True,
                metadata_message="Header rewritten in place",
                hashes={"SHA256": "A" * 64, "AutoV1": "B" * 8, "AutoV2": "C" * 10, "AutoV3": "D" * 12, "CRC32": "E" * 8},
            )

            text = Path(recipe).read_text()
            self.assertIn("DaSiWa Quantization Recipe", text)
            self.assertIn("Output:            video_fp8.safetensors", text)
            self.assertIn("Format:            FP8", text)
            self.assertIn("Metadata injected: yes", text)
            self.assertIn("AutoV3:            DDDDDDDDDDDD", text)

    def test_quantization_bits_labels_match_actual_quant_target(self):
        self.assertEqual(
            normalize_quantization_bits("INT8 Row-wise ConvRot Runtime"),
            "INT8 Row-wise ConvRot (runtime)",
        )
        self.assertEqual(
            normalize_quantization_bits("INT8 Row-wise ConvRot"),
            "INT8 Tensor-wise",
        )

    def test_metadata_preview_uses_target_quantization_placeholder(self):
        preview = update_metadata_preview("preview-model", "WAN 2.2")
        self.assertIn('"quantization.bits": "{target_quantization}"', preview)

    def test_manual_injection_preserves_existing_quantization_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "int4.safetensors"
            quantization_metadata = json.dumps({
                "format_version": "1.0",
                "layers": {"transformer.weight": {"format": "convrot_w4a4"}},
            })
            save_file(
                {"transformer.weight": torch.ones(2, 2)},
                str(path),
                metadata={
                    "_quantization_metadata": quantization_metadata,
                    "existing.custom": "keep",
                },
            )

            ok, message = inject_metadata(str(path), {"modelspec.title": "Updated title"})

            self.assertTrue(ok, message)
            with safe_open(path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
            self.assertEqual(metadata["_quantization_metadata"], quantization_metadata)
            self.assertEqual(metadata["existing.custom"], "keep")
            self.assertEqual(metadata["modelspec.title"], "Updated title")


if __name__ == "__main__":
    unittest.main()
