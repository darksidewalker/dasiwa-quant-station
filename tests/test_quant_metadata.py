import tempfile
import unittest
from pathlib import Path

import json
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from core.metadata_manager import (
    inject_metadata,
    merge_custom_metadata,
    normalize_quantization_bits,
    update_metadata_preview,
)
from core.safetensors_engine import write_quant_recipe


class QuantMetadataTests(unittest.TestCase):
    def test_hash_keys_are_never_added_to_metadata(self):
        """civitai.hash.* / modelspec.hash_sha256 describe the source
        checkpoint and are not applicable after merging/quantization —
        generated metadata must never carry them."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            path.write_bytes(b"DaSiWa quant hash test")

            meta = merge_custom_metadata("WAN 2.2", "hash-test", str(path), bits="FP8")
            for key in meta:
                self.assertFalse(
                    key.startswith("civitai.hash.") or key == "modelspec.hash_sha256",
                    f"stale hash key {key!r} leaked into generated metadata",
                )

    def test_pre_write_metadata_carries_no_hash_keys(self):
        """No dead placeholders: a not-yet-existing target must not plant any
        hash fields at all."""
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "does_not_exist_yet.safetensors")

            meta = merge_custom_metadata("WAN 2.2", "pre-write", missing, bits="BF16")
            for key in list(meta):
                self.assertFalse(
                    key.startswith("civitai.hash.") or key == "modelspec.hash_sha256",
                    f"stale hash key {key!r} leaked into pre-write metadata",
                )

    def test_preview_mode_carries_no_hash_keys(self):
        preview = json.loads(update_metadata_preview("preview-model", "LTX-2.3"))
        self.assertNotIn("modelspec.hash_sha256", preview)
        for key in preview:
            self.assertFalse(key.startswith("civitai.hash."))

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
            )

            text = Path(recipe).read_text()
            self.assertIn("DaSiWa Quantization Recipe", text)
            self.assertIn("Output:            video_fp8.safetensors", text)
            self.assertIn("Format:            FP8", text)
            self.assertIn("Metadata injected: yes", text)
            self.assertNotIn("Civitai/Common Hashes", text)
            self.assertNotIn("AutoV3", text)

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
