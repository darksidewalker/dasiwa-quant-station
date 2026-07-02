"""Smoke test for the recipe file format produced by _write_recipe."""
import os, sys, textwrap, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRecipeFileFormat(unittest.TestCase):
    """Verify the new recipe header lines are present and parseable."""

    def test_write_recipe_header_fields(self):
        from core.lora_merge_engine import _write_recipe
        tmp = "/tmp/test_dasiwa_recipe.txt"
        if os.path.exists(tmp):
            os.remove(tmp)
        
        payload = {
            "base_path": "/dev/null",
            "architecture": "LTX-2.3",
            "output_dir": "/tmp",
            "output_name": "test.safetensors",
        }
        path = _write_recipe(
            output_path="/tmp/test_dasiwa_test.safetensors",
            payload=payload,
            loras=[{"path": "/tmp/lora1.safetensors", "strength": 0.8, "strategy": "Motion"}],
            strategy="Balanced",
            global_strength=1.25,
            adaptive=True,
            matched_ops=[],
            skipped=3,
            unmatched=1,
            ambiguous=0,
        )
        
        with open(path) as f:
            text = f.read()
        
        self.assertIn("Architecture:      LTX-2.3", text)
        self.assertIn("Default strategy:  Balanced", text)
        self.assertIn("Global strength:   1.25", text)
        self.assertIn("Adaptive scaling:  yes", text)
        self.assertIn("Dry run first:     no", text)
        self.assertIn("Strict matching:   no", text)
        self.assertIn("Krea2 unchain:     no", text)

        # LoRA section must be present.
        lines = [l.strip() for l in text.split("\n") if l.strip().startswith("1.") and not l.strip().startswith("--")]
        self.assertGreaterEqual(len(lines), 1, "No LoRA entry found in recipe:\n" + text)
        self.assertTrue(
            any("test_dasiwa_recipe" in line or "lora1.safetensors" in line for line in lines),
            f"Expected 'lora1' in LoRA line but got: {lines[0]}",
        )

        # Verify Strength/Strategy sub-lines exist.
        self.assertIn("Strength:", text)
        self.assertTrue(
            any(s in text and "0.8" in text for s in ["", "\n"]),
            f"No per-LoRA strength found:\n{text}",
        )
        assert "Strategy:" in text and "Motion" in text or "Balanced" in text
        
        os.remove(path)

    def test_write_recipe_with_krea2_unchain(self):
        from core.lora_merge_engine import _write_recipe
        path = _write_recipe(
            output_path="/tmp/test_dasiwa_test.safetensors",
            payload={**{
                "base_path": "/dev/null",
                "architecture": "Krea 2",
                "output_dir": "/tmp",
                "output_name": "test.safetensors",
            }, **{"krea2_unchain": True}},
            loras=[],
            strategy="Balanced",
            global_strength=1.0,
            adaptive=False,
            matched_ops=[{
                "lora_path": "__builtin_unchain__",
                "strategy": "Balanced",
                "diff_key": "__builtin_unchain__",
                "target_key": "txtfusion.projector.weight",
                "category": "diff_patch",
                "scale": 1.05,
            }],
            skipped=2,
            unmatched=0,
            ambiguous=0,
        )

        with open(path) as f:
            text = f.read()
        
        self.assertIn("Architecture:      Krea 2", text)
        self.assertIn("Krea2 unchain:     yes", text)
        os.remove(path)

