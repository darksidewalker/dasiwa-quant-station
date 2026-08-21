"""
Tests for the model-level merge engine (h3_hybrid recipe).

Covers:
  - order-agnostic role resolution (fl2va/ref2va in either argument order)
  - byte-identical reversed-order output
  - overlay-only keys being discarded (output keeps the base key set)
  - missing overlay keys / shape-dtype mismatch failing closed
  - dry-run not writing output
"""

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from core.model_merge_engine import run_model_merge, list_recipes


def _h3_tensors(block_count=4, distinct_blocks=(), base=0.0, distinct=1.0):
    """Tiny H3-like tensor dict: shared keys + per-block adaln values.

    adaln values are ``distinct`` for blocks in ``distinct_blocks``, ``base``
    elsewhere, so FL/REF pairs with different base/distinct values prove
    overlay provenance.
    """
    t = {}
    for i in range(block_count):
        lo = i in distinct_blocks
        t[f"blocks.{i}.adaln_proj.linear.weight"] = torch.full((8, 4), distinct if lo else base)
        t[f"blocks.{i}.adaln_proj.linear.bias"] = torch.full((4,), distinct if lo else base)
        t[f"blocks.{i}.attn.qkv_proj.weight"] = torch.full((4, 4), float(i + 1))
        t[f"blocks.{i}.norm1.weight"] = torch.full((4,), 2.0)
    t["adaln_t_table"] = torch.full((1025, 8), 0.5)
    t["video_patch_proj.weight"] = torch.full((4, 4), 3.0)
    return t


def _read_safetensors_keys(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return [k for k in header if k != "__metadata__"]


def _read_safetensors_tensor(path, key):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        spec = header[key]
        start, end = spec["data_offsets"]
        f.seek(8 + n + start)
        raw = f.read(end - start)
    if spec["dtype"] == "F32":
        return torch.frombuffer(bytearray(raw), dtype=torch.float32).clone().reshape(spec["shape"])
    raise NotImplementedError(spec["dtype"])


def _payload(base, overlay, out_dir, **extra):
    return {
        "base_path": base,
        "overlay_path": overlay,
        "architecture": "MiniMax H3",
        "recipe": "h3_hybrid",
        "output_dir": out_dir,
        "output_name": "out_hybrid.safetensors",
        "watermark": False,
        **extra,
    }


def _events(payload):
    return list(run_model_merge(payload))


class H3ModelMergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.fl = Path(self.dir) / "minimax_h3_fl2va_test.safetensors"
        self.ref = Path(self.dir) / "minimax_h3_ref2va_test.safetensors"

    def tearDown(self):
        self.tmp.cleanup()

    def _write_pair(self, ref_extra_keys=(), ref_mismatch=False):
        """FL and REF share the same keys; adaln provenance values differ."""
        save_file(_h3_tensors(distinct_blocks=(), base=0.0), self.fl)
        ref_t = _h3_tensors(distinct_blocks=(2, 3), base=0.0, distinct=1.0)
        if ref_mismatch:
            # shape-mismatched adaln weight on block 2 (the overlay side)
            ref_t["blocks.2.adaln_proj.linear.weight"] = ref_t[
                "blocks.2.adaln_proj.linear.weight"
            ].reshape(16, 2)
        for key in ref_extra_keys:
            ref_t[key] = torch.zeros(4)
        save_file(ref_t, self.ref)

    def test_recipes_listed(self):
        ids = [r["id"] for r in list_recipes()]
        self.assertIn("h3_hybrid", ids)

    def test_order_agnostic_roles(self):
        self._write_pair()
        out1 = Path(self.dir) / "out_a.safetensors"
        out2 = Path(self.dir) / "out_b.safetensors"
        p1 = _payload(str(self.fl), str(self.ref), str(self.dir), output_name="out_a.safetensors")
        p2 = _payload(str(self.ref), str(self.fl), str(self.dir), output_name="out_b.safetensors")
        _events(p1)
        _events(p2)
        self.assertTrue(out1.exists())
        self.assertTrue(out2.exists())
        self.assertEqual(out1.read_bytes(), out2.read_bytes())

    def test_overlay_provenance(self):
        self._write_pair()
        _events(_payload(str(self.fl), str(self.ref), str(self.dir)))
        out = Path(self.dir) / "out_hybrid.safetensors"
        # overlay (blocks 2..3) adaln must come from REF (value 1.0)
        self.assertEqual(_read_safetensors_tensor(out, "blocks.2.adaln_proj.linear.weight").mean().item(), 1.0)
        self.assertEqual(_read_safetensors_tensor(out, "blocks.3.adaln_proj.linear.bias").mean().item(), 1.0)
        # base side (blocks 0..1) adaln must come from FL (all-0 base values)
        self.assertEqual(_read_safetensors_tensor(out, "blocks.0.adaln_proj.linear.weight").mean().item(), 0.0)
        self.assertEqual(_read_safetensors_tensor(out, "blocks.1.adaln_proj.linear.bias").mean().item(), 0.0)
        # non-adaln keys always from FL
        self.assertEqual(_read_safetensors_tensor(out, "blocks.2.attn.qkv_proj.weight").mean().item(), 3.0)
        self.assertEqual(_read_safetensors_tensor(out, "video_patch_proj.weight").mean().item(), 3.0)

    def test_metadata_baked_marker(self):
        self._write_pair()
        _events(_payload(str(self.fl), str(self.ref), str(self.dir)))
        out = Path(self.dir) / "out_hybrid.safetensors"
        with open(out, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n)).get("__metadata__", {})
        self.assertEqual(meta.get("minimax_h3_hybrid"), "baked")
        self.assertEqual(meta.get("base_model"), "minimax_h3_fl2va_test.safetensors")
        self.assertEqual(meta.get("overlay_model"), "minimax_h3_ref2va_test.safetensors")

    def test_extra_overlay_keys_discarded(self):
        """Extra overlay-only marker keys (comfy_quant-style) must not break the
        merge; the output keeps the base key set."""
        self._write_pair(ref_extra_keys=("blocks.2.attn.qkv_proj.comfy_quant", "rope.comfy_quant"))
        events = _events(_payload(str(self.fl), str(self.ref), str(self.dir)))
        self.assertEqual(events[-1]["status"], "finished")
        out = Path(self.dir) / "out_hybrid.safetensors"
        keys = set(_read_safetensors_keys(out))
        self.assertNotIn("blocks.2.attn.qkv_proj.comfy_quant", keys)
        self.assertNotIn("rope.comfy_quant", keys)
        self.assertIn("blocks.2.adaln_proj.linear.weight", keys)

    def test_dry_run_writes_nothing(self):
        self._write_pair()
        out = Path(self.dir) / "out_hybrid.safetensors"
        events = _events(_payload(str(self.fl), str(self.ref), str(self.dir), dry_run=True))
        self.assertEqual(events[-1]["status"], "dry-run complete")
        self.assertFalse(out.exists())

    def test_missing_overlay_key_fails(self):
        self._write_pair()
        # REF missing an overlay-side adaln key
        ref_t = {k: v for k, v in _h3_tensors(distinct_blocks=(2, 3)).items()
                 if k != "blocks.2.adaln_proj.linear.weight"}
        save_file(ref_t, self.ref)
        events = _events(_payload(str(self.fl), str(self.ref), str(self.dir)))
        self.assertEqual(events[-1]["status"], "failed")
        self.assertFalse((Path(self.dir) / "out_hybrid.safetensors").exists())

    def test_overlay_shape_mismatch_fails(self):
        self._write_pair(ref_mismatch=True)
        events = _events(_payload(str(self.fl), str(self.ref), str(self.dir)))
        self.assertEqual(events[-1]["status"], "failed")
        self.assertFalse((Path(self.dir) / "out_hybrid.safetensors").exists())

    def test_unclassifiable_filename_fails(self):
        self._write_pair()
        other = Path(self.dir) / "mystery_checkpoint.safetensors"
        save_file(_h3_tensors(), other)
        events = _events(_payload(str(self.fl), str(other), str(self.dir)))
        self.assertEqual(events[-1]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
