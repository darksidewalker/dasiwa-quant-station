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
from unittest.mock import patch
from pathlib import Path

import torch
from safetensors.torch import save_file

from core.model_merge_engine import (
    run_model_merge,
    list_recipes,
    _classify_h3_family,
    _is_svd_eligible,
    _randomized_svd_cap,
    _commutator_conflict,
    _larv_tier_scales,
    _larv_block_scales,
)


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
        # Test fixtures use 4 blocks; default range (30-49) would select none.
        "block_range_start": extra.pop("block_range_start", 0),
        "block_range_end": extra.pop("block_range_end", 3),
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
        p1 = _payload(str(self.fl), str(self.ref), str(self.dir), output_name="out_a.safetensors",
                      block_range_start=0, block_range_end=3)
        p2 = _payload(str(self.ref), str(self.fl), str(self.dir), output_name="out_b.safetensors",
                      block_range_start=0, block_range_end=3)
        _events(p1)
        _events(p2)
        self.assertTrue(out1.exists())
        self.assertTrue(out2.exists())
        self.assertEqual(out1.read_bytes(), out2.read_bytes())

    def test_overlay_provenance(self):
        self._write_pair()
        # Explicit range 0-3: all blocks overlaid in this 4-block fixture
        _events(_payload(str(self.fl), str(self.ref), str(self.dir),
                         block_range_start=0, block_range_end=3))
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

    def test_base_hash_metadata_not_inherited(self):
        """Hash fields describe the source checkpoint, not the merged output,
        so a base carrying civitai.hash.* / modelspec.hash_sha256 (real or
        placeholder) must not propagate them into merge results."""
        save_file(
            _h3_tensors(distinct_blocks=(), base=0.0),
            self.fl,
            metadata={
                "civitai.hash.AutoV1": "DEADBEEF",
                "civitai.hash.AutoV2": "DEADBEEF00",
                "civitai.hash.AutoV3": "DEADBEEF0000",
                "civitai.hash.SHA256": "DEADBEEF" * 8,
                "civitai.hash.CRC32": "CAFEF00D",
                "modelspec.hash_sha256": "0xDEADBEEF",
                "modelspec.title": "base-title",
            },
        )
        save_file(_h3_tensors(distinct_blocks=(2, 3), base=0.0, distinct=1.0), self.ref)
        _events(_payload(str(self.fl), str(self.ref), str(self.dir)))
        out = Path(self.dir) / "out_hybrid.safetensors"
        with open(out, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n)).get("__metadata__", {})
        for key in list(meta):
            self.assertFalse(
                key.startswith("civitai.hash.") or key == "modelspec.hash_sha256",
                f"inherited base hash key {key!r} leaked into merged output",
            )
        # non-hash metadata is still carried forward
        self.assertEqual(meta.get("modelspec.title"), "base-title")

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

    def test_hybrid_writes_companion_recipe_for_custom_blocks(self):
        self._write_pair()
        fl = _h3_tensors(distinct_blocks=(), base=0.0)
        ref = _h3_tensors(distinct_blocks=(2, 3), base=0.0, distinct=1.0)
        fl["final_layer.adaln_proj.linear.weight"] = torch.zeros(8, 4)
        ref["final_layer.adaln_proj.linear.weight"] = torch.ones(8, 4)
        save_file(fl, self.fl)
        save_file(ref, self.ref)
        name = "out_recipe.safetensors"
        events = _events(_payload(str(self.ref), str(self.fl), str(self.dir),
                                  output_name=name, overlay_blocks=[1, 3],
                                  block_range_start=0, block_range_end=2,
                                  final_adaln_from_overlay=True,
                                  preserve_loader_metadata=False))
        recipe_path = Path(self.dir) / "out_recipe.txt"
        self.assertEqual(events[-1]["status"], "finished")
        self.assertTrue(recipe_path.is_file())
        content = recipe_path.read_text(encoding="utf-8")
        for expected in ("Recipe: h3_hybrid", "Architecture: MiniMax H3",
                         f"Base checkpoint: {self.fl}", f"Overlay checkpoint: {self.ref}",
                         "Blend mode: binary", "Block selection: custom",
                         "Selected blocks: 1, 3", "Final AdaLN requested: yes",
                         "Final AdaLN from overlay: yes",
                         "Preserve loader metadata: no", "Watermark: off",
                         "Overlay tensors: 5"):
            self.assertIn(expected, content)
        self.assertIn(str(recipe_path), "".join(event.get("text", "") for event in events))

    def test_hybrid_range_recipe_records_actual_block_set(self):
        self._write_pair()
        _events(_payload(str(self.fl), str(self.ref), str(self.dir),
                         output_name="range_recipe.safetensors",
                         block_range_start=2, block_range_end=3))
        recipe = (Path(self.dir) / "range_recipe.txt").read_text(encoding="utf-8")
        self.assertIn("Block selection: range", recipe)
        self.assertIn("Block range: 2–3", recipe)
        self.assertIn("Selected blocks: 2, 3", recipe)

    def test_dry_run_writes_nothing(self):
        self._write_pair()
        out = Path(self.dir) / "out_hybrid.safetensors"
        events = _events(_payload(str(self.fl), str(self.ref), str(self.dir), dry_run=True))
        self.assertEqual(events[-1]["status"], "dry-run complete")
        self.assertFalse(out.exists())
        self.assertFalse((Path(self.dir) / "out_hybrid.txt").exists())

    def test_hybrid_streams_progress_events(self):
        """h3_hybrid streams `h3_hybrid N/M tensors` progress ticks as
        `type: "progress"` events carrying only `text` — the UI renders them
        as a self-updating single line inside the console window (no
        status-bar output, nothing appended to the scrollback)."""
        self._write_pair()
        events = _events(_payload(str(self.fl), str(self.ref), str(self.dir)))
        progress = [
            e for e in events
            if e.get("type") == "progress" and "h3_hybrid" in e.get("text", "")
        ]
        total = len(_read_safetensors_keys(self.fl))
        # Fixture is small (< 100 keys) → progress_every = 1 → one progress
        # event per tensor.
        self.assertGreaterEqual(len(progress), total)
        self.assertTrue(progress[0]["text"].startswith(f"h3_hybrid 1/{total}"))
        self.assertTrue(progress[-1]["text"].startswith(f"h3_hybrid {total}/{total}"))
        # Progress ticks must not touch the topbar status line: no `status` field.
        self.assertFalse(any(e.get("status") for e in progress))
        self.assertEqual(events[-1], {"type": "done", "status": "finished"})

    def test_custom_block_range(self):
        """Only specified block range is overlaid; others stay base."""
        self._write_pair()
        out_name = "out_range.safetensors"
        # Only overlay blocks 2-3 (last 2 of 4)
        _events(_payload(str(self.fl), str(self.ref), str(self.dir),
                         output_name=out_name,
                         block_range_start=2, block_range_end=3))
        out = Path(self.dir) / out_name
        # Blocks 2-3: overlay (value 1.0)
        self.assertEqual(_read_safetensors_tensor(out, "blocks.2.adaln_proj.linear.weight").mean().item(), 1.0)
        self.assertEqual(_read_safetensors_tensor(out, "blocks.3.adaln_proj.linear.weight").mean().item(), 1.0)
        # Blocks 0-1: base (value 0.0)
        self.assertEqual(_read_safetensors_tensor(out, "blocks.0.adaln_proj.linear.weight").mean().item(), 0.0)
        self.assertEqual(_read_safetensors_tensor(out, "blocks.1.adaln_proj.linear.weight").mean().item(), 0.0)

    def test_sparse_block_selection_overlays_only_requested_adaln(self):
        self._write_pair()
        save_file(_h3_tensors(distinct_blocks=(0, 1, 2, 3), base=0.0, distinct=1.0), self.ref)
        name = "out_sparse.safetensors"
        events = _events(_payload(str(self.fl), str(self.ref), str(self.dir),
                                  output_name=name, block_range_start=0,
                                  block_range_end=3, overlay_blocks=[0, 2]))
        self.assertEqual(events[-1]["status"], "finished")
        for i in range(4):
            for kind in ("weight", "bias"):
                value = _read_safetensors_tensor(Path(self.dir) / name,
                    f"blocks.{i}.adaln_proj.linear.{kind}").mean().item()
                self.assertEqual(value, 1.0 if i in (0, 2) else 0.0)
        meta, _ = _read_meta(Path(self.dir) / name)
        self.assertEqual(json.loads(meta["h3_hybrid_overlay_blocks"]), [0, 2])
        self.assertEqual(meta["h3_hybrid_block_selection"], "custom")
        self.assertIn("blocks 0,2", "".join(event.get("text", "") for event in events))

    def test_sparse_selection_rejects_invalid_indices_without_output(self):
        self._write_pair()
        for selection in ([], [0, 0], [0, 4], [-1, 2], [0, "2"], "0,2", [True]):
            name = "out_invalid.safetensors"
            events = _events(_payload(str(self.fl), str(self.ref), str(self.dir),
                                      output_name=name, overlay_blocks=selection))
            self.assertEqual(events[-1]["status"], "failed", selection)
            self.assertFalse((Path(self.dir) / name).exists(), selection)

    def test_sparse_selection_and_final_layer_are_independent(self):
        fl = _h3_tensors(base=0.0)
        ref = _h3_tensors(distinct_blocks=(1,), base=0.0, distinct=1.0)
        fl["final_layer.adaln_proj.linear.weight"] = torch.zeros(8, 4)
        ref["final_layer.adaln_proj.linear.weight"] = torch.full((8, 4), 9.0)
        save_file(fl, self.fl)
        save_file(ref, self.ref)
        name = "out_sparse_final.safetensors"
        _events(_payload(str(self.fl), str(self.ref), str(self.dir),
                         output_name=name, overlay_blocks=[1], final_adaln_from_overlay=True))
        self.assertEqual(_read_safetensors_tensor(Path(self.dir) / name,
            "final_layer.adaln_proj.linear.weight").mean().item(), 9.0)
        self.assertEqual(_read_safetensors_tensor(Path(self.dir) / name,
            "blocks.2.adaln_proj.linear.weight").mean().item(), 0.0)

    def test_single_block_range(self):
        """A single-block range overlays only that block."""
        self._write_pair()
        out_name = "out_single.safetensors"
        _events(_payload(str(self.fl), str(self.ref), str(self.dir),
                         output_name=out_name,
                         block_range_start=2, block_range_end=2))
        out = Path(self.dir) / out_name
        self.assertEqual(_read_safetensors_tensor(out, "blocks.2.adaln_proj.linear.weight").mean().item(), 1.0)
        for i in (0, 1, 3):
            self.assertEqual(_read_safetensors_tensor(out, f"blocks.{i}.adaln_proj.linear.weight").mean().item(), 0.0)

    def test_range_clamping(self):
        """Range beyond actual block count is clamped to valid range."""
        self._write_pair()
        out_name = "out_clamped.safetensors"
        # Request blocks 10-20 but only 4 exist → clamped to 3-3
        _events(_payload(str(self.fl), str(self.ref), str(self.dir),
                         output_name=out_name,
                         block_range_start=10, block_range_end=20))
        out = Path(self.dir) / out_name
        # Only last block (3) should be overlaid after clamping
        self.assertEqual(_read_safetensors_tensor(out, "blocks.3.adaln_proj.linear.weight").mean().item(), 1.0)
        for i in (0, 1, 2):
            self.assertEqual(_read_safetensors_tensor(out, f"blocks.{i}.adaln_proj.linear.weight").mean().item(), 0.0)

    def test_final_adaln_from_overlay(self):
        """final_adaln_from_overlay includes final_layer.adaln_proj in overlay."""
        # Build fixture with final_layer.adaln_proj
        fl = _h3_tensors(distinct_blocks=(), base=0.0)
        ref = _h3_tensors(distinct_blocks=(2, 3), base=0.0, distinct=1.0)
        fl["final_layer.adaln_proj.linear.weight"] = torch.full((8, 4), 0.0)
        fl["final_layer.adaln_proj.linear.bias"] = torch.full((4,), 0.0)
        ref["final_layer.adaln_proj.linear.weight"] = torch.full((8, 4), 9.0)
        ref["final_layer.adaln_proj.linear.bias"] = torch.full((4,), 9.0)
        save_file(fl, self.fl)
        save_file(ref, self.ref)

        # Without final_adaln toggle: final layer stays base (0.0)
        out_name1 = "out_no_final.safetensors"
        _events(_payload(str(self.fl), str(self.ref), str(self.dir),
                         output_name=out_name1,
                         block_range_start=2, block_range_end=3))
        self.assertEqual(
            _read_safetensors_tensor(Path(self.dir) / out_name1, "final_layer.adaln_proj.linear.weight").mean().item(),
            0.0
        )

        # With final_adaln toggle: final layer comes from overlay (9.0)
        out_name2 = "out_with_final.safetensors"
        _events(_payload(str(self.fl), str(self.ref), str(self.dir),
                         output_name=out_name2,
                         block_range_start=2, block_range_end=3,
                         final_adaln_from_overlay=True))
        self.assertEqual(
            _read_safetensors_tensor(Path(self.dir) / out_name2, "final_layer.adaln_proj.linear.weight").mean().item(),
            9.0
        )

    def test_modality_slices_keep_audio_and_select_video_by_block(self):
        fl, ref = {}, {}
        for block in range(4):
            for suffix in ("weight", "bias"):
                key = f"blocks.{block}.adaln_proj.linear.{suffix}"
                shape = (36, 4) if suffix == "weight" else (36,)
                fl[key] = torch.zeros(shape)
                ref[key] = torch.full(shape, 1.0 + block)
        fl["final_layer.audio_out.weight"] = torch.zeros((2, 6))
        fl["final_layer.audio_out.bias"] = torch.zeros(2)
        ref["final_layer.audio_out.weight"] = torch.ones((2, 6))
        ref["final_layer.audio_out.bias"] = torch.ones(2)
        save_file(fl, self.fl)
        save_file(ref, self.ref)
        events = _events(_payload(str(self.fl), str(self.ref), self.dir,
                                 overlay_blocks=[1, 3], modality_mode="split",
                                 audio_blocks="all", text_blocks="base",
                                 audio_out_from_overlay=True))
        self.assertEqual(events[-1]["status"], "finished")
        out = Path(self.dir) / "out_hybrid.safetensors"
        from safetensors import safe_open
        with safe_open(out, framework="pt") as f:
            for block in range(4):
                for suffix in ("weight", "bias"):
                    tensor = f.get_tensor(f"blocks.{block}.adaln_proj.linear.{suffix}")
                    rows = tensor.reshape(3, 6, 2, -1)
                    self.assertEqual(rows[0].unique().tolist(), [float(block + 1)] if block in (1, 3) else [0.0])
                    self.assertEqual(rows[1].unique().tolist(), [0.0])
                    self.assertEqual(rows[2].unique().tolist(), [float(block + 1)])
            self.assertEqual(f.get_tensor("final_layer.audio_out.bias").tolist(), [1.0, 1.0])
            meta = f.metadata()
            self.assertEqual(meta["h3_hybrid_modality_mode"], "split")
            self.assertEqual(meta["h3_hybrid_audio_blocks"], "[0, 1, 2, 3]")
            self.assertEqual(meta["h3_hybrid_text_blocks"], "[]")
            self.assertEqual(meta["h3_hybrid_video_blocks"], "[1, 3]")
        recipe = (Path(self.dir) / "out_hybrid.txt").read_text()
        self.assertIn("Audio blocks: 0, 1, 2, 3", recipe)
        self.assertIn("Video blocks: 1, 3", recipe)
        self.assertIn("Audio output head from overlay: yes", recipe)

    def test_modality_split_dry_run_reports_recipe_without_files(self):
        fl, ref = {}, {}
        for block in range(4):
            for suffix in ("weight", "bias"):
                key = f"blocks.{block}.adaln_proj.linear.{suffix}"
                shape = (36, 4) if suffix == "weight" else (36,)
                fl[key] = torch.zeros(shape)
                ref[key] = torch.ones(shape)
        save_file(fl, self.fl)
        save_file(ref, self.ref)
        events = _events(_payload(str(self.fl), str(self.ref), self.dir,
                                 overlay_blocks=[2], modality_mode="split",
                                 audio_blocks="selected", text_blocks="base",
                                 dry_run=True))
        self.assertEqual(events[-1]["status"], "dry-run complete")
        self.assertIn("audio=[2]", "".join(event.get("text", "") for event in events))
        self.assertFalse((Path(self.dir) / "out_hybrid.safetensors").exists())
        self.assertFalse((Path(self.dir) / "out_hybrid.txt").exists())

    def test_modality_split_rejects_incompatible_adaln_before_writing(self):
        self._write_pair()
        events = _events(_payload(str(self.fl), str(self.ref), self.dir,
                                 overlay_blocks=[2], modality_mode="split"))
        self.assertEqual(events[-1]["status"], "failed")
        self.assertFalse((Path(self.dir) / "out_hybrid.safetensors").exists())
        self.assertFalse((Path(self.dir) / "out_hybrid.txt").exists())

    def test_larv_mode_is_rejected_before_any_output_write(self):
        """H3 LARV is blocked after black-video regressions; it must not write output."""
        self._write_pair()
        out_name = "out_larv_rejected.safetensors"
        events = _events(_payload(
            str(self.fl), str(self.ref), str(self.dir), output_name=out_name,
            block_range_start=2, block_range_end=3, blend_mode="larv_tiered",
        ))
        self.assertEqual(events[-1]["status"], "failed")
        self.assertIn("LARV is disabled", "".join(event.get("text", "") for event in events))
        self.assertFalse((Path(self.dir) / out_name).exists())

    def test_metadata_records_block_range(self):
        """Merged output metadata records the block range used."""
        self._write_pair()
        out_name = "out_meta.safetensors"
        _events(_payload(str(self.fl), str(self.ref), str(self.dir),
                         output_name=out_name,
                         block_range_start=2, block_range_end=3))
        meta, _ = _read_meta(Path(self.dir) / out_name)
        self.assertEqual(meta.get("h3_hybrid_block_start"), "2")
        self.assertEqual(meta.get("h3_hybrid_block_end"), "3")

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


# ---------------------------------------------------------------------------
# h3_delta recipe
# ---------------------------------------------------------------------------

def _delta_pair_tensors(ref_shift=1.0):
    """
    Tiny H3-pruned-like pair: FL and REF share keys; REF = FL + shift on
    a few tensors, identical on the rest. Shapes/dtypes chosen to hit
    every SVD-eligibility branch:
      - 2-D trunk matrices (qkv_proj, fc1, adaln_proj.linear.weight) → SVD-eligible
      - 1-D biases / norms / adaln_t_table / rope → exact families
      - adaln_t_table only in this fixture (pruned variant marker)
    """
    fl = {
        "blocks.0.attn.qkv_proj.weight": torch.full((8, 8), 1.0),
        "blocks.0.mlp.fc1.weight": torch.full((8, 8), 2.0),
        "blocks.0.mlp.fc2.weight": torch.full((8, 8), 3.0),
        "blocks.0.adaln_proj.linear.weight": torch.full((4, 8), 0.5),
        "blocks.0.adaln_proj.linear.bias": torch.full((4,), 0.0),
        "blocks.0.norm1.weight": torch.full((8,), 1.0),
        "adaln_t_table": torch.full((16, 8), 2.0),
        "rope.inv_freq": torch.full((16,), 0.1),
        "video_patch_proj.weight": torch.full((8, 8), 4.0),
    }
    ref = {}
    for k, v in fl.items():
        if k in ("blocks.0.attn.qkv_proj.weight", "blocks.0.adaln_proj.linear.bias",
                 "blocks.0.norm1.weight"):
            ref[k] = v + ref_shift
        else:
            ref[k] = v.clone()
    return fl, ref


def _delta_payload(base, overlay, out_dir, out_name="out_delta.safetensors", **extra):
    return {
        "base_path": base,
        "overlay_path": overlay,
        "architecture": "MiniMax H3",
        "recipe": "h3_delta",
        "output_dir": out_dir,
        "output_name": out_name,
        "watermark": False,
        **extra,
    }


def _read_meta(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header.get("__metadata__", {}), header


class H3DeltaRecipeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.fl = Path(self.dir) / "minimax_h3_fl2va_delta_test.safetensors"
        self.ref = Path(self.dir) / "minimax_h3_ref2va_delta_test.safetensors"

    def tearDown(self):
        self.tmp.cleanup()

    def _write_delta_pair(self, missing_in_ref=(), ref_shape_mismatch=False):
        fl, ref = _delta_pair_tensors()
        save_file(fl, self.fl)
        for key in missing_in_ref:
            del ref[key]
        if ref_shape_mismatch:
            ref["blocks.0.mlp.fc1.weight"] = ref["blocks.0.mlp.fc1.weight"].reshape(4, 16)
        save_file(ref, self.ref)

    # -- recipe registration ------------------------------------------------
    def test_recipes_listed_includes_delta(self):
        ids = [r["id"] for r in list_recipes()]
        self.assertIn("h3_delta", ids)

    def test_delta_writes_companion_recipe_with_resolved_parameters(self):
        self._write_delta_pair()
        events = _events(_delta_payload(str(self.fl), str(self.ref), str(self.dir),
                                        rank=0, strength=0))
        self.assertEqual(events[-1]["status"], "finished")
        recipe_path = Path(self.dir) / "out_delta.txt"
        content = recipe_path.read_text(encoding="utf-8")
        for expected in ("Recipe: h3_delta", f"Base checkpoint: {self.fl}",
                         f"Overlay checkpoint: {self.ref}", "Delta rank: 0",
                         "Delta mode: exact", "Delta strength: 1.0",
                         "Watermark: off"):
            self.assertIn(expected, content)
        self.assertIn(str(recipe_path), "".join(event.get("text", "") for event in events))

    # -- exact delta ---------------------------------------------------------
    def test_exact_delta_applies_shift(self):
        self._write_delta_pair()
        events = _events(_delta_payload(str(self.fl), str(self.ref), str(self.dir)))
        self.assertEqual(events[-1]["status"], "finished")
        out = Path(self.dir) / "out_delta.safetensors"
        # shifted trunk tensor: 1.0 + 1.0 = 2.0
        self.assertEqual(_read_safetensors_tensor(out, "blocks.0.attn.qkv_proj.weight").mean().item(), 2.0)
        # unshifted trunk tensor stays
        self.assertEqual(_read_safetensors_tensor(out, "blocks.0.mlp.fc1.weight").mean().item(), 2.0)
        # exact-family bias shifted
        self.assertEqual(_read_safetensors_tensor(out, "blocks.0.adaln_proj.linear.bias").mean().item(), 1.0)
        # untouched tensors keep base values (float32 round-trip on 0.1)
        self.assertEqual(_read_safetensors_tensor(out, "adaln_t_table").mean().item(), 2.0)
        self.assertAlmostEqual(_read_safetensors_tensor(out, "rope.inv_freq").mean().item(), 0.1, places=6)

    def test_exact_delta_metadata(self):
        self._write_delta_pair()
        _events(_delta_payload(str(self.fl), str(self.ref), str(self.dir)))
        meta, _ = _read_meta(Path(self.dir) / "out_delta.safetensors")
        self.assertEqual(meta.get("minimax_h3_delta"), "baked")
        self.assertEqual(meta.get("h3_delta_mode"), "exact")
        self.assertEqual(meta.get("h3_delta_rank"), "0")
        self.assertEqual(meta.get("h3_delta_strength"), "1.000000")
        self.assertEqual(meta.get("h3_delta_variant"), "pruned")
        self.assertEqual(meta.get("base_model"), "minimax_h3_fl2va_delta_test.safetensors")
        self.assertEqual(meta.get("overlay_model"), "minimax_h3_ref2va_delta_test.safetensors")
        self.assertNotIn("h3_delta_energy", meta)  # energy report only in SVD mode

    def test_strength_scales_delta(self):
        self._write_delta_pair()
        out_name = "out_s.safetensors"
        _events(_delta_payload(str(self.fl), str(self.ref), str(self.dir),
                               out_name=out_name, strength=0.5))
        out = Path(self.dir) / out_name
        # shifted trunk tensor: 1.0 + 0.5 * 1.0 = 1.5
        self.assertAlmostEqual(_read_safetensors_tensor(out, "blocks.0.attn.qkv_proj.weight").mean().item(), 1.5)
        # unshifted trunk unchanged
        self.assertEqual(_read_safetensors_tensor(out, "blocks.0.mlp.fc1.weight").mean().item(), 2.0)

    def test_strength_zero_defaults_to_full(self):
        self._write_delta_pair()
        out_name = "out_s0.safetensors"
        _events(_delta_payload(str(self.fl), str(self.ref), str(self.dir),
                               out_name=out_name, strength=0.0))
        out = Path(self.dir) / out_name
        meta, _ = _read_meta(out)
        self.assertEqual(meta.get("h3_delta_strength"), "1.000000")
        self.assertEqual(_read_safetensors_tensor(out, "blocks.0.attn.qkv_proj.weight").mean().item(), 2.0)

    # -- SVD mode ------------------------------------------------------------
    def test_svd_rank_cap_full_rank_is_exact(self):
        """Rank >= min-dim on a rank<=r matrix → exact reconstruction."""
        self._write_delta_pair()
        out_name = "out_svd.safetensors"
        events = _events(_delta_payload(str(self.fl), str(self.ref), str(self.dir),
                                        out_name=out_name, rank=256))
        self.assertEqual(events[-1]["status"], "finished")
        out = Path(self.dir) / out_name
        # cap non-binding on 8x8 deltas → same result as exact
        self.assertEqual(_read_safetensors_tensor(out, "blocks.0.attn.qkv_proj.weight").mean().item(), 2.0)
        self.assertEqual(_read_safetensors_tensor(out, "blocks.0.adaln_proj.linear.bias").mean().item(), 1.0)
        meta, _ = _read_meta(out)
        self.assertEqual(meta.get("h3_delta_mode"), "svd-r256")
        energy = json.loads(meta.get("h3_delta_energy", "{}"))
        # all SVD-eligible tensors: cap non-binding → 1.0 captured
        for fam, rec in energy.items():
            self.assertAlmostEqual(rec["avg_captured"], 1.0)

    def test_svd_rank_low_captures_some_energy(self):
        """Rank-1 cap on a random full-rank delta loses energy but keeps most."""
        fl, ref = _delta_pair_tensors()
        torch.manual_seed(0)
        big_fl = torch.randn(64, 64)
        big_ref = big_fl + torch.randn(64, 64)
        fl["blocks.1.attn.qkv_proj.weight"] = big_fl
        ref["blocks.1.attn.qkv_proj.weight"] = big_ref
        save_file(fl, self.fl)
        save_file(ref, self.ref)
        out_name = "out_svd_low.safetensors"
        _events(_delta_payload(str(self.fl), str(self.ref), str(self.dir),
                               out_name=out_name, rank=1))
        out = Path(self.dir) / out_name
        got = _read_safetensors_tensor(out, "blocks.1.attn.qkv_proj.weight")
        meta, _ = _read_meta(out)
        energy = json.loads(meta["h3_delta_energy"])
        self.assertIn("attn", energy)
        self.assertGreater(energy["attn"]["avg_captured"], 0.5)
        self.assertLess(energy["attn"]["avg_captured"], 1.0)
        # result is base + low-rank approximation (fp32 delta, fp32 base)
        expected_delta = big_ref - big_fl
        best_r1, _cap, _ach = _randomized_svd_cap(expected_delta, 1)
        expected = (big_fl.float() + best_r1).float()
        self.assertTrue(torch.allclose(got, expected, atol=1e-5))

    # -- validation ----------------------------------------------------------
    def test_delta_missing_overlay_key_fails(self):
        self._write_delta_pair(missing_in_ref=("blocks.0.attn.qkv_proj.weight",))
        events = _events(_delta_payload(str(self.fl), str(self.ref), str(self.dir)))
        self.assertEqual(events[-1]["status"], "failed")
        self.assertFalse((Path(self.dir) / "out_delta.safetensors").exists())

    def test_delta_shape_mismatch_fails(self):
        self._write_delta_pair(ref_shape_mismatch=True)
        events = _events(_delta_payload(str(self.fl), str(self.ref), str(self.dir)))
        self.assertEqual(events[-1]["status"], "failed")

    def test_delta_dry_run_writes_nothing(self):
        self._write_delta_pair()
        out = Path(self.dir) / "out_delta.safetensors"
        events = _events(_delta_payload(str(self.fl), str(self.ref), str(self.dir),
                                        dry_run=True, rank=64))
        self.assertEqual(events[-1]["status"], "dry-run complete")
        self.assertFalse(out.exists())
        self.assertFalse((Path(self.dir) / "out_delta.txt").exists())

    def test_delta_order_agnostic_roles(self):
        self._write_delta_pair()
        out_a = "out_da.safetensors"
        out_b = "out_db.safetensors"
        _events(_delta_payload(str(self.fl), str(self.ref), str(self.dir), out_name=out_a))
        _events(_delta_payload(str(self.ref), str(self.fl), str(self.dir), out_name=out_b))
        self.assertTrue(Path(self.dir, out_a).exists())
        self.assertTrue(Path(self.dir, out_b).exists())
        # Both outputs must agree on trunk values (roles resolved internally),
        # but they are NOT byte-identical: the base key order differs by file
        # only when key sets differ — here key sets are identical, so the
        # tensor content is the same while header metadata may differ.
        ta = _read_safetensors_tensor(Path(self.dir, out_a), "blocks.0.attn.qkv_proj.weight")
        tb = _read_safetensors_tensor(Path(self.dir, out_b), "blocks.0.attn.qkv_proj.weight")
        self.assertEqual(ta.mean().item(), 2.0)
        self.assertEqual(tb.mean().item(), 2.0)

    def test_delta_streams_progress_events(self):
        """h3_delta streams `h3_delta N/M tensors` progress events (~100 ticks),
        each a distinct `type: "progress"` event carrying ONLY `text` — the UI
        renders those as a self-updating single line inside the console window,
        without appending to the scrollback or writing the topbar status line
        (regression guard for the h3_delta "no progress in console" bug)."""
        self._write_delta_pair()
        events = _events(_delta_payload(str(self.fl), str(self.ref), str(self.dir)))
        progress = [
            e for e in events
            if e.get("type") == "progress" and "h3_delta" in e.get("text", "")
        ]
        total = len(_read_safetensors_keys(self.fl))
        # Fixture is small (< 100 keys) → progress_every = 1 → one progress
        # event per tensor. Real 532-key models emit ~106 events.
        self.assertGreaterEqual(len(progress), total)
        self.assertTrue(progress[0]["text"].startswith(f"h3_delta 1/{total}"))
        self.assertTrue(progress[-1]["text"].startswith(f"h3_delta {total}/{total}"))
        # Progress ticks must not touch the topbar status line: no `status` field.
        self.assertFalse(any(e.get("status") for e in progress))
        # The done event is separate; completion log lines stay type "log".
        self.assertEqual(events[-1], {"type": "done", "status": "finished"})

    def test_delta_dry_run_streams_no_progress(self):
        self._write_delta_pair()
        events = _events(_delta_payload(str(self.fl), str(self.ref), str(self.dir),
                                        dry_run=True, rank=64))
        self.assertEqual(events[-1]["status"], "dry-run complete")
        self.assertNotIn("h3_delta", events[-1]["status"])

    def test_full_variant_detected(self):
        """time_embedder keys (no adaln_t_table) → variant 'full'."""
        fl, ref = _delta_pair_tensors()
        del fl["adaln_t_table"]
        del ref["adaln_t_table"]
        fl["time_embedder.proj_in.weight"] = torch.full((8, 8), 5.0)
        fl["time_embedder.proj_in.bias"] = torch.zeros(8)
        ref["time_embedder.proj_in.weight"] = torch.full((8, 8), 6.0)
        ref["time_embedder.proj_in.bias"] = torch.ones(8)
        save_file(fl, self.fl)
        save_file(ref, self.ref)
        out_name = "out_full.safetensors"
        _events(_delta_payload(str(self.fl), str(self.ref), str(self.dir), out_name=out_name))
        meta, _ = _read_meta(Path(self.dir, out_name))
        self.assertEqual(meta.get("h3_delta_variant"), "full")
        # time_embedder applied exactly
        self.assertEqual(_read_safetensors_tensor(Path(self.dir, out_name), "time_embedder.proj_in.weight").mean().item(), 6.0)


# ---------------------------------------------------------------------------
# Family classifier
# ---------------------------------------------------------------------------

class LARVMetricTests(unittest.TestCase):
    def test_rectangular_conflict_matches_symmetric_paper_definition(self):
        """The memory-safe conflict metric retains both paper Gram-commutator terms."""
        base = torch.tensor([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
        delta = torch.tensor([[2.0, -1.0], [0.5, 4.0], [-3.0, 1.0]])
        denom = torch.norm(base) * torch.norm(delta) + 1e-10
        left = torch.norm(base @ delta.t() - delta @ base.t()) / denom
        right = torch.norm(base.t() @ delta - delta.t() @ base) / denom
        expected = 0.5 * (left + right)
        self.assertAlmostEqual(_commutator_conflict(base, delta), float(expected), places=6)

    def test_h3_tiered_gate_never_extrapolates_past_ref2va(self):
        scales = _larv_tier_scales({22: 0.2, 23: 0.5, 24: 0.9})
        self.assertEqual(scales, {22: 0.5, 23: 0.75, 24: 1.0})

    def test_h3_continuous_gate_never_extrapolates_past_ref2va(self):
        scales, _ = _larv_block_scales(
            {22: (20.0, 0.01), 23: (50.0, 0.02)}, total_layers=50, gate="continuous",
        )
        self.assertTrue(all(0.0 <= scale <= 1.0 for scale in scales.values()))


class H3FamilyClassificationTests(unittest.TestCase):
    def test_pruned_block_families(self):
        self.assertEqual(_classify_h3_family("blocks.3.attn.qkv_proj.weight"), "attn")
        self.assertEqual(_classify_h3_family("blocks.3.mlp.fc1.weight"), "mlp")
        self.assertEqual(_classify_h3_family("blocks.3.mlp.fc2.weight"), "mlp")
        self.assertEqual(_classify_h3_family("blocks.3.adaln_proj.linear.weight"), "ada")
        self.assertEqual(_classify_h3_family("blocks.3.adaln_proj.linear.bias"), "ada")
        self.assertEqual(_classify_h3_family("blocks.3.norm1.weight"), "norm")
        self.assertEqual(_classify_h3_family("blocks.3.norm2.weight"), "norm")
        self.assertEqual(_classify_h3_family("blocks.3.attn.out_proj.weight"), "attn")
        # q/k_norm are 1-D RMS scale tensors → norm family (mirrors the
        # NVFP4 filter, which skips attn.[qk]_norm)
        self.assertEqual(_classify_h3_family("blocks.3.attn.q_norm.weight"), "norm")
        self.assertEqual(_classify_h3_family("blocks.3.attn.k_norm.weight"), "norm")

    def test_nonblock_families(self):
        self.assertEqual(_classify_h3_family("adaln_t_table"), "timestep_table")
        self.assertEqual(_classify_h3_family("time_embedder.proj_in.weight"), "time_embedder")
        self.assertEqual(_classify_h3_family("time_embedder.proj_out.bias"), "time_embedder")
        self.assertEqual(_classify_h3_family("rope.inv_freq"), "rope")
        self.assertEqual(_classify_h3_family("video_patch_proj.weight"), "proj")
        self.assertEqual(_classify_h3_family("audio_patch_proj.bias"), "bias")
        self.assertEqual(_classify_h3_family("condition_proj.weight"), "proj")
        self.assertEqual(_classify_h3_family("final_layer.norm.weight"), "norm")

    def test_svd_eligibility(self):
        self.assertTrue(_is_svd_eligible("blocks.3.attn.qkv_proj.weight", (4096, 4096)))
        self.assertTrue(_is_svd_eligible("blocks.3.mlp.fc1.weight", (8192, 4096)))
        self.assertTrue(_is_svd_eligible("blocks.3.adaln_proj.linear.weight", (512, 512)))
        self.assertFalse(_is_svd_eligible("blocks.3.adaln_proj.linear.bias", (512,)))
        self.assertFalse(_is_svd_eligible("blocks.3.norm1.weight", (4096,)))
        self.assertFalse(_is_svd_eligible("adaln_t_table", (1025, 8)))
        self.assertFalse(_is_svd_eligible("time_embedder.proj_in.weight", (2688, 2688)))
        self.assertFalse(_is_svd_eligible("rope.inv_freq", (16,)))
        # token_refiner 2-D weights are SVD-eligible (compressible trunk)
        self.assertTrue(_is_svd_eligible("token_refiner.blocks.0.attn.qkv_proj.weight", (512, 512)))


if __name__ == "__main__":
    unittest.main()
