"""MiniMax H3 NVFP4 quant-profile detection + mixed-profile recognition.

Builds a synthetic 50-block H3 file (tiny tensors, a few KB total) so these
tests run in milliseconds without the real 12 GB community references:

  - pure NVFP4        : all 200 main-matrix heavy linears U8-packed
  - NVFP4 HQ mixed    : exactly the 30-layer HQ plan (27 out_proj + 3 fc2,
                        single source of truth in layer_config_builder) stays
                        BF16, the other 170 are U8-packed
  - FP8 adaln tier    : all 200 U8 + per-block adaln_proj.linear F8_E4M3
  - unknown mixed     : a kept layer outside any known plan
  - unquantized BF16  : detection returns None (no profile to report)

The end-to-end tests run audit_patterns() on those files and prove the
"no false flagging" requirement: recognized mixed profiles report their
retained heavy linears as MIXED-KEPT variants, never as SUSPICIOUS.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file

from core.layer_config_builder import H3_NVFP4_HQ_LAYER_PLAN
from utils.pattern_audit import audit_patterns, detect_h3_quant_profile, _read_layers

H3_BLOCKS = 50
MAIN_MATRIX_KINDS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


def _tensor(dtype):
    if dtype == "U8":
        return torch.zeros(2, 2, dtype=torch.uint8)
    if dtype == "F8_E4M3":
        return torch.zeros(2, 2, dtype=torch.float8_e4m3fn)
    return torch.zeros(2, 2, dtype=torch.bfloat16)


def _build_h3_file(path, main_matrix_dtypes, adaln_dtype="BF16",
                   extra_kept=None):
    """Write a synthetic H3 safetensors file.

    main_matrix_dtypes: dict (block, kind) -> dtype for all 200 heavy
    linears. extra_kept: list of (key, dtype) for out-of-plan retentions.
    Structural layers + arch markers always BF16 unless overridden.
    """
    tensors = {"adaln_t_table": _tensor("BF16")}  # pruned-variant marker
    for b in range(H3_BLOCKS):
        tensors[f"blocks.{b}.attn.q_norm.weight"] = _tensor("BF16")  # marker
        tensors[f"blocks.{b}.attn.k_norm.weight"] = _tensor("BF16")
        tensors[f"blocks.{b}.norm1.weight"] = _tensor("BF16")
        tensors[f"blocks.{b}.norm2.weight"] = _tensor("BF16")
        tensors[f"blocks.{b}.adaln_proj.linear.weight"] = _tensor(adaln_dtype)
        for kind in MAIN_MATRIX_KINDS:
            dt = main_matrix_dtypes.get((b, kind), "BF16")
            tensors[f"blocks.{b}.{kind}.weight"] = _tensor(dt)
    for key, dt in (extra_kept or []):
        tensors[key] = _tensor(dt)
    save_file(tensors, path)


def _all_dtypes(dt):
    return {(b, k): dt for b in range(H3_BLOCKS) for k in MAIN_MATRIX_KINDS}


def _hq_mixed_dtypes():
    plan = {(b, k) for b, k in H3_NVFP4_HQ_LAYER_PLAN}
    return {
        (b, k): ("BF16" if (b, k) in plan else "U8")
        for b in range(H3_BLOCKS)
        for k in MAIN_MATRIX_KINDS
    }


class H3NVFP4ProfileDetectionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "h3_test.safetensors")

    def tearDown(self):
        self._tmp.cleanup()

    def _detect(self):
        return detect_h3_quant_profile(_read_layers(self.path))

    def test_pure_nvfp4_profile(self):
        _build_h3_file(self.path, _all_dtypes("U8"))
        p = self._detect()
        self.assertEqual(p["profile"], "nvfp4_pure")
        self.assertEqual(p["packed"], 200)
        self.assertEqual(len(p["kept"]), 0)
        self.assertFalse(p["fp8_adaln"])

    def test_hq_mixed_profile_matches_plan_exactly(self):
        _build_h3_file(self.path, _hq_mixed_dtypes())
        p = self._detect()
        self.assertEqual(p["profile"], "nvfp4_hq_mixed")
        self.assertEqual(p["packed"], 170)
        self.assertEqual(len(p["kept"]), 30)
        kept_set = {(b, k) for b, k, _d, _key in p["kept"]}
        self.assertEqual(kept_set, {(b, k) for b, k in H3_NVFP4_HQ_LAYER_PLAN})

    def test_fp8_adaln_tier_profile(self):
        _build_h3_file(self.path, _all_dtypes("U8"), adaln_dtype="F8_E4M3")
        p = self._detect()
        self.assertEqual(p["profile"], "nvfp4_fp8_adaln_mixed")
        self.assertEqual(p["packed"], 200)
        self.assertTrue(p["fp8_adaln"])

    def test_unknown_mixed_profile_reports_soft_note(self):
        dtypes = _all_dtypes("U8")
        dtypes[(5, "attn.qkv_proj")] = "BF16"  # outside any known plan
        _build_h3_file(self.path, dtypes)
        p = self._detect()
        self.assertEqual(p["profile"], "nvfp4_mixed_unknown")
        self.assertEqual(p["packed"], 199)
        self.assertEqual(len(p["kept"]), 1)

    def test_unquantized_source_has_no_profile(self):
        _build_h3_file(self.path, _all_dtypes("BF16"))
        self.assertIsNone(self._detect())


class H3NVFP4NoFalseFlaggingTests(unittest.TestCase):
    """End-to-end: recognized mixed profiles never flag their retained
    heavy linears as suspicious pattern misses."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "h3_test.safetensors")

    def tearDown(self):
        self._tmp.cleanup()

    def test_hq_mixed_file_audits_as_recognized_variant(self):
        _build_h3_file(self.path, _hq_mixed_dtypes())
        report = audit_patterns(self.path, "MiniMax H3")
        self.assertIn("NVFP4 HQ mixed", report)
        self.assertIn("MIXED-KEPT", report)
        self.assertIn("attn.out_proj (blocks", report)
        self.assertIn("mlp.fc2 (blocks", report)
        self.assertIn("VERDICT: Patterns fully cover", report)
        # The 30 intentionally-kept layers must NOT be flagged anywhere.
        self.assertNotIn("SUSPICIOUS", report)

    def test_unknown_mixed_file_audits_clean_but_notes_unrecognized_plan(self):
        dtypes = _all_dtypes("U8")
        dtypes[(5, "attn.qkv_proj")] = "BF16"
        _build_h3_file(self.path, dtypes)
        report = audit_patterns(self.path, "MiniMax H3")
        self.assertIn("NVFP4 mixed (unrecognized plan)", report)
        self.assertIn("MIXED-KEPT", report)
        self.assertIn("VERDICT: Patterns fully cover", report)
        self.assertNotIn("SUSPICIOUS", report)

    def test_unquantized_h3_source_does_not_flag_fc2(self):
        # Regression: the generic LLaMA-style `\.fc2$` suspicious keyword
        # used to flag H3's 50 mlp.fc2 layers on every H3 file.
        _build_h3_file(self.path, _all_dtypes("BF16"))
        report = audit_patterns(self.path, "MiniMax H3")
        self.assertIn("VERDICT: Patterns fully cover", report)
        self.assertNotIn("SUSPICIOUS", report)


if __name__ == "__main__":
    unittest.main()
