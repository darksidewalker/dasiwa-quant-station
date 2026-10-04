"""H3 command policy tests; real headers, mocked conversion only."""
import os
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from core.safetensors_engine import run_safe_conversion
from tests.test_safetensors_engine_commands import _FakeProcess

FMT = "INT8 Row-wise ConvRot Runtime"


def h3_fixture(path, pruned=False, prefix=""):
    torch.manual_seed(43)
    tensors = {
        "blocks.0.attn.qkv_proj.weight": torch.randn(256, 256).to(torch.bfloat16),
        "blocks.0.attn.out_proj.weight": torch.randn(256, 256).to(torch.bfloat16),
        "blocks.0.mlp.fc1.weight": torch.randn(256, 256).to(torch.bfloat16),
        "blocks.0.mlp.fc2.weight": torch.randn(256, 256).to(torch.bfloat16),
        "blocks.0.adaln_proj.linear.weight": torch.randn(256, 8 if pruned else 256).to(torch.bfloat16),
        "blocks.0.adaln_proj.linear.bias": torch.randn(256).to(torch.bfloat16),
        "final_layer.adaln_proj.linear.weight": torch.randn(256, 8 if pruned else 256).to(torch.bfloat16),
        "final_layer.adaln_proj.linear.bias": torch.randn(256).to(torch.bfloat16),
        "video_patch_proj.weight": torch.randn(256, 256).to(torch.bfloat16),
        "token_refiner.blocks.0.attn.qkv_proj.weight": torch.randn(256, 256).to(torch.bfloat16),
    }
    if pruned:
        tensors["adaln_t_table"] = torch.randn(32, 8)
    else:
        for name in ("proj_in", "proj_out"):
            tensors[f"time_embedder.{name}.weight"] = torch.randn(256, 256).to(torch.bfloat16)
            tensors[f"time_embedder.{name}.bias"] = torch.randn(256).to(torch.bfloat16)
    save_file({prefix + k: v for k, v in tensors.items()}, str(path))
    return {prefix + k: v for k, v in tensors.items()}


class H3PolicyTests(unittest.TestCase):
    def capture(self, pruned=False, prefix="", arch="MiniMax H3", fmt=FMT,
                strategy="Simple", filename="misleading_pruned.safetensors", **kwargs):
        commands = []
        def popen(cmd, *args, **kw):
            commands.append(cmd)
            return _FakeProcess(cmd)
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, filename)
            h3_fixture(source, pruned, prefix)
            with mock.patch("core.safetensors_engine.subprocess.Popen", side_effect=popen), \
                 mock.patch("core.safetensors_engine.verify_architecture_match", return_value=(True, "ok")), \
                 mock.patch("core.safetensors_engine.FILTERS_DIR", tmp), \
                 mock.patch("core.safetensors_engine.inject_metadata", return_value=(True, "ok")), \
                 mock.patch("core.safetensors_engine.write_quant_recipe") as recipe, \
                 mock.patch("core.safetensors_engine.save_log"):
                events = list(run_safe_conversion(tmp, source, [fmt], "test", arch,
                    "prodigy", strategy, "", **kwargs))
                recipes = recipe.call_args_list
        return commands, events, recipes

    def test_h3_aborts_if_installed_converter_lacks_required_preset(self):
        with mock.patch("core.safetensors_engine.h3_ctq_capability", return_value=(False, "missing --minimaxh3")):
            cmds, events, _ = self.capture()
        self.assertEqual(cmds, [])
        self.assertIn("Aborted: unsupported converter", events[-1][1])

    def test_policy_and_verbose_validation_happens_before_output_or_launch(self):
        cases = [dict(h3_quant_policy="bogus"), dict(verbose_level="LOUD"),
                 dict(h3_quant_policy="upstream_int8_convrot", arch="WAN 2.2"),
                 dict(h3_quant_policy="upstream_int8_convrot", fmt="NVFP4"),
                 dict(h3_quant_policy="upstream_int8_convrot", strategy="Optimizer-driven")]
        for args in cases:
            with self.subTest(args=args):
                cmds, events, recipes = self.capture(**args)
                self.assertEqual(cmds, [])
                self.assertEqual(recipes, [])
                self.assertIn("Aborted", events[-1][1])

    def test_pruned_exclusion_is_header_based_bare_and_prefixed_for_both_policies(self):
        for prefix in ("", "model.diffusion_model."):
            for policy in ("preserve_structural", "upstream_int8_convrot"):
                with self.subTest(prefix=prefix, policy=policy):
                    cmds, _, _ = self.capture(pruned=True, prefix=prefix,
                        filename="full.safetensors", h3_quant_policy=policy)
                    self.assertEqual(len(cmds), 1)
                    cmd = cmds[0]
                    self.assertEqual(cmd.count("--exclude_layers"), 1)
                    self.assertEqual(cmd[cmd.index("--exclude_layers") + 1], "(adaln_t_table|adaln_proj)")
                    self.assertEqual("--layer-config" in cmd, policy == "preserve_structural")

    def test_structural_full_preserves_time_embedder_dtype_after_upstream_cast(self):
        import re
        cmds, _, _ = self.capture(h3_quant_policy='preserve_structural')
        self.assertIn('--preserve-layers', cmds[0])
        pattern = cmds[0][cmds[0].index('--preserve-layers') + 1]
        self.assertIsNotNone(re.search(pattern, 'time_embedder.proj_in.weight'))

    def test_opt_in_full_matches_reference_without_any_layer_override(self):
        cmds, events, recipes = self.capture(h3_quant_policy="upstream_int8_convrot")
        self.assertEqual(len(cmds), 1)
        cmd = cmds[0]
        for flag in ("--comfy_quant", "--save-quant-metadata", "--verbose", "--low-memory",
                     "--minimaxh3", "--int8", "--scaling_mode", "--convrot",
                     "--convrot-group-size", "--simple"):
            self.assertEqual(cmd.count(flag), 1, flag)
        self.assertEqual(cmd[cmd.index("--verbose") + 1], "VERBOSE")
        self.assertEqual(cmd[cmd.index("--scaling_mode") + 1], "row")
        self.assertEqual(cmd[cmd.index("--convrot-group-size") + 1], "256")
        self.assertNotIn("--layer-config", cmd)
        self.assertNotIn("--exclude_layers", cmd)
        self.assertTrue(recipes[0].args[7], "recipe must record effective low-memory")
        self.assertIn("upstream_int8_convrot", events[-1][0])


if __name__ == "__main__":
    unittest.main()
