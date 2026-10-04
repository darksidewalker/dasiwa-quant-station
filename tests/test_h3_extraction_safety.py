import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch
import torch
from safetensors.torch import save_file, load_file
from core.lora_extract_engine import run_lora_extract
from tests.test_h3_adapter_convert_engine import AdapterConversionTests


class ExtractionSafetyTests(unittest.TestCase):
    def test_bias_only_final_adaln_is_exported(self):
        with TemporaryDirectory() as d:
            base, ref, _, out, _, _ = AdapterConversionTests().fixture(d)
            changed = load_file(base)
            changed['final_layer.adaln_proj.linear.bias'] += .2
            modified = os.path.join(d, 'modified.safetensors'); save_file(changed, modified)
            with patch('core.lora_extract_engine.verify_architecture_match', return_value=(True, 'ok')):
                list(run_lora_extract(dict(base_path=base, merged_path=modified, pruned_target_path=ref, output_path=out, architecture='MiniMax H3', recipe='h3_pruned')))
            got = load_file(out)
            torch.testing.assert_close(got['diffusion_model.final_layer.adaln_proj.linear.diff_b'], torch.full((2,), .2), atol=1e-6, rtol=1e-6)

    def test_changed_time_curve_is_rejected_before_artifact(self):
        with TemporaryDirectory() as d:
            base, ref, _, out, _, _ = AdapterConversionTests().fixture(d)
            changed = load_file(base)
            changed['time_embedder.proj_out.bias'] += .2
            changed['blocks.0.attn.qkv_proj.weight'] += .2
            modified = os.path.join(d, 'modified.safetensors'); save_file(changed, modified)
            with patch('core.lora_extract_engine.verify_architecture_match', return_value=(True, 'ok')):
                with self.assertRaisesRegex(ValueError, 'time|Time'):
                    list(run_lora_extract(dict(base_path=base, merged_path=modified, pruned_target_path=ref, output_path=out, architecture='MiniMax H3', recipe='h3_pruned')))
            self.assertFalse(os.path.exists(out))

    def test_pruned_adaln_weights_are_read_in_rows_not_materialized(self):
        from safetensors import safe_open as real_open
        with TemporaryDirectory() as d:
            base, ref, _, out, _, _ = AdapterConversionTests().fixture(d)
            modified = os.path.join(d, 'modified.safetensors')
            changed = load_file(base); changed['blocks.0.adaln_proj.linear.weight'] += .1
            save_file(changed, modified)
            class RowOnly:
                def __init__(self, *a, **kw): self.handle = real_open(*a, **kw)
                def __enter__(self): self.handle.__enter__(); return self
                def __exit__(self, *a): return self.handle.__exit__(*a)
                def __getattr__(self, k): return getattr(self.handle, k)
                def get_tensor(self, k):
                    if k.endswith('adaln_proj.linear.weight'): raise AssertionError('Dense AdaLN weight materialized')
                    return self.handle.get_tensor(k)
            with patch('core.lora_extract_engine.verify_architecture_match', return_value=(True, 'ok')), patch('core.lora_extract_engine.safe_open', RowOnly):
                list(run_lora_extract(dict(base_path=base, merged_path=modified, pruned_target_path=ref, output_path=out, architecture='MiniMax H3', recipe='h3_pruned')))
            self.assertIn('diffusion_model.blocks.0.adaln_proj.linear.diff', load_file(out))

    def test_full_bias_only_changes_are_exported(self):
        with TemporaryDirectory() as d:
            base, modified, out = [os.path.join(d, name + '.safetensors') for name in ('base', 'modified', 'out')]
            save_file({'head.weight': torch.zeros(2, 2), 'head.bias': torch.zeros(2)}, base)
            save_file({'head.weight': torch.zeros(2, 2), 'head.bias': torch.full((2,), .4)}, modified)
            with patch('core.lora_extract_engine.verify_architecture_match', return_value=(True, 'ok')):
                list(run_lora_extract(dict(base_path=base, merged_path=modified, output_path=out, architecture='WAN 2.2', recipe='generic')))
            got = load_file(out)
            torch.testing.assert_close(got['diffusion_model.head.diff_b'], torch.full((2,), .4))
            with open(out + '.txt') as f: recipe = f.read()
            self.assertIn('Output mode: full', recipe)
            self.assertIn('Output: ' + out, recipe)
            self.assertIn('Bias direct patches: 1', recipe)

    def test_existing_recipe_is_never_overwritten(self):
        with TemporaryDirectory() as d:
            base, _, _, out, _, _ = AdapterConversionTests().fixture(d)
            modified = os.path.join(d, 'modified.safetensors')
            changed = load_file(base); changed['blocks.0.attn.qkv_proj.weight'] += .2
            save_file(changed, modified)
            with open(out + '.txt', 'w') as f: f.write('existing recipe')
            with self.assertRaises(FileExistsError):
                list(run_lora_extract(dict(base_path=base, merged_path=modified, output_path=out, architecture='MiniMax H3', recipe='generic')))
            self.assertFalse(os.path.exists(out))

    def test_rank_bounds_are_validated(self):
        with TemporaryDirectory() as d:
            base, ref, _, out, _, _ = AdapterConversionTests().fixture(d)
            with self.assertRaisesRegex(ValueError, 'rank'):
                list(run_lora_extract(dict(base_path=base, merged_path=base, output_path=out, architecture='MiniMax H3', recipe='generic', min_rank=8, max_rank=2)))
