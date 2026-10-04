import os
import unittest
import torch
from tempfile import TemporaryDirectory
from safetensors.torch import save_file


def full_tensors(dtype=torch.float32, prefix=''):
    torch.manual_seed(4)
    items = {'time_embedder.proj_in.weight': torch.randn(5, 4)*.5,
             'time_embedder.proj_in.bias': torch.randn(5)*.5,
             'time_embedder.proj_out.weight': torch.randn(3, 5)*.5,
             'time_embedder.proj_out.bias': torch.randn(3)*.5,
             'blocks.0.adaln_proj.linear.weight': torch.randn(6, 3),
             'blocks.0.adaln_proj.linear.bias': torch.randn(6),
             'final_layer.adaln_proj.linear.weight': torch.randn(2, 3),
             'final_layer.adaln_proj.linear.bias': torch.randn(2),
             'blocks.0.attn.qkv_proj.weight': torch.randn(4, 3)}
    return {prefix+k: v.to(dtype) for k,v in items.items()}


class VariantTests(unittest.TestCase):
    def test_full_pruned_and_mixed_are_distinguished(self):
        from core.h3_curve import inspect_h3_variant, time_curve
        with TemporaryDirectory() as d:
            path = os.path.join(d, 'm.safetensors')
            items = full_tensors(prefix='model.diffusion_model.')
            save_file(items, path)
            result = inspect_h3_variant(path)
            self.assertEqual(result['variant'], 'full')
            self.assertEqual(result['source_width'], 3)
            prefix = result['prefix']
            from core.h3_curve import TIME_KEYS
            table = time_curve([items[prefix+k] for k in TIME_KEYS], torch.linspace(0, 1, 65)).float()
            for k in TIME_KEYS: del items[prefix+k]
            items[prefix+'adaln_t_table'] = table
            save_file(items, path)
            self.assertEqual(inspect_h3_variant(path)['variant'], 'pruned')
            items[prefix+TIME_KEYS[0]] = torch.randn(5,4)
            save_file(items, path)
            with self.assertRaisesRegex(ValueError, 'mixed|ambiguous'): inspect_h3_variant(path)
