import os
import unittest
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import load_file, save_file

from core.h3_adapter_convert_engine import run_h3_adapter_convert
from core.lora_merge_engine import run_lora_merge
from tests import test_h3_adapter_convert_engine as conversion_tests
from utils.minimax_h3_layer_profiles import (
    is_h3_turbo_update_key,
    is_minimax_h3_preserved_key,
)


class H3CompletePipelineTests(unittest.TestCase):
    modules = ('blocks.0.adaln_proj.linear', 'final_layer.adaln_proj.linear')

    def adapter(self, target):
        tensors, deltas = {}, {}
        for module in self.modules:
            rows, width = target[module + '.weight'].shape
            down = torch.tensor([[.3, -.1, .5], [-.4, .7, .2]])[:, :width]
            up = torch.arange(1, rows * 2 + 1, dtype=torch.float32).reshape(rows, 2) / 10
            bias = torch.arange(1, rows + 1, dtype=torch.float32) / 7
            tensors.update({module + '.lora_A.weight': down,
                            module + '.lora_B.weight': up,
                            module + '.alpha': torch.tensor(3.),
                            module + '.diff_b': bias})
            deltas[module] = (1.5 * (up @ down), bias)
        return tensors, deltas

    def bake(self, base, adapter, output):
        events = list(run_lora_merge(dict(
            base_path=base, output_path=output,
            loras=[dict(path=adapter, strength=.8)], global_strength=.5,
            architecture='MiniMax H3', merge_algorithm='additive',
            h3_turbo_complete=True, adaptive=False, strict_matching=True,
            dry_run=False, merge_device='cpu', protect_token_refiner=False,
        )))
        self.assertEqual(events[-1]['status'], 'finished')
        text = ''.join(event.get('text', '') for event in events)
        self.assertIn('application: complete; omissions=0 protected_refiner=0', text)
        return load_file(output)

    def assert_baked(self, actual, base, deltas):
        self.assertEqual(set(actual), set(base))
        touched = set()
        for module, (weight_delta, bias_delta) in deltas.items():
            for suffix, delta in (('.weight', weight_delta), ('.bias', bias_delta)):
                key = module + suffix
                touched.add(key)
                self.assertFalse(torch.equal(actual[key], base[key]), key)
                torch.testing.assert_close(actual[key], base[key] + .4 * delta,
                                           atol=2e-5, rtol=2e-5)
        for key in set(base) - touched:
            self.assertTrue(torch.equal(actual[key], base[key]), key)

    def test_converted_pruned_complete_bake_applies_real_adaln_weights_and_bias_once(self):
        with TemporaryDirectory() as directory:
            base, ref, adapter, converted, target, curve = conversion_tests.AdapterConversionTests().fixture(directory)
            tensors, deltas = self.adapter(target)
            save_file(tensors, adapter)
            events = list(run_h3_adapter_convert(dict(
                base_path=base, pruned_target_path=ref, adapter_path=adapter,
                output_path=converted, merge_device='cpu',
            )))
            self.assertEqual(events[-1]['status'], 'finished')
            center = curve.mean(0)
            expected = {module: (weight, bias + weight @ center)
                        for module, (weight, bias) in deltas.items()}
            actual = self.bake(ref, converted, os.path.join(directory, 'pruned_baked.safetensors'))
            self.assert_baked(actual, target, expected)

    def test_full_complete_bake_applies_real_adaln_weights_and_bias_once(self):
        with TemporaryDirectory() as directory:
            base, _, adapter, _, _, _ = conversion_tests.AdapterConversionTests().fixture(directory)
            source = load_file(base)
            tensors, deltas = self.adapter(source)
            save_file(tensors, adapter)
            actual = self.bake(base, adapter, os.path.join(directory, 'full_baked.safetensors'))
            self.assert_baked(actual, source, deltas)

    def test_turbo_exception_keeps_norm_and_io_preservation_narrow(self):
        for key in ('blocks.0.norm.weight', 'blocks.0.adaln.weight',
                    'blocks.0.modulation.bias', 'blocks.0.rope.weight',
                    'video_patch_proj.weight', 'audio_patch_proj.weight', 'time_embedder.proj_in.weight',
                    'final_layer.linear.weight', 'vae.decoder.weight'):
            with self.subTest(key=key):
                self.assertTrue(is_minimax_h3_preserved_key(key), key)
                self.assertFalse(is_h3_turbo_update_key(key), key)
        for key in ('blocks.0.adaln_proj.linear.extra.weight',
                    'blocks.x.adaln_proj.linear.weight',
                    'final_layer.adaln_proj.norm.weight'):
            with self.subTest(key=key):
                self.assertFalse(is_h3_turbo_update_key(key), key)


if __name__ == '__main__':
    unittest.main()
