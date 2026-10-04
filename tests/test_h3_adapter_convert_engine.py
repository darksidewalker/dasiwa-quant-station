import os
import unittest
from tempfile import TemporaryDirectory
import torch
from safetensors.torch import save_file, load_file
from safetensors import safe_open
from tests.test_h3_variant_contract import full_tensors
from core.h3_curve import TIME_KEYS, time_curve, interpolate_table


class AdapterConversionTests(unittest.TestCase):
    def fixture(self, directory):
        source = full_tensors()
        curve = time_curve([source[k] for k in TIME_KEYS], torch.linspace(0, 1, 1025)).float()
        center = curve.mean(0)
        target = {k: v.clone() for k, v in source.items() if k not in TIME_KEYS}
        target['adaln_t_table'] = curve - center
        for key in list(target):
            if key.endswith('adaln_proj.linear.bias'):
                target[key] += source[key[:-4] + 'weight'] @ center
        base, ref, adapter, output = [os.path.join(directory, x + '.safetensors') for x in ('base', 'ref', 'adapter', 'output')]
        save_file(source, base); save_file(target, ref)
        return base, ref, adapter, output, target, curve

    def test_factors_and_bias_preserve_affine_update_and_scale(self):
        from core.h3_adapter_convert_engine import run_h3_adapter_convert
        with TemporaryDirectory() as d:
            base, ref, adapter, out, target, curve = self.fixture(d)
            module = 'blocks.0.adaln_proj.linear'
            A = torch.tensor([[.3, -.1, .5], [-.4, .7, .2]])
            B = torch.randn(6, 2)
            bias = torch.randn(6)
            save_file({module + '.lora_A.weight': A, module + '.lora_B.weight': B,
                       module + '.alpha': torch.tensor(3.), module + '.diff_b': bias}, adapter)
            events = list(run_h3_adapter_convert(dict(base_path=base, pruned_target_path=ref, adapter_path=adapter, output_path=out, cuda_device='cuda:0')))
            self.assertEqual(events[-1]['status'], 'finished')
            got = load_file(out)
            t = torch.tensor([.07123, .42217, .91231], dtype=torch.float64)
            x = time_curve([full_tensors()[k] for k in TIME_KEYS], t)
            q = interpolate_table(target['adaln_t_table'], t)
            for strength in (-.7, 0., .3, 1.):
                actual = strength * (1.5 * q @ got[module + '.lora_A.weight'].double().T @ got[module + '.lora_B.weight'].double().T + got[module + '.diff_b'].double())
                expected = strength * (1.5 * x @ A.double().T @ B.double().T + bias.double())
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            with safe_open(out, framework='pt') as h: self.assertIn('adaln_coordinate_table_sha256', h.metadata())

    def test_direct_patches_and_unknown_or_time_updates_fail_without_output(self):
        from core.h3_adapter_convert_engine import run_h3_adapter_convert
        with TemporaryDirectory() as d:
            base, ref, adapter, out, target, curve = self.fixture(d)
            module = 'final_layer.adaln_proj.linear'
            W = torch.randn(2, 3)
            save_file({module + '.diff': W}, adapter)
            list(run_h3_adapter_convert(dict(base_path=base, pruned_target_path=ref, adapter_path=adapter, output_path=out)))
            self.assertIn(module + '.diff_b', load_file(out))
            for bad in ('time_embedder.proj_in.diff', 'standalone.fc1.weight'):
                save_file({bad: torch.randn(2, 3)}, adapter)
                with self.assertRaises(ValueError):
                    list(run_h3_adapter_convert(dict(base_path=base, pruned_target_path=ref, adapter_path=adapter, output_path=out + '.new')))
                self.assertFalse(os.path.exists(out + '.new.safetensors'))

    def test_nonfinite_ordinary_up_factor_is_rejected_and_cleaned(self):
        from core.h3_adapter_convert_engine import run_h3_adapter_convert
        with TemporaryDirectory() as d:
            base, ref, adapter, out, _, _ = self.fixture(d)
            module = 'blocks.0.attn.qkv_proj'
            save_file({module + '.lora_A.weight': torch.ones(1, 3), module + '.lora_B.weight': torch.full((4, 1), float('nan'))}, adapter)
            with self.assertRaisesRegex(ValueError, 'finite|NaN'):
                list(run_h3_adapter_convert(dict(base_path=base, pruned_target_path=ref, adapter_path=adapter, output_path=out)))
            self.assertFalse(os.path.exists(out))
            self.assertFalse(any(name.startswith('.h3_stage_') for name in os.listdir(d)))

    def test_cuda_string_accepted_by_prune_preflight(self):
        from core.h3_prune_engine import require_h3
        require_h3({'cuda_device': 'cuda:1'})
