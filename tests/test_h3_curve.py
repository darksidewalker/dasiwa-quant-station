import hashlib
import unittest
import torch
from safetensors.torch import save_file
from tempfile import TemporaryDirectory


class CurveTests(unittest.TestCase):
    def test_curve_matches_time_embedder_formula(self):
        from core.h3_curve import time_curve, table_sha256
        torch.manual_seed(4)
        tensors = [torch.randn(5, 4), torch.randn(5), torch.randn(3, 5), torch.randn(3)]
        t = torch.tensor([0., .13, .53, 1.], dtype=torch.float64)
        freq = torch.tensor([1., .01], dtype=torch.float64)
        emb = torch.cat(((t[:, None]*freq).cos(), (t[:, None]*freq).sin()), 1)
        w1, b1, w2, b2 = [x.double() for x in tensors]
        expected = torch.nn.functional.silu(torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(emb, w1, b1)), w2, b2))
        torch.testing.assert_close(time_curve(tensors, t), expected)
        self.assertEqual(table_sha256(expected), hashlib.sha256(expected.float().numpy().tobytes()).hexdigest())

    def test_reference_gauge_recovery_and_projection_validation(self):
        from core.h3_curve import recover_gauge, TIME_KEYS
        from safetensors import safe_open
        from tests.test_h3_variant_contract import full_tensors
        import os
        items = full_tensors()
        from core.h3_curve import time_curve
        table = time_curve([items[k] for k in TIME_KEYS], torch.linspace(0,1,65)).float()
        target = {k:v.clone() for k,v in items.items() if k not in TIME_KEYS}
        target['adaln_t_table'] = table
        with TemporaryDirectory() as d:
            a,b = os.path.join(d,'a.safetensors'), os.path.join(d,'b.safetensors')
            save_file(items,a); save_file(target,b)
            gauge = recover_gauge(a,b)
            torch.testing.assert_close(gauge['basis'], torch.eye(3,dtype=torch.float64), atol=1e-4,rtol=1e-4)
            self.assertLess(gauge['interpolation_relative_error'], 1e-4)
            target['final_layer.adaln_proj.linear.weight'] *= -1
            save_file(target,b)
            with self.assertRaisesRegex(ValueError, 'consistency'): recover_gauge(a,b)

if __name__ == '__main__': unittest.main()
