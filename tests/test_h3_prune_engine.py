import os
import unittest
import torch
from tempfile import TemporaryDirectory
from safetensors.torch import save_file, load_file
from safetensors import safe_open
from tests.test_h3_variant_contract import full_tensors
from core.h3_curve import TIME_KEYS, time_curve


def reference(items, prefix=''):
    table = time_curve([items[prefix+k] for k in TIME_KEYS], torch.linspace(0,1,65)).float()
    result = {k:v.clone() for k,v in items.items() if k not in {prefix+t for t in TIME_KEYS}}
    result[prefix+'adaln_t_table'] = table
    return result


class PruneTests(unittest.TestCase):
    def test_independent_fold_has_explicit_reproducibility_record(self):
        from core.h3_prune_engine import run_h3_prune
        from core.h3_curve import interpolate_table
        import json
        with TemporaryDirectory() as d:
            a,out=[os.path.join(d,n+'.safetensors') for n in ('base','independent')]
            items=full_tensors(); save_file(items,a)
            list(run_h3_prune(dict(base_path=a,output_path=out,fold_mode='independent',merge_device='cpu')))
            got=load_file(out)
            t=torch.tensor([.123,.765],dtype=torch.float64)
            expected=time_curve([items[k] for k in TIME_KEYS],t)@items['final_layer.adaln_proj.linear.weight'].double().T+items['final_layer.adaln_proj.linear.bias'].double()
            actual=interpolate_table(got['adaln_t_table'],t)@got['final_layer.adaln_proj.linear.weight'].double().T+got['final_layer.adaln_proj.linear.bias'].double()
            torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
            with open(out+'.txt') as f: text=f.read()
            self.assertIn('basis_values',text)
            self.assertIn('ecosystem_compatible',text)

    def test_reference_fold_preserves_table_dtype_and_unrelated_bytes(self):
        from core.h3_prune_engine import run_h3_prune
        with TemporaryDirectory() as d:
            a,b,out = [os.path.join(d,n+'.safetensors') for n in ('base','ref','out')]
            items = full_tensors(torch.bfloat16)
            target = reference(items)
            save_file(items,a); save_file(target,b)
            events = list(run_h3_prune(dict(base_path=a, reference_path=b, output_path=out, merge_device='cpu', row_chunk_size=2)))
            self.assertEqual(events[-1]['status'],'finished')
            got = load_file(out)
            self.assertEqual(set(got),set(target))
            self.assertTrue(torch.equal(got['adaln_t_table'],target['adaln_t_table']))
            self.assertTrue(torch.equal(got['blocks.0.attn.qkv_proj.weight'],items['blocks.0.attn.qkv_proj.weight']))
            self.assertEqual(got['final_layer.adaln_proj.linear.weight'].dtype,torch.bfloat16)
            self.assertTrue(os.path.isfile(out+'.txt'))
            with safe_open(out,framework='pt') as h:
                self.assertIn('adaln_coordinate_table_sha256',h.metadata())
            with self.assertRaises(FileExistsError): list(run_h3_prune(dict(base_path=a,reference_path=b,output_path=out)))
