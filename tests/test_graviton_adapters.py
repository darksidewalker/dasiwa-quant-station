"""CPU artifact contracts for GRAVITON through the existing adapter workflows."""
import json
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from core.lora_merge_engine import run_lora_merge
from core.lora_compose_engine import run_lora_compose
from core.model_merge_engine import _graviton_merge_matrix
from core.metadata_configs import MODEL_METADATA_CONFIGS
from unittest.mock import patch


def test_graviton_mixed_factors_alpha_lokr_bias_order_and_zero():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        key = 'diffusion_model.blocks.0.attn.qkv_proj'
        save_file({key + '.lora_A.weight': torch.tensor([[1., 2.]]),
                   key + '.lora_B.weight': torch.tensor([[1.], [3.]]),
                   key + '.alpha': torch.tensor(2.)}, str(root / 'a.safetensors'))
        save_file({key + '.lokr_w1': torch.tensor([[.5]]), key + '.lokr_w2': torch.eye(2),
                   key + '.alpha': torch.tensor(float('inf')),
                   key + '.diff_b': torch.tensor([.25, -.5])}, str(root / 'b.safetensors'))
        save_file({key + '.diff': torch.ones(2, 2) * 1000}, str(root / 'zero.safetensors'))
        specs = [dict(path=str(root / 'a.safetensors'), strength=-.5),
                 dict(path=str(root / 'b.safetensors'), strength=2.),
                 dict(path=str(root / 'zero.safetensors'), strength=0.)]
        payload = dict(loras=specs, architecture='MiniMax H3', merge_algorithm='graviton',
                       merge_device='cpu', output_adapter='lora', output_rank=2, frobenius_energy=1.)
        for name, values in [('first', specs), ('reverse', list(reversed(specs)))]:
            list(run_lora_compose(dict(payload, loras=values, output_path=str(root / (name + '.safetensors')))))
        first, reverse = [load_file(str(root / (name + '.safetensors'))) for name in ('first', 'reverse')]
        for name in first:
            torch.testing.assert_close(first[name], reverse[name], rtol=0, atol=0)
        torch.testing.assert_close(first[key + '.diff_b'], torch.tensor([.5, -1.]))
        raw = -torch.tensor([[1., 2.], [3., 6.]]) + torch.eye(2)
        expected = _graviton_merge_matrix(torch.zeros_like(raw), raw, 1., -1, polar=False)[0]
        rebuilt = first[key + '.lora_B.weight'].float() @ first[key + '.lora_A.weight'].float()
        torch.testing.assert_close(rebuilt, expected, atol=.04, rtol=.01)
        dry = root / 'dry.safetensors'
        list(run_lora_compose(dict(payload, output_path=str(dry), dry_run=True)))
        assert not dry.exists() and not Path(str(dry) + '.txt').exists()


def test_graviton_bake_nonmatrix_preservation_and_oom_retry():
    from core.lora_merge_engine import _merge_target_with_policy
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = {'blocks.0.attn.qkv_proj.bias': torch.tensor([1., 2.]),
                'blocks.0.adaln_proj.linear.weight': torch.eye(2),
                'vae.weight': torch.eye(2)}
        save_file(base, str(root / 'base.safetensors'))
        save_file({key.removesuffix('.weight') + ('.diff_b' if key.endswith('.bias') else '.diff'): torch.ones_like(value)
                   for key, value in base.items() if not key.endswith('.bias')}, str(root / 'preserve.safetensors'))
        save_file({'blocks.0.attn.qkv_proj.diff_b': torch.tensor([.5, -.5])}, str(root / 'bias.safetensors'))
        out = root / 'out.safetensors'
        list(run_lora_merge(dict(base_path=str(root / 'base.safetensors'),
            loras=[dict(path=str(root / 'preserve.safetensors')), dict(path=str(root / 'bias.safetensors'))],
            architecture='MiniMax H3', merge_algorithm='graviton', merge_device='cpu', output_path=str(out))))
        result = load_file(str(out))
        torch.testing.assert_close(result['blocks.0.attn.qkv_proj.bias'], torch.tensor([1.5, 1.5]))
        torch.testing.assert_close(result['blocks.0.adaln_proj.linear.weight'], torch.eye(2))
        torch.testing.assert_close(result['vae.weight'], torch.eye(2))
    with patch('core.lora_merge_engine._cuda_available', return_value=True), \
         patch('core.lora_merge_engine._has_cuda_headroom', return_value=True), \
         patch('core.graviton_adapter.bake_target', side_effect=[torch.cuda.OutOfMemoryError(), torch.eye(2)]) as merge:
        tensor, used, reason = _merge_target_with_policy(torch.eye(2), [], {}, False, 'auto', 'cuda:0', 1024, 'graviton')
        assert used == 'cpu' and reason == 'cuda_oom'
        assert merge.call_args_list[0].args[-1] == 'cuda:0'
        assert merge.call_args_list[1].args[-1] == 'cpu'


@pytest.mark.parametrize('workflow', ['bake', 'compose'])
def test_graviton_nonfinite_failure_does_not_replace_output(workflow):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        key = 'blocks.0.attn.qkv_proj'
        save_file({key + '.weight': torch.eye(2)}, str(root / 'base.safetensors'))
        save_file({key + '.diff': torch.full((2, 2), float('nan'))}, str(root / 'a.safetensors'))
        save_file({key + '.diff': torch.eye(2)}, str(root / 'b.safetensors'))
        out = root / 'out.safetensors'
        out.write_bytes(b'previous artifact')
        payload = dict(base_path=str(root / 'base.safetensors'),
            loras=[dict(path=str(root / 'a.safetensors')), dict(path=str(root / 'b.safetensors'))],
            architecture='MiniMax H3', merge_algorithm='graviton', merge_device='cpu', output_path=str(out))
        run = run_lora_merge if workflow == 'bake' else run_lora_compose
        with pytest.raises(FileExistsError):
            list(run(payload))
        assert out.read_bytes() == b'previous artifact'
        out.unlink()
        with pytest.raises(ValueError, match='finite'):
            list(run(payload))
        assert sorted(p.name for p in root.iterdir()) == ['a.safetensors', 'b.safetensors', 'base.safetensors']


@pytest.mark.parametrize('workflow', ['bake', 'compose'])
def test_graviton_dry_run_rejects_incompatible_factor_ranks(workflow):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        key = 'blocks.0.attn.qkv_proj'
        save_file({key + '.weight': torch.eye(2)}, str(root / 'base.safetensors'))
        for name in ['a', 'b']:
            save_file({key + '.lora_A.weight': torch.ones(1, 2),
                       key + '.lora_B.weight': torch.ones(2, 2)}, str(root / (name + '.safetensors')))
        with pytest.raises(ValueError, match='factor shapes'):
            list((run_lora_merge if workflow == 'bake' else run_lora_compose)(dict(
                base_path=str(root / 'base.safetensors'), loras=[dict(path=str(root / (name + '.safetensors'))) for name in ['a', 'b']],
                architecture='MiniMax H3', merge_algorithm='graviton', dry_run=True, merge_device='cpu')))


def test_graviton_direct_lokr_output_retains_anchor_contract():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        key = 'diffusion_model.blocks.0.attn.qkv_proj'
        for name in ['a', 'b']:
            save_file({key + '.lokr_w1': torch.ones(1, 1), key + '.lokr_w2': torch.eye(2)}, str(root / (name + '.safetensors')))
        list(run_lora_compose(dict(loras=[dict(path=str(root / (name + '.safetensors'))) for name in ['a','b']],
            architecture='MiniMax H3', merge_algorithm='graviton', merge_device='cpu', output_adapter='lokr', output_path=str(root / 'out.safetensors'))))
        tensors = load_file(str(root / 'out.safetensors'))
        rebuilt = torch.kron(tensors[key + '.lokr_w1'].float(), tensors[key + '.lokr_w2'].float())
        torch.testing.assert_close(rebuilt, torch.eye(2)*2, atol=.02, rtol=.01)


def test_graviton_zero_strength_lokr_does_not_change_auto_representation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        key = 'diffusion_model.blocks.0.attn.qkv_proj'
        for name in ['a', 'b']:
            save_file({key + '.lora_A.weight': torch.eye(2), key + '.lora_B.weight': torch.eye(2)}, str(root / (name + '.safetensors')))
        save_file({key + '.lokr_w1': torch.ones(1, 1), key + '.lokr_w2': torch.eye(2)}, str(root / 'zero.safetensors'))
        list(run_lora_compose(dict(loras=[dict(path=str(root / (name + '.safetensors')), strength=0. if name == 'zero' else 1.) for name in ['a','b','zero']],
            architecture='MiniMax H3', merge_algorithm='graviton', merge_device='cpu', output_adapter='auto', output_path=str(root / 'out.safetensors'))))
        assert key + '.lora_A.weight' in load_file(str(root / 'out.safetensors'))


def test_graviton_zero_strength_is_byte_exact_for_float64():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        key = 'blocks.0.attn.qkv_proj'
        base = torch.tensor([[1.123456789, 2.987654321], [3., 4.]], dtype=torch.float64)
        save_file({key + '.weight': base}, str(root / 'base.safetensors'))
        save_file({key + '.diff': torch.ones(2, 2)}, str(root / 'a.safetensors'))
        list(run_lora_merge(dict(base_path=str(root / 'base.safetensors'),
            loras=[dict(path=str(root / 'a.safetensors'), strength=0.)], architecture='MiniMax H3',
            merge_algorithm='graviton', merge_device='cpu', output_path=str(root / 'out.safetensors'))))
        assert torch.equal(load_file(str(root / 'out.safetensors'))[key + '.weight'], base)


def test_graviton_conditioning_uses_linear_spectral_and_conv_diff_stays_additive():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = {'time_projection.weight': torch.tensor([[1., 2.], [3., 5.]]),
                'conv.weight': torch.ones(2, 2, 1, 1)}
        save_file(base, str(root / 'base.safetensors'))
        save_file({key.removesuffix('.weight') + '.diff': torch.ones_like(value) * -.2
                   for key, value in base.items()}, str(root / 'a.safetensors'))
        list(run_lora_merge(dict(base_path=str(root / 'base.safetensors'),
            loras=[dict(path=str(root / 'a.safetensors'))], architecture='Flux.2',
            merge_algorithm='graviton', merge_device='cpu', output_path=str(root / 'out.safetensors'))))
        result = load_file(str(root / 'out.safetensors'))
        expected = _graviton_merge_matrix(base['time_projection.weight'], base['time_projection.weight'] - .2, 1., -1, polar=False)[0]
        torch.testing.assert_close(result['time_projection.weight'], expected)
        torch.testing.assert_close(result['conv.weight'], base['conv.weight'] - .2)


def test_graviton_bake_uses_combined_logical_delta():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = torch.tensor([[1., 2.], [3., 4.]])
        delta = torch.tensor([[.2, -.3], [.4, .1]])
        save_file({'blocks.0.attn.qkv_proj.weight': base}, str(root / 'base.safetensors'))
        save_file({'diffusion_model.blocks.0.attn.qkv_proj.diff': delta}, str(root / 'a.safetensors'))
        out = root / 'out.safetensors'
        events = list(run_lora_merge(dict(base_path=str(root / 'base.safetensors'),
            loras=[dict(path=str(root / 'a.safetensors'), strength=-.5)],
            architecture='MiniMax H3', merge_algorithm='graviton', merge_device='cpu', output_path=str(out))))
        expected = _graviton_merge_matrix(base, base - .5 * delta, 1., -1)[0]
        torch.testing.assert_close(load_file(str(out))['blocks.0.attn.qkv_proj.weight'], expected)
        assert events[-1]['status'] == 'finished'
        assert 'Merge algorithm:   graviton' in Path(str(out) + '.txt').read_text()


@pytest.mark.parametrize('architecture', MODEL_METADATA_CONFIGS)
def test_graviton_all_architectures_bake(architecture):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        key = 'blocks.0.attn.qkv_proj.weight'
        save_file({key: torch.eye(2), 'untouched': torch.tensor([7.])}, str(root / 'base.safetensors'))
        save_file({'diffusion_model.blocks.0.attn.qkv_proj.diff': torch.ones(2, 2) * .1}, str(root / 'a.safetensors'))
        payload = dict(base_path=str(root / 'base.safetensors'), loras=[dict(path=str(root / 'a.safetensors'))],
                       architecture=architecture, merge_algorithm='graviton', merge_device='cpu', output_path=str(root / 'out.safetensors'))
        assert list(run_lora_merge(payload))[-1]['status'] == 'finished'
        torch.testing.assert_close(load_file(str(root / 'out.safetensors'))['untouched'], torch.tensor([7.]))


def test_graviton_mismatch_cannot_silently_drop_a_layer_even_with_skip():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        save_file({'blocks.0.attn.qkv_proj.weight': torch.eye(2)}, str(root / 'base.safetensors'))
        save_file({'diffusion_model.blocks.1.attn.qkv_proj.diff': torch.eye(2)}, str(root / 'a.safetensors'))
        with pytest.raises(ValueError, match='incompatible|unmatched'):
            list(run_lora_merge(dict(base_path=str(root / 'base.safetensors'),
                loras=[dict(path=str(root / 'a.safetensors'))], architecture='MiniMax H3',
                merge_algorithm='graviton', strict_matching=False, dry_run=True)))
        assert not list(root.glob('*.txt'))


def test_graviton_compose_is_signed_base_free_spectral_sum():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        key = 'diffusion_model.blocks.0.attn.qkv_proj'
        delta = torch.tensor([[1., 2.], [3., 5.]])
        for name, value in [('a', delta), ('b', delta * .25)]:
            save_file({key + '.diff': value}, str(root / (name + '.safetensors')))
        out = root / 'out.safetensors'
        events = list(run_lora_compose(dict(loras=[dict(path=str(root / 'a.safetensors'), strength=-1.),
            dict(path=str(root / 'b.safetensors'), strength=1.)], architecture='MiniMax H3',
            merge_algorithm='graviton', merge_device='cpu', output_adapter='lora',
            output_rank=2, frobenius_energy=1., output_path=str(out))))
        tensors = load_file(str(out))
        actual = tensors[key + '.lora_B.weight'].float() @ tensors[key + '.lora_A.weight'].float()
        expected = _graviton_merge_matrix(torch.zeros_like(delta), -.75 * delta, 1., -1, polar=False)[0]
        torch.testing.assert_close(actual, expected, atol=.04, rtol=.01)
        assert events[-1]['status'] == 'finished'
        assert 'base-free spectral' in Path(str(out) + '.txt').read_text()
