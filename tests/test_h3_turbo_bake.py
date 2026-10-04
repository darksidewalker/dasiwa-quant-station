import hashlib
import tempfile
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file, load_file
from core.lora_merge_engine import run_lora_merge


def test_complete_bakes_adaln_bias_and_final_without_relaxing_default():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base, adapter, out = [root / (s + '.safetensors') for s in ('base', 'adapter', 'out')]
        keys = ['blocks.0.adaln_proj', 'final_layer.adaln_proj']
        save_file({k + suffix: torch.zeros(shape) for k in keys
                   for suffix, shape in [('.weight', (4, 3)), ('.bias', (4,))]}, str(base))
        save_file({k + suffix: torch.ones(shape) for k in keys
                   for suffix, shape in [('.lora_A.weight', (1, 3)), ('.lora_B.weight', (4, 1)), ('.diff_b', (4,))]}, str(adapter))
        payload = dict(base_path=str(base), output_path=str(out), architecture='MiniMax H3',
                       loras=[dict(path=str(adapter), strength=-0.5)], merge_device='cpu')
        default = list(run_lora_merge(dict(payload, dry_run=True)))
        assert 'skipped_preserve=2' in ''.join(e.get('text', '') for e in default)
        events = list(run_lora_merge(dict(payload, h3_turbo_complete=True)))
        assert events[-1]['status'] == 'finished'
        for value in load_file(str(out)).values():
            torch.testing.assert_close(value, torch.full_like(value, -0.5))
        assert 'H3 Turbo complete: yes' in Path(str(out) + '.txt').read_text()


@pytest.mark.parametrize('arch,algorithm', [('WAN 2.2', 'additive'), ('MiniMax H3', 'consensus')])
def test_complete_rejects_wrong_arch_or_consensus(arch, algorithm):
    with pytest.raises(ValueError, match='H3.*additive'):
        list(run_lora_merge(dict(base_path='absent', architecture=arch, merge_algorithm=algorithm,
                                 h3_turbo_complete=True)))
@pytest.mark.parametrize('metadata', [{}, {'adaln_coordinate_table_sha256': '0' * 64}])
def test_pruned_complete_requires_exact_gauge(metadata):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base, adapter, out = [root / (s + '.safetensors') for s in ('base', 'adapter', 'out')]
        save_file({'adaln_t_table': torch.ones(5, 2), 'blocks.0.adaln_proj.weight': torch.zeros(4, 2)}, str(base))
        save_file({'blocks.0.adaln_proj.diff': torch.ones(4, 2)}, str(adapter), metadata=metadata)
        with pytest.raises(ValueError, match='gauge'):
            list(run_lora_merge(dict(base_path=str(base), output_path=str(out), architecture='MiniMax H3',
                                    loras=[dict(path=str(adapter))], h3_turbo_complete=True)))
        assert not out.exists()


def test_table_patch_rejected_before_baking():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base, adapter = root / 'base.safetensors', root / 'adapter.safetensors'
        save_file({'adaln_t_table': torch.ones(5, 2)}, str(base))
        save_file({'adaln_t_table.diff': torch.ones(5, 2)}, str(adapter))
        with pytest.raises(ValueError, match='table'):
            list(run_lora_merge(dict(base_path=str(base), loras=[dict(path=str(adapter))])))
