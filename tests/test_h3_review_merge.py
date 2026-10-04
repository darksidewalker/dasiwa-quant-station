import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from core.lora_compose_engine import run_lora_compose
from core.lora_merge_engine import run_lora_merge


class H3ReviewMergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def adapter(self, name, module='blocks.0.adaln_proj', gauge=None, extra=None):
        path = self.root / (name + '.safetensors')
        tensors = {module+'.lora_A.weight': torch.tensor([[1.12345, 0.23]]),
                   module+'.lora_B.weight': torch.tensor([[0.7345], [1.2345]])}
        tensors.update(extra or {})
        save_file(tensors, str(path), metadata={} if gauge is None else {'adaln_coordinate_table_sha256': gauge})
        return {'path': str(path)}

    def compose(self, specs, **kw):
        return list(run_lora_compose(dict(loras=specs, architecture='MiniMax H3', merge_device='cpu', merge_algorithm='additive', output_path=str(self.root/'out.safetensors'), **kw)))

    def test_baking_atomic_recipe_and_unhandled_audit(self):
        base = self.root/'base.safetensors'
        save_file({'blocks.0.attn.qkv_proj.weight': torch.zeros(2, 2), 'untouched': torch.tensor([7], dtype=torch.int64)}, str(base))
        adapter = self.adapter('turbo', module='blocks.0.attn.qkv_proj', extra={'unknown.alpha': torch.tensor(1.)})
        payload = dict(base_path=str(base), loras=[adapter], architecture='MiniMax H3', h3_turbo_complete=True, strict_matching=False, merge_device='cpu', output_path=str(self.root/'baked.safetensors'))
        list(run_lora_merge(payload))
        recipe = self.root/'baked.safetensors.txt'
        self.assertTrue(recipe.exists())
        self.assertIn('unknown.alpha', recipe.read_text())
        self.assertIn('partial', recipe.read_text())
        with safe_open(payload['output_path'], framework='pt') as h:
            self.assertEqual(h.get_tensor('untouched').item(), 7)
        with self.assertRaises(FileExistsError):
            list(run_lora_merge(payload))

    def test_baking_failure_cleans_staged_output(self):
        base = self.root/'base.safetensors'
        save_file({'blocks.0.attn.qkv_proj.weight': torch.zeros(2, 2)}, str(base))
        specs = [self.adapter('a', module='blocks.0.attn.qkv_proj')]
        with patch('core.lora_merge_engine._merge_target_with_policy', side_effect=ValueError('injected')):
            with self.assertRaisesRegex(ValueError, 'injected'):
                list(run_lora_merge(dict(base_path=str(base), loras=specs, architecture='MiniMax H3', output_path=str(self.root/'baked.safetensors'))))
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['a.safetensors', 'base.safetensors'])

    def test_baking_consensus_reconstructs_bounded_rows(self):
        from core.lora_merge_engine import _merge_target_consensus
        from core.consensus_merge import resolve_consensus_preset
        specs = [self.adapter('a', module='x'), self.adapter('b', module='x')]
        for s in specs:
            save_file({'x.diff': torch.ones(513, 2)}, s['path'])
        ops = [dict(lora_path=s['path'], is_diff=True, diff_key='x.diff', scale=1.) for s in specs]
        import contextlib
        original = torch.stack
        def bounded(values, *a, **kw):
            self.assertLessEqual(values[0].shape[0], 256)
            return original(values, *a, **kw)
        with contextlib.ExitStack() as contexts:
            handles = {s['path']: contexts.enter_context(safe_open(s['path'], framework='pt')) for s in specs}
            with patch('core.lora_merge_engine.torch.stack', side_effect=bounded):
                result = _merge_target_consensus(torch.zeros(513, 2), ops, handles, resolve_consensus_preset('neutral', 'MiniMax H3'), 'cpu')
        torch.testing.assert_close(result, torch.ones(513, 2))

    def test_composition_streams_layers_and_publishes_atomic_recipe(self):
        from core.safetensors_stream import TensorSpool
        specs = [self.adapter('a', module='blocks.0.attn.qkv_proj', extra={'blocks.1.attn.qkv_proj.diff': torch.ones(2, 2)}), self.adapter('b', module='blocks.0.attn.qkv_proj')]
        from core.lora_compose_engine import _compose_layer
        writes = []
        original = TensorSpool.tensor
        def record(spool, key, tensor):
            writes.append(key)
            return original(spool, key, tensor)
        def check(*args):
            if args[0][0]['kind'] == 'diff':
                self.assertTrue(writes, 'previous layer was retained instead of spooled')
            return _compose_layer(*args)
        with patch.object(TensorSpool, 'tensor', record), patch('core.lora_compose_engine._compose_layer', side_effect=check):
            self.compose(specs)
        self.assertTrue((self.root/'out.safetensors.txt').is_file())
        with self.assertRaises(FileExistsError):
            self.compose(specs)

    def test_composition_failure_publishes_neither_artifact_nor_recipe(self):
        from core.lora_compose_engine import _compose_layer
        specs = [self.adapter('a', module='blocks.0.attn.qkv_proj', extra={'blocks.1.attn.qkv_proj.diff': torch.ones(2, 2)}), self.adapter('b', module='blocks.0.attn.qkv_proj')]
        def fail(*args):
            if args[0][0]['kind'] == 'diff':
                raise ValueError('injected second layer failure')
            return _compose_layer(*args)
        with patch('core.lora_compose_engine._compose_layer', side_effect=fail):
            with self.assertRaisesRegex(ValueError, 'second layer'):
                self.compose(specs)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['a.safetensors', 'b.safetensors'])

    def test_consensus_reconstructs_bounded_rows(self):
        from core.lora_compose_engine import _compose_layer
        from core.consensus_merge import resolve_consensus_preset
        paths = [self.adapter('a', module='blocks.0.attn.qkv_proj'), self.adapter('b', module='blocks.0.attn.qkv_proj')]
        for spec in paths:
            save_file({'blocks.0.attn.qkv_proj.lora_A.weight': torch.ones(1, 2), 'blocks.0.attn.qkv_proj.lora_B.weight': torch.ones(513, 1)}, spec['path'])
        import contextlib
        items = [dict(path=s['path'], kind='lora', down_key='blocks.0.attn.qkv_proj.lora_A.weight', up_key='blocks.0.attn.qkv_proj.lora_B.weight', rank=1, shape=(513, 2), scale=1.) for s in paths]
        stack = torch.stack
        sizes = []
        def bounded(values, *a, **kw):
            sizes.append(values[0].shape[0])
            self.assertLessEqual(values[0].shape[0], 256)
            return stack(values, *a, **kw)
        with contextlib.ExitStack() as contexts:
            handles = {s['path']: contexts.enter_context(safe_open(s['path'], framework='pt')) for s in paths}
            with patch('core.lora_compose_engine.torch.stack', side_effect=bounded):
                _compose_layer(items, handles, 'cpu', 'consensus', resolve_consensus_preset('neutral', 'MiniMax H3'), 'lora', None, 1, .99)
        self.assertGreaterEqual(len(sizes), 3)

    def test_report_measures_bf16_serialized_delta(self):
        self.compose([self.adapter('a', module='blocks.0.attn.qkv_proj'), self.adapter('b', module='blocks.0.attn.qkv_proj')])
        recipe = (self.root/'out.safetensors.txt').read_text()
        reports = json.loads(recipe.split('Layer report:\n')[1])
        with safe_open(str(self.root/'out.safetensors'), framework='pt') as h:
            rebuilt = h.get_tensor('diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight').float() @ h.get_tensor('diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight').float()
        original = 2 * torch.tensor([[0.7345], [1.2345]]) @ torch.tensor([[1.12345, 0.23]])
        expected = (torch.linalg.vector_norm(original-rebuilt)/torch.linalg.vector_norm(original)).item()
        self.assertAlmostEqual(reports[0]['relative_error'], expected, places=6)

    def test_whole_layer_oom_retries_cpu_and_logs_retry(self):
        from core.lora_compose_engine import _compose_layer
        calls = []
        def fail_once(*args):
            calls.append(args[2])
            if len(calls) == 1:
                raise torch.cuda.OutOfMemoryError('injected reconstruction OOM')
            return _compose_layer(*args)
        with patch('core.lora_compose_engine._device', return_value='cuda:0'), patch('core.lora_compose_engine._compose_layer', side_effect=fail_once):
            events = self.compose([self.adapter('a'), self.adapter('b')])
        self.assertEqual(calls, ['cuda:0', 'cpu'])
        self.assertIn('CPU retry', ''.join(e.get('text', '') for e in events))
        reports = json.loads((self.root/'out.safetensors.txt').read_text().split('Layer report:\n')[1])
        self.assertEqual(reports[0]['device'], 'cpu')
        self.assertEqual(reports[0]['fallback_reason'], 'cuda_oom')

    def test_complete_source_accounting_keeps_protected_refiner_partial(self):
        base = self.root/'base.safetensors'
        save_file({'blocks.0.attn.qkv_proj.weight': torch.zeros(2, 2),
                   'blocks.1.attn.qkv_proj.weight': torch.zeros(2, 2),
                   'blocks.1.attn.qkv_proj.bias': torch.zeros(2),
                   'token_refiner.blocks.0.attn.qkv_proj.weight': torch.zeros(2, 2)}, str(base))
        spec = self.adapter('turbo', module='blocks.0.attn.qkv_proj', extra={
            'blocks.0.attn.qkv_proj.alpha': torch.tensor(1.),
            'blocks.1.attn.qkv_proj.diff': torch.ones(2, 2),
            'blocks.1.attn.qkv_proj.diff_b': torch.ones(2),
            'token_refiner.blocks.0.attn.qkv_proj.diff': torch.ones(2, 2)})
        payload = dict(base_path=str(base), loras=[spec], architecture='MiniMax H3', h3_turbo_complete=True, dry_run=True)
        text = ''.join(e.get('text', '') for e in run_lora_merge(payload))
        self.assertIn('application: complete; omissions=0', text)
        text = ''.join(e.get('text', '') for e in run_lora_merge(dict(payload, protect_token_refiner=True)))
        self.assertIn('application: partial; omissions=0 protected_refiner=1', text)

    def bake(self, strict):
        base = self.root/'base.safetensors'
        save_file({'blocks.0.attn.qkv_proj.weight': torch.zeros(2, 2)}, str(base))
        adapter = self.adapter('turbo', module='blocks.0.attn.qkv_proj', extra={'unknown.lora_A.weight': torch.ones(1, 2)})
        return list(run_lora_merge(dict(base_path=str(base), loras=[adapter], architecture='MiniMax H3', h3_turbo_complete=True, strict_matching=strict, dry_run=True)))

    def test_complete_bake_rejects_orphan(self):
        with self.assertRaisesRegex(ValueError, 'unknown.lora_A.weight'):
            self.bake(True)

    def test_lenient_bake_reports_orphan_partial(self):
        text = ''.join(e.get('text', '') for e in self.bake(False))
        self.assertIn('application: partial; omissions=1', text)
        self.assertIn('unknown.lora_A.weight', text)

    def test_composition_checks_bound_ordinary_input_gauge_when_adaln_present(self):
        with self.assertRaisesRegex(ValueError, 'gauge'):
            self.compose([self.adapter('a', gauge='a'), self.adapter('b', module='blocks.0.attn.qkv_proj', gauge='b')])

    def test_factorization_reports_do_not_rebuild_dense_delta(self):
        from core.adapter_factorization import factorize_lora, factorize_lokr
        delta = torch.eye(4)
        with patch('core.adapter_factorization.reconstruct_lora', side_effect=AssertionError('dense reconstruction')):
            down, up, report = factorize_lora(delta, max_rank=2)
        expected = float(torch.linalg.vector_norm(delta-up@down)/torch.linalg.vector_norm(delta))
        self.assertAlmostEqual(report.relative_error, expected, places=6)
        with patch('core.adapter_factorization.reconstruct_lokr', side_effect=AssertionError('dense reconstruction')):
            w1, w2, report = factorize_lokr(delta, (2, 2), (2, 2))
        expected = float(torch.linalg.vector_norm(delta-torch.kron(w1, w2))/torch.linalg.vector_norm(delta))
        self.assertAlmostEqual(report.relative_error, expected, places=6)

    def test_auto_rank_cap_is_reported_as_legacy_cap(self):
        self.compose([self.adapter('a', module='blocks.0.attn.qkv_proj'), self.adapter('b', module='blocks.0.attn.qkv_proj')])
        reports = json.loads((self.root/'out.safetensors.txt').read_text().split('Layer report:\n')[1])
        self.assertEqual(reports[0]['effective_rank_cap'], 1)
        self.assertEqual(reports[0]['error_basis'], 'serialized_vs_original_merged_delta')
        self.assertEqual(reports[0]['storage_dtype'], 'BF16')

    def test_composition_rejects_conflicting_adaln_gauges(self):
        with self.assertRaisesRegex(ValueError, 'gauge'):
            self.compose([self.adapter('a', gauge='a'), self.adapter('b', gauge='b')])

    def test_composition_rejects_bound_unbound_adaln_mix(self):
        with self.assertRaisesRegex(ValueError, 'gauge'):
            self.compose([self.adapter('a', gauge='a'), self.adapter('b')])

    def test_composition_preserves_matching_gauge_with_ordinary_unbound(self):
        self.compose([self.adapter('a', gauge='a'), self.adapter('b', gauge='a'), self.adapter('c', module='blocks.0.attn.qkv_proj')])
        with safe_open(str(self.root/'out.safetensors'), framework='pt') as h:
            self.assertEqual(h.metadata().get('adaln_coordinate_table_sha256'), 'a')
