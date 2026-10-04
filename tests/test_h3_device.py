"""Bounded H3 projection device-policy contracts."""
import unittest
from unittest.mock import patch


class DeviceTests(unittest.TestCase):
    def test_bare_cuda_preflight(self):
        from core.h3_prune_engine import require_h3
        require_h3({'cuda_device': 'cuda'})

    def test_explicit_unavailable_cuda_reports_cpu_rows(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('core.h3_device'), 'shared row device policy missing')
        import torch
        from core.h3_device import H3DevicePolicy
        with patch('torch.cuda.is_available', return_value=False):
            policy = H3DevicePolicy({'merge_device': 'cuda', 'cuda_device': 'cuda:2'})
            result = policy.run(lambda w, v: w @ v, (torch.ones(2, 3), torch.ones(3, 1)), 2)
        torch.testing.assert_close(result, torch.full((2, 1), 3., dtype=torch.float64))
        self.assertEqual(policy.summary()['requested'], 'cuda')
        self.assertEqual(policy.summary()['cuda_device'], 'cuda:2')
        self.assertEqual(policy.summary()['cpu'], 1)
        self.assertEqual(policy.summary()['cuda'], 0)
        self.assertEqual(policy.summary()['cuda_unavailable'], 1)
        self.assertIn('cuda_unavailable', policy.log())

    def test_headroom_checked_on_selected_device_each_chunk(self):
        import torch
        from core.h3_device import H3DevicePolicy
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=3), patch('torch.cuda.mem_get_info', return_value=(1024, 2048)) as memory:
            policy = H3DevicePolicy({'cuda_device': 'cuda:2', 'vram_headroom_mb': 1})
            for _ in range(2):
                policy.run(lambda x: x + 1, (torch.ones(2),), 2)
        self.assertEqual(memory.call_args_list, [unittest.mock.call('cuda:2')] * 2)
        self.assertEqual(policy.summary()['insufficient_vram'], 2)
        self.assertEqual(policy.summary()['cpu'], 2)
        self.assertEqual(policy.summary()['cuda_unavailable'], 0)

    def test_oom_at_transfer_math_or_readback_retries_whole_chunk(self):
        import torch
        from core.h3_device import H3DevicePolicy
        original_to, original_cpu = torch.Tensor.to, torch.Tensor.cpu
        for stage in ('transfer', 'math', 'readback'):
            with self.subTest(stage=stage):
                attempted = []
                def transfer(t, *args, **kwargs):
                    if str(kwargs.get('device', '')).startswith('cuda'):
                        attempted.append(kwargs['device'])
                        if stage == 'transfer' and len(attempted) == 1:
                            raise torch.cuda.OutOfMemoryError('injected transfer OOM')
                        kwargs['device'] = 'cpu'
                    return original_to(t, *args, **kwargs)
                calls = []
                def math(x):
                    calls.append(1)
                    if stage == 'math' and len(calls) == 1:
                        raise torch.cuda.OutOfMemoryError('injected math OOM')
                    return x + 2
                reads = []
                def readback(t):
                    reads.append(1)
                    if stage == 'readback' and len(reads) == 1:
                        raise torch.cuda.OutOfMemoryError('injected readback OOM')
                    return original_cpu(t)
                with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=3), patch('torch.cuda.mem_get_info', return_value=(2**30, 2**31)), patch.object(torch.Tensor, 'to', transfer), patch.object(torch.Tensor, 'cpu', readback), patch('torch.cuda.device'), patch('torch.cuda.empty_cache'):
                    policy = H3DevicePolicy({'merge_device': 'cuda', 'cuda_device': 'cuda:2', 'vram_headroom_mb': 0})
                    result = policy.run(math, (torch.ones(2),), 2)
                torch.testing.assert_close(result, torch.full((2,), 3., dtype=torch.float64))
                self.assertEqual(attempted[0], 'cuda:2')
                self.assertEqual(policy.summary()['oom'], 1)
                self.assertEqual(policy.summary()['cpu'], 1)
                self.assertEqual(policy.summary()['cuda'], 0)

    def test_engines_publish_actual_unavailable_device_accounting(self):
        import os, json, torch
        from tempfile import TemporaryDirectory
        from safetensors.torch import save_file
        from tests.test_h3_adapter_convert_engine import AdapterConversionTests
        from core.h3_prune_engine import run_h3_prune
        from core.h3_adapter_convert_engine import run_h3_adapter_convert
        with TemporaryDirectory() as d:
            base, ref, adapter, out, _, _ = AdapterConversionTests().fixture(d)
            module = 'blocks.0.adaln_proj.linear'
            save_file({module + '.lora_A.weight': torch.ones(2, 3), module + '.lora_B.weight': torch.ones(6, 2), module + '.alpha': torch.tensor(3.)}, adapter)
            common = dict(base_path=base, merge_device='cuda', cuda_device='cuda:2', row_chunk_size=2)
            with patch('torch.cuda.is_available', return_value=False):
                prune_events = list(run_h3_prune(dict(common, reference_path=ref, output_path=out)))
                convert_events = list(run_h3_adapter_convert(dict(common, pruned_target_path=ref, adapter_path=adapter, output_path=out+'.adapter.safetensors')))
            for path, events in ((out, prune_events), (out+'.adapter.safetensors', convert_events)):
                with open(path+'.txt') as handle:
                    report = json.loads(handle.read().split('\n', 1)[1])
                devices = report['summary']['devices']
                self.assertEqual(devices['requested'], 'cuda')
                self.assertGreater(devices['cpu'], 0)
                self.assertEqual(devices['cuda'], 0)
                self.assertEqual(devices['cuda_unavailable'], devices['cpu'])
                self.assertTrue(any('H3 row device summary:' in e.get('text', '') for e in events))

    def test_high_rank_adapter_projection_never_transfers_more_than_256_rows(self):
        import torch
        from tempfile import TemporaryDirectory
        from safetensors.torch import save_file, load_file
        from tests.test_h3_adapter_convert_engine import AdapterConversionTests
        from core.h3_adapter_convert_engine import run_h3_adapter_convert
        from core.h3_device import H3DevicePolicy
        calls = []
        original = H3DevicePolicy.run
        def track(policy, operation, tensors, output_elements):
            calls.append(tensors[0].shape[0])
            return original(policy, operation, tensors, output_elements)
        with TemporaryDirectory() as d:
            base, ref, adapter, out, _, _ = AdapterConversionTests().fixture(d)
            module = 'blocks.0.adaln_proj.linear'
            up = torch.ones(6, 300)
            save_file({module+'.lora_A.weight': torch.ones(300, 3), module+'.lora_B.weight': up}, adapter)
            with patch.object(H3DevicePolicy, 'run', track):
                list(run_h3_adapter_convert(dict(base_path=base, pruned_target_path=ref, adapter_path=adapter, output_path=out, merge_device='cpu', row_chunk_size=1024)))
            self.assertTrue(calls)
            self.assertLessEqual(max(calls), 256)
            self.assertTrue(torch.equal(load_file(out)[module+'.lora_B.weight'], up))

    def test_memory_query_failure_is_unavailability_not_headroom(self):
        import torch
        from core.h3_device import H3DevicePolicy
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=1), patch('torch.cuda.mem_get_info', side_effect=RuntimeError('driver unavailable')):
            policy = H3DevicePolicy({'merge_device': 'cuda'})
            policy.run(lambda x: x, (torch.ones(1),), 1)
        self.assertEqual(policy.summary()['cuda_unavailable'], 1)
        self.assertEqual(policy.summary()['insufficient_vram'], 0)
        self.assertEqual(policy.summary()['used'], 'cpu')
        self.assertEqual(policy.summary()['fallbacks'], 1)

    def test_discovery_or_oom_cache_failure_still_falls_back_to_cpu(self):
        import torch
        from core.h3_device import H3DevicePolicy
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', side_effect=RuntimeError('driver failed')):
            policy = H3DevicePolicy({'merge_device': 'cuda'})
            result = policy.run(lambda x: x, (torch.ones(1),), 1)
        self.assertEqual(policy.summary()['cuda_unavailable'], 1)
        self.assertEqual(result.device.type, 'cpu')
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=1), patch('torch.cuda.mem_get_info', return_value=(2**30, 2**31)), patch.object(torch.Tensor, 'to', side_effect=[torch.cuda.OutOfMemoryError('transfer'), torch.ones(1, dtype=torch.float64)]), patch('torch.cuda.device'), patch('torch.cuda.empty_cache', side_effect=RuntimeError('cache cleanup failed')):
            policy = H3DevicePolicy({'merge_device': 'cuda', 'vram_headroom_mb': 0})
            result = policy.run(lambda x: x, (torch.ones(1),), 1)
        self.assertEqual(policy.summary()['oom'], 1)
        self.assertEqual(policy.summary()['used'], 'cpu')

    def test_dry_runs_report_no_device_work_without_artifacts(self):
        import os, torch
        from tempfile import TemporaryDirectory
        from safetensors.torch import save_file
        from tests.test_h3_adapter_convert_engine import AdapterConversionTests
        from core.h3_prune_engine import run_h3_prune
        from core.h3_adapter_convert_engine import run_h3_adapter_convert
        with TemporaryDirectory() as d:
            base, ref, adapter, out, _, _ = AdapterConversionTests().fixture(d)
            module = 'blocks.0.adaln_proj.linear'
            save_file({module+'.lora_A.weight': torch.ones(2, 3), module+'.lora_B.weight': torch.ones(6, 2)}, adapter)
            for engine, options in ((run_h3_prune, dict(reference_path=ref)), (run_h3_adapter_convert, dict(pruned_target_path=ref, adapter_path=adapter))):
                events = list(engine(dict(base_path=base, output_path=out, dry_run=True, merge_device='cuda', **options)))
                device_logs = [e['text'] for e in events if e.get('text', '').startswith('H3 row device summary:')]
                self.assertEqual(len(device_logs), 1)
                self.assertIn('"used": "none"', device_logs[0])
                self.assertFalse(os.path.exists(out))
                self.assertFalse(os.path.exists(out+'.txt'))

    def test_cpu_request_never_queries_cuda(self):
        import torch
        from core.h3_device import H3DevicePolicy
        with patch('torch.cuda.is_available', side_effect=AssertionError('CPU must not query CUDA')):
            policy = H3DevicePolicy({'merge_device': 'cpu'})
            policy.run(lambda x: x * 2, (torch.ones(2),), 2)
        self.assertEqual(policy.summary()['used'], 'cpu')
        self.assertEqual(policy.summary()['fallbacks'], 0)

    def test_non_oom_math_error_is_not_hidden_by_cpu_fallback(self):
        import torch
        from core.h3_device import H3DevicePolicy
        original_to = torch.Tensor.to
        def transfer(t, *args, **kwargs):
            kwargs['device'] = 'cpu'
            return original_to(t, *args, **kwargs)
        def invalid(x):
            raise ValueError('malformed input')
        policy = H3DevicePolicy({'vram_headroom_mb': 0})
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=1), patch('torch.cuda.mem_get_info', return_value=(2**30, 2**31)), patch.object(torch.Tensor, 'to', transfer):
            with self.assertRaisesRegex(ValueError, 'malformed'):
                policy.run(invalid, (torch.ones(1),), 1)
        self.assertEqual(policy.summary()['fallbacks'], 0)
        self.assertEqual(policy.summary()['cpu'], 0)

    def test_guarded_real_cuda_matches_cpu_prune_and_affine_adapter(self):
        import json, torch
        from tempfile import TemporaryDirectory
        from safetensors.torch import save_file, load_file
        from tests.test_h3_adapter_convert_engine import AdapterConversionTests
        from core.h3_prune_engine import run_h3_prune
        from core.h3_adapter_convert_engine import run_h3_adapter_convert
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable')
        try:
            free, _ = torch.cuda.mem_get_info('cuda:0')
        except Exception as exc:
            self.skipTest(f'CUDA memory query unavailable: {exc}')
        if free < 512 * 1024**2: self.skipTest('Keep 512 MiB free for tiny CUDA verification')
        with TemporaryDirectory() as d:
            base, ref, adapter, out, _, _ = AdapterConversionTests().fixture(d)
            m = 'blocks.0.adaln_proj.linear'
            final = 'final_layer.adaln_proj.linear'
            save_file({m+'.lora_A.weight': torch.tensor([[.3, -.1, .5], [-.4, .7, .2]]),
                       m+'.lora_B.weight': torch.arange(12).reshape(6, 2).float() / 13,
                       m+'.alpha': torch.tensor(3.), m+'.diff_b': torch.arange(6).float()/10,
                       final+'.diff': torch.tensor([[.1, -.2, .3], [-.4, .5, -.6]]),
                       final+'.diff_b': torch.tensor([.2, -.3])}, adapter)
            common = dict(base_path=base, cuda_device='cuda:0', row_chunk_size=2, vram_headroom_mb=256)
            for engine, options in ((run_h3_prune, dict(reference_path=ref)),
                                    (run_h3_adapter_convert, dict(pruned_target_path=ref, adapter_path=adapter))):
                paths = []
                for device in ('cpu', 'cuda'):
                    path = out + f'.{engine.__name__}.{device}.safetensors'
                    paths.append(path)
                    list(engine(dict(common, **options, merge_device=device, output_path=path)))
                    with open(path+'.txt') as handle:
                        summary = json.loads(handle.read().split('\n', 1)[1])['summary']['devices']
                    self.assertEqual(summary['used'], device)
                    self.assertGreater(summary[device], 0)
                    self.assertEqual(summary['fallbacks'], 0)
                cpu, cuda = map(load_file, paths)
                self.assertEqual(set(cpu), set(cuda))
                for key in cpu:
                    torch.testing.assert_close(cuda[key], cpu[key], rtol=2e-5, atol=1e-6)
                if engine == run_h3_adapter_convert:
                    self.assertTrue(torch.equal(cuda[m+'.lora_B.weight'], load_file(adapter)[m+'.lora_B.weight']))

