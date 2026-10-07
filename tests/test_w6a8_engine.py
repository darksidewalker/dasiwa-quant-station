"""CPU-only contract and publication tests for the native H3 W6A8 engine."""
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def test_request_and_explicit_h3_allowlist():
    engine = importlib.import_module("core.w6a8_engine")
    assert engine.W6A8_SUPPORTED_ARCHITECTURES == {"MiniMax H3"}
    assert engine.validate_w6a8_request("MiniMax H3", "Simple") is None
    assert "MiniMax H3" in engine.validate_w6a8_request("WAN 2.2", "Simple")
    assert "Simple" in engine.validate_w6a8_request("MiniMax H3", "Balanced")
    weight = torch.ones(2, 256, dtype=torch.bfloat16)
    for prefix in ("", "diffusion_model.", "model.diffusion_model."):
        for family in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
            assert engine.validate_quantizable_tensor(prefix + "blocks.0." + family + ".weight", weight) is None
    for key in ("blocks.0.attn.other.weight", "blocks.0.mlp.fc3.weight",
                "text_encoder.blocks.0.mlp.fc1.weight", "token_refiner.blocks.0.mlp.fc1.weight",
                "vae.blocks.0.mlp.fc1.weight", "final_layer.video_out.weight",
                "blocks.x.mlp.fc1.weight", "blocks.0.mlp.fc1.bias"):
        assert engine.validate_quantizable_tensor(key, weight)
    for key in ("blocks.0.adaln_proj.linear.weight", "blocks.0.attn.q_norm.weight",
                "blocks.0.norm1.weight", "adaln_t_table", "time_embedder.proj_in.weight",
                "token_refiner.final_norm.weight", "vae.decoder.weight"):
        assert engine.is_preserved_key("MiniMax H3", key)


def test_upstream_cpu_quantization_and_reconstruction():
    from comfy_kitchen.tensor import AsymW4A8Int8Layout
    from core.w6a8_engine import quantize_weight
    weight = torch.randn(2, 256, generator=torch.Generator().manual_seed(7), dtype=torch.bfloat16)
    result = quantize_weight(weight)
    assert set(result) == {"", "_s_rel", "_s_channel"}
    assert result[""].dtype == torch.int8 and result[""].shape == (2, 192)
    assert result["_s_rel"].dtype == torch.float8_e4m3fn and result["_s_rel"].shape == (2, 8)
    assert result["_s_channel"].dtype == torch.float32 and result["_s_channel"].shape == (2,)
    params = AsymW4A8Int8Layout.Params(
        scale=result["_s_rel"], s_channel=result["_s_channel"], correction=None,
        codebook=None, orig_dtype=weight.dtype, orig_shape=tuple(weight.shape),
        group_size=32, convrot_groupsize=256)
    reconstructed = AsymW4A8Int8Layout.dequantize(result[""], params)
    assert reconstructed.shape == weight.shape
    assert ((reconstructed.float() - weight.float()).norm() / weight.float().norm()).item() < 0.06


def test_legacy_kwargs_layout_is_not_w6_capable():
    from core.w6a8_engine import quantize_weight
    class LegacyLayout:
        @classmethod
        def quantize(cls, tensor, **kwargs):
            raise AssertionError("An older kwargs-only layout must not be called")
    with mock.patch("comfy_kitchen.tensor.AsymW4A8Int8Layout", LegacyLayout):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "bits|0.2.37"):
            quantize_weight(torch.ones(2, 256))


def test_silent_w4_packing_is_rejected():
    from core.w6a8_engine import quantize_weight
    class BrokenLayout:
        @classmethod
        def quantize(cls, tensor, bits=4, scale_search=True, **kwargs):
            assert bits == 6 and scale_search is True
            assert kwargs['group_size'] == 32 and kwargs['convrot_groupsize'] == 256
            assert kwargs['symmetric'] is True and kwargs['codebook'] is False
            return torch.zeros(2, 128, dtype=torch.int8), SimpleNamespace(
                scale=torch.ones(2, 8).to(torch.float8_e4m3fn),
                s_channel=torch.ones(2), correction=None, codebook=None)
    with mock.patch("comfy_kitchen.tensor.AsymW4A8Int8Layout", BrokenLayout):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "packing|shape"):
            quantize_weight(torch.ones(2, 256))


def _convert(tmp, tensors, metadata=None, model_name="output"):
    from core.w6a8_engine import run_w6a8_conversion
    source = Path(tmp) / "source.safetensors"
    save_file(tensors, str(source), metadata=metadata)
    with mock.patch("core.w6a8_engine.verify_architecture_match", return_value=(True, "ok")):
        return list(run_w6a8_conversion(str(tmp), str(source), model_name,
                                       "MiniMax H3", "Simple", False))


def test_native_streaming_output_preserves_non_allowlisted_bytes():
    from core.w6a8_engine import build_w6a8_layer_metadata
    from core.safetensors_stream import DTYPES
    before = dict(DTYPES)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "owned-stage"
        stage.mkdir()
        prefix = "diffusion_model."
        tensors = {prefix + "blocks.0." + family + ".weight": torch.randn(2, 256)
                   for family in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")}
        preserved = {"token_refiner.blocks.0.mlp.fc1.weight": torch.randn(2, 256),
                     "vae.blocks.0.mlp.fc1.weight": torch.randn(2, 256),
                     prefix + "final_layer.video_out.weight": torch.randn(2, 256),
                     prefix + "blocks.0.attn.other.weight": torch.randn(2, 256),
                     prefix + "blocks.0.norm1.weight": torch.randn(2),
                     "rope.inv_freq": torch.arange(4, dtype=torch.float64)}
        with mock.patch.dict("os.environ", {"DASIWA_H3_STAGE_DIR": str(stage)}):
            events = _convert(tmp, {**tensors, **preserved}, {"config": '{"x":1}'})
        assert events[-1][1] == "W6A8 complete", events
        assert "4 quantized" in events[-1][0] and "6 preserved" in events[-1][0]
        output = Path(tmp) / "output_w6a8.safetensors"
        with safe_open(output, framework="pt", device="cpu") as handle:
            assert len(handle.keys()) == 22
            assert handle.metadata()['config'] == '{"x":1}'
            assert handle.metadata()['quantization.bits'] == "W6A8"
            for weight_key in tensors:
                module = weight_key[:-len('.weight')]
                packed = handle.get_tensor(weight_key)
                assert packed.dtype == torch.int8 and packed.shape == (2, 192)
                assert handle.get_tensor(weight_key + '_s_rel').dtype == torch.float8_e4m3fn
                assert handle.get_tensor(weight_key + '_s_channel').dtype == torch.float32
                native = handle.get_tensor(module + '.comfy_quant')
                assert native.dtype == torch.uint8 and native.ndim == 1
                assert json.loads(bytes(native.tolist())) == build_w6a8_layer_metadata()
                assert weight_key + '_codebook' not in handle.keys()
                assert weight_key + '_correction' not in handle.keys()
            for key, tensor in preserved.items():
                assert torch.equal(handle.get_tensor(key), tensor)
        recipe = Path(str(output) + '.txt')
        assert recipe.exists() and 'bits=6' in recipe.read_text()
        assert 'codebook=False' in recipe.read_text() and 'scale_search=True' in recipe.read_text()
        assert not list(stage.iterdir())
        assert dict(DTYPES) == before


def test_strict_source_guard_rejects_manifest_and_quantization_markers():
    weight = {"blocks.0.mlp.fc1.weight": torch.ones(2, 256)}
    cases = [({"blocks.0.mlp.fc1.weight": torch.ones(2, 256, dtype=dtype)}, None)
             for dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2)]
    cases += [(dict(weight, **{marker: torch.ones(1)}), None) for marker in
              ("blocks.0.mlp.fc1.comfy_quant", "blocks.0.mlp.fc1.weight_s_rel",
               "blocks.0.mlp.fc1.weight_scale", "blocks.0.mlp.fc1.weight_codebook",
               "blocks.0.mlp.fc1.weight_correction", "blocks.0.mlp.fc1.input_scale")]
    cases += [(weight, {"_quantization_metadata": value}) for value in ("{}", "garbage", "[]")]
    cases += [(weight, {"quantization.bits": value}) for value in ("FP8", "W6A8", "INT8", "NVFP4")]
    for tensors, metadata in cases:
        with tempfile.TemporaryDirectory() as tmp:
            events = _convert(tmp, tensors, metadata)
            assert events[-1][1] == "Aborted: lossy source", events
            assert not (Path(tmp) / "output_w6a8.safetensors").exists()
            assert not list(Path(tmp).glob('.h3_stage_*'))


def test_zero_quantized_layers_never_publishes():
    with tempfile.TemporaryDirectory() as tmp:
        events = _convert(tmp, {"final_layer.video_out.weight": torch.ones(2, 256)})
        assert events[-1][1].startswith("Aborted"), events
        assert "zero" in events[-1][0].lower(), events
        assert not (Path(tmp) / "output_w6a8.safetensors").exists()
        assert not (Path(tmp) / "output_w6a8.safetensors.txt").exists()
        assert not list(Path(tmp).glob('.h3_stage_*'))


def test_quantization_header_format_markers_are_rejected():
    weight = {"blocks.0.mlp.fc1.weight": torch.ones(2, 256)}
    for metadata in ({"quantization.format": "w6a8_int8"}, {"format": "nvfp4"},
                     {"quantization.dtype": "int8"}):
        with tempfile.TemporaryDirectory() as tmp:
            events = _convert(tmp, weight, metadata)
            assert events[-1][1] == "Aborted: lossy source", events


def test_packing_contract_rejects_wrong_scale_and_extra_companions():
    from core.w6a8_engine import quantize_weight
    for change in ('scale_dtype', 'scale_shape', 'channel_dtype', 'channel_shape',
                   'codebook', 'correction', 'nonfinite'):
        class BrokenLayout:
            @classmethod
            def quantize(cls, tensor, bits=4, scale_search=True, **kwargs):
                params = SimpleNamespace(scale=torch.ones(2, 8).to(torch.float8_e4m3fn),
                                         s_channel=torch.ones(2), correction=None, codebook=None)
                if change == 'scale_dtype': params.scale = params.scale.float()
                if change == 'scale_shape': params.scale = torch.ones(2, 16).to(torch.float8_e4m3fn)
                if change == 'channel_dtype': params.s_channel = params.s_channel.bfloat16()
                if change == 'channel_shape': params.s_channel = torch.ones(2, 1)
                if change == 'codebook': params.codebook = torch.ones(64)
                if change == 'correction': params.correction = torch.ones(8, 2)
                if change == 'nonfinite': params.s_channel[0] = float('nan')
                return torch.zeros(2, 192, dtype=torch.int8), params
        with mock.patch('comfy_kitchen.tensor.AsymW4A8Int8Layout', BrokenLayout):
            with unittest.TestCase().assertRaises(RuntimeError):
                quantize_weight(torch.ones(2, 256))


def test_invalid_shapes_and_nonfinite_weights_are_rejected():
    from core.w6a8_engine import quantize_weight, validate_quantizable_tensor
    key = 'blocks.0.mlp.fc1.weight'
    for tensor, message in ((torch.ones(256), '2D'), (torch.ones(2, 248), '32'),
                            (torch.ones(2, 224), '256'), (torch.ones(0, 256), 'Empty'),
                            (torch.ones(2, 256, dtype=torch.int8), 'floating-point')):
        assert message in (validate_quantizable_tensor(key, tensor) or '')
        with unittest.TestCase().assertRaises(ValueError):
            quantize_weight(tensor)
    for value in (float('nan'), float('inf'), 1e100):
        with unittest.TestCase().assertRaisesRegex(ValueError, 'Nonfinite|overflow'):
            quantize_weight(torch.full((2, 256), value, dtype=torch.float64))


def test_preexisting_output_and_recipe_are_never_overwritten():
    for suffix in ('', '.txt'):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / ('output_w6a8.safetensors' + suffix)
            target.write_bytes(b'existing')
            events = _convert(tmp, {'blocks.0.mlp.fc1.weight': torch.ones(2, 256)})
            assert events[-1][1].startswith('Aborted')
            assert target.read_bytes() == b'existing'
            assert not list(Path(tmp).glob('.h3_stage_*'))


def test_artifact_publication_failure_rolls_back_only_owned_recipe():
    from core import safetensors_stream
    real_link = safetensors_stream.os.link
    with tempfile.TemporaryDirectory() as tmp:
        unrelated = Path(tmp) / '.h3_stage_other'
        unrelated.mkdir()
        (unrelated / 'keep').write_text('other job')
        def fail_artifact(src, dst):
            if str(dst).endswith('.safetensors'):
                assert Path(str(dst) + '.txt').exists(), 'Recipe must precede publication'
                raise OSError('publication blocked')
            return real_link(src, dst)
        with mock.patch('core.safetensors_stream.os.link', side_effect=fail_artifact):
            events = _convert(tmp, {'blocks.0.mlp.fc1.weight': torch.ones(2, 256)})
        assert events[-1][1].startswith('Aborted') and 'publication blocked' in events[-1][0]
        assert not (Path(tmp) / 'output_w6a8.safetensors').exists()
        assert not (Path(tmp) / 'output_w6a8.safetensors.txt').exists()
        assert list(Path(tmp).glob('.h3_stage_*')) == [unrelated]
        assert (unrelated / 'keep').read_text() == 'other job'


def test_generator_close_cleans_owned_stage_without_publishing():
    from core.w6a8_engine import run_w6a8_conversion
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / 'source.safetensors'
        save_file({'blocks.0.mlp.fc1.weight': torch.ones(2, 256)}, str(source))
        with mock.patch('core.w6a8_engine.verify_architecture_match', return_value=(True, 'ok')):
            generator = run_w6a8_conversion(tmp, str(source), 'output', 'MiniMax H3', 'Simple', False)
            assert next(generator)[1] == 'running'
            assert len(list(Path(tmp).glob('.h3_stage_*'))) == 1
            generator.close()
        assert not list(Path(tmp).glob('.h3_stage_*'))
        assert not (Path(tmp) / 'output_w6a8.safetensors').exists()


def test_metadata_failure_does_not_publish():
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch('core.w6a8_engine.merge_custom_metadata', side_effect=ValueError('metadata blocked')):
            events = _convert(tmp, {'blocks.0.mlp.fc1.weight': torch.ones(2, 256)})
        assert events[-1][1].startswith('Aborted') and 'metadata blocked' in events[-1][0]
        assert not (Path(tmp) / 'output_w6a8.safetensors').exists()
        assert not (Path(tmp) / 'output_w6a8.safetensors.txt').exists()
        assert not list(Path(tmp).glob('.h3_stage_*'))


def test_minimum_kitchen_version_is_enforced():
    from core.w6a8_engine import quantize_weight
    with mock.patch('importlib.metadata.version', return_value='0.2.36'):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, '0.2.37'):
            quantize_weight(torch.ones(2, 256))


def test_preserved_tensors_are_copied_without_loading():
    from core.w6a8_engine import run_w6a8_conversion
    import core.w6a8_engine as engine
    real_open = engine.safe_open
    loaded = []
    class TrackedSource:
        def __init__(self, *args, **kwargs): self.source = real_open(*args, **kwargs)
        def __enter__(self): self.handle = self.source.__enter__(); return self
        def __exit__(self, *args): return self.source.__exit__(*args)
        def get_tensor(self, key): loaded.append(key); return self.handle.get_tensor(key)
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / 'source.safetensors'
        save_file({'blocks.0.mlp.fc1.weight': torch.ones(2, 256),
                   'final_layer.video_out.weight': torch.ones(2, 256)}, str(source))
        with mock.patch('core.w6a8_engine.safe_open', TrackedSource), mock.patch(
                'core.w6a8_engine.verify_architecture_match', return_value=(True, 'ok')):
            events = list(run_w6a8_conversion(tmp, str(source), 'output', 'MiniMax H3', 'Simple', False))
        assert events[-1][1] == 'W6A8 complete', events
        assert loaded == ['blocks.0.mlp.fc1.weight']


def test_bad_request_and_architecture_mismatch_do_not_stage():
    from core.w6a8_engine import run_w6a8_conversion
    with tempfile.TemporaryDirectory() as tmp:
        for architecture, strategy in (('WAN 2.2', 'Simple'), ('MiniMax H3', 'Balanced')):
            events = list(run_w6a8_conversion(tmp, 'missing-file', 'output', architecture, strategy, False))
            assert events[-1][1] == 'Aborted: unsupported W6A8 request'
        source = Path(tmp) / 'source.safetensors'
        save_file({'blocks.0.mlp.fc1.weight': torch.ones(2, 256)}, str(source))
        with mock.patch('core.w6a8_engine.verify_architecture_match', return_value=(False, 'wrong arch')):
            events = list(run_w6a8_conversion(tmp, str(source), 'output', 'MiniMax H3', 'Simple', False))
        assert events[-1] == ('wrong arch', 'Aborted: architecture mismatch')
        assert not list(Path(tmp).glob('.h3_stage_*'))


def test_bf16_merged_source_is_accepted_without_weakening_tensor_guards():
    from core.w6a8_engine import validate_unquantized_source
    metadata = {"quantization.bits": "BF16 merged"}
    header = {"__metadata__": metadata,
              "blocks.0.mlp.fc1.weight": {"dtype": "BF16"}}
    assert validate_unquantized_source(header) is None
    for dtype in ("I8", "U8", "F8_E4M3", "F8_E5M2"):
        packed = {**header, "blocks.0.mlp.fc1.weight": {"dtype": dtype}}
        assert validate_unquantized_source(packed)
    for marker in ("blocks.0.mlp.fc1.comfy_quant", "blocks.0.mlp.fc1.weight_s_rel"):
        assert validate_unquantized_source({**header, marker: {"dtype": "U8"}})
    with tempfile.TemporaryDirectory() as tmp:
        events = _convert(tmp, {"blocks.0.mlp.fc1.weight": torch.ones(2, 256, dtype=torch.bfloat16)}, metadata)
        assert events[-1][1] == "W6A8 complete", events
        assert (Path(tmp) / "output_w6a8.safetensors").exists()


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith("test_") and callable(value))
