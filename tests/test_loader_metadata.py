import json
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from core import metadata_manager as mm
from core.lora_merge_engine import run_lora_merge


def test_shared_ui_preservation_checkbox_and_payloads():
    root = Path(__file__).resolve().parents[1]
    html = (root / 'web/index.html').read_text()
    js = (root / 'web/app.js').read_text()
    assert 'id="preserve-loader-metadata" type="checkbox" checked' in html
    assert js.count('preserve_loader_metadata: $("preserve-loader-metadata").checked') == 3
    assert 's.preserveLoaderMetadata ?? true' in js



@pytest.mark.parametrize('module,function', [
    ('safetensors_engine', 'run_safe_conversion'),
    ('int4_convrot_engine', 'run_int4_convrot_conversion'),
    ('w4a8_engine', 'run_w4a8_conversion'),
    ('gguf_engine', 'run_gguf_conversion'),
])
def test_quant_engines_default_source_metadata_preservation_on(module, function):
    import importlib
    import inspect
    engine = importlib.import_module('core.' + module)
    assert inspect.signature(getattr(engine, function)).parameters['preserve_loader_metadata'].default is True


def test_conversion_preserves_config_but_not_source_quant_layout():
    source = {'config': '{"actual":true}', '_quantization_metadata': 'stale', 'custom': 'keep'}
    meta = mm.merge_custom_metadata('MiniMax H3', 'test', 'PREVIEW_MODE',
                                    source_metadata=source)
    assert meta['config'] == source['config']
    assert '_quantization_metadata' not in meta
    disabled = mm.merge_custom_metadata('MiniMax H3', 'test', 'PREVIEW_MODE',
                                        source_metadata=source, preserve_loader_metadata=False)
    assert disabled['config'] == source['config']
    assert 'custom' not in disabled
    assert meta['custom'] == 'keep'


def test_manual_injection_protects_existing_config_and_runtime_layout():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / 'test.safetensors')
        existing = {'config': '{"actual":true}', '_quantization_metadata': 'original', 'custom': 'kept'}
        save_file({'x': torch.zeros(1)}, path, metadata=existing)
        ok, _ = mm.inject_metadata(path, {'config': 'wrong', '_quantization_metadata': 'wrong', 'modelspec.title': 'edited'})
        assert ok
        actual = mm.read_source_metadata(path)
        for key, value in existing.items():
            assert actual[key] == value
        assert actual['modelspec.title'] == 'edited'


@pytest.mark.parametrize('dtype,metadata', [
    (torch.int8, {}),
    (torch.float32, {'_quantization_metadata': '{"layers":{"x":{"format":"asym_w4a8_int8"}}}'}),
    (torch.float8_e4m3fn, {}),
])
@pytest.mark.parametrize('dry_run', [False, True])
def test_lora_rejects_quantized_base_before_matching(dtype, metadata, dry_run):
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / 'base.safetensors'
        out = Path(tmp) / 'out.safetensors'
        save_file({'blocks.0.attn.wq.weight': torch.zeros(3, 4).to(dtype)}, str(base), metadata=metadata)
        with pytest.raises(ValueError, match='quantized'):
            list(run_lora_merge({'base_path': str(base), 'output_path': str(out),
                                 'loras': [], 'dry_run': dry_run, 'preserve_loader_metadata': False}))
        assert not out.exists()


def test_model_merge_preserve_loader_metadata_flag():
    from core.model_merge_engine import _read_base_metadata
    source = {'config': '{"actual":true}', 'modelspec.architecture': 'minimax-h3',
              'custom': 'keep', 'modelspec.hash_sha256': '0xdead'}
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / 'base.safetensors')
        save_file({'x': torch.zeros(1)}, path, metadata=source)
        kept = _read_base_metadata(path)
        assert kept['config'] == source['config']
        assert kept['modelspec.architecture'] == source['modelspec.architecture']
        assert kept['custom'] == 'keep'
        assert 'modelspec.hash_sha256' not in kept
        reduced = _read_base_metadata(path, preserve_loader_metadata=False)
        assert reduced['config'] == source['config']
        assert reduced['modelspec.architecture'] == source['modelspec.architecture']
        assert 'custom' not in reduced
        assert 'modelspec.hash_sha256' not in reduced
