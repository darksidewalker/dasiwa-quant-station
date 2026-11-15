"""HyperFlow loader-contract preservation in quantization."""
import json, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.layer_config_builder import build_layer_config_dict
from core.metadata_manager import merge_custom_metadata

HF_KEYS_KEEP = [
    "endpoint_time_embedder.proj_in.weight",
    "hyperflow_gate", "hyperflow_sigmas",
    "blocks.1.attn.q_proj.down.weight", "blocks.1.attn.k_proj.up.weight",
    "blocks.46.mlp.fc1.down.weight", "blocks.46.mlp.fc2.up.weight",
    "token_refiner.blocks.0.mlp.fc1.weight",
]

def _skipped(config, key):
    return any(not pat.startswith("_") and re.fullmatch(pat, key) and cfg.get("skip")
               for pat, cfg in config.items() if isinstance(cfg, dict))

def test_hyperflow_tensors_preserved_all_h3_formats():
    for fmt in ("FP8", "INT8 Tensor-wise", "INT8 Row-wise ConvRot Runtime", "NVFP4", "NVFP4 HQ"):
        config, _ = build_layer_config_dict("MiniMax H3", fmt)
        for key in HF_KEYS_KEEP:
            assert _skipped(config, key), f"{fmt} would pack {key}"
        assert not _skipped(config, "blocks.0.attn.qkv_proj.weight")  # dense heavy linears still quantize

def test_hyperflow_metadata_survives_quant_rewrite():
    src = {"hyperflow": json.dumps({"format": 1, "variant": "fl2va", "weights": "int8_convrot", "blocks": 47}),
           "format": "pt"}
    out = merge_custom_metadata("MiniMax H3", "model", "/tmp/x.safetensors", bits="INT8 Row-wise ConvRot Runtime",
                                source_metadata=src, preserve_loader_metadata=True)
    hf = json.loads(out["hyperflow"])
    assert hf["format"] == 1 and hf["variant"] == "fl2va" and hf["blocks"] == 47
    assert hf["weights"] == "int8_row-wise_convrot_runtime"
    assert hf["requantized_by"] == "dasiwa-quant-station"

def test_absent_hyperflow_entry_stays_absent():
    out = merge_custom_metadata("MiniMax H3", "model", "/tmp/x.safetensors", bits="FP8",
                                source_metadata={"format": "pt"}, preserve_loader_metadata=True)
    assert "hyperflow" not in out
