"""Streaming W4A8 asymmetric ConvRot quantization for MiniMax H3.

Produces outputs compatible with the reference MiniMax w4a8 quants:

    blocks.N.attn.out_proj.weight         I8  [N, K//2]  (packed 4-bit qdata)
    blocks.N.attn.out_proj.weight_s_rel   F8_E4M3 [N, K//16]
    blocks.N.attn.out_proj.weight_s_channel F32 [N]
    blocks.N.attn.out_proj.weight_codebook F32 [16]
    (+ weight_correction F32 [K//16, N] only for asymmetric layers)

with per-layer `_quantization_metadata` entries
`{"format": "asym_w4a8_int8", "group_size": 16, "convrot": true,
"convrot_groupsize": 256}` — identical to the reference quants.
"""

import json
import os
import re
import shutil
import struct
import tempfile
from typing import Any, Optional

import torch
from safetensors import safe_open

from core.layer_config_builder import BAKED_VAE_PATTERNS, PRESERVE_PATTERNS
from core.metadata_manager import inject_metadata, merge_custom_metadata, read_source_metadata
from core.safetensors_engine import write_quant_recipe
from utils.arch_detector import verify_architecture_match

W4A8_FORMAT = "asym_w4a8_int8"
W4A8_QUANT_GROUP_SIZE = 16
W4A8_CONVROT_GROUP_SIZE = 256
W4A8_SUPPORTED_ARCHITECTURES = {"MiniMax H3"}

_LOSSY_FORMATS = {"nvfp4", "mxfp8", "convrot_w4a4", "asym_w4a8_int8"}

_DTYPE_NAMES = {
    torch.float64: "F64", torch.float32: "F32", torch.float16: "F16",
    torch.bfloat16: "BF16", torch.int64: "I64", torch.int32: "I32",
    torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL",
    torch.float8_e4m3fn: "F8_E4M3", torch.float8_e5m2: "F8_E5M2",
}
_DTYPE_SIZES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8,
                "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
                "F8_E4M3": 1, "F8_E5M2": 1}

_PRESERVE_RX = {
    arch: [re.compile(pattern) for pattern in PRESERVE_PATTERNS[arch] + BAKED_VAE_PATTERNS]
    for arch in W4A8_SUPPORTED_ARCHITECTURES
}


def validate_w4a8_request(architecture: str, strategy: str) -> Optional[str]:
    if architecture not in W4A8_SUPPORTED_ARCHITECTURES:
        return "W4A8 (asym_w4a8_int8) supports only MiniMax H3."
    if strategy != "Simple":
        return "W4A8 requires the deterministic Simple strategy."
    return None


def is_preserved_key(architecture: str, key: str) -> bool:
    return any(pattern.search(key) for pattern in _PRESERVE_RX[architecture])


def build_w4a8_layer_metadata() -> dict[str, int | str]:
    """Mirror the reference quant's per-layer metadata entry."""
    return {
        "format": W4A8_FORMAT,
        "group_size": W4A8_QUANT_GROUP_SIZE,
        "convrot": True,
        "convrot_groupsize": W4A8_CONVROT_GROUP_SIZE,
    }


def validate_quantizable_tensor(key: str, tensor: torch.Tensor) -> Optional[str]:
    if not key.endswith(".weight"):
        return "Only .weight tensors are eligible for W4A8."
    if not tensor.dtype.is_floating_point:
        return "Only floating-point tensors are eligible for W4A8."
    if tensor.ndim != 2:
        return "Only 2D weight tensors are eligible for W4A8."
    k = tensor.shape[1]
    if k % 16:
        return "Input dimension must be divisible by 16."
    if k % W4A8_CONVROT_GROUP_SIZE:
        return f"Input dimension must be divisible by {W4A8_CONVROT_GROUP_SIZE}."
    if k % W4A8_QUANT_GROUP_SIZE:
        return f"Input dimension must be divisible by {W4A8_QUANT_GROUP_SIZE}."
    return None


def quantize_weight(tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """Quantize one full-precision weight into the W4A8 companion set.

    Returns the tensors that get written under the reference key names:
    {"weight": qdata, "weight_s_rel", "weight_s_channel",
     "weight_codebook"(optional), "weight_correction"(optional)}.
    """
    error = validate_quantizable_tensor("weight.weight", tensor)
    if error:
        raise ValueError(error)
    try:
        from comfy_kitchen.tensor import AsymW4A8Int8Layout
    except ImportError as exc:
        raise RuntimeError(
            "W4A8 requires a comfy-kitchen build with AsymW4A8Int8Layout. "
            "Run Update & Restart to install the current build."
        ) from exc
    qdata, params = AsymW4A8Int8Layout.quantize(
        tensor.to(dtype=torch.bfloat16).contiguous(),
        group_size=W4A8_QUANT_GROUP_SIZE,
        convrot_groupsize=W4A8_CONVROT_GROUP_SIZE,
        symmetric=True,
        scale_dtype=torch.float8_e4m3fn,
        codebook=True,
    )
    companions = {
        "": qdata,
        "_s_rel": params.scale,
        "_s_channel": params.s_channel,
    }
    if params.codebook is not None:
        companions["_codebook"] = params.codebook
    if params.correction is not None:
        companions["_correction"] = params.correction
    return companions


def _is_lossy_source(metadata: dict[str, str] | None) -> bool:
    if not metadata:
        return False
    try:
        layers = json.loads(metadata.get("_quantization_metadata", "{}") or "{}").get("layers", {})
    except json.JSONDecodeError:
        return True
    return any(info.get("format") in _LOSSY_FORMATS or info.get("convrot") for info in layers.values())


def _write_tensor_bytes(handle: Any, tensor: torch.Tensor) -> None:
    tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tofile(handle)


def _tensor_spec(tensor: torch.Tensor, offset: int) -> tuple[dict[str, Any], int]:
    dtype = _DTYPE_NAMES.get(tensor.dtype)
    if dtype is None:
        raise ValueError(f"Unsupported safetensors dtype: {tensor.dtype}")
    size = tensor.numel() * _DTYPE_SIZES[dtype]
    return {"dtype": dtype, "shape": list(tensor.shape), "data_offsets": [offset, offset + size]}, offset + size


def _header(specs: dict[str, dict[str, Any]], metadata: dict[str, str]) -> bytes:
    return json.dumps({"__metadata__": metadata, **specs}, separators=(",", ":")).encode("utf-8")


def run_w4a8_conversion(output_dir: str, source_path: str, model_name: str,
                        architecture: str, strategy: str, is_full_checkpoint: bool,
                        custom_metadata: dict | None = None, preserve_loader_metadata=True):
    request_error = validate_w4a8_request(architecture, strategy)
    if request_error:
        yield request_error, "Aborted: unsupported W4A8 request"
        return
    arch_ok, arch_msg = verify_architecture_match(source_path, architecture)
    if not arch_ok:
        yield arch_msg, "Aborted: architecture mismatch"
        return
    with safe_open(source_path, framework="pt", device="cpu") as source:
        if _is_lossy_source(source.metadata()):
            yield "Refusing lossy/re-quantized source; use a BF16/FP16 source.\n", "Aborted: lossy source"
            return

    output_path = os.path.join(output_dir, f"{model_name}_w4a8.safetensors")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    spool_path = None
    tmp_output = output_path + ".tmp"
    try:
        fd, spool_path = tempfile.mkstemp(prefix=".w4a8-", dir=os.path.dirname(output_path))
        specs: dict[str, dict[str, Any]] = {}
        quant_layers: dict[str, dict[str, int | str]] = {}
        offset = 0
        total = quantized = preserved = 0
        with os.fdopen(fd, "wb") as spool, safe_open(source_path, framework="pt", device="cpu") as source:
            keys = list(source.keys())
            progress_interval = max(1, len(keys) // 100)
            yield f"W4A8: preparing {len(keys)} tensors.\n", "running"
            for index, key in enumerate(keys, start=1):
                total += 1
                tensor = source.get_tensor(key)
                eligible = not is_preserved_key(architecture, key) and validate_quantizable_tensor(key, tensor) is None
                if eligible:
                    companions = quantize_weight(tensor)
                    base_key = key  # the .weight key carries the packed qdata
                    for suffix, tensor_ in companions.items():
                        target_key = base_key + suffix
                        spec, offset = _tensor_spec(tensor_, offset)
                        specs[target_key] = spec
                        _write_tensor_bytes(spool, tensor_)
                    quant_layers[key[:-len(".weight")]] = build_w4a8_layer_metadata()
                    quantized += 1
                else:
                    spec, offset = _tensor_spec(tensor, offset)
                    specs[key] = spec
                    _write_tensor_bytes(spool, tensor)
                    preserved += 1
                if index == len(keys) or index % progress_interval == 0:
                    yield (f"W4A8 progress: {index}/{len(keys)} tensors "
                           f"({index * 100 // len(keys)}%), {quantized} quantized.\n"), "running"
        metadata = merge_custom_metadata(architecture, model_name, output_path, bits="W4A8",
                                         custom_meta=custom_metadata,
                                         source_metadata=read_source_metadata(source_path),
                                         preserve_loader_metadata=preserve_loader_metadata)
        metadata["_quantization_metadata"] = json.dumps({"format_version": "1.0", "layers": quant_layers})
        header = _header(specs, {str(k): str(v) for k, v in metadata.items()})
        with open(tmp_output, "wb") as output, open(spool_path, "rb") as spool:
            output.write(struct.pack("<Q", len(header)))
            output.write(header)
            shutil.copyfileobj(spool, output, length=1024 * 1024)
        os.replace(tmp_output, output_path)
        yield "W4A8: finalizing metadata.\n", "running"
        injected, metadata_msg = inject_metadata(output_path, metadata)
        recipe = write_quant_recipe(output_path, source_path, model_name, architecture, "W4A8",
                                   strategy, "n/a", False, False, is_full_checkpoint,
                                   f"{architecture} preserve policy",
                                   ["comfy-kitchen", "AsymW4A8Int8Layout"], injected, metadata_msg,
                                   preserve_loader_metadata=preserve_loader_metadata)
        yield (f"{arch_msg}\nW4A8: {quantized} quantized / {preserved} preserved / {total} total tensors.\n"
               f"Output: {output_path}\nRecipe: {recipe}\n"), "W4A8 complete"
    except Exception as exc:
        try:
            os.remove(tmp_output)
        except OSError:
            pass
        yield f"W4A8 failed: {exc}\n", "Aborted: W4A8 failed"
    finally:
        if spool_path:
            try:
                os.remove(spool_path)
            except OSError:
                pass
