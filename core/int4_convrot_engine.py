"""Conservative streaming INT4 ConvRot conversion for supported ComfyUI architectures."""

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
from core.metadata_manager import calculate_civitai_hashes, inject_metadata, merge_custom_metadata
from core.safetensors_engine import write_quant_recipe
from utils.arch_detector import verify_architecture_match
from utils.ltx23_layer_profiles import is_ltx23_preserved_key

INT4_CONVROT_FORMAT = "convrot_w4a4"
CONVROT_GROUP_SIZE = 256
INT4_QUANT_GROUP_SIZE = 64
_LOSSY_FORMATS = {"nvfp4", "mxfp8", "convrot_w4a4"}
_DTYPE_NAMES = {
    torch.float64: "F64", torch.float32: "F32", torch.float16: "F16",
    torch.bfloat16: "BF16", torch.int64: "I64", torch.int32: "I32",
    torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL",
}
_DTYPE_SIZES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8,
                "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
_PRESERVE_RX = {
    arch: [re.compile(pattern) for pattern in PRESERVE_PATTERNS[arch] + BAKED_VAE_PATTERNS]
    for arch in ("WAN 2.2", "Krea 2")
}


def validate_int4_convrot_request(architecture: str, strategy: str) -> Optional[str]:
    if architecture not in {"LTX-2.3", "WAN 2.2", "Krea 2"}:
        return "INT4 ConvRot supports only LTX-2.3, WAN 2.2, and Krea 2."
    if strategy != "Simple":
        return "INT4 ConvRot requires the deterministic Simple strategy."
    return None


def is_preserved_key(architecture: str, key: str) -> bool:
    if architecture == "LTX-2.3":
        return is_ltx23_preserved_key(key)
    return any(pattern.search(key) for pattern in _PRESERVE_RX[architecture])


def build_quant_layer_metadata() -> dict[str, int | str]:
    return {"format": INT4_CONVROT_FORMAT, "convrot_groupsize": CONVROT_GROUP_SIZE,
            "quant_group_size": INT4_QUANT_GROUP_SIZE}


def validate_quantizable_tensor(key: str, tensor: torch.Tensor) -> Optional[str]:
    if not key.endswith(".weight"):
        return "Only .weight tensors are eligible for INT4 ConvRot."
    if not tensor.dtype.is_floating_point:
        return "Only floating-point tensors are eligible for INT4 ConvRot."
    if tensor.ndim != 2:
        return "Only 2D weight tensors are eligible for INT4 ConvRot."
    if tensor.shape[1] % CONVROT_GROUP_SIZE:
        return f"Input dimension must be divisible by {CONVROT_GROUP_SIZE}."
    return None


def quantize_weight(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    error = validate_quantizable_tensor("weight.weight", tensor)
    if error:
        raise ValueError(error)
    try:
        from comfy_kitchen.tensor import TensorCoreConvRotW4A4Layout
    except ImportError as exc:
        raise RuntimeError("INT4 ConvRot requires the pinned comfy-kitchen build. Run Update & Restart.") from exc
    qdata, params = TensorCoreConvRotW4A4Layout.quantize(
        tensor.to(dtype=torch.bfloat16).contiguous(), convrot_groupsize=CONVROT_GROUP_SIZE,
        quant_group_size=INT4_QUANT_GROUP_SIZE,
    )
    return qdata, params.scale


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


def run_int4_convrot_conversion(output_dir: str, source_path: str, model_name: str,
                                 architecture: str, strategy: str, is_full_checkpoint: bool,
                                 custom_metadata: dict | None = None):
    request_error = validate_int4_convrot_request(architecture, strategy)
    if request_error:
        yield request_error, "Aborted: unsupported INT4 ConvRot request"
        return
    arch_ok, arch_msg = verify_architecture_match(source_path, architecture)
    if not arch_ok:
        yield arch_msg, "Aborted: architecture mismatch"
        return
    with safe_open(source_path, framework="pt", device="cpu") as source:
        if _is_lossy_source(source.metadata()):
            yield "Refusing lossy/re-quantized source; use BF16/FP16 source.\n", "Aborted: lossy source"
            return

    output_path = os.path.join(output_dir, f"{model_name}_int4_convrot.safetensors")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    spool_path = None
    tmp_output = output_path + ".tmp"
    try:
        fd, spool_path = tempfile.mkstemp(prefix=".int4-convrot-", dir=os.path.dirname(output_path))
        specs: dict[str, dict[str, Any]] = {}
        quant_layers: dict[str, dict[str, int | str]] = {}
        offset = 0
        total = quantized = preserved = 0
        with os.fdopen(fd, "wb") as spool, safe_open(source_path, framework="pt", device="cpu") as source:
            keys = list(source.keys())
            progress_interval = max(1, len(keys) // 100)
            yield f"INT4 ConvRot: preparing {len(keys)} tensors.\n", "running"
            for index, key in enumerate(keys, start=1):
                total += 1
                tensor = source.get_tensor(key)
                eligible = not is_preserved_key(architecture, key) and validate_quantizable_tensor(key, tensor) is None
                if eligible:
                    qdata, scale = quantize_weight(tensor)
                    spec, offset = _tensor_spec(qdata, offset)
                    specs[key] = spec
                    _write_tensor_bytes(spool, qdata)
                    scale_key = key[:-len(".weight")] + ".weight_scale"
                    spec, offset = _tensor_spec(scale, offset)
                    specs[scale_key] = spec
                    _write_tensor_bytes(spool, scale)
                    quant_layers[key[:-len(".weight")]] = build_quant_layer_metadata()
                    quantized += 1
                else:
                    spec, offset = _tensor_spec(tensor, offset)
                    specs[key] = spec
                    _write_tensor_bytes(spool, tensor)
                    preserved += 1
                if index == len(keys) or index % progress_interval == 0:
                    yield (f"INT4 ConvRot progress: {index}/{len(keys)} tensors "
                           f"({index * 100 // len(keys)}%), {quantized} quantized.\n"), "running"
        metadata = merge_custom_metadata(architecture, model_name, output_path, bits="INT4 ConvRot",
                                         custom_meta=custom_metadata)
        metadata["_quantization_metadata"] = json.dumps({"format_version": "1.0", "layers": quant_layers})
        header = _header(specs, {str(k): str(v) for k, v in metadata.items()})
        with open(tmp_output, "wb") as output, open(spool_path, "rb") as spool:
            output.write(struct.pack("<Q", len(header)))
            output.write(header)
            shutil.copyfileobj(spool, output, length=1024 * 1024)
        os.replace(tmp_output, output_path)
        yield "INT4 ConvRot: calculating output hashes and finalizing metadata.\n", "running"
        hashes = calculate_civitai_hashes(output_path)
        metadata.update({f"civitai.hash.{name}": value for name, value in hashes.items()})
        metadata["modelspec.hash_sha256"] = f"0x{hashes['SHA256'].lower()}"
        injected, metadata_msg = inject_metadata(output_path, metadata)
        recipe = write_quant_recipe(output_path, source_path, model_name, architecture, "INT4 ConvRot",
                                    strategy, "n/a", False, False, is_full_checkpoint, f"{architecture} preserve policy",
                                    ["comfy-kitchen", "TensorCoreConvRotW4A4Layout"], injected, metadata_msg, hashes)
        yield (f"{arch_msg}\nINT4 ConvRot: {quantized} quantized / {preserved} preserved / {total} total tensors.\n"
               f"Output: {output_path}\nRecipe: {recipe}\n"), "INT4 ConvRot complete"
    except Exception as exc:
        try:
            os.remove(tmp_output)
        except OSError:
            pass
        yield f"INT4 ConvRot failed: {exc}\n", "Aborted: INT4 ConvRot failed"
    finally:
        if spool_path:
            try:
                os.remove(spool_path)
            except OSError:
                pass
