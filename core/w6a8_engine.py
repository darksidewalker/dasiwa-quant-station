"""Native, streaming MiniMax H3 W6A8 (group32 / ConvRot256) conversion."""
import json
import os
import re
from typing import Optional

import torch
from safetensors import safe_open

from core.layer_config_builder import BAKED_VAE_PATTERNS, PRESERVE_PATTERNS
from core.metadata_manager import merge_custom_metadata
from core.safetensors_stream import TensorSpool, destination, read_header
from utils.arch_detector import verify_architecture_match

W6A8_FORMAT = "w6a8_int8"
W6A8_QUANT_GROUP_SIZE = 32
W6A8_CONVROT_GROUP_SIZE = 256
W6A8_SUPPORTED_ARCHITECTURES = {"MiniMax H3"}

_HEAVY_WEIGHT = re.compile(
    r"^(?:(?:model\.)?diffusion_model\.)?blocks\.\d+\."
    r"(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2))\.weight$"
)
_PRESERVE_RX = {
    arch: [re.compile(pattern) for pattern in PRESERVE_PATTERNS[arch] + BAKED_VAE_PATTERNS]
    for arch in W6A8_SUPPORTED_ARCHITECTURES
}


def validate_w6a8_request(architecture: str, strategy: str) -> Optional[str]:
    if architecture not in W6A8_SUPPORTED_ARCHITECTURES:
        return "W6A8 (w6a8_int8) supports only MiniMax H3."
    if strategy != "Simple":
        return "W6A8 requires the deterministic Simple strategy."
    return None


def is_preserved_key(architecture: str, key: str) -> bool:
    return any(pattern.search(key) for pattern in _PRESERVE_RX[architecture])


def build_w6a8_layer_metadata() -> dict:
    return {"format": W6A8_FORMAT, "group_size": W6A8_QUANT_GROUP_SIZE,
            "convrot": True, "convrot_groupsize": W6A8_CONVROT_GROUP_SIZE}


def validate_quantizable_tensor(key: str, tensor: torch.Tensor) -> Optional[str]:
    if not _HEAVY_WEIGHT.fullmatch(key):
        return "Only the four H3 blocks.N heavy .weight families are eligible for W6A8."
    if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        return "Only unquantized floating-point tensors are eligible for W6A8."
    if tensor.ndim != 2:
        return "Only 2D weight tensors are eligible for W6A8."
    if any(dim == 0 for dim in tensor.shape):
        return "Empty weight tensors are not eligible for W6A8."
    if tensor.shape[1] % W6A8_QUANT_GROUP_SIZE:
        return "Input dimension must be divisible by 32."
    if tensor.shape[1] % W6A8_CONVROT_GROUP_SIZE:
        return "Input dimension must be divisible by 256."
    return None


def _w6_layout():
    """Reject old layouts that silently swallow bits=6 through **kwargs."""
    import inspect
    from importlib.metadata import version, PackageNotFoundError
    try:
        from comfy_kitchen.tensor import AsymW4A8Int8Layout
        installed = version("comfy-kitchen")
        numbers = re.match(r"^(\d+)\.(\d+)\.(\d+)", installed)
        if not numbers or tuple(map(int, numbers.groups())) < (0, 2, 37):
            raise RuntimeError("W6A8 requires comfy-kitchen >= 0.2.37; run Update & Restart.")
    except (ImportError, PackageNotFoundError) as exc:
        raise RuntimeError("W6A8 requires comfy-kitchen >= 0.2.37; run Update & Restart.") from exc
    parameters = inspect.signature(AsymW4A8Int8Layout.quantize).parameters
    if any(name not in parameters or parameters[name].kind in (
            inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
           for name in ("bits", "scale_search")):
        raise RuntimeError("W6A8 requires explicit bits and scale_search support (comfy-kitchen >= 0.2.37).")
    return AsymW4A8Int8Layout


def quantize_weight(tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return packed weight under '' and exactly two reference scale suffixes."""
    error = validate_quantizable_tensor("blocks.0.mlp.fc1.weight", tensor)
    if error:
        raise ValueError(error)
    if not torch.isfinite(tensor).all():
        raise ValueError("Nonfinite W6A8 source weight")
    layout = _w6_layout()
    prepared = tensor.to(dtype=torch.bfloat16).contiguous()
    if not torch.isfinite(prepared).all():
        raise ValueError("BF16 overflow in W6A8 source weight")
    qdata, params = layout.quantize(
        prepared, group_size=W6A8_QUANT_GROUP_SIZE,
        convrot_groupsize=W6A8_CONVROT_GROUP_SIZE, bits=6, symmetric=True,
        codebook=False, scale_search=True, scale_dtype=torch.float8_e4m3fn)
    n, k = tensor.shape
    companions = {"": qdata, "_s_rel": params.scale, "_s_channel": params.s_channel}
    expected = {"": (torch.int8, (n, 3 * k // 4)),
                "_s_rel": (torch.float8_e4m3fn, (n, k // W6A8_QUANT_GROUP_SIZE)),
                "_s_channel": (torch.float32, (n,))}
    for suffix, (dtype, shape) in expected.items():
        value = companions[suffix]
        if not isinstance(value, torch.Tensor) or value.dtype != dtype or tuple(value.shape) != shape:
            raise RuntimeError(f"Invalid W6A8 packing/shape/dtype for weight{suffix}; expected {dtype} {shape}.")
        if value.is_floating_point() and not torch.isfinite(value.float()).all():
            raise RuntimeError(f"Nonfinite W6A8 scale: weight{suffix}")
    if params.codebook is not None or params.correction is not None:
        raise RuntimeError("W6A8 must not emit codebook or correction tensors.")
    return companions


_QUANT_KEY = re.compile(
    r"(?:^|\.)(?:comfy_quant|_quantization_metadata|quant_state|qweight|qzeros|"
    r"packed_weight|weight_s_rel|weight_s_channel|weight_codebook|weight_correction|"
    r"weight_scale[^.]*|scale_weight[^.]*|input_scale[^.]*|scale_input[^.]*|"
    r"output_scale[^.]*|absmax)(?:$|\.)"
)
_FULL_PRECISION = {"F16", "BF16", "F32", "F64"}


def validate_unquantized_source(header: dict) -> Optional[str]:
    """Inspect the entire source manifest before any tensor is loaded/staged."""
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        return "Malformed source metadata"
    for key, value in metadata.items():
        if key in {"_quantization_metadata", "comfy_quant", "quantization_config",
                   "quantization_method", "quantization", "quant_method"}:
            return f"Quantization header marker: {key}"
        if key in {"quantization.format", "quantization.dtype"}:
            return f"Quantization header marker: {key}"
        if key == "format" and str(value).lower() in {
                "nvfp4", "mxfp8", "convrot_w4a4", "asym_w4a8_int8", "w6a8_int8",
                "int8", "fp8", "gguf", "gptq", "awq"}:
            return f"Quantized source format: {value}"
        if key == "quantization.bits" and str(value).upper() not in {
                "BF16", "FP16", "F16", "FP32", "F32", "FP64", "F64", "16", "32", "64"}:
            return f"Quantized source precision: {value}"
    for key, spec in header.items():
        if key == "__metadata__":
            continue
        if _QUANT_KEY.search(key):
            return f"Quantization tensor marker: {key}"
        dtype = spec["dtype"]
        if dtype.startswith("F8_") or (key.endswith(".weight") and dtype not in _FULL_PRECISION):
            return f"Quantized/packed source dtype {dtype}: {key}"
    return None


class StagedSafetensorsOutput(TensorSpool):
    """Keep FP8 handling local; shared spool registries remain unchanged."""
    def tensor(self, key, tensor):
        if tensor.dtype != torch.float8_e4m3fn:
            return super().tensor(key, tensor)
        tensor = tensor.detach().cpu().contiguous()
        if not torch.isfinite(tensor.float()).all():
            raise ValueError(f"Nonfinite W6A8 FP8 scale: {key}")
        # The byte count is identical; only the audited header dtype differs.
        self.chunks(key, "U8", list(tensor.shape), (tensor.view(torch.uint8),))
        self.header[key]["dtype"] = "F8_E4M3"


def _recipe(output, source, model_name, architecture, strategy, is_full_checkpoint,
            preserve_loader_metadata, quantized, preserved):
    return "\n".join([
        "DaSiWa Quantization Recipe", f"Output path: {output}", f"Source path: {source}",
        f"Model name: {model_name}", f"Architecture: {architecture}",
        "Format: W6A8 (w6a8_int8)", f"Strategy: {strategy}",
        f"Full checkpoint: {'yes' if is_full_checkpoint else 'no'}",
        f"Preserve loader metadata: {'yes' if preserve_loader_metadata else 'no'}",
        "Layer policy: H3 four heavy block families only; preserve H3 structural and baked components",
        f"Layers: {quantized} quantized / {preserved} preserved",
        "Metadata: native per-layer U8 .comfy_quant JSON; merged before atomic publication",
        "Command: comfy-kitchen AsymW4A8Int8Layout.quantize "
        "bits=6 group_size=32 convrot_groupsize=256 symmetric=True codebook=False "
        "scale_search=True scale_dtype=torch.float8_e4m3fn", ""])


def run_w6a8_conversion(output_dir: str, source_path: str, model_name: str,
                        architecture: str, strategy: str, is_full_checkpoint: bool,
                        custom_metadata: dict | None = None, preserve_loader_metadata=True):
    """Stream one source tensor at a time, then publish audited artifact + recipe."""
    request_error = validate_w6a8_request(architecture, strategy)
    if request_error:
        yield request_error, "Aborted: unsupported W6A8 request"
        return
    try:
        header, data_start = read_header(source_path)
        source_error = validate_unquantized_source(header)
        if source_error:
            yield (f"Refusing lossy/re-quantized source; use an unquantized BF16/FP16 source. "
                   f"{source_error}\n"), "Aborted: lossy source"
            return
        arch_ok, arch_msg = verify_architecture_match(source_path, architecture)
        if not arch_ok:
            yield arch_msg, "Aborted: architecture mismatch"
            return
        output = destination({"output_dir": output_dir}, f"{model_name}_w6a8.safetensors",
                             inputs=(source_path,))
        _w6_layout()
        keys = [key for key in header if key != "__metadata__"]
        quantized = preserved = 0
        with StagedSafetensorsOutput(output) as spool, safe_open(source_path, framework="pt", device="cpu") as source:
            yield f"W6A8: preparing {len(keys)} tensors.\n", "running"
            interval = max(1, len(keys) // 100)
            for index, key in enumerate(keys, 1):
                spec = header[key]
                eligible = (not is_preserved_key(architecture, key) and _HEAVY_WEIGHT.fullmatch(key)
                            and spec["dtype"] in _FULL_PRECISION and len(spec["shape"]) == 2
                            and all(spec["shape"]) and spec["shape"][1] % W6A8_CONVROT_GROUP_SIZE == 0)
                if eligible:
                    tensor = source.get_tensor(key)
                    companions = quantize_weight(tensor)
                    for suffix, companion in companions.items():
                        spool.tensor(key + suffix, companion)
                        del companion
                    native = json.dumps(build_w6a8_layer_metadata(), separators=(",", ":")).encode("utf-8")
                    spool.tensor(key[:-len(".weight")] + ".comfy_quant",
                                 torch.tensor(list(native), dtype=torch.uint8))
                    quantized += 1
                    del tensor, companions
                else:
                    spool.copy(key, source_path, spec, data_start)
                    preserved += 1
                if index == len(keys) or index % interval == 0:
                    yield (f"W6A8 progress: {index}/{len(keys)} tensors "
                           f"({index * 100 // len(keys)}%), {quantized} quantized.\n"), "running"
            if not quantized:
                raise ValueError("Zero quantized layers: no compatible H3 heavy weights were found.")
            metadata = merge_custom_metadata(
                architecture, model_name, output, bits="W6A8", custom_meta=custom_metadata,
                is_full=is_full_checkpoint, source_metadata=header.get("__metadata__", {}),
                preserve_loader_metadata=preserve_loader_metadata)
            # Native tensor records own runtime layout; custom metadata cannot inject stale layouts.
            for key in ("_quantization_metadata", "comfy_quant", "quantization_config",
                        "quantization_method", "quantization", "quant_method"):
                metadata.pop(key, None)
            metadata["quantization.bits"] = "W6A8"
            yield "W6A8: auditing and publishing output + recipe.\n", "running"
            recipe_path = spool.publish(metadata, _recipe(
                output, source_path, model_name, architecture, strategy, is_full_checkpoint,
                preserve_loader_metadata, quantized, preserved))
        yield (f"{arch_msg}\nW6A8: {quantized} quantized / {preserved} preserved / {len(keys)} total tensors.\n"
               f"Output: {output}\nRecipe: {recipe_path}\n"), "W6A8 complete"
    except Exception as exc:
        yield f"W6A8 failed: {exc}\n", "Aborted: W6A8 failed"
