import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from safetensors import safe_open


@dataclass(frozen=True)
class TensorInfo:
    key: str
    shape: Tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class LoraPair:
    base_name: str
    down_key: str
    up_key: str
    alpha_key: Optional[str]
    rank: int
    down_shape: Tuple[int, ...]
    up_shape: Tuple[int, ...]
    target_candidates: Tuple[str, ...]


@dataclass(frozen=True)
class DiffPatch:
    """A direct additive weight delta (ComfyUI .diff format)."""
    diff_key: str
    diff_shape: Tuple[int, ...]
    target_candidates: Tuple[str, ...]


def read_safetensors_manifest(path: str) -> Dict[str, TensorInfo]:
    manifest: Dict[str, TensorInfo] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            sl = f.get_slice(key)
            manifest[key] = TensorInfo(
                key=key,
                shape=tuple(sl.get_shape()),
                dtype=str(sl.get_dtype()),
            )
    return manifest


def discover_lora_pairs(manifest: Dict[str, TensorInfo]) -> List[LoraPair]:
    grouped: Dict[str, Dict[str, str]] = {}
    alpha: Dict[str, str] = {}
    for key in manifest:
        role_base = _split_lora_key(key)
        if role_base is None:
            continue
        role, base = role_base
        if role == "alpha":
            alpha[base] = key
        else:
            grouped.setdefault(base, {})[role] = key

    pairs: List[LoraPair] = []
    for base in sorted(grouped):
        roles = grouped[base]
        if "down" not in roles or "up" not in roles:
            continue
        down = manifest[roles["down"]]
        up = manifest[roles["up"]]
        if len(down.shape) < 2 or len(up.shape) < 2:
            continue
        rank = _infer_rank(down.shape, up.shape)
        pairs.append(LoraPair(
            base_name=base,
            down_key=down.key,
            up_key=up.key,
            alpha_key=alpha.get(base),
            rank=rank,
            down_shape=down.shape,
            up_shape=up.shape,
            target_candidates=tuple(_target_candidates(base)),
        ))
    return pairs


def discover_diff_patches(manifest: Dict[str, TensorInfo]) -> List[DiffPatch]:
    """Find .diff keys (ComfyUI direct-weight-patch format) and map them to targets."""
    patches: List[DiffPatch] = []
    for key in sorted(manifest):
        if not key.endswith(".diff"):
            continue
        info = manifest[key]
        # Strip .diff suffix, append .weight → base name for candidate generation
        base = key[:-len(".diff")] + ".weight"
        # Build candidates: the base itself plus prefix-normalized variants
        candidates = tuple(_diff_target_candidates(base))
        patches.append(DiffPatch(
            diff_key=key,
            diff_shape=info.shape,
            target_candidates=candidates,
        ))
    return patches


def _diff_target_candidates(base: str) -> List[str]:
    """Generate key variants for a .diff→target mapping (prefix normalization).

    Returns candidates both with and without a trailing .weight suffix so that
    tensors like ``blocks.0.mod.lin`` (scale, no .weight) are also found.
    """
    bases = [base]
    if base.startswith("diffusion_model."):
        bases.append(base[len("diffusion_model."):])
        bases.append("model." + base)
    if base.startswith("model.diffusion_model."):
        bases.append(base[len("model."):])
    if base.startswith("base_model.model.diffusion_model."):
        bases.append(base.replace("base_model.model.diffusion_model.", "model.diffusion_model.", 1))
    out: List[str] = []
    for b in bases:
        # Emit the key as-is (covers .weight and bare keys alike)
        if b not in out:
            out.append(b)
        # Also emit with / without .weight to cover both conventions
        if b.endswith(".weight"):
            bare = b[:-len(".weight")]
            if bare not in out:
                out.append(bare)
        else:
            with_weight = b + ".weight"
            if with_weight not in out:
                out.append(with_weight)
    return out


def _split_lora_key(key: str) -> Optional[Tuple[str, str]]:
    patterns = [
        ("down", ".lora_A.weight"),
        ("up", ".lora_B.weight"),
        ("down", ".lora_down.weight"),
        ("up", ".lora_up.weight"),
        ("alpha", ".alpha"),
    ]
    for role, suffix in patterns:
        if key.endswith(suffix):
            return role, key[: -len(suffix)]
    return None


def _infer_rank(down_shape: Tuple[int, ...], up_shape: Tuple[int, ...]) -> int:
    common = set(down_shape) & set(up_shape)
    if common:
        return min(common)
    return min(min(down_shape), min(up_shape))


def _target_candidates(base: str) -> List[str]:
    bases = [base]
    krea_block_match = re.fullmatch(r"lora_unet_(blocks_\d+)_(attn|mlp)_(gate|wk|wo|wq|wv|down|up)", base)
    if krea_block_match:
        block, family, leaf = krea_block_match.groups()
        bases.append(f"{block.replace('_', '.')}.{family}.{leaf}")
    krea_simple_match = re.fullmatch(r"lora_unet_(first|last_linear|tmlp_\d+|tproj_\d+|txtmlp_\d+|txtfusion_projector)", base)
    if krea_simple_match:
        name = krea_simple_match.group(1)
        name = name.replace("last_linear", "last.linear").replace("txtfusion_projector", "txtfusion.projector")
        bases.append(name.replace("_", "."))
    krea_txtfusion_match = re.fullmatch(
        r"lora_unet_txtfusion_((?:layerwise|refiner)_blocks_\d+)_(attn|mlp)_(gate|wk|wo|wq|wv|down|up)",
        base,
    )
    if krea_txtfusion_match:
        block, family, leaf = krea_txtfusion_match.groups()
        block = block.replace("layerwise_blocks_", "layerwise_blocks.").replace("refiner_blocks_", "refiner_blocks.")
        bases.append(f"txtfusion.{block}.{family}.{leaf}")
    if base.startswith("diffusion_model."):
        bases.append("model." + base)
        # Krea 2 base checkpoints use bare keys like blocks.X.attn.gate.weight
        bare = base[len("diffusion_model."):]
        bases.append(bare)
    if base.startswith("base_model.model.transformer_blocks."):
        bases.append(base.replace("base_model.model.transformer_blocks.", "model.diffusion_model.transformer_blocks.", 1))
    if base.startswith("base_model.model.diffusion_model."):
        bases.append(base.replace("base_model.model.diffusion_model.", "model.diffusion_model.", 1))
    if base.startswith("model.transformer_blocks."):
        bases.append(base.replace("model.transformer_blocks.", "model.diffusion_model.transformer_blocks.", 1))

    out: List[str] = []
    for b in bases:
        key = b if b.endswith(".weight") else b + ".weight"
        if key not in out:
            out.append(key)
    return out
