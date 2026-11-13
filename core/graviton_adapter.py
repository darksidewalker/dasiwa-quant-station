"""GRAVITON in logical adapter-delta space; checkpoint geometry needs a real base."""
import re

import torch

from core.model_merge_engine import _graviton_merge_matrix


BAKE_SEMANTICS = ('GRAVITON: sum signed effective deltas, B=base+sum, t=1, GD-auto rank; '
                  'polar on trunk matrices, spectral-linear on conditioning, additive on non-2D.')
COMPOSE_SEMANTICS = ('GRAVITON: base-free spectral sum; GD-auto rank; no polar checkpoint geometry. '
                     'Non-2D biases remain additive. Error includes spectral filtering and storage/compression loss.')


def validate_pairs(pairs):
    for pair in pairs:
        if pair.kind == "lokr_decomposed":
            raise ValueError(f"GRAVITON factorized LoKr input is unsupported: {pair.base_name}")
        down, up = pair.down_shape, pair.up_shape
        if (len(down) != 2 or len(up) != 2 or min(*down, *up) <= 0 or
                (pair.kind == "lora" and down[0] != up[1])):
            raise ValueError(f"GRAVITON incompatible factor shapes: {pair.base_name}: {down}, {up}")


def preserved_key(key, architecture, profile_preserved):
    from core.layer_config_builder import PRESERVE_PATTERNS, BAKED_VAE_PATTERNS
    from core.metadata_configs import MODEL_METADATA_CONFIGS
    if architecture not in MODEL_METADATA_CONFIGS:
        raise ValueError(f'unsupported checkpoint architecture: {architecture}')
    # Existing dedicated bake profiles remain authoritative where present.
    if architecture in {'LTX-2.3', 'WAN 2.2', 'Krea 2', 'MiniMax H3'}:
        return profile_preserved(key)
    return any(re.search(pattern, key) for pattern in
               list(PRESERVE_PATTERNS.get(architecture, [])) + list(BAKED_VAE_PATTERNS))


def spectral_target(base, delta, *, polar):
    if not torch.isfinite(base).all() or not torch.isfinite(delta).all():
        raise ValueError('GRAVITON requires finite base weights and effective deltas')
    if delta.ndim != 2 or not delta.numel() or not torch.count_nonzero(delta):
        return base + delta
    return _graviton_merge_matrix(base, base + delta, 1., -1, polar=polar)[0]


def bake_target(base, ops, handles, device):
    from core.lora_merge_engine import _materialize_effective_delta
    original = base.to(device=device, dtype=torch.float32)
    delta = torch.zeros_like(original)
    # Stable accumulation makes the adapter selection order irrelevant.
    for op in sorted(ops, key=lambda op: (op['lora_path'], op.get('down_key', op.get('diff_key', '')), op['scale'])):
        if op['scale']:
            delta += _materialize_effective_delta(op, handles, tuple(base.shape), device)
    if not torch.count_nonzero(delta):
        return base.detach().cpu()
    key = ops[0]['target_key']
    conditioning = re.search(r'ada|modulat|norm|time|embed|caption|condition|rope|bias|gate|refiner|txtfusion', key, re.I)
    merged = spectral_target(original, delta, polar=not conditioning)
    if not torch.isfinite(merged).all():
        raise ValueError(f'GRAVITON produced non-finite weights: {key}')
    result = merged.to(device='cpu', dtype=base.dtype)
    if not torch.isfinite(result).all():
        raise ValueError(f'GRAVITON overflow in source dtype: {key}')
    return result


def compose_layer(contributors, handles, device, actual, anchor, cap, energy):
    from core.lora_compose_engine import _full_delta, _serialized_report
    from core.adapter_factorization import factorize_lora, factorize_lokr, FactorizationReport
    from core.consensus_merge import ConsensusStats
    delta = torch.zeros(contributors[0]['shape'], device=device, dtype=torch.float32)
    for item in sorted(contributors, key=lambda item: (item['path'], item.get('down_key', item.get('diff_key', '')), item['scale'])):
        if item['scale']:
            delta += _full_delta(handles[item['path']], item, device)
    merged = spectral_target(torch.zeros_like(delta), delta, polar=False).cpu()
    if actual == 'bias':
        tensors = {'.diff_b': merged}
        report = FactorizationReport('bias', 0, 1., 0.)
    elif actual == 'lokr':
        w1, w2, report = factorize_lokr(merged, anchor[0], anchor[1])
        tensors = {'.lokr_w1': w1.to(torch.bfloat16), '.lokr_w2': w2.to(torch.bfloat16)}
    else:
        down, up, report = factorize_lora(merged, max_rank=cap, energy=energy)
        tensors = {'.lora_A.weight': down.to(torch.bfloat16), '.lora_B.weight': up.to(torch.bfloat16)}
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError('GRAVITON produced non-finite adapter factors')
    # Compare against the original weighted sum, including spectral filtering loss.
    return tensors, _serialized_report(contributors, handles, 'additive', None, tensors, report), ConsensusStats()
