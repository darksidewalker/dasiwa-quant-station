"""Diffusers MiniMax H3 LoRA -> ComfyUI-native H3 LoRA, streamed one module at a time.

Layout evidence (bit-exact Diffusers vs Comfy-Org H3 rows): native qkv_proj rows are [q_all; k_all; v_all],
and native SwiGLU fc1 consumes [gate, value] while Diffusers ff.net.0.proj stores [value, gate].
ComfyUI has no slice targets for H3 LoRA keys, so q/k/v fuse into A=[A_q;A_k;A_v], B=blockdiag(B_q,B_k,B_v).
No .alpha is written: alpha/rank is folded into B and ComfyUI uses scale 1.0 without alpha.
"""
import json
import os
import re
import statistics
from collections import Counter
from contextlib import ExitStack

import torch
from safetensors import safe_open

from core.adapter_factorization import factorize_additive_lora
from core.h3_prune_engine import event
from core.safetensors_stream import TensorSpool, destination, read_header
from utils.lora_inspector import discover_lora_pairs, read_safetensors_manifest

QKV = ('attn.to_q', 'attn.to_k', 'attn.to_v')
FAMILY = {'attn.to_out.0': 'attn.out_proj', 'ff.net.0.proj': 'mlp.fc1', 'ff.net.2': 'mlp.fc2'}
MODULE = re.compile(r'(?:base_model\.model\.)?(?:transformer\.)?(transformer_blocks|token_refiner\.refiner_blocks)\.(\d+)\.(.+)')
CURVE = re.compile(r'(.+)\.curve_([AB])\.weight')
NATIVE = re.compile(r'(?:(?:model\.)?diffusion_model\.)?(?:token_refiner\.)?blocks\.\d+\.|lora_unet_(?:token_refiner_)?blocks_\d+_')
FLOATS = {'F16', 'BF16', 'F32', 'F64'}
HIGH_ERROR = 0.05
LICENSE_NOTE = ('Derived from MiniMax H3 adapters: use and redistribution remain subject to the MiniMax H3 Community '
                'License and any adapter licence; this conversion grants no licence.')


def target_of(name, curve=False):
    """Diffusers module name -> (native module, q/k/v slot or None), or None when unmapped."""
    m = MODULE.fullmatch(name)
    if not m:
        return None
    stack, n, mod = m.groups()
    prefix = ('blocks.' if stack == 'transformer_blocks' else 'token_refiner.blocks.') + n
    if curve:
        return (prefix + '.adaln_proj.linear', None) if stack == 'transformer_blocks' and mod == 'adaln_proj.linear' else None
    if mod in QKV:
        return prefix + '.attn.qkv_proj', QKV.index(mod)
    return (f'{prefix}.{FAMILY[mod]}', None) if mod in FAMILY else None


def native_shapes(checkpoint):
    """Header-only {native weight name: shape}, accepting bare, diffusion_model. or model.diffusion_model. keys."""
    header, _ = read_header(checkpoint)
    header.pop('__metadata__', None)
    return {re.sub(r'^(?:model\.)?diffusion_model\.', '', k): v['shape'] for k, v in header.items()}


def plan(lora, curve, shapes):
    """Return ({target: (parts, rank, out, width)}, errors, skipped); parts are (path, A, B, alpha, A shape, B shape)."""
    groups, errors, skipped = {}, [], []

    def add(target, slot, part, label):
        if slot in groups.setdefault(target, {}):
            errors.append(f'duplicate update for {target}: {label}')
        else:
            groups[target][slot] = part

    manifest = read_safetensors_manifest(lora)
    native = sorted(k for k in manifest if NATIVE.match(k))
    if native:
        raise ValueError(f'Input is already ComfyUI-native, nothing to convert (e.g. {native[0]})')
    consumed = set()
    for pair in discover_lora_pairs(manifest):
        consumed.update(k for k in (pair.down_key, pair.up_key, pair.alpha_key) if k)
        hit = target_of(pair.base_name) if pair.kind == 'lora' else None
        if not hit:
            errors.append(f'unsupported module: {pair.base_name} ({pair.kind})')
        elif len(pair.down_shape) != 2 or len(pair.up_shape) != 2 or pair.up_shape[1] != pair.down_shape[0]:
            errors.append(f'incompatible factors: {pair.base_name} A{list(pair.down_shape)} B{list(pair.up_shape)}')
        else:
            add(*hit, (lora, pair.down_key, pair.up_key, pair.alpha_key, pair.down_shape, pair.up_shape), pair.base_name)
    errors += [f'unhandled tensor (orphan factor or unknown key): {k}' for k in sorted(set(manifest) - consumed)]
    errors += [f'unsupported dtype {i.dtype}: {k}' for k, i in sorted(manifest.items()) if i.dtype not in FLOATS]
    if curve:
        if shapes is None:
            raise ValueError('Curve branches need the target checkpoint: they apply only to curve-form checkpoints (adaln_t_table)')
        if 'adaln_t_table' not in shapes:
            skipped.append('curve branches: target checkpoint has no adaln_t_table (not curve-form), '
                           'so blocks.N.adaln_proj.linear does not take the 8-D curve input')
        else:
            sides = {}
            for key, info in sorted(read_safetensors_manifest(curve).items()):
                m = CURVE.fullmatch(key)
                if not m or not target_of(m[1], curve=True) or info.dtype not in FLOATS or len(info.shape) != 2:
                    errors.append(f'unsupported curve tensor: {key}')
                else:
                    sides.setdefault(m[1], {})[m[2]] = (key, info.shape)
            for base, s in sorted(sides.items()):
                if set(s) != {'A', 'B'} or s['B'][1][1] != s['A'][1][0]:
                    errors.append(f'curve branch needs matching curve_A and curve_B: {base}')
                else:
                    add(*target_of(base, curve=True), (curve, s['A'][0], s['B'][0], None, s['A'][1], s['B'][1]), base)
    layout = {}
    for target, slots in sorted(groups.items()):
        qkv = target.endswith('qkv_proj')
        if qkv and set(slots) != {0, 1, 2}:
            errors.append(f'{target}: missing q/k/v sibling(s) {[QKV[i] for i in sorted({0, 1, 2} - set(slots))]}')
            continue
        parts = [slots[i] for i in (0, 1, 2)] if qkv else [slots[None]]
        if qkv and len({p[5][0] for p in parts}) != 1:
            errors.append(f'{target}: expected equal q/k/v output rows, got {[p[5][0] for p in parts]}')
            continue
        widths = {p[4][1] for p in parts}
        rank, out = sum(p[4][0] for p in parts), sum(p[5][0] for p in parts)
        if len(widths) != 1 or (target.endswith('mlp.fc1') and out % 2):
            errors.append(f'{target}: incompatible input widths {sorted(widths)} / output rows {out}')
            continue
        width = widths.pop()
        if shapes is not None and list(shapes.get(target + '.weight', ())) != [out, width]:
            errors.append(f'{target}: delta {[out, width]} != target checkpoint weight {shapes.get(target + ".weight")}')
            continue
        layout[target] = (parts, rank, out, width)
    return layout, errors, skipped


def build(target, parts, handles):
    """Native float32 (A, B) for one module; alpha/rank folded into B."""
    pairs = []
    for path, down, up, alpha, _, _ in parts:
        h = handles[path]
        a, b = h.get_tensor(down).float(), h.get_tensor(up).float()
        if alpha:
            value = h.get_tensor(alpha).float()
            if value.numel() != 1 or not torch.isfinite(value).all():
                raise ValueError(f'alpha must be a finite scalar: {alpha}')
            b = b * (value.item() / a.shape[0])
        if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise ValueError(f'non-finite LoRA factors: {down}')
        pairs.append((a, b))
    if len(pairs) == 3:
        return torch.cat([a for a, _ in pairs]), torch.block_diag(*[b for _, b in pairs])
    a, b = pairs[0]
    if target.endswith('mlp.fc1'):
        half = b.shape[0] // 2  # Diffusers [value, gate] -> native [gate, value]
        b = torch.cat((b[half:], b[:half]))
    return a, b


def family(target):
    return re.sub(r'^.*?\d+\.', '', target)


def svd_summary(stats, energy, max_rank):
    errors = [e for _, _, _, e in stats]
    worst = max(stats, key=lambda s: s[3])
    return {'modules': len(stats), 'median_relative_error': statistics.median(errors),
            'worst_relative_error': worst[3], 'worst_module': worst[0],
            f'over_{HIGH_ERROR}_error': sum(e > HIGH_ERROR for e in errors),
            'below_energy_target': sum(r < energy - 1e-6 for _, _, r, _ in stats),  # float32 energy rounding
            'at_rank_cap': sum(k == max_rank for _, k, _, _ in stats) if max_rank else 0,
            'ranks': {'min': min(s[1] for s in stats), 'max': max(s[1] for s in stats)}}


def run_h3_diffusers_lora(payload):
    if payload.get('architecture', 'MiniMax H3') != 'MiniMax H3':
        raise ValueError('Requires MiniMax H3 architecture')
    real = lambda key: os.path.realpath(os.path.expanduser(payload[key])) if payload.get(key) else ''
    lora, checkpoint, curve = real('adapter_path'), real('checkpoint_path'), real('curve_path')
    if not lora:
        raise ValueError('adapter_path (Diffusers H3 LoRA) is required')
    representation = payload.get('representation') or 'exact'
    energy, max_rank = float(payload.get('energy') or 0.999), int(payload.get('max_rank') or 0)
    if representation not in {'exact', 'svd'} or not 0 < energy <= 1 or max_rank < 0:
        raise ValueError('representation must be exact or svd, energy in (0, 1], max_rank >= 0')
    dry = bool(payload.get('dry_run', False))
    output = '' if dry else destination(payload, 'minimax_h3_native_lora.safetensors', [p for p in (lora, checkpoint, curve) if p])
    shapes = native_shapes(checkpoint) if checkpoint else None
    layout, errors, skipped = plan(lora, curve, shapes)
    families = Counter(family(t) for t in layout)
    ranks = {f: [r for t, (_, r, _, _) in layout.items() if family(t) == f] for f in families}
    alpha_keys = sum(bool(p[3]) for parts, *_ in layout.values() for p in parts)
    exact_bytes = sum(2 * r * (o + w) for _, r, o, w in layout.values())
    svd_bytes = sum(2 * min(r, max_rank or r) * (o + w) for _, r, o, w in layout.values())
    mapping = {'modules': len(layout), 'families': dict(families),
               'fused_rank_range': {f: [min(v), max(v)] for f, v in ranks.items()},
               'alpha_keys_folded': alpha_keys, 'target_checkpoint': 'names and shapes verified' if shapes is not None else 'not given (unverified)',
               'skipped': skipped, 'errors': errors}
    size = (f'exact output {exact_bytes / 2**30:.2f} GiB' if representation == 'exact'
            else f'SVD output <= {svd_bytes / 2**30:.2f} GiB (exact would be {exact_bytes / 2**30:.2f} GiB)')
    yield event('log', 'Diffusers -> native H3 LoRA plan\n' + json.dumps({**mapping, 'estimate': size}, indent=2) + '\n')
    if errors:
        raise ValueError(f'{len(errors)} unsupported or invalid adapter entries (listed above); nothing written')
    if not layout:
        raise ValueError('No convertible Diffusers H3 LoRA modules found')
    if dry:
        yield event('done', status='dry-run complete')
        return
    svd = representation == 'svd'
    stats, zero = [], []
    metadata = {'format': 'pt', 'h3_lora_conversion': 'diffusers_to_native', 'h3_lora_representation': representation,
                'source_adapter': os.path.basename(lora), 'license_note': LICENSE_NOTE}
    if any(t.endswith('.adaln_proj.linear') for t in layout):
        from core.h3_curve import table_sha256
        with safe_open(checkpoint, framework='pt', device='cpu') as handle:
            key = next(k for k in handle.keys() if re.sub(r'^(?:model\.)?diffusion_model\.', '', k) == 'adaln_t_table')
            metadata['adaln_coordinate_table_sha256'] = table_sha256(handle.get_tensor(key))
    with TensorSpool(output) as spool, ExitStack() as stack:
        handles = {p: stack.enter_context(safe_open(p, framework='pt', device='cpu')) for p in (lora, curve) if p}
        for index, (target, (parts, _, _, _)) in enumerate(layout.items(), 1):
            a, b = build(target, parts, handles)
            if not b.any():
                zero.append(target)
            if svd:
                a, b, report = factorize_additive_lora([(a, b, 1.0)], max_rank=max_rank, energy=energy)
                stats.append((target, report.rank, report.retained_energy, report.relative_error))
            spool.tensor(f'diffusion_model.{target}.lora_A.weight', a.to(torch.bfloat16))
            spool.tensor(f'diffusion_model.{target}.lora_B.weight', b.to(torch.bfloat16))
            if index % max(1, len(layout) // 100) == 0 or index == len(layout):
                yield event('progress', f'Converted modules: {index}/{len(layout)}')
        summary = {'written_tensors': 2 * len(layout), 'zero_delta_modules': len(zero), 'output_dtype': 'BF16'}
        if svd:
            summary['svd'] = {**svd_summary(stats, energy, max_rank), 'device': 'cpu',
                              'note': 'float32 factorization error before BF16 storage',
                              'per_module': {t: {'rank': k, 'retained_energy': r, 'relative_error': e} for t, k, r, e in stats}}
            yield event('log', 'SVD re-rank: ' + json.dumps({k: v for k, v in summary['svd'].items() if k != 'per_module'}) + '\n')
        if zero:
            yield event('log', f'Warning: {len(zero)} module(s) have an all-zero delta and change nothing (e.g. {zero[0]})\n')
        settings = {'adapter_path': lora, 'checkpoint_path': checkpoint, 'curve_path': curve, 'representation': representation,
                    'energy': energy, 'max_rank': max_rank, 'output_path': output}
        recipe = spool.publish(metadata, 'DaSiWa H3 Diffusers LoRA Conversion Recipe\n' + json.dumps(
            {'settings': settings, 'mapping': mapping, 'summary': summary, 'license_note': LICENSE_NOTE,
             'inference_validation': 'unverified'}, indent=2, allow_nan=False) + '\n')
    yield event('log', f'Wrote adapter: {output} ({os.path.getsize(output) / 2**30:.2f} GiB)\nWrote recipe: {recipe}\n')
    yield event('done', status='finished')
