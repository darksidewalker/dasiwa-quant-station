"""Full-width H3 adapters to an exact reference-pruned affine coordinate system."""
import os
import torch
from safetensors import safe_open
from core.h3_device import H3DevicePolicy
from core.h3_curve import recover_gauge, finite, TIME_KEYS
from core.h3_prune_engine import require_h3, event, numeric_recipe
from core.safetensors_stream import TensorSpool, destination, read_header
from utils.lora_inspector import read_safetensors_manifest, discover_lora_pairs, discover_diff_patches


def normalize(key):
    for prefix in ('base_model.model.diffusion_model.', 'model.diffusion_model.', 'diffusion_model.'):
        if key.startswith(prefix): return key[len(prefix):]
    return key


def run_h3_adapter_convert(payload):
    require_h3(payload)
    base, target, adapter = [os.path.realpath(os.path.expanduser(payload[k])) for k in ('base_path', 'pruned_target_path', 'adapter_path')]
    output = destination(payload, 'minimax_h3_pruned_adapter.safetensors', (base, target, adapter))
    header, data_start = read_header(adapter)
    manifest = read_safetensors_manifest(adapter)
    source_manifest = {normalize(k): v for k, v in read_safetensors_manifest(base).items()}
    target_manifest = {normalize(k): v for k, v in read_safetensors_manifest(target).items()}
    groups, consumed = {}, set()
    def resolve(candidates):
        found = {normalize(k) for k in candidates if normalize(k) in source_manifest}
        if len(found) != 1: raise ValueError(f'Unknown/ambiguous adapter target: {candidates}')
        key = found.pop()
        if key.startswith('time_embedder.') or 'adaln_t_table' in key:
            raise ValueError('Time-changing adapters require a paired-checkpoint fold, not reusable adapter conversion')
        if key not in target_manifest: raise ValueError(f'Missing pruned target: {key}')
        return key
    for pair in discover_lora_pairs(manifest):
        if pair.kind != 'lora' or len(pair.down_shape) != 2 or len(pair.up_shape) != 2:
            raise ValueError('Conversion supports standard 2D LoRA factors, not LoKr or convolution adapters')
        key = resolve(pair.target_candidates)
        module = key[:-len('.weight')]
        group = groups.setdefault(module, {})
        if 'pair' in group or 'diff' in group: raise ValueError(f'Duplicate weight update: {module}')
        if pair.rank <= 0 or pair.up_shape[1] != pair.down_shape[0] or source_manifest[key].shape != (pair.up_shape[0], pair.down_shape[1]):
            raise ValueError(f'Incompatible LoRA shapes: {module}')
        if not key.endswith('adaln_proj.linear.weight') and source_manifest[key].shape != target_manifest[key].shape:
            raise ValueError(f'Non-AdaLN target shape mismatch: {module}')
        group['pair'] = pair
        consumed.update((pair.down_key, pair.up_key))
        if pair.alpha_key: consumed.add(pair.alpha_key)
    for patch in discover_diff_patches(manifest):
        key = resolve(patch.target_candidates)
        if not key.endswith(('.weight', '.bias')): raise ValueError(f'Unsupported buffer patch: {key}')
        if source_manifest[key].shape != patch.diff_shape: raise ValueError(f'Incompatible patch shape: {key}')
        module = key.rsplit('.', 1)[0]
        role = 'bias' if key.endswith('.bias') else 'diff'
        group = groups.setdefault(module, {})
        if role in group or (role == 'diff' and 'pair' in group): raise ValueError(f'Duplicate update: {module}')
        group[role] = patch.diff_key
        consumed.add(patch.diff_key)
    unhandled = set(manifest) - consumed
    if unhandled: raise ValueError(f'Unsupported/unhandled adapter tensors: {sorted(unhandled)}')
    if not groups: raise ValueError('No supported adapter contributors')
    for key, info in manifest.items():
        if info.dtype not in {'F16', 'BF16', 'F32', 'F64'}:
            raise ValueError(f'Unsupported adapter dtype: {key}')
    yield event('log', 'Validating exact source/reference gauge and affine offsets on CPU')
    gauge = recover_gauge(base, target)
    summary = {'converted': 0, 'copied': 0, 'consumed': len(consumed), 'unsupported': 0, 'written': 0, 'devices': {}}
    policy = H3DevicePolicy(payload)
    summary['devices'] = policy.summary()
    yield event('log', f"H3 row policy requested={policy.requested}, CUDA device={policy.cuda_device}; gauge validation stays CPU double")
    if payload.get('dry_run', False):
        yield event('log', policy.log())
        yield event('log', numeric_recipe('H3 adapter compatibility audit', payload, gauge, summary))
        yield event('done', status='dry-run complete'); return
    rows = max(1, min(int(payload.get('row_chunk_size', 256)), 256))
    basis, center = gauge['basis'], gauge['center']
    metadata = {k: v for k, v in header.get('__metadata__', {}).items() if not k.startswith('civitai.hash.') and k not in {'modelspec.hash_sha256', '_quantization_metadata'}}
    metadata.update(adaln_coordinate_table_sha256=gauge['table_sha256'], h3_adapter_conversion='affine_reference', format='pt')
    with TensorSpool(output) as spool, safe_open(adapter, framework='pt', device='cpu') as handle:
        for index, (module, group) in enumerate(sorted(groups.items()), 1):
            folded = module.endswith('adaln_proj.linear')
            pair = group.get('pair')
            bias_key = group.get('bias')
            diff_key = group.get('diff')
            down_center = None
            alpha_scale = 1.
            if pair:
                if pair.alpha_key:
                    alpha = finite(handle.get_tensor(pair.alpha_key), pair.alpha_key)
                    if alpha.numel() != 1: raise ValueError('LoRA alpha must be scalar')
                    alpha_scale = float(alpha.item()) / pair.rank
                    spool.tensor(module + '.alpha', alpha.reshape(()))
                    summary['written'] += 1
                down_center = torch.empty(pair.rank, dtype=torch.float64) if folded else None
                def down_chunks():
                    for lo in range(0, pair.rank, rows):
                        down = finite(handle.get_slice(pair.down_key)[lo:lo + rows], pair.down_key)
                        if folded:
                            # Project A and its affine center together, without forming B@A.
                            result = policy.run(lambda a, v, c: torch.cat((a @ v, (a @ c)[:, None]), dim=1),
                                                (down, basis, center), down.shape[0] * (basis.shape[1] + 1))
                            down_center[lo:lo + down.shape[0]] = finite(result[:, -1], module)
                            yield finite(result[:, :-1].to(down.dtype), module)
                        else:
                            yield down
                shape = [pair.rank, basis.shape[1]] if folded else list(pair.down_shape)
                spool.chunks(module + '.lora_A.weight', header[pair.down_key]['dtype'], shape, down_chunks())
                for lo in range(0, pair.up_shape[0], rows):
                    finite(handle.get_slice(pair.up_key)[lo:lo + rows], pair.up_key)
                spool.copy(module + '.lora_B.weight', adapter, header[pair.up_key], data_start)
                summary['written'] += 2
            elif diff_key:
                info = header[diff_key]
                def diff_chunks():
                    for lo in range(0, info['shape'][0], rows):
                        w = finite(handle.get_slice(diff_key)[lo:lo + rows], diff_key)
                        yield finite(policy.run(lambda w, v: w @ v, (w, basis), w.shape[0] * basis.shape[1]).to(w.dtype), diff_key) if folded else w
                shape = [info['shape'][0], basis.shape[1]] if folded else info['shape']
                spool.chunks(module + '.diff', info['dtype'], shape, diff_chunks())
                summary['written'] += 1
            if bias_key or (folded and (pair or diff_key)):
                out_rows = source_manifest[module + '.bias'].shape[0]
                def bias_chunks():
                    for lo in range(0, out_rows, rows):
                        count = min(rows, out_rows - lo)
                        offset = torch.zeros(count, dtype=torch.float64)
                        if bias_key: offset += finite(handle.get_slice(bias_key)[lo:lo + rows], bias_key).double()
                        if folded and pair:
                            up = finite(handle.get_slice(pair.up_key)[lo:lo + rows], pair.up_key).double()
                            offset += policy.run(lambda b, ac: alpha_scale * (b @ ac), (up, down_center), count)
                        elif folded and diff_key:
                            w = finite(handle.get_slice(diff_key)[lo:lo + rows], diff_key)
                            offset += policy.run(lambda w, c: w @ c, (w, center), count)
                        yield finite(offset.float(), module + '.diff_b')
                spool.chunks(module + '.diff_b', 'F32', [out_rows], bias_chunks())
                summary['written'] += 1
            summary['converted' if folded else 'copied'] += 1
            if index % max(1, len(groups) // 100) == 0:
                yield event('progress', f'Converted adapter modules: {index}/{len(groups)}')
        effective = dict(payload, base_path=base, pruned_target_path=target, adapter_path=adapter, output_path=output)
        summary['devices'] = policy.summary()
        yield event('log', policy.log())
        recipe = spool.publish(metadata, numeric_recipe('DaSiWa H3 Adapter Conversion Recipe', effective, gauge, summary))
    yield event('log', f'Wrote adapter: {output}\nWrote recipe: {recipe}')
    yield event('done', status='finished')
