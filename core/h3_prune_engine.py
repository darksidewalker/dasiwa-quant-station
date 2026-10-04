"""Streamed, explicit MiniMax H3 checkpoint pruning."""
import json
import os
import torch
from safetensors import safe_open
from core.h3_device import H3DevicePolicy
from core.h3_curve import TIME_KEYS, inspect_h3_variant, recover_gauge, finite, time_curve, table_sha256, relative_error, interpolate_table
from core.safetensors_stream import TensorSpool, destination, read_header


def event(kind,text='',status=''):
    result={'type':kind}
    if text: result['text']=text
    if status: result['status']=status
    return result


def require_h3(payload):
    if payload.get('architecture','MiniMax H3')!='MiniMax H3': raise ValueError('Requires MiniMax H3 architecture')
    if payload.get('merge_device','auto') not in {'auto','cpu','cuda'}: raise ValueError('merge_device must be auto, cpu, or cuda')
    import re
    cuda_device = str(payload.get('cuda_device', 'cuda:0'))
    if int(payload.get('vram_headroom_mb', 1024)) < 0 or not re.fullmatch(r'cuda|(?:cuda:)?\d+', cuda_device):
        raise ValueError('Invalid CUDA device/headroom')
    if payload.get('fold_mode','reference') not in {'reference','independent'}: raise ValueError('fold_mode must be reference or independent')


def numeric_recipe(title,payload,gauge,summary):
    detail={k:v for k,v in gauge.items() if not isinstance(v,torch.Tensor)}
    return title+'\n'+json.dumps({'settings':payload,'gauge':detail,'summary':summary,'inference_validation':'unverified'},indent=2,allow_nan=False,default=str)+'\n'


def run_h3_prune(payload):
    require_h3(payload)
    base=os.path.realpath(os.path.expanduser(payload['base_path']))
    reference=payload.get('reference_path')
    inputs=[base]+([reference] if reference else [])
    out=destination(payload,'minimax_h3_pruned.safetensors',inputs)
    header,start=read_header(base)
    variant=inspect_h3_variant(base)
    if variant['variant']!='full': raise ValueError('Pruning requires an original full floating H3 checkpoint')
    mode = payload.get('fold_mode', 'reference')
    if mode == 'reference':
        if not reference: raise ValueError('reference_path required for compatible reference fold')
        yield event('log','Preparing compatible reference gauge on CPU (bounded AdaLN row samples)')
        gauge = recover_gauge(base, reference)
    else:
        yield event('log', 'Preparing independent CPU SVD gauge; existing pruned adapters are NOT automatically compatible')
        with safe_open(base, framework='pt', device='cpu') as source:
            tensors = [source.get_tensor(variant['prefix'] + k) for k in TIME_KEYS]
        curve = time_curve(tensors, torch.linspace(0, 1, 1025, dtype=torch.float64))
        u, singular, vh = torch.linalg.svd(curve, full_matrices=False)
        rank = min(8, len(singular))
        basis = vh[:rank].T.contiguous()
        table = (u[:, :rank] * singular[:rank]).float().contiguous()
        center = torch.zeros(curve.shape[1], dtype=torch.float64)
        t = (torch.arange(129, dtype=torch.float64) + .37) / 129
        gauge = {'basis': basis, 'center': center, 'table': table,
                 'table_sha256': table_sha256(table), 'basis_values': basis.tolist(),
                 'ecosystem_compatible': False, 'svd_dtype': 'F64', 'svd_device': 'cpu',
                 'fit_relative_error': relative_error(table.double() @ basis.T, curve),
                 'interpolation_relative_error': relative_error(interpolate_table(table, t) @ basis.T, time_curve(tensors, t))}
        if gauge['fit_relative_error'] > .005 or gauge['interpolation_relative_error'] > .005:
            raise ValueError('Independent rank-8 fold exceeds numerical curve tolerance')
    prefix=variant['prefix']; basis=gauge['basis']; center=gauge['center']
    summary={'copied':0,'folded':0,'written':0,'devices': {}}
    policy = H3DevicePolicy(payload)
    summary['devices'] = policy.summary()
    yield event('log', f"H3 row policy requested={policy.requested}, CUDA device={policy.cuda_device}; gauge validation stays CPU double")
    effective=dict(payload,base_path=base,reference_path=os.path.realpath(reference) if reference else '',output_path=out)
    yield event('log',numeric_recipe('Gauge numerical audit (not inference proof)',{},gauge,{}))
    if payload.get('dry_run',False):
        yield event('log', policy.log())
        yield event('done',status='dry-run complete'); return
    rows=max(1,min(int(payload.get('row_chunk_size', 256)), 256))
    metadata={k:v for k,v in header.get('__metadata__',{}).items() if not k.startswith('civitai.hash.') and k not in {'modelspec.hash_sha256','_quantization_metadata'}}
    if 'config' in metadata:
        config=json.loads(metadata['config'])
        config.setdefault('transformer',{}).update(adaln_curve_grid=len(gauge['table']),time_embed_dim=basis.shape[1])
        metadata['config']=json.dumps(config,separators=(',',':'))
    metadata.update(format='pt',adaln_coordinate_table_sha256=gauge['table_sha256'],h3_fold_mode=mode)
    keys=[k for k in header if k!='__metadata__' and k not in {prefix+t for t in TIME_KEYS}]
    with TensorSpool(out) as spool, safe_open(base,framework='pt',device='cpu') as handle:
        spool.tensor(prefix+'adaln_t_table',gauge['table'])
        for i,key in enumerate(keys,1):
            info=header[key]
            if key.endswith('adaln_proj.linear.weight'):
                shape=[info['shape'][0],basis.shape[1]]
                def weight_chunks():
                    for lo in range(0,shape[0],rows):
                        w=finite(handle.get_slice(key)[lo:lo+rows],key)
                        yield finite(policy.run(lambda w, v: w @ v, (w, basis), w.shape[0] * basis.shape[1]).to(w.dtype),key)
                spool.chunks(key,info['dtype'],shape,weight_chunks()); summary['folded']+=1
            elif key.endswith('adaln_proj.linear.bias'):
                wk=key[:-4]+'weight'
                def bias_chunks():
                    for lo in range(0,info['shape'][0],rows):
                        b=finite(handle.get_slice(key)[lo:lo+rows],key)
                        w=finite(handle.get_slice(wk)[lo:lo+rows],wk)
                        yield finite(policy.run(lambda b, w, c: b + w @ c, (b, w, center), b.numel()).to(b.dtype),key)
                spool.chunks(key,info['dtype'],info['shape'],bias_chunks()); summary['folded']+=1
            else:
                spool.copy(key,base,info,start); summary['copied']+=1
            summary['written']+=1
            if i%max(1,len(keys)//100)==0 or i==len(keys): yield event('progress',f'Pruning tensors: {i}/{len(keys)}')
        summary['written']+=1
        summary['devices'] = policy.summary()
        yield event('log', policy.log())
        recipe=spool.publish(metadata,numeric_recipe('DaSiWa H3 Prune Recipe',effective,gauge,summary))
    yield event('log',f'Wrote checkpoint: {out}\nWrote recipe: {recipe}')
    yield event('done',status='finished')
