"""MiniMax H3 time-curve math. Gauge identity is canonical F32 table bytes."""
import hashlib
import math
import torch
import torch.nn.functional as F
from safetensors import safe_open
from utils.lora_inspector import read_safetensors_manifest


def inspect_h3_variant(path):
    """Strict header-level full/pruned classification; small table read for identity."""
    manifest = read_safetensors_manifest(path)
    time = [k for k in manifest if any(k.endswith(s) for s in TIME_KEYS)]
    tables = [k for k in manifest if k.endswith('adaln_t_table')]
    weights = [k for k in manifest if k.endswith(ADALN_WEIGHT)]
    if not weights or (time and tables) or len(tables) > 1:
        raise ValueError('ambiguous/mixed or missing H3 structure')
    if tables:
        prefix = tables[0][:-len('adaln_t_table')]
        shape = manifest[tables[0]].shape
        if len(shape) != 2 or shape[0] < 3 or not 1 <= shape[1] <= 8 or manifest[tables[0]].dtype != 'F32':
            raise ValueError('Malformed H3 table; expected F32 [grid>=3, width<=8]')
        width = shape[1]
        variant = 'pruned'
    else:
        prefixes = {k[:-len(s)] for k in time for s in TIME_KEYS if k.endswith(s)}
        if len(prefixes) != 1:
            raise ValueError('Expected exactly one full H3 time embedder')
        prefix = prefixes.pop()
        if any(prefix+k not in manifest for k in TIME_KEYS):
            raise ValueError('Missing full H3 time quartet')
        w1,b1,w2,b2 = [manifest[prefix+k] for k in TIME_KEYS]
        if len(w1.shape)!=2 or w1.shape[1]%2 or w1.shape[1]<2 or b1.shape!=(w1.shape[0],) or len(w2.shape)!=2 or w2.shape[1]!=w1.shape[0] or b2.shape!=(w2.shape[0],):
            raise ValueError('Malformed H3 time embedder shapes')
        width = w2.shape[0]
        variant = 'full'
    structural = weights + [k[:-6]+'bias' for k in weights] + time + tables
    for k in structural:
        if k not in manifest:
            raise ValueError(f'Missing AdaLN paired bias: {k}')
        if manifest[k].dtype not in {'F16','BF16','F32'}:
            raise ValueError(f'Quantized/unsupported H3 structural dtype: {k}')
        if not k.startswith(prefix):
            raise ValueError('mixed H3 prefixes')
    for k in weights:
        if len(manifest[k].shape)!=2 or manifest[k].shape[1]!=width or manifest[k[:-6]+'bias'].shape!=(manifest[k].shape[0],):
            raise ValueError(f'Malformed AdaLN projection shape: {k}')
    if any('.comfy_quant' in k or 'weight_scale' in k or 'weight_s_rel' in k for k in manifest) or any(v.dtype not in {'F16','BF16','F32','F64'} for k,v in manifest.items() if k.endswith('.weight')):
        raise ValueError('Quantized/packed H3 source is unsupported; use original floating checkpoint')
    with safe_open(path, framework='pt', device='cpu') as h:
        metadata = h.metadata() or {}
        if metadata.get('_quantization_metadata'):
            raise ValueError('Quantized H3 source is unsupported')
        identity = table_sha256(h.get_tensor(tables[0])) if tables else ''
    return {'variant': variant, 'prefix': prefix, 'source_width': width,
            'adaln_coordinate_table_sha256': identity, 'source_dtype': sorted({manifest[k].dtype for k in weights}),
            'adaln_weights': weights, 'quantized': False}

TIME_KEYS = ('time_embedder.proj_in.weight', 'time_embedder.proj_in.bias',
             'time_embedder.proj_out.weight', 'time_embedder.proj_out.bias')
ADALN_WEIGHT = 'adaln_proj.linear.weight'


def finite(tensor, label='tensor'):
    if not torch.isfinite(tensor).all():
        raise ValueError(f'{label} contains nonfinite values (NaN or infinity)')
    return tensor


def relative_error(actual, expected):
    return float(torch.linalg.vector_norm(actual-expected) / torch.linalg.vector_norm(expected).clamp_min(1e-30))


def interpolate_table(table, t):
    position = t.double().clamp(0, 1)*(len(table)-1)
    lo = position.floor().long().clamp(max=len(table)-2)
    frac = (position-lo).unsqueeze(1)
    return table.double()[lo]*(1-frac) + table.double()[lo+1]*frac


def recover_gauge(base_path, target_path, *, fit_tolerance=0.005, consistency_tolerance=0.02):
    full = inspect_h3_variant(base_path)
    target = inspect_h3_variant(target_path)
    if full['variant'] != 'full' or target['variant'] != 'pruned':
        raise ValueError('Gauge recovery requires a full source and exact pruned target')
    p,q = full['prefix'],target['prefix']
    with safe_open(base_path, framework='pt', device='cpu') as a, safe_open(target_path, framework='pt', device='cpu') as b:
        tensors = [a.get_tensor(p+k) for k in TIME_KEYS]
        table = finite(b.get_tensor(q+'adaln_t_table'), 'table')
        curve = time_curve(tensors, torch.linspace(0,1,len(table),dtype=torch.float64))
        design = torch.cat((table.double(), torch.ones(len(table),1,dtype=torch.float64)),1)
        # Normalize columns before checking rank; gauges can have very different scales.
        norms = torch.linalg.vector_norm(design,dim=0)
        scaled = design / norms.clamp_min(1e-30)
        singular = torch.linalg.svdvals(scaled)
        if singular[-1] <= singular[0]*1e-12:
            raise ValueError('Reference table affine design is rank deficient/ill-conditioned')
        affine = torch.linalg.lstsq(scaled,curve,driver='gelsd',rcond=1e-12).solution / norms[:,None]
        basis,center = affine[:-1].T.contiguous(),affine[-1].contiguous()
        residual = relative_error(design@affine,curve)
        if residual > fit_tolerance:
            raise ValueError(f'Unrelated reference gauge: curve residual {residual:.3e} exceeds {fit_tolerance}')
        t = (torch.arange(129,dtype=torch.float64)+.37)/129
        off_curve = time_curve(tensors,t)
        approx = interpolate_table(table,t)@basis.T+center
        interpolation_error = relative_error(approx,off_curve)
        if interpolation_error > fit_tolerance:
            raise ValueError(f'Off-grid interpolation residual {interpolation_error:.3e} exceeds tolerance')
        worst = 0.
        target_manifest = read_safetensors_manifest(target_path)
        # Bounded first/last rows of EVERY AdaLN including final; do not trust table fit alone.
        for key in full['adaln_weights']:
            tk = q+key[len(p):]
            bk = key[:-6]+'bias'
            tbk = tk[:-6]+'bias'
            if tk not in target_manifest or target_manifest[tk].shape[0] != read_safetensors_manifest(base_path)[key].shape[0]:
                raise ValueError(f'Source-target consistency: missing/incompatible {tk}')
            rows = target_manifest[tk].shape[0]
            indices = sorted(set([0,rows//2,rows-1]))
            for row in indices:
                w = finite(a.get_slice(key)[row:row+1].double(),key)
                bias = finite(a.get_slice(bk)[row:row+1].double(),bk)
                tw = finite(b.get_slice(tk)[row:row+1].double(),tk)
                bias_target = finite(b.get_slice(tbk)[row:row+1].double(),tbk)
                err = relative_error(interpolate_table(table,t)@tw.T + bias_target, off_curve@w.T + bias)
                worst = max(worst,err)
                if err > consistency_tolerance:
                    raise ValueError(f'Source-target AdaLN consistency failed for {key} row {row}: {err:.3e}')
    return {'basis':basis,'center':center,'table':table,'table_sha256':table_sha256(table),
            'fit_relative_error':residual,'interpolation_relative_error':interpolation_error,
            'consistency_relative_error':worst,'condition_number':float(singular[0]/singular[-1]),
            'source_prefix':p,'target_prefix':q}


def table_sha256(table):
    return hashlib.sha256(finite(table, 'AdaLN table').detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def time_curve(tensors, t):
    w1, b1, w2, b2 = [finite(x, 'time embedder').to(dtype=torch.float64, device='cpu') for x in tensors]
    t = t.to(dtype=torch.float64, device='cpu')
    if w1.ndim != 2 or w1.shape[1] < 2 or w1.shape[1] % 2 or b1.shape != (w1.shape[0],) or w2.ndim != 2 or w2.shape[1] != w1.shape[0] or b2.shape != (w2.shape[0],):
        raise ValueError('Malformed H3 time embedder shapes')
    half = w1.shape[1] // 2
    freqs = torch.exp(-math.log(10000.) * torch.arange(half, dtype=torch.float64) / half)
    emb = torch.cat(((t[:, None] * freqs).cos(), (t[:, None] * freqs).sin()), dim=1)
    return finite(F.silu(F.linear(F.silu(F.linear(emb, w1, b1)), w2, b2)), 'time curve')
