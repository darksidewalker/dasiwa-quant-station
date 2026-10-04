"""Bounded safetensors spool with exclusive, durable artifact+recipe publication.

Only header descriptors live in memory. Tensor chunks are immediately spooled.
Final and recipe are staged in the destination filesystem; publication is no-clobber.
"""
import json
import math
import os
import shutil
import struct
import tempfile
from pathlib import Path
import torch
from safetensors import safe_open

DTYPES = {torch.float32:'F32',torch.float64:'F64',torch.float16:'F16',torch.bfloat16:'BF16',torch.uint8:'U8',torch.int8:'I8',torch.int16:'I16',torch.int32:'I32',torch.int64:'I64',torch.bool:'BOOL'}
SIZES = {'F64':8,'F32':4,'F16':2,'BF16':2,'I64':8,'I32':4,'I16':2,'U8':1,'I8':1,'BOOL':1,'F8_E4M3':1,'F8_E5M2':1,'U16':2,'U32':4,'U64':8}


def read_header(path):
    size = os.path.getsize(path)
    with open(path,'rb') as f:
        prefix = f.read(8)
        if len(prefix)!=8: raise ValueError('Truncated safetensors header')
        length = struct.unpack('<Q',prefix)[0]
        if length < 2 or length > min(size-8,100_000_000): raise ValueError('Invalid safetensors header size')
        def unique(pairs):
            result = {}
            for k,v in pairs:
                if k in result: raise ValueError(f'Duplicate header key: {k}')
                result[k]=v
            return result
        header = json.loads(f.read(length),object_pairs_hook=unique)
    cursor=0
    for k,v in sorted(((k,v) for k,v in header.items() if k!='__metadata__'),key=lambda kv:kv[1]['data_offsets']):
        shape=v['shape']; start,end=v['data_offsets']
        if v['dtype'] not in SIZES or any(not isinstance(n,int) or n<0 for n in shape) or start!=cursor or end-start!=math.prod(shape)*SIZES[v['dtype']]:
            raise ValueError(f'Invalid tensor offsets/shape/dtype: {k}')
        cursor=end
    if cursor+8+length!=size: raise ValueError('Invalid safetensors file length')
    return header,8+length


def destination(payload, default_name, inputs=()):
    output = payload.get('output_path') or os.path.join(payload.get('output_dir') or os.path.dirname(inputs[0]),payload.get('output_name') or default_name)
    output=os.path.realpath(os.path.expanduser(output))
    if not output.lower().endswith('.safetensors'): output+='.safetensors'
    for path in inputs:
        if output==os.path.realpath(path) or (os.path.exists(output) and os.path.samefile(output,path)):
            raise ValueError('Output must not alias an input')
    if os.path.lexists(output) or os.path.lexists(output+'.txt'):
        raise FileExistsError(f'Refusing to overwrite existing output/recipe: {output}')
    return output


class TensorSpool:
    def __init__(self, output):
        self.output=output
        os.makedirs(os.path.dirname(output),exist_ok=True)
        # The job runner owns this exact private root and removes it after a killed
        # child exits. Never sweep other staging directories in the output folder.
        stage_root = os.environ.get('DASIWA_H3_STAGE_DIR') or os.path.dirname(output)
        self.temp=tempfile.TemporaryDirectory(prefix='.h3_stage_',dir=stage_root)
        self.data=open(os.path.join(self.temp.name,'payload'),'w+b')
        self.header={}

    def __enter__(self): return self

    def __exit__(self,*exc):
        self.data.close()
        self.temp.cleanup()

    def tensor(self,key,tensor):
        tensor=tensor.detach().cpu().contiguous()
        self.chunks(key,DTYPES[tensor.dtype],list(tensor.shape),(tensor,))

    def chunks(self,key,dtype,shape,chunks):
        if key in self.header: raise ValueError(f'Duplicate output tensor: {key}')
        start=self.data.tell()
        for tensor in chunks:
            tensor=tensor.detach().cpu().contiguous()
            if DTYPES.get(tensor.dtype)!=dtype: raise ValueError('Output chunk dtype mismatch')
            if tensor.is_floating_point() and not torch.isfinite(tensor).all(): raise ValueError(f'Nonfinite/overflow output: {key}')
            self.data.write(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        end=self.data.tell()
        if end-start!=math.prod(shape)*SIZES[dtype]: raise ValueError(f'Output tensor size mismatch: {key}')
        self.header[key]={'dtype':dtype,'shape':list(shape),'data_offsets':[start,end]}

    def copy(self,key,path,info,data_start):
        if key in self.header: raise ValueError(f'Duplicate output tensor: {key}')
        start=self.data.tell()
        lo,hi=info['data_offsets']
        with open(path,'rb') as source:
            source.seek(data_start+lo)
            remaining=hi-lo
            while remaining:
                data=source.read(min(remaining,8*1024*1024))
                if not data: raise ValueError(f'Truncated source tensor {key}')
                self.data.write(data); remaining-=len(data)
        self.header[key]={'dtype':info['dtype'],'shape':info['shape'],'data_offsets':[start,self.data.tell()]}

    def publish(self,metadata,recipe):
        header=dict(self.header)
        header['__metadata__']={str(k):str(v) for k,v in metadata.items()}
        raw=json.dumps(header,separators=(',',':'),allow_nan=False).encode()
        raw+=b' '*((-len(raw))%8)
        staged=os.path.join(self.temp.name,'artifact.safetensors')
        with open(staged,'wb') as out:
            out.write(struct.pack('<Q',len(raw))); out.write(raw)
            self.data.flush(); self.data.seek(0)
            shutil.copyfileobj(self.data,out,8*1024*1024)
            out.flush(); os.fsync(out.fileno())
        audited,_=read_header(staged)
        if audited!=header: raise ValueError('Output header audit failed')
        with safe_open(staged,framework='pt',device='cpu') as h:
            if set(h.keys())!=set(self.header): raise ValueError('Output key audit failed')
        side=os.path.join(self.temp.name,'recipe.txt')
        with open(side,'w',encoding='utf-8') as out:
            out.write(recipe); out.flush(); os.fsync(out.fileno())
        # Hard-link publication is atomic and exclusive (rename would overwrite races).
        published_side=False; published_artifact=False
        try:
            os.link(side,self.output+'.txt'); published_side=True
            os.link(staged,self.output); published_artifact=True
            fd=os.open(os.path.dirname(self.output),os.O_RDONLY|os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
        except BaseException:
            if published_artifact: os.unlink(self.output)
            if published_side: os.unlink(self.output+'.txt')
            raise
        return self.output+'.txt'
