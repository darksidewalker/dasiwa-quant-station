# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import os
import gguf
import re
import torch
import logging
import argparse
import hashlib
from tqdm import tqdm
from safetensors.torch import load_file, save_file

QUANTIZATION_THRESHOLD = 1024
REARRANGE_THRESHOLD = 512
MAX_TENSOR_NAME_LENGTH = 127
MAX_TENSOR_DIMS = 4

class ModelTemplate:
    arch = "invalid"
    shape_fix = False 
    keys_detect = []  
    keys_banned = []  
    keys_hiprec = []  
    keys_skip   = []
    keys_ignore = []  

    def __init__(self):
        self.nd_tensors = {}

    def handle_nd_tensor(self, key, data, src_path=None):
        raise NotImplementedError(f"Tensor detected that exceeds dims supported by C++ code! ({key} @ {data.shape})")

class ModelFlux(ModelTemplate):
    arch = "flux"
    keys_detect = [
        ("transformer_blocks.0.attn.norm_added_k.weight",),
        ("double_blocks.0.img_attn.proj.weight",),
    ]
    keys_banned = ["transformer_blocks.0.attn.norm_added_k.weight",]

class ModelSD3(ModelTemplate):
    arch = "sd3"
    keys_detect = [
        ("transformer_blocks.0.attn.add_q_proj.weight",),
        ("joint_blocks.0.x_block.attn.qkv.weight",),
    ]
    keys_banned = ["transformer_blocks.0.attn.add_q_proj.weight",]

class ModelAura(ModelTemplate):
    arch = "aura"
    keys_detect = [
        ("double_layers.3.modX.1.weight",),
        ("joint_transformer_blocks.3.ff_context.out_projection.weight",),
    ]
    keys_banned = ["joint_transformer_blocks.3.ff_context.out_projection.weight",]

class ModelHiDream(ModelTemplate):
    arch = "hidream"
    keys_detect = [
        (
            "caption_projection.0.linear.weight",
            "double_stream_blocks.0.block.ff_i.shared_experts.w3.weight"
        )
    ]
    keys_hiprec = [
        # nn.parameter, can't load from BF16 ver
        ".ff_i.gate.weight",
        "img_emb.emb_pos"
    ]

class CosmosPredict2(ModelTemplate):
    arch = "cosmos"
    keys_detect = [
        (
            "blocks.0.mlp.layer1.weight",
            "blocks.0.adaln_modulation_cross_attn.1.weight",
        )
    ]
    keys_hiprec = ["pos_embedder"]
    keys_ignore = ["_extra_state", "accum_"]

class ModelHyVid(ModelTemplate):
    arch = "hyvid"
    keys_detect = [
        (
            "double_blocks.0.img_attn_proj.weight",
            "txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight",
        )
    ]

    def handle_nd_tensor(self, key, data, src_path=None):
        """Stash preserved tensors in memory to be saved as a single batch later."""
        self.nd_tensors[key] = torch.from_numpy(data)

class ModelWan(ModelHyVid):
    arch = "wan"
    keys_detect = [
        ("blocks.0.self_attn.norm_q.weight", "text_embedding.2.weight", "head.modulation",)
    ]
    keys_hiprec = [".modulation"]
    keys_skip = [
        r"(^|\.)modulation($|\.)",
        r"(^|\.)patch_embedding\.",
        r"(^|\.)text_embedding\.",
        r"(^|\.)time_projection\.",
        r"(^|\.)time_embedding\.",
        r"(^|\.)img_emb\.",
        r"(^|\.)head\.",
        r"vae\.", r"audio_vae\.", r"vocoder\.", r"encoder\.", r"decoder\."
    ]

class ModelLTXV(ModelHyVid):
    arch = "ltxv"
    keys_detect = [
        (
            "adaln_single.emb.timestep_embedder.linear_2.weight",
            "transformer_blocks.27.scale_shift_table",
            "caption_projection.linear_2.weight",
        )
    ]
    keys_hiprec = [
        "scale_shift_table", # nn.parameter, can't load from BF16 base quant
        "encoder.",
        "decoder.",
    ]
    keys_skip = [
        r"adaln_single\.",
        r"(audio|video)_embeddings_connector\.",
        r"(^|\.)caption_projection\.",
        r"(^|\.)patchify_proj($|\.)",
        r"(^|\.)proj_out($|\.)",
        r"(^|\.)audio_patchify_proj($|\.)",
        r"(^|\.)audio_proj_out($|\.)",
        r"scale_shift_table",
        r"\.to_gate_logits$",
        r"vae\.", r"audio_vae\.", r"vocoder\.", r"encoder\.", r"decoder\."
    ]

class ModelSDXL(ModelTemplate):
    arch = "sdxl"
    shape_fix = True
    keys_detect = [
        ("down_blocks.0.downsamplers.0.conv.weight", "add_embedding.linear_1.weight",),
        (
            "input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight",
            "output_blocks.2.2.conv.weight", "output_blocks.5.2.conv.weight",
        ), # Non-diffusers
        ("label_emb.0.0.weight",),
    ]

class ModelSD1(ModelTemplate):
    arch = "sd1"
    shape_fix = True
    keys_detect = [
        ("down_blocks.0.downsamplers.0.conv.weight",),
        (
            "input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight", "input_blocks.9.0.op.weight",
            "output_blocks.2.1.conv.weight", "output_blocks.5.2.conv.weight", "output_blocks.8.2.conv.weight"
        ), # Non-diffusers
    ]

class ModelLumina2(ModelTemplate):
    arch = "lumina2"
    keys_detect = [
        ("cap_embedder.1.weight", "context_refiner.0.attention.qkv.weight")
    ]

arch_list = [ModelFlux, ModelSD3, ModelAura, ModelHiDream, CosmosPredict2, 
             ModelLTXV, ModelHyVid, ModelWan, ModelSDXL, ModelSD1, ModelLumina2]

def is_model_arch(model, state_dict):
    # check if model is correct
    matched = False
    invalid = False
    for match_list in model.keys_detect:
        if all(key in state_dict for key in match_list):
            matched = True
            invalid = any(key in state_dict for key in model.keys_banned)
            break
    assert not invalid, "Model architecture not allowed for conversion! (i.e. reference VS diffusers format)"
    return matched

def detect_arch(state_dict):
    model_arch = None
    for arch in arch_list:
        if is_model_arch(arch, state_dict):
            model_arch = arch()
            break
    assert model_arch is not None, "Unknown model architecture!"
    return model_arch

def parse_args():
    parser = argparse.ArgumentParser(description="Generate F16 GGUF files from single UNET")
    parser.add_argument("--src", required=True, help="Source model ckpt file.")
    parser.add_argument("--dst", help="Output unet gguf file.")
    parser.add_argument("--arch", help="Specify architecture (e.g. ltxv, wan, flux). Bypasses auto-detection.")
    args = parser.parse_args()

    if not os.path.isfile(args.src):
        parser.error("No input provided!")

    return args

def strip_prefix(state_dict):
    # prefix for mixed state dict
    prefix = None
    for pfx in ["model.diffusion_model.", "model."]:
        if any([x.startswith(pfx) for x in state_dict.keys()]):
            prefix = pfx
            break

    # prefix for uniform state dict
    if prefix is None:
        for pfx in ["net."]:
            if all([x.startswith(pfx) for x in state_dict.keys()]):
                prefix = pfx
                break

    # strip prefix if found
    if prefix is not None:
        logging.info(f"State dict prefix found: '{prefix}'")
        sd = {}
        for k, v in state_dict.items():
            if prefix not in k:
                continue
            k = k.replace(prefix, "")
            sd[k] = v
    else:
        logging.debug("State dict has no prefix")
        sd = state_dict

    return sd

def load_state_dict(path):
    if any(path.endswith(x) for x in [".ckpt", ".pt", ".bin", ".pth"]):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        for subkey in ["model", "module"]:
            if subkey in state_dict:
                state_dict = state_dict[subkey]
                break
    else:
        state_dict = load_file(path)
    return strip_prefix(state_dict)

def handle_tensors(writer, state_dict, model_arch, src_path=None):
    name_lengths = tuple(sorted(
        ((key, len(key)) for key in state_dict.keys()),
        key=lambda item: item[1],
        reverse=True,
    ))
    if not name_lengths:
        return
    
    max_name_len = name_lengths[0][1]
    for key, data in tqdm(state_dict.items()):
        old_dtype = data.dtype

        if any(x in key for x in model_arch.keys_ignore):
            continue

        if data.dtype == torch.bfloat16:
            data = data.to(torch.float32).numpy()
        elif data.dtype in [getattr(torch, "float8_e4m3fn", "_invalid"), getattr(torch, "float8_e5m2", "_invalid")]:
            data = data.to(torch.float16).numpy()
        else:
            data = data.numpy()

        n_dims = len(data.shape)
        data_shape = data.shape
        data_qtype = gguf.GGMLQuantizationType.BF16 if old_dtype == torch.bfloat16 else gguf.GGMLQuantizationType.F16

        # Preservation Logic: Extract 5D tensors OR keys matching skip patterns
        if len(data.shape) > MAX_TENSOR_DIMS or any(re.search(pat, key) for pat in model_arch.keys_skip):
            model_arch.handle_nd_tensor(key, data, src_path=src_path)
            continue 

        n_params = data.size
        if old_dtype in (torch.float32, torch.bfloat16):
            if n_dims == 1 or n_params <= QUANTIZATION_THRESHOLD or any(x in key for x in model_arch.keys_hiprec):
                data_qtype = gguf.GGMLQuantizationType.F32

        if (model_arch.shape_fix and n_dims > 1 and n_params >= REARRANGE_THRESHOLD 
            and (n_params / 256).is_integer() and not (data.shape[-1] / 256).is_integer()):
            orig_shape = data.shape
            data = data.reshape(n_params // 256, 256)
            writer.add_array(f"comfy.gguf.orig_shape.{key}", tuple(int(dim) for dim in orig_shape))

        try:
            data = gguf.quants.quantize(data, data_qtype)
        except (AttributeError, gguf.QuantError):
            data_qtype = gguf.GGMLQuantizationType.F16
            data = gguf.quants.quantize(data, data_qtype)

        shape_str = f"{{{', '.join(str(n) for n in reversed(data.shape))}}}"
        tqdm.write(f"{f'%-{max_name_len + 4}s' % key} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")
        writer.add_tensor(key, data, raw_dtype=data_qtype)

def convert_file(path, dst_path=None, interact=True, overwrite=False, arch=None):
    state_dict = load_state_dict(path)

    model_arch = None
    if arch:
        for a_cls in arch_list:
            if a_cls.arch == arch:
                model_arch = a_cls()
                break
        if model_arch is None:
            logging.warning(f"Architecture override '{arch}' not found in registry; using auto-detection.")

    if model_arch is None:
        model_arch = detect_arch(state_dict)
    
    dtypes = [x.dtype for x in state_dict.values()]
    main_dtype = max(set(dtypes), key=dtypes.count)
    ftype_name = "BF16" if main_dtype == torch.bfloat16 else "F16"
    ftype_gguf = gguf.LlamaFileType.MOSTLY_BF16 if main_dtype == torch.bfloat16 else gguf.LlamaFileType.MOSTLY_F16

    if dst_path is None:
        dst_path = f"{os.path.splitext(path)[0]}-{ftype_name}.gguf"

    # Clean start for sidecar file
    file_hash = hashlib.md5(os.path.basename(path).encode()).hexdigest()[:8]
    fix_path = f"./fix_5d_tensors_{model_arch.arch}_{file_hash}.safetensors"
    if os.path.exists(fix_path):
        os.remove(fix_path)

    if os.path.isfile(dst_path) and not overwrite:
        if not interact: raise OSError("Output exists!")
        input("Output exists enter to continue...")

    writer = gguf.GGUFWriter(path=None, arch=model_arch.arch)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    if ftype_gguf is not None:
        writer.add_file_type(ftype_gguf)

    handle_tensors(writer, state_dict, model_arch, src_path=path)

    # Memory-efficient save: Write all preserved tensors at once.
    if model_arch.nd_tensors:
        save_file(model_arch.nd_tensors, fix_path)
        tqdm.write(f"🎯 Successfully preserved {len(model_arch.nd_tensors)} structural tensors: {fix_path}")

    writer.write_header_to_file(path=dst_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    file_hash = hashlib.md5(os.path.basename(path).encode()).hexdigest()[:8]
    fix = f"./fix_5d_tensors_{model_arch.arch}_{file_hash}.safetensors"
    if os.path.isfile(fix):
        logging.warning(f"\n### Success! Unique fix file created at '{fix}'")
        logging.warning(" The UI engine will now use this file for the 5D Tensor Fix step.")

    return dst_path, model_arch

if __name__ == "__main__":
    args = parse_args()
    convert_file(args.src, args.dst, arch=args.arch)
