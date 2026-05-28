# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import os
import gguf
import torch
import argparse
from tqdm import tqdm
from safetensors.torch import load_file

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--fix", required=False, help="Defaults to ./fix_5d_tensors_[arch].pt")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.src):
        parser.error(f"Invalid source file '{args.src}'")
    if not args.overwrite and os.path.exists(args.dst):
        parser.error(f"Output exists, use '--overwrite' ({args.dst})")

    return args

def get_arch_str(reader):
    field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if not field: return "unknown"
    return field.parts[-1].tobytes().decode("utf-8").strip("\x00")

def get_file_type(reader):
    field = reader.get_field(gguf.Keys.General.FILE_TYPE)
    if not field: return gguf.LlamaFileType.MOSTLY_F16
    val = field.contents()
    ft = int(val[0] if isinstance(val, list) else val)
    return gguf.LlamaFileType(ft)

if __name__ == "__main__":
    args = get_args()

    # read existing
    reader = gguf.GGUFReader(args.src)
    arch = get_arch_str(reader)
    file_type = get_file_type(reader)
    print(f"Detected arch: '{arch}' (ftype: {str(file_type)})")

    # prep fix
    if args.fix is None:
        args.fix = f"./fix_5d_tensors_{arch}.safetensors"
 
    if not os.path.isfile(args.fix):
        raise OSError(f"No 5D tensor fix file: {args.fix}")

    sd5d = load_file(args.fix)
    # NumPy does not support BFloat16 via the PyTorch bridge. We cast to Float32 
    # here to ensure the data can be processed by GGUFWriter.
    sd5d = {k: v.to(torch.float32).numpy() for k, v in sd5d.items()}
    print("5D tensors:", sd5d.keys())

    # prep output
    writer = gguf.GGUFWriter(path=None, arch=arch)
    
    # Carry over all non-tensor metadata from the source to prevent "Could not detect model type"
    # and ensure other standard GGUF keys are preserved.
    for key, field in reader.fields.items():
        if key == gguf.Keys.General.ARCHITECTURE:
            continue # Already handled by GGUFWriter constructor
        if key.startswith("GGUF."):
            continue # Virtual fields handled by writer
        
        val_type = field.types[0]
        value = field.contents()
        writer.add_key_value(key, value, val_type)

    added = []
    def add_extra_key(writer, key, data):
        global added
        data_qtype = gguf.GGMLQuantizationType.F32
        # TRANSPOSE FIX: GGUFWriter transposes 2D arrays, but loaders only
        # transpose back if the name ends in .weight. For structural tables,
        # we pre-transpose so the final GGUF shape matches Torch expectation.
        if len(data.shape) == 2 and not key.endswith(".weight"):
            data = data.T.copy()

        data = gguf.quants.quantize(data, data_qtype)
        tqdm.write(f"Adding key {key} ({data.shape})")
        writer.add_tensor(key, data, raw_dtype=data_qtype)
        added.append(key)

    # main loop to add missing 5D tensor(s)
    for tensor in tqdm(reader.tensors):
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        key5d = tensor.name.replace(".bias", ".weight")
        if key5d in sd5d.keys():
            add_extra_key(writer, key5d, sd5d[key5d])

    # brute force for any missed
    for key, data in sd5d.items():
        if key not in added:
            add_extra_key(writer, key, data)

    writer.write_header_to_file(path=args.dst)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
