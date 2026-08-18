import os
import json
import struct
import datetime
import hashlib
import zlib
from core.metadata_configs import MODEL_METADATA_CONFIGS, COMMON_METADATA
from core.watermark import watermark_for
from safetensors.torch import load_file, save_file
from safetensors import safe_open

try:
    import gguf
except ImportError:
    gguf = None


# Safetensors spec: header JSON is capped at 100MB. We stay well under that.
_SAFETENSORS_HEADER_MAX = 100 * 1024 * 1024
_SPACER_KEY = "__spacer"
# Loader-critical runtime metadata is authored by the quantizer and must not
# be replaced by arbitrary JSON pasted into the manual metadata editor.
_PROTECTED_EXISTING_METADATA_KEYS = {"_quantization_metadata"}

def calculate_sha256(file_path):
    """Calculates a clean 0x-prefixed SHA256 hash of the target file."""
    if not os.path.exists(file_path) or file_path == "PREVIEW_MODE":
        return "0x[HASH_WILL_BE_CALCULATED_ON_SAVE]"
    
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in 64kb chunks for memory efficiency
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return f"0x{sha256_hash.hexdigest()}"


def calculate_civitai_hashes(file_path):
    """Calculate common Civitai file hash fields for metadata/recipes."""
    if not os.path.exists(file_path) or file_path == "PREVIEW_MODE":
        placeholder = "HASH_WILL_BE_CALCULATED_ON_SAVE"
        return {
            "AutoV1": placeholder,
            "AutoV2": placeholder,
            "AutoV3": placeholder,
            "SHA256": placeholder,
            "CRC32": placeholder,
        }

    sha256_hash = hashlib.sha256()
    crc = 0
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(1024 * 1024), b""):
            sha256_hash.update(byte_block)
            crc = zlib.crc32(byte_block, crc)

    sha = sha256_hash.hexdigest().upper()
    return {
        "AutoV1": sha[:8],
        "AutoV2": sha[:10],
        "AutoV3": sha[:12],
        "SHA256": sha,
        "CRC32": f"{crc & 0xFFFFFFFF:08X}",
    }


def _civitai_hash_metadata(file_path):
    hashes = calculate_civitai_hashes(file_path)
    return {f"civitai.hash.{name}": value for name, value in hashes.items()}


def normalize_quantization_bits(bits):
    """Return the canonical metadata label for a selected quant target."""
    labels = {
        "FP8": "FP8",
        "NVFP4": "NVFP4",
        "MXFP8": "MXFP8",
        "Hybrid MXFP8": "Hybrid MXFP8",
        "INT8 Tensor-wise": "INT8 Tensor-wise",
        "INT8 Row-wise ConvRot Runtime": "INT8 Row-wise ConvRot (runtime)",
        "INT4 ConvRot Runtime": "INT4 ConvRot",
        # Stale sessions may still send the old value. It intentionally maps
        # to the non-ConvRot command path in safetensors_engine, so metadata
        # must describe the real output instead of the stale UI label.
        "INT8 Row-wise ConvRot": "INT8 Tensor-wise",
    }
    return labels.get(bits, bits)


def get_current_meta(model_name, architecture, bits="FP8"):
    """
    Standard asset-based metadata template (Fallback logic).
    Used if no specialized JSON dump is found in the core folder.
    """
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    bits = normalize_quantization_bits(bits)
    
    # Get architecture-specific fields (fallback to WAN 2.2 if not found)
    base_config = MODEL_METADATA_CONFIGS.get(architecture, MODEL_METADATA_CONFIGS.get("WAN 2.2"))
    
    # Merge with common fields (Date, Tool, Bits). Architecture-specific
    # overrides (e.g. Anima -> BF16) take precedence over the common defaults.
    full_template = {**COMMON_METADATA, **base_config}
    
    # Perform string replacement for all template tokens
    final_meta = {}
    for k, v in full_template.items():
        if isinstance(v, str):
            final_meta[k] = (v.replace("{model_name}", model_name)
                              .replace("{date}", date_str)
                              .replace("{bits}", bits))
        else:
            final_meta[k] = v
            
    return final_meta

def get_specialized_meta(architecture, model_name, final_file_path, bits="FP8", is_full=False):
    """
    PRIORITY 1: Loads the FULL content of {Arch}_metadata.json from /core.
    PRIORITY 2: Falls back to asset-based template.
    """
    bits = normalize_quantization_bits(bits)
    core_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Cleaning logic: "LTX-2.3" -> "LTX23", "WAN 2.2" -> "WAN22"
    clean_arch = architecture.replace("-", "").replace(".", "").replace(" ", "")
    
    seed_filename = f"{clean_arch}_metadata.json"
    if is_full:
        vae_filename = f"{clean_arch}_metadata_vae.json"
        if os.path.exists(os.path.join(core_dir, vae_filename)):
            seed_filename = vae_filename
            
    seed_path = os.path.join(core_dir, seed_filename)

    if os.path.exists(seed_path):
        try:
            with open(seed_path, 'r', encoding='utf-8') as f:
                # 1. Load the FULL dump (License, Architecture, Resolution, etc.)
                meta = json.load(f)
            
            # 2. OVERWRITE ONLY our specific UI/Session fields
            meta["modelspec.title"] = model_name 
            meta["modelspec.hash_sha256"] = calculate_sha256(final_file_path)
            meta["modelspec.date"] = datetime.datetime.now().strftime("%Y-%m-%d")
            meta["quantization.bits"] = bits
            meta["quantization.tool"] = "https://github.com/darksidewalker/dasiwa-quant-station"
            
            # Ensure the spacer exists for future header edits
            if "__spacer" not in meta:
                meta["__spacer"] = " " * 2048
            
            return meta
        except Exception as e:
            print(f"❌ Error merging {seed_filename}: {e}")

    # Fallback to standard assets.py config if JSON seed is missing
    return get_current_meta(model_name, architecture, bits)

def update_metadata_preview(name, architecture="WAN 2.2", is_full=False):
    """Called by UI to generate the preview for the Gradio JSON box."""
    # Preview is architecture-only; the real quant target is selected later by
    # the quantization job, so show a placeholder instead of a misleading FP8.
    meta = get_specialized_meta(
        architecture,
        name,
        "PREVIEW_MODE",
        bits="{target_quantization}",
        is_full=is_full,
    )
    return json.dumps(meta, indent=4)

# Keys that MUST be present for a functional LTX 2.3 checkpoint.
# These are overwritten by get_specialized_meta() and must never be
# replaced by arbitrary user edits.
_LTX23_REQUIRED_KEYS = {
    "modelspec.architecture",
    "modelspec.implementation",
    "modelspec.license",
    "modelspec.resolution",
    "modelspec.resolution_hints",
    "modelspec.resolution_native",
    "modelspec.resolution_aspect",
    "quantization.bits",
    "quantization.tool",
}

def merge_custom_metadata(architecture, model_name, file_path, bits="BF16",
                          custom_meta=None, is_full=False, extra_meta=None):
    """
    Build final metadata dict for safetensors output.

    Priority order:
      1. get_specialized_meta(...) provides the base template with all
         required LTX 2.3 functional fields, resolution hints, etc.
      2. custom_meta (user-edited JSON from UI) overlays on top, but
         REQUIRED_KEYS are protected so user edits cannot break the
         checkpoint.
      3. extra_meta (e.g. merge provenance) overlays last.

    Returns dict of string -> string ready for save_file(..., metadata=...).
    """
    base = get_specialized_meta(architecture, model_name, file_path, bits, is_full=is_full)
    base.update(_civitai_hash_metadata(file_path))

    if custom_meta:
        for k, v in custom_meta.items():
            if k in _LTX23_REQUIRED_KEYS:
                continue
            base[k] = v

    if extra_meta:
        base.update(extra_meta)

    # EC watermark (X25519 + AES-256-GCM) written into the modelspec.watermark
    # field (no-op if no passphrase is configured).
    wm = watermark_for(architecture, model_name, file_path, bits=bits)
    if wm:
        base.update(wm)

    # Ensure spacer exists for future header edits
    if "__spacer" not in base:
        base["__spacer"] = " " * 2048

    return base

def _read_safetensors_header(file_path):
    """
    Read the raw safetensors header without loading any tensor data.

    Returns: (header_dict, header_size, header_bytes)
        header_dict: parsed JSON header (includes tensor specs and __metadata__)
        header_size: int, byte length of the JSON header (NOT including the 8-byte prefix)
        header_bytes: the raw header bytes as read from disk
    """
    with open(file_path, "rb") as f:
        size_prefix = f.read(8)
        if len(size_prefix) != 8:
            raise ValueError("File too small to be a valid safetensors file")
        header_size = struct.unpack("<Q", size_prefix)[0]
        if header_size <= 0 or header_size > _SAFETENSORS_HEADER_MAX:
            raise ValueError(f"Invalid header size: {header_size}")
        header_bytes = f.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError("Truncated header")
    header_dict = json.loads(header_bytes.decode("utf-8"))
    return header_dict, header_size, header_bytes


def _try_inplace_metadata_rewrite(file_path, meta_dict):
    """
    Attempt to rewrite the safetensors header in place by swapping only the
    __metadata__ section. Tensor specs and tensor data are not touched.

    Strategy:
      1. Read the existing header.
      2. Build a new header with the same tensor specs but new __metadata__.
      3. If the new JSON fits in the original byte allocation, pad the
         __spacer field with whitespace so the total length matches exactly,
         then write the new 8-byte prefix + JSON over the original.
      4. If it doesn't fit even after shrinking the spacer to a minimum,
         return False so the caller falls back to a full rewrite.

    Returns: (success: bool, message: str)
    """
    try:
        header_dict, original_header_size, _ = _read_safetensors_header(file_path)
    except Exception as e:
        return False, f"Could not read header: {e}"

    # safetensors stores all values as strings; coerce to keep the format valid.
    meta_strings = {k: (v if isinstance(v, str) else json.dumps(v))
                    for k, v in meta_dict.items()}

    # Preserve tensor specs (everything that isn't __metadata__).
    new_header = {k: v for k, v in header_dict.items() if k != "__metadata__"}
    new_header["__metadata__"] = dict(meta_strings)

    # Serialize once to measure size. safetensors uses compact JSON (no indent).
    candidate = json.dumps(new_header, separators=(",", ":"), ensure_ascii=False)
    candidate_bytes = candidate.encode("utf-8")

    if len(candidate_bytes) > original_header_size:
        # Try to shrink the spacer field to make room.
        current_spacer = new_header["__metadata__"].get(_SPACER_KEY, "")
        overshoot = len(candidate_bytes) - original_header_size
        if len(current_spacer) >= overshoot:
            new_spacer_len = len(current_spacer) - overshoot
            new_header["__metadata__"][_SPACER_KEY] = " " * new_spacer_len
            candidate = json.dumps(new_header, separators=(",", ":"), ensure_ascii=False)
            candidate_bytes = candidate.encode("utf-8")
        if len(candidate_bytes) > original_header_size:
            return False, "New header exceeds original allocation"

    # Pad up to exactly the original size by extending the spacer.
    if len(candidate_bytes) < original_header_size:
        padding_needed = original_header_size - len(candidate_bytes)
        existing_spacer = new_header["__metadata__"].get(_SPACER_KEY, "")
        new_header["__metadata__"][_SPACER_KEY] = existing_spacer + " " * padding_needed
        candidate = json.dumps(new_header, separators=(",", ":"), ensure_ascii=False)
        candidate_bytes = candidate.encode("utf-8")

        # JSON overhead (the spacer key itself if it didn't exist) can shift the
        # length by a few bytes. Trim or pad once more to land exactly.
        delta = original_header_size - len(candidate_bytes)
        if delta != 0:
            spacer_now = new_header["__metadata__"][_SPACER_KEY]
            if delta > 0:
                new_header["__metadata__"][_SPACER_KEY] = spacer_now + " " * delta
            else:
                if len(spacer_now) + delta < 0:
                    return False, "Could not align header size via spacer"
                new_header["__metadata__"][_SPACER_KEY] = spacer_now[:delta]
            candidate = json.dumps(new_header, separators=(",", ":"), ensure_ascii=False)
            candidate_bytes = candidate.encode("utf-8")

    if len(candidate_bytes) != original_header_size:
        return False, (
            f"Could not align header to original size "
            f"({len(candidate_bytes)} vs {original_header_size})"
        )

    # Write the new 8-byte length prefix (unchanged value) and the new header.
    # Tensor data after this region is untouched.
    try:
        with open(file_path, "r+b") as f:
            f.seek(0)
            f.write(struct.pack("<Q", original_header_size))
            f.write(candidate_bytes)
    except Exception as e:
        return False, f"Write failed: {e}"

    return True, "Header rewritten in place"


def inject_metadata(file_path, meta_dict):
    """
    Write metadata into a safetensors file header.

    Fast path: rewrite only the header bytes if the new metadata fits in the
    original allocation (typical when only updating session fields like title,
    date, hash). For multi-GB models this avoids reading and rewriting the
    entire file.

    Slow path: load all tensors and re-save (used when the new header would
    grow beyond the original allocation, or if the in-place rewrite fails).
    """
    try:
        header_dict, _, _ = _read_safetensors_header(file_path)
        existing_meta = header_dict.get("__metadata__", {})
        merged_meta = dict(existing_meta)
        merged_meta.update(meta_dict)
        for key in _PROTECTED_EXISTING_METADATA_KEYS:
            if key in existing_meta:
                merged_meta[key] = existing_meta[key]
    except Exception:
        # _try_inplace_metadata_rewrite below will return the useful header
        # read error; retain the requested metadata for its fallback path.
        merged_meta = meta_dict

    ok, msg = _try_inplace_metadata_rewrite(file_path, merged_meta)
    if ok:
        return True, f"Metadata Injected (in-place: {msg})"

    # Slow path fallback. Logs the in-place failure reason so it's debuggable.
    try:
        # Safetensors requires all metadata values to be strings.
        meta_strings = {k: (v if isinstance(v, str) else json.dumps(v))
                        for k, v in merged_meta.items()}
        tensors = load_file(file_path)
        save_file(tensors, file_path, metadata=meta_strings)
        return True, f"Metadata Injected (full rewrite; in-place skipped: {msg})"
    except Exception as e:
        return False, str(e)

def _gguf_value_for_type(field):
    """Extract a (value, gguf_value_type) pair from a GGUFReader field."""
    val_type = field.types[0]
    return field.contents(), val_type


def _gguf_add_existing_field(writer, field):
    """
    Carry an existing GGUF metadata field into the new writer using only
    methods available across gguf library versions. Skips arrays whose subtype
    we can't reliably forward without `sub_type` kwarg support.

    Returns True if the field was successfully added, False if it was skipped.
    """
    name = field.name
    val_type = field.types[0]
    value = field.contents()

    # Map of GGUFValueType -> writer method
    type_dispatch = {
        gguf.GGUFValueType.UINT8: writer.add_uint8,
        gguf.GGUFValueType.INT8: writer.add_int8,
        gguf.GGUFValueType.UINT16: writer.add_uint16,
        gguf.GGUFValueType.INT16: writer.add_int16,
        gguf.GGUFValueType.UINT32: writer.add_uint32,
        gguf.GGUFValueType.INT32: writer.add_int32,
        gguf.GGUFValueType.FLOAT32: writer.add_float32,
        gguf.GGUFValueType.UINT64: writer.add_uint64,
        gguf.GGUFValueType.INT64: writer.add_int64,
        gguf.GGUFValueType.FLOAT64: writer.add_float64,
        gguf.GGUFValueType.BOOL: writer.add_bool,
        gguf.GGUFValueType.STRING: writer.add_string,
    }

    add_fn = type_dispatch.get(val_type)
    if add_fn is not None:
        add_fn(name, value)
        return True

    # Array case: try to forward via add_array. Some writer versions accept
    # this directly for typed arrays; others require the sub_type kwarg of
    # add_key_value. Try add_key_value with sub_type first, then plain
    # add_array, then give up cleanly.
    if val_type == gguf.GGUFValueType.ARRAY:
        sub_type = field.types[-1] if len(field.types) > 1 else None
        try:
            writer.add_key_value(name, value, val_type, sub_type=sub_type)
            return True
        except TypeError:
            # Older signature: add_key_value(key, val, vtype) without sub_type
            try:
                writer.add_array(name, value)
                return True
            except Exception:
                return False
        except Exception:
            return False

    return False


def write_gguf_meta(file_path, model_name, architecture, bits="FP8", is_full=False):
    """
    Inject modelspec.* metadata into an existing GGUF file by reading it,
    copying tensors and KV pairs to a new file with the additional metadata,
    and atomically replacing the original.

    The `architecture` argument is the source model architecture string
    (e.g. "wan", "ltxv") used to look up the right specialized metadata
    template. The GGUF file's own `general.architecture` value is preserved
    from the input (set by upstream tools like convert.py / llama-quantize).

    `bits` is a label used in the metadata (e.g. "Q8_0", "FP8") and is not
    used to recompute the actual on-disk quantization.

    Returns (success: bool, message: str).
    """
    if not gguf:
        return False, "gguf library not installed"

    if not os.path.isfile(file_path):
        return False, f"File not found: {file_path}"

    try:
        reader = gguf.GGUFReader(file_path, "r")
    except Exception as e:
        return False, f"GGUF read failed: {e}"

    # Preserve the GGUF's own architecture (set by upstream tooling).
    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch_field is None:
        return False, "Source GGUF has no general.architecture field"
    
    # Robustly extract architecture string using contents()
    gguf_arch_str = arch_field.parts[-1].tobytes().decode("utf-8").strip("\x00")

    # Honor any custom alignment used by the source.
    alignment_field = reader.get_field(gguf.Keys.General.ALIGNMENT)
    custom_alignment = None
    if alignment_field is not None:
        try:
            custom_alignment = int(alignment_field.parts[alignment_field.data[-1]])
        except Exception:
            custom_alignment = None

    # Build the modelspec metadata to merge in.
    new_meta = get_specialized_meta(architecture, model_name, file_path, bits, is_full=is_full)
    # Coerce non-string values to strings (the format we already use elsewhere).
    new_meta_strings = {
        k: (v if isinstance(v, str) else json.dumps(v))
        for k, v in new_meta.items()
    }

    # EC watermark (X25519 + AES-256-GCM) written into the modelspec.watermark
    # field (no-op if no passphrase is configured).
    wm = watermark_for(architecture, model_name, file_path, bits=bits)
    if wm:
        for k, v in wm.items():
            new_meta_strings[k] = v if isinstance(v, str) else json.dumps(v)

    # Write to a sibling temp file, then atomically replace.
    dst_dir = os.path.dirname(file_path) or "."
    tmp_path = os.path.join(
        dst_dir, f".{os.path.basename(file_path)}.meta.tmp"
    )
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    try:
        writer = gguf.GGUFWriter(
            tmp_path, arch=gguf_arch_str, endianess=reader.endianess
        )
    except TypeError:
        # Older signatures may not accept `endianess` as kwarg
        writer = gguf.GGUFWriter(tmp_path, arch=gguf_arch_str)

    if custom_alignment is not None:
        writer.data_alignment = custom_alignment

    # Track which keys we'll override so we don't double-add them.
    override_keys = set(new_meta_strings.keys())
    skipped_fields = []

    # GGUFWriter._init_ already calls add_architecture(). Skip that field
    # and any GGUFWriter virtual fields.
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE:
            continue
        if field.name.startswith("GGUF."):
            continue
        if field.name in override_keys:
            # Will be added below with the new value
            continue
        if not _gguf_add_existing_field(writer, field):
            skipped_fields.append(f"{field.name} ({field.types})")

    # Add the new modelspec.* / quantization.* fields as strings.
    for k, v in new_meta_strings.items():
        try:
            writer.add_string(k, v)
        except ValueError:
            # Duplicated key -- shouldn't happen given override_keys logic, but
            # if it does, skip rather than crash.
            pass

    # Register tensor info from source so the output gets a valid tensor
    # section. Tensor data is streamed below via write_tensor_data.
    for tensor in reader.tensors:
        # Determine if the tensor uses a quantized type that requires physical byte-shape conversion.
        # Non-quantized types (F32, F16, BF16) use logical shapes in GGUFWriter.
        is_quantized = tensor.tensor_type not in (
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
            getattr(gguf.GGMLQuantizationType, 'BF16', 30)
        )

        if is_quantized:
            # For quantized tensors, GGUFWriter.add_tensor_info expects the "byte shape"
            # in Torch-order [outer, ..., inner_bytes].
            # GGUFReader.shape is GGUF-order [inner_logical, outer, ...].
            outer_dims = tensor.shape[1:]
            
            prod_outer = 1
            for d in outer_dims:
                prod_outer *= d
            
            # Calculate the physical bytes occupied by a single "row" (the contiguous dimension)
            bytes_per_row = tensor.data.nbytes // prod_outer
            t_shape = list(outer_dims[::-1]) + [bytes_per_row]
        else:
            # For standard types, just reverse the GGUF-order shape to Torch-order.
            t_shape = tensor.shape[::-1]

        writer.add_tensor_info(
            tensor.name,
            t_shape,
            tensor.data.dtype,
            tensor.data.nbytes,
            tensor.tensor_type,
        )

    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for tensor in reader.tensors:
            try:
                writer.write_tensor_data(
                    tensor.data, tensor_endianess=reader.endianess
                )
            except TypeError:
                # Older signature: write_tensor_data(data) only
                writer.write_tensor_data(tensor.data)
        writer.close()
    except Exception as e:
        try:
            writer.close()
        except Exception:
            pass
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return False, f"GGUF write failed: {e}"

    # Atomic replace.
    try:
        os.replace(tmp_path, file_path)
    except Exception as e:
        return False, f"Atomic replace failed: {e}"

    msg = "GGUF Meta Injected"
    if skipped_fields:
        msg += f" (skipped {len(skipped_fields)} unforwardable field(s))"
    return True, msg

def read_any_metadata(MODELS_DIR, file_name):
    """Reads and returns the header metadata from Safetensors or GGUF for the terminal."""
    if not file_name: 
        return "❌ No file selected."
    
    MODELS_DIR = os.path.expanduser(MODELS_DIR)
    path = os.path.join(MODELS_DIR, file_name)
    if not os.path.exists(path):
        return f"❌ File not found at: {path}"

    # GGUF Reading Logic
    if file_name.lower().endswith(".gguf"):
        if not gguf: return "❌ 'gguf' library missing."
        try:
            reader = gguf.GGUFReader(path, "r")
            lines = [f"🔍 GGUF DIAGNOSTIC: {file_name}", "="*60]
            
            # Metadata Summary
            lines.append(f"  Endianness : {reader.endianess.name}")
            lines.append(f"  KV Pairs   : {len(reader.fields)}")
            lines.append(f"  Tensors    : {len(reader.tensors)}")
            
            lines.append("\n📊 KEY-VALUE DATA:")
            for key, field in reader.fields.items():
                try:
                    val, _ = _gguf_value_for_type(field)
                    lines.append(f"  {key:<40}: {val}")
                except Exception:
                    lines.append(f"  {key:<40}: [Unparseable Binary Data]")

            # Tensor Audit - Useful for verifying the 5D self-healing logic
            lines.append("\n🧠 TENSOR AUDIT (Top structural layers):")
            has_5d = False
            for tensor in reader.tensors:
                dim_count = len(tensor.shape)
                if dim_count >= 5: has_5d = True
                
                # Display structural tensors (weights or high-dim)
                if dim_count > 2 or tensor.name.endswith(".weight"):
                    shape_str = "x".join(map(str, tensor.shape))
                    marker = " ⭐ 5D" if dim_count >= 5 else ""
                    lines.append(f"  {tensor.name:<45} | {shape_str:>15} | {tensor.tensor_type.name}{marker}")
            
            if not has_5d:
                lines.append("\nℹ️  No 5D tensors detected. Verify if self-healing was applied.")
                
            return "\n".join(lines)
        except Exception as e:
            return f"🔥 GGUF Read Error: {str(e)}"

    # Safetensors Reading Logic
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            meta = f.metadata()
            if not meta:
                return f"🔍 HEADER: {file_name}\n" + "-"*40 + "\nEmpty or no metadata found."
            return f"🔍 SAFETENSORS HEADER: {file_name}\n" + "-"*40 + "\n" + json.dumps(meta, indent=4)
    except Exception as e:
        return f"🔥 Read Error: {str(e)}"