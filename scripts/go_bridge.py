#!/usr/bin/env python3
import argparse
import ctypes
import gc
import json
import os
import subprocess
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

VENV_DIR = os.path.join(ROOT_DIR, ".venv")
VENV_BIN = os.path.join(VENV_DIR, "bin")
if os.path.isdir(VENV_BIN):
    os.environ["VIRTUAL_ENV"] = VENV_DIR
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if VENV_BIN not in path_parts:
        os.environ["PATH"] = os.pathsep.join([VENV_BIN] + path_parts)

from core.gguf_engine import run_gguf_conversion
from core.metadata_manager import (
    calculate_sha256,
    inject_metadata,
    read_any_metadata,
    update_metadata_preview,
)
from core.lora_merge_engine import run_lora_merge
from core.int4_convrot_engine import run_int4_convrot_conversion
from core.safetensors_engine import run_safe_conversion
from core.watermark import verify_watermark as wm_verify, save_key as wm_save_key, watermark_status as wm_status
from utils.arch_detector import inspect_checkpoint
from utils.file_ops import ensure_dirs
from utils.pattern_audit import audit_patterns
from utils.scanner_5d import scan_5d_tensors


def _emit(obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def _load_payload(args):
    if args.json:
        return json.loads(args.json)
    return json.load(sys.stdin)


def _watermark_context(payload):
    """
    Apply the per-run watermark switch from the payload and return a one-line
    hint describing the effective watermark state.

    - ``watermark`` (bool, default True) gates whether a modelspec.watermark
      token is written for this job.
    - The hint tells the user when watermarking is enabled but no key is
      configured, so outputs silently lack provenance.
    """
    enabled = bool(payload.get("watermark", True))
    if enabled:
        os.environ.pop("DASIWA_WATERMARK_DISABLED", None)
    else:
        os.environ["DASIWA_WATERMARK_DISABLED"] = "1"

    status = wm_status()
    if not enabled:
        return "Watermarking disabled for this run — no modelspec.watermark will be written."
    if status["available"]:
        return f"Watermarking ON: modelspec.watermark will be added (secret from {status['source']})."
    return ("Watermarking ON but no key configured — outputs will NOT carry "
            "modelspec.watermark. Set DASIWA_WATERMARK_PASSPHRASE or run "
            "`go_bridge.py watermark-key` to enable provenance.")


def cmd_inspect(args):
    path = os.path.realpath(os.path.expanduser(args.path))
    arch, is_full, log = inspect_checkpoint(path)
    _emit({"architecture": arch, "full_checkpoint": is_full, "log": log})


def cmd_metadata(args):
    _emit({
        "metadata": update_metadata_preview(
            args.name or "TreasureChest",
            args.architecture or "WAN 2.2",
            is_full=args.full,
        )
    })


def cmd_read_metadata(args):
    _emit({"text": read_any_metadata(args.models_dir, args.file)})


def cmd_read_metadata_path(args):
    path = os.path.realpath(os.path.expanduser(args.path))
    _emit({
        "text": read_any_metadata(os.path.dirname(path), os.path.basename(path))
    })


def cmd_inject_metadata_path(args):
    payload = _load_payload(args)
    path = os.path.realpath(os.path.expanduser(payload["path"]))
    if not path.lower().endswith(".safetensors"):
        _emit({"ok": False, "text": "Metadata injection currently supports safetensors sources."})
        return
    meta = json.loads(payload["metadata"])
    meta["modelspec.hash_sha256"] = calculate_sha256(path)
    ok, msg = inject_metadata(path, meta)
    _emit({"ok": ok, "text": msg})


def cmd_scan(args):
    _emit({"text": scan_5d_tensors(args.path)})


def cmd_audit(args):
    _emit({"text": audit_patterns(args.path, args.architecture)})


def cmd_clean_memory(args):
    lines = [
        "Released DaSiWa Go heap pressure and ran Python garbage collection.",
    ]
    collected = gc.collect()
    lines.append(f"Python GC collected {collected} objects.")

    try:
        libc = ctypes.CDLL("libc.so.6")
        if libc.malloc_trim(0) == 1:
            lines.append("Returned free libc heap pages to the OS.")
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            lines.append("Requested PyTorch CUDA cache release.")
        else:
            lines.append("PyTorch CUDA is not active.")
    except Exception as exc:
        lines.append(f"PyTorch cleanup skipped: {exc}")

    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
        lines.append("Requested CuPy memory pool release.")
    except Exception as exc:
        lines.append(f"CuPy cleanup skipped: {exc}")

    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
        lines.append("Requested TensorFlow/Keras session cleanup.")
    except Exception as exc:
        lines.append(f"TensorFlow cleanup skipped: {exc}")

    holders = _nvidia_memory_holders()
    if holders:
        lines.append("")
        lines.append("Processes still holding NVIDIA VRAM:")
        lines.extend(holders)
        lines.append("Close or stop those processes to release their model memory.")

    _emit({"text": "\n".join(lines)})


def _nvidia_memory_holders():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []

    holders = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        gpu_uuid, pid, name, used_mb = parts
        holders.append(f"- PID {pid}: {name} on {gpu_uuid} using {used_mb}MB")
    return holders


def cmd_quantize(args):
    ensure_dirs()
    payload = _load_payload(args)
    models_dir = os.path.realpath(os.path.expanduser(payload["models_dir"]))
    output_dir = os.path.realpath(os.path.expanduser(payload.get("output_dir") or models_dir))
    source_path = os.path.realpath(os.path.expanduser(payload["source_path"]))
    model_name = payload["model_name"]
    formats = payload.get("formats") or []
    model_type = payload.get("architecture") or "Not set"
    strategy = payload.get("strategy") or "Optimizer-driven"
    optimizer = payload.get("optimizer") or "prodigy"
    low_vram = bool(payload.get("low_vram"))
    is_full = bool(payload.get("full_checkpoint"))
    custom_metadata = payload.get("custom_metadata")  # user-edited metadata from UI
    wm_hint = _watermark_context(payload)

    log_acc = (
        f"Initializing Pipeline for: {model_name}\n"
        f"Target Architecture: {model_type}\n"
        f"Full Checkpoint: {'Yes' if is_full else 'No'}\n"
        f"Output directory: {output_dir}\n"
        f"{wm_hint}\n"
        + "-" * 40 + "\n"
    )
    safe_fmts = [
        f for f in formats
        if f in {
            "FP8",
            "NVFP4",
            "MXFP8",
            "Hybrid MXFP8",
            "INT8 Tensor-wise",
            "INT8 Row-wise ConvRot Runtime",
            "INT8 Row-wise ConvRot",
        }
    ]
    int4_convrot = "INT4 ConvRot Runtime" in formats
    gguf_fmts = [f for f in formats if f.startswith("GGUF_")]
    last_log = ""

    def stream(gen):
        nonlocal last_log
        for log, status in gen:
            delta = log[len(last_log):] if log.startswith(last_log) else log
            last_log = log
            if delta:
                _emit({"type": "log", "text": delta})
            _emit({"type": "status", "status": status})

    if safe_fmts:
        stream(run_safe_conversion(
            output_dir,
            source_path,
            safe_fmts,
            model_name,
            model_type,
            optimizer,
            strategy,
            log_acc,
            low_vram=low_vram,
            is_full_checkpoint=is_full,
            custom_metadata=custom_metadata,
        ))
        log_acc = last_log or log_acc

    if int4_convrot:
        stream(run_int4_convrot_conversion(
            output_dir, source_path, model_name, model_type, strategy, is_full,
            custom_metadata=custom_metadata,
        ))
        log_acc = last_log or log_acc

    if gguf_fmts:
        stream(run_gguf_conversion(
            output_dir,
            source_path,
            gguf_fmts,
            model_name,
            log_acc,
            model_type=model_type,
            is_full=is_full,
        ))

    _emit({"type": "done", "status": "Finished"})


def cmd_lora_merge(args):
    ensure_dirs()
    payload = _load_payload(args)
    _emit({"type": "log", "text": _watermark_context(payload) + "\n"})
    for event in run_lora_merge(payload):
        _emit(event)


def cmd_watermark(args):
    path = os.path.realpath(os.path.expanduser(args.path))
    _emit(wm_verify(path))


def cmd_watermark_key(args):
    passphrase = args.passphrase
    if not passphrase:
        import getpass
        passphrase = getpass.getpass("Watermark passphrase (will be saved to ~/.dasiwa/watermark.key): ")
    if not passphrase:
        _emit({"ok": False, "error": "Passphrase must not be empty."})
        return
    path = wm_save_key(passphrase)
    _emit({
        "ok": True,
        "path": path,
        "note": ("Watermark passphrase saved to " + path +
                 " (0600, outside the repository). Quant outputs will now "
                 "carry an EC-watermarked modelspec.watermark field."),
    })


def cmd_watermark_status(args):
    _emit(wm_status())


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect")
    p.add_argument("path")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("metadata")
    p.add_argument("--name", default="TreasureChest")
    p.add_argument("--architecture", default="WAN 2.2")
    p.add_argument("--full", action="store_true")
    p.set_defaults(func=cmd_metadata)

    p = sub.add_parser("read-metadata")
    p.add_argument("--models-dir", required=True)
    p.add_argument("--file", required=True)
    p.set_defaults(func=cmd_read_metadata)

    p = sub.add_parser("read-metadata-path")
    p.add_argument("path")
    p.set_defaults(func=cmd_read_metadata_path)

    p = sub.add_parser("inject-metadata-path")
    p.add_argument("--json")
    p.set_defaults(func=cmd_inject_metadata_path)

    p = sub.add_parser("scan")
    p.add_argument("path")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("audit")
    p.add_argument("path")
    p.add_argument("--architecture", required=True)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("clean-memory")
    p.set_defaults(func=cmd_clean_memory)

    p = sub.add_parser("quantize")
    p.add_argument("--json")
    p.set_defaults(func=cmd_quantize)

    p = sub.add_parser("lora-merge")
    p.add_argument("--json")
    p.set_defaults(func=cmd_lora_merge)

    p = sub.add_parser("watermark", help="Decode the EC watermark in the modelspec.watermark field of a quant output.")
    p.add_argument("path")
    p.set_defaults(func=cmd_watermark)

    p = sub.add_parser(
        "watermark-key",
        help="Save the DaSiWa watermark passphrase (~/.dasiwa/watermark.key, 0600).",
    )
    p.add_argument("--passphrase", default="",
                   help="Passphrase to save (omit to be prompted securely).")
    p.set_defaults(func=cmd_watermark_key)

    p = sub.add_parser(
        "watermark-status",
        help="Report whether a watermark secret is configured (for the UI).",
    )
    p.set_defaults(func=cmd_watermark_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
