#!/usr/bin/env python3
import argparse
import json
import os
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
from core.safetensors_engine import run_safe_conversion
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


def cmd_quantize(args):
    ensure_dirs()
    payload = _load_payload(args)
    models_dir = os.path.realpath(os.path.expanduser(payload["models_dir"]))
    source_path = os.path.realpath(os.path.expanduser(payload["source_path"]))
    model_name = payload["model_name"]
    formats = payload.get("formats") or []
    model_type = payload.get("architecture") or "Not set"
    strategy = payload.get("strategy") or "Optimizer-driven"
    optimizer = payload.get("optimizer") or "prodigy"
    low_vram = bool(payload.get("low_vram"))
    is_full = bool(payload.get("full_checkpoint"))

    log_acc = (
        f"Initializing Pipeline for: {model_name}\n"
        f"Target Architecture: {model_type}\n"
        f"Full Checkpoint: {'Yes' if is_full else 'No'}\n"
        + "-" * 40 + "\n"
    )
    safe_fmts = [
        f for f in formats
        if f in {
            "FP8",
            "NVFP4",
            "INT8 Tensor-wise",
            "INT8 Row-wise ConvRot Runtime",
            "INT8 Row-wise ConvRot",
        }
    ]
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
            models_dir,
            source_path,
            safe_fmts,
            model_name,
            model_type,
            optimizer,
            strategy,
            log_acc,
            low_vram=low_vram,
            is_full_checkpoint=is_full,
        ))
        log_acc = last_log or log_acc

    if gguf_fmts:
        stream(run_gguf_conversion(
            models_dir,
            source_path,
            gguf_fmts,
            model_name,
            log_acc,
            model_type=model_type,
            is_full=is_full,
        ))

    _emit({"type": "done", "status": "Finished"})


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

    p = sub.add_parser("quantize")
    p.add_argument("--json")
    p.set_defaults(func=cmd_quantize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
