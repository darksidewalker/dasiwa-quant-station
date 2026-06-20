import os
import psutil
import platform
import re
import shutil
import subprocess

try:
    import torch
except Exception:
    torch = None


def _run_probe(cmd):
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout.strip()
    except Exception:
        return ""


def _find_command(names, fallback_paths=None):
    for name in names:
        path = shutil.which(name)
        if path:
            return path

    for path in fallback_paths or []:
        if shutil.which(path) or os.path.exists(path):
            return path

    return None


def _format_gb(value_mb):
    try:
        return float(value_mb) / 1024
    except (TypeError, ValueError):
        return None


def _extract_metric_number(line):
    metric_text = line.rsplit(":", 1)[-1]
    match = re.search(r"(\d+(?:\.\d+)?)\s*%?", metric_text)
    if not match:
        return None
    return round(float(match.group(1)))


def _get_nvidia_info():
    nvidia_smi = _find_command(
        ["nvidia-smi"],
        fallback_paths=[
            "/usr/bin/nvidia-smi",
            "/usr/local/bin/nvidia-smi",
            "/usr/lib/nvidia-smi",
        ],
    )
    if not nvidia_smi:
        return None

    output = _run_probe([
        nvidia_smi,
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if not output:
        return None

    utils = []
    used_total_gb = 0.0
    total_total_gb = 0.0
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            utils.append(int(parts[0]))
        except ValueError:
            utils.append(None)

        used_gb = _format_gb(parts[1])
        total_gb = _format_gb(parts[2])
        if used_gb is not None and total_gb is not None:
            used_total_gb += used_gb
            total_total_gb += total_gb

    if not utils and not total_total_gb:
        return None

    if not utils or all(val is None for val in utils):
        gpu_load = "Active"
    elif len(utils) == 1:
        gpu_load = f"{utils[0]}%" if utils[0] is not None else "Active"
    else:
        gpu_load = " ".join(
            f"G{i}:{val}%" if val is not None else f"G{i}:Active"
            for i, val in enumerate(utils)
        )
    vram_info = (
        f"{used_total_gb:.1f}/{total_total_gb:.1f}GB"
        if total_total_gb
        else "Active"
    )
    return gpu_load, vram_info


def _get_rocm_info():
    rocm_cmd = _find_command(
        ["rocm-smi", "rocm-smi.py", "roc-smi"],
        fallback_paths=[
            "/usr/bin/rocm-smi",
            "/usr/bin/rocm-smi.py",
            "/opt/rocm/bin/rocm-smi",
            "/opt/rocm/bin/rocm-smi.py",
        ],
    )
    if not rocm_cmd:
        return None

    output = _run_probe([rocm_cmd, "--showuse", "--showmemuse"])
    if not output:
        return None

    utils = []
    mem_utils = []
    for line in output.splitlines():
        lower = line.lower()
        if "gpu use" in lower or "gpu busy" in lower:
            metric = _extract_metric_number(line)
            if metric is not None:
                utils.append(metric)
        elif "memory use" in lower or "vram" in lower:
            metric = _extract_metric_number(line)
            if metric is not None:
                mem_utils.append(metric)

    if not utils:
        return None

    gpu_load = (
        f"{utils[0]}%"
        if len(utils) == 1
        else " ".join(f"G{i}:{val}%" for i, val in enumerate(utils))
    )
    vram_info = (
        f"{mem_utils[0]}%"
        if len(mem_utils) == 1
        else " ".join(f"G{i}:{val}%" for i, val in enumerate(mem_utils))
        if mem_utils
        else "Active"
    )
    return gpu_load, vram_info


def _get_torch_vram_info():
    if torch is None or not torch.cuda.is_available():
        return None

    try:
        free_b, total_b = torch.cuda.mem_get_info()
        used_gb = (total_b - free_b) / (1024**3)
        total_gb = total_b / (1024**3)
        return f"{used_gb:.1f}/{total_gb:.1f}GB"
    except Exception:
        return None


def _get_gpu_info():
    vendor_info = _get_nvidia_info() or _get_rocm_info()
    if vendor_info:
        return vendor_info

    vram_info = _get_torch_vram_info()
    if vram_info:
        return "Active", vram_info

    return "Active", "Active"



