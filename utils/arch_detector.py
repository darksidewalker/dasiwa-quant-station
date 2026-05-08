# utils/arch_detector.py
"""
Detects the architecture of a safetensors file by examining its layer keys.

Used as a safety check before quantization: if the user-selected architecture
doesn't match the source file's actual architecture, abort with a clear error.
This prevents the failure mode where a user accidentally leaves "WAN 2.2"
selected while loading an LTX-2.3 file, causing convert_to_quant to apply
the wrong preset and produce a damaged output (every layer FP8'd, including
LTX-specific structural layers that should have been preserved).

The detector reads the safetensors header only (no weight loading) and
matches against marker patterns characteristic of each architecture.
"""
import os
import re


# Marker patterns: substrings or regex fragments that uniquely identify
# the architecture. Source: real layer keys from uploaded WAN 2.2 i2v 14B
# and LTX-2.3 22B templates.
#
# Each architecture has multiple markers; we require at least one to match
# to claim a confident detection. If multiple architectures match, that's
# treated as ambiguous (shouldn't happen in practice given how distinct
# the naming conventions are).
ARCH_MARKERS = {
    "LTX-2.3": [
        # LTX-2.3 keys are prefixed with model.diffusion_model.
        re.compile(r"^model\.diffusion_model\.transformer_blocks\."),
        # LTX-2 specific: audio-video cross-attention (the AV variant)
        re.compile(r"\.adaln_single\."),
        re.compile(r"_embeddings_connector\."),
        # Diffusers attention naming
        re.compile(r"transformer_blocks\.\d+\.attn\d+\.to_v"),
    ],
    "WAN 2.2": [
        # WAN keys are naked (no model.diffusion_model. prefix).
        # Split q/k/v/o (never fused) is the giveaway.
        re.compile(r"^blocks\.\d+\.self_attn\.[qkvo]$"),
        re.compile(r"^blocks\.\d+\.cross_attn\.[qkvo]$"),
        re.compile(r"^text_embedding\.\d+\."),
        re.compile(r"^time_projection\.\d+\."),
        re.compile(r"^head\.head\."),
    ],
}


def _read_keys_only(safetensors_path, max_keys=None):
    """Read just the layer key list from a safetensors header. No weights."""
    from safetensors import safe_open
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    if max_keys is not None:
        keys = keys[:max_keys]
    return keys


def detect_architecture(safetensors_path):
    """
    Inspect the safetensors header and return the detected architecture name.
    
    Returns:
        (arch_name, confidence_log) where:
          - arch_name is "LTX-2.3", "WAN 2.2", "UNKNOWN", or "AMBIGUOUS"
          - confidence_log is a list of strings explaining the detection
    
    Errors raised by file I/O are propagated to the caller.
    """
    keys = _read_keys_only(safetensors_path)
    if not keys:
        return "UNKNOWN", [f"No keys found in {os.path.basename(safetensors_path)}"]
    
    matches = {}  # arch -> [(marker_index, hit_count), ...]
    for arch, patterns in ARCH_MARKERS.items():
        hits = []
        for i, rx in enumerate(patterns):
            n = sum(1 for k in keys if rx.search(k))
            if n > 0:
                hits.append((i, n))
        if hits:
            matches[arch] = hits
    
    log = [f"Inspected {len(keys)} layer keys"]
    
    if not matches:
        log.append("No known architecture markers matched")
        return "UNKNOWN", log
    
    if len(matches) == 1:
        arch = next(iter(matches))
        n_markers = len(matches[arch])
        total_hits = sum(n for _, n in matches[arch])
        log.append(f"Detected {arch}: {n_markers} marker pattern(s) hit, "
                   f"{total_hits} layer(s) matched")
        return arch, log
    
    # Multiple architectures matched - report which won and by how much
    log.append("⚠️  Multiple architectures matched (ambiguous):")
    for arch, hits in matches.items():
        total = sum(n for _, n in hits)
        log.append(f"  - {arch}: {len(hits)} pattern(s), {total} layer hits")
    return "AMBIGUOUS", log


def verify_architecture_match(safetensors_path, declared_arch):
    """
    Verify the user's declared architecture matches what's actually in the file.
    
    Returns:
        (ok: bool, message: str)
    
    ok=True means safe to proceed (match, or unknown which we allow).
    ok=False means abort the batch.
    """
    try:
        detected, log = detect_architecture(safetensors_path)
    except Exception as e:
        return False, f"Failed to read source file header: {e}"
    
    log_text = " | ".join(log)
    
    if detected == "UNKNOWN":
        # Unknown architecture - allow with a notice. Could be a future model
        # variant that doesn't match our markers. The user takes responsibility.
        return True, (
            f"⚠️  Could not auto-detect architecture from layer keys. "
            f"Proceeding with user-selected '{declared_arch}'. "
            f"({log_text})"
        )
    
    if detected == "AMBIGUOUS":
        # Markers from multiple architectures matched - this is suspicious.
        # Don't abort, but warn loudly.
        return True, (
            f"⚠️  Source file matched markers from multiple architectures. "
            f"Proceeding with user-selected '{declared_arch}', but verify the file. "
            f"({log_text})"
        )
    
    if detected == declared_arch:
        return True, f"✅ Architecture verified: {detected}"
    
    # Mismatch: hard abort
    return False, (
        f"❌ ARCHITECTURE MISMATCH\n"
        f"   You selected: {declared_arch}\n"
        f"   File appears to be: {detected}\n"
        f"   ({log_text})\n"
        f"   Change the Architecture Selection radio button to '{detected}'\n"
        f"   and click START again. Aborting to prevent damaged output."
    )
