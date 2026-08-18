# core/watermark.py
# Keyed, only-you-can-decode watermark for DaSiWa quant outputs.
#
# The watermark is a 32-byte AES-256-GCM ciphertext of a small JSON payload
# (tool, arch, model name, timestamp, a random nonce, and the SHA-256 of the
# quantised output). It is stored as a hex string under the "dasiwa.watermark"
# metadata key in every DaSiWa quant output (safetensors __metadata__ and
# GGUF custom kv).
#
# Decoding requires the secret key, which is NEVER committed to the repo.
# Key sources (in order):
#   1. DASIWA_WATERMARK_KEY env var
#   2. ~/.dasiwa/watermark.key  (created via `go_bridge.py watermark-key`)
#
# Without the key the ciphertext is opaque — an attacker who sees the hex
# string cannot recover the payload or forge a valid one (GCM authentication
# means any tampering is detected).

import base64
import hashlib
import json
import os
import struct
import time

_WATERMARK_KEY_ENV = "DASIWA_WATERMARK_KEY"
_WATERMARK_KEY_FILE = os.path.expanduser("~/.dasiwa/watermark.key")
_WATERMARK_METADATA_KEY = "dasiwa.watermark"
_TOOL_ID = "dasiwa-quant-station"


def _load_key() -> bytes | None:
    """Return the 32-byte AES key, or None if not configured."""
    env_key = os.environ.get(_WATERMARK_KEY_ENV)
    if env_key:
        # Accept a hex-encoded 32-byte key from the env.
        try:
            raw = bytes.fromhex(env_key)
            if len(raw) == 32:
                return raw
        except ValueError:
            pass
        # Also accept a raw 32-char passphrase hashed to 32 bytes.
        return hashlib.sha256(env_key.encode()).digest()

    if os.path.isfile(_WATERMARK_KEY_FILE):
        try:
            with open(_WATERMARK_KEY_FILE, "rb") as f:
                raw = f.read().strip()
            if len(raw) == 32:
                return raw
            # If the file holds a passphrase, hash it.
            return hashlib.sha256(raw).digest()
        except OSError:
            return None
    return None


def generate_key() -> tuple[bytes, str]:
    """Generate a new random 32-byte key. Returns (key_bytes, hex_string)."""
    key = os.urandom(32)
    return key, key.hex()


def save_key(key: bytes | None = None) -> str:
    """Write a key (or a freshly generated one) to ~/.dasiwa/watermark.key.

    Returns the absolute path.
    """
    if key is None:
        key, _ = generate_key()
    d = os.path.expanduser("~/.dasiwa")
    os.makedirs(d, exist_ok=True)
    path = _WATERMARK_KEY_FILE
    with open(path, "wb") as f:
        f.write(key)
    os.chmod(path, 0o600)
    return path


def _encrypt(key: bytes, payload: dict) -> str:
    """Encrypt a payload dict with AES-256-GCM. Returns a hex string."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(12)
    plaintext = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    # nonce (12) + ciphertext + GCM tag (16, embedded by AESGCM) -> hex
    return (nonce + ciphertext).hex()


def _decrypt(key: bytes, token_hex: str) -> dict | None:
    """Decrypt a watermark hex string. Returns payload dict or None."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    try:
        raw = bytes.fromhex(token_hex)
        if len(raw) < 13:
            return None
        nonce, ciphertext = raw[:12], raw[12:]
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return json.loads(plaintext.decode())
    except Exception:
        return None


def watermark_for(
    architecture: str,
    model_name: str,
    output_path: str | None = None,
    bits: str = "",
) -> dict:
    """Build the watermark payload dict. Returns {"key": "dasiwa.watermark", "value": hex}
    or {} if no key is configured.
    """
    key = _load_key()
    if key is None:
        return {}

    sha256_hex = ""
    if output_path and os.path.isfile(output_path):
        sha256_hex = calculate_sha256(output_path)  # reuse metadata_manager's

    payload = {
        "t": _TOOL_ID,
        "arch": architecture,
        "name": model_name,
        "bits": bits,
        "ts": int(time.time()),
        "nonce": os.urandom(8).hex(),
        "sha": sha256_hex,
    }
    return {_WATERMARK_METADATA_KEY: _encrypt(key, payload)}


def verify_watermark(output_path: str) -> dict:
    """Read the dasiwa.watermark metadata key from a safetensors or GGUF file
    and attempt to decrypt it. Returns a dict with:
      ok: bool          — key was found and decrypted successfully
      source: str       — "safetensors" | "gguf"
      payload: dict    — decoded watermark payload (if ok)
      raw: str         — the hex token (for debugging)
      note: str        — human-readable explanation
    """
    key = _load_key()

    # Try safetensors header first.
    raw_token = None
    source = ""
    try:
        import struct as _struct

        with open(output_path, "rb") as f:
            n = _struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        md = header.get("__metadata__", {})
        raw_token = md.get(_WATERMARK_METADATA_KEY)
        source = "safetensors"
    except (OSError, json.JSONDecodeError, KeyError):
        pass

    # Try GGUF KV (custom field accessed by name string).
    if raw_token is None:
        try:
            import gguf as _gguf

            reader = _gguf.GGUFReader(output_path, "r")
            field = reader.get_field(_WATERMARK_METADATA_KEY)
            if field is not None:
                raw_token = field.contents()
                source = "gguf"
        except Exception:
            pass

    if raw_token is None:
        return {
            "ok": False,
            "source": source or "unknown",
            "payload": None,
            "raw": None,
            "note": "No dasiwa.watermark metadata key found in this file.",
        }

    if key is None:
        return {
            "ok": False,
            "source": source,
            "payload": None,
            "raw": raw_token[:64] + "..." if len(raw_token) > 64 else raw_token,
            "note": (
                "Watermark present but no key configured. "
                "Set DASIWA_WATERMARK_KEY or run `go_bridge.py watermark-key` "
                "to generate one."
            ),
        }

    payload = _decrypt(key, raw_token)
    if payload is None:
        return {
            "ok": False,
            "source": source,
            "payload": None,
            "raw": raw_token[:64] + "..." if len(raw_token) > 64 else raw_token,
            "note": "Decryption failed — wrong key or tampered token.",
        }

    return {
        "ok": True,
        "source": source,
        "payload": payload,
        "raw": raw_token[:64] + "..." if len(raw_token) > 64 else raw_token,
        "note": "Watermark verified.",
    }


def calculate_sha256(file_path: str) -> str:
    """SHA-256 of a file, 0x-prefixed hex (matches metadata_manager's format)."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"0x{h.hexdigest()}"
