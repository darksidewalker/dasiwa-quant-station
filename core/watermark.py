# core/watermark.py
# EC-based, only-you-can-decode watermark for DaSiWa quant outputs.
#
# Design:
#   * The ``modelspec.watermark`` metadata field carries the watermark token.
#     No plaintext author string is written into quant outputs at all.
#   * The token is an ephemeral X25519 key wrapped around an AES-256-GCM
#     ciphertext (standard ECIES pattern). Only a holder of the secret key
#     can decrypt it. The ephemeral public key is random for every output,
#     so each watermarked file carries a unique, individually identifiable
#     token.
#   * The secret key is derived from a user passphrase (PBKDF2-HMAC-SHA256).
#     The passphrase itself is read from the environment and is NOT stored
#     in the repository.
#
# Passphrase / key resolution order (first hit wins):
#   1. env DASIWA_WATERMARK_PASSPHRASE
#   2. env DASIWA_WATERMARK_KEY      (32 raw hex bytes, pre-derived key)
#   3. ~/.dasiwa/watermark.key       (created via `go_bridge.py watermark-key`)
#      containing either a raw passphrase line or `key:<64 hex>`
#
# When nothing is configured, ``watermark_for`` returns {} and no author
# field is added at all (clean no-op, no plaintext author leaks).
#
# Token layout (bytes):  32 B ephemeral X25519 pubkey || 16 B salt || 12 B nonce || ciphertext(+16 tag)
# Encoded as base64url in metadata, prefixed with "dswm1.".

import base64
import hashlib
import json
import os
import secrets
import time

try:
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    _CIPHER_OK = True
except ImportError:  # pragma: no cover
    _CIPHER_OK = False

_WATERMARK_KEY_FILE = os.path.expanduser("~/.dasiwa/watermark.key")
_WATERMARK_FIELD = "modelspec.watermark"
_TOKEN_PREFIX = "dswm1."
_SALT_LEN = 16
_NONCE_LEN = 12
_PBKDF2_ITERATIONS = 310_000
_TOKEN_VERSION = "1"


def _derive_key(passphrase: bytes, salt: bytes) -> bytes:
    """Derive a 32-byte AES-256 key from a passphrase + random salt."""
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt,
                     iterations=_PBKDF2_ITERATIONS)
    return kdf.derive(passphrase)


def _x25519_clamp(raw: bytes) -> bytes:
    """
    Clamp a 32-byte scalar into a valid X25519 private scalar (RFC 7748 §5).

    Clears the low 3 bits of the first byte, clears the top bit, and sets
    bit 254. Applied to the passphrase-derived static key so the X25519 DH
    uses a spec-valid scalar on both the encrypt and decrypt sides.
    """
    b = bytearray(raw)
    b[0] &= 0xFC          # clear low 3 bits
    b[31] &= 0x7F         # clear top bit
    b[31] |= 0x40         # set bit 254
    return bytes(b)


def _load_secret():
    """
    Resolve the watermark secret: returns a passphrase string, or a
    pre-derived 32-byte hex key (detected via the ``key:`` prefix / 64-hex
    env ``DASIWA_WATERMARK_KEY``).

    Returns None when nothing is configured.
    """
    pp = os.environ.get("DASIWA_WATERMARK_PASSPHRASE", "").strip()
    if pp:
        return pp

    key_hex = os.environ.get("DASIWA_WATERMARK_KEY", "").strip()
    if key_hex:
        if key_hex.startswith("key:"):
            key_hex = key_hex[len("key:"):]
        if len(key_hex) == 64 and all(c in "0123456789abcdefABCDEF" for c in key_hex):
            return "key:" + key_hex.lower()
        # Treat a non-hex env value as a passphrase.
        return key_hex

    if os.path.isfile(_WATERMARK_KEY_FILE):
        try:
            with open(_WATERMARK_KEY_FILE, "r", encoding="utf-8") as fh:
                content = fh.read().strip()
            if content:
                return content
        except Exception:
            return None
    return None


def _ecies_encrypt(key: bytes, payload_bytes: bytes) -> bytes:
    """
    Ephemeral X25519 + AES-256-GCM (ECIES).

    An ephemeral X25519 pair is generated per token; the shared secret is
    ``x25519(ephemeral_priv, static_pub)``. Only the holder of the static
    private key (derived from the passphrase) can unwrap it. The ephemeral
    public key embedded in the token is what makes every output unique.

    Layout: ephemeral_pub(32) || salt(16) || nonce(12) || gcm_ciphertext(+tag)
    """
    ephemeral = X25519PrivateKey.generate()
    static_priv = X25519PrivateKey.from_private_bytes(key)
    static_pub = static_priv.public_key()
    shared = ephemeral.exchange(static_pub)

    salt = os.urandom(_SALT_LEN)
    nonce = os.urandom(_NONCE_LEN)
    # Bind the per-message AES key to the shared secret + salt so two
    # watermarks of the same file are still independent.
    aes_key = hashlib.sha256(b"dasiwa.ecies.v1" + shared + salt).digest()
    ct = AESGCM(aes_key).encrypt(nonce, payload_bytes, None)

    eph_pub = ephemeral.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return eph_pub + salt + nonce + ct


def _ecies_decrypt(key: bytes, token_bytes: bytes) -> bytes:
    """Reverse of _ecies_encrypt. Returns payload bytes or raises."""
    ephemeral_pub = X25519PublicKey.from_public_bytes(token_bytes[:32])
    salt = token_bytes[32:48]
    nonce = token_bytes[48:60]
    ciphertext = token_bytes[60:]

    static_priv = X25519PrivateKey.from_private_bytes(key)
    shared = static_priv.exchange(ephemeral_pub)
    aes_key = hashlib.sha256(b"dasiwa.ecies.v1" + shared + salt).digest()
    return AESGCM(aes_key).decrypt(nonce, ciphertext, None)


def _file_sha256(file_path: str) -> str:
    """SHA-256 of the file (hex). Returns '' when the file is missing."""
    if not file_path or not os.path.isfile(file_path):
        return ""
    h = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_payload(architecture: str, model_name: str, file_path,
                   bits: str) -> dict:
    return {
        "v": _TOKEN_VERSION,
        "t": "dasiwa-quant-station",
        "arch": architecture,
        "name": model_name,
        "bits": bits,
        "nonce": secrets.token_hex(8),
        "ts": int(time.time()),
        "sha": _file_sha256(file_path),
    }


def _static_key_from_secret(secret: str):
    """
    Build the 32-byte X25519 static private scalar from the resolved secret.

    - ``key:<64 hex>``  -> the 32 raw bytes, clamped to a valid scalar
    - plain passphrase  -> PBKDF2 (fixed salt) 32-byte key, clamped

    Returns a 32-byte ``bytes`` or ``None`` when malformed.
    """
    if secret.startswith("key:"):
        raw_hex = secret[4:]
        try:
            key = bytes.fromhex(raw_hex)
        except ValueError:
            return None
        if len(key) != 32:
            return None
        return _x25519_clamp(key)
    key = _derive_key(secret.encode("utf-8"), b"dasiwa.static-key.v1")
    if key is None or len(key) != 32:
        return None
    return _x25519_clamp(key)


def _token_hex_for(payload: dict, key: bytes) -> str:
    payload_bytes = json.dumps(payload, separators=(",", ":"),
                              sort_keys=True).encode("utf-8")
    blob = _ecies_encrypt(key, payload_bytes)
    return _TOKEN_PREFIX + base64.urlsafe_b64encode(blob).decode("ascii")


def _token_payload_from_hex(token_hex: str, key: bytes):
    """
    Decrypt a watermark token (without or with the ``dswm1.`` prefix).

    Returns the payload dict, or None when the secret/key does not match
    (wrong passphrase, tampered token, or legacy token format).
    """
    if not token_hex:
        return None
    body = token_hex
    if body.startswith(_TOKEN_PREFIX):
        body = body[len(_TOKEN_PREFIX):]
    try:
        blob = base64.urlsafe_b64decode(body.encode("ascii"))
    except Exception:
        return None
    if len(blob) < 48 + 16:  # pubkey+salt+nonce+min tag
        return None
    try:
        payload_bytes = _ecies_decrypt(key, blob)
        payload = json.loads(payload_bytes.decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _watermark_disabled() -> bool:
    """
    Per-run kill switch. The bridge sets DASIWA_WATERMARK_DISABLED=1 when the
    user opts out of watermarking for this job. Reading (verifying) existing
    watermarks is unaffected; this only stops new tokens from being written.
    """
    val = os.environ.get("DASIWA_WATERMARK_DISABLED", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def watermark_for(architecture: str, model_name: str, file_path: str = None,
                  bits: str = "FP8") -> dict:
    """
    Build the EC-watermarked author metadata.

    Returns:
      {}                     when watermarking is disabled / no secret
      {"modelspec.watermark": "dswm1.<hex>"}  when a secret is available
    """
    if _watermark_disabled():
        return {}
    secret = _load_secret()
    if secret is None:
        return {}
    if not _CIPHER_OK:
        return {}

    payload = _build_payload(architecture, model_name, file_path, bits)
    static_key = _static_key_from_secret(secret)
    if static_key is None:
        return {}

    token_hex = _token_hex_for(payload, static_key)
    return {_WATERMARK_FIELD: token_hex}


def verify_watermark(output_path: str) -> dict:
    """
    Read the ``modelspec.watermark`` field from a safetensors or GGUF file
    and attempt to decode it.

    Returns a dict:
      ok: bool          — token present and successfully decoded
      payload: dict|None — decoded watermark payload (None on failure)
      field: str        — raw watermark field value (or '' when absent)
      note: str         — human-readable explanation
    """
    meta = {}
    if not output_path or not os.path.isfile(output_path):
        return {
            "ok": False,
            "payload": None,
            "field": "",
            "note": f"File not found: {output_path}",
        }

    if output_path.endswith(".safetensors"):
        try:
            import struct as _struct
            with open(output_path, "rb") as fh:
                size_prefix = fh.read(8)
                if len(size_prefix) != 8:
                    return {"ok": False, "payload": None, "field": "",
                            "note": "File too small to be a valid safetensors file"}
                header_size = _struct.unpack("<Q", size_prefix)[0]
                if header_size <= 0 or header_size > 64 * 1024 * 1024:
                    return {"ok": False, "payload": None, "field": "",
                            "note": "Invalid header size"}
                header = json.loads(fh.read(header_size).decode("utf-8"))
            meta = header.get("__metadata__", {}) or {}
        except Exception as e:
            return {"ok": False, "payload": None, "field": "",
                    "note": f"Could not read safetensors header: {e}"}
    elif output_path.endswith(".gguf"):
        try:
            from gguf import GGUFReader
            reader = GGUFReader(output_path, "r")
            # The watermark is written as a plain string field named
            # ``modelspec.watermark`` (see write_gguf_meta -> add_string).
            # Fall back to ``general.author`` for legacy files.
            for field_name in ("modelspec.watermark", "general.author"):
                f = reader.get_field(field_name)
                if f is not None:
                    try:
                        val = f.contents()
                    except Exception:
                        val = f.parts[0].tobytes().decode("utf-8", "ignore")
                    if isinstance(val, (bytes, bytearray)):
                        val = val.decode("utf-8", "ignore").strip("\x00")
                    if val:
                        meta[_WATERMARK_FIELD] = val
                        break
        except Exception as e:
            return {"ok": False, "payload": None, "field": "",
                    "note": f"Could not read GGUF metadata: {e}"}
    else:
        return {"ok": False, "payload": None, "field": "",
                "note": f"Unsupported file type: {output_path}"}

    wm_val = meta.get(_WATERMARK_FIELD, "")
    if not wm_val:
        return {
            "ok": False,
            "payload": None,
            "field": "",
            "note": "No modelspec.watermark watermark found in this file.",
        }

    secret = _load_secret()
    if secret is None:
        return {
            "ok": False,
            "payload": None,
            "field": wm_val,
            "note": ("Watermark token present but no passphrase is configured "
                     "(set DASIWA_WATERMARK_PASSPHRASE or run "
                     "`go_bridge.py watermark-key`)."),
        }
    if not _CIPHER_OK:
        return {
            "ok": False,
            "payload": None,
            "field": wm_val,
            "note": "cryptography package is required to decode EC watermarks.",
        }

    static_key = _static_key_from_secret(secret)
    if static_key is None:
        return {"ok": False, "payload": None, "field": wm_val,
                "note": "Malformed DASIWA_WATERMARK_KEY (expected 64 hex chars)."}

    payload = _token_payload_from_hex(wm_val, static_key)
    if payload is None:
        return {
            "ok": False,
            "payload": None,
            "field": wm_val,
            "note": "Decryption failed — wrong passphrase, tampered token, or legacy format.",
        }
    payload["watermark_field"] = wm_val
    return {
        "ok": True,
        "payload": payload,
        "field": wm_val,
        "note": "EC watermark (X25519 + AES-256-GCM) decoded successfully.",
    }


def save_key(passphrase: str) -> str:
    """
    Persist the passphrase (plaintext, 0600) to ``~/.dasiwa/watermark.key``
    for later automatic use. The passphrase stays OUT of the repository;
    this file only lives in $HOME.

    Returns the absolute path written.
    """
    os.makedirs(os.path.dirname(_WATERMARK_KEY_FILE), exist_ok=True)
    with open(_WATERMARK_KEY_FILE, "w", encoding="utf-8") as fh:
        fh.write(passphrase)
    os.chmod(_WATERMARK_KEY_FILE, 0o600)
    return _WATERMARK_KEY_FILE


def watermark_status() -> dict:
    """
    Report whether a watermark secret is currently resolvable, for the UI.

    Returns a dict with:
      available: bool      — a secret is configured and usable
      cipher_ok: bool       — the cryptography package is importable
      source: str           — where the secret was found ("env" / "key_file"
                              / "" when none). Never the secret value itself.
      note: str            — human-readable guidance when not available
    """
    secret = _load_secret()
    if secret is None:
        return {
            "available": False,
            "cipher_ok": _CIPHER_OK,
            "source": "",
            "note": ("No watermark key configured. Set DASIWA_WATERMARK_PASSPHRASE "
                     "or run `go_bridge.py watermark-key` to add provenance "
                     "watermarks to quant outputs."),
        }
    if not _CIPHER_OK:
        source = "key_file"
        if os.environ.get("DASIWA_WATERMARK_PASSPHRASE") or os.environ.get("DASIWA_WATERMARK_KEY"):
            source = "env"
        return {
            "available": False,
            "cipher_ok": False,
            "source": source,
            "note": "cryptography package is required to use EC watermarks.",
        }
    source = "env"
    if not (os.environ.get("DASIWA_WATERMARK_PASSPHRASE") or os.environ.get("DASIWA_WATERMARK_KEY")):
        source = "key_file"
    return {
        "available": True,
        "cipher_ok": True,
        "source": source,
        "note": "Watermark secret available; quant outputs will carry modelspec.watermark.",
    }
