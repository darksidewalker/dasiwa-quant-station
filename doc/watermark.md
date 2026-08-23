# Provenance Watermark

Back to [README](../README.md)

Every quantized and LoRA-merged output carries an EC-based provenance token
in the `modelspec.watermark` field. No plaintext author string is written,
and the rest of the custom metadata is left untouched — only
`modelspec.watermark` is added.

## Scheme

- **Scheme:** ephemeral X25519 (ECIES) wrapping an AES-256-GCM ciphertext. The static key is derived from your passphrase via PBKDF2-HMAC-SHA256 (clamped to a valid X25519 scalar). A fresh ephemeral key is generated per output, so every token is unique and only you can decode it.
- **Payload:** tool, architecture, model name, bit width, timestamp, a random nonce, and the SHA-256 of the output (when the file exists).
- **Decode:** with the correct passphrase the token decodes to the provenance payload; a wrong or tampered token fails (GCM authentication). Without any configured secret the field is simply not written (clean no-op).

## Secret resolution (first hit wins)

1. `DASIWA_WATERMARK_PASSPHRASE` (environment)
2. `DASIWA_WATERMARK_KEY` (environment; 64-hex pre-derived key or a passphrase)
3. `~/.dasiwa/watermark.key` (0600, written by `go_bridge.py watermark-key`)

**Passphrase location:** kept in your environment / a 0600 key file
**outside the repository** — never committed to Gitea or GitHub.

## UI

A **Watermark outputs** checkbox (on by default, shared by Quantize and LoRA
modes) toggles watermarking per run; a live hint below it tells you whether a
key is available, whether no key is configured (no token written), or that
watermarking is off for this run. Unchecking it sets a per-job kill switch so
that run's outputs skip `modelspec.watermark`.

**Status:** `GET /api/watermark` (and `go_bridge.py watermark-status`)
reports whether a secret is resolvable — without ever returning the secret
value.

## Commands

```bash
# Persist the passphrase (0600, outside the repo)
python scripts/go_bridge.py watermark-key --passphrase "your-passphrase"

# Decode the watermark in a quant output (safetensors or GGUF)
python scripts/go_bridge.py watermark path/to/output.safetensors

# Check whether a watermark key is currently configured (for the UI)
python scripts/go_bridge.py watermark-status
```
