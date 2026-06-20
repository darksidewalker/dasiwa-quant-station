# Agent Guidance for DaSiWa Quant Station

This document explains the expected behavior and conventions for automated agents (AI assistants) working on this repository.

## Architecture

- **Go server** (`cmd/dasiwa/main.go`, `internal/app/server.go`) serves a static frontend at `http://127.0.0.1:7878`.
- **Python bridge** (`scripts/go_bridge.py`) is invoked by the Go server for all quantization, LoRA merge, scan, and audit operations.
- **Core engines** live in `core/` — gguf_engine, safetensors_engine, lora_merge_engine, layer_config_builder, metadata_manager, metadata_configs.
- **Utilities** live in `utils/` — arch_detector, lora_inspector, layer profiles, scanner, audit, system monitoring, file_ops.
- **Frontend** is vanilla JS/HTML/CSS in `web/`.

## Pathing

- Use `config.MODELS_DIR` as the canonical source-of-truth for model storage paths.
- The Go server provides filesystem browser endpoints (`/api/browse`, `/api/files`) — no Python file listing is needed.

## Long-running Processes

- Use `subprocess.Popen` or streaming approaches for long-running conversions so the Go UI remains responsive.
- Prefer the existing patterns in `core/gguf_engine.py` and `core/safetensors_engine.py`.
- Go streams progress via SSE to the browser.

## Editing Guidance

- When changing architecture registries or UI labels, update all three places: `core/safetensors_engine.ARCH_REGISTRY`, Go API format list in `internal/app/server.go`, and metadata templates in `core/metadata_configs.py`.
- Avoid altering unrelated files in the same commit; make small, focused edits and run the app locally to verify behavior.

## Verification

- Run `python3 -m pytest tests/ -v` after any core engine changes.
- Check `logs/`, browser network/SSE output, and the Go server terminal for debugging.
