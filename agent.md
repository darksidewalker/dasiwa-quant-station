# Agent Guidance for DaSiWa Quant Station

This document explains the expected behavior and conventions for automated agents (AI assistants) working on this repository.

## UI & Pathing
- The UI exposes a single `Model Directory` dropdown (`models_dir_dd`) that lists the repository `MODELS_DIR` and its immediate subfolders. Agents should prefer this dropdown pattern when referencing or populating UI tests.
- Do not attempt to treat the UI as an upload interface. The app operates on files already present on the host filesystem under `MODELS_DIR`.
- Use `config.MODELS_DIR` as the canonical source-of-truth for model storage paths. Helper `utils.file_ops.list_dirs()` returns a `gr.update` suitable to populate the directory dropdown.

## Callback Conventions
- Callbacks expect a directory path string (not an uploaded file object). If a callback receives a file path, the code normalizes it to its containing directory.
- The primary file dropdowns are populated from `utils.file_listing.get_model_list(models_dir)`.

## Long-running Processes
- Use `subprocess.Popen` or streaming approaches for long-running conversions so UI remains responsive; prefer the existing patterns in `core/gguf_engine.py` and `core/safetensors_engine.py`.

## Editing Guidance
- When changing architecture registries or UI labels, update all three places: `core/safetensors_engine.ARCH_REGISTRY`, `ui/layout.py` Dropdown choices, and `ui/assets.py` `MODEL_METADATA_CONFIGS` keys.
- Avoid altering unrelated files in the same commit; make small, focused edits and run the app locally to verify UI wiring.

## Verification
- To validate directory-dropdown behavior: run the app and ensure `Model Directory` lists `MODELS_DIR` and subfolders; selecting one should populate the `Safetensors file` dropdown.

## Contacts & Credits
- See `README.md` for project credits and external dependencies.

