# config.py
import os

# --- DIRECTORIES ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# --- BINARIES & TOOLS ---
LLAMA_BIN = os.path.join(ROOT_DIR, "llama.cpp", "build", "bin", "llama-quantize")
