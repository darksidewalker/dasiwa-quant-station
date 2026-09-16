#!/usr/bin/env bash
set -e

# --- 1. CONFIGURATION ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_DIR/.venv"
LLAMA_DIR="$PROJECT_DIR/llama.cpp"
PATCH_FILE="$PROJECT_DIR/lcpp.patch"
MODE="${1:---go}"

echo "📂 Project Root: $PROJECT_DIR"
echo "🧭 Launch Mode: $MODE"

# --- 1.5 MULTI-DISTRO DEPENDENCIES ---
# Prebuilt Python wheels mean Quant Station no longer needs to compile
# comfy-kitchen locally. Only install missing launcher/build commands; the
# in-app update path must not repeatedly invoke the system package manager.
MISSING_COMMANDS=()
for cmd in curl unzip go; do
    command -v "$cmd" &> /dev/null || MISSING_COMMANDS+=("$cmd")
done

if [ "${#MISSING_COMMANDS[@]}" -gt 0 ] && [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "🔍 Detected System: $NAME"
    echo "📦 Missing required commands: ${MISSING_COMMANDS[*]}"
    case "$ID" in
        arch|manjaro)
            sudo pacman -S --needed --noconfirm curl unzip go
            ;;
        ubuntu|debian|mint)
            sudo apt update
            sudo apt install -y curl unzip golang-go
            ;;
        *)
            echo "❌ Unsupported distribution ($ID). Install curl, unzip, and Go, then retry."
            exit 1
            ;;
    esac
elif [ "${#MISSING_COMMANDS[@]}" -gt 0 ]; then
    echo "❌ Missing required commands: ${MISSING_COMMANDS[*]}"
    exit 1
else
    echo "✅ System prerequisites are already installed."
fi

if ! command -v uv &> /dev/null; then
    echo "⚙️ uv not found. Installing via official script..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
else
    echo "✅ uv is already installed."
fi

# The official standalone install can update itself. Distribution-packaged uv
# builds reject this operation; keep those under the OS package manager instead.
echo "📦 Checking uv for updates..."
if ! uv self update; then
    echo "⚠️ uv self-update unavailable; keeping the package-manager version."
fi

# --- 2. GGUF ENGINE SETUP ---

# --- 2.5 UPDATE GGUFY BINARY ---
GGUFY_BIN="$PROJECT_DIR/bin/ggufy"
GGUFY_VERSION_FILE="$PROJECT_DIR/bin/ggufy.version"
GGUFY_STAGE="$PROJECT_DIR/bin/.ggufy-update"

ARCH=$(uname -m)
case "$ARCH" in
    x86_64) GGUFY_ASSET="ggufy-linux-x86_64" ;;
    aarch64) GGUFY_ASSET="ggufy-linux-arm64" ;;
    *)
        echo "❌ GGUFY has no configured Linux asset for architecture: $ARCH"
        exit 1
        ;;
esac

GGUFY_RELEASE_URL=$(curl -LsS --fail -o /dev/null -w '%{url_effective}' \
    "https://github.com/qskousen/ggufy/releases/latest")
GGUFY_LATEST_VERSION="${GGUFY_RELEASE_URL##*/}"
GGUFY_INSTALLED_VERSION=""
[ -f "$GGUFY_VERSION_FILE" ] && GGUFY_INSTALLED_VERSION=$(<"$GGUFY_VERSION_FILE")

GGUFY_VALID=false
if [ -f "$GGUFY_BIN" ]; then
    FILE_SIZE=$(stat -c%s "$GGUFY_BIN" 2>/dev/null || echo 0)
    if [ "$FILE_SIZE" -ge 500000 ] && head -c 4 "$GGUFY_BIN" | grep -q $'\x7fELF' 2>/dev/null; then
        GGUFY_VALID=true
    fi
fi

if [ "$GGUFY_VALID" != true ] || [ "$GGUFY_INSTALLED_VERSION" != "$GGUFY_LATEST_VERSION" ]; then
    echo "📥 Updating GGUFY ${GGUFY_INSTALLED_VERSION:-unknown} → $GGUFY_LATEST_VERSION..."
    rm -rf "$GGUFY_STAGE"
    mkdir -p "$GGUFY_STAGE"
    GGUFY_ARCHIVE="$GGUFY_STAGE/${GGUFY_ASSET}.zip"

    curl -L --fail --retry 3 --retry-delay 2 -o "$GGUFY_ARCHIVE" \
        "https://github.com/qskousen/ggufy/releases/download/${GGUFY_LATEST_VERSION}/${GGUFY_ASSET}.zip"
    unzip -q -o "$GGUFY_ARCHIVE" -d "$GGUFY_STAGE/extracted"
    GGUFY_CANDIDATE=$(find "$GGUFY_STAGE/extracted" -type f \
        \( -name "ggufy" -o -name "$GGUFY_ASSET" \) -print -quit)

    if [ -z "$GGUFY_CANDIDATE" ]; then
        echo "❌ GGUFY extraction failed: binary not found in archive."
        rm -rf "$GGUFY_STAGE"
        exit 1
    fi

    FILE_SIZE=$(stat -c%s "$GGUFY_CANDIDATE" 2>/dev/null || echo 0)
    if [ "$FILE_SIZE" -lt 500000 ] || ! head -c 4 "$GGUFY_CANDIDATE" | grep -q $'\x7fELF' 2>/dev/null; then
        echo "❌ Downloaded GGUFY binary failed validation."
        rm -rf "$GGUFY_STAGE"
        exit 1
    fi

    chmod +x "$GGUFY_CANDIDATE"
    mkdir -p "$PROJECT_DIR/bin"
    mv -f "$GGUFY_CANDIDATE" "$GGUFY_BIN"
    printf '%s\n' "$GGUFY_LATEST_VERSION" > "$GGUFY_VERSION_FILE"
    rm -rf "$GGUFY_STAGE"
    echo "✅ GGUFY $GGUFY_LATEST_VERSION installed."
else
    echo "✅ GGUFY $GGUFY_INSTALLED_VERSION is current."
fi

# --- 3. LOCAL VENV SETUP ---
if [ ! -x "$VENV_PATH/bin/python" ]; then
    echo "⚙️ Creating local Python 3.12 environment..."
    uv venv --python 3.12 "$VENV_PATH"
fi

# Resolve the complete environment once. uv.lock records the compatible set;
# --upgrade refreshes every direct and transitive package before syncing it.
# Keeping comfy-kitchen on its published wheel avoids a local CUDA/CMake build,
# and omitting its cublas extra lets Torch own the compatible cuBLAS version.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
export UV_RETRY_COUNT="${UV_RETRY_COUNT:-5}"
export UV_MAX_CONCURRENT_DOWNLOADS="${UV_MAX_CONCURRENT_DOWNLOADS:-2}"

echo "📦 Upgrading and syncing the Python environment..."
uv sync --upgrade --no-dev --python "$VENV_PATH/bin/python"
uv pip check --python "$VENV_PATH/bin/python"

"$VENV_PATH/bin/python" - <<'PY'
from importlib.metadata import version

import torch
from comfy_kitchen.tensor import AsymW4A8Int8Layout, TensorCoreConvRotW4A4Layout
import convert_to_quant

print(f"✅ Torch {torch.__version__} (CUDA {torch.version.cuda or 'not available'})")
print(f"✅ convert-to-quant {version('convert-to-quant')}")
print(f"✅ comfy-kitchen {version('comfy-kitchen')}")
print("✅ INT4/W4A8 layouts:", TensorCoreConvRotW4A4Layout.__name__, AsymW4A8Int8Layout.__name__)
PY

# --- 4. LAUNCH ---
export VIRTUAL_ENV="$VENV_PATH"
export PATH="$VENV_PATH/bin:$PATH"

if [ "$MODE" = "--setup-only" ]; then
    echo "✅ Setup complete."
    exit 0
fi

if ! command -v go &> /dev/null; then
    echo "❌ Go toolchain not found. Install Go."
    exit 1
fi

echo "🔨 Building Go UI ..."
go build -o "$PROJECT_DIR/quantstation" ./cmd/quantstation

echo "🚀 Starting Quant Station Go UI ..."
echo "🌐 Open: http://127.0.0.1:7878"
"$PROJECT_DIR/quantstation"
