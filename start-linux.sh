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
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "🔍 Detected System: $NAME"

    # 1. Install System Build Tools
    case "$ID" in
        arch|manjaro)
            echo "📦 Installing for Arch-based system..."
            sudo pacman -S --needed --noconfirm base-devel cmake cuda curl unzip go
            ;;
        ubuntu|debian|mint)
            echo "📦 Installing for Debian-based system..."
            sudo apt update
            sudo apt install -y build-essential cmake nvidia-cuda-toolkit curl unzip golang-go
            ;;
        *)
            echo "⚠️ Unrecognized distribution ($ID). Ensure build-essential, cmake, and cuda are manualy installed."
            ;;
    esac

    # 2. Install 'uv' (Python Package Manager)
    if ! command -v uv &> /dev/null; then
        echo "⚙️ uv not found. Installing via official script..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Source the cargo env to make uv immediately available in this session
        source $HOME/.cargo/env
    else
        echo "✅ uv is already installed."
    fi
else
    echo "❌ Could not detect OS via /etc/os-release. Skipping system package install."
fi

# --- 2. GGUF ENGINE SETUP ---

# --- 2.5 DOWNLOAD GGUFY BINARY ---
GGUFY_BIN="$PROJECT_DIR/bin/ggufy"

# Safety check: remove invalid binary if it's far too small or not a valid ELF
if [ -f "$GGUFY_BIN" ]; then
    FILE_SIZE=$(stat -c%s "$GGUFY_BIN" 2>/dev/null || echo 0)
    if [ "$FILE_SIZE" -lt 500000 ] || ! head -c 4 "$GGUFY_BIN" | grep -q $'\x7fELF' 2>/dev/null; then
        echo "⚠️  Detected corrupted, tiny, or non-ELF GGUFY binary ($FILE_SIZE bytes). Removing for re-download..."
        rm -f "$GGUFY_BIN"
    fi
fi

if [ ! -f "$GGUFY_BIN" ]; then
    echo "📥 GGUFY binary not found. Downloading latest release..."
    mkdir -p "$PROJECT_DIR/bin"
    
    # Detect architecture and select correct asset
    ARCH=$(uname -m)
    GGUFY_ASSET="ggufy-linux-x86_64"
    [ "$ARCH" = "aarch64" ] && GGUFY_ASSET="ggufy-linux-arm64"

    echo "📥 Downloading GGUFY archive..."
    if curl -L --fail --retry 3 --retry-delay 2 -o "$PROJECT_DIR/bin/ggufy.zip" "https://github.com/qskousen/ggufy/releases/latest/download/${GGUFY_ASSET}.zip"; then
        echo "📦 Extracting binary (handles nested folders)..."
        unzip -q -o "$PROJECT_DIR/bin/ggufy.zip" -d "$PROJECT_DIR/bin/extracted"
        find "$PROJECT_DIR/bin/extracted" -type f \( -name "ggufy" -o -name "$GGUFY_ASSET" \) -exec mv {} "$GGUFY_BIN" \;
        
        if [ -f "$GGUFY_BIN" ]; then
            chmod +x "$GGUFY_BIN"
            rm -rf "$PROJECT_DIR/bin/ggufy.zip" "$PROJECT_DIR/bin/extracted"
            echo "✅ GGUFY installed successfully."
        else
            echo "❌ Extraction failed: binary not found in zip."
            exit 1
        fi
    else
        echo "❌ Failed to download GGUFY binary. Check your connection or GitHub access."
        exit 1
    fi
fi

# --- 3. LOCAL VENV SETUP ---
if [ ! -d "$VENV_PATH" ]; then
    echo "⚙️ Creating local virtual environment..."
    uv venv "$VENV_PATH"
fi

echo "📦 Syncing Python dependencies..."
uv pip install --refresh -r requirements.txt

echo "💎 Installing FP Quantization Tools..."
uv pip install --refresh git+https://github.com/silveroxides/convert_to_quant.git@main#egg=convert_to_quant --no-deps --force-reinstall

echo "🍳 Installing Comfy Kitchen [CUBLAS + INT4 ConvRot]..."
# Pin a revision that exports TensorCoreConvRotW4A4Layout. The PyPI 0.1.0
# package lacks this layout, while its package metadata can still resolve it
# for unconstrained installs.
uv pip install --python "$VENV_PATH/bin/python" --refresh --upgrade \
    "comfy-kitchen[cublas] @ git+https://github.com/Comfy-Org/comfy-kitchen.git@911d47e6c355f31a1f66fe74ea64a1760fad581a"

"$VENV_PATH/bin/python" - <<'PY'
from comfy_kitchen.tensor import TensorCoreConvRotW4A4Layout
print("✅ Comfy Kitchen INT4 ConvRot layout available:", TensorCoreConvRotW4A4Layout.__name__)
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
