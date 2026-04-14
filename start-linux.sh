#!/usr/bin/env bash
set -e

# --- 1. CONFIGURATION ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_DIR/.venv"
LLAMA_DIR="$PROJECT_DIR/llama.cpp"
PATCH_FILE="$PROJECT_DIR/lcpp.patch"

echo "📂 Project Root: $PROJECT_DIR"

# --- 1.5 MULTI-DISTRO DEPENDENCIES ---
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "🔍 Detected System: $NAME"

    # 1. Install System Build Tools
    case "$ID" in
        arch|manjaro)
            echo "📦 Installing for Arch-based system..."
            sudo pacman -S --needed --noconfirm base-devel cmake cuda curl
            ;;
        ubuntu|debian|mint)
            echo "📦 Installing for Debian-based system..."
            sudo apt update
            sudo apt install -y build-essential cmake nvidia-cuda-toolkit curl
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

# --- 2. CLONE, PATCH & BUILD LLAMA.CPP (SKIP IF DONE) ---
cd "$PROJECT_DIR"

# Define the path to the final binary we need
QUANT_EXE="$LLAMA_DIR/build/bin/llama-quantize"

if [ -f "$QUANT_EXE" ]; then
    echo "✅ Patched llama-quantize found. Skipping patch/build steps."
else
    echo "🔨 Binary not found or incomplete. Starting build process..."

    # 1. Clone if missing
    if [ ! -d "$LLAMA_DIR" ]; then
        git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
    fi

    cd "$LLAMA_DIR"

    # 2. Hard Reset and Patch (Only if we are actually building)
    echo "🏷️ Preparing source for Wan 2.2..."
    git reset --hard
    git checkout tags/b3962 -f
    
    if [ -f "$PROJECT_DIR/lcpp.patch" ]; then
        echo "🧹 Normalizing and Applying patch..."
        # Fix line endings on the fly
        python3 -c "f=open('$PROJECT_DIR/lcpp.patch','rb');c=f.read().replace(b'\r\n',b'\n');open('$PROJECT_DIR/lcpp.patch','wb').write(c)"
        git apply "$PROJECT_DIR/lcpp.patch"
    fi

    # 3. Build with modern CUDA + C++17 flags
    mkdir -p build && cd build
    cmake .. -DGGML_CUDA=ON \
             -DCMAKE_CUDA_ARCHITECTURES=native \
             -DCMAKE_CXX_STANDARD=17 \
             -DCMAKE_CUDA_STANDARD=17
             
    cmake --build . --config Release -j$(nproc)
    
    cd "$PROJECT_DIR"
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

echo "🍳 Installing Comfy Kitchen [CUBLAS]..."
uv pip install --refresh "comfy-kitchen[cublas]"

# --- 4. LAUNCH ---
echo "🚀 Starting Quant Station ..."
export VIRTUAL_ENV="$VENV_PATH"
export PATH="$VENV_PATH/bin:$PATH"

# Ensure Python knows we aren't using RAMDisk anymore
python app.py