#!/bin/bash
set -euo pipefail

# --- Section 1: Python 3.11 Installation ---

echo "Checking for Python 3.11..."
if ! command -v python3.11 &> /dev/null; then
    echo "Python 3.11 not found. Installing..."
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends python3.11 python3.11-dev python3.11-venv
    sudo apt-get clean
    echo "Python 3.11 installed."
else
    echo "Python 3.11 is already installed."
fi

# --- Section 2: Virtual Environment Setup ---

echo "Setting up virtual environment..."
mkdir -p /quickpod/quickpod_worker
cd /quickpod/quickpod_worker
python3.11 -m venv venv_worker
source venv_worker/bin/activate
echo "Virtual environment created and activated."

# --- Section 3: Environment Variables ---

echo "Setting environment variables..."
export MODEL_KEY="image_upscaler_realesrgan"

export HEARTBEAT_INTERVAL="10"
export MAX_CONCURRENT_JOBS="1"

export WORKSPACE_DIR="/quickpod/quickpod_worker"

export HF_HOME="/models_storage/all_models"
export HF_HUB_CACHE="/models_storage/all_models"

export XFORMERS_OFFLOAD_CPU="1"
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:512,expandable_segments:True"

echo "Environment variables set."

# --- Section 4: Dynamic CUDA and PyTorch Installation ---

echo "Checking for existing CUDA installation..."
CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version 2>/dev/null | grep 'release' | awk '{print $NF}' | sed 's/V//' | cut -d'.' -f1,2 || true)

if [ -z "$CUDA_VERSION" ] || [ "$(echo "$CUDA_VERSION < 12.8" | bc -l)" -eq 1 ]; then
    echo "CUDA not found or version < 12.8. Installing CUDA 12.9..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
    sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-ubuntu2404-12-9-local_12.9.0-575.51.03-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2404-12-9-local_12.9.0-575.51.03-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2404-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update -qq
    sudo apt-get -y install cuda-toolkit-12-9
    CUDA_VERSION="12.9"
fi

# Install PyTorch matching detected CUDA version
pip install -U pip
case "$CUDA_VERSION" in
    12.8)
        echo "Detected CUDA $CUDA_VERSION. Installing PyTorch 2.7.1..."
        pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
        ;;
    12.9)
        echo "Detected CUDA $CUDA_VERSION. Installing PyTorch for CUDA 12.9..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
        ;;
    *)
        echo "Unsupported CUDA version: $CUDA_VERSION"
        exit 1
        ;;
esac

echo "CUDA and PyTorch setup complete."


# --- Section 5: Get and Unpack Code ---

echo "Downloading and unpacking code..."
cd /quickpod/quickpod_worker
wget -q -O image_upscaler_realesrgan_inference_app.tar.gz https://<r2_url>/<your_code_archive.tar.gz>
tar -xzf image_upscaler_realesrgan_inference_app.tar.gz
rm image_upscaler_realesrgan_inference_app.tar.gz
echo "Code unpacked."

# Install dependencies
echo "Installing Python dependencies..."
cd /quickpod/quickpod_worker/worker_image_upscaler_realesrgan
pip install -U -r requirements.txt
echo "Dependencies installed."


# --- Section 6: Download Hugging Face Models (RealESRGAN) ---
log() { echo "[$(date -Is)] $*"; }

CACHE_DIR="${HF_HOME:-/models_storage/all_models}"
MODEL_DIR="$CACHE_DIR/realesrgan"
mkdir -p "$MODEL_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"

python - <<'PY'
import os, traceback
from huggingface_hub import hf_hub_download

# Model repo (weights live here)
REPO_ID = "ai-forever/Real-ESRGAN"  # fallback if not hosted, adjust if you have another

# Mapping of file names
files = [
    "realesr-general-x4v3.pth",
    "realesr-general-wdn-x4v3.pth",
    "RealESRGAN_x4plus.pth",
]

cache_dir = os.environ.get("HF_HOME", "/models_storage/all_models")
model_dir = os.path.join(cache_dir, "realesrgan")
os.makedirs(model_dir, exist_ok=True)

for fname in files:
    target = os.path.join(model_dir, fname)
    if os.path.exists(target):
        print(f"✓ Found {fname}, skipping.")
        continue
    try:
        print(f"⬇️ Downloading {fname} ...")
        path = hf_hub_download(repo_id=REPO_ID, filename=fname, cache_dir=cache_dir, resume_download=True)
        os.rename(path, target)
        print(f"✓ Downloaded {fname} -> {target}")
    except Exception as e:
        print(f"❌ Failed to download {fname}: {e}")
        traceback.print_exc()
PY

export REALESRGAN_X4_GENERAL_PATH="$MODEL_DIR/realesr-general-x4v3.pth"
export REALESRGAN_X4_GENERAL_WDN_PATH="$MODEL_DIR/realesr-general-wdn-x4v3.pth"
export REALESRGAN_X4_PLUS_PATH="$MODEL_DIR/RealESRGAN_x4plus.pth"
log "RealESRGAN models ready in $MODEL_DIR"


# --- Section 7: Start the Application (run as current/default user) ---
echo "Starting the application (as current user: $(whoami))..."
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
LOGFILE="/quickpod/quickpod_worker/image_upscaler_realesrgan_app.log"

# Make sure the log file exists and is writable by the current user
mkdir -p "$(dirname "$LOGFILE")"
touch "$LOGFILE"
chmod 644 "$LOGFILE" || true

# Activate virtualenv and run uvicorn in background
# Note: 'nohup' + '&' means the process will keep running after this script exits.
cd /quickpod/quickpod_worker/worker_image_upscaler_realesrgan
source /quickpod/quickpod_worker/venv_worker/bin/activate
nohup uvicorn src.app:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --workers 1 \
  --proxy-headers \
  --loop uvloop \
  --http httptools \
  > "$LOGFILE" 2>&1 &

echo "Application started (logs in $LOGFILE)."
