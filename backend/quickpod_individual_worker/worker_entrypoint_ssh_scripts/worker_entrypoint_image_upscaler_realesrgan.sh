#!/bin/bash
set -euo pipefail

# --- Section 1: Virtual Environment Setup ---
echo "Setting up virtual environment..."
mkdir -p /quickpod/quickpod_worker
cd /quickpod/quickpod_worker
if [ ! -d "venv_worker" ]; then
    python3 -m venv venv_worker
    echo "✓ Virtual environment created."
fi
source venv_worker/bin/activate
echo "✓ Virtual environment activated."

# --- Section 2: Environment Variables ---
echo "Setting environment variables..."
export MODEL_KEY="image_upscaler_realesrgan"

export HEARTBEAT_INTERVAL="10"
export MAX_CONCURRENT_JOBS="1"

export WORKSPACE_DIR="/quickpod/quickpod_worker"

export HF_HOME="/models_storage/all_models"
export HF_HUB_CACHE="/models_storage/all_models"

export XFORMERS_OFFLOAD_CPU="1"
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:512,expandable_segments:True"

echo "✓ Environment variables set."

# --- Section 3: Install PyTorch (CUDA 12.9) ---
echo "Installing PyTorch for CUDA 12.9..."
pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
echo "✓ PyTorch installed."

# --- Section 4: Get and Unpack Code ---
echo "Downloading and unpacking worker code..."
cd /quickpod/quickpod_worker
if [ ! -d "worker_image_upscaler_realesrgan" ]; then
    apt-get update -qq && apt-get install -y unzip
    wget -q -O worker_image_upscaler_realesrgan.zip https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/worker_image_upscaler_realesrgan.zip
    unzip -o worker_image_upscaler_realesrgan.zip
    rm worker_image_upscaler_realesrgan.zip
    echo "✓ Worker code unpacked."
else
    echo "✓ Worker code already present."
fi

# Install Python dependencies
echo "Installing Python dependencies..."
cd /quickpod/quickpod_worker/worker_image_upscaler_realesrgan
pip install -U -r requirements.txt
echo "✓ Dependencies installed."

# --- Section 5: Download Hugging Face Models (RealESRGAN) ---
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

REPO_ID = "ai-forever/Real-ESRGAN"
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
        print(f"⬇️ Downloading {fname} from {REPO_ID} ...")
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

# --- Section 6: Start the Application ---
echo "Starting the application (as current user: $(whoami))..."
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
LOGFILE="/quickpod/quickpod_worker/image_upscaler_realesrgan_app.log"

mkdir -p "$(dirname "$LOGFILE")"
touch "$LOGFILE"
chmod 644 "$LOGFILE" || true

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

echo "✅ Application started (logs in $LOGFILE)"
