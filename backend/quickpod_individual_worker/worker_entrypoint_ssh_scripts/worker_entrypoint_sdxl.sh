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
export MODEL_KEY="sdxl"
export HEARTBEAT_INTERVAL="10"
export MAX_CONCURRENT_JOBS="1"

export WORKSPACE_DIR="/quickpod/quickpod_worker"

export HF_HOME="/models_storage/all_models"
export HF_HUB_CACHE="/models_storage/all_models"

export XFORMERS_OFFLOAD_CPU="1"
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:512,expandable_segments:True"

echo "✓ Environment variables set."

# --- Section 3: PyTorch Installation (CUDA 12.9 preinstalled in base image) ---
echo "Installing PyTorch for CUDA 12.9..."
pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
echo "✓ PyTorch installed."

# --- Section 4: Get and Unpack Code ---
echo "Downloading and unpacking code..."
cd /quickpod/quickpod_worker
if [ ! -d "worker_stable_diffusion_xl" ]; then
    apt-get update -qq && apt-get install -y unzip git
    wget -q -O worker_stable_diffusion_xl.zip https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/worker_stable_diffusion_xl.zip
    unzip -o worker_stable_diffusion_xl.zip
    rm worker_stable_diffusion_xl.zip
    echo "✓ Worker code unpacked."
else
    echo "✓ Worker code already present."
fi

# Install dependencies
echo "Installing Python dependencies..."
cd /quickpod/quickpod_worker/worker_stable_diffusion_xl
pip install -U -r requirements.txt
echo "✓ Dependencies installed."

# --- Section 5: Download Hugging Face Models (SDXL Realism Illustrious) ---
log() { echo "[$(date -Is)] $*"; }

CACHE_DIR="${HF_HOME:-/models_storage/all_models}"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"

SDXL_REPO="John6666/realism-illustrious-by-stable-yogi-v50-fp16-sdxl"
SDXL_LOCAL="$CACHE_DIR/hf_models/$(echo "$SDXL_REPO" | tr '/' '_')"

python - <<'PY'
import os, traceback
from huggingface_hub import snapshot_download

repo_id = "John6666/realism-illustrious-by-stable-yogi-v50-fp16-sdxl"
cache_dir = os.environ.get("HF_HOME", "/models_storage/all_models")
target = os.path.join(cache_dir, "hf_models", repo_id.replace("/", "_"))

if os.path.exists(target) and os.listdir(target):
    print(f"✓ Found {repo_id} at {target}, skipping.")
else:
    try:
        print(f"⬇️ Downloading {repo_id} into {target} ...")
        snapshot_download(repo_id, cache_dir=cache_dir, local_dir=target, resume_download=True)
        print(f"✓ Downloaded {repo_id} -> {target}")
    except Exception as e:
        print(f"❌ Failed to download {repo_id}: {e}")
        traceback.print_exc()
PY

export SDXL_LOCAL_PATH="$SDXL_LOCAL"
export SDXL_EMBEDDINGS_PATH="$SDXL_LOCAL/embeddings"

log "SDXL model ready at $SDXL_LOCAL_PATH"
log "Embeddings directory: $SDXL_EMBEDDINGS_PATH"

# --- Section 6: Start the Application ---
echo "Starting the application (as current user: $(whoami))..."
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
LOGFILE="/quickpod/quickpod_worker/sdxl_app.log"

mkdir -p "$(dirname "$LOGFILE")"
touch "$LOGFILE"
chmod 644 "$LOGFILE" || true

cd /quickpod/quickpod_worker/worker_stable_diffusion_xl
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
