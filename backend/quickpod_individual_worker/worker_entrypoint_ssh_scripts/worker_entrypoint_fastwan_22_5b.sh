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
export MODEL_KEY="fastwan_22_5b"
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
if [ ! -d "worker_fastwan_22_5b" ]; then
    apt-get update -qq && apt-get install -y unzip git
    wget -q -O worker_fastwan_22_5b.zip https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/worker_fastwan_22_5b.zip
    unzip -o worker_fastwan_22_5b.zip
    rm worker_fastwan_22_5b.zip
    echo "✓ Worker code unpacked."
else
    echo "✓ Worker code already present."
fi

# Install dependencies
echo "Installing Python dependencies..."
cd /quickpod/quickpod_worker/worker_fastwan_22_5b
pip install -U -r requirements.txt
echo "✓ Dependencies installed."

# --- Section 5: Pre-download Hugging Face model (FastWan 2.1 1.3B) ---
log() { echo "[$(date -Is)] $*"; }

CACHE_DIR="${HF_HOME:-/models_storage/all_models}"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"

FASTWAN_1B_REPO="FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
FASTWAN_1B_LOCAL_DIR="$CACHE_DIR/hf_models/$(echo "$FASTWAN_1B_REPO" | tr '/' '_')"

log "Preparing to download FastWan 2.1 1.3B into $FASTWAN_1B_LOCAL_DIR"

python - <<'PY' || true
import os, traceback
from huggingface_hub import snapshot_download

repo_id = os.environ.get("FASTWAN_1B_REPO", "FastVideo/FastWan2.1-T2V-1.3B-Diffusers")
cache_dir = os.environ.get("HF_HOME", "/models_storage/all_models")
target = os.path.join(cache_dir, "hf_models", repo_id.replace("/", "_"))

if os.path.exists(target) and os.listdir(target):
    print(f"FastWan 1B already exists at {target}, skipping download.")
else:
    try:
        print(f"⬇️ Downloading {repo_id} into {target} ...")
        snapshot_download(repo_id, cache_dir=cache_dir, local_dir=target, resume_download=True)
        print(f"✓ Downloaded {repo_id} -> {target}")
    except Exception as e:
        print("❌ Snapshot download failed:", e)
        traceback.print_exc()
PY

if [ ! -d "$FASTWAN_1B_LOCAL_DIR" ] || [ -z "$(ls -A "$FASTWAN_1B_LOCAL_DIR" 2>/dev/null)" ]; then
  if [ -n "${HF_TOKEN:-}" ]; then
    log "FastWan 1B snapshot missed — falling back to git clone"
    mkdir -p "$FASTWAN_1B_LOCAL_DIR"
    git -C "$FASTWAN_1B_LOCAL_DIR" init
    git -C "$FASTWAN_1B_LOCAL_DIR" remote add origin "https://user:${HF_TOKEN}@huggingface.co/${FASTWAN_1B_REPO}" || true
    git -C "$FASTWAN_1B_LOCAL_DIR" pull origin main || true
  else
    log "No HF_TOKEN: cannot fallback clone for FastWan 1B."
  fi
else
  log "FastWan 1B already available at $FASTWAN_1B_LOCAL_DIR"
fi

export FASTWAN_1B_LOCAL_PATH="$FASTWAN_1B_LOCAL_DIR"
log "FastWan 1B model ready at $FASTWAN_1B_LOCAL_PATH"

# --- Section 6: Start the Application ---
echo "Starting the application (as current user: $(whoami))..."
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
LOGFILE="/quickpod/quickpod_worker/fastwan_22_5b_app.log"

mkdir -p "$(dirname "$LOGFILE")"
touch "$LOGFILE"
chmod 644 "$LOGFILE" || true

cd /quickpod/quickpod_worker/worker_fastwan_22_5b
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
