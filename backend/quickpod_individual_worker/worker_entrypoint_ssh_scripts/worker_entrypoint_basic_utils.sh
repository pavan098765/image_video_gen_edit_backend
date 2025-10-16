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
export MODEL_KEY="basic_utils"
export HEARTBEAT_INTERVAL="10"
export MAX_CONCURRENT_JOBS="1"

export WORKSPACE_DIR="/quickpod/quickpod_worker"

export HF_HOME="/models_storage/all_models"
export HF_HUB_CACHE="/models_storage/all_models"

export LAMA_MODEL="/models_storage/all_models"
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
if [ ! -d "worker_basic_utils" ]; then
    apt-get update -qq && apt-get install -y unzip git
    wget -q -O worker_basic_utils.zip https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/worker_basic_utils.zip
    unzip -o worker_basic_utils.zip
    rm worker_basic_utils.zip
    echo "✓ Worker code unpacked."
else
    echo "✓ Worker code already present."
fi

# Install dependencies
echo "Installing Python dependencies..."
cd /quickpod/quickpod_worker/worker_basic_utils
pip install -U -r requirements.txt
echo "✓ Dependencies installed."

# --- Section 5: Pre-download Hugging Face Models (RMBG + optional LaMa) ---
log() { echo "[$(date -Is)] $*"; }

CACHE_DIR="${HF_HOME:-/models_storage/all_models}"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"

RMBG_HF_REPO="${RMBG_HF_REPO:-briaai/RMBG-1.4}"
LAMA_HF_REPO="${LAMA_HF_REPO:-}"

RMBG_LOCAL_DIR="$CACHE_DIR/hf_models/$(echo "$RMBG_HF_REPO" | tr '/' '_')"
LAMA_LOCAL_DIR="$CACHE_DIR/hf_models/$(echo "$LAMA_HF_REPO" | tr '/' '_')"

log "Preparing to download models into $CACHE_DIR"
python - <<'PY' || true
import os, traceback
from huggingface_hub import snapshot_download
def dl(repo_id, cache_dir):
    if not repo_id: return None
    target = os.path.join(cache_dir, "hf_models", repo_id.replace("/", "_"))
    if os.path.exists(target) and os.listdir(target):
        print(f"✓ Found existing {repo_id}, skipping")
        return target
    try:
        path = snapshot_download(repo_id, cache_dir=cache_dir, local_dir=target, resume_download=True)
        print(f"✓ Downloaded {repo_id} -> {path}")
        return path
    except Exception as e:
        print(f"❌ Failed {repo_id}:", e)
        traceback.print_exc()
        return None

cache = os.environ.get("HF_HOME", "/models_storage/all_models")
rmbg = os.environ.get("RMBG_HF_REPO", "briaai/RMBG-1.4")
lama = os.environ.get("LAMA_HF_REPO", "")
dl(rmbg, cache)
if lama: dl(lama, cache)
PY

export RMBG_LOCAL_PATH="$RMBG_LOCAL_DIR"
export LAMA_LOCAL_PATH="$LAMA_LOCAL_DIR"
log "Model download step complete. RMBG_LOCAL_PATH=$RMBG_LOCAL_PATH, LAMA_LOCAL_PATH=$LAMA_LOCAL_PATH"

# --- Section 6: Start the Application ---
echo "Starting the application (as current user: $(whoami))..."
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
LOGFILE="/quickpod/quickpod_worker/basic_utils_app.log"

mkdir -p "$(dirname "$LOGFILE")"
touch "$LOGFILE"
chmod 644 "$LOGFILE" || true

cd /quickpod/quickpod_worker/worker_basic_utils
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
