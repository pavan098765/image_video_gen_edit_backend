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
export MODEL_KEY="qwen_image_edit"

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
echo "Downloading and unpacking code..."
cd /quickpod/quickpod_worker
if [ ! -d "worker_qwen_image_edit" ]; then
    apt-get update -qq && apt-get install -y unzip git
    wget -q -O worker_qwen_image_edit.zip https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/worker_qwen_image_edit.zip
    unzip -o worker_qwen_image_edit.zip
    rm worker_qwen_image_edit.zip
    echo "✓ Worker code unpacked."
else
    echo "✓ Worker code already present."
fi

# Install dependencies
echo "Installing Python dependencies..."
cd /quickpod/quickpod_worker/worker_qwen_image_edit
pip install -U -r requirements.txt
echo "✓ Dependencies installed."

# --- Section 5: Pre-download Hugging Face Models (Qwen Image Edit split) ---
log() { echo "[$(date -Is)] $*"; }

CACHE_DIR="${HF_HOME:-/models_storage/all_models}"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"

QWEN_REPO="Qwen/Qwen-Image-Edit"
NUNCHAKU_REPO="nunchaku-tech/nunchaku-qwen-image-edit"

QWEN_LOCAL_DIR="$CACHE_DIR/hf_models/$(echo "$QWEN_REPO" | tr '/' '_')"
NUNCHAKU_LOCAL_DIR="$CACHE_DIR/hf_models/$(echo "$NUNCHAKU_REPO" | tr '/' '_')"

python - <<'PY'
import os, traceback
from huggingface_hub import snapshot_download, hf_hub_download
from nunchaku.utils import get_precision

cache_dir = os.environ.get("HF_HOME", "/models_storage/all_models")

# 1. Download Qwen repo without transformer
qwen_repo = "Qwen/Qwen-Image-Edit"
qwen_target = os.path.join(cache_dir, "hf_models", qwen_repo.replace("/", "_"))
try:
    print(f"⬇️ Downloading {qwen_repo} (skipping transformer weights)...")
    snapshot_download(
        qwen_repo,
        cache_dir=cache_dir,
        local_dir=qwen_target,
        resume_download=True,
        allow_patterns=["*", "!(transformer/*)"],
    )
    print(f"✓ Qwen repo (without transformer) downloaded to {qwen_target}")
except Exception as e:
    print("❌ Qwen repo download failed:", e)
    traceback.print_exc()

# 2. Download only transformer file from Nunchaku
nunchaku_repo = "nunchaku-tech/nunchaku-qwen-image-edit"
precision = get_precision()
rank = 128
steps = 8
filename = f"svdq-{precision}_r{rank}-qwen-image-edit-lightningv1.0-{steps}steps.safetensors"

nunchaku_target_dir = os.path.join(cache_dir, "hf_models", nunchaku_repo.replace("/", "_"))
os.makedirs(nunchaku_target_dir, exist_ok=True)
nunchaku_target = os.path.join(nunchaku_target_dir, filename)

if not os.path.exists(nunchaku_target):
    try:
        print(f"⬇️ Downloading {nunchaku_repo}/{filename} ...")
        path = hf_hub_download(
            repo_id=nunchaku_repo,
            filename=filename,
            cache_dir=cache_dir,
            resume_download=True
        )
        print(f"✓ Downloaded transformer to {path}")
    except Exception as e:
        print("❌ Transformer download failed:", e)
        traceback.print_exc()
else:
    print(f"✓ Transformer already exists at {nunchaku_target}")
PY

export QWEN_LOCAL_PATH="$QWEN_LOCAL_DIR"
export NUNCHAKU_QWEN_LOCAL_PATH="$NUNCHAKU_LOCAL_DIR"
log "Qwen repo ready at $QWEN_LOCAL_PATH"
log "Nunchaku transformer ready at $NUNCHAKU_QWEN_LOCAL_PATH"

# --- Section 6: Start the Application ---
echo "Starting the application (as current user: $(whoami))..."
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
LOGFILE="/quickpod/quickpod_worker/qwen_image_edit_app.log"

mkdir -p "$(dirname "$LOGFILE")"
touch "$LOGFILE"
chmod 644 "$LOGFILE" || true

cd /quickpod/quickpod_worker/worker_qwen_image_edit
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
