#!/bin/bash
set -euo pipefail

# --- Section 1: Virtual Environment Setup ---
echo "Checking virtual environment..."
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
export MODEL_KEY="multi-model-worker"
export HEARTBEAT_INTERVAL="10"
export MAX_CONCURRENT_JOBS="1"
export WORKSPACE_DIR="/quickpod/quickpod_worker"
export HF_HOME="/models_storage/all_models"
export HF_HUB_CACHE="/models_storage/all_models"
export LAMA_MODEL="/models_storage/all_models"
export XFORMERS_OFFLOAD_CPU="1"
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:512,expandable_segments:True"
echo "✓ Environment variables set."

# --- Section 3: PyTorch Installation (CUDA 12.9) ---
if ! python -c "import torch" &>/dev/null; then
    echo "Installing PyTorch for CUDA 12.9..."
    pip install -U pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
    echo "✓ PyTorch installed."
else
    echo "✓ PyTorch already installed."
fi

# --- Section 4: Get and Unpack Code ---
echo "Checking code package..."
cd /quickpod/quickpod_worker
if [ ! -d "worker_warm_pod" ]; then
    apt-get update -qq && apt-get install -y unzip
    wget -q -O worker_warm_pod.zip https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/worker_warm_pod.zip
    unzip -o worker_warm_pod.zip
    rm worker_warm_pod.zip
    echo "✓ Code unpacked."
else
    echo "✓ Code already present, skipping download."
fi

# --- Section 5: Install Dependencies (cached) ---
cd /quickpod/quickpod_worker/worker_warm_pod
if [ ! -f ".deps_installed" ]; then
    echo "Installing Python dependencies..."
    pip install -U -r requirements.txt
    touch .deps_installed
    echo "✓ Dependencies installed."
else
    echo "✓ Dependencies already installed."
fi

# --- Section 6: Start the Application ---
echo "Starting the application (as current user: $(whoami))..."
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
LOGFILE="/quickpod/quickpod_worker/app.log"

mkdir -p "$(dirname "$LOGFILE")"
touch "$LOGFILE"
chmod 644 "$LOGFILE" || true

cd /quickpod/quickpod_worker/worker_warm_pod
source /quickpod/quickpod_worker/venv_worker/bin/activate

# Kill previous uvicorn if running
pkill -f "uvicorn src.app:app" || true

# Start fresh
nohup uvicorn src.app:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --workers 1 \
  --proxy-headers \
  --loop uvloop \
  --http httptools \
  > "$LOGFILE" 2>&1 &

echo "✅ Application started (logs in $LOGFILE)."
