# worker_entrypoint_basic_utils.sh

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
export MODEL_KEY="basic_utils"
export HEARTBEAT_INTERVAL="10"
export MAX_CONCURRENT_JOBS="1"

export WORKSPACE_DIR="/quickpod/quickpod_worker"

export HF_HOME="/models_storage/all_models"
export HF_HUB_CACHE="/models_storage/all_models"

export LAMA_MODEL="/models_storage/all_models"
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
wget -q -O basic_utils_inference_app.tar.gz https://<r2_url>/<your_code_archive.tar.gz>
tar -xzf basic_utils_inference_app.tar.gz
rm basic_utils_inference_app.tar.gz
echo "Code unpacked."

# Install dependencies
echo "Installing Python dependencies..."
cd /quickpod/quickpod_worker/worker_basic_utils
pip install -U -r requirements.txt
echo "Dependencies installed."

# --- Section 6: Pre-download Hugging Face models (RMBG + optional LaMa) ---
log() { echo "[$(date -Is)] $*"; }

# Ensure cache dir exists and env vars for HF caches are set
CACHE_DIR="${HF_HOME:-/models_storage/all_models}"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"

# Model repo IDs (can override via env when creating pod)
# RMBG default (used by your rmbg_remover.py pipeline)
RMBG_HF_REPO="${RMBG_HF_REPO:-briaai/RMBG-1.4}"
# LAMA repo (optional) - set LAMA_HF_REPO env if you want a specific repo
LAMA_HF_REPO="${LAMA_HF_REPO:-}"   # e.g. "saicoder/lama" or your custom repo

# Where to put snapshots inside cache
RMBG_LOCAL_DIR="$CACHE_DIR/hf_models/$(echo "$RMBG_HF_REPO" | tr '/' '_')"
LAMA_LOCAL_DIR="$CACHE_DIR/hf_models/$(echo "$LAMA_HF_REPO" | tr '/' '_')"

log "Preparing to download models into $CACHE_DIR"
if [ -z "${HF_TOKEN:-}" ]; then
  log "WARNING: HF_TOKEN not set — downloads may fail for gated models."
fi

# Install huggingface_hub if missing (quiet)
python - <<'PY' || true
import sys, subprocess, pkgutil
if pkgutil.find_loader("huggingface_hub") is None:
    print("Installing huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-input", "huggingface_hub"])
else:
    print("huggingface_hub already installed")
PY

export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"

# Function to download with huggingface_hub.snapshot_download (idempotent)
python - <<'PY'
import os, sys, traceback
from huggingface_hub import snapshot_download
def dl(repo_id, cache_dir):
    if not repo_id:
        print("No repo_id provided, skipping.")
        return None
    # map repo to unique local path inside cache
    target = os.path.join(cache_dir, "hf_models", repo_id.replace("/", "_"))
    if os.path.exists(target) and os.listdir(target):
        print(f"Found existing model at {target}, skipping download.")
        return target
    try:
        print(f"Downloading snapshot for {repo_id} into {target} ...")
        path = snapshot_download(repo_id, cache_dir=cache_dir, local_dir=target, resume_download=True)
        print(f"Downloaded {repo_id} -> {path}")
        return path
    except Exception as e:
        print("Snapshot download failed:", e)
        traceback.print_exc()
        return None

cache_dir = os.environ.get("HF_HOME", "/models_storage/all_models")
rmbg = os.environ.get("RMBG_HF_REPO", "briaai/RMBG-1.4")
lama = os.environ.get("LAMA_HF_REPO", "")

r = dl(rmbg, cache_dir)
if r:
    print("RMBG snapshot path:", r)
else:
    print("RMBG download failed (will attempt git fallback if HF_TOKEN provided).")

if lama:
    rl = dl(lama, cache_dir)
    if rl:
        print("LaMa snapshot path:", rl)
    else:
        print("LaMa download failed.")
else:
    print("No LAMA_HF_REPO specified; skipping LaMa download.")
PY

# If snapshot_download failed for any repo, fallback to git clone using HF_TOKEN if provided
if [ ! -d "$RMBG_LOCAL_DIR" ] || [ -z "$(ls -A "$RMBG_LOCAL_DIR" 2>/dev/null)" ]; then
  if [ -n "${HF_TOKEN:-}" ]; then
    log "RMBG snapshot missed — falling back to git clone for $RMBG_HF_REPO"
    mkdir -p "$RMBG_LOCAL_DIR"
    git -C "$RMBG_LOCAL_DIR" init
    git -C "$RMBG_LOCAL_DIR" remote add origin "https://user:${HF_TOKEN}@huggingface.co/${RMBG_HF_REPO}" || true
    git -C "$RMBG_LOCAL_DIR" pull origin main || true
  else
    log "No HF_TOKEN: cannot fallback clone for RMBG. If model is gated, set HF_TOKEN."
  fi
else
  log "RMBG already available at $RMBG_LOCAL_DIR"
fi

if [ -n "$LAMA_HF_REPO" ]; then
  if [ ! -d "$LAMA_LOCAL_DIR" ] || [ -z "$(ls -A "$LAMA_LOCAL_DIR" 2>/dev/null)" ]; then
    if [ -n "${HF_TOKEN:-}" ]; then
      log "LaMa snapshot missed — falling back to git clone for $LAMA_HF_REPO"
      mkdir -p "$LAMA_LOCAL_DIR"
      git -C "$LAMA_LOCAL_DIR" init
      git -C "$LAMA_LOCAL_DIR" remote add origin "https://user:${HF_TOKEN}@huggingface.co/${LAMA_HF_REPO}" || true
      git -C "$LAMA_LOCAL_DIR" pull origin main || true
    else
      log "No HF_TOKEN: cannot fallback clone for LaMa. If model is gated, set HF_TOKEN or provide LAMA_HF_REPO public repo."
    fi
  else
    log "LaMa already available at $LAMA_LOCAL_DIR"
  fi
else
  log "No LAMA_HF_REPO set; skipping LaMa fallback."
fi

# Export local paths for Python code to use
export RMBG_LOCAL_PATH="$RMBG_LOCAL_DIR"
export LAMA_LOCAL_PATH="$LAMA_LOCAL_DIR"
log "Model download step complete. RMBG_LOCAL_PATH=${RMBG_LOCAL_PATH}, LAMA_LOCAL_PATH=${LAMA_LOCAL_PATH}"


# --- Section 7: Start the Application (run as current/default user) ---
echo "Starting the application (as current user: $(whoami))..."
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
LOGFILE="/quickpod/quickpod_worker/basic_utils_app.log"

# Make sure the log file exists and is writable by the current user
mkdir -p "$(dirname "$LOGFILE")"
touch "$LOGFILE"
chmod 644 "$LOGFILE" || true

# Activate virtualenv and run uvicorn in background
# Note: 'nohup' + '&' means the process will keep running after this script exits.
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

echo "Application started (logs in $LOGFILE)."
