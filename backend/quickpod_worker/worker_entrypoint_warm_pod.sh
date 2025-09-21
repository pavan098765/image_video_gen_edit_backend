#!/bin/bash
set -euo pipefail

# # --- Section 1: Python 3.11 Installation ---

# echo "Checking for Python 3.11..."
# if ! command -v python3.11 &> /dev/null; then
#     echo "Python 3.11 not found. Installing..."
#     sudo apt-get update -qq
#     sudo apt-get install -y --no-install-recommends software-properties-common
#     sudo add-apt-repository -y ppa:deadsnakes/ppa
#     sudo apt-get update -qq
#     sudo apt-get install -y --no-install-recommends python3.11 python3.11-dev python3.11-venv
#     sudo apt-get clean
#     echo "Python 3.11 installed."
# else
#     echo "Python 3.11 is already installed."
# fi

# --- Section 2: Virtual Environment Setup ---

# echo "Setting up virtual environment..."
# mkdir -p /quickpod/quickpod_worker
# cd /quickpod/quickpod_worker
# python3.11 -m venv venv_worker
# source venv_worker/bin/activate
# echo "Virtual environment created and activated."

# --- Section 3: Environment Variables ---

echo "Setting environment variables..."
export MODEL_KEY="multi-model-worker"
export CONTROLLER_URL="https://ky3libvtbr544cocs4lk4lqp6q.apigateway.ap-hyderabad-1.oci.customer-oci.com/imgvidgen"
export CONTROLLER_SHARED_SECRET="SECRETKEY"
export CONTROLLER_API_KEY="SECRETKEY"
export HEARTBEAT_INTERVAL="10"
export MAX_CONCURRENT_JOBS="1"

export WORKSPACE_DIR="/quickpod/quickpod_worker"

export HF_HOME="/models_storage/all_models"
export HF_HUB_CACHE="/models_storage/all_models"
export HF_TOKEN="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"

export LAMA_MODEL="/models_storage/all_models"
export XFORMERS_OFFLOAD_CPU="1"
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:512,expandable_segments:True"

echo "Environment variables set."

# --- Section 4: Dynamic CUDA and PyTorch Installation ---

# echo "Checking for existing CUDA installation..."
# CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version 2>/dev/null | grep 'release' | awk '{print $NF}' | sed 's/V//' | cut -d'.' -f1,2 || true)

# if [ -z "$CUDA_VERSION" ] || [ "$(echo "$CUDA_VERSION < 12.8" | bc -l)" -eq 1 ]; then
#     echo "CUDA not found or version < 12.8. Installing CUDA 12.9..."
#     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
#     sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
#     wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-ubuntu2404-12-9-local_12.9.0-575.51.03-1_amd64.deb
#     sudo dpkg -i cuda-repo-ubuntu2404-12-9-local_12.9.0-575.51.03-1_amd64.deb
#     sudo cp /var/cuda-repo-ubuntu2404-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
#     sudo apt-get update -qq
#     sudo apt-get -y install cuda-toolkit-12-9
#     CUDA_VERSION="12.9"
# fi

# # Install PyTorch matching detected CUDA version
# pip install -U pip
# case "$CUDA_VERSION" in
#     12.8)
#         echo "Detected CUDA $CUDA_VERSION. Installing PyTorch 2.7.1..."
#         pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
#         ;;
#     12.9)
#         echo "Detected CUDA $CUDA_VERSION. Installing PyTorch for CUDA 12.9..."
#         pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
#         ;;
#     *)
#         echo "Unsupported CUDA version: $CUDA_VERSION"
#         exit 1
#         ;;
# esac

# echo "CUDA and PyTorch setup complete."

# --- Section 5: Download Hugging Face Models ---

# echo "Downloading Hugging Face models..."
# mkdir -p /models_storage/fastwan
# cd /models_storage/fastwan
# git lfs install

# echo "Attempting anonymous clone..."
# if git clone https://huggingface.co/<your_model_repo> .; then
#     echo "Model cloned anonymously."
# else
#     echo "Anonymous clone failed."
#     if [ -n "$HF_TOKEN" ]; then
#         echo "Retrying with HF_TOKEN..."
#         git clone https://user:$HF_TOKEN@huggingface.co/<your_model_repo> .
#         echo "Model cloned using token."
#     else
#         echo "Error: Model is gated and no HF_TOKEN provided."
#         exit 1
#     fi
# fi


# --- Section 7: Start the Application (run as current/default user) ---
echo "Starting the application (as current user: $(whoami))..."
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
LOGFILE="/quickpod/quickpod_worker/app.log"

# Make sure the log file exists and is writable by the current user
mkdir -p "$(dirname "$LOGFILE")"
touch "$LOGFILE"
chmod 644 "$LOGFILE" || true

# Activate virtualenv and run uvicorn in background
# Note: 'nohup' + '&' means the process will keep running after this script exits.
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
