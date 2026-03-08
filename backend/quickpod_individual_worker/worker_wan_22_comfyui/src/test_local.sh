#!/bin/bash
# =============================================================================
# test_local.sh - Run the Wan 2.2 I2V ComfyUI Worker locally on a pod
# =============================================================================
#
# Usage (from the worker_wan_22_comfyui directory):
#
#   # First time - download everything + install deps:
#   python src/prep_model.py --no-split --no-dockerfile --install-deps
#
#   # Then start the worker:
#   bash src/test_local.sh
#
#   # Or to do everything in one go:
#   bash src/test_local.sh --prep
#
# The script sets up env vars pointing to local directories (instead of
# Docker /app/... paths), then starts the FastAPI worker which internally
# launches ComfyUI.
#
# Test with:
#   curl http://localhost:8000/health
#
#   # Send a test job (base64 encode an image first):
#   IMG_B64=$(base64 -w0 test_image.png)
#   curl -X POST http://localhost:8000/run \
#     -H "Content-Type: application/json" \
#     -d '{
#       "job_id": "test-001",
#       "prompt": "a beautiful sunset timelapse",
#       "model": "wan_22_comfyui",
#       "params": {
#         "init_image": "'$IMG_B64'",
#         "width": 832,
#         "height": 480,
#         "steps": 8,
#         "cfg": 1.0,
#         "num_frames": 81
#       },
#       "upload_url": "http://dummy-upload/test",
#       "result_key": "test-result"
#     }'
# =============================================================================

set -e

# Resolve the worker root directory (parent of src/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "Wan 2.2 I2V ComfyUI Worker - Local Test Mode"
echo "============================================"
echo "  Worker root:  $WORKER_ROOT"

# ---------------------------------------------------------------------------
# Optional: run prep_model first
# ---------------------------------------------------------------------------
if [[ "$1" == "--prep" ]]; then
    echo ""
    echo "--- Running prep_model.py (no-split, no-dockerfile, install-deps) ---"
    cd "$WORKER_ROOT"
    python src/prep_model.py --no-split --no-dockerfile --install-deps
    echo ""
fi

# ---------------------------------------------------------------------------
# Set environment variables for local paths
# ---------------------------------------------------------------------------
export COMFYUI_DIR="$WORKER_ROOT/comfyui"
export MODELS_DIR="$COMFYUI_DIR/models"
export WORKFLOW_DIR="$WORKER_ROOT/workflow"
export WORKSPACE_DIR="$WORKER_ROOT/workspace"
export RIFE_WEIGHTS_DIR="$WORKER_ROOT/rife_weights"
export PYTHONPATH="$WORKER_ROOT/src:${PYTHONPATH:-}"

# Worker identity (defaults for local testing)
export CONTAINER_LABEL="${CONTAINER_LABEL:-local-test}"
export PUBLIC_IPADDR="${PUBLIC_IPADDR:-127.0.0.1}"
export MODEL_KEY="${MODEL_KEY:-wan-22-comfyui-worker}"

# No controller for local testing (unless explicitly set)
export CONTROLLER_URL="${CONTROLLER_URL:-}"

echo "  COMFYUI_DIR:  $COMFYUI_DIR"
echo "  MODELS_DIR:   $MODELS_DIR"
echo "  WORKFLOW_DIR: $WORKFLOW_DIR"
echo "  WORKSPACE:    $WORKSPACE_DIR"
echo "  PYTHONPATH:   $PYTHONPATH"
echo ""

# ---------------------------------------------------------------------------
# Verify ComfyUI is present
# ---------------------------------------------------------------------------
if [ ! -f "$COMFYUI_DIR/main.py" ]; then
    echo "ERROR: ComfyUI not found at $COMFYUI_DIR"
    echo "  Run first: python src/prep_model.py --no-split --no-dockerfile --install-deps"
    exit 1
fi

# ---------------------------------------------------------------------------
# Verify at least one model file exists
# ---------------------------------------------------------------------------
MODEL_COUNT=$(find "$MODELS_DIR" -name "*.safetensors" 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "ERROR: No .safetensors files found in $MODELS_DIR"
    echo "  Run first: python src/prep_model.py --no-split --no-dockerfile --install-deps"
    exit 1
fi
echo "  Found $MODEL_COUNT model file(s)"

# ---------------------------------------------------------------------------
# Verify workflow exists
# ---------------------------------------------------------------------------
NEW_WORKFLOW="WAN2.2 T2V-I2V-T2I-S2V K3NK v2.5.4 SVI.json"
if [ ! -f "$WORKFLOW_DIR/$NEW_WORKFLOW" ]; then
    echo "ERROR: Workflow not found at $WORKFLOW_DIR/$NEW_WORKFLOW"
    exit 1
fi
echo "  Workflow found: $NEW_WORKFLOW"

# ---------------------------------------------------------------------------
# Create workspace directory
# ---------------------------------------------------------------------------
mkdir -p "$WORKSPACE_DIR"
mkdir -p "$COMFYUI_DIR/output"
mkdir -p "$COMFYUI_DIR/temp"
mkdir -p "$COMFYUI_DIR/input"
echo "  Directories ready"
echo ""

# ---------------------------------------------------------------------------
# Start the worker
# ---------------------------------------------------------------------------
echo "--- Starting FastAPI worker ---"
echo "  FastAPI on port 8000, ComfyUI on port 8188 (internal)"
echo "  Health check: curl http://localhost:8000/health"
echo ""

cd "$WORKER_ROOT"
exec python3 -m uvicorn src.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --timeout-keep-alive 300
