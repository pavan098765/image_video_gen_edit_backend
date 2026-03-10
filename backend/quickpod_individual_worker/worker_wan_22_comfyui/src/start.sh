#!/bin/bash
# =============================================================================
# start.sh - Container entrypoint for Wan 2.2 I2V ComfyUI Worker
# =============================================================================
#
# This script:
# 1. Merges split safetensors files (from Docker layer optimization)
# 2. Starts the FastAPI worker (which internally starts ComfyUI)
#
# The FastAPI app manages ComfyUI's lifecycle (start/stop/health) via
# comfyui_bridge.py, so we only need to launch one process here.
# =============================================================================

set -e

echo "============================================"
echo "Wan 2.2 I2V ComfyUI Worker Starting"
echo "============================================"
echo "  COMFYUI_DIR:  ${COMFYUI_DIR:-/app/comfyui}"
echo "  MODELS_DIR:   ${MODELS_DIR:-/app/comfyui/models}"
echo "  WORKFLOW_DIR: ${WORKFLOW_DIR:-/app/worker/workflow}"
echo "  WORKSPACE:    ${WORKSPACE_DIR:-/workspace}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Merge split safetensors
# ---------------------------------------------------------------------------
echo "--- Step 1: Merging split safetensors ---"
cd /app/worker
python src/merge.py
echo ""

# ---------------------------------------------------------------------------
# Step 2: Ensure ComfyUI directories exist
# ---------------------------------------------------------------------------
echo "--- Step 2: Ensuring directories ---"
mkdir -p "${COMFYUI_DIR:-/app/comfyui}/output"
mkdir -p "${COMFYUI_DIR:-/app/comfyui}/temp"
mkdir -p "${COMFYUI_DIR:-/app/comfyui}/input"
mkdir -p "${WORKSPACE_DIR:-/workspace}"
echo "  Directories ready"
echo ""

# ---------------------------------------------------------------------------
# Step 3: Start FastAPI worker
# ---------------------------------------------------------------------------
# The FastAPI app (app.py) handles:
#   - Starting ComfyUI server as a subprocess (via comfyui_bridge.py)
#   - Waiting for ComfyUI to be ready
#   - Registering with the controller
#   - Processing jobs via /run endpoint
#   - Uploading results to R2
#   - Heartbeat / health monitoring
echo "--- Step 3: Starting FastAPI worker ---"
echo "  FastAPI will start ComfyUI internally via comfyui_bridge"
echo ""

cd /app/worker
exec uvicorn src.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --timeout-keep-alive 300
