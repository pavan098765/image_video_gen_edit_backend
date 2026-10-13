#!/bin/bash
set -e

echo "Qwen Image 2.1 Turbo ComfyUI worker"
echo "  COMFYUI_DIR:  ${COMFYUI_DIR:-/app/comfyui}"
echo "  WORKFLOW_DIR: ${WORKFLOW_DIR:-/app/worker/workflow}"

mkdir -p "${COMFYUI_DIR:-/app/comfyui}/output"
mkdir -p "${COMFYUI_DIR:-/app/comfyui}/temp"
mkdir -p "${COMFYUI_DIR:-/app/comfyui}/input"
mkdir -p "${COMFYUI_DIR:-/app/comfyui}/user/default/workflows"
mkdir -p "${WORKSPACE_DIR:-/workspace}"

if ls "${WORKFLOW_DIR:-/app/worker/workflow}"/*.json >/dev/null 2>&1; then
  cp "${WORKFLOW_DIR:-/app/worker/workflow}"/*.json "${COMFYUI_DIR:-/app/comfyui}/user/default/workflows/" || true
fi

cd /app/worker
exec uvicorn src.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --timeout-keep-alive 300
