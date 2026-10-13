#!/bin/bash
set -e

echo "ComfyUI generic worker"
echo "  COMFYUI_DIR:  ${COMFYUI_DIR:-/app/comfyui}"
echo "  WORKFLOW_DIR: ${WORKFLOW_DIR:-/app/worker/workflow}"

mkdir -p "${COMFYUI_DIR:-/app/comfyui}/output"
mkdir -p "${COMFYUI_DIR:-/app/comfyui}/temp"
mkdir -p "${COMFYUI_DIR:-/app/comfyui}/input"
mkdir -p "${WORKSPACE_DIR:-/workspace}"

cd /app/worker
exec uvicorn src.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --timeout-keep-alive 300
