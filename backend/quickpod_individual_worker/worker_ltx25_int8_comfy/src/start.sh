#!/bin/bash
set -e

export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

# Triton/SolAttn kernel cache. Same rules as the H3 worker.
# NFS plan: backend/quickpod_individual_worker/NFS_VOLUME.md
# Do not mkdir /data — that hides a missing mount.
_default_triton=/root/.triton
if [ -z "${TRITON_CACHE_DIR:-}" ] || [ "${TRITON_CACHE_DIR}" = "${_default_triton}" ]; then
  if [ -n "${QUICKPOD_DATA_DIR:-}" ] && [ -d "${QUICKPOD_DATA_DIR%/}/triton" ]; then
    TRITON_CACHE_DIR="${QUICKPOD_DATA_DIR%/}/triton"
  elif [ -d /data/triton ]; then
    TRITON_CACHE_DIR=/data/triton
  else
    TRITON_CACHE_DIR="${_default_triton}"
  fi
fi
if ! mkdir -p "${TRITON_CACHE_DIR}" 2>/dev/null; then
  echo "WARN: cannot write Triton cache at ${TRITON_CACHE_DIR}, falling back to /root/.triton"
  TRITON_CACHE_DIR=/root/.triton
  mkdir -p "${TRITON_CACHE_DIR}"
fi
export TRITON_CACHE_DIR

echo "LTX-2.5 INT8 ComfyUI worker"
echo "  COMFYUI_DIR:  ${COMFYUI_DIR:-/app/comfyui}"
echo "  WORKFLOW_DIR: ${WORKFLOW_DIR:-/app/worker/workflow}"
echo "  CC:           ${CC}  TRITON_CACHE_DIR: ${TRITON_CACHE_DIR}"

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
