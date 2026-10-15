#!/bin/bash
set -e

export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

# Triton/SolAttn kernel cache. Default is local /root/.triton.
# Shared NFS plan (weights + this cache): ../NFS_VOLUME.md
# NFS (later, not auto-created):
#   TRITON_CACHE_DIR=/data/triton
#   or QUICKPOD_DATA_DIR=/data with /data/triton already on the volume
#   or just mkdir /data/triton on the mount — we pick it up if still on the default path.
# Do not mkdir /data ourselves — that would hide a missing mount.
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

echo "MiniMax H3 ComfyUI worker"
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
