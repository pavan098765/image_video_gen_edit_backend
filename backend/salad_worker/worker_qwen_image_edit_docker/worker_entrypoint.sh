#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8080}"

echo "[worker-entrypoint] Starting worker"
echo "  - Model Key: ${MODEL_KEY:-unset}"
echo "  - Worker ID: ${WORKER_ID:-unset}"
echo "  - Controller: ${CONTROLLER_URL:-unset}"
echo "  - Port: ${PORT}"

# Start FastAPI app
exec uvicorn src.app:app --host 0.0.0.0 --port "$PORT" --workers 1
