#!/usr/bin/env bash
set -euo pipefail

# Default port (can override with env PORT=xxxx)
PORT="${PORT:-8080}"

echo "[worker-entrypoint] Starting worker on port ${PORT}"
echo "[worker-entrypoint] Model=${MODEL_KEY:-unset}, Worker ID=${WORKER_ID:-unset}"

# Start FastAPI app (app.py must define `app = FastAPI()`)
exec uvicorn app:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1
