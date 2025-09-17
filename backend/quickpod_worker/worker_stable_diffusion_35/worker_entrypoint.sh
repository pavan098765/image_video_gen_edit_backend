#!/usr/bin/env bash
set -euo pipefail

# Optional env vars:
# WORKER_ID, MODEL_KEY, CONTROLLER_URL, CONTROLLER_SHARED_SECRET, CONTROLLER_API_KEY (if using presign auth)
# R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT, R2_BUCKET (if worker uploads directly)
# WORKER_PUBLIC_URL - public reachable URL for this worker (controller uses this)
# PORT - port to listen on (default 8080)

PORT="${PORT:-8080}"

echo "[worker-entrypoint] Starting worker (model=${MODEL_KEY:-unset}) on port ${PORT}"
exec uvicorn app:app --host 0.0.0.0 --port "$PORT" --workers 1
