#!/bin/bash
# =============================================================================
# test_job.sh - Send a test job to the locally running worker
# =============================================================================
#
# Usage:
#   bash src/test_job.sh <path_to_image.png> [num_segments] [enable_rife] [auto_prompts]
#   bash src/test_job.sh test_image.png          # 1 segment, rife on, auto_prompts on
#   bash src/test_job.sh test_image.png 2        # 2 segments
#   bash src/test_job.sh test_image.png 2 true   # 2 segments, rife on
#   bash src/test_job.sh test_image.png 3 false false  # 3 segments, no rife, no auto prompts
#
# Prerequisites:
#   - Worker must be running (bash src/test_local.sh)
#   - Provide a reference image as argument
#
# =============================================================================

set -e

WORKER_URL="${WORKER_URL:-http://localhost:8000}"

# ---------------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------------
echo "--- Health Check ---"
HEALTH=$(curl -s "$WORKER_URL/health")
echo "$HEALTH" | python3 -m json.tool 2>/dev/null || echo "$HEALTH"

STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "unknown")
if [ "$STATUS" != "ok" ]; then
    echo ""
    echo "WARNING: Worker status is '$STATUS' (not 'ok')."
    echo "  If 'starting', wait for ComfyUI to finish loading and retry."
    echo "  If 'unhealthy', check worker logs."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""

# ---------------------------------------------------------------------------
# 2. Encode reference image
# ---------------------------------------------------------------------------
IMAGE_PATH="${1:-}"

if [ -z "$IMAGE_PATH" ]; then
    echo "ERROR: No image path provided."
    echo "Usage: bash src/test_job.sh <path_to_image.png>"
    exit 1
fi

if [ ! -f "$IMAGE_PATH" ]; then
    echo "ERROR: Image file not found: $IMAGE_PATH"
    exit 1
fi

NUM_SEGMENTS="${2:-1}"  # Default to 1 segment for fast testing
ENABLE_RIFE="${3:-true}"  # Default to true
AUTO_PROMPTS="${4:-true}"  # Default to true
RIFE_TARGET_FPS="${RIFE_TARGET_FPS:-30}"  # Override via env var

echo "--- Preparing Payload ---"
echo "  Image: $IMAGE_PATH"
echo "  Segments: $NUM_SEGMENTS (of 10)"
echo "  RIFE: $ENABLE_RIFE (target ${RIFE_TARGET_FPS}fps)"
echo "  Auto prompts: $AUTO_PROMPTS"
IMG_SIZE=$(stat --format="%s" "$IMAGE_PATH" 2>/dev/null || stat -f "%z" "$IMAGE_PATH" 2>/dev/null)
echo "  Size:  $IMG_SIZE bytes"

JOB_ID="test-$(date +%s)"
PROMPT="${TEST_PROMPT:-Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the felines intricate details and the refreshing atmosphere of the seaside. Add cinematic motion and smooth animation.}"

# Build the JSON payload via Python to avoid shell argument size limits
PAYLOAD_FILE=$(mktemp /tmp/test_job_payload_XXXXXX.json)
trap "rm -f $PAYLOAD_FILE" EXIT

python3 -c "
import base64, json, sys
with open('$IMAGE_PATH', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()
print(f'  Base64 length: {len(img_b64)} chars', file=sys.stderr)
payload = {
    'job_id': '$JOB_ID',
    'prompt': '$PROMPT',
    'model': 'wan_22_comfyui',
    'params': {
        'init_image': img_b64,
        'width': 832,
        'height': 480,
        'steps': 8,
        'cfg': 1.0,
        'num_frames': 81,
        'num_segments': $NUM_SEGMENTS,
        'enable_rife': '$ENABLE_RIFE' == 'true',
        'rife_target_fps': $RIFE_TARGET_FPS,
        'auto_prompts': '$AUTO_PROMPTS' == 'true',
    },
    'upload_url': 'http://dummy-upload/$JOB_ID',
    'result_key': 'test-result-$JOB_ID',
}
with open('$PAYLOAD_FILE', 'w') as f:
    json.dump(payload, f)
print(f'  Payload file: {len(json.dumps(payload))} bytes', file=sys.stderr)
"
echo ""

# ---------------------------------------------------------------------------
# 3. Send test job
# ---------------------------------------------------------------------------
echo "--- Sending Test Job ---"
echo "  Job ID:       $JOB_ID"
echo "  Prompt:       ${PROMPT:0:80}..."
echo "  Segments:     $NUM_SEGMENTS"
echo "  RIFE:         $ENABLE_RIFE (target ${RIFE_TARGET_FPS}fps)"
echo "  Auto prompts: $AUTO_PROMPTS"
echo ""

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$WORKER_URL/run" \
  -H "Content-Type: application/json" \
  -d @"$PAYLOAD_FILE")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)

echo "  HTTP Status: $HTTP_CODE"
echo "  Response:"
echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
echo ""

if [ "$HTTP_CODE" != "200" ]; then
    echo "ERROR: Job submission failed (HTTP $HTTP_CODE)"
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. Poll job status
# ---------------------------------------------------------------------------
echo "--- Polling Job Status ---"
echo "  (Ctrl+C to stop polling, job continues in background)"
echo ""

while true; do
    sleep 5
    STATUS_RESP=$(curl -s "$WORKER_URL/status")
    JOB_STATUS=$(echo "$STATUS_RESP" | python3 -c "
import sys, json
data = json.load(sys.stdin)
jobs = data.get('jobs', {})
job = jobs.get('$JOB_ID', {})
status = job.get('status', 'not_found')
print(status)
" 2>/dev/null || echo "unknown")

    TIMESTAMP=$(date '+%H:%M:%S')

    if [ "$JOB_STATUS" == "completed" ]; then
        echo "  [$TIMESTAMP] Job COMPLETED!"
        RESULT=$(echo "$STATUS_RESP" | python3 -c "
import sys, json
data = json.load(sys.stdin)
job = data.get('jobs', {}).get('$JOB_ID', {})
print(f\"  Processing time: {job.get('processing_time', '?')}s\")
print(f\"  Result key: {job.get('result_r2_key', '?')}\")
" 2>/dev/null)
        echo "$RESULT"
        echo ""
        WS_DIR="${WORKSPACE_DIR:-/workspace}"
        echo "  Output video should be at: $WS_DIR/$JOB_ID.mp4"
        if [ -f "$WS_DIR/$JOB_ID.mp4" ]; then
            FILE_SIZE=$(stat --format="%s" "$WS_DIR/$JOB_ID.mp4" 2>/dev/null || stat -f "%z" "$WS_DIR/$JOB_ID.mp4" 2>/dev/null)
            echo "  File found! Size: $FILE_SIZE bytes"
        else
            echo "  (checking alternative locations...)"
            find /work_space -name "${JOB_ID}*" 2>/dev/null | head -5
        fi
        break
    elif [ "$JOB_STATUS" == "failed" ]; then
        echo "  [$TIMESTAMP] Job FAILED!"
        ERROR=$(echo "$STATUS_RESP" | python3 -c "
import sys, json
data = json.load(sys.stdin)
job = data.get('jobs', {}).get('$JOB_ID', {})
print(f\"  Error: {job.get('error', '?')}\")
print(f\"  Type: {job.get('error_type', '?')}\")
" 2>/dev/null)
        echo "$ERROR"
        exit 1
    elif [ "$JOB_STATUS" == "not_found" ]; then
        # Job might have been cleaned up already (completed fast)
        echo "  [$TIMESTAMP] Job not found in status (may have completed and been cleaned up)"
        echo "  Check: ls -la workspace/$JOB_ID.*"
        break
    else
        echo "  [$TIMESTAMP] Status: $JOB_STATUS ..."
    fi
done
