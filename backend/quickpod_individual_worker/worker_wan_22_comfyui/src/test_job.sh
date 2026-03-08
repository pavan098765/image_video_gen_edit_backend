#!/bin/bash
# =============================================================================
# test_job.sh - Wrapper to run test_job.py (Wan 2.2 ComfyUI worker tests)
# =============================================================================
#
# Usage: same as test_job.py
#   bash src/test_job.sh [test_name] [image_path]
#   bash src/test_job.sh all <image_path>
#
# test_name: i2v | t2v | t2i | s2v | i2v-no-interp | i2v-no-upscale | i2v-contexts | all
#
# Prerequisites: Worker running (bash src/test_local.sh)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/test_job.py" "$@"
