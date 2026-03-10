#!/usr/bin/env python3
"""
test_job.py - Send test job(s) to the locally running Wan 2.2 ComfyUI worker.

Usage:
  python src/test_job.py [test_name] [image_path]
  python src/test_job.py all <image_path>

test_name:
  i2v              Image-to-video, 1 context, interpolation+upscale on
  t2v              Text-to-video (no image)
  t2i              Text-to-image (no image)
  s2v              Subject-to-video (needs image)
  i2v-no-interp    I2V with enable_interpolation=False
  i2v-no-upscale   I2V with enable_upscale=False
  i2v-contexts     I2V with num_contexts=2
  all              Run all of the above in sequence (image_path required for image-based tests)

Prerequisites: Worker running (e.g. bash src/test_local.sh)

If you get "Unknown model: wan_22_comfyui", the server may be a different worker.
  - Either run this worker (test_local.sh) so the test discovers the model from /status.
  - Or set TEST_MODEL to the model name your worker expects, e.g.:
      TEST_MODEL=wan_22_fp8_lightx2v_i2v python src/test_job.py all test.jpg
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

# Default prompt (override with TEST_PROMPT env)
DEFAULT_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
    "Cinematic motion and smooth animation."
)
PROMPT = os.environ.get("TEST_PROMPT", DEFAULT_PROMPT)

WORKER_URL = os.environ.get("WORKER_URL", "http://localhost:8000")
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR")
if not WORKSPACE_DIR:
    _script_dir = Path(__file__).resolve().parent
    WORKSPACE_DIR = str(_script_dir.parent / "workspace")

# Model name: from TEST_MODEL env, or discovered from worker /status "models", or default
WORKER_MODEL = os.environ.get("TEST_MODEL", "").strip()

# Test definitions: (name, mode, need_image, extra_params_dict)
TESTS = [
    ("i2v", "i2v", True, {}),
    ("t2v", "t2v", False, {}),
    ("t2i", "t2i", False, {}),
    ("s2v", "s2v", True, {}),
    ("i2v-no-interp", "i2v", True, {"enable_interpolation": False}),
    ("i2v-no-upscale", "i2v", True, {"enable_upscale": False}),
    ("i2v-contexts", "i2v", True, {"num_contexts": 2}),
]


def health_check() -> tuple[bool, str]:
    """Returns (ok, model_name). Discovers model from worker /status if WORKER_MODEL not set."""
    global WORKER_MODEL
    try:
        r = requests.get(f"{WORKER_URL}/health", timeout=10)
        data = r.json()
        status = data.get("status", "unknown")
        if status != "ok":
            print(f"WARNING: Worker status is {status!r}. Continue anyway? (y/N) ", end="")
            if input().strip().lower() != "y":
                return False, ""
    except Exception as e:
        print(f"WARNING: Health check failed: {e}. Continue? (y/N) ", end="")
        if input().strip().lower() != "y":
            return False, ""
    # Discover model from worker if not set
    if not WORKER_MODEL:
        try:
            r = requests.get(f"{WORKER_URL}/status", timeout=10)
            data = r.json()
            models = data.get("models") or data.get("model_list")
            if models and len(models) > 0:
                WORKER_MODEL = models[0] if isinstance(models[0], str) else str(models[0])
                print(f"  Using worker model: {WORKER_MODEL}")
        except Exception:
            pass
    if not WORKER_MODEL:
        WORKER_MODEL = "wan_22_comfyui"
        print("  Using default model: wan_22_comfyui (set TEST_MODEL if worker expects another name)")
    return True, WORKER_MODEL


def build_params(need_image: bool, image_path: str | None, extra: dict) -> dict:
    params = {
        "width": 832,
        "height": 480,
        "steps": 8,
        "shift": 6.9,
        "num_frames": 81,
        "num_contexts": 1,
        "enable_interpolation": True,
        "interpolation_multiplier": 2,
        "enable_upscale": True,
    }
    params.update(extra)
    if need_image and image_path:
        with open(image_path, "rb") as f:
            params["init_image"] = base64.b64encode(f.read()).decode()
    return params


def run_one_test(
    test_name: str,
    mode: str,
    need_image: bool,
    extra_params: dict,
    image_path: str | None,
    model: str,
) -> str | None:
    job_id = f"test-{test_name}-{int(time.time())}"
    params = build_params(need_image, image_path, extra_params)
    payload = {
        "job_id": job_id,
        "prompt": PROMPT,
        "model": model,
        "mode": mode,
        "params": params,
        "upload_url": f"http://dummy-upload/{job_id}",
        "result_key": f"test-result-{job_id}",
    }
    print(f"  Job ID: {job_id}")
    try:
        r = requests.post(
            f"{WORKER_URL}/run",
            json=payload,
            timeout=30,
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        return None
    if r.status_code != 200:
        print(f"  FAILED (HTTP {r.status_code}): {r.text}")
        return None
    print("  Submitted OK")
    return job_id


def poll_until_done(job_id: str) -> bool:
    while True:
        time.sleep(5)
        try:
            r = requests.get(f"{WORKER_URL}/status", timeout=10)
            data = r.json()
            jobs = data.get("jobs", {})
            job = jobs.get(job_id, {})
            status = job.get("status", "not_found")
        except Exception as e:
            print(f"  [{job_id}] status error: {e}")
            continue
        if status == "completed":
            print(f"  [{job_id}] COMPLETED")
            print(f"  Time: {job.get('processing_time', '?')}s, key: {job.get('result_r2_key', '?')}")
            out_path = os.path.join(WORKSPACE_DIR, f"{job_id}.mp4")
            if os.path.isfile(out_path):
                print(f"  File: {out_path} ({os.path.getsize(out_path)} bytes)")
            return True
        if status == "failed":
            print(f"  [{job_id}] FAILED")
            print(f"  Error: {job.get('error', '?')}")
            return False
        if status == "not_found":
            print(f"  [{job_id}] not_found (may have been cleaned up)")
            return True
        print(f"  [{job_id}] {status} ...")


def run_test(
    test_name: str,
    mode: str,
    need_image: bool,
    extra_params: dict,
    image_path: str | None,
    model: str,
) -> bool:
    print("")
    print(f"========== Test: {test_name} (mode={mode}) ==========")
    if need_image and not (image_path and os.path.isfile(image_path)):
        print(f"  ERROR: Image required. Provide image path for this test.")
        return False
    job_id = run_one_test(test_name, mode, need_image, extra_params, image_path, model)
    if not job_id:
        return False
    return poll_until_done(job_id)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send test job(s) to Wan 2.2 ComfyUI worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "test_name",
        nargs="?",
        default="i2v",
        choices=[t[0] for t in TESTS] + ["all"],
        help="Test to run",
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default="",
        help="Path to image (required for i2v, s2v, i2v-no-*, i2v-contexts; required when test_name=all)",
    )
    args = parser.parse_args()
    test_name = args.test_name
    image_path = args.image_path.strip() or None

    print("============================================")
    print("Wan 2.2 ComfyUI Worker - Test Job(s)")
    print("============================================")
    print(f"  Test:    {test_name}")
    print(f"  Image:   {image_path or '<none>'}")
    print(f"  Worker:  {WORKER_URL}")
    print(f"  Outputs: {WORKSPACE_DIR}/<job_id>.mp4")
    print("")

    ok, model = health_check()
    if not ok:
        return 1
    print("")

    if test_name == "all":
        if not image_path or not os.path.isfile(image_path):
            print("ERROR: For 'all' tests, provide image path: python src/test_job.py all <image_path>")
            return 1
        failed = 0
        for name, mode, need_image, extra in TESTS:
            if not run_test(name, mode, need_image, extra, image_path, model):
                failed += 1
        print("")
        print("============================================")
        if failed == 0:
            print("All tests completed successfully.")
            print("")
            print("Outputs (one file per test):")
            print(f"  Directory: {WORKSPACE_DIR}")
            for f in sorted(Path(WORKSPACE_DIR).glob("test-*.mp4")):
                print(f"    {f.name} ({f.stat().st_size} bytes)")
        else:
            print(f"{failed} test(s) failed.")
            return 1
        print("============================================")
        return 0

    # Single test
    for name, mode, need_image, extra in TESTS:
        if name == test_name:
            ok = run_test(name, mode, need_image, extra, image_path, model)
            return 0 if ok else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
