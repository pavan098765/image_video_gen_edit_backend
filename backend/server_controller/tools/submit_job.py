"""
submit_flux.py
Hardcoded job submitter for the flux_kontext_dev worker via controller.
Sends to /submit (controller), which handles presigned R2 URLs automatically.
"""

import os
import sys
import json
import base64
import hmac
import hashlib
import requests
from io import BytesIO

# -----------------------------
# Hardcoded config
# -----------------------------
CONTROLLER_URL = os.getenv("CONTROLLER_URL", "https://genedit.mooo.com")
CONTROLLER_API_KEY = os.getenv("CONTROLLER_API_KEY", "SECRETKEY")
CONTROLLER_SHARED_SECRET = os.getenv("CONTROLLER_SHARED_SECRET", "SECRET")  # must match worker/controller env

MODEL_NAME = "flux_kontext_dev"  # must match worker model name
PROMPT = "Change her body orientation so that she is facing the camera directly, with her upper body visible from the front. Her arms should be relaxed and resting on her sides naturally."
NEGATIVE_PROMPT = "blurry, distorted, lowres"

INIT_IMAGE_URL = (
    "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/Snapinsta.app_262322643_4937545659591345_7487236998017046274_n_1080.jpg"
)

TIMEOUT = 300
PARAMS = {
    "steps": 8,
    "guidance_scale": 3.0,
    "scale_image": True,
    "scale_factor": 4,
    "seed": 42,
    "randomize_seed": True,
    "height": 512,
    "width": 512,
    "negative_prompt": NEGATIVE_PROMPT,
    "timeout": TIMEOUT,
}

# -----------------------------
# Helper: Fetch and encode image
# -----------------------------
def fetch_and_encode_image(url: str) -> str:
    """Fetch image and return base64-encoded string."""
    try:
        print(f"🖼️  Downloading init image from: {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        image_data = BytesIO(resp.content)
        return base64.b64encode(image_data.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"❌ Failed to fetch image from {url}: {e}")
        sys.exit(1)

# -----------------------------
# Helper: Sign payload
# -----------------------------
def sign_payload(payload: dict) -> dict:
    """Return headers with X-Signature + Authorization."""
    payload_bytes = json.dumps(payload).encode("utf-8")
    signature = "sha256=" + hmac.new(
        CONTROLLER_SHARED_SECRET.encode("utf-8"), payload_bytes, hashlib.sha256
    ).hexdigest()
    headers = {
        "X-Signature": signature,
        "Content-Type": "application/json",
    }
    if CONTROLLER_API_KEY:
        headers["Authorization"] = f"Bearer {CONTROLLER_API_KEY}"
    return headers

# -----------------------------
# Submit job to controller
# -----------------------------
def main():
    # Encode init image
    init_image_b64 = fetch_and_encode_image(INIT_IMAGE_URL)
    PARAMS["init_image"] = init_image_b64

    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "params": PARAMS,
    }

    headers = sign_payload(payload)
    url = CONTROLLER_URL.rstrip("/") + "/submit"

    print(f"\n🚀 Submitting job to controller: {url}\n")
    # print(json.dumps(payload, indent=2))
    print(f"\n🔑 Using Authorization: Bearer {CONTROLLER_API_KEY[:4]}...")

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=1600)
    except Exception as e:
        print(f"❌ Request failed: {e}")
        sys.exit(2)

    print("\nStatus:", resp.status_code)
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception:
        print(resp.text)


if __name__ == "__main__":
    main()
