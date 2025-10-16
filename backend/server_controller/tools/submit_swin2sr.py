"""
submit_swin2sr.py
Job submitter for the Swin2SR upscaler model via controller.
Supports x2, x4, and x4_classic upscaling modes.
"""

import os
import sys
import json
import base64
import hmac
import hashlib
import requests
from io import BytesIO
from typing import Optional

# -----------------------------
# Hardcoded config
# -----------------------------
CONTROLLER_URL = os.getenv("CONTROLLER_URL", "https://genedit.mooo.com")
CONTROLLER_API_KEY = os.getenv("CONTROLLER_API_KEY", "SECRETKEY")
CONTROLLER_SHARED_SECRET = os.getenv("CONTROLLER_SHARED_SECRET", "SECRET")

MODEL_NAME = "swin2sr"  # must match worker model name

# Sample images for testing different resolutions
SAMPLE_IMAGES = {
    "portrait": "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/Snapinsta.app_262322643_4937545659591345_7487236998017046274_n_1080.jpg",
    "landscape": "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/Snapinsta.app_262322643_4937545659591345_7487236998017046274_n_1080.jpg",
}



# Available upscale models
SCALE_MODELS = {
    "2x": "swin2sr_x2",
    "4x": "swin2sr_x4",
    "4x_classic": "swin2sr_x4_classic"
}


# Default parameters
DEFAULT_PARAMS = {
    "timeout": 300,
    "scale": "swin2sr_x2",  # default to 2x upscaling
}

# -----------------------------
# Helper: Fetch and encode image
# -----------------------------
def fetch_and_encode_image(url: str) -> str:
    """Fetch image and return base64-encoded string."""
    try:
        print(f"🖼️  Downloading image from: {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        image_data = BytesIO(resp.content)
        return base64.b64encode(image_data.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"❌ Failed to fetch image from {url}: {e}")
        sys.exit(1)

def encode_local_image(path: str) -> str:
    """Encode a local image file as base64."""
    try:
        print(f"📂 Reading local image: {path}")
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"❌ Failed to read local image {path}: {e}")
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
def submit_upscale_job(
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    scale: str = "2x",
    timeout: int = 300
) -> dict:
    """
    Submit an upscaling job to the controller.
    
    Args:
        image_path: Local path to image file (optional)
        image_url: URL of image to upscale (optional)
        scale: Upscale factor ("2x", "4x", or "4x_classic")
        timeout: Job timeout in seconds
        
    At least one of image_path or image_url must be provided.
    """
    # Parameter validation
    if not image_path and not image_url:
        raise ValueError("Must provide either image_path or image_url")
    
    if scale not in SCALE_MODELS:
        raise ValueError(f"Invalid scale '{scale}'. Must be one of: {list(SCALE_MODELS.keys())}")
    
    # Get image data
    if image_path:
        init_image = encode_local_image(image_path)
    else:
        init_image = fetch_and_encode_image(image_url)
    
    # Prepare parameters
    params = DEFAULT_PARAMS.copy()
    params.update({
        "init_image": init_image,
        "scale": SCALE_MODELS[scale],
        "timeout": timeout
    })
    
    # Prepare payload
    payload = {
        "model": MODEL_NAME,
        "prompt": "",  # not used for upscaling
        "params": params
    }
    
    # Submit request
    headers = sign_payload(payload)
    url = CONTROLLER_URL.rstrip("/") + "/submit"
    
    print(f"\n🚀 Submitting {scale} upscale job to: {url}")
    print(f"📋 Parameters:")
    print(f"  - Scale: {scale} ({SCALE_MODELS[scale]})")
    print(f"  - Timeout: {timeout}s")
    print(f"\n🔑 Using Authorization: Bearer {CONTROLLER_API_KEY[:4]}...")
    
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()
        print("\n✅ Job submitted successfully!")
        print(f"📝 Job ID: {result.get('job_id')}")
        if result.get('download_url'):
            print(f"🔗 Result will be available at: {result['download_url']}")
        return result
    except Exception as e:
        print(f"\n❌ Job submission failed: {e}")
        if hasattr(e, 'response'):
            try:
                print("\nError details:", e.response.json())
            except:
                print("\nError response:", e.response.text)
        sys.exit(1)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Submit image upscaling job using Swin2SR")
    
    # Image source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", help="Path to local image file")
    source_group.add_argument("--url", help="URL of image to upscale")
    source_group.add_argument("--sample", choices=list(SAMPLE_IMAGES.keys()), 
                            help="Use a sample image")
    
    # Other parameters
    parser.add_argument("--scale", choices=list(SCALE_MODELS.keys()),
                       default="2x", help="Upscale factor")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Job timeout in seconds")
    
    args = parser.parse_args()
    
    # Determine image source
    image_path = None
    image_url = None
    
    if args.image:
        image_path = args.image
    elif args.url:
        image_url = args.url
    else:  # args.sample
        image_url = SAMPLE_IMAGES[args.sample]
    
    # Submit job
    result = submit_upscale_job(
        image_path=image_path,
        image_url=image_url,
        scale=args.scale,
        timeout=args.timeout
    )
    
    print("\nFull response:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()