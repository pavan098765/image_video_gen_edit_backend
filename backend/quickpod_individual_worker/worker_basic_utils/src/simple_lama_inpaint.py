"""
Simple LaMa inpainting wrapper with outpainting-friendly mask preparation.
"""

import base64
import os
from io import BytesIO
from typing import Dict, Any, Optional, Union
from PIL import Image
import torch
from simple_lama_inpainting import SimpleLama
from upscaler import upscale_image

# ----------------------------
# Globals
# ----------------------------
DEVICE = "cuda"  # enforce GPU only
DEFAULT_OUTPUT_DIR = "./outputs_outpaint"
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# Local baked TorchScript path (set in container build or startup)
LAMA_LOCAL_PATH = os.environ.get("LAMA_LOCAL_PATH", "/app/models/basic_utils/lama/lama.pt").strip()
os.environ["LAMA_MODEL"] = LAMA_LOCAL_PATH

# Initialize model once
try:
    if not os.path.exists(LAMA_LOCAL_PATH):
        raise FileNotFoundError(f"❌ LaMa model file not found at {LAMA_LOCAL_PATH}")
    SIMPLE_LAMA = SimpleLama()
    print(f"✅ SimpleLama initialized from: {LAMA_LOCAL_PATH}")
except Exception as e:
    raise RuntimeError(f"❌ Failed to initialize SimpleLama. Error: {e}")

# ----------------------------
# Helpers
# ----------------------------
def _safe_b64decode(data: Union[str, bytes]) -> bytes:
    """Safely decode base64 string or bytes."""
    if isinstance(data, str):
        raw = data.encode()
    elif isinstance(data, bytes):
        raw = data
    else:
        raise ValueError("Base64 input must be str or bytes")

    missing_padding = len(raw) % 4
    if missing_padding:
        raw += b"=" * (4 - missing_padding)

    return base64.b64decode(raw)


def _load_image(img: Optional[Union[str, Image.Image, bytes]], force_gray: bool = False) -> Optional[Image.Image]:
    """Load an image from path, PIL, bytes, or base64."""
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("L" if force_gray else "RGB")
    if isinstance(img, bytes):
        return Image.open(BytesIO(img)).convert("L" if force_gray else "RGB")

    s = str(img)
    if os.path.exists(s):
        return Image.open(s).convert("L" if force_gray else "RGB")

    try:
        decoded = _safe_b64decode(s)
        return Image.open(BytesIO(decoded)).convert("L" if force_gray else "RGB")
    except Exception as e:
        raise ValueError(f"Unable to decode image: {e}")


def process_with_lama(job_id: str, init_image: Image.Image, mask_image: Image.Image, output_dir: str, params: Dict[str, Any]) -> str:
    """Run LaMa inpainting given an image and mask."""

    out_path = os.path.join(output_dir, f"lama_inpaint_{job_id}.png")
    result = SIMPLE_LAMA(init_image, mask_image)
    result.save(out_path)

    # -------------------------
    # Optional Upscaling and compulsory metadata removal
    # -------------------------
    scale_image = params.get("scale_image", True)
    scale_factor = params.get("scale_factor", 2) if scale_image else 0
    model_name = "RealESRGAN_x2plus" if scale_factor == 2 else "RealESRGAN_x4plus"

    upscaled = upscale_image(out_path,  model_name=model_name, scale_factor=scale_factor)

    upscaled_path = os.path.join(output_dir, f"{job_id}_upscaled.png")
    upscaled.save(upscaled_path)
    return os.path.abspath(upscaled_path)

# ----------------------------
# Core Function
# ----------------------------
def run_inference_simple_lama_inpaint(job_id: str, prompt: Optional[str], params: Dict[str, Any]) -> str:
    """
    Run LaMa inpainting.
    - job_id: unique job identifier
    - prompt: optional (ignored for LaMa, included for API consistency)
    - params: dict with keys:
        init_image (required) - base64, path, or PIL
        mask_image (required) - base64, path, or PIL
        output_dir (optional) - where to save output
    """
    init_image = _load_image(params.get("init_image"))
    mask_img = _load_image(params.get("mask_image"), force_gray=True)

    if init_image is None:
        raise ValueError("init_image is required for inpainting")
    if mask_img is None:
        raise ValueError("mask_image is required for inpainting")
    if init_image.size != mask_img.size:
        raise ValueError("init_image and mask_image must be the same size")

    output_dir = params.get("output_dir", DEFAULT_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    return process_with_lama(job_id, init_image, mask_img, output_dir, params)

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    input_path = "./input_image.jpg"
    mask_path = "./mask_image.png"

    if not os.path.exists(input_path) or not os.path.exists(mask_path):
        raise SystemExit("❌ Please provide input_image.jpg and mask_image.png to test this script.")

    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    with open(mask_path, "rb") as f:
        mask_b64 = base64.b64encode(f.read()).decode("utf-8")

    test_params = {
        "init_image": img_b64,
        "mask_image": mask_b64,
        "output_dir": "./outputs_outpaint",
    }

    print("Using device:", DEVICE)
    result_path = run_inference_simple_lama_inpaint("simple_lama_test", None, test_params)
    print("✅ Saved:", result_path)
