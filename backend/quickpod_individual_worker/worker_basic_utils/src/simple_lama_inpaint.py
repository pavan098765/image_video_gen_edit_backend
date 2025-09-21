"""
Simple LaMa inpainting wrapper with outpainting-friendly mask preparation.
"""

import base64
import os
from io import BytesIO
from typing import Dict, Any, Optional, Union
from PIL import Image
import torch

# ----------------------------
# Environment Setup
# ----------------------------
os.environ["XFORMERS_OFFLOAD_CPU"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "garbage_collection_threshold:0.6,max_split_size_mb:512,expandable_segments:True"
)

# Force HuggingFace model caches into /models_storage
CACHE_DIR = "/models_storage/all_models"
for var in ["LAMA_MODEL", "HF_HOME", "HF_HUB_CACHE", "XDG_CACHE_HOME"]:
    os.environ[var] = CACHE_DIR

from simple_lama_inpainting import SimpleLama

# ----------------------------
# Globals
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_OUTPUT_DIR = "./outputs_outpaint"
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# Optional local snapshot path (set in startup script)
LAMA_LOCAL_PATH = os.environ.get("LAMA_LOCAL_PATH", "").strip() or None

# Initialize model once
try:
    if LAMA_LOCAL_PATH and os.path.exists(LAMA_LOCAL_PATH):
        SIMPLE_LAMA = SimpleLama(model_path=LAMA_LOCAL_PATH)
        print(f"✅ SimpleLama initialized from local path: {LAMA_LOCAL_PATH}")
    else:
        SIMPLE_LAMA = SimpleLama()
        if LAMA_LOCAL_PATH:
            print(f"⚠️ LAMA_LOCAL_PATH={LAMA_LOCAL_PATH} not found, falling back to default download.")
        else:
            print("ℹ️ No LAMA_LOCAL_PATH set, using default SimpleLama() download.")
except Exception as e:
    raise RuntimeError(
        f"Failed to initialize SimpleLama. "
        f"If model is gated, ensure HF_TOKEN is set in env, "
        f"or set LAMA_LOCAL_PATH to a local snapshot. Error: {e}"
    )

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


def _load_image(img: Optional[Union[str, Image.Image, bytes]]) -> Optional[Image.Image]:
    """Load an image from path, PIL, bytes, or base64."""
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, bytes):
        return Image.open(BytesIO(img)).convert("RGB")

    s = str(img)
    if os.path.exists(s):
        return Image.open(s).convert("RGB")

    try:
        decoded = _safe_b64decode(s)
        return Image.open(BytesIO(decoded)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Unable to decode image: {e}")


def process_with_lama(init_image: Image.Image, mask_image: Image.Image, output_path: str) -> str:
    """Run LaMa inpainting given an image and mask."""
    result = SIMPLE_LAMA.inpaint(init_image, mask_image)
    result.save(output_path)
    return os.path.abspath(output_path)


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
    mask_img = _load_image(params.get("mask_image"))

    if init_image is None:
        raise ValueError("init_image is required for inpainting")
    if mask_img is None:
        raise ValueError("mask_image is required for inpainting")
    if init_image.size != mask_img.size:
        raise ValueError("init_image and mask_image must be the same size")

    output_dir = params.get("output_dir", DEFAULT_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, f"lama_inpaint_{job_id}.png")
    return process_with_lama(init_image, mask_img, out_path)


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
