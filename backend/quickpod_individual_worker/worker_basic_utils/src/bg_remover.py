# rmbg_remover.py

import base64
import os
from typing import Dict, Any, Optional, Union
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from transformers import pipeline

# -------------------------
# Setup: HF cache defaults (ensure these are set before importing transformers)
# -------------------------
# If your startup script sets HF_HOME / HF_HUB_CACHE, those will be used.
# Otherwise we provide a sensible default.
HF_CACHE_DIR = os.environ.get("HF_HOME", os.environ.get("HF_HUB_CACHE", "/models_storage/all_models"))
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = HF_CACHE_DIR

# Allow an optional env var to point to a pre-downloaded local model snapshot directory
# (set this in your startup script after snapshot_download)
RMBG_LOCAL_PATH = os.environ.get("RMBG_LOCAL_PATH", "").strip() or None

# -------------------------
# Helper: device selection
# -------------------------
DEVICE_INDEX = 0 if torch.cuda.is_available() else -1
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

# Force torch.load to respect GPU if available (keeps previous behaviour)
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(
    *args, **kwargs, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# -------------------------
# Initialize RMBG model (background remover) with preference for local path
# -------------------------
_model_source = RMBG_LOCAL_PATH or "briaai/RMBG-1.4"
try:
    # Passing device argument to pipeline: 0 for cuda, -1 for cpu
    bg_remover = pipeline(
        task="image-segmentation",
        model=_model_source,
        trust_remote_code=True,
        device=DEVICE_INDEX,
    )
    # If successful and we used a local path, that's ideal to avoid downloads
    init_msg = f"RMBG pipeline initialized from local path: {_model_source}" if RMBG_LOCAL_PATH else f"RMBG pipeline initialized from HF model id: {_model_source}"
    print(init_msg)
except Exception as e:
    # Restore original torch.load before raising to avoid leaving global monkeypatch applied
    torch.load = original_load
    raise RuntimeError(
        f"Failed to initialize RMBG pipeline from {_model_source}. "
        "If model is private, ensure HF_TOKEN is set in env, or set RMBG_LOCAL_PATH to a local snapshot. "
        f"Original error: {e}"
    )

# Restore original torch.load after model init
torch.load = original_load

# -------------------------
# Helpers
# -------------------------
def _load_image(img: Optional[Union[str, Image.Image, bytes]]) -> Optional[Image.Image]:
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
        decoded = base64.b64decode(s)
        return Image.open(BytesIO(decoded)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Unable to decode image from input: {e}")


# -------------------------
# Core background remover
# -------------------------
def remove_background(input_image: np.ndarray, return_mask: bool = False) -> np.ndarray:
    """
    Remove background from input image using RMBG.
    return_mask=True -> returns segmentation mask (white foreground, black background)
    return_mask=False -> returns image with transparent background (RGBA)
    """
    if input_image is None:
        raise ValueError("No input image provided for background removal.")

    pil_image = Image.fromarray(input_image).convert("RGB")

    try:
        if return_mask:
            # Many RMBG implementations support return_mask to get a PIL mask
            mask = bg_remover(pil_image, return_mask=True)
            # If the pipeline returned a dict/list, try to extract image
            if isinstance(mask, (list, tuple)) and len(mask) > 0:
                mask = mask[0]
            if hasattr(mask, "convert"):
                return np.array(mask.convert("L"))  # grayscale mask
            else:
                # fallback: try to interpret as ndarray-like
                return np.array(mask)
        else:
            result = bg_remover(pil_image)  # expected to return a PIL image with alpha
            if isinstance(result, (list, tuple)) and len(result) > 0:
                result = result[0]
            if hasattr(result, "convert"):
                return np.array(result.convert("RGBA"))
            else:
                return np.array(result)
    except Exception as e:
        raise RuntimeError(f"RMBG inference failed: {e}")


# -------------------------
# Inference wrapper
# -------------------------
def run_inference_rmbg(job_id: str, prompt:Optional[str], params: Dict[str, Any]) -> str:
    init_image = _load_image(params.get("init_image"))
    if init_image is None:
        raise ValueError("init_image is required for background removal")

    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    return_mask = bool(params.get("return_mask", False))
    suffix = "mask" if return_mask else "nobg"
    out_path = os.path.join(output_dir, f"rmbg_{suffix}_{job_id}.png")

    result = remove_background(np.array(init_image), return_mask=return_mask)

    # If mask (grayscale) save as L, else save RGBA
    if return_mask:
        Image.fromarray(result).convert("L").save(out_path)
    else:
        # ensure result has alpha channel
        img = Image.fromarray(result)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        img.save(out_path)

    return out_path


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    job_id = "rmbg_debug"
    input_path = "./input_image.jpg"

    if not os.path.exists(input_path):
        raise SystemExit("Provide input_image.jpg to test this script.")

    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Remove background (transparent output)
    params1 = {
        "init_image": img_b64,
        "output_dir": "./outputs",
        "return_mask": False,  # set True if you want just the mask
    }
    out1 = run_inference_rmbg(job_id, params=params1)
    print("✅ Background removed image saved at:", out1)

    # Save mask instead
    params2 = {
        "init_image": img_b64,
        "output_dir": "./outputs",
        "return_mask": True,
    }
    out2 = run_inference_rmbg(job_id + "_mask", params=params2)
    print("✅ Mask saved at:", out2)
