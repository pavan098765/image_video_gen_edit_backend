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
# Setup
# -------------------------
os.environ["HF_HOME"] = "/models_storage/all_models"
os.environ["HF_HUB_CACHE"] = "/models_storage/all_models"

# -------------------------
# Globals
# -------------------------
_bg_remover = None  # cache


def _get_bg_remover():
    """Lazy load RMBG pipeline (only once)."""
    global _bg_remover
    if _bg_remover is None:
        # Temporarily override torch.load to respect GPU if available
        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(
            *args,
            **kwargs,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        _bg_remover = pipeline(
            task="image-segmentation",
            model="briaai/RMBG-1.4",
            trust_remote_code=True,
        )

        # Restore torch.load
        torch.load = original_load

    return _bg_remover


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
    if os.path.exists(str(img)):
        return Image.open(str(img)).convert("RGB")
    try:
        decoded = base64.b64decode(img)
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
    return_mask=False -> returns image with transparent background
    """
    if input_image is None:
        raise ValueError("No input image provided for background removal.")

    pil_image = Image.fromarray(input_image)
    bg_remover = _get_bg_remover()

    if return_mask:
        mask = bg_remover(pil_image, return_mask=True)
        return np.array(mask.convert("L"))  # grayscale mask
    else:
        result = bg_remover(pil_image)  # transparent PNG
        return np.array(result.convert("RGBA"))


# -------------------------
# Inference wrapper
# -------------------------
def run_inference_rmbg(job_id: str, params: Dict[str, Any]) -> str:
    init_image = _load_image(params.get("init_image"))
    if init_image is None:
        raise ValueError("init_image is required for background removal")

    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    return_mask = bool(params.get("return_mask", False))
    suffix = "mask" if return_mask else "nobg"
    out_path = os.path.join(output_dir, f"rmbg_{suffix}_{job_id}.png")

    result = remove_background(np.array(init_image), return_mask=return_mask)
    Image.fromarray(result).save(out_path)

    return out_path
