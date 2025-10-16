# rmbg_remover.py

import base64
import os
from typing import Dict, Any, Optional
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from transformers import pipeline
from upscaler import upscale_image

# -------------------------
# Config: model location
# -------------------------
# If you bake RMBG into the container, set RMBG_LOCAL_PATH accordingly.
RMBG_LOCAL_PATH = os.environ.get("RMBG_LOCAL_PATH", "/models/rmbg").strip()

# -------------------------
# Initialize RMBG model (GPU only)
# -------------------------
_model_source = RMBG_LOCAL_PATH or "briaai/RMBG-1.4"

try:
    bg_remover = pipeline(
        task="image-segmentation",
        model=_model_source,
        trust_remote_code=True,
        device=0,  # force GPU index 0
    )
    print(f"✅ RMBG pipeline initialized from: {_model_source}")
except Exception as e:
    raise RuntimeError(
        f"❌ Failed to initialize RMBG pipeline from {_model_source}. "
        f"Ensure the model is baked into the container. Error: {e}"
    )

# -------------------------
# Helpers
# -------------------------
def _load_image(img: Optional[str | Image.Image | bytes]) -> Optional[Image.Image]:
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
    return_mask=True -> segmentation mask (white foreground, black background)
    return_mask=False -> image with transparent background (RGBA)
    """
    if input_image is None:
        raise ValueError("No input image provided for background removal.")

    pil_image = Image.fromarray(input_image).convert("RGB")

    try:
        if return_mask:
            mask = bg_remover(pil_image, return_mask=True)
            if isinstance(mask, (list, tuple)) and len(mask) > 0:
                mask = mask[0]
            if hasattr(mask, "convert"):
                return np.array(mask.convert("L"))
            return np.array(mask)
        else:
            result = bg_remover(pil_image)
            if isinstance(result, (list, tuple)) and len(result) > 0:
                result = result[0]
            if hasattr(result, "convert"):
                return np.array(result.convert("RGBA"))
            return np.array(result)
    except Exception as e:
        raise RuntimeError(f"RMBG inference failed: {e}")

# -------------------------
# Inference wrapper
# -------------------------
def run_inference_rmbg(job_id: str, prompt: Optional[str], params: Dict[str, Any]) -> str:
    init_image = _load_image(params.get("init_image"))
    if init_image is None:
        raise ValueError("init_image is required for background removal")

    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    return_mask = bool(params.get("return_mask", False))
    suffix = "mask" if return_mask else "nobg"
    out_path = os.path.join(output_dir, f"rmbg_{suffix}_{job_id}.png")

    result = remove_background(np.array(init_image), return_mask=return_mask)

    if return_mask:
        Image.fromarray(result).convert("L").save(out_path)
    else:
        img = Image.fromarray(result)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        img.save(out_path)
    
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

    # Remove background
    params1 = {"init_image": img_b64, "output_dir": "./outputs", "return_mask": False}
    out1 = run_inference_rmbg(job_id, prompt=None, params=params1)
    print("✅ Background removed image saved at:", out1)

    # Save mask
    params2 = {"init_image": img_b64, "output_dir": "./outputs", "return_mask": True}
    out2 = run_inference_rmbg(job_id + "_mask", prompt=None, params=params2)
    print("✅ Mask saved at:", out2)
