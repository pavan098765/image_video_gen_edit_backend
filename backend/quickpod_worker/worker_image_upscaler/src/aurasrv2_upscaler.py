# aurasrv2_upscaler.py

import os
os.environ["HF_HOME"] = "/models_storage/all_models"
os.environ["HF_HUB_CACHE"] = "/models_storage/all_models"
import base64
from typing import Dict, Any, Optional, Union
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from aura_sr import AuraSR

# -------------------------
# Setup
# -------------------------

# Patch torch.load temporarily for device mapping
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(
    *args,
    **kwargs,
    map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# -------------------------
# Lazy globals
# -------------------------
_aura_sr = None


def _get_aura_sr() -> AuraSR:
    """Lazy-load AuraSR model once and cache it."""
    global _aura_sr
    if _aura_sr is None:
        print("[Init] Loading AuraSR-v2 model...")
        _aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")
        print("[Init] AuraSR-v2 model ready")
    return _aura_sr


# Restore original torch.load
torch.load = _original_load


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
# Core upscale function
# -------------------------
def upscale_image(input_image: np.ndarray, scale: int = 4) -> np.ndarray:
    """
    Upscale an image using AuraSR.
    AuraSR natively supports 4x; we emulate 2x by downscaling the 4x output.
    """
    if input_image is None:
        raise ValueError("No input image provided for upscaling.")

    pil_image = Image.fromarray(input_image)
    aura_sr = _get_aura_sr()

    # AuraSR only supports 4x
    upscaled_image = aura_sr.upscale_4x(pil_image)

    if scale == 2:
        w, h = upscaled_image.size
        upscaled_image = upscaled_image.resize((w // 2, h // 2), Image.LANCZOS)

    return np.array(upscaled_image)


# -------------------------
# Inference wrapper
# -------------------------
def run_inference_aurasr_upscale(job_id: str, params: Dict[str, Any]) -> str:
    init_image = _load_image(params.get("init_image"))
    if init_image is None:
        raise ValueError("init_image is required for upscaling")

    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"aurasr_upscaled_{job_id}.png")

    scale = int(params.get("scale", 4))  # default 4x
    result = upscale_image(np.array(init_image), scale=scale)
    Image.fromarray(result).save(out_path)

    return out_path


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    job_id = "upscale_debug"
    input_path = "./input_image.jpg"

    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    params = {
        "init_image": img_b64,
        "output_dir": "./outputs",
        "scale": 2,  # choose 2 or 4
    }

    out = run_inference_aurasr_upscale(job_id, params=params)
    print("✅ Upscaled image saved at:", out)
