# aurasrv2_upscaler.py
# ✅ Lazy-loading implementation for AuraSR-v2
# Supports 2x or 4x upscale, auto-handles multiple calls efficiently.

import base64
import os
from typing import Dict, Any, Optional, Union
from io import BytesIO
from PIL import Image
import httpx
import numpy as np
import torch

torch.cuda.empty_cache()

from aura_sr import AuraSR

# --- Auto-merge split safetensors if merge.py exists ---
try:
    from merge import main as merge_main
    print("🔍 Scanning for .safetensors.part-* files...")
    merge_main()
except Exception as e:
    print(f"⚠️ merge.py auto-merge skipped or failed: {e}")

# -------------------------
# Model Setup (Lazy Loading)
# -------------------------
AURASR_LOCAL = os.environ.get("AURASR_LOCAL_PATH")
if not AURASR_LOCAL or not os.path.exists(AURASR_LOCAL):
    raise RuntimeError("❌ AuraSR local model not found. Ensure it is baked or downloaded to AURASR_LOCAL_PATH.")

_model_cache = {
    "aura_sr": None
}

def get_aura_sr() -> AuraSR:
    """
    Lazily load AuraSR-v2 model once, cache it globally for reuse.
    """
    global _model_cache
    if _model_cache["aura_sr"] is not None:
        return _model_cache["aura_sr"]

    print("🧠 Loading AuraSR-v2 model...")
    aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")
    _model_cache["aura_sr"] = aura_sr
    print("✅ AuraSR-v2 model loaded and cached.")
    return aura_sr


def unload_aura_sr():
    """
    Optional cleanup: unload AuraSR and free VRAM/CPU cache.
    """
    global _model_cache
    _model_cache["aura_sr"] = None
    torch.cuda.empty_cache()
    print("🧹 AuraSR model cache cleared and GPU memory freed.")


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
    Upscale an image by 4x using AuraSR.
    Supports 2x via resizing down from 4x.
    """
    if input_image is None:
        raise ValueError("No input image provided for upscaling.")

    aura_sr = get_aura_sr()
    pil_image = Image.fromarray(input_image)

    print(f"⚙️ Running AuraSR upscale ({scale}x)...")
    upscaled_image = aura_sr.upscale_4x(pil_image)

    if scale == 2:
        w, h = upscaled_image.size
        upscaled_image = upscaled_image.resize((w // 2, h // 2), Image.LANCZOS)
        print("🔁 Downscaled from 4x to 2x for final output.")

    return np.array(upscaled_image)


# -------------------------
# Inference wrapper
# -------------------------
def run_inference_aurasr_upscale(job_id: str, params: Dict[str, Any]) -> str:
    """
    Run AuraSR-v2 inference for 2x or 4x upscaling.
    Returns path to the upscaled image.
    """
    print(f"🧩 Starting AuraSR upscale job: {job_id}")
    init_image = _load_image(params.get("init_image"))
    if init_image is None:
        raise ValueError("init_image is required for upscaling")

    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"aurasr_upscaled_{job_id}.png")

    scale = int(params.get("scale", 4))
    if scale not in [2, 4]:
        raise ValueError("Scale must be 2 or 4 for AuraSR upscaling.")

    result = upscale_image(np.array(init_image), scale=scale)
    Image.fromarray(result).save(out_path)

    print(f"✅ Upscaled image saved to {out_path}")
    return os.path.abspath(out_path)


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    job_id = "upscale_debug"
    url = "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/Snapinsta.app_262322643_4937545659591345_7487236998017046274_n_1080.jpg"

    with httpx.Client(timeout=60) as c:
        img_bytes = c.get(url).content
    img_b64 = base64.b64encode(img_bytes).decode()


    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    params = {
        "init_image": img_b64,
        "output_dir": "./outputs",
        "scale": 2,  # choose 2 or 4
    }

    out = run_inference_aurasr_upscale(job_id, params=params)
    print("✅ Upscaled image saved at:", out)
