# upscale_realesrgan_local.py
import base64
import os
os.environ["HF_HOME"] = "/models_storage/all_models"
os.environ["HF_HUB_CACHE"] = "/models_storage/all_models"
from io import BytesIO
from typing import Dict, Any, Optional, Union
from PIL import Image
import torch
from realesrgan import RealESRGAN

# -------------------------
# Config / paths
# -------------------------
MODEL_DIR = os.path.join("/models_storage", "all_models", "realesrgan")
os.makedirs(MODEL_DIR, exist_ok=True)

DEFAULT_OUTPUT_DIR = "./outputs"
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# Model mapping for scales -> local .pth files
MODEL_MAP = {
    "x4_general": os.environ.get("REALESRGAN_X4_GENERAL_PATH"),
    "x4_general_wdn": os.environ.get("REALESRGAN_X4_GENERAL_WDN_PATH"),
    "x4_plus": os.environ.get("REALESRGAN_X4_PLUS_PATH"),
}

# Cache loaded models to avoid reloading
LOADED_MODELS = {}

# -------------------------
# Helpers
# -------------------------
def _load_image(img: Optional[Union[str, Image.Image, bytes]]) -> Optional[Image.Image]:
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, bytes):
        return Image.open(BytesIO(img)).convert("RGB")
    if os.path.exists(str(img)):
        return Image.open(str(img)).convert("RGB")
    try:
        decoded = base64.b64decode(img)
        return Image.open(BytesIO(decoded)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Unable to decode image from input: {e}")

def _get_model(scale: str):
    """Load and cache the local RealESRGAN model"""
    scale = scale.lower()
    if scale not in MODEL_MAP:
        raise ValueError(f"Unsupported scale '{scale}'. Supported: {list(MODEL_MAP.keys())}")

    if scale in LOADED_MODELS:
        return LOADED_MODELS[scale]

    model_path = os.path.join(MODEL_DIR, MODEL_MAP[scale])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RealESRGAN(device, scale=int(scale[1]))
    model.load_weights(model_path, download=False)
    model.eval()

    LOADED_MODELS[scale] = (model, device)
    return model, device

# -------------------------
# Main Upscale function
# -------------------------
def run_inference_realesrgan(job_id: str, params: Dict[str, Any]) -> str:
    input_image = _load_image(params.get("image"))
    if input_image is None:
        raise ValueError("input_image is required for upscaling")

    scale = params.get("scale", "x4_general")
    model, device = _get_model(scale)

    output_dir = params.get("output_dir", DEFAULT_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"upscaled_{scale}_{job_id}.png")

    # Upscale
    with torch.no_grad():
        out_img = model.predict(input_image)

    out_img.save(out_path)
    return out_path

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    job_id = "realesrgan_local_test"
    input_path = "./input_image.jpg"

    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    params = {
        "image": img_b64,
        "scale": "x4_plus",  # x2, x4, x8
        "output_dir": "./outputs",
    }

    out = run_inference_realesrgan(job_id, params=params)
    print("✅ Upscale result saved at:", out)
