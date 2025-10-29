# upscale_swin_dynamic.py

import base64
import os
from io import BytesIO
from typing import Dict, Any, Optional, Union
from PIL import Image
import torch
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

# -------------------------
# Config / paths
# -------------------------
MODEL_DIR = "/models/swin2sr"
os.makedirs(MODEL_DIR, exist_ok=True)

DEFAULT_OUTPUT_DIR = "./outputs"
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# -------------------------
# Model mapping for scales
# -------------------------
MODEL_MAP = {
    "swin2sr_x2": os.environ.get("SWIN2SR_X2_LOCAL_PATH"),
    "swin2sr_x4_real": os.environ.get("SWIN2SR_X4_LOCAL_PATH"),
    "swin2sr_x4": os.environ.get("SWIN2SR_X4_CLASSIC_LOCAL_PATH"),
}

# Cache loaded models
LOADED_MODELS = {}

# -------------------------
# Helpers
# -------------------------
def _load_image(img: Optional[Union[str, Image.Image, bytes]]) -> Optional[Image.Image]:
    """Load image from path, PIL.Image, bytes, or base64 string."""
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

def _get_model(scale: str):
    """Load and cache the Swin2SR model for the given scale."""
    if isinstance(scale, int) or str(scale).isdigit():
        scale = f"swin2sr_x{scale}"
    else:
        scale = str(scale).lower()
    if scale not in MODEL_MAP:
        raise ValueError(f"Unsupported scale '{scale}'. Supported: {list(MODEL_MAP.keys())}")

    if scale in LOADED_MODELS:
        return LOADED_MODELS[scale]

    model_path = MODEL_MAP[scale]
    if not model_path or not os.path.exists(model_path):
        raise RuntimeError(f"❌ Swin2SR model not found for '{scale}' at {model_path}")

    device = "cuda"
    processor = AutoImageProcessor.from_pretrained(model_path, cache_dir=MODEL_DIR)
    model = Swin2SRForImageSuperResolution.from_pretrained(model_path, cache_dir=MODEL_DIR).to(device)
    model.eval()

    LOADED_MODELS[scale] = (processor, model, device)
    return processor, model, device

# -------------------------
# Main Upscale function
# -------------------------
def run_inference_swin2sr(job_id: str, params: Dict[str, Any]) -> str:
    input_image = _load_image(params.get("init_image"))
    if input_image is None:
        raise ValueError("init_image is required for upscaling")

    scale = params.get("scale", "swin2sr_x2")
    processor, model, device = _get_model(scale)

    output_dir = params.get("output_dir", DEFAULT_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"upscaled_{scale}_{job_id}.png")

    # Preprocess
    inputs = processor(input_image, return_tensors="pt").to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Postprocess
    out_img = outputs.reconstruction.squeeze().float().cpu().clamp_(0, 1).numpy()
    out_img = (out_img * 255.0).round().astype("uint8")
    out_img = Image.fromarray(out_img.transpose(1, 2, 0))  # C,H,W -> H,W,C

    out_img.save(out_path)
    return out_path

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    job_id = "swin_upscale_test"

    input_path = "./input_image.jpg"
    if not os.path.exists(input_path):
        raise SystemExit("❌ Please provide input_image.jpg to test this script.")

    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    params = {
        "output_dir": "./outputs",
        "init_image": img_b64,
        "scale": "swin2sr_x4",  # choose: swin2sr_x2, swin2sr_x4, swin2sr_x4_classic
    }

    out = run_inference_swin2sr(job_id, params=params)
    print("✅ Upscale result saved at:", out)
