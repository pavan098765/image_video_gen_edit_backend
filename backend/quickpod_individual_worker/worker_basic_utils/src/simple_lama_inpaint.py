"""
Simple LaMa inpainting wrapper with outpainting-friendly mask preparation.
Optimized for GPU performance with FP16 and inference mode.
"""

import base64
import os
from io import BytesIO
from typing import Dict, Any, Optional, Union
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from upscaler import upscale_image

# ----------------------------
# Globals
# ----------------------------
DEVICE = "cuda"  # enforce GPU only
DEFAULT_OUTPUT_DIR = "./outputs_outpaint"
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# Local baked TorchScript path (set in container build or startup)
LAMA_LOCAL_PATH = os.environ.get("LAMA_LOCAL_PATH", "/app/models/basic_utils/lama/lama.pt").strip()

# Global model reference
_LAMA_MODEL = None

def get_lama_model():
    """
    Lazily load and cache the LaMa model with optimizations.
    Uses FP16 for faster inference on modern GPUs.
    """
    global _LAMA_MODEL
    if _LAMA_MODEL is not None:
        return _LAMA_MODEL
    
    if not os.path.exists(LAMA_LOCAL_PATH):
        raise FileNotFoundError(f"❌ LaMa model file not found at {LAMA_LOCAL_PATH}")
    
    print(f"🧠 Loading LaMa model from: {LAMA_LOCAL_PATH}")
    
    # Load TorchScript model
    _LAMA_MODEL = torch.jit.load(LAMA_LOCAL_PATH, map_location=DEVICE)
    _LAMA_MODEL = _LAMA_MODEL.to(DEVICE)
    _LAMA_MODEL.eval()
    
    # Try to convert to half precision for faster inference
    try:
        _LAMA_MODEL = _LAMA_MODEL.half()
        print("✅ LaMa model loaded with FP16 optimization")
    except Exception as e:
        print(f"⚠️ FP16 not supported, using FP32: {e}")
    
    # Warm up the model with a dummy inference
    try:
        with torch.inference_mode():
            dummy_img = torch.zeros(1, 3, 512, 512, device=DEVICE, dtype=_LAMA_MODEL.parameters().__next__().dtype if hasattr(_LAMA_MODEL, 'parameters') else torch.float16)
            dummy_mask = torch.zeros(1, 1, 512, 512, device=DEVICE, dtype=dummy_img.dtype)
            _ = _LAMA_MODEL(dummy_img, dummy_mask)
        print("✅ LaMa model warmed up successfully")
    except Exception as e:
        print(f"⚠️ Warmup failed (non-critical): {e}")
    
    return _LAMA_MODEL


def _run_lama_inference(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Run optimized LaMa inference on a single image-mask pair.
    """
    model = get_lama_model()
    
    # Determine model dtype
    try:
        model_dtype = next(model.parameters()).dtype
    except (StopIteration, AttributeError):
        model_dtype = torch.float16
    
    orig_size = image.size  # (W, H)
    
    # Pad to multiple of 8 for model compatibility
    def pad_to_multiple(img, multiple=8):
        w, h = img.size
        new_w = ((w + multiple - 1) // multiple) * multiple
        new_h = ((h + multiple - 1) // multiple) * multiple
        if new_w == w and new_h == h:
            return img, (0, 0)
        padded = Image.new(img.mode, (new_w, new_h), 0 if img.mode == 'L' else (0, 0, 0))
        padded.paste(img, (0, 0))
        return padded, (new_w - w, new_h - h)
    
    image_padded, img_pad = pad_to_multiple(image.convert("RGB"))
    mask_padded, _ = pad_to_multiple(mask.convert("L"))
    
    # Convert to tensors
    img_np = np.array(image_padded).astype(np.float32) / 255.0
    mask_np = np.array(mask_padded).astype(np.float32) / 255.0
    
    # Ensure mask is binary (threshold at 0.5)
    mask_np = (mask_np > 0.5).astype(np.float32)
    
    # To tensor: (H, W, C) -> (1, C, H, W)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE, dtype=model_dtype)
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(DEVICE, dtype=model_dtype)
    
    # Run inference with optimizations
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=(model_dtype == torch.float16)):
            result = model(img_tensor, mask_tensor)
    
    # Convert back to PIL
    result = result.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(result)
    
    # Crop back to original size if padded
    if img_pad != (0, 0):
        result_img = result_img.crop((0, 0, orig_size[0], orig_size[1]))
    
    return result_img


# Initialize model at import time for faster first request
try:
    get_lama_model()
except Exception as e:
    print(f"⚠️ LaMa model lazy init deferred: {e}")

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
    """Run optimized LaMa inpainting given an image and mask."""

    out_path = os.path.join(output_dir, f"lama_inpaint_{job_id}.png")
    result = _run_lama_inference(init_image, mask_image)
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
