# rmbg_remover.py

import base64
import os
from typing import Dict, Any, Optional
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from upscaler import upscale_image

# -------------------------
# Config: model location
# -------------------------
RMBG_LOCAL_PATH = os.environ.get("RMBG_LOCAL_PATH", "/models/rmbg").strip()

# -------------------------
# Initialize RMBG model (GPU only) - using direct model loading like official HF space
# -------------------------
from briarmbg import BriaRMBG

_model_source = RMBG_LOCAL_PATH if RMBG_LOCAL_PATH and os.path.exists(RMBG_LOCAL_PATH) else "briaai/RMBG-1.4"

try:
    _rmbg_model = BriaRMBG.from_pretrained(_model_source)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _rmbg_model.to(_device)
    _rmbg_model.eval()
    print(f"✅ RMBG model initialized from: {_model_source} on {_device}")
except Exception as e:
    raise RuntimeError(
        f"❌ Failed to initialize RMBG model from {_model_source}. "
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


def _preprocess_image(image: Image.Image) -> tuple[torch.Tensor, tuple[int, int]]:
    """Preprocess image for RMBG model - matches official HF space implementation."""
    orig_size = image.size  # (w, h)
    
    # Resize to model input size
    model_input_size = (1024, 1024)
    image_resized = image.resize(model_input_size, Image.BILINEAR)
    
    # Convert to tensor
    im_np = np.array(image_resized)
    im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = torch.unsqueeze(im_tensor, 0)
    im_tensor = torch.divide(im_tensor, 255.0)
    im_tensor = normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    
    if torch.cuda.is_available():
        im_tensor = im_tensor.cuda()
    
    return im_tensor, orig_size


def _postprocess_mask(result: torch.Tensor, orig_size: tuple[int, int]) -> Image.Image:
    """Postprocess model output to mask - matches official HF space implementation."""
    w, h = orig_size
    
    # Get first output and interpolate to original size
    result = torch.squeeze(F.interpolate(result[0][0], size=(h, w), mode='bilinear'), 0)
    
    # Normalize mask values
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    
    # Convert to PIL mask
    result_array = (result * 255).cpu().data.numpy().astype(np.uint8)
    pil_mask = Image.fromarray(np.squeeze(result_array))
    
    return pil_mask


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
        # Preprocess
        im_tensor, orig_size = _preprocess_image(pil_image)
        
        # Inference
        with torch.no_grad():
            result = _rmbg_model(im_tensor)
        
        # Postprocess to get mask
        pil_mask = _postprocess_mask(result, orig_size)
        
        if return_mask:
            return np.array(pil_mask)
        else:
            # Apply mask as alpha channel to original image
            result_image = pil_image.copy()
            result_image.putalpha(pil_mask)
            return np.array(result_image)
            
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
