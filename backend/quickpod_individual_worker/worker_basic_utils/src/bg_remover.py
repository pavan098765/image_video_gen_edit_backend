# bg_remover.py

import base64
import os
from typing import Dict, Any, Optional
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from upscaler import upscale_image

# -------------------------
# Config: model location
# -------------------------
BIREFNET_LOCAL_PATH = os.environ.get("BIREFNET_LOCAL_PATH", "/app/models/basic_utils/birefnet").strip()

# Set float32 matmul precision for better performance
torch.set_float32_matmul_precision(["high", "highest"][0])

# -------------------------
# Initialize BiRefNet model (GPU only)
# -------------------------
_model_source = BIREFNET_LOCAL_PATH if BIREFNET_LOCAL_PATH and os.path.exists(BIREFNET_LOCAL_PATH) else "ZhengPeng7/BiRefNet"

try:
    _birefnet_model = AutoModelForImageSegmentation.from_pretrained(
        _model_source, trust_remote_code=True
    )
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _birefnet_model.to(_device)
    _birefnet_model.eval()
    print(f"✅ BiRefNet model initialized from: {_model_source} on {_device}")
except Exception as e:
    raise RuntimeError(
        f"❌ Failed to initialize BiRefNet model from {_model_source}. "
        f"Ensure the model is baked into the container. Error: {e}"
    )

# -------------------------
# Image transforms for BiRefNet
# -------------------------
_transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
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


def _process_image(image: Image.Image) -> Image.Image:
    """
    Apply BiRefNet-based image segmentation to remove the background.
    
    This function preprocesses the input image, runs it through a BiRefNet segmentation model to obtain a mask,
    and applies the mask as an alpha (transparency) channel to the original image.
    
    Args:
        image (PIL.Image): The input RGB image.
    
    Returns:
        PIL.Image: The image with the background removed, using the segmentation mask as transparency.
    """
    image_size = image.size
    input_images = _transform_image(image).unsqueeze(0).to(_device)
    
    # Prediction
    with torch.no_grad():
        preds = _birefnet_model(input_images)[-1].sigmoid().cpu()
    
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    
    # Create a copy to avoid modifying the original image
    result_image = image.copy()
    result_image.putalpha(mask)
    return result_image


# -------------------------
# Core background remover
# -------------------------
def remove_background(input_image: np.ndarray, return_mask: bool = False) -> np.ndarray:
    """
    Remove background from input image using BiRefNet.
    return_mask=True -> segmentation mask (white foreground, black background)
    return_mask=False -> image with transparent background (RGBA)
    """
    if input_image is None:
        raise ValueError("No input image provided for background removal.")

    pil_image = Image.fromarray(input_image).convert("RGB")

    try:
        if return_mask:
            # Get mask only
            image_size = pil_image.size
            input_images = _transform_image(pil_image).unsqueeze(0).to(_device)
            
            with torch.no_grad():
                preds = _birefnet_model(input_images)[-1].sigmoid().cpu()
            
            pred = preds[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred)
            mask = pred_pil.resize(image_size)
            return np.array(mask.convert("L"))
        else:
            # Get image with transparent background
            processed_image = _process_image(pil_image)
            return np.array(processed_image)
            
    except Exception as e:
        raise RuntimeError(f"BiRefNet inference failed: {e}")

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
    out_path = os.path.join(output_dir, f"birefnet_{suffix}_{job_id}.png")

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
