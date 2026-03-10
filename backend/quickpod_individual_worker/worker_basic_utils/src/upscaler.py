# upscale_realesrgan_local.py
import sys
import os
import base64
import requests
from io import BytesIO
from typing import Union
from PIL import Image
import numpy as np
import cv2
import torch

# --- Compatibility shim for basicsr + newer torchvision ---
try:
    from torchvision.transforms import functional as F
    sys.modules["torchvision.transforms.functional_tensor"] = F
except ImportError:
    pass
# -----------------------------------------------------------

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


# -------------------------
# Config / Model Paths
# -------------------------
MODEL_DIR: str = os.environ.get("REALESRGAN_LOCAL_PATH", "/app/models/realesrgan")

MODEL_MAP = {
    "RealESRGAN_x4plus": os.path.join(MODEL_DIR, "RealESRGAN_x4plus.pth"),
    "RealESRGAN_x2plus": os.path.join(MODEL_DIR, "RealESRGAN_x2plus.pth"),
}

MODEL_CACHE = {}

# -------------------------
# Helpers
# -------------------------
def _decode_image(img: Union[str, bytes, Image.Image]) -> np.ndarray:
    """Decode input into OpenCV BGR or BGRA uint8 array (preserves alpha if present)."""
    if isinstance(img, Image.Image):
        if img.mode == "RGBA":
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
        return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    if isinstance(img, bytes):
        arr = np.frombuffer(img, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if os.path.exists(str(img)):
        return cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
    try:
        decoded = base64.b64decode(img)
        arr = np.frombuffer(decoded, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        raise ValueError(f"Unable to decode image: {e}")


def _strip_metadata(pil_img: Image.Image) -> Image.Image:
    """
    Remove all EXIF, ICC, and ancillary metadata by re-saving to PNG in memory.
    Ensures output is clean and deterministic. Preserves image mode (RGB/RGBA).
    """
    original_mode = pil_img.mode
    with BytesIO() as buf:
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return Image.open(buf).convert(original_mode)


def _get_model(model_name: str, gpu_id: int = 0) -> RealESRGANer:
    """Load a RealESRGAN model from local weights (supports only 2x, 4x)."""
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    if model_name not in MODEL_MAP:
        raise ValueError(f"Unsupported model '{model_name}'. Available: {list(MODEL_MAP.keys())}")

    model_path = MODEL_MAP[model_name]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Missing model weights at {model_path}")

    # Select model architecture and scale
    if model_name == "RealESRGAN_x4plus":
        scale = 4
        model = RRDBNet(num_in_ch=3, num_out_ch=3,
                        num_feat=64, num_block=23,
                        num_grow_ch=32, scale=scale)
    elif model_name == "RealESRGAN_x2plus":
        scale = 2
        model = RRDBNet(num_in_ch=3, num_out_ch=3,
                        num_feat=64, num_block=23,
                        num_grow_ch=32, scale=scale)
    else:
        raise ValueError(f"Model not supported: {model_name}")

    # Preemptively check model file structure to avoid KeyError
    # RealESRGANer expects 'params' or 'params_ema' keys, but some model files
    # have the weights directly without these wrapper keys
    print(f"🔍 Checking model file structure: {model_path}")
    loadnet = torch.load(model_path, map_location='cpu')
    
    # Check if model file has the expected structure
    needs_reformat = 'params' not in loadnet and 'params_ema' not in loadnet
    
    if needs_reformat:
        print(f"⚠️ Model file missing 'params'/'params_ema' keys, reformatting preemptively...")
        
        # Model file has weights directly without wrapper keys - use it as-is
        state_dict = loadnet
        
        # Create a temporary model file with the correct structure for RealESRGANer
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            tmp_path = tmp_file.name
            torch.save({'params': state_dict}, tmp_path)
        
        # Use the reformatted temp file
        final_model_path = tmp_path
        cleanup_temp = True
    else:
        # Model file is already properly formatted
        final_model_path = model_path
        cleanup_temp = False
    
    try:
        # Load with RealESRGANer (will work since we've ensured proper structure)
        upsampler = RealESRGANer(
            scale=scale,
            model_path=final_model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),
            gpu_id=gpu_id if torch.cuda.is_available() else None
        )
    except Exception as e:
        # For any error, re-raise it with more context
        raise RuntimeError(f"Failed to load RealESRGAN model '{model_name}' from {model_path}: {e}")
    finally:
        # Clean up temporary file if we created one
        if cleanup_temp:
            try:
                os.unlink(final_model_path)
            except Exception:
                pass

    MODEL_CACHE[model_name] = upsampler
    return upsampler


# -------------------------
# Main Function
# -------------------------
def upscale_image(
    input_image: Union[str, bytes, Image.Image],
    model_name: str = "RealESRGAN_x4plus",
    scale_factor: float = 4.0,
    remove_metadata: bool = True,
) -> Image.Image:
    """
    Upscale an image using RealESRGAN_x2plus or RealESRGAN_x4plus.
    Preserves alpha channel (transparency) if present in the input.
    Args:
        input_image: path, bytes, base64, or PIL.Image
        model_name: 'RealESRGAN_x2plus' or 'RealESRGAN_x4plus'
        scale_factor: typically 2 or 4
        remove_metadata: remove EXIF/ICC metadata from output
    Returns:
        PIL.Image (RGB or RGBA if input had alpha)
    """
    img = _decode_image(input_image)
    has_alpha = len(img.shape) == 3 and img.shape[2] == 4

    if has_alpha:
        # Separate alpha channel before upscaling (RealESRGAN only handles BGR)
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        bgr = img

    upsampler = _get_model(model_name)

    try:
        output_bgr, _ = upsampler.enhance(bgr, outscale=scale_factor)
    except RuntimeError as e:
        raise RuntimeError(f"❌ RealESRGAN inference failed: {e}")

    # Convert upscaled BGR to RGB
    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(output_rgb)

    if has_alpha:
        # Resize alpha to match upscaled dimensions and reattach
        h, w = output_bgr.shape[:2]
        alpha_resized = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LANCZOS4)
        pil_img.putalpha(Image.fromarray(alpha_resized))

    if remove_metadata:
        pil_img = _strip_metadata(pil_img)

    return pil_img


# -------------------------
# Example CLI Usage
# -------------------------
def fetch_image_bytes(url: str) -> bytes:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


if __name__ == "__main__":
    url = "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/Snapinsta.app_262322643_4937545659591345_7487236998017046274_n_1080.jpg"
    out_path = "out_upscaled_x2_clean.png"

    out_img = upscale_image(
        input_image=fetch_image_bytes(url),
        model_name="RealESRGAN_x2plus",  # or RealESRGAN_x4plus
        scale_factor=2.0,
        remove_metadata=True
    )

    out_img.save(out_path)
    print(f"✅ Saved upscaled, metadata-free image to {out_path}")
