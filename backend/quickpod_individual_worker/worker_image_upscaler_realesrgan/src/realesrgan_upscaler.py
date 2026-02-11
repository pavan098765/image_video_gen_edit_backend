# upscale_realesrgan_local.py
# ✅ Lazy-loading implementation with GPU memory optimizations

import gc
import sys
import os
import base64
from io import BytesIO
from typing import Dict, Any, Optional, Union
import numpy as np
import cv2
from PIL import Image
import torch

# Aggressive GPU cleanup at import to prevent OOM
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# --- Compatibility shim for basicsr + newer torchvision ---
try:
    from torchvision.transforms import functional as F
    sys.modules["torchvision.transforms.functional_tensor"] = F
except ImportError:
    pass
# -----------------------------------------------------------

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer


# -------------------------
# Config / paths
# -------------------------
MODEL_DIR = os.environ.get("REALESRGAN_LOCAL_PATH", "/app/models/realesrgan")
GFPGAN_MODEL_PATH = os.path.join(MODEL_DIR, "GFPGANv1.3.pth")
DEFAULT_OUTPUT_DIR = "./outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

MODEL_MAP = {
    "RealESRGAN_x4plus": os.path.join(MODEL_DIR, "RealESRGAN_x4plus.pth"),
    "RealESRNet_x4plus": os.path.join(MODEL_DIR, "RealESRNet_x4plus.pth"),
    "RealESRGAN_x4plus_anime_6B": os.path.join(MODEL_DIR, "RealESRGAN_x4plus_anime_6B.pth"),
    "RealESRGAN_x2plus": os.path.join(MODEL_DIR, "RealESRGAN_x2plus.pth"),
    "realesr-general-x4v3": os.path.join(MODEL_DIR, "realesr-general-x4v3.pth"),
    "realesr-general-wdn-x4v3": os.path.join(MODEL_DIR, "realesr-general-wdn-x4v3.pth"),
}

LOADED_MODELS: Dict[str, RealESRGANer] = {}

# -------------------------
# Helpers
# -------------------------
def _load_image(img: Optional[Union[str, Image.Image, bytes]]) -> Optional[Image.Image]:
    """Load image from PIL, bytes, path, or base64 string."""
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

def _strip_metadata(img: Image.Image) -> Image.Image:
    """Remove EXIF/ICC metadata safely."""
    with BytesIO() as buf:
        img.save(buf, format="PNG")
        buf.seek(0)
        return Image.open(buf).convert("RGB")

def _get_model(model_name: str, params: Dict[str, Any]) -> RealESRGANer:
    """Load and cache RealESRGAN model from local weights."""
    if model_name in LOADED_MODELS:
        return LOADED_MODELS[model_name]

    if model_name not in MODEL_MAP:
        raise ValueError(f"Unsupported model '{model_name}'. Available: {list(MODEL_MAP.keys())}")

    model_path = MODEL_MAP[model_name]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Missing model weights at {model_path}")

    # Pick correct architecture
    if model_name in ["RealESRGAN_x4plus", "RealESRNet_x4plus"]:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == "RealESRGAN_x4plus_anime_6B":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == "RealESRGAN_x2plus":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif model_name == "realesr-general-x4v3":
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3,
                                num_feat=64, num_conv=32,
                                upscale=4, act_type="prelu")
        netscale = 4
    else:
        raise ValueError(f"Model architecture not implemented: {model_name}")

    # Handle optional DNI blending
    dni_weight = None
    if model_name == "realesr-general-x4v3":
        denoise_strength = float(params.get("denoise_strength", 1.0))
        if denoise_strength != 1.0:
            wdn_model_path = MODEL_MAP["realesr-general-wdn-x4v3"]
            model_path = [model_path, wdn_model_path]
            dni_weight = [denoise_strength, 1 - denoise_strength]

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        dni_weight=dni_weight,
        tile=int(params.get("tile", 0)),
        tile_pad=int(params.get("tile_pad", 10)),
        pre_pad=int(params.get("pre_pad", 0)),
        half=not params.get("fp32", False),
        gpu_id=0 if torch.cuda.is_available() else None
    )

    LOADED_MODELS[model_name] = upsampler
    print(f"✅ RealESRGAN model '{model_name}' loaded and cached.")
    return upsampler


def unload_realesrgan_models():
    """
    Optional cleanup: unload RealESRGAN models and free VRAM/CPU cache.
    """
    global LOADED_MODELS
    LOADED_MODELS.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("🧹 RealESRGAN model cache cleared and GPU memory freed.")


# -------------------------
# Main Upscale Function
# -------------------------
def run_inference_realesrgan(job_id: str, params: Dict[str, Any]) -> str:
    """
    Run RealESRGAN (optionally with GFPGAN) and return output path.
    Supports all models listed in MODEL_MAP.
    """
    print(f"🧩 Starting RealESRGAN upscale job: {job_id}")
    
    # Pre-inference GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    
    input_image = _load_image(params.get("init_image"))
    if input_image is None:
        raise ValueError("❌ 'init_image' parameter required")

    # Cap output to 4K: resize input if upscaled result would exceed 4K
    outscale = float(params.get("outscale", 4))
    max_output_dim = 4096  # 4K cap
    orig_w, orig_h = input_image.size
    target_w = int(orig_w * outscale)
    target_h = int(orig_h * outscale)
    
    if max(target_w, target_h) > max_output_dim:
        # Calculate scale factor to cap at 4K
        scale_factor = max_output_dim / max(target_w, target_h)
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        print(f"⚠️ Input image ({orig_w}x{orig_h}) would produce {target_w}x{target_h} output (>4K). Resizing to {new_w}x{new_h} before upscaling.")
        input_image = input_image.resize((new_w, new_h), Image.LANCZOS)

    model_name = params.get("model_name", "RealESRGAN_x4plus")
    upsampler = _get_model(model_name, params)

    # Optional GFPGAN for face enhancement
    face_enhance = bool(params.get("face_enhance", False))
    face_enhancer = None
    if face_enhance:
        if not os.path.exists(GFPGAN_MODEL_PATH):
            raise FileNotFoundError("❌ GFPGAN weights missing in container.")
        face_enhancer = GFPGANer(
            model_path=GFPGAN_MODEL_PATH,
            upscale=params.get("outscale", 4),
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=upsampler,
        )

    output_dir = params.get("output_dir", DEFAULT_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    suffix = params.get("suffix", "out")
    out_path = os.path.join(output_dir, f"upscaled_{job_id}_{suffix}.png")

    try:
        # Convert input RGB → BGR for OpenCV / RealESRGAN
        np_input = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

        with torch.inference_mode():
            if face_enhancer:
                _, _, output = face_enhancer.enhance(
                    np_input, has_aligned=False, only_center_face=False, paste_back=True
                )
            else:
                output, _ = upsampler.enhance(np_input, outscale=outscale)

        # Convert BGR → RGB for correct color display
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output_rgb = np.clip(output_rgb, 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(output_rgb)
        clean_img = _strip_metadata(pil_img)  # ✅ Remove metadata before saving
        clean_img.save(out_path)

    except Exception as e:
        raise RuntimeError(f"❌ Inference failed (try smaller tile or fp32=True): {e}")

    # Post-inference GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

    print(f"✅ Upscaled image saved to {out_path}")
    return out_path


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    job_id = "realesrgan_local_test"
    input_path = "./input_image.jpg"
    if not os.path.exists(input_path):
        raise SystemExit("❌ Please provide input_image.jpg to test this script.")

    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    params = {
        "init_image": img_b64,
        "model_name": "RealESRGAN_x4plus",  # any from MODEL_MAP
        "outscale": 4,
        "face_enhance": False,
        "tile": 0,
        "tile_pad": 10,
        "fp32": False,
        "suffix": "test",
        "output_dir": "./outputs",
        "denoise_strength": 0.8,  # optional for general-x4v3
    }

    out = run_inference_realesrgan(job_id, params=params)
    print("✅ Upscale result saved at:", out)
