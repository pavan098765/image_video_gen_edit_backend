# ------------------------------------------------------------------
# Face Enhancer Inference (GFPGAN + RealESRGAN)
# - Extracted from huggingface_app.py
# - Supports all GFPGAN versions (v1.2, v1.3, v1.4, RestoreFormer, CodeFormer, RealESR-General-x4v3)
# - Clean inference API without Gradio/UI code
# ------------------------------------------------------------------

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
# NOTE: Don't call ipc_collect() at import time - it initializes CUDA too early
# and may fail if CUDA isn't ready or if PyTorch version doesn't support certain options
gc.collect()
try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Don't call ipc_collect() at import - it can cause initialization errors
        # It will be called later when CUDA is actually in use
except Exception:
    pass  # CUDA may not be ready at import time

# --- Auto-merge split safetensors if merge.py exists ---
try:
    import time
    from merge import main as merge_main
    print("🔍 Scanning for .safetensors.part-* files...")
    merge_start = time.time()
    merge_main()
    merge_time = time.time() - merge_start
    print(f"✅ Merge completed in {merge_time:.1f}s")
except Exception as e:
    print(f"⚠️ merge.py auto-merge skipped or failed: {e}")

# --- Compatibility shim for basicsr + newer torchvision ---
try:
    from torchvision.transforms import functional as F
    sys.modules["torchvision.transforms.functional_tensor"] = F
except ImportError:
    pass

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer

# -------------------------
# Config / paths
# -------------------------
MODEL_DIR = os.environ.get("REALESRGAN_LOCAL_PATH", "/app/models/realesrgan")
DEFAULT_OUTPUT_DIR = "/app/worker/outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# GFPGAN face model paths (must match prep_model.py filenames)
# Frontend sends model="face_enhancer" and mode=<one of FACE_ENHANCER_MODES>
GFPGAN_MODELS = {
    "v1.2": os.path.join(MODEL_DIR, "GFPGANv1.2.pth"),
    "v1.3": os.path.join(MODEL_DIR, "GFPGANv1.3.pth"),
    "v1.4": os.path.join(MODEL_DIR, "GFPGANv1.4.pth"),
    "RestoreFormer": os.path.join(MODEL_DIR, "RestoreFormer.pth"),
    "CodeFormer": os.path.join(MODEL_DIR, "CodeFormer.pth"),
    "RealESR-General-x4v3": os.path.join(MODEL_DIR, "realesr-general-x4v3.pth"),
}
FACE_ENHANCER_MODES = list(GFPGAN_MODELS.keys())

# RealESRGAN background upsampler models (used for upscaling after face enhancement)
REALESRGAN_BG_MODELS = {
    "realesr-general-x4v3": {
        "model_path": os.path.join(MODEL_DIR, "realesr-general-x4v3.pth"),
        "scale": 4,
        "arch": "srvgg",
    },
    "RealESRGAN_x2plus": {
        "model_path": os.path.join(MODEL_DIR, "RealESRGAN_x2plus.pth"),
        "scale": 2,
        "arch": "rrdb",
    },
    "RealESRGAN_x4plus_anime_6B": {
        "model_path": os.path.join(MODEL_DIR, "RealESRGAN_x4plus_anime_6B.pth"),
        "scale": 4,
        "arch": "rrdb_anime",
    },
}

# Default background upsampler
DEFAULT_BG_MODEL = "realesr-general-x4v3"
REALESRGAN_BG_MODES = list(REALESRGAN_BG_MODELS.keys())

# Cache for loaded models (keyed by model name)
LOADED_BG_UPSAMPLERS: Dict[str, RealESRGANer] = {}

# -------------------------
# Helpers
# -------------------------
def _load_image(img: Optional[Union[str, Image.Image, bytes]]) -> Optional[Image.Image]:
    """Load image from PIL, bytes, path, URL, or base64 string."""
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, bytes):
        return Image.open(BytesIO(img)).convert("RGB")
    s = str(img).strip()
    if os.path.exists(s):
        return Image.open(s).convert("RGB")
    # URL: download then open (server often sends presigned R2 URL instead of base64)
    if s.startswith("http://") or s.startswith("https://"):
        try:
            import requests
            resp = requests.get(s, timeout=120)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Unable to download image from URL: {e}")
    # Base64: fix padding then decode (avoids "Incorrect padding" when padding was stripped)
    try:
        raw = s.encode("utf-8")
        pad = 4 - (len(raw) % 4)
        if pad and pad != 4:
            raw += b"=" * pad
        decoded = base64.b64decode(raw)
        return Image.open(BytesIO(decoded)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Unable to decode image from input: {e}")

def _strip_metadata(img: Image.Image) -> Image.Image:
    """Remove EXIF/ICC metadata safely."""
    with BytesIO() as buf:
        img.save(buf, format="PNG")
        buf.seek(0)
        return Image.open(buf).convert("RGB")

def _get_background_upsampler(model_name: str = DEFAULT_BG_MODEL) -> RealESRGANer:
    """Load and cache RealESRGAN background upsampler.
    
    Args:
        model_name: One of "realesr-general-x4v3", "RealESRGAN_x2plus", "RealESRGAN_x4plus_anime_6B"
    
    Returns:
        RealESRGANer instance
    """
    global LOADED_BG_UPSAMPLERS
    
    if model_name in LOADED_BG_UPSAMPLERS:
        print(f"  ℹ️ Using cached background upsampler: {model_name}")
        return LOADED_BG_UPSAMPLERS[model_name]
    
    if model_name not in REALESRGAN_BG_MODELS:
        raise ValueError(
            f"Unsupported background upsampler '{model_name}'. "
            f"Available: {list(REALESRGAN_BG_MODELS.keys())}"
        )
    
    model_info = REALESRGAN_BG_MODELS[model_name]
    model_path = model_info["model_path"]
    scale = model_info["scale"]
    arch = model_info["arch"]
    
    print(f"  📦 Loading RealESRGAN background upsampler ({model_name}) from disk...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Missing background upsampler weights at {model_path}")
    
    # Create appropriate model architecture
    print(f"  📦 Creating model architecture ({arch})...")
    if arch == "srvgg":
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32,
            upscale=scale, act_type="prelu"
        )
    elif arch == "rrdb":
        if model_name == "RealESRGAN_x2plus":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=2)
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=4)
    elif arch == "rrdb_anime":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                       num_block=6, num_grow_ch=32, scale=4)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    print(f"  📦 Initializing RealESRGANer (this may take a moment)...")
    half = True if torch.cuda.is_available() else False
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=half,
        gpu_id=0 if torch.cuda.is_available() else None
    )
    
    LOADED_BG_UPSAMPLERS[model_name] = upsampler
    print(f"✅ Background upsampler ({model_name}) loaded and cached.")
    return upsampler

# -------------------------
# Main Face Enhancement Function
# -------------------------
def run_inference_face_enhancer(job_id: str, params: Dict[str, Any]) -> str:
    """
    Run GFPGAN face enhancement with RealESRGAN background upsampling,
    or general upscale only (RealESRGAN without GFPGAN) when face_enhance=False.
    
    Args:
        job_id: unique job identifier (for output filenames)
        params:
            - init_image (required): input image (base64, URL, or path)
            - face_enhance (bool, default: True): if False, run only bg_model (general upscale, no GFPGAN)
            - mode (str): GFPGAN model mode - overrides version if set. One of FACE_ENHANCER_MODES.
            - version (str): legacy; used if mode not set. GFPGAN version - same values as mode.
            - scale (float, default: 2.0): rescaling factor (final output scale)
            - bg_model (str): background upsampler - one of REALESRGAN_BG_MODES
            - output_dir (str): output directory
    
    Returns:
        absolute path to final image
    """
    print(f"🧩 Starting face enhancement job: {job_id}")
    
    # Pre-inference GPU cleanup
    print(f"  [1/7] Cleaning GPU memory...")
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except (RuntimeError, AttributeError) as e:
                if "expandable_segments" in str(e) or "CachingAllocator" in str(e):
                    pass
                else:
                    raise
    except Exception:
        pass
    
    print(f"  [2/7] Loading input image...")
    # Support both init_image and image (upscale capability uses input id "image")
    input_image = _load_image(params.get("init_image") or params.get("image"))
    if input_image is None:
        raise ValueError("❌ 'init_image' or 'image' parameter required")
    print(f"  ✅ Input image loaded: {input_image.size}")
    
    # face_enhance=False → general upscale only (RealESRGAN), no GFPGAN
    face_enhance = params.get("face_enhance", True)
    if isinstance(face_enhance, str):
        face_enhance = face_enhance.lower() in ("true", "1", "yes")
    
    # Get scale (rescaling factor, default: 2.0)
    scale = float(params.get("scale", 2.0))
    
    # Get background upsampler model name (default: realesr-general-x4v3)
    bg_model_name = params.get("bg_model", DEFAULT_BG_MODEL)
    print(f"  [3/7] Loading background upsampler (RealESRGAN: {bg_model_name})...")
    bg_upsampler = _get_background_upsampler(bg_model_name)
    print(f"  ✅ Background upsampler loaded")
    
    if not face_enhance:
        # General upscale only: run RealESRGAN directly on full image (no GFPGAN)
        print(f"  [4/7] General upscale mode (no face enhancement)...")
        img_array = np.array(input_image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[0:2]
        with torch.inference_mode():
            output, _ = bg_upsampler.enhance(img_bgr, outscale=scale)
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output_rgb = np.clip(output_rgb, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(output_rgb)
        clean_img = _strip_metadata(pil_img)
        output_dir = params.get("output_dir", DEFAULT_OUTPUT_DIR)
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"face_enhanced_{job_id}.png")
        clean_img.save(out_path, format="PNG")
        print(f"✅ General upscale saved to {out_path}")
        return os.path.abspath(out_path)
    
    # Face enhancement path: GFPGAN + bg_upsampler
    version = params.get("mode") or params.get("version", "v1.4")
    print(f"  [4/7] Validating GFPGAN mode/version: {version}")
    if version not in GFPGAN_MODELS:
        raise ValueError(
            f"Unsupported GFPGAN version '{version}'. "
            f"Available: {list(GFPGAN_MODELS.keys())}"
        )
    
    gfpgan_model_path = GFPGAN_MODELS[version]
    if not os.path.exists(gfpgan_model_path):
        raise FileNotFoundError(f"❌ Missing GFPGAN model weights at {gfpgan_model_path}")
    print(f"  ✅ Model path verified: {gfpgan_model_path}")
    
    # Determine GFPGAN architecture based on version
    print(f"  [5/7] Determining architecture for version: {version}")
    if version == "RestoreFormer":
        arch = "RestoreFormer"
    elif version == "CodeFormer":
        arch = "CodeFormer"
    elif version == "RealESR-General-x4v3":
        arch = "realesr-general"
    else:
        arch = "clean"
    print(f"  ✅ Architecture: {arch}")
    
    # Create GFPGAN face enhancer
    print(f"  [6/7] Loading GFPGAN model ({version})...")
    face_enhancer = GFPGANer(
        model_path=gfpgan_model_path,
        upscale=2,  # GFPGAN internal upscale (always 2x)
        arch=arch,
        channel_multiplier=2,
        bg_upsampler=bg_upsampler,
    )
    print(f"  ✅ GFPGAN model loaded")
    
    output_dir = params.get("output_dir", DEFAULT_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"face_enhanced_{job_id}.png")
    
    print(f"  [7/7] Running face enhancement inference...")
    try:
        # Convert PIL Image to numpy array (RGB)
        img_array = np.array(input_image)
        
        # Convert RGB → BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Handle image modes (RGBA, grayscale)
        img_mode = None
        if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4:
            img_mode = "RGBA"
        elif len(img_bgr.shape) == 2:
            img_mode = "GRAY"
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        
        # Resize if image is too small (< 300px height)
        h, w = img_bgr.shape[0:2]
        if h < 300:
            print(f"⚠️ Image height ({h}px) < 300px, resizing to {w*2}x{h*2}")
            img_bgr = cv2.resize(img_bgr, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        
        # Run face enhancement
        with torch.inference_mode():
            _, _, output = face_enhancer.enhance(
                img_bgr,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )
        
        # Apply final rescaling if scale != 2
        if scale != 2.0:
            h, w = img_bgr.shape[0:2]
            interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
            output = cv2.resize(
                output,
                (int(w * scale / 2), int(h * scale / 2)),
                interpolation=interpolation
            )
        
        # Convert BGR → RGB for correct color display
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output_rgb = np.clip(output_rgb, 0, 255).astype(np.uint8)
        
        # Convert to PIL and save
        pil_img = Image.fromarray(output_rgb)
        clean_img = _strip_metadata(pil_img)
        clean_img.save(out_path, format="PNG")
        
    except Exception as e:
        raise RuntimeError(f"❌ Face enhancement failed: {e}")
    finally:
        del face_enhancer
    
    # Post-inference GPU cleanup
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # ipc_collect() may fail if PyTorch version doesn't support certain allocator options
            try:
                torch.cuda.ipc_collect()
            except (RuntimeError, AttributeError) as e:
                if "expandable_segments" in str(e) or "CachingAllocator" in str(e):
                    # PyTorch version doesn't support this option - skip it
                    pass
                else:
                    raise
    except Exception:
        pass  # CUDA may not be available
    
    print(f"✅ Face enhanced image saved to {out_path}")
    return os.path.abspath(out_path)


def unload_models():
    """Free cached RealESRGAN background upsamplers and reclaim GPU memory."""
    global LOADED_BG_UPSAMPLERS
    for name, upsampler in list(LOADED_BG_UPSAMPLERS.items()):
        try:
            if hasattr(upsampler, "model"):
                del upsampler.model
            del upsampler
        except Exception:
            pass
    LOADED_BG_UPSAMPLERS.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -------------------------
# CLI test mode
# -------------------------
if __name__ == "__main__":
    import httpx
    job_id = "face_enhancer_test"
    url = "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/Snapinsta.app_262322643_4937545659591345_7487236998017046274_n_1080.jpg"
    
    with httpx.Client(timeout=60) as c:
        img_bytes = c.get(url).content
    img_b64 = base64.b64encode(img_bytes).decode()
    
    params = {
        "init_image": img_b64,
        "version": "v1.4",
        "scale": 2.0,
        "output_dir": "/app/worker/outputs",
    }
    
    out = run_inference_face_enhancer(job_id, params)
    print("✅ Final output path:", out)
