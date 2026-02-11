# upscale_swin_dynamic.py
# ✅ Lazy-loading implementation with GPU memory optimizations

import base64
import gc
import os
from io import BytesIO
from typing import Dict, Any, Optional, Union
from PIL import Image
import torch
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

# Aggressive GPU cleanup at import to prevent OOM
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

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
# GPU Memory Detection
# -------------------------
def get_gpu_memory() -> float:
    """Get GPU memory in GB."""
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024 ** 3)
        return 0.0
    except Exception:
        return 0.0

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

def _get_model(scale: str, use_cpu_offload: bool = False):
    """Load and cache the Swin2SR model for the given scale.
    
    Args:
        scale: Model scale identifier
        use_cpu_offload: If True, model stays on CPU and is moved to GPU only during inference
    """
    if isinstance(scale, int) or str(scale).isdigit():
        scale = f"swin2sr_x{scale}"
    else:
        scale = str(scale).lower()
    if scale not in MODEL_MAP:
        raise ValueError(f"Unsupported scale '{scale}'. Supported: {list(MODEL_MAP.keys())}")

    # Check cache with offload flag
    cache_key = f"{scale}_offload_{use_cpu_offload}"
    if cache_key in LOADED_MODELS:
        return LOADED_MODELS[cache_key]

    model_path = MODEL_MAP[scale]
    if not model_path or not os.path.exists(model_path):
        raise RuntimeError(f"❌ Swin2SR model not found for '{scale}' at {model_path}")

    processor = AutoImageProcessor.from_pretrained(model_path, cache_dir=MODEL_DIR)
    
    # Load model with appropriate device placement
    if use_cpu_offload:
        # Load on CPU, will move to GPU during inference
        device = "cpu"
        model = Swin2SRForImageSuperResolution.from_pretrained(model_path, cache_dir=MODEL_DIR)
        model.eval()
        print(f"✅ Swin2SR model '{scale}' loaded on CPU (will offload to GPU during inference).")
    else:
        # Load directly on GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Swin2SRForImageSuperResolution.from_pretrained(model_path, cache_dir=MODEL_DIR).to(device)
        model.eval()
        print(f"✅ Swin2SR model '{scale}' loaded and cached on {device}.")
    
    # Try to use half precision for memory efficiency (if GPU available)
    if device == "cuda" and torch.cuda.is_available():
        try:
            model = model.half()
            print(f"✅ Model converted to half precision (fp16) for memory efficiency.")
        except Exception as e:
            print(f"⚠️ Could not convert to half precision: {e}")

    LOADED_MODELS[cache_key] = (processor, model, device, use_cpu_offload)
    return processor, model, device, use_cpu_offload


def unload_swin2sr_models():
    """
    Optional cleanup: unload Swin2SR models and free VRAM/CPU cache.
    """
    global LOADED_MODELS
    LOADED_MODELS.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("🧹 Swin2SR model cache cleared and GPU memory freed.")

# -------------------------
# Main Upscale function
# -------------------------
def run_inference_swin2sr(job_id: str, params: Dict[str, Any]) -> str:
    print(f"🧩 Starting Swin2SR upscale job: {job_id}")
    
    # Pre-inference GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    
    input_image = _load_image(params.get("init_image"))
    if input_image is None:
        raise ValueError("init_image is required for upscaling")

    scale = params.get("scale", "swin2sr_x2")
    is_4x_model = "x4" in str(scale).lower()
    
    # VRAM-aware optimizations
    vram_gb = get_gpu_memory()
    print(f"🔍 Detected GPU VRAM: {vram_gb:.1f} GB")
    
    # Determine if we should use CPU offloading
    # For 4x models on lower VRAM cards, use CPU offloading to allow 4K output
    use_cpu_offload = False
    if is_4x_model and vram_gb > 0:
        if vram_gb < 24:
            use_cpu_offload = True
            print(f"🧠 Low VRAM ({vram_gb:.1f}GB) for 4x model → enabling CPU offloading to support 4K output")
        elif vram_gb <= 16:
            use_cpu_offload = True
            print(f"🧠 Very Low VRAM ({vram_gb:.1f}GB) for 4x model → enabling CPU offloading to support 4K output")
    
    # Cap output to 4K: resize input if upscaled result would exceed 4K
    # Determine scale factor from model name
    if is_4x_model:
        upscale_factor = 4
    else:
        upscale_factor = 2
    
    max_output_dim = 4096  # 4K cap
    orig_w, orig_h = input_image.size
    target_w = int(orig_w * upscale_factor)
    target_h = int(orig_h * upscale_factor)
    
    if max(target_w, target_h) > max_output_dim:
        # Calculate scale factor to cap at 4K
        scale_factor = max_output_dim / max(target_w, target_h)
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        print(f"⚠️ Input image ({orig_w}x{orig_h}) would produce {target_w}x{target_h} output (>4K). Resizing to {new_w}x{new_h} before upscaling.")
        input_image = input_image.resize((new_w, new_h), Image.LANCZOS)
    
    processor, model, device, use_cpu_offload_flag = _get_model(scale, use_cpu_offload)

    output_dir = params.get("output_dir", DEFAULT_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"upscaled_{scale}_{job_id}.png")

    # Preprocess
    inputs = processor(input_image, return_tensors="pt")
    
    # Move inputs to appropriate device and handle CPU offloading
    if use_cpu_offload_flag:
        # For CPU offload: move model to GPU temporarily for inference
        inference_device = "cuda" if torch.cuda.is_available() else "cpu"
        if inference_device == "cuda":
            # Temporarily move model to GPU
            model = model.to(inference_device)
            inputs = {k: v.to(inference_device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.inference_mode():
                outputs = model(**inputs)
            
            # Move model back to CPU to free GPU memory
            model = model.cpu()
            # Update cache with CPU model
            cache_key = f"{scale}_offload_{use_cpu_offload}"
            LOADED_MODELS[cache_key] = (processor, model, "cpu", use_cpu_offload)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        else:
            # CPU-only inference
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = model(**inputs)
    else:
        # Standard GPU inference
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)

    # Postprocess
    out_img = outputs.reconstruction.squeeze().float().cpu().clamp_(0, 1).numpy()
    out_img = (out_img * 255.0).round().astype("uint8")
    out_img = Image.fromarray(out_img.transpose(1, 2, 0))  # C,H,W -> H,W,C

    out_img.save(out_path)

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
