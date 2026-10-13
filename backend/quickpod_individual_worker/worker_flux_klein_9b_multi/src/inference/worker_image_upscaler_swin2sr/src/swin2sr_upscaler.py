# upscale_swin_dynamic.py
# ✅ Lazy-loading implementation with GPU memory optimizations

import base64
import gc
import os
from io import BytesIO
from typing import Dict, Any, Optional, Union
import numpy as np
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
# HF processor/model cache (not the same as SWIN2SR_*_LOCAL_PATH weight dirs).
MODEL_DIR = os.environ.get("SWIN2SR_HF_CACHE_DIR", "/app/models/swin2sr_hf_cache")
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
        if scale in ("4x", "2x"):
            scale = "swin2sr_x4" if scale == "4x" else "swin2sr_x2"
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
# Tiled inference constants (Swin2SR window_size=8)
# -------------------------
WINDOW_SIZE = 8
TILE_SIZE = 256  # multiple of 8
TILE_OVERLAP = 32
# Max input dimension above which we use tiling for 4x to avoid OOM
TILE_THRESHOLD_4X = 512


def _run_one_tile(
    tile_pil: Image.Image,
    processor,
    model,
    device: str,
    use_cpu_offload: bool,
    model_dtype: torch.dtype,
    upscale_factor: int,
    scale: str,
) -> np.ndarray:
    """Run Swin2SR on a single tile; returns (H, W, C) float [0,1] for tile content only (no padding).

    Caller must pass tiles whose width and height are multiples of WINDOW_SIZE (8) so the
    processor does not add padding and output alignment is 1:1. Output is cropped to exactly
    (tile_h * scale, tile_w * scale) from the top-left of the model output.
    """
    tw, th = tile_pil.size
    inputs = processor(tile_pil, return_tensors="pt")
    inference_device = "cuda" if torch.cuda.is_available() else "cpu"

    if use_cpu_offload and inference_device == "cuda":
        model = model.to(inference_device)
    inputs = {k: v.to(device=inference_device, dtype=model_dtype) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    out_tensor = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1)
    # Crop to exact tile content size (input_tile * scale); removes any processor padding
    content_h = int(min(th * upscale_factor, out_tensor.shape[1]))
    content_w = int(min(tw * upscale_factor, out_tensor.shape[2]))
    out_tensor = out_tensor[:, :content_h, :content_w]
    out_np = np.moveaxis(out_tensor.numpy(), 0, -1)  # C,H,W -> H,W,C

    if use_cpu_offload and inference_device == "cuda":
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
    return out_np


def _run_tiled(
    input_image: Image.Image,
    processor,
    model,
    device: str,
    use_cpu_offload: bool,
    model_dtype: torch.dtype,
    upscale_factor: int,
    scale: str,
) -> np.ndarray:
    """Upscale full image by tiles; stitches all tiles into one canvas; returns (H, W, C) float [0,1]."""
    orig_w, orig_h = input_image.size
    out_h = orig_h * upscale_factor
    out_w = orig_w * upscale_factor
    stride = TILE_SIZE - TILE_OVERLAP
    assert stride > 0 and TILE_SIZE % WINDOW_SIZE == 0, "TILE_SIZE must be multiple of 8"

    h_idx_list = (
        list(range(0, orig_h - TILE_SIZE, stride)) + [orig_h - TILE_SIZE]
        if orig_h > TILE_SIZE
        else [0]
    )
    w_idx_list = (
        list(range(0, orig_w - TILE_SIZE, stride)) + [orig_w - TILE_SIZE]
        if orig_w > TILE_SIZE
        else [0]
    )

    E = np.zeros((out_h, out_w, 3), dtype=np.float64)
    W = np.zeros((out_h, out_w), dtype=np.float64)
    num_tiles = len(h_idx_list) * len(w_idx_list)
    print(f"🧱 Tiled inference: {len(h_idx_list)}x{len(w_idx_list)} = {num_tiles} tiles (output {out_h}x{out_w})")

    tile_count = 0
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            # Option A: round tile size down to multiple of WINDOW_SIZE (8) so processor does not pad
            raw_tile_h = min(TILE_SIZE, orig_h - h_idx)
            raw_tile_w = min(TILE_SIZE, orig_w - w_idx)
            tile_h = max(WINDOW_SIZE, (raw_tile_h // WINDOW_SIZE) * WINDOW_SIZE)
            tile_w = max(WINDOW_SIZE, (raw_tile_w // WINDOW_SIZE) * WINDOW_SIZE)

            # Overlap-with-crop-back: extract with margin, then crop output to center region
            pad_top = max(0, h_idx - TILE_OVERLAP // 2)
            pad_bottom = min(orig_h, h_idx + tile_h + TILE_OVERLAP // 2)
            pad_left = max(0, w_idx - TILE_OVERLAP // 2)
            pad_right = min(orig_w, w_idx + tile_w + TILE_OVERLAP // 2)
            padded_w = pad_right - pad_left
            padded_h = pad_bottom - pad_top
            extract_w = (padded_w // WINDOW_SIZE) * WINDOW_SIZE
            extract_h = (padded_h // WINDOW_SIZE) * WINDOW_SIZE
            margin_w = w_idx - pad_left
            margin_h = h_idx - pad_top
            if extract_w < margin_w + tile_w or extract_h < margin_h + tile_h:
                extract_w = tile_w
                extract_h = tile_h
                pad_left = w_idx
                pad_top = h_idx
                margin_w = 0
                margin_h = 0
            tile_pil = input_image.crop((pad_left, pad_top, pad_left + extract_w, pad_top + extract_h))

            tile_out = _run_one_tile(
                tile_pil, processor, model, device, use_cpu_offload, model_dtype, upscale_factor, scale
            )
            mh_s = margin_h * upscale_factor
            mw_s = margin_w * upscale_factor
            th_s = tile_h * upscale_factor
            tw_s = tile_w * upscale_factor
            center = tile_out[mh_s : mh_s + th_s, mw_s : mw_s + tw_s]
            ah, aw = center.shape[0], center.shape[1]

            oh0 = int(h_idx * upscale_factor)
            ow0 = int(w_idx * upscale_factor)
            end_h = min(oh0 + ah, out_h)
            end_w = min(ow0 + aw, out_w)
            use_h, use_w = end_h - oh0, end_w - ow0
            E[oh0:end_h, ow0:end_w] += center[:use_h, :use_w]
            W[oh0:end_h, ow0:end_w] += 1.0

            if tile_count < 4:
                print(f"  Tile ({h_idx},{w_idx}) -> input ({tile_h}x{tile_w}) -> out ({ah}x{aw}) -> canvas [{oh0}:{end_h}, {ow0}:{end_w}]")
            tile_count += 1

    if np.any(W < 0.5):
        print(f"⚠️ Some output pixels had no tile coverage; filling with nearest.")
        W[W < 1e-6] = 1.0
    else:
        W[W < 1e-6] = 1.0
    out = (E / W[:, :, np.newaxis]).astype(np.float32)
    stitched = np.clip(out, 0.0, 1.0)
    print(f"  Final stitched shape: {stitched.shape}, expected: ({out_h}, {out_w}, 3)")
    assert stitched.shape == (out_h, out_w, 3), (
        f"Stitched shape {stitched.shape} != expected ({out_h}, {out_w}, 3)"
    )
    return stitched


# -------------------------
# Main Upscale function
# -------------------------
def run_inference_swin2sr(job_id: str, params: Dict[str, Any]) -> str:
    """
    Upscale image with Swin2SR. Full output resolution by default.
    Params: init_image, scale (swin2sr_x2 / x4 / x4_real), output_dir.
    Optional: use_tiles (bool) - force tiled inference; max_output_dim (int) - cap output long edge (e.g. 2048).
    """
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
    
    # For 4x models on lower VRAM cards, use CPU offloading when not using tiles
    use_cpu_offload = False
    if is_4x_model and vram_gb > 0:
        if vram_gb < 24:
            use_cpu_offload = True
            print(f"🧠 Low VRAM ({vram_gb:.1f}GB) for 4x model → enabling CPU offloading")
        elif vram_gb <= 16:
            use_cpu_offload = True
            print(f"🧠 Very Low VRAM ({vram_gb:.1f}GB) for 4x model → enabling CPU offloading")
    
    if is_4x_model:
        upscale_factor = 4
    else:
        upscale_factor = 2
    
    orig_w, orig_h = input_image.size
    target_w = int(orig_w * upscale_factor)
    target_h = int(orig_h * upscale_factor)
    
    # Use tiled inference for large 4x images to avoid OOM while keeping full resolution
    max_output_dim = params.get("max_output_dim")  # None = no cap
    use_tiles = params.get("use_tiles")
    if use_tiles is None:
        use_tiles = is_4x_model and (orig_h > TILE_THRESHOLD_4X or orig_w > TILE_THRESHOLD_4X)
    if use_tiles and (orig_h > TILE_SIZE or orig_w > TILE_SIZE):
        # Tiled path: no input resize; full output
        pass
    else:
        # Optional 2K cap only when not tiling (so we don't crop user content by resizing)
        if max_output_dim is not None and max(target_w, target_h) > max_output_dim:
            scale_factor = max_output_dim / max(target_w, target_h)
            new_w = int(orig_w * scale_factor)
            new_h = int(orig_h * scale_factor)
            print(f"⚠️ Capping: input ({orig_w}x{orig_h}) → ({new_w}x{new_h}) before upscaling (max_output_dim={max_output_dim})")
            input_image = input_image.resize((new_w, new_h), Image.LANCZOS)
    
    pre_upscale_w, pre_upscale_h = input_image.size
    processor, model, device, use_cpu_offload_flag = _get_model(scale, use_cpu_offload)

    output_dir = params.get("output_dir", DEFAULT_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"upscaled_{scale}_{job_id}.png")

    model_dtype = next(model.parameters()).dtype

    # Tiled path: run all tiles, stitch into one image, then save once (full output)
    if use_tiles and (pre_upscale_h > TILE_SIZE or pre_upscale_w > TILE_SIZE):
        out_np = _run_tiled(
            input_image, processor, model, device, use_cpu_offload_flag,
            model_dtype, upscale_factor, scale,
        )
        # Stitched array is full (pre_upscale_h*scale x pre_upscale_w*scale); save only after complete
        out_img = (out_np * 255.0).round().astype(np.uint8)
        out_img = Image.fromarray(out_img)
        out_img.save(out_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        print(f"✅ Upscaled image (tiled, full {out_img.width}x{out_img.height}) saved to {out_path}")
        return out_path

    # Whole-image path
    inputs = processor(input_image, return_tensors="pt")

    if use_cpu_offload_flag:
        inference_device = "cuda" if torch.cuda.is_available() else "cpu"
        if inference_device == "cuda":
            model = model.to(inference_device)
            inputs = {k: v.to(device=inference_device, dtype=model_dtype) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = model(**inputs)
            model = model.cpu()
            cache_key = f"{scale}_offload_{use_cpu_offload}"
            LOADED_MODELS[cache_key] = (processor, model, "cpu", use_cpu_offload)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        else:
            inputs = {k: v.to("cpu", dtype=model_dtype) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = model(**inputs)
    else:
        inputs = {k: v.to(device=device, dtype=model_dtype) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)

    out_tensor = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1)
    # Crop only processor padding: content is (pre_upscale_h*scale, pre_upscale_w*scale)
    crop_h = min(pre_upscale_h * upscale_factor, out_tensor.shape[1])
    crop_w = min(pre_upscale_w * upscale_factor, out_tensor.shape[2])
    out_tensor = out_tensor[:, :crop_h, :crop_w]
    out_img = np.moveaxis(out_tensor.numpy(), 0, -1)
    out_img = (out_img * 255.0).round().astype(np.uint8)
    out_img = Image.fromarray(out_img)
    out_img.save(out_path)

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
