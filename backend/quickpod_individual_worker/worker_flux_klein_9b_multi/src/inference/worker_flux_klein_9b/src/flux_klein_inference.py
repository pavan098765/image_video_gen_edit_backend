# flux_klein_inference.py
# ✅ Lazy Loading for model and pipeline
# ✅ GPU memory cleanup before/after inference (OOM prevention)
# ✅ SDNQ 4-bit quantization with INT8 MatMul support
# Dynamic resolution (keeps aspect ratio, multiple-of-64), seed, CFG, steps

import os
import base64
import gc
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import httpx
import torch
import diffusers
from sdnq import SDNQConfig  # import sdnq to register it into diffusers and transformers
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model

from upscaler import upscale_image

# --- Auto-merge split safetensors if merge.py exists ---
try:
    from merge import main as merge_main
    print("🔍 Scanning for .safetensors.part-* files...")
    merge_main()
except Exception as e:
    print(f"⚠️ merge.py auto-merge skipped or failed: {e}")

# -------------------------
# Paths / Model config
# -------------------------
MODEL_REPO = "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32"
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
MODEL_LOCAL_DIR = os.path.join(MODELS_DIR, "flux2_klein_9b_sdnq_4bit_svd_r32")

# -------------------------
# Lazy Pipeline Global
# -------------------------
_pipeline = None


def get_pipeline():
    """Lazy-initialize the Flux2Klein pipeline with SDNQ quantization."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    total_start = time.time()

    # CRITICAL: Clear GPU cache before loading to maximize free memory
    gc.collect()
    torch.cuda.empty_cache()

    print(f"🧠 Loading Flux2Klein SDNQ pipeline from {MODEL_REPO}...")
    print(f"   Local directory: {MODEL_LOCAL_DIR}")
    
    # Verify model directory exists
    if not os.path.exists(MODEL_LOCAL_DIR):
        raise FileNotFoundError(f"Model directory not found: {MODEL_LOCAL_DIR}")

    pipeline_start = time.time()
    pipe = diffusers.Flux2KleinPipeline.from_pretrained(
        MODEL_LOCAL_DIR,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    pipeline_time = time.time() - pipeline_start
    print(f"✅ Base pipeline loaded in {pipeline_time:.1f}s")

    # Enable INT8 MatMul for AMD, Intel ARC and Nvidia GPUs:
    if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
        print("⚡ Applying SDNQ INT8 MatMul optimizations...")
        try:
            pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
            print("   ✓ Transformer INT8 MatMul enabled")
        except Exception as e:
            print(f"   ⚠️  Failed to apply SDNQ to transformer: {e}")
        
        try:
            pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
            print("   ✓ Text encoder INT8 MatMul enabled")
        except Exception as e:
            print(f"   ⚠️  Failed to apply SDNQ to text encoder: {e}")
        
        # Optional: torch.compile for faster speeds (can be enabled if needed)
        # pipe.transformer = torch.compile(pipe.transformer)
    else:
        print("⚠️  Triton not available, skipping INT8 MatMul optimizations")

    # Apply memory management (CPU offload for better VRAM management)
    # print("⚡ Applying memory optimizations...")
    # try:
    #     pipe.enable_model_cpu_offload()
    #     print("✓ Model CPU offload enabled")
    # except Exception as e:
    #     print(f"   ⚠️  CPU offload failed: {e}")
    #     # Fallback: move to device directly
    #     if torch.cuda.is_available():
    #         pipe = pipe.to("cuda")
    #         print(f"   ✓ Model moved to cuda directly")
    #     else:
    #         pipe = pipe.to("cpu")
    #         print(f"   ✓ Model moved to cpu")
    pipe = pipe.to("cuda")
    _pipeline = pipe
    total_time = time.time() - total_start
    print(f"✅ Pipeline initialized in {total_time:.1f}s (base: {pipeline_time:.1f}s)")
    return _pipeline


def unload_models():
    """Frees up GPU memory by unloading models from cache."""
    global _pipeline
    _pipeline = None
    gc.collect()
    torch.cuda.empty_cache()
    print("🧹 Model cache cleared and GPU memory freed.")


# -------------------------
# Helpers
# -------------------------
def _load_image_from_base64(img_b64: str) -> Image.Image:
    """Load image from base64 string."""
    try:
        decoded_bytes = base64.b64decode(img_b64)
        return Image.open(BytesIO(decoded_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}")


def _load_image_from_init_string(s: str) -> Image.Image:
    """
    Load from presigned HTTP(S) URL (R2), data URL, or raw base64 — matches controller dispatch.
    """
    s = (s or "").strip()
    if not s:
        raise ValueError("empty init_image string")
    if s.startswith(("http://", "https://")):
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            resp = client.get(s)
            resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    if s.startswith("data:image"):
        comma = s.find(",")
        if comma < 0:
            raise ValueError("invalid data:image URL")
        b64 = s[comma + 1 :].strip()
        return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    return _load_image_from_base64(s)


def _normalize_flux_init_images(
    raw: Any,
) -> Optional[List[Image.Image]]:
    """Return list of RGB PIL images, or None for T2I."""
    if raw is None:
        return None
    if isinstance(raw, Image.Image):
        return [raw.convert("RGB")]
    if isinstance(raw, str):
        return [_load_image_from_init_string(raw)]
    if isinstance(raw, (list, tuple)):
        out: List[Image.Image] = []
        for item in raw:
            if item is None:
                continue
            if isinstance(item, str):
                out.append(_load_image_from_init_string(item))
            elif isinstance(item, Image.Image):
                out.append(item.convert("RGB"))
            else:
                raise ValueError(f"Unsupported init_image list element: {type(item)}")
        return out or None
    raise ValueError(f"Unsupported init_image type: {type(raw)}")


def _get_dynamic_size(img: Image.Image, max_dim: int = 1024) -> tuple[int, int]:
    """Return width/height close to input, within max_dim, multiple-of-64, preserving aspect ratio."""
    w, h = img.size
    aspect = w / h
    
    # First, scale down if needed
    if w > h:
        new_w = min(w, max_dim)
        new_h = int(round(new_w / aspect))
    else:
        new_h = min(h, max_dim)
        new_w = int(round(new_h * aspect))
    
    # Round to multiples of 64 while preserving aspect ratio
    if new_w >= new_h:
        new_w = max(64, (new_w // 64) * 64)
        new_h = int(round(new_w / aspect))
        new_h = max(64, (new_h // 64) * 64)
        new_w = int(round(new_h * aspect))
        new_w = max(64, (new_w // 64) * 64)
    else:
        new_h = max(64, (new_h // 64) * 64)
        new_w = int(round(new_h * aspect))
        new_w = max(64, (new_w // 64) * 64)
        new_h = int(round(new_w / aspect))
        new_h = max(64, (new_h // 64) * 64)
    
    return new_w, new_h


# -------------------------
# Main inference
# -------------------------
def run_inference_flux_klein(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run Flux2Klein inference with SDNQ quantization.
    Supports both T2I (text-to-image) and I2I (image-to-image/editing) modes.
    
    Args:
        job_id: unique job identifier
        prompt: text description for image generation/editing
        params:
            - output_dir (str)
            - init_image: single base64 / data URL / presigned URL, or list of those (multi-ref bulk)
            - width (int): output width for T2I mode (default: 1024)
            - height (int): output height for T2I mode (default: 1024)
            - num_inference_steps (int): default 4
            - guidance_scale (float): default 1.0
            - seed (int): optional random seed
            - scale_image (bool): enable upscaling
            - scale_factor (int): upscale factor (2 or 4)
    
    Returns: absolute path to saved PNG file
    """
    print(f"🧩 Starting inference job: {job_id}")
    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    # I2I/editing: init_image from controller is URL(s) or base64 (single or list for multi-ref bulk)
    init_list = _normalize_flux_init_images(params.get("init_image"))
    is_i2i_mode = bool(init_list)

    if is_i2i_mode:
        assert init_list is not None
        print(
            f"📸 I2I/Editing mode: {len(init_list)} reference image(s)"
        )
        width, height = _get_dynamic_size(init_list[0], max_dim=1024)
        resized: List[Image.Image] = []
        for img in init_list:
            if img.size == (width, height):
                resized.append(img)
            else:
                resized.append(img.resize((width, height), Image.LANCZOS))
        pipeline_image: Union[Image.Image, List[Image.Image]] = (
            resized[0] if len(resized) == 1 else resized
        )
        print(f"   Primary input size: {init_list[0].size[0]}x{init_list[0].size[1]}")
        print(f"   Output size: {width}x{height} (adjusted to multiple of 64)")
    else:
        # T2I mode: use provided dimensions or defaults
        print("📝 T2I mode: text-to-image generation")
        width = int(params.get("width", 1024))
        height = int(params.get("height", 1024))
        # Ensure dimensions are multiples of 64
        width = max(64, (width // 64) * 64)
        height = max(64, (height // 64) * 64)
        pipeline_image = None

    steps = int(params.get("num_inference_steps", 4))
    cfg = float(params.get("guidance_scale", 1.0))
    seed = params.get("seed", None)
    if seed is None:
        import random
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = int(seed)

    pipeline = get_pipeline()

    # CRITICAL: Clear GPU memory before inference to maximize available VRAM
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Log memory state before inference
    try:
        if torch.cuda.is_available():
            free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            print(f"🔍 GPU memory before inference: {free_mem / 1024**3:.2f} GB free")
    except Exception:
        pass

    mode_str = "I2I/Editing" if is_i2i_mode else "T2I"
    print(f"⚙️ Running {mode_str} pipeline at {width}x{height} | Steps={steps} | CFG={cfg} | Seed={seed}")

    try:
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
        
        with torch.inference_mode():
            # Build pipeline call arguments
            pipeline_kwargs = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "guidance_scale": cfg,
                "num_inference_steps": steps,
                "generator": generator
            }
            
            # Add image parameter for I2I/editing mode
            if is_i2i_mode:
                pipeline_kwargs["image"] = pipeline_image
            
            output = pipeline(**pipeline_kwargs).images[0]
    finally:
        # CRITICAL: Clean up GPU memory after inference
        gc.collect()
        torch.cuda.empty_cache()

    out_path = os.path.join(output_dir, f"flux_klein_{job_id}.png")
    output.save(out_path)
    print(f"🖼️ Saved generated image to {out_path}")

    # -------------------------
    # Optional Upscaling
    # -------------------------
    scale_image = params.get("scale_image", False)
    if scale_image:
        scale_factor = int(params.get("scale_factor", 2))
        model_name = params.get("scale_model_name", f"RealESRGAN_x{scale_factor}plus")
        
        try:
            upscaled = upscale_image(out_path, model_name=model_name, scale_factor=scale_factor)
            upscaled_path = os.path.join(output_dir, f"{job_id}_upscaled.png")
            upscaled.save(upscaled_path)
            print(f"🚀 Upscaled image saved to {upscaled_path}")
            return os.path.abspath(upscaled_path)
        except Exception as e:
            print(f"⚠️  Upscaling failed: {e}, returning original image")
            return os.path.abspath(out_path)

    return os.path.abspath(out_path)


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Example 1: T2I mode
    job_id = "flux_klein_test_t2i"
    prompt = "A cat holding a sign that says hello world"
    
    params_t2i = {
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "seed": 0,
        "output_dir": "./outputs",
        "scale_image": False,
    }

    out = run_inference_flux_klein(job_id, prompt, params_t2i)
    print("✅ T2I Output image saved at:", out)
    
    # Example 2: I2I/Editing mode (requires base64 image)
    # Uncomment and provide a base64 image string to test
    # job_id_i2i = "flux_klein_test_i2i"
    # prompt_i2i = "Change the background to a sunset"
    # img_url = "https://example.com/image.jpg"
    # with httpx.Client(timeout=60) as c:
    #     img_bytes = c.get(img_url).content
    # img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    # 
    # params_i2i = {
    #     "init_image": img_b64,
    #     "num_inference_steps": 4,
    #     "guidance_scale": 1.0,
    #     "seed": 42,
    #     "output_dir": "./outputs",
    #     "scale_image": False,
    # }
    # 
    # out_i2i = run_inference_flux_klein(job_id_i2i, prompt_i2i, params_i2i)
    # print("✅ I2I Output image saved at:", out_i2i)
