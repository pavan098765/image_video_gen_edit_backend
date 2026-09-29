# ------------------------------------------------------------------
# SDXL Img2Img Inference with SDXL-Lightning LoRA
# - Lazy pipeline loading (prevents OOM during FastAPI startup)
# - SDXL-Lightning LoRA fused for fast 8-step inference
# - Dynamic resolution + optional upscaling
# - GPU memory cleanup before/after inference (OOM prevention)
# ------------------------------------------------------------------

import base64
import os
import gc
import time
import random
from io import BytesIO
from typing import Dict, Any, Optional, Union
from PIL import Image
import torch
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    EulerDiscreteScheduler,
)
from upscaler import upscale_image

# -------------------------
# Paths / Config
# -------------------------
MODEL_ID = os.environ.get("SDXL_LOCAL_PATH")
if not MODEL_ID or not os.path.exists(MODEL_ID):
    raise RuntimeError("❌ SDXL_LOCAL_PATH is not set or invalid. Ensure local model path is correct.")

LIGHTNING_LORA_DIR = os.environ.get("SDXL_LIGHTNING_LORA_PATH", "/app/models/sdxl_lightning")
LIGHTNING_LORA_FILE = "sdxl_lightning_8step_lora.safetensors"
LIGHTNING_LORA_PATH = os.path.join(LIGHTNING_LORA_DIR, LIGHTNING_LORA_FILE)

# SDXL-Lightning defaults: 8 steps, EulerDiscrete trailing, guidance_scale ~1
DEFAULT_STEPS = 8
DEFAULT_CFG = 1.0

DEFAULT_OUTPUT_DIR = "./outputs"
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# -------------------------
# Lazy Pipeline Global (will load on first use)
# -------------------------
_pipeline = None


def get_pipeline():
    """Lazy-initialize the SDXL Img2Img pipeline with Lightning LoRA fused."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    total_start = time.time()

    # MEMORY-OPTIMIZED: Check if T2I pipeline is already loaded and share components
    # This prevents duplicate model loading on low-VRAM GPUs (<24GB)
    try:
        from sdxl_T2I_inference import _pipeline as t2i_pipeline
        if t2i_pipeline is not None:
            print(f"🧠 T2I pipeline already loaded → creating I2I from shared components (saves ~7-8GB VRAM)")
            pipeline_start = time.time()
            pipe = StableDiffusionXLImg2ImgPipeline(**t2i_pipeline.components)
            pipeline_time = time.time() - pipeline_start
            print(f"✅ I2I pipeline created from shared components in {pipeline_time:.1f}s")
            
            # Set EulerDiscreteScheduler with trailing timesteps (required by SDXL-Lightning)
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config, timestep_spacing="trailing"
            )
            
            # Load and fuse SDXL-Lightning LoRA if not already fused in T2I
            # Note: If T2I already has LoRA fused, we need to check/apply LoRA here too
            # For now, we'll load LoRA if it exists (T2I doesn't use Lightning LoRA)
            if os.path.exists(LIGHTNING_LORA_PATH):
                try:
                    lora_start = time.time()
                    print(f"⚡ Loading SDXL-Lightning LoRA from {LIGHTNING_LORA_PATH}")
                    pipe.load_lora_weights(LIGHTNING_LORA_DIR, weight_name=LIGHTNING_LORA_FILE)
                    pipe.fuse_lora()
                    lora_time = time.time() - lora_start
                    print(f"✅ Lightning LoRA loaded and fused in {lora_time:.1f}s")
                except Exception as e:
                    print(f"⚠️ Failed to load Lightning LoRA: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️ Lightning LoRA not found at {LIGHTNING_LORA_PATH}, running without LoRA")
            
            _pipeline = pipe
            total_time = time.time() - total_start
            print(f"✅ SDXL Img2Img pipeline initialized in {total_time:.1f}s (shared components: {pipeline_time:.1f}s)")
            return _pipeline
    except (ImportError, AttributeError):
        # T2I not loaded yet, fall through to full load
        pass

    # Fallback: Full pipeline load (if T2I not available or sharing failed)
    # CRITICAL: Clear GPU cache before loading to maximize free memory
    gc.collect()
    torch.cuda.empty_cache()

    print(f"🧠 Loading SDXL Img2Img pipeline from {MODEL_ID}")
    pipeline_start = time.time()
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    pipeline_time = time.time() - pipeline_start
    print(f"✅ Base pipeline loaded in {pipeline_time:.1f}s")

    # Load and fuse SDXL-Lightning LoRA for fast inference
    if os.path.exists(LIGHTNING_LORA_PATH):
        try:
            lora_start = time.time()
            print(f"⚡ Loading SDXL-Lightning LoRA from {LIGHTNING_LORA_PATH}")
            pipe.load_lora_weights(LIGHTNING_LORA_DIR, weight_name=LIGHTNING_LORA_FILE)
            pipe.fuse_lora()
            lora_time = time.time() - lora_start
            print(f"✅ Lightning LoRA loaded and fused in {lora_time:.1f}s")
        except Exception as e:
            print(f"⚠️ Failed to load Lightning LoRA: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ Lightning LoRA not found at {LIGHTNING_LORA_PATH}, running without LoRA")

    # Set EulerDiscreteScheduler with trailing timesteps (required by SDXL-Lightning)
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    print("✅ Scheduler set to EulerDiscreteScheduler (trailing timesteps)")

    _pipeline = pipe
    total_time = time.time() - total_start
    print(f"✅ SDXL Img2Img pipeline initialized in {total_time:.1f}s (base: {pipeline_time:.1f}s)")
    return _pipeline


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
    s = str(img)
    if os.path.exists(s):
        return Image.open(s).convert("RGB")
    try:
        decoded = base64.b64decode(s)
        return Image.open(BytesIO(decoded)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Unable to decode image: {e}")


def _get_dynamic_size(img: Image.Image, max_dim: int = 1024) -> tuple[int, int]:
    """Compute output resolution keeping aspect ratio, multiple-of-64, within max_dim."""
    w, h = img.size
    aspect = w / h
    if w > h:
        new_w = min(w, max_dim)
        new_h = int(new_w / aspect)
    else:
        new_h = min(h, max_dim)
        new_w = int(new_h * aspect)
    new_w = max(64, (new_w // 64) * 64)
    new_h = max(64, (new_h // 64) * 64)
    return new_w, new_h


# -------------------------
# Main Img2Img Function
# -------------------------
def run_inference_sdxl_i2i(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run SDXL Img2Img inference with Lightning LoRA.

    Args:
        job_id: unique job identifier (for output filenames)
        prompt: text prompt for image editing
        params:
        - output_dir (str)
        - init_image (base64/path/PIL)
        - negative_prompt (str)
        - sample_steps (int) - default 8 for Lightning
        - cfg_scale (float) - default 1.0 for Lightning
        - strength (float) - default 0.7
        - seed (int)
        - scale_image (bool)
        - scale_factor (int)

    Returns: absolute path to final image (upscaled if requested)
    """
    # Consistent global output path
    default_out = "/app/worker/outputs"
    output_dir = os.path.abspath(params.get("output_dir", default_out))
    os.makedirs(output_dir, exist_ok=True)

    pipe = get_pipeline()

    init_image = _load_image(params.get("init_image"))
    if init_image is None:
        raise ValueError("init_image is required for img2img")

    width, height = _get_dynamic_size(init_image, max_dim=1024)
    negative_prompt = params.get("negative_prompt", "")

    # Parameters (SDXL-Lightning defaults: 8 steps, cfg=1)
    steps = int(params.get("sample_steps", DEFAULT_STEPS))
    cfg = float(params.get("cfg_scale", DEFAULT_CFG))
    strength = float(params.get("strength", 0.7))
    seed = int(params.get("seed", random.randint(0, 2**32 - 1)))

    # CRITICAL: Clear GPU memory before inference to maximize available VRAM
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Log memory state before inference
    try:
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        print(f"🔍 GPU memory before inference: {free_mem / 1024**3:.2f} GB free")
    except Exception:
        pass

    print(f"⚙️ Running Img2Img | Steps={steps}, CFG={cfg}, Strength={strength}, Seed={seed}, Size={width}x{height}")
    generator = torch.manual_seed(seed) if seed >= 0 else None

    # Run inference with automatic mixed precision for memory efficiency
    try:
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg,
                strength=strength,
                generator=generator,
            )
            image = result.images[0]
    finally:
        # CRITICAL: Clean up GPU memory after inference
        gc.collect()
        torch.cuda.empty_cache()

    out_path = os.path.join(output_dir, f"sdxl_i2i_{job_id}.png")
    image.save(out_path)
    print(f"✅ Saved base image: {out_path}")

    # -------------------------
    # Optional Upscaling
    # -------------------------
    if params.get("scale_image", False):
        scale_factor = int(params.get("scale_factor", 2))
        model_name = params.get("scale_model_name", f"RealESRGAN_x{scale_factor}plus")
        try:
            upscaled = upscale_image(out_path, model_name=model_name, scale_factor=scale_factor)
            up_path = os.path.join(output_dir, f"sdxl_i2i_{job_id}_upscaled.png")
            upscaled.save(up_path)
            print(f"✅ Saved upscaled image: {up_path}")
            return os.path.abspath(up_path)
        except Exception as e:
            print(f"⚠️ Upscaling failed: {e}")
            return os.path.abspath(out_path)

    return os.path.abspath(out_path)


# -------------------------
# CLI Test
# -------------------------
if __name__ == "__main__":
    job_id = "test_sdxl_i2i"
    prompt = "A fantasy cinematic portrait, highly detailed, realistic"

    input_path = "./input_image.jpg"
    if not os.path.exists(input_path):
        raise SystemExit("❌ Please provide input_image.jpg to test this script.")

    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    params = {
        "sample_steps": 8,
        "cfg_scale": 1.0,
        "strength": 0.65,
        "output_dir": "./outputs",
        "negative_prompt": "lowres, blurry, bad anatomy, extra limbs",
        "init_image": img_b64,
        "scale_image": False,
    }

    out = run_inference_sdxl_i2i(job_id, prompt, params)
    print("✅ Img2Img result saved at:", out)
