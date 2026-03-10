# ------------------------------------------------------------------
# SDXL Inpaint Inference with SDXL-Lightning LoRA
# - Lazy pipeline loading (prevents OOM during FastAPI startup)
# - SDXL-Lightning LoRA fused for fast 8-step inference
# - Dynamic resolution + mask preprocessing
# - GPU memory cleanup before/after inference (OOM prevention)
# ------------------------------------------------------------------

import base64
import os
import gc
import time
import random
from typing import Dict, Any, Optional, Union
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import torch
from diffusers import (
    StableDiffusionXLInpaintPipeline,
    EulerDiscreteScheduler,
)
from upscaler import upscale_image

# -------------------------
# Paths / Config
# -------------------------
MODEL_ID = os.environ.get("SDXL_LOCAL_PATH")
if not MODEL_ID or not os.path.exists(MODEL_ID):
    raise RuntimeError("❌ SDXL_LOCAL_PATH is not set or invalid. Ensure local model path exists.")

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
    """Lazy-initialize the SDXL Inpaint pipeline with Lightning LoRA fused."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    total_start = time.time()

    # MEMORY-OPTIMIZED: Check if T2I pipeline is already loaded and share components
    # This prevents duplicate model loading on low-VRAM GPUs (<24GB)
    try:
        from sdxl_T2I_inference import _pipeline as t2i_pipeline
        if t2i_pipeline is not None:
            print(f"🧠 T2I pipeline already loaded → creating Inpaint from shared components (saves ~7-8GB VRAM)")
            pipeline_start = time.time()
            pipe = StableDiffusionXLInpaintPipeline(**t2i_pipeline.components)
            pipeline_time = time.time() - pipeline_start
            print(f"✅ Inpaint pipeline created from shared components in {pipeline_time:.1f}s")
            
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
            print(f"✅ SDXL Inpaint pipeline initialized in {total_time:.1f}s (shared components: {pipeline_time:.1f}s)")
            return _pipeline
    except (ImportError, AttributeError):
        # T2I not loaded yet, fall through to full load
        pass

    # Fallback: Full pipeline load (if T2I not available or sharing failed)
    # CRITICAL: Clear GPU cache before loading to maximize free memory
    gc.collect()
    torch.cuda.empty_cache()

    print(f"🧠 Loading SDXL Inpaint pipeline from {MODEL_ID}")
    pipeline_start = time.time()
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
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
    print(f"✅ SDXL Inpaint pipeline initialized in {total_time:.1f}s (base: {pipeline_time:.1f}s)")
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


def _process_mask_for_diffusers(
    mask: Optional[Image.Image],
    target_size: Optional[tuple] = None,
    binarize: bool = True,
    threshold: int = 127,
    invert: bool = False,
    resize_nearest: bool = False,
    debug_save_path: Optional[str] = None,
) -> Optional[Image.Image]:
    """Prepare binary mask for inpainting."""
    if mask is None:
        return None
    if mask.mode != "L":
        mask = mask.convert("L")
    if target_size is not None and mask.size != target_size:
        resample = Image.NEAREST if resize_nearest else Image.BILINEAR
        mask = mask.resize(target_size, resample=resample)
    if binarize:
        mask = mask.point(lambda p: 255 if p > threshold else 0)
    if invert:
        mask = ImageOps.invert(mask)
    if debug_save_path:
        try:
            os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
            mask.save(debug_save_path)
        except Exception:
            pass
    return mask


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
# Main Inference Function
# -------------------------
def run_inference_sdxl_inpaint(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run SDXL inpainting with Lightning LoRA.

    Args:
        job_id: unique job identifier (for output filenames)
        prompt: text prompt for inpainting
        params:
        - output_dir (str)
        - init_image (base64/path/PIL)
        - mask_image (base64/path/PIL)
        - negative_prompt (str)
        - sample_steps (int) - default 8 for Lightning
        - cfg_scale (float) - default 1.0 for Lightning
        - seed (int)
        - binarize_mask (bool)
        - invert_mask (bool)
        - mask_threshold (int)
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
        raise ValueError("init_image is required for inpainting")

    raw_mask = _load_image(params.get("mask_image"))
    binarize = params.get("binarize_mask", True)
    invert_mask = params.get("invert_mask", False)
    mask_threshold = params.get("mask_threshold", 127)
    debug_save_mask = params.get("debug_save_mask", False)

    width, height = _get_dynamic_size(init_image, max_dim=1024)

    mask_image = None
    if raw_mask is not None:
        mask_image = _process_mask_for_diffusers(
            raw_mask,
            target_size=(width, height),
            binarize=binarize,
            threshold=mask_threshold,
            invert=invert_mask,
            resize_nearest=True,
            debug_save_path=(
                os.path.join(output_dir, f"debug_{job_id}_mask.png")
                if debug_save_mask
                else None
            ),
        )

        arr = np.array(mask_image)
        white = int((arr >= 128).sum())
        total = arr.size
        print(f"[mask diagnostics] white={white}/{total} ({white/total*100:.2f}%) invert={invert_mask}")

        if white == 0:
            raise ValueError("Processed mask has no white pixels. Nothing to inpaint.")

    negative_prompt = params.get("negative_prompt", "")

    # Parameters (SDXL-Lightning defaults: 8 steps, cfg=1)
    steps = int(params.get("sample_steps", DEFAULT_STEPS))
    cfg = float(params.get("cfg_scale", DEFAULT_CFG))
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

    print(f"⚙️ Inpainting | Steps={steps}, CFG={cfg}, Seed={seed}, Size={width}x{height}")
    generator = torch.manual_seed(seed) if seed >= 0 else None

    # Run inference with automatic mixed precision for memory efficiency
    try:
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                width=width,
                height=height,
                negative_prompt=negative_prompt,
                image=init_image,
                mask_image=mask_image,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
            )
            image = result.images[0]
    finally:
        # CRITICAL: Clean up GPU memory after inference
        gc.collect()
        torch.cuda.empty_cache()

    out_path = os.path.join(output_dir, f"sdxl_inpaint_{job_id}.png")
    image.save(out_path)
    print(f"✅ Saved base inpaint result: {out_path}")

    # -------------------------
    # Optional Upscaling
    # -------------------------
    if params.get("scale_image", False):
        scale_factor = int(params.get("scale_factor", 2))
        model_name = params.get("scale_model_name", f"RealESRGAN_x{scale_factor}plus")
        try:
            upscaled = upscale_image(out_path, model_name=model_name, scale_factor=scale_factor)
            up_path = os.path.join(output_dir, f"sdxl_inpaint_{job_id}_upscaled.png")
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
    job_id = "test_inpaint"
    prompt = "Cleanly remove and replace masked area"

    input_path = "./input_image.jpg"
    mask_path = "./mask.png"

    if not os.path.exists(input_path) or not os.path.exists(mask_path):
        raise SystemExit("❌ Provide input_image.jpg and mask.png to test this script.")

    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    with open(mask_path, "rb") as f:
        mask_b64 = base64.b64encode(f.read()).decode("utf-8")

    params = {
        "init_image": img_b64,
        "mask_image": mask_b64,
        "binarize_mask": True,
        "invert_mask": True,
        "mask_threshold": 127,
        "debug_save_mask": True,
        "sample_steps": 8,
        "cfg_scale": 1.0,
        "output_dir": "./outputs",
        "negative_prompt": "low quality, blurry",
        "scale_image": False,
    }

    out = run_inference_sdxl_inpaint(job_id, prompt, params)
    print("✅ Inpainted image saved at:", out)
