# realism_illustrious_inpaint.py
# ✅ Lazy-loading implementation
# ✅ GPU memory cleanup before/after inference (OOM prevention)
# SDXL Inpaint with dynamic resolution + mask preprocessing + embedding auto-prepend (GPU-only, baked local models)

import base64
import os
import gc
import random
from typing import Dict, Any, Optional, Union
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import torch
from diffusers import (
    StableDiffusionXLInpaintPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
)
from upscaler import upscale_image

# -------------------------
# Paths / Config
# -------------------------
MODEL_ID = os.environ.get("SDXL_LOCAL_PATH")
if not MODEL_ID or not os.path.exists(MODEL_ID):
    raise RuntimeError("❌ SDXL_LOCAL_PATH is not set or invalid. Ensure local model path exists.")

EMBEDDINGS_DIR = os.environ.get("SDXL_EMBEDDINGS_PATH", os.path.join(MODEL_ID, "embeddings"))
POS_EMB_PATH = os.path.join(EMBEDDINGS_DIR, "Stable_Yogis_Illustrious_Positives.safetensors")
NEG_EMB_PATH = os.path.join(EMBEDDINGS_DIR, "Stable_Yogis_Illustrious_Negatives-neg.safetensors")

POS_TOKEN = "illustrious_pos"
NEG_TOKEN = "illustrious_neg"

DEFAULT_OUTPUT_DIR = "./outputs"
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# -------------------------
# Lazy Pipeline Global
# -------------------------
_pipeline = None
_scheduler_config = None


def get_inpaint_pipe():
    """Lazy-load SDXL inpaint pipeline with embeddings and cache it."""
    global _pipeline, _scheduler_config
    if _pipeline is not None:
        return _pipeline, _scheduler_config

    import time
    total_start = time.time()

    # CRITICAL: Clear GPU cache before loading to maximize free memory
    gc.collect()
    torch.cuda.empty_cache()

    print("🧠 Loading SDXL Inpaint pipeline...")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")

    scheduler_config = pipe.scheduler.config
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)

    # Load embeddings
    if os.path.exists(POS_EMB_PATH):
        try:
            pipe.load_textual_inversion(POS_EMB_PATH, token=POS_TOKEN)
            print(f"✅ Loaded positive embedding: {POS_EMB_PATH}")
        except Exception as e:
            print(f"⚠️ Failed to load POS embedding: {e}")

    if os.path.exists(NEG_EMB_PATH):
        try:
            pipe.load_textual_inversion(NEG_EMB_PATH, token=NEG_TOKEN)
            print(f"✅ Loaded negative embedding: {NEG_EMB_PATH}")
        except Exception as e:
            print(f"⚠️ Failed to load NEG embedding: {e}")

    _pipeline = pipe
    _scheduler_config = scheduler_config
    total_time = time.time() - total_start
    print(f"✅ SDXL Inpaint pipeline loaded in {total_time:.1f}s")
    return pipe, scheduler_config


def unload_inpaint_pipe():
    """Optional cleanup: unload SDXL pipeline and clear GPU/CPU memory."""
    global _pipeline, _scheduler_config
    _pipeline = None
    _scheduler_config = None
    gc.collect()
    torch.cuda.empty_cache()
    print("🧹 SDXL Inpaint pipeline cache cleared and GPU memory freed.")


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


def _get_dynamic_size(img: Image.Image, max_dim: int = 1152) -> tuple[int, int]:
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
    """Run SDXL inpainting job with lazy loading, embedding prepend, and mask handling."""
    print(f"🧩 Starting SDXL Inpaint job: {job_id}")
    pipe, scheduler_config = get_inpaint_pipe()

    init_image = _load_image(params.get("init_image"))
    if init_image is None:
        raise ValueError("init_image is required for inpainting")

    raw_mask = _load_image(params.get("mask_image"))
    binarize = params.get("binarize_mask", True)
    invert_mask = params.get("invert_mask", False)
    mask_threshold = params.get("mask_threshold", 127)
    debug_save_mask = params.get("debug_save_mask", False)

    width, height = _get_dynamic_size(init_image, max_dim=1152)

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
                os.path.join(params.get("output_dir", DEFAULT_OUTPUT_DIR), f"debug_{job_id}_mask.png")
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
    if POS_TOKEN not in prompt:
        prompt = f"{POS_TOKEN}, {prompt}"
    if NEG_TOKEN not in str(negative_prompt):
        negative_prompt = f"{NEG_TOKEN}, {negative_prompt}"

    sampler_name = params.get("sampler", "DPMSolverMultistep")
    if sampler_name == "DPMSolverMultistep":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)
    elif sampler_name == "EulerA":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config)
    else:
        print(f"⚠️ Unknown sampler '{sampler_name}', using DPMSolverMultistep.")

    steps = int(params.get("sample_steps", 30))
    cfg = float(params.get("cfg_scale", 5))
    seed = int(params.get("seed", random.randint(0, 2**32 - 1)))
    output_dir = params.get("output_dir", DEFAULT_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

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

    print(f"⚙️ Inpainting | Steps={steps}, CFG={cfg}, Sampler={sampler_name}, Seed={seed}")
    generator = torch.manual_seed(seed) if seed >= 0 else None

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
    print(f"🖼️ Base inpaint result saved to {out_path}")

    # -------------------------
    # Optional Upscaling and metadata removal
    # -------------------------
    scale_image = params.get("scale_image", True)
    scale_factor = params.get("scale_factor", 2) if scale_image else 0
    model_name = "RealESRGAN_x2plus" if scale_factor == 2 else "RealESRGAN_x4plus"

    upscaled = upscale_image(out_path, model_name=model_name, scale_factor=scale_factor)

    upscaled_path = os.path.join(output_dir, f"{job_id}_upscaled.png")
    upscaled.save(upscaled_path)
    print(f"🚀 Upscaled image saved at {upscaled_path}")

    return os.path.abspath(upscaled_path)


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
        "sample_steps": 30,
        "cfg_scale": 5,
        "output_dir": "./outputs",
        "negative_prompt": "low quality, blurry",
        "sampler": "DPMSolverMultistep",
    }

    out = run_inference_sdxl_inpaint(job_id, prompt, params)
    print("✅ Inpainted image saved at:", out)
