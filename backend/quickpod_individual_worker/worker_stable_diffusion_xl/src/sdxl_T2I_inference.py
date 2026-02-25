# ------------------------------------------------------------------
# SDXL Txt2Img Inference with SDXL-Lighting-8-steps
# - Lazy pipeline loading (prevents OOM during FastAPI startup)
# - Fast 8-step inference with guidance_scale=0
# - Optional upscaling
# - GPU memory cleanup before/after inference (OOM prevention)
# ------------------------------------------------------------------

import os
import gc
import time
import random
from typing import Dict, Any
from PIL import Image
import torch
from diffusers import StableDiffusionXLPipeline
from upscaler import upscale_image

# --- Auto-merge split safetensors if merge.py exists ---
# NOTE: Models are always in parts for Docker image size management
# Merge happens every time on startup before model loading
try:
    from merge import main as merge_main
    print("🔍 Scanning for .safetensors.part-* files...")
    merge_start = time.time()
    merge_main()
    merge_time = time.time() - merge_start
    print(f"✅ Merge completed in {merge_time:.1f}s")
except Exception as e:
    print(f"⚠️ merge.py auto-merge skipped or failed: {e}")

# -------------------------
# Paths / Config
# -------------------------
MODEL_ID = os.environ.get("SDXL_LOCAL_PATH")
if not MODEL_ID or not os.path.exists(MODEL_ID):
    raise RuntimeError("❌ SDXL_LOCAL_PATH is not set or invalid. Ensure the model is baked and available.")

# SDXL-Lighting-8-steps defaults: 8 steps, guidance_scale=0
DEFAULT_STEPS = 8
DEFAULT_CFG = 0.0

DEFAULT_OUTPUT_DIR = "./outputs"
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# -------------------------
# Lazy Pipeline Global (will load on first use)
# -------------------------
_pipeline = None


def get_pipeline():
    """Lazy-initialize the SDXL Txt2Img pipeline with SDXL-Lighting-8-steps."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    total_start = time.time()

    # CRITICAL: Clear GPU cache before loading to maximize free memory
    gc.collect()
    torch.cuda.empty_cache()

    print(f"🧠 Loading SDXL Txt2Img pipeline from {MODEL_ID}")
    pipeline_start = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    pipeline_time = time.time() - pipeline_start
    print(f"✅ Pipeline loaded in {pipeline_time:.1f}s")

    _pipeline = pipe
    total_time = time.time() - total_start
    print(f"✅ SDXL Txt2Img pipeline initialized in {total_time:.1f}s")
    return _pipeline


# -------------------------
# Main Inference
# -------------------------
def run_inference_sdxl_t2i(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run SDXL Txt2Img inference with SDXL-Lighting-8-steps.

    Args:
        job_id: unique job identifier (for output filenames)
        prompt: text prompt for image generation
        params:
        - output_dir (str)
        - negative_prompt (str)
        - num_inference_steps (int) - default 8
        - guidance_scale (float) - default 0.0
        - width (int) - default 768
        - height (int) - default 768
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

    negative_prompt = params.get("negative_prompt", "")

    # Parameters (SDXL-Lighting-8-steps defaults: 8 steps, guidance_scale=0)
    steps = int(params.get("num_inference_steps", params.get("sample_steps", DEFAULT_STEPS)))
    width = int(params.get("width", 768))
    height = int(params.get("height", 768))
    cfg = float(params.get("guidance_scale", params.get("cfg_scale", DEFAULT_CFG)))
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

    print(f"⚙️ Running Txt2Img | Steps={steps}, Guidance={cfg}, Seed={seed}, Size={width}x{height}")
    generator = torch.Generator(device="cuda").manual_seed(seed) if seed >= 0 else None

    # Run inference with automatic mixed precision for memory efficiency
    try:
        with torch.inference_mode():
            pipe_kwargs = {
                "prompt": prompt,
                "num_inference_steps": steps,
                "guidance_scale": cfg,
                "generator": generator,
            }
            if negative_prompt:
                pipe_kwargs["negative_prompt"] = negative_prompt
            # Width/height are optional - if not provided, pipeline uses defaults
            if width and height:
                pipe_kwargs["width"] = width
                pipe_kwargs["height"] = height
            
            result = pipe(**pipe_kwargs)
            image = result.images[0]
    finally:
        # CRITICAL: Clean up GPU memory after inference
        gc.collect()
        torch.cuda.empty_cache()

    out_path = os.path.join(output_dir, f"sdxl_t2i_{job_id}.png")
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
            up_path = os.path.join(output_dir, f"sdxl_t2i_{job_id}_upscaled.png")
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
    job_id = "test_rlsxl"
    prompt = "a beautiful indian girl, highly detailed, cinematic lighting, ultra realistic"

    params = {
        "num_inference_steps": 8,
        "guidance_scale": 0.0,
        "output_dir": "./outputs",
        "negative_prompt": "",
        "width": 768,
        "height": 512,
        "scale_image": False,
    }

    out = run_inference_sdxl_t2i(job_id, prompt, params)
    print("✅ Output image saved at:", out)
