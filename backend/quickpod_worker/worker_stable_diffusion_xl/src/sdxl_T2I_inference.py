# inference_sdxl_t2i.py
# ✅ Lazy-loading implementation
# SDXL Txt2Img with scheduler switch, embedding auto-prepend, and upscaling
# GPU-only, assumes baked local models

import os
import random
from typing import Dict, Any
from PIL import Image
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
)
from upscaler import upscale_image

torch.cuda.empty_cache()

# -------------------------
# Paths / Config
# -------------------------
MODEL_ID = os.environ.get("SDXL_LOCAL_PATH")
if not MODEL_ID or not os.path.exists(MODEL_ID):
    raise RuntimeError("❌ SDXL_LOCAL_PATH is not set or invalid. Ensure the model is baked and available.")

EMBEDDINGS_DIR = os.environ.get("SDXL_EMBEDDINGS_PATH", os.path.join(MODEL_ID, "embeddings"))
POS_EMBED_PATH = os.path.join(EMBEDDINGS_DIR, "Stable_Yogis_Illustrious_Positives.safetensors")
NEG_EMBED_PATH = os.path.join(EMBEDDINGS_DIR, "Stable_Yogis_Illustrious_Negatives-neg.safetensors")

POS_TOKEN = "illustrious_pos"
NEG_TOKEN = "illustrious_neg"

DEFAULT_OUTPUT_DIR = "./outputs"
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# -------------------------
# Lazy Loader Cache
# -------------------------
_model_cache = {"pipe": None, "scheduler_config": None}


def get_sdxl_t2i_pipe():
    """Lazy-load the SDXL Txt2Img pipeline and cache it."""
    global _model_cache
    if _model_cache["pipe"] is not None:
        return _model_cache["pipe"], _model_cache["scheduler_config"]

    print("🧠 Loading SDXL Txt2Img pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")

    scheduler_config = pipe.scheduler.config
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)

    # Load textual inversion embeddings (optional)
    if os.path.exists(POS_EMBED_PATH):
        try:
            pipe.load_textual_inversion(POS_EMBED_PATH, token=POS_TOKEN)
            print(f"✅ Loaded positive embedding: {POS_EMBED_PATH}")
        except Exception as e:
            print(f"⚠️ Failed to load POS embedding: {e}")

    if os.path.exists(NEG_EMBED_PATH):
        try:
            pipe.load_textual_inversion(NEG_EMBED_PATH, token=NEG_TOKEN)
            print(f"✅ Loaded negative embedding: {NEG_EMBED_PATH}")
        except Exception as e:
            print(f"⚠️ Failed to load NEG embedding: {e}")

    _model_cache["pipe"] = pipe
    _model_cache["scheduler_config"] = scheduler_config
    print("✅ SDXL Txt2Img pipeline loaded and cached.")
    return pipe, scheduler_config


def unload_sdxl_t2i_pipe():
    """Optional cleanup: unload SDXL Txt2Img pipeline and free GPU memory."""
    global _model_cache
    _model_cache["pipe"] = None
    _model_cache["scheduler_config"] = None
    torch.cuda.empty_cache()
    print("🧹 SDXL Txt2Img pipeline cache cleared and GPU memory freed.")


# -------------------------
# Main Inference
# -------------------------
def run_inference_sdxl_t2i(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """Run SDXL Txt2Img inference with lazy loading and embedding auto-prepend."""
    print(f"🧩 Starting SDXL Txt2Img job: {job_id}")
    pipe, scheduler_config = get_sdxl_t2i_pipe()

    negative_prompt = params.get("negative_prompt", "")

    # Auto-prepend embeddings if not already in prompt
    if POS_TOKEN not in prompt:
        prompt = f"{POS_TOKEN}, {prompt}"
    if NEG_TOKEN not in str(negative_prompt):
        negative_prompt = f"{NEG_TOKEN}, {negative_prompt}"

    # Scheduler selection
    sampler_name = params.get("sampler", "DPMSolverMultistep")
    if sampler_name == "DPMSolverMultistep":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)
    elif sampler_name == "EulerA":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config)
    else:
        print(f"⚠️ Unknown sampler '{sampler_name}', defaulting to DPMSolverMultistep.")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)

    # Parameters
    steps = int(params.get("sample_steps", 30))
    width = int(params.get("width", 768))
    height = int(params.get("height", 768))
    cfg = float(params.get("cfg_scale", 5))
    seed = int(params.get("seed", random.randint(0, 2**32 - 1)))
    output_dir = params.get("output_dir", DEFAULT_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print(f"⚙️ Running Txt2Img | Steps={steps}, CFG={cfg}, Sampler={sampler_name}, Seed={seed}")
    generator = torch.manual_seed(seed) if seed >= 0 else None

    # Inference
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
    )

    image = result.images[0]
    out_path = os.path.join(output_dir, f"sdxl_t2i_{job_id}.png")
    image.save(out_path)
    print(f"🖼️ Base image saved at {out_path}")

    # -------------------------
    # Optional Upscaling and metadata removal
    # -------------------------
    scale_image = params.get("scale_image", True)
    scale_factor = params.get("scale_factor", 2) if scale_image else 0
    model_name = "RealESRGAN_x2plus" if scale_factor == 2 else "RealESRGAN_x4plus"

    upscaled = upscale_image(out_path,  model_name=model_name, scale_factor=scale_factor)

    upscaled_path = os.path.join(output_dir, f"{job_id}_upscaled.png")
    upscaled.save(upscaled_path)
    print(f"🚀 Upscaled image saved at {upscaled_path}")

    return os.path.abspath(upscaled_path)


# -------------------------
# CLI Test
# -------------------------
if __name__ == "__main__":
    job_id = "test_rlsxl"
    prompt = "a beautiful indian girl, highly detailed, cinematic lighting, ultra realistic"

    params = {
        "sample_steps": 30,
        "cfg_scale": 5,
        "output_dir": "./outputs",
        "negative_prompt": "lowres, bad anatomy, blurry, extra limbs, deformed, unrealistic skin",
        "sampler": "DPMSolverMultistep",
        "width": 768,
        "height": 512,
    }

    out = run_inference_sdxl_t2i(job_id, prompt, params)
    print("✅ Output image saved at:", out)
