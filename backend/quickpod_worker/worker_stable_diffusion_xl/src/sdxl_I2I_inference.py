# realism_illustrious_img2img.py
# Robust SDXL Img2Img with dynamic resolution + embedding auto-prepend

import base64
import os
import random
from io import BytesIO
from typing import Dict, Any, Optional, Union
from PIL import Image
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler

# -------------------------
# Config / paths
# -------------------------
MODEL_DIR = os.path.join("/models_storage", "all_models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_ID = "John6666/realism-illustrious-by-stable-yogi-v50-fp16-sdxl"
DEFAULT_OUTPUT_DIR = "./outputs"
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

EMB_DIR = os.path.join(MODEL_DIR, "embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

POS_EMB_FILENAME = "Stable_Yogis_Illustrious_Positives.safetensors"
NEG_EMB_FILENAME = "Stable_Yogis_Illustrious_Negatives-neg.safetensors"
POS_TOKEN = "illustrious_pos"
NEG_TOKEN = "illustrious_neg"
POS_EMB = f"embedding:{POS_EMB_FILENAME}"
NEG_EMB = f"embedding:{NEG_EMB_FILENAME}"

# -------------------------
# Load Img2Img pipeline
# -------------------------
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
    cache_dir=MODEL_DIR,
).to("cuda")

# Store default scheduler config
default_scheduler_config = pipe.scheduler.config
# Recommended default scheduler for V5
pipe.scheduler = DPMSolverMultistepScheduler.from_config(default_scheduler_config)

# -------------------------
# Helpers
# -------------------------
def _load_image(img: Optional[Union[str, Image.Image, bytes]]) -> Optional[Image.Image]:
    """Load image from path, PIL.Image, bytes, or base64 string."""
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, bytes):
        return Image.open(BytesIO(img)).convert("RGB")
    if os.path.exists(str(img)):
        return Image.open(str(img)).convert("RGB")
    try:
        decoded = base64.b64decode(img)
        return Image.open(BytesIO(decoded)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Unable to decode image from input: {e}")

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
    new_w = (new_w // 64) * 64
    new_h = (new_h // 64) * 64
    new_w = max(64, new_w)
    new_h = max(64, new_h)
    return new_w, new_h

# -------------------------
# Main Img2Img function
# -------------------------
def run_inference_sdxl_i2i(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    init_image = _load_image(params.get("image"))
    if init_image is None:
        raise ValueError("init_image is required for img2img")

    # Dynamic resolution
    width, height = _get_dynamic_size(init_image, max_dim=1152)

    negative_prompt = params.get("negative_prompt", "")

    # Auto-prepend embeddings
    if POS_EMB and POS_EMB not in prompt:
        prompt = f"{POS_EMB}, {prompt}"
    if NEG_EMB and NEG_EMB not in str(negative_prompt):
        negative_prompt = f"{NEG_EMB}, {negative_prompt}"

    # Scheduler option (optional)
    sampler_name = params.get("sampler", "DPMSolverMultistep")
    if sampler_name == "DPMSolverMultistep":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(default_scheduler_config)
    elif sampler_name == "EulerA":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(default_scheduler_config)

    steps = params.get("sample_steps", 30)
    cfg = params.get("cfg_scale", 5)
    strength = params.get("strength", 0.7)
    seed = params.get("seed", random.randint(0, 2**32 - 1))
    output_dir = params.get("output_dir", DEFAULT_OUTPUT_DIR)
    generator = torch.manual_seed(seed) if seed is not None and seed >= 0 else None

    # Run Img2Img
    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        negative_prompt=negative_prompt,
        image=init_image,
        num_inference_steps=steps,
        guidance_scale=cfg,
        strength=strength,
        generator=generator
    )

    image = result.images[0]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"img2img_{job_id}.png")
    image.save(out_path)
    return out_path

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    job_id = "img2img_debug"
    prompt = "Undress her, remove her clothes"

    input_path = "/quickpod/quickpod_workers/worker_stable_diffusion_xl/src/input_image.jpg"
    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    params = {
        "sample_steps": 30,
        "cfg_scale": 5,
        "strength": 0.65,
        "output_dir": "./outputs",
        "negative_prompt": "lowres, blurry, bad anatomy, extra limbs",
        "image": img_b64,
        "sampler": "DPMSolverMultistep"
    }

    out = run_inference_sdxl_i2i(job_id, prompt, params=params)
    print("✅ Img2Img result saved at:", out)
