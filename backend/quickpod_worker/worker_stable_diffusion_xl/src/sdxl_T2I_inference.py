# realism_illustrious_infer.py
# Robust SDXL Txt2Img with dynamic resolution + embedding auto-prepend

import os
os.environ["HF_HOME"] = "/models_storage/all_models"
os.environ["HF_HUB_CACHE"] = "/models_storage/all_models"
import random
from typing import Dict, Any, Optional, Union
from PIL import Image
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler
)

# -------------------------
# Environment
# -------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_DIR = os.path.join("/models_storage", "all_models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_ID = "John6666/realism-illustrious-by-stable-yogi-v50-fp16-sdxl"

EMBEDDINGS_DIR = os.path.join(MODEL_DIR, "realism_illustrious", "embeddings")
POS_EMBED_PATH = os.path.join(EMBEDDINGS_DIR, "Stable_Yogis_Illustrious_Positives.safetensors")
NEG_EMBED_PATH = os.path.join(EMBEDDINGS_DIR, "Stable_Yogis_Illustrious_Negatives-neg.safetensors")

DEFAULT_OUTPUT_DIR = "./outputs"
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# -------------------------
# Load pipeline
# -------------------------
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
    cache_dir=MODEL_DIR
).to("cuda")

# Keep default scheduler config for optional re-init
default_scheduler_config = pipe.scheduler.config

# Load embeddings
POS_EMB_PREPEND = "illustrious_pos"
NEG_EMB_PREPEND = "illustrious_neg"
if os.path.exists(POS_EMBED_PATH):
    pipe.load_textual_inversion(POS_EMBED_PATH, token=POS_EMB_PREPEND)
if os.path.exists(NEG_EMBED_PATH):
    pipe.load_textual_inversion(NEG_EMBED_PATH, token=NEG_EMB_PREPEND)

# -------------------------
# Helpers
# -------------------------
def _get_dynamic_size(ref_img: Optional[Union[str, Image.Image]] = None, max_dim: int = 1152) -> tuple[int, int]:
    """
    Compute width/height for Txt2Img keeping aspect ratio of reference image.
    If no image is provided, defaults to max_dim x max_dim (multiple of 64).
    """
    if isinstance(ref_img, str):
        ref_img = Image.open(ref_img).convert("RGB")
    if isinstance(ref_img, Image.Image):
        w, h = ref_img.size
        aspect = w / h
        if w > h:
            new_w = min(w, max_dim)
            new_h = int(new_w / aspect)
        else:
            new_h = min(h, max_dim)
            new_w = int(new_h * aspect)
    else:
        new_w = new_h = max_dim
    # multiples of 64
    new_w = max(64, (new_w // 64) * 64)
    new_h = max(64, (new_h // 64) * 64)
    return new_w, new_h

# -------------------------
# Main Txt2Img inference
# -------------------------
def run_inference_sdxl_t2i(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Txt2Img inference with Realism Illustrious SDXL and embedding auto-prepend.
    params may include:
      - ref_image: optional PIL.Image or path to control aspect ratio
      - sample_steps, cfg_scale, seed, output_dir, negative_prompt, sampler
    """
    negative_prompt = params.get("negative_prompt", "")

    # prepend embeddings if not already included
    if POS_EMB_PREPEND and POS_EMB_PREPEND not in prompt:
        prompt = f"{POS_EMB_PREPEND}, {prompt}"
    if NEG_EMB_PREPEND and NEG_EMB_PREPEND not in str(negative_prompt):
        negative_prompt = f"{NEG_EMB_PREPEND}, {negative_prompt}"

    # Scheduler selection
    sampler_name = params.get("sampler", "DPMSolverMultistep")
    if sampler_name == "DPMSolverMultistep":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(default_scheduler_config)
    elif sampler_name == "EulerA":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(default_scheduler_config)

    steps = params.get("sample_steps", 30)
    width = params.get("width", 512)
    height = params.get("height", 512)
    cfg = params.get("cfg_scale", 5)
    seed = params.get("seed", random.randint(0, 2**32 - 1))
    output_dir = params.get("output_dir", DEFAULT_OUTPUT_DIR)
    generator = torch.manual_seed(seed) if seed is not None and seed >= 0 else None

    # Run inference
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
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"realism_illustrious_{job_id}.png")
    image.save(out_path)
    return out_path

# -------------------------
# CLI test
# -------------------------
if __name__ == "__main__":
    job_id = "test_rlsxl"
    prompt = "a beautiful indian girl in bikini, highly detailed, cinematic lighting, ultra realistic"

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
