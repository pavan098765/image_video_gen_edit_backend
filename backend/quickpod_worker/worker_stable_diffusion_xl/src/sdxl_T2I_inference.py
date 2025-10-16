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
    EulerAncestralDiscreteScheduler,
)

# -------------------------
# Environment
# -------------------------

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_DIR = os.path.join("/models_storage", "all_models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_ID = "John6666/realism-illustrious-by-stable-yogi-v50-fp16-sdxl"

EMB_DIR = os.path.join(MODEL_DIR, "realism_illustrious", "embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

POS_EMB_FILENAME = "Stable_Yogis_Illustrious_Positives.safetensors"
NEG_EMB_FILENAME = "Stable_Yogis_Illustrious_Negatives-neg.safetensors"
POS_EMB_PATH = os.path.join(EMB_DIR, POS_EMB_FILENAME)
NEG_EMB_PATH = os.path.join(EMB_DIR, NEG_EMB_FILENAME)

DEFAULT_OUTPUT_DIR = "./outputs"
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# -------------------------
# Cached pipeline
# -------------------------
PIPE: Optional[StableDiffusionXLPipeline] = None
DEFAULT_SCHEDULER_CONFIG = None
POS_EMB_PREPEND: Optional[str] = None
NEG_EMB_PREPEND: Optional[str] = None


def _get_pipeline() -> StableDiffusionXLPipeline:
    """Lazy-load pipeline + textual inversion embeddings."""
    global PIPE, DEFAULT_SCHEDULER_CONFIG, POS_EMB_PREPEND, NEG_EMB_PREPEND
    if PIPE is not None:
        return PIPE

    PIPE = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
        cache_dir=MODEL_DIR,
    ).to("cuda")

    DEFAULT_SCHEDULER_CONFIG = PIPE.scheduler.config

    # Try loading textual inversion embeddings
    POS_EMB_PREPEND = "illustrious_pos"
    NEG_EMB_PREPEND = "illustrious_neg"

    try:
        if os.path.exists(POS_EMB_PATH):
            PIPE.load_textual_inversion(POS_EMB_PATH, token=POS_EMB_PREPEND)
        else:
            POS_EMB_PREPEND = f"embedding:{POS_EMB_FILENAME}"
    except Exception:
        POS_EMB_PREPEND = f"embedding:{POS_EMB_FILENAME}"

    try:
        if os.path.exists(NEG_EMB_PATH):
            PIPE.load_textual_inversion(NEG_EMB_PATH, token=NEG_EMB_PREPEND)
        else:
            NEG_EMB_PREPEND = f"embedding:{NEG_EMB_FILENAME}"
    except Exception:
        NEG_EMB_PREPEND = f"embedding:{NEG_EMB_FILENAME}"

    print("[Init] SDXL Txt2Img pipeline loaded")
    return PIPE


# -------------------------
# Helpers
# -------------------------
def _get_dynamic_size(
    ref_img: Optional[Union[str, Image.Image]] = None, max_dim: int = 1152
) -> tuple[int, int]:
    """
    Compute width/height for Txt2Img keeping aspect ratio of reference image.
    If no image is provided, defaults to max_dim x max_dim (multiple of 64).
    """
    if isinstance(ref_img, str) and os.path.exists(ref_img):
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

    new_w = max(64, (new_w // 64) * 64)
    new_h = max(64, (new_h // 64) * 64)
    return new_w, new_h


# -------------------------
# Main Txt2Img inference
# -------------------------
def run_inference_sdxl_t2i(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    pipe = _get_pipeline()

    negative_prompt = params.get("negative_prompt", "")

    # Auto-prepend embeddings
    if POS_EMB_PREPEND and POS_EMB_PREPEND not in prompt:
        prompt = f"{POS_EMB_PREPEND}, {prompt}"
    if NEG_EMB_PREPEND and NEG_EMB_PREPEND not in str(negative_prompt):
        negative_prompt = f"{NEG_EMB_PREPEND}, {negative_prompt}"

    # Scheduler selection
    sampler_name = params.get("sampler", "DPMSolverMultistep")
    if sampler_name == "DPMSolverMultistep":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(DEFAULT_SCHEDULER_CONFIG)
    elif sampler_name == "EulerA":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(DEFAULT_SCHEDULER_CONFIG)

    steps = int(params.get("sample_steps", 30))
    width = int(params.get("width", 512))
    height = int(params.get("height", 512))
    cfg = float(params.get("cfg_scale", 5))
    seed = int(params.get("seed", random.randint(0, 2**32 - 1)))
    output_dir = params.get("output_dir", DEFAULT_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    generator = torch.manual_seed(seed) if seed >= 0 else None

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
