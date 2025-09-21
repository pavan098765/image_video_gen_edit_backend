# realism_illustrious_inpaint.py
# Robust SDXL inpaint with dynamic resolution + mask preprocessing + embedding auto-prepend

import base64
import os
import random
from typing import Dict, Any, Optional, Union
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import torch
from diffusers import (
    StableDiffusionXLInpaintPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler
)

# -------------------------
# Configuration / paths
# -------------------------
MODEL_DIR = os.environ.get("HF_HOME", "/models_storage/all_models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_ID = os.environ.get(
    "SDXL_LOCAL_PATH",
    os.path.join(MODEL_DIR, "hf_models", "John6666_realism-illustrious-by-stable-yogi-v50-fp16-sdxl")
)

DEFAULT_OUTPUT_DIR = "./outputs"
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

EMBEDDINGS_DIR = os.environ.get("SDXL_EMBEDDINGS_PATH", os.path.join(MODEL_ID, "embeddings"))
POS_EMB_PATH = os.path.join(EMBEDDINGS_DIR, "Stable_Yogis_Illustrious_Positives.safetensors")
NEG_EMB_PATH = os.path.join(EMBEDDINGS_DIR, "Stable_Yogis_Illustrious_Negatives-neg.safetensors")

POS_TOKEN = "illustrious_pos"
NEG_TOKEN = "illustrious_neg"

# -------------------------
# Load pipeline
# -------------------------
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
    cache_dir=MODEL_DIR,
).to("cuda")

# Keep a copy of default scheduler config for safe re-init
default_scheduler_config = pipe.scheduler.config
pipe.scheduler = DPMSolverMultistepScheduler.from_config(default_scheduler_config)

# -------------------------
# Load embeddings
# -------------------------
POS_EMB_PREPEND = None
NEG_EMB_PREPEND = None

if os.path.exists(POS_EMB_PATH):
    try:
        pipe.load_textual_inversion(POS_EMB_PATH, token=POS_TOKEN)
        POS_EMB_PREPEND = POS_TOKEN
    except Exception:
        POS_EMB_PREPEND = f"embedding:{os.path.basename(POS_EMB_PATH)}"
else:
    POS_EMB_PREPEND = f"embedding:{os.path.basename(POS_EMB_PATH)}"

if os.path.exists(NEG_EMB_PATH):
    try:
        pipe.load_textual_inversion(NEG_EMB_PATH, token=NEG_TOKEN)
        NEG_EMB_PREPEND = NEG_TOKEN
    except Exception:
        NEG_EMB_PREPEND = f"embedding:{os.path.basename(NEG_EMB_PATH)}"
else:
    NEG_EMB_PREPEND = f"embedding:{os.path.basename(NEG_EMB_PATH)}"

# -------------------------
# Helpers
# -------------------------
def _load_image(img: Optional[Union[str, Image.Image, bytes]]) -> Optional[Image.Image]:
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


def _process_mask_for_diffusers(
    mask: Image.Image,
    target_size: Optional[tuple] = None,
    binarize: bool = True,
    threshold: int = 127,
    invert: bool = False,
    resize_nearest: bool = False,
    debug_save_path: Optional[str] = None
) -> Image.Image:
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
# Main inpaint function
# -------------------------
def run_inference_sdxl_inpaint(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
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
            debug_save_path=(os.path.join(params.get("output_dir", DEFAULT_OUTPUT_DIR),
                                          f"debug_{job_id}_processed_mask.png")
                             if debug_save_mask else None)
        )
        arr = np.array(mask_image)
        white = int((arr >= 128).sum())
        total = arr.size
        print(f"[mask diagnostics] white pixels (to be inpainted): {white}/{total} "
              f"({white/total*100:.2f}%) ; invert_mask={invert_mask} ; binarize={binarize}")
        if white == 0:
            raise ValueError("Processed mask has zero white pixels (nothing to inpaint). "
                             "Check invert_mask/binarize settings and mask orientation.")

    negative_prompt = params.get("negative_prompt", "")
    if POS_EMB_PREPEND and POS_EMB_PREPEND not in prompt:
        prompt = f"{POS_EMB_PREPEND}, {prompt}"
    if NEG_EMB_PREPEND and NEG_EMB_PREPEND not in str(negative_prompt):
        negative_prompt = f"{NEG_EMB_PREPEND}, {negative_prompt}"

    sampler_name = params.get("sampler", "DPMSolverMultistep")
    if sampler_name == "DPMSolverMultistep":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(default_scheduler_config)
    elif sampler_name == "EulerA":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(default_scheduler_config)

    steps = params.get("sample_steps", 30)
    cfg = params.get("cfg_scale", 5)
    seed = params.get("seed", random.randint(0, 2**32 - 1))
    output_dir = params.get("output_dir", DEFAULT_OUTPUT_DIR)
    generator = torch.manual_seed(seed) if seed is not None and seed >= 0 else None

    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator
    )

    out_img = result.images[0]
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"inpaint_{job_id}.png")
    out_img.save(out_path)
    return out_path

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    job_id = "inpaint_debug"
    prompt = "remove the mask area cleanly and fill background"

    input_path = "/quickpod/quickpod_workers/worker_stable_diffusion_xl/src/input_image.jpg"
    mask_path = "/quickpod/quickpod_workers/worker_stable_diffusion_xl/src/mask.png"

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
        "negative_prompt": "blurry, low quality",
        "sampler": "DPMSolverMultistep"
    }

    out = run_inference_sdxl_inpaint(job_id, prompt, params=params)
    print("✅ Inpainted image saved at:", out)
