# realism_illustrious_img2img.py
# ✅ Lazy-loading implementation
# SDXL Img2Img with dynamic resolution + embedding auto-prepend (GPU-only, baked local models)

import base64
import os
import random
from io import BytesIO
from typing import Dict, Any, Optional, Union
from PIL import Image
import torch
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
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
    raise RuntimeError("❌ SDXL_LOCAL_PATH is not set or invalid. Ensure local model path is correct.")

EMBEDDINGS_DIR = os.environ.get(
    "SDXL_EMBEDDINGS_PATH", os.path.join(MODEL_ID, "embeddings")
)
POS_EMBED_PATH = os.path.join(EMBEDDINGS_DIR, "Stable_Yogis_Illustrious_Positives.safetensors")
NEG_EMBED_PATH = os.path.join(EMBEDDINGS_DIR, "Stable_Yogis_Illustrious_Negatives-neg.safetensors")

POS_TOKEN = "illustrious_pos"
NEG_TOKEN = "illustrious_neg"

DEFAULT_OUTPUT_DIR = "./outputs"
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# -------------------------
# Lazy Loader Cache
# -------------------------
_model_cache = {"sdxl_pipe": None, "default_scheduler_config": None}


def get_sdxl_pipe():
    """
    Lazily loads the SDXL Img2Img pipeline and caches it.
    Loads embeddings (if available) and sets default scheduler.
    """
    global _model_cache
    if _model_cache["sdxl_pipe"] is not None:
        return _model_cache["sdxl_pipe"], _model_cache["default_scheduler_config"]

    print("🧠 Loading SDXL Img2Img pipeline...")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")

    default_scheduler_config = pipe.scheduler.config
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(default_scheduler_config)

    # Load embeddings (optional)
    if os.path.exists(POS_EMBED_PATH):
        pipe.load_textual_inversion(POS_EMBED_PATH, token=POS_TOKEN)
        print(f"✅ Loaded positive embedding: {POS_EMBED_PATH}")
    if os.path.exists(NEG_EMBED_PATH):
        pipe.load_textual_inversion(NEG_EMBED_PATH, token=NEG_TOKEN)
        print(f"✅ Loaded negative embedding: {NEG_EMBED_PATH}")

    _model_cache["sdxl_pipe"] = pipe
    _model_cache["default_scheduler_config"] = default_scheduler_config
    print("✅ SDXL Img2Img pipeline loaded and cached.")
    return pipe, default_scheduler_config


def unload_sdxl_pipe():
    """Optional cleanup: unload SDXL pipeline and free GPU/CPU memory."""
    global _model_cache
    _model_cache["sdxl_pipe"] = None
    _model_cache["default_scheduler_config"] = None
    torch.cuda.empty_cache()
    print("🧹 SDXL pipeline cache cleared and GPU memory freed.")


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
# Main Img2Img Function
# -------------------------
def run_inference_sdxl_i2i(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Runs SDXL Img2Img inference with dynamic resolution and embedding auto-prepend.
    Saves and optionally upscales the result.
    """
    print(f"🧩 Starting SDXL Img2Img job: {job_id}")
    pipe, default_scheduler_config = get_sdxl_pipe()

    init_image = _load_image(params.get("init_image"))
    if init_image is None:
        raise ValueError("init_image is required for img2img")

    width, height = _get_dynamic_size(init_image, max_dim=1152)
    negative_prompt = params.get("negative_prompt", "")

    # Auto-prepend embedding tokens
    if POS_TOKEN and POS_TOKEN not in prompt:
        prompt = f"{POS_TOKEN}, {prompt}"
    if NEG_TOKEN and NEG_TOKEN not in str(negative_prompt):
        negative_prompt = f"{NEG_TOKEN}, {negative_prompt}"

    # Scheduler selection
    sampler_name = params.get("sampler", "DPMSolverMultistep")
    if sampler_name == "DPMSolverMultistep":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(default_scheduler_config)
    elif sampler_name == "EulerA":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(default_scheduler_config)
    else:
        print(f"⚠️ Unknown sampler '{sampler_name}', defaulting to DPMSolverMultistep")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(default_scheduler_config)

    steps = int(params.get("sample_steps", 30))
    cfg = float(params.get("cfg_scale", 5))
    strength = float(params.get("strength", 0.7))
    seed = int(params.get("seed", random.randint(0, 2**32 - 1)))
    output_dir = params.get("output_dir", DEFAULT_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print(f"⚙️ Running Img2Img | Steps={steps}, CFG={cfg}, Strength={strength}, Sampler={sampler_name}, Seed={seed}")
    generator = torch.manual_seed(seed) if seed >= 0 else None

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
    out_path = os.path.join(output_dir, f"sdxl_i2i_{job_id}.png")
    image.save(out_path)
    print(f"🖼️ Base image saved at {out_path}")

    # -------------------------
    # Optional Upscaling and compulsory metadata removal
    # -------------------------
    scale_image = params.get("scale_image", True)
    scale_factor = params.get("scale_factor", 2) if scale_image else 0
    model_name = "RealESRGAN_x2plus" if scale_factor == 2 else "RealESRGAN_x4plus"

    upscaled = upscale_image(out_path,  model_name=model_name, scale_factor=scale_factor)

    upscaled_path = os.path.join(output_dir, f"{job_id}_upscaled.png")
    upscaled.save(upscaled_path)
    print(f"🚀 Upscaled image saved to {upscaled_path}")

    return os.path.abspath(upscaled_path)


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
        "sample_steps": 30,
        "cfg_scale": 5,
        "strength": 0.65,
        "output_dir": "./outputs",
        "negative_prompt": "lowres, blurry, bad anatomy, extra limbs",
        "init_imagege": img_b64,
        "sampler": "DPMSolverMultistep",
    }

    out = run_inference_sdxl_i2i(job_id, prompt, params)
    print("✅ Img2Img result saved at:", out)
