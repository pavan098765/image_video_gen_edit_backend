# sd35_inpaint_inference.py
# ✅ Lazy-loading implementation
# SD 3.5 Img2Img using pre-downloaded GGUF + text encoders (GPU-only, baked models)

import os
import base64
from io import BytesIO
from typing import Any, Dict, Optional, Union
from PIL import Image
import torch

from stable_diffusion_cpp import StableDiffusion
from upscaler import upscale_image

torch.cuda.empty_cache()

# -------------------------
# Environment / Paths
# -------------------------
MODEL_DIR = "/models/sd35"

SD35_LOCAL_PATH = os.environ.get(
    "SD35_LOCAL_PATH",
    os.path.join(MODEL_DIR, "hf_models", "second-state_stable-diffusion-3.5-large-GGUF"),
)

# Model component filenames (baked in image)
MODEL_FILES = {
    "DIFFUSION_MODEL": "sd3.5_large-Q8_0.gguf",
    "CLIP_L_PATH": "clip_l.safetensors",
    "CLIP_G_PATH": "clip_g-Q8_0.gguf",
    "T5XXL_PATH": "t5xxl-Q8_0.gguf",
}

MODEL_PATHS = {k: os.path.join(SD35_LOCAL_PATH, v) for k, v in MODEL_FILES.items()}

# Validate model files exist
for key, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Missing required model file: {path}")
    else:
        print(f"✓ {key} found at {path}")

# -------------------------
# Lazy Loader Cache
# -------------------------
_model_cache = {
    "sd35": None
}

def get_sd35() -> StableDiffusion:
    """
    Lazily load the Stable Diffusion 3.5 model.
    Cached globally for reuse across Img2Img jobs.
    """
    global _model_cache
    if _model_cache["sd35"] is not None:
        return _model_cache["sd35"]

    print("🧠 Loading Stable Diffusion 3.5 model (GGUF + CLIP + T5)...")
    sd = StableDiffusion(
        model_path=MODEL_PATHS["DIFFUSION_MODEL"],
        clip_l_path=MODEL_PATHS["CLIP_L_PATH"],
        clip_g_path=MODEL_PATHS["CLIP_G_PATH"],
        t5xxl_path=MODEL_PATHS["T5XXL_PATH"],
        keep_vae_on_cpu=True,
        keep_clip_on_cpu=True
    )
    _model_cache["sd35"] = sd
    print("✅ Stable Diffusion 3.5 model loaded and cached.")
    return sd


def unload_sd35():
    """
    Optional cleanup: unload SD3.5 model and free GPU/CPU memory.
    """
    global _model_cache
    _model_cache["sd35"] = None
    torch.cuda.empty_cache()
    print("🧹 Stable Diffusion 3.5 model cache cleared and GPU memory freed.")


# -------------------------
# Helpers
# -------------------------
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


def _load_image(img: Optional[Union[str, Image.Image, bytes]]) -> Optional[Image.Image]:
    """Load image from path, PIL.Image, bytes, or base64 string."""
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, bytes):
        return Image.open(BytesIO(img)).convert("RGB")
    if os.path.exists(str(img)):
        return Image.open(str(img)).convert("RGB")
    try:
        decoded = base64.b64decode(img)
        return Image.open(BytesIO(decoded)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Unable to decode image: {e}")


# -------------------------
# Main Inference
# -------------------------
def run_inference_sd35_i2i(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run SD 3.5 Img2Img inference and save the result locally.
    Returns the path to the saved PNG file.
    """
    print(f"🧩 Starting SD 3.5 Img2Img job: {job_id}")
    sd = get_sd35()

    negative_prompt = params.get("negative_prompt", "lowres, bad anatomy, error, cropped, worst quality")
    strength = float(params.get("strength", 0.65))
    init_image = _load_image(params.get("init_image"))
    steps = int(params.get("sample_steps", 25))
    cfg = float(params.get("cfg_scale", 4.5))
    method = params.get("sample_method", "euler")
    seed = int(params.get("seed", -1))
    output_dir = params.get("output_dir", "./outputs")

    if init_image is None:
        raise ValueError("init_image is required for img2img")

    # Compute output resolution
    width, height = _get_dynamic_size(init_image, max_dim=1152)

    os.makedirs(output_dir, exist_ok=True)
    print(f"⚙️ Running SD3.5 Inference | Steps={steps}, CFG={cfg}, Strength={strength}, Method={method}")

    # Run inference
    output = sd.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        init_image=init_image,
        strength=strength,
        width=width,
        height=height,
        sample_steps=steps,
        cfg_scale=cfg,
        sample_method=method,
        seed=seed,
    )

    out_path = os.path.join(output_dir, f"sd_35_i2i_{job_id}.png")

    if isinstance(output, list) and isinstance(output[0], Image.Image):
        output[0].save(out_path)
    elif isinstance(output, Image.Image):
        output.save(out_path)
    else:
        Image.fromarray(output).save(out_path)

    print(f"🖼️ Base generated image saved to {out_path}")

    # -------------------------
    # Optional Upscaling and Metadata Removal
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
    job_id = "test_sd35_i2i"
    prompt = "A cinematic landscape with mountains at sunset, high detail, photorealistic"
    input_path = "./input_image.jpg"

    if not os.path.exists(input_path):
        raise SystemExit("❌ Please provide input_image.jpg to test this script.")

    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    params = {
        "sample_steps": 30,
        "cfg_scale": 5,
        "sample_method": "dpmpp2mv2",
        "seed": -1,
        "output_dir": "./outputs",
        "images": img_b64,
        "negative_prompt": (
            "lowres, bad anatomy, error body, blurry, fuzzy, out of focus, malformed, "
            "extra limbs, cloned face, gross proportions, deformed, disfigured"
        ),
    }

    out = run_inference_sd35_i2i(job_id, prompt, params)
    print("✅ Output image saved at:", out)
