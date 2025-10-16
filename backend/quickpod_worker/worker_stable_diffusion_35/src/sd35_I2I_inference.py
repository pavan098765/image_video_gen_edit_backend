# sd35_i2i_inference.py
# Stable Diffusion 3.5 Img2Img (GGUF backend)

import os
os.environ["HF_HOME"] = "/models_storage/all_models"
os.environ["HF_HUB_CACHE"] = "/models_storage/all_models"
import base64
from io import BytesIO
from typing import Any, Dict, Optional, Union
from PIL import Image
from stable_diffusion_cpp import StableDiffusion
from huggingface_hub import hf_hub_download

# -------------------------
# Environment setup
# -------------------------

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_DIR = os.path.join("/models_storage", "all_models")
os.makedirs(MODEL_DIR, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "hf_xxx")

# -------------------------
# Model definitions
# -------------------------
MODEL_FILES = {
    "DIFFUSION_MODEL": (
        "second-state/stable-diffusion-3.5-large-GGUF",
        "sd3.5_large-Q8_0.gguf",
    ),
    "CLIP_L_PATH": (
        "second-state/stable-diffusion-3.5-large-GGUF",
        "clip_l.safetensors",
    ),
    "CLIP_G_PATH": (
        "second-state/stable-diffusion-3.5-large-GGUF",
        "clip_g-Q8_0.gguf",
    ),
    "T5XXL_PATH": (
        "second-state/stable-diffusion-3.5-large-GGUF",
        "t5xxl-Q8_0.gguf",
    ),
}

# Cache pipeline
STABLE_DIFFUSION: Optional[StableDiffusion] = None


# -------------------------
# Helpers
# -------------------------
def _download_models():
    """Download required models into MODEL_DIR."""
    local_paths = {}
    for key, (repo_id, filename) in MODEL_FILES.items():
        print(f"[Download] {key} → {repo_id}/{filename}")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=HF_TOKEN,
            cache_dir=MODEL_DIR,
        )
        local_paths[key] = path
    return local_paths


def _get_pipeline() -> StableDiffusion:
    """Lazy init Stable Diffusion 3.5 pipeline."""
    global STABLE_DIFFUSION
    if STABLE_DIFFUSION is not None:
        return STABLE_DIFFUSION

    model_paths = _download_models()
    STABLE_DIFFUSION = StableDiffusion(
        model_path=model_paths["DIFFUSION_MODEL"],
        clip_l_path=model_paths["CLIP_L_PATH"],
        clip_g_path=model_paths["CLIP_G_PATH"],
        t5xxl_path=model_paths["T5XXL_PATH"],
    )
    print("[Init] Stable Diffusion 3.5 i2i pipeline loaded")
    return STABLE_DIFFUSION


def _get_dynamic_size(img: Image.Image, max_dim: int = 1152) -> tuple[int, int]:
    """Keep aspect ratio, round to multiple-of-64, max dimension capped."""
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


# -------------------------
# Main inference
# -------------------------
def run_inference_sd35_i2i(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run SD 3.5 Img2Img and save result locally.
    Returns PNG path.
    """
    pipe = _get_pipeline()

    negative_prompt = params.get("negative_prompt")
    strength = float(params.get("strength", 0.65))
    init_image = _load_image(params.get("images"))
    steps = int(params.get("sample_steps", 25))
    cfg = float(params.get("cfg_scale", 4.5))
    method = params.get("sample_method", "euler")
    seed = int(params.get("seed", -1))
    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    if init_image is None:
        raise ValueError("init_image is required for img2img")

    width, height = _get_dynamic_size(init_image, max_dim=1152)

    out = pipe.generate_image(
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

    out_path = os.path.join(output_dir, f"sd35_i2i_{job_id}.png")

    # Save robustly
    img = out[0] if isinstance(out, list) else out
    if isinstance(img, Image.Image):
        img.save(out_path)
    else:
        Image.fromarray(img).save(out_path)

    return out_path


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    job_id = "test2"
    prompt = "high quality, photorealistic, intricate details"

    image_input_path = "./input_image.jpg"
    with open(image_input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    params = {
        "sample_steps": 30,
        "cfg_scale": 5,
        "sample_method": "dpmpp2mv2",
        "seed": -1,
        "output_dir": "./outputs",
        "images": img_b64,
        "negative_prompt": (
            "lowres, bad anatomy, error body, error hands, error fingers, "
            "deformed, blurry, extra limbs, cloned face, malformed limbs"
        ),
    }

    out = run_inference_sd35_i2i(job_id, prompt, params)
    print("✅ Output image saved at:", out)
