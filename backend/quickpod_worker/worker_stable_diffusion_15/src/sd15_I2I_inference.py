# sd_15_i2i_inference.py
# Stable Diffusion 1.5 Img2Img inference (GGUF backend)

import os
os.environ["HF_HOME"] = "/models_storage/all_models"
os.environ["HF_HUB_CACHE"] = "/models_storage/all_models"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import base64
from io import BytesIO
from typing import Dict, Any, Optional, Union
from PIL import Image

from huggingface_hub import hf_hub_download
from stable_diffusion_cpp import StableDiffusion

# -------------------------
# Environment setup
# -------------------------


MODEL_DIR = os.path.join("/models_storage", "all_models")
os.makedirs(MODEL_DIR, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "hf_xxx")  # set in env

# -------------------------
# Model definitions
# -------------------------
MODEL_FILES = {
    "DIFFUSION_MODEL": (
        "second-state/stable-diffusion-v1-5-GGUF",
        "stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf"
    ),
    "VAE_PATH": (
        "stabilityai/sd-vae-ft-mse-original",
        "vae-ft-mse-840000-ema-pruned.safetensors"
    ),
}

# Cache pipeline instance
STABLE_DIFFUSION: Optional[StableDiffusion] = None


# -------------------------
# Helpers
# -------------------------
def _get_dynamic_size(img: Image.Image, max_dim: int = 1152) -> tuple[int, int]:
    """Compute resolution keeping aspect ratio, multiple-of-64, within max_dim."""
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


def _download_models():
    """Download required GGUF + VAE into MODEL_DIR."""
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
    """Lazy init of Stable Diffusion model."""
    global STABLE_DIFFUSION
    if STABLE_DIFFUSION is not None:
        return STABLE_DIFFUSION

    model_paths = _download_models()
    STABLE_DIFFUSION = StableDiffusion(
        model_path=model_paths["DIFFUSION_MODEL"],
        vae_path=model_paths["VAE_PATH"],
    )
    print("[Init] Stable Diffusion 1.5 Img2Img pipeline loaded")
    return STABLE_DIFFUSION


# -------------------------
# Main inference
# -------------------------
def run_inference_sd15_i2i(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run SD 1.5 Img2Img inference and save PNG locally.
    Returns: path to PNG
    """
    pipe = _get_pipeline()

    init_image = _load_image(params.get("init_image") or params.get("images"))
    if init_image is None:
        raise ValueError("init_image is required for img2img")

    width, height = _get_dynamic_size(init_image, max_dim=1152)

    negative_prompt = params.get("negative_prompt")
    strength = float(params.get("strength", 0.65))
    steps = int(params.get("sample_steps", 30))
    cfg = float(params.get("cfg_scale", 7.5))
    method = params.get("sample_method", "euler")
    seed = int(params.get("seed", -1))
    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Run inference
    result = pipe.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        images=init_image,
        strength=strength,
        width=width,
        height=height,
        sample_steps=steps,
        cfg_scale=cfg,
        sample_method=method,
        seed=seed,
    )

    out_path = os.path.join(output_dir, f"sd15_i2i_{job_id}.png")

    if isinstance(result, list) and isinstance(result[0], Image.Image):
        result[0].save(out_path)
    elif isinstance(result, Image.Image):
        result.save(out_path)
    else:
        Image.fromarray(result).save(out_path)

    return out_path


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    job_id = "test2"
    prompt = "make the indian flag into usa flag"
    input_path = "./input.png"

    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    params = {
        "sample_steps": 30,
        "cfg_scale": 5,
        "sample_method": "dpmpp2mv2",
        "seed": -1,
        "output_dir": "./outputs",
        "strength": 0.4,
        "init_image": img_b64,
        "negative_prompt": (
            "lowres, bad anatomy, error body, error hands, error fingers, "
            "error legs, error arms, disfigured, deformed, blurry, fuzzy, out of focus, "
            "extra limbs, cloned face, gross proportions, malformed limbs, missing arms, "
            "missing legs, extra arms, extra legs, mutated hands and fingers, poorly drawn hands, "
            "fused fingers, too many fingers, long neck"
        ),
    }

    out = run_inference_sd15_i2i(job_id, prompt, params)
    print("✅ Output image saved at:", out)
