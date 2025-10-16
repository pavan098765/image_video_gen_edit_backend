# inpaint_refactor.py
# Stable Diffusion 1.5 Inpainting (GGUF backend)

import os
os.environ["HF_HOME"] = "/models_storage/all_models"
os.environ["HF_HUB_CACHE"] = "/models_storage/all_models"
import base64
from io import BytesIO
from typing import Any, Dict, Optional, Union
from PIL import Image, ImageOps
import numpy as np

from stable_diffusion_cpp import StableDiffusion
from huggingface_hub import hf_hub_download

# -------------------------
# Environment setup
# -------------------------

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
def _download_models():
    """Download required GGUF + VAE models into MODEL_DIR."""
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
    """Lazy init Stable Diffusion pipeline."""
    global STABLE_DIFFUSION
    if STABLE_DIFFUSION is not None:
        return STABLE_DIFFUSION

    model_paths = _download_models()
    if model_paths.get("VAE_PATH"):
        STABLE_DIFFUSION = StableDiffusion(
            model_path=model_paths["DIFFUSION_MODEL"],
            vae_path=model_paths["VAE_PATH"],
        )
    else:
        STABLE_DIFFUSION = StableDiffusion(model_path=model_paths["DIFFUSION_MODEL"])

    print("[Init] Stable Diffusion 1.5 Inpaint pipeline loaded")
    return STABLE_DIFFUSION


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


def _process_mask(
    mask: Image.Image,
    target_size: Optional[tuple] = None,
    binarize: bool = True,
    threshold: int = 127,
    invert: bool = False,
    resize_nearest: bool = False,
    debug_save_path: Optional[str] = None,
) -> Image.Image:
    """Prepare binary inpainting mask for StableDiffusion."""
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
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        mask.save(debug_save_path)
    return mask


# -------------------------
# Main inference
# -------------------------
def run_inference_sd15_inpaint(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run SD 1.5 Inpainting and save output.
    Accepts init_image and mask_image as base64 strings, paths, or PIL.Image.
    """
    pipe = _get_pipeline()

    init_image = _load_image(params.get("init_image") or params.get("images"))
    if init_image is None:
        raise ValueError("init_image is required for inpainting")

    raw_mask = _load_image(params.get("mask_image"))

    width, height = _get_dynamic_size(init_image, max_dim=1152)

    # Process mask if given
    mask_image = None
    if raw_mask is not None:
        mask_image = _process_mask(
            raw_mask,
            target_size=(width, height),
            binarize=bool(params.get("binarize_mask", True)),
            threshold=int(params.get("mask_threshold", 127)),
            invert=bool(params.get("invert_mask", False)),
            resize_nearest=True,
            debug_save_path=(
                os.path.join(params.get("output_dir", "./outputs"),
                             f"debug_{job_id}_processed_mask.png")
                if params.get("debug_save_mask", False) else None
            ),
        )
        arr = np.array(mask_image)
        white = int((arr >= 128).sum())
        total = arr.size
        print(f"[Mask diagnostics] white pixels (to be inpainted): {white}/{total} "
              f"({white/total*100:.2f}%) ; invert={params.get('invert_mask', False)}")

        if white == 0:
            raise ValueError("Processed mask has zero white pixels. "
                             "Check invert_mask/binarize settings.")

    # Params
    strength = float(params.get("strength", 0.4))
    steps = int(params.get("sample_steps", 30))
    cfg = float(params.get("cfg_scale", 7.5))
    method = params.get("sample_method", "euler")
    negative = params.get("negative_prompt")
    seed = int(params.get("seed", -1))
    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, f"sd15_inpaint_{job_id}.png")

    # Run inference
    out = pipe.generate_image(
        prompt=prompt,
        negative_prompt=negative,
        init_image=init_image,
        mask_image=mask_image,
        strength=strength,
        width=width,
        height=height,
        sample_steps=steps,
        cfg_scale=cfg,
        sample_method=method,
        seed=seed,
    )

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
    job_id = "test_inpaint"
    prompt = "make the indian flag into usa flag"

    input_path = "./input.png"
    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    mask_path = "./mask.png"
    with open(mask_path, "rb") as f:
        mask_b64 = base64.b64encode(f.read()).decode("utf-8")

    params = {
        "sample_steps": 30,
        "cfg_scale": 5,
        "sample_method": "dpmpp2mv2",
        "seed": -1,
        "output_dir": "./outputs",
        "init_image": img_b64,
        "mask_image": mask_b64,
        "negative_prompt": (
            "lowres, bad anatomy, error body, error hands, error fingers, "
            "error legs, error arms, disfigured, deformed, blurry, fuzzy, out of focus, "
            "extra limbs, cloned face, gross proportions, malformed limbs, missing arms, "
            "missing legs, extra arms, extra legs, mutated hands and fingers, poorly drawn hands, "
            "fused fingers, too many fingers, long neck"
        ),
    }

    out = run_inference_sd15_inpaint(job_id, prompt, params)
    print("✅ Output image saved at:", out)
