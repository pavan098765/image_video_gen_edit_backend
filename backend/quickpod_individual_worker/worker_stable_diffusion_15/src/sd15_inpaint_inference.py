# inpaint_refactor.py
# SD 1.5 Inpaint using pre-downloaded GGUF + VAE

import os
import base64
import numpy as np
from io import BytesIO
from typing import Any, Dict, Optional, Union
from PIL import Image, ImageOps
from stable_diffusion_cpp import StableDiffusion

# -------------------------
# Environment / Paths
# -------------------------
MODEL_DIR = os.environ.get("HF_HOME", "/models_storage/all_models")

SD15_LOCAL_PATH = os.environ.get(
    "SD15_LOCAL_PATH",
    os.path.join(MODEL_DIR, "hf_models", "second-state_stable-diffusion-v1-5-GGUF")
)
SD15_VAE_PATH = os.environ.get(
    "SD15_VAE_PATH",
    os.path.join(MODEL_DIR, "hf_models", "stabilityai_sd-vae-ft-mse-original")
)

MODEL_FILES = {
    "DIFFUSION_MODEL": "stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf",
    "VAE_PATH": "vae-ft-mse-840000-ema-pruned.safetensors",
}

MODEL_PATHS = {
    "DIFFUSION_MODEL": os.path.join(SD15_LOCAL_PATH, MODEL_FILES["DIFFUSION_MODEL"]),
    "VAE_PATH": os.path.join(SD15_VAE_PATH, MODEL_FILES["VAE_PATH"]),
}

# Validate presence
for key, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required model file: {path}")
    else:
        print(f"✓ {key} found at {path}")

# -------------------------
# Init StableDiffusion
# -------------------------
sd = StableDiffusion(
    model_path=MODEL_PATHS["DIFFUSION_MODEL"],
    vae_path=MODEL_PATHS["VAE_PATH"],
)

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

def _process_mask_for_diffusers(
    mask: Image.Image,
    target_size: Optional[tuple] = None,
    binarize: bool = True,
    threshold: int = 127,
    invert: bool = False,
    resize_nearest: bool = False,
    debug_save_path: Optional[str] = None,
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

# -------------------------
# Inference
# -------------------------
def run_inference_sd15_inpaint(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run SD1.5 inpainting and save output.
    Accepts init_image and mask_image as base64 strings, paths, or PIL.Image.
    """
    init_image = _load_image(params.get("images"))
    raw_mask = _load_image(params.get("mask_image"))

    if init_image is None:
        raise ValueError("init_image is required for inpainting")

    binarize = params.get("binarize_mask", True)
    invert_mask = params.get("invert_mask", False)
    mask_threshold = params.get("mask_threshold", 127)
    debug_save_mask = params.get("debug_save_mask", False)

    # Dynamic resolution
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
            debug_save_path=(os.path.join(params.get("output_dir", "./outputs"),
                                          f"debug_{job_id}_processed_mask.png")
                             if debug_save_mask else None),
        )
        arr = np.array(mask_image)
        white = int((arr >= 128).sum())
        total = arr.size
        print(f"[mask diagnostics] white pixels (to be inpainted): {white}/{total} "
              f"({white/total*100:.2f}%) ; invert_mask={invert_mask} ; binarize={binarize}")
        if white == 0:
            raise ValueError("Processed mask has zero white pixels (nothing to inpaint).")

    strength = params.get("strength", 0.4)
    steps = params.get("sample_steps", 30)
    cfg = params.get("cfg_scale", 7.5)
    method = params.get("sample_method", "euler")
    negative = params.get("negative_prompt", None)
    seed = params.get("seed", -1)
    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"sd_15_inpaint_{job_id}.png")

    out = sd.generate_image(
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
# CLI Test
# -------------------------
if __name__ == "__main__":
    job_id = "test_inpaint"
    prompt = "make the indian flag to usa flag"

    input_path = "/quickpod/quickpod_workers/worker_sd15/input.png"
    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    mask_image_path = "/quickpod/quickpod_workers/worker_sd15/mask.png"
    with open(mask_image_path, "rb") as f:
        mask_b64 = base64.b64encode(f.read()).decode("utf-8")

    params = {
        "sample_steps": 30,
        "cfg_scale": 5,
        "sample_method": "dpmpp2mv2",
        "seed": -1,
        "output_dir": "./outputs",
        "images": img_b64,
        "mask_image": mask_b64,
        "negative_prompt": (
            "lowres, bad anatomy, error body, error hands, error fingers, error legs, "
            "error arms, disfigured, deformed, blurry, fuzzy, out of focus, extra limbs, "
            "cloned face, gross proportions, malformed limbs, missing arms, missing legs, "
            "extra arms, extra legs, mutated hands and fingers, poorly drawn hands and fingers, "
            "fused fingers, too many fingers, long neck"
        ),
    }

    out = run_inference_sd15_inpaint(job_id, prompt, params)
    print("✅ Output image saved at:", out)
