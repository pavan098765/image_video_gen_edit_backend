# inpaint_refactor.py
# ✅ Lazy-loading implementation with GPU memory optimizations
# SD 1.5 Inpaint using pre-downloaded GGUF + VAE (GPU-only, baked models)

import gc
import os
import base64
import numpy as np
from io import BytesIO
from typing import Any, Dict, Optional, Union
from PIL import Image, ImageOps
import torch

from stable_diffusion_cpp import StableDiffusion
from upscaler import upscale_image

# Aggressive GPU cleanup at import to prevent OOM
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# -------------------------
# Environment / Paths
# -------------------------
MODEL_DIR = "/models/sd15"

SD15_LOCAL_PATH = os.environ.get(
    "SD15_LOCAL_PATH",
    os.path.join(MODEL_DIR, "hf_models", "second-state_stable-diffusion-v1-5-GGUF"),
)
SD15_VAE_PATH = os.environ.get(
    "SD15_VAE_PATH",
    os.path.join(MODEL_DIR, "hf_models", "stabilityai_sd-vae-ft-mse-original"),
)

MODEL_FILES = {
    "DIFFUSION_MODEL": "stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf",
    "VAE_PATH": "vae-ft-mse-840000-ema-pruned.safetensors",
}

MODEL_PATHS = {
    "DIFFUSION_MODEL": os.path.join(SD15_LOCAL_PATH, MODEL_FILES["DIFFUSION_MODEL"]),
    "VAE_PATH": os.path.join(SD15_VAE_PATH, MODEL_FILES["VAE_PATH"]),
}

# Validate model files exist (fail fast)
for key, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Missing required model file: {path}")
    else:
        print(f"✓ {key} found at {path}")

# -------------------------
# Lazy Loader Cache
# -------------------------
_model_cache = {
    "sd_inpaint": None
}

def get_sd15_inpaint() -> StableDiffusion:
    """
    Lazily load the Stable Diffusion 1.5 Inpaint model.
    Cached globally for reuse.
    """
    global _model_cache
    if _model_cache["sd_inpaint"] is not None:
        return _model_cache["sd_inpaint"]

    print("🧠 Loading Stable Diffusion 1.5 Inpaint model (GGUF + VAE)...")
    sd = StableDiffusion(
        model_path=MODEL_PATHS["DIFFUSION_MODEL"],
        vae_path=MODEL_PATHS["VAE_PATH"],
    )
    _model_cache["sd_inpaint"] = sd
    print("✅ Stable Diffusion 1.5 Inpaint model loaded and cached.")
    return sd


def unload_sd15_inpaint():
    """
    Optional cleanup: unload the inpaint model and free VRAM/CPU cache.
    """
    global _model_cache
    _model_cache["sd_inpaint"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("🧹 Stable Diffusion Inpaint model cache cleared and GPU memory freed.")


# -------------------------
# Helpers
# -------------------------
def _load_image(img: Optional[Union[str, Image.Image, bytes]]) -> Optional[Image.Image]:
    """Load image from PIL.Image, bytes, path, or base64 string. Returns RGB PIL.Image."""
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


def _process_mask_for_diffusers(
    mask: Optional[Image.Image],
    target_size: Optional[tuple] = None,
    binarize: bool = True,
    threshold: int = 127,
    invert: bool = False,
    resize_nearest: bool = False,
    debug_save_path: Optional[str] = None,
) -> Optional[Image.Image]:
    """
    Prepare a mask for diffusers-style inpainting.
    Converts to L mode, resizes, binarizes, optionally inverts, and can save debug copy.
    """
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
            pass  # non-fatal debug save failure

    return mask


# -------------------------
# Inference
# -------------------------
def run_inference_sd15_inpaint(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run SD1.5 inpainting and save output.
    Returns the absolute path of the final upscaled PNG.
    """
    print(f"🧩 Starting SD 1.5 Inpaint job: {job_id}")
    
    # Pre-inference GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    sd = get_sd15_inpaint()

    init_image = _load_image(params.get("init_image"))
    raw_mask = _load_image(params.get("mask_image"))
    if init_image is None:
        raise ValueError("init_image is required for inpainting")

    binarize = bool(params.get("binarize_mask", True))
    invert_mask = bool(params.get("invert_mask", False))
    mask_threshold = int(params.get("mask_threshold", 127))
    debug_save_mask = bool(params.get("debug_save_mask", False))

    width, height = _get_dynamic_size(init_image, max_dim=int(params.get("max_dim", 1152)))

    # Process mask if provided
    mask_image = None
    if raw_mask is not None:
        mask_image = _process_mask_for_diffusers(
            raw_mask,
            target_size=(width, height),
            binarize=binarize,
            threshold=mask_threshold,
            invert=invert_mask,
            resize_nearest=True,
            debug_save_path=(
                os.path.join(params.get("output_dir", "./outputs"),
                             f"debug_{job_id}_processed_mask.png")
                if debug_save_mask else None
            ),
        )

        arr = np.array(mask_image)
        white = int((arr >= 128).sum())
        total = arr.size
        print(f"[mask diagnostics] white pixels (to be inpainted): {white}/{total} "
              f"({white/total*100:.2f}%) ; invert_mask={invert_mask} ; binarize={binarize}")
        if white == 0:
            raise ValueError("Processed mask has zero white pixels (nothing to inpaint).")

    # Inference parameters
    strength = float(params.get("strength", 0.4))
    steps = int(params.get("sample_steps", 30))
    cfg = float(params.get("cfg_scale", 7.5))
    method = params.get("sample_method", "euler")
    negative = params.get("negative_prompt", None)
    seed = int(params.get("seed", -1))
    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    print(f"⚙️ Inpainting | Steps={steps}, CFG={cfg}, Strength={strength}, Method={method}")

    with torch.inference_mode():
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

    out_path = os.path.join(output_dir, f"sd_15_inpaint_{job_id}.png")
    img = out[0] if isinstance(out, list) else out
    if isinstance(img, Image.Image):
        img.save(out_path)
    else:
        Image.fromarray(img).save(out_path)

    # Post-inference GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print(f"🖼️ Inpainted base image saved to {out_path}")

    # -------------------------
    # Optional Upscaling + Metadata Removal
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
    job_id = "test_inpaint"
    prompt = "Make the Indian flag into the USA flag"
    input_path = "./input.png"
    mask_path = "./mask.png"

    if not os.path.exists(input_path) or not os.path.exists(mask_path):
        raise SystemExit("❌ Please provide input.png and mask.png to test this script.")

    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
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
            "lowres, bad anatomy, error body, error hands, error fingers, error legs, "
            "error arms, disfigured, deformed, blurry, fuzzy, out of focus, extra limbs, "
            "cloned face, gross proportions, malformed limbs, missing arms, missing legs, "
            "extra arms, extra legs, mutated hands and fingers, poorly drawn hands and fingers, "
            "fused fingers, too many fingers, long neck"
        ),
    }

    out = run_inference_sd15_inpaint(job_id, prompt, params)
    print("✅ Output image saved at:", out)
