# sd_15_i2i_inference.py
# ✅ Lazy-loading implementation with GPU memory optimizations
# SD 1.5 Img2Img using pre-downloaded GGUF + VAE

import base64
import gc
from io import BytesIO
import os
from typing import Dict, Any, Optional, Union
from PIL import Image
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

# Validate model files
for key, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Missing required model file: {path}")
    else:
        print(f"✓ {key} found at {path}")

# -------------------------
# Lazy Loader Cache
# -------------------------
_model_cache = {
    "stable_diffusion": None
}


def get_stable_diffusion() -> StableDiffusion:
    """
    Lazily load the Stable Diffusion 1.5 model and cache it for reuse.
    """
    global _model_cache
    if _model_cache["stable_diffusion"] is not None:
        return _model_cache["stable_diffusion"]

    print("🧠 Loading Stable Diffusion 1.5 (GGUF + VAE)...")
    stable_diffusion = StableDiffusion(
        model_path=MODEL_PATHS["DIFFUSION_MODEL"],
        vae_path=MODEL_PATHS["VAE_PATH"],
    )
    _model_cache["stable_diffusion"] = stable_diffusion
    print("✅ Stable Diffusion 1.5 model loaded and cached.")
    return stable_diffusion


def unload_sd15():
    """
    Optional cleanup: unload SD15 model and free VRAM/CPU memory.
    """
    global _model_cache
    _model_cache["stable_diffusion"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("🧹 Stable Diffusion model cache cleared and GPU memory freed.")


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
# Main Inference Function
# -------------------------
def run_inference_sd15_i2i(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run SD 1.5 Img2Img inference and save the result locally.
    Returns the path to the saved PNG file.
    """
    print(f"🧩 Starting SD 1.5 Img2Img job: {job_id}")
    
    # Pre-inference GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    stable_diffusion = get_stable_diffusion()

    negative_prompt = params.get("negative_prompt", None)
    init_image = _load_image(params.get("init_image"))
    strength = float(params.get("strength", 0.65))
    steps = int(params.get("sample_steps", 30))
    cfg = float(params.get("cfg_scale", 7.5))
    method = params.get("sample_method", "euler")
    seed = int(params.get("seed", -1))
    output_dir = params.get("output_dir", "./outputs")

    if init_image is None:
        raise ValueError("init_image is required for img2img")

    width, height = _get_dynamic_size(init_image, max_dim=1152)
    os.makedirs(output_dir, exist_ok=True)

    print(f"⚙️ Running inference: Steps={steps}, CFG={cfg}, Strength={strength}, Method={method}")

    with torch.inference_mode():
        output = stable_diffusion.generate_image(
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

    out_path = os.path.join(output_dir, f"sd_15_i2i_{job_id}.png")

    if isinstance(output, list) and isinstance(output[0], Image.Image):
        output[0].save(out_path)
    elif isinstance(output, Image.Image):
        output.save(out_path)
    else:
        Image.fromarray(output).save(out_path)

    # Post-inference GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print(f"🖼️ Base output saved to {out_path}")

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
    job_id = "test2"
    prompt = "make the indian flag to usa flag"
    input_path = "./input.png"

    if not os.path.exists(input_path):
        raise SystemExit("❌ Please provide input.png to test this script.")

    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    params = {
        "sample_steps": 30,
        "cfg_scale": 5,
        "sample_method": "dpmpp2mv2",
        "seed": -1,
        "output_dir": "./outputs",
        "strength": 0.4,
        "images": img_b64,
        "negative_prompt": (
            "lowres, bad anatomy, error body, error hands, error fingers, error legs, "
            "error arms, disfigured, deformed, blurry, fuzzy, out of focus, extra limbs, "
            "cloned face, gross proportions, malformed limbs, missing arms, missing legs, "
            "extra arms, extra legs, mutated hands and fingers, poorly drawn hands and fingers, "
            "fused fingers, too many fingers, long neck"
        ),
    }

    out = run_inference_sd15_i2i(job_id, prompt, params)
    print("✅ Output image saved at:", out)
