# inference_sd15_t2i.py
# ✅ Lazy-loading implementation
# SD 1.5 Txt2Img using pre-downloaded GGUF + VAE (GPU-only, baked models)

import os
from typing import Dict, Any
from PIL import Image
import torch

from stable_diffusion_cpp import StableDiffusion
from upscaler import upscale_image

torch.cuda.empty_cache()

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

# Model component filenames
MODEL_FILES = {
    "DIFFUSION_MODEL": "stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf",
    "VAE_PATH": "vae-ft-mse-840000-ema-pruned.safetensors",
}

# Resolve to local paths
MODEL_PATHS = {
    "DIFFUSION_MODEL": os.path.join(SD15_LOCAL_PATH, MODEL_FILES["DIFFUSION_MODEL"]),
    "VAE_PATH": os.path.join(SD15_VAE_PATH, MODEL_FILES["VAE_PATH"]),
}

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
    "stable_diffusion_t2i": None
}

def get_stable_diffusion_t2i() -> StableDiffusion:
    """
    Lazily load the Stable Diffusion 1.5 Txt2Img model (GGUF + VAE).
    Cached globally for reuse across inference calls.
    """
    global _model_cache
    if _model_cache["stable_diffusion_t2i"] is not None:
        return _model_cache["stable_diffusion_t2i"]

    print("🧠 Loading Stable Diffusion 1.5 Txt2Img model (GGUF + VAE)...")
    sd = StableDiffusion(
        model_path=MODEL_PATHS["DIFFUSION_MODEL"],
        vae_path=MODEL_PATHS["VAE_PATH"],
        vae_decode_only=True,
    )
    _model_cache["stable_diffusion_t2i"] = sd
    print("✅ Stable Diffusion 1.5 Txt2Img model loaded and cached.")
    return sd


def unload_sd15_t2i():
    """
    Optional cleanup: unload Txt2Img model and free VRAM/CPU cache.
    """
    global _model_cache
    _model_cache["stable_diffusion_t2i"] = None
    torch.cuda.empty_cache()
    print("🧹 Stable Diffusion Txt2Img model cache cleared and GPU memory freed.")


# -------------------------
# Main Inference Function
# -------------------------
def run_inference_sd15_t2i(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run SD 1.5 Txt2Img inference and save result locally.
    Returns the path to the saved PNG file.
    """
    print(f"🧩 Starting SD 1.5 Txt2Img job: {job_id}")
    stable_diffusion = get_stable_diffusion_t2i()

    negative_prompt = params.get("negative_prompt", None)
    width = int(params.get("width", 512))
    height = int(params.get("height", 512))
    steps = int(params.get("sample_steps", 25))
    cfg = float(params.get("cfg_scale", 7.5))
    method = params.get("sample_method", "euler")
    seed = int(params.get("seed", -1))
    output_dir = params.get("output_dir", "./outputs")

    os.makedirs(output_dir, exist_ok=True)

    print(f"⚙️ Running Txt2Img | Steps={steps}, CFG={cfg}, Size={width}x{height}, Seed={seed}, Sampler={method}")

    output = stable_diffusion.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        sample_steps=steps,
        cfg_scale=cfg,
        sample_method=method,
        seed=seed,
    )

    out_path = os.path.join(output_dir, f"sd_15_t2i_{job_id}.png")

    if isinstance(output, list) and isinstance(output[0], Image.Image):
        output[0].save(out_path)
    elif isinstance(output, Image.Image):
        output.save(out_path)
    else:
        Image.fromarray(output).save(out_path)

    print(f"🖼️ Base generated image saved to {out_path}")

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
# CLI test
# -------------------------
if __name__ == "__main__":
    job_id = "test2"
    prompt = "a beautiful indian girl, high quality, detailed, intricate, digital art"

    params = {
        "width": 768,
        "height": 1024,
        "sample_steps": 30,
        "cfg_scale": 5,
        "sample_method": "dpmpp2mv2",
        "seed": -1,
        "output_dir": "./outputs",
        "negative_prompt": (
            "lowres, bad anatomy, error body, error hands, error fingers, error legs, "
            "error arms, disfigured, deformed, blurry, fuzzy, out of focus, extra limbs, "
            "cloned face, gross proportions, malformed limbs, missing arms, missing legs, "
            "extra arms, extra legs, mutated hands and fingers, poorly drawn hands and fingers, "
            "fused fingers, too many fingers, long neck"
        ),
    }

    out = run_inference_sd15_t2i(job_id, prompt, params)
    print("✅ Output image saved at:", out)
