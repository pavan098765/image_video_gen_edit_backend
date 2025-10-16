# sd35_t2i_inference.py
# ✅ Lazy-loading implementation
# SD 3.5 Text-to-Image using pre-downloaded GGUF + text encoders (GPU-only, baked models)

import os
from typing import Dict, Any
from PIL import Image
import torch

from stable_diffusion_cpp import StableDiffusion
from upscaler import upscale_image

torch.cuda.empty_cache()

# -------------------------
# Model Paths
# -------------------------
MODEL_DIR = "/models/sd35"

SD35_LOCAL_PATH = os.environ.get(
    "SD35_LOCAL_PATH",
    os.path.join(MODEL_DIR, "hf_models", "second-state_stable-diffusion-3.5-large-GGUF"),
)

# Model component filenames (assumed baked into container)
MODEL_FILES = {
    "DIFFUSION_MODEL": "sd3.5_large-Q8_0.gguf",
    "CLIP_L_PATH": "clip_l.safetensors",
    "CLIP_G_PATH": "clip_g-Q8_0.gguf",
    "T5XXL_PATH": "t5xxl-Q8_0.gguf",
}

MODEL_PATHS = {k: os.path.join(SD35_LOCAL_PATH, v) for k, v in MODEL_FILES.items()}

# Validate files exist
for key, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Missing required model file: {path}")
    else:
        print(f"✓ {key} found at {path}")

# -------------------------
# Lazy Loader Cache
# -------------------------
_model_cache = {
    "sd35_t2i": None
}

def get_sd35_t2i() -> StableDiffusion:
    """
    Lazily load the Stable Diffusion 3.5 Text-to-Image model.
    Cached globally for reuse across multiple inference calls.
    """
    global _model_cache
    if _model_cache["sd35_t2i"] is not None:
        return _model_cache["sd35_t2i"]

    print("🧠 Loading Stable Diffusion 3.5 T2I model (GGUF + CLIP + T5)...")
    sd = StableDiffusion(
        model_path=MODEL_PATHS["DIFFUSION_MODEL"],
        clip_l_path=MODEL_PATHS["CLIP_L_PATH"],
        clip_g_path=MODEL_PATHS["CLIP_G_PATH"],
        t5xxl_path=MODEL_PATHS["T5XXL_PATH"],
        vae_decode_only=True,  # safe since it's T2I
        keep_vae_on_cpu=True,
        keep_clip_on_cpu=True
    )
    _model_cache["sd35_t2i"] = sd
    print("✅ Stable Diffusion 3.5 T2I model loaded and cached.")
    return sd


def unload_sd35_t2i():
    """
    Optional cleanup: unload the T2I model and free GPU/CPU memory.
    """
    global _model_cache
    _model_cache["sd35_t2i"] = None
    torch.cuda.empty_cache()
    print("🧹 SD3.5 T2I model cache cleared and GPU memory freed.")


# -------------------------
# Inference
# -------------------------
def run_inference_sd35_t2i(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run SD 3.5 Text-to-Image inference and save output.
    Returns the absolute path of the saved upscaled PNG file.
    """
    print(f"🧩 Starting SD 3.5 Txt2Img job: {job_id}")
    stable_diffusion = get_sd35_t2i()

    negative_prompt = params.get("negative_prompt", "lowres, bad anatomy, error, cropped, worst quality")
    width = int(params.get("width", 512))
    height = int(params.get("height", 512))
    steps = int(params.get("sample_steps", 25))
    cfg = float(params.get("cfg_scale", 4.5))
    method = params.get("sample_method", "euler")
    seed = int(params.get("seed", -1))
    output_dir = params.get("output_dir", "./outputs")

    os.makedirs(output_dir, exist_ok=True)
    print(f"⚙️ Generating Image | Steps={steps}, CFG={cfg}, Size={width}x{height}, Method={method}, Seed={seed}")

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

    out_path = os.path.join(output_dir, f"sd35_t2i_{job_id}.png")

    # Save defensively
    if isinstance(output, list) and isinstance(output[0], Image.Image):
        output[0].save(out_path)
    elif isinstance(output, Image.Image):
        output.save(out_path)
    else:
        Image.fromarray(output).save(out_path)

    print(f"🖼️ Base generated image saved to {out_path}")

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
    job_id = "test_t2i"
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

    out = run_inference_sd35_t2i(job_id, prompt, params)
    print("✅ Output image saved at:", out)
