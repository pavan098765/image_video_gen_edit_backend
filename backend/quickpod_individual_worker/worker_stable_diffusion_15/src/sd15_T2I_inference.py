# inference_sd15_t2i.py

import os
from typing import Dict, Any
from PIL import Image
from stable_diffusion_cpp import StableDiffusion

# -------------------------
# Environment
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

for key, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required model file: {path}")
    else:
        print(f"✓ {key} found at {path}")

# -------------------------
# Initialize SD 1.5
# -------------------------
stable_diffusion = StableDiffusion(
    model_path=MODEL_PATHS["DIFFUSION_MODEL"],
    vae_path=MODEL_PATHS["VAE_PATH"],
    vae_decode_only=True,
)

# -------------------------
# Inference
# -------------------------
def run_inference_sd15_t2i(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    negative_prompt = params.get("negative_prompt", None)
    width = params.get("width", 512)
    height = params.get("height", 512)
    steps = params.get("sample_steps", 25)
    cfg = params.get("cfg_scale", 7.5)
    method = params.get("sample_method", "euler")
    seed = params.get("seed", -1)
    output_dir = params.get("output_dir", "./outputs")

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

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"sd_15_t2i_{job_id}.png")

    if isinstance(output, list) and isinstance(output[0], Image.Image):
        output[0].save(out_path)
    elif isinstance(output, Image.Image):
        output.save(out_path)
    else:
        Image.fromarray(output).save(out_path)

    return out_path

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
