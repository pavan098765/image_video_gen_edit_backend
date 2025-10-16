# sd15_t2i_inference.py
# Stable Diffusion 1.5 Text-to-Image (GGUF backend)

import os
os.environ["HF_HOME"] = "/models_storage/all_models"
os.environ["HF_HUB_CACHE"] = "/models_storage/all_models"
import base64
from typing import Dict, Any, Optional, Union
from PIL import Image
from stable_diffusion_cpp import StableDiffusion
from huggingface_hub import hf_hub_download

# -------------------------
# Environment setup
# -------------------------

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_DIR = os.path.join("/models_storage", "all_models")
os.makedirs(MODEL_DIR, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "hf_xxx")  # must be set in env

# -------------------------
# Model definitions
# -------------------------
MODEL_FILES = {
    "DIFFUSION_MODEL": (
        "second-state/stable-diffusion-v1-5-GGUF",
        "stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf",
    ),
    "VAE_PATH": (
        "stabilityai/sd-vae-ft-mse-original",
        "vae-ft-mse-840000-ema-pruned.safetensors",
    ),
}

# Cache pipeline
STABLE_DIFFUSION: Optional[StableDiffusion] = None


# -------------------------
# Helpers
# -------------------------
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
    """Lazy init Stable Diffusion pipeline."""
    global STABLE_DIFFUSION
    if STABLE_DIFFUSION is not None:
        return STABLE_DIFFUSION

    model_paths = _download_models()
    STABLE_DIFFUSION = StableDiffusion(
        model_path=model_paths["DIFFUSION_MODEL"],
        vae_path=model_paths["VAE_PATH"],
        vae_decode_only=True,  # for T2I, decode only
    )

    print("[Init] Stable Diffusion 1.5 T2I pipeline loaded")
    return STABLE_DIFFUSION


# -------------------------
# Main inference
# -------------------------
def run_inference_sd15_t2i(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run SD 1.5 Text-to-Image and save output.
    Returns local PNG path.
    """
    pipe = _get_pipeline()

    # Params
    negative_prompt = params.get("negative_prompt")
    width = int(params.get("width", 512))
    height = int(params.get("height", 512))
    steps = int(params.get("sample_steps", 25))
    cfg = float(params.get("cfg_scale", 7.5))
    method = params.get("sample_method", "euler")
    seed = int(params.get("seed", -1))
    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, f"sd15_t2i_{job_id}.png")

    # Run inference
    out = pipe.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        sample_steps=steps,
        cfg_scale=cfg,
        sample_method=method,
        seed=seed,
    )

    # Save result safely
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
            "extra arms, extra legs, mutated hands and fingers, poorly drawn hands, "
            "fused fingers, too many fingers, long neck"
        ),
    }

    out = run_inference_sd15_t2i(job_id, prompt, params)
    print("✅ Output image saved at:", out)
