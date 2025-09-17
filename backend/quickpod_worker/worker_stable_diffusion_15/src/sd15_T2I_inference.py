# SD 1.5 model https://huggingface.co/second-state/stable-diffusion-v1-5-GGUF/blob/main/stable-diffusion-v1-5-pruned-emaonly-Q5_1.gguf
# Download Vae : https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors

import os
os.environ["HF_HOME"] = "/models_storage/all_models/sd_15"
os.environ["HF_HUB_CACHE"] = "/models_storage/all_models/sd_15"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
from typing import Dict, Any
from PIL import Image
from stable_diffusion_cpp import StableDiffusion
from huggingface_hub import hf_hub_download

# Always use /quickpod/all_models inside cloud GPU
MODEL_DIR = os.path.join("/models_storage", "all_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Hugging Face token (must be set in env for gated models)
HF_TOKEN = f"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"

# Define models
MODEL_FILES = {
    "DIFFUSION_MODEL": ("second-state/stable-diffusion-v1-5-GGUF", "stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf"),
    "VAE_PATH": ("stabilityai/sd-vae-ft-mse-original", "vae-ft-mse-840000-ema-pruned.safetensors"),
}

def download_models():
    """Download required models into /quickpod/all_models using huggingface_hub"""
    local_paths = {}
    for key, (repo_id, filename) in MODEL_FILES.items():
        print(f"[Download] {key} → {repo_id}/{filename}")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=HF_TOKEN,
            cache_dir=MODEL_DIR
        )
        local_paths[key] = path
    return local_paths

# Ensure all models are available
MODEL_PATHS = download_models()

# Initialize once at import
stable_diffusion = StableDiffusion(
    model_path=MODEL_PATHS["DIFFUSION_MODEL"],
    vae_path=MODEL_PATHS["VAE_PATH"],
    vae_decode_only=True, # Can be True if we are not generating image to image
)

def run_inference_sd15_t2i(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run SD 1.5 Txt2Img inference and save the result locally.
    Returns the path to the saved PNG file.
    """
    negative_prompt = params.get("negative_prompt", None)
    width           = params.get("width", 512)
    height          = params.get("height", 512)
    steps           = params.get("sample_steps", 25)
    cfg             = params.get("cfg_scale", 7.5)
    method          = params.get("sample_method", "euler")
    seed            = params.get("seed", -1)
    output_dir      = params.get("output_dir", "/quickpod")

    # Run inference
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

    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"sd_15_t2i_{job_id}.png")

    if isinstance(output, list) and isinstance(output[0], Image.Image):
        output[0].save(out_path)
    elif isinstance(output, Image.Image):
        output.save(out_path)
    else:
        Image.fromarray(output).save(out_path)

    return out_path

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
        "output_dir": "/quickpod",
        "negative_prompt": "lowres, bad anatomy, error body, error hands, error fingers, error legs, error arms, disfigured, deformed, blurry, fuzzy, out of focus, extra limbs, cloned face, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands and fingers, poorly drawn hands and fingers, fused fingers, too many fingers, long neck",
    }

    out = run_inference_sd15_t2i(job_id, prompt, params)
    print("✅ Output image saved at:", out)
