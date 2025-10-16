# sd35_t2i_inference.py
# Stable Diffusion 3.5 Txt2Img (GGUF backend)

import os
import base64
from typing import Dict, Any, Optional
from PIL import Image
from stable_diffusion_cpp import StableDiffusion
from huggingface_hub import hf_hub_download

# -------------------------
# Environment setup
# -------------------------
os.environ["HF_HOME"] = "/models_storage/all_models"
os.environ["HF_HUB_CACHE"] = "/models_storage/all_models"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_DIR = os.path.join("/models_storage", "all_models")
os.makedirs(MODEL_DIR, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "hf_xxx")

# -------------------------
# Model definitions
# -------------------------
MODEL_FILES = {
    "DIFFUSION_MODEL": (
        "city96/stable-diffusion-3.5-large-gguf",
        "sd3.5_large-Q5_0.gguf",
    ),
    "CLIP_L_PATH": (
        "Comfy-Org/stable-diffusion-3.5-fp8",
        "text_encoders/clip_l.safetensors",
    ),
    "CLIP_G_PATH": (
        "Comfy-Org/stable-diffusion-3.5-fp8",
        "text_encoders/clip_g.safetensors",
    ),
    "T5XXL_PATH": (
        "Comfy-Org/stable-diffusion-3.5-fp8",
        "text_encoders/t5xxl_fp16.safetensors",
    ),
}

STABLE_DIFFUSION: Optional[StableDiffusion] = None


# -------------------------
# Helpers
# -------------------------
def _download_models():
    """Download all required model files into MODEL_DIR."""
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
    """Lazy init Stable Diffusion 3.5 pipeline (cached)."""
    global STABLE_DIFFUSION
    if STABLE_DIFFUSION is not None:
        return STABLE_DIFFUSION

    model_paths = _download_models()
    STABLE_DIFFUSION = StableDiffusion(
        model_path=model_paths["DIFFUSION_MODEL"],
        clip_l_path=model_paths["CLIP_L_PATH"],
        clip_g_path=model_paths["CLIP_G_PATH"],
        t5xxl_path=model_paths["T5XXL_PATH"],
        vae_decode_only=True,  # safe for Txt2Img
    )
    print("[Init] Stable Diffusion 3.5 Txt2Img pipeline loaded")
    return STABLE_DIFFUSION


# -------------------------
# Inference
# -------------------------
def run_inference_sd35_t2i(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run SD 3.5 Txt2Img inference and save the result locally.
    Returns the path to the saved PNG file.
    """
    pipe = _get_pipeline()

    negative_prompt = params.get("negative_prompt")
    width = int(params.get("width", 512))
    height = int(params.get("height", 512))
    steps = int(params.get("sample_steps", 25))
    cfg = float(params.get("cfg_scale", 4.5))
    method = params.get("sample_method", "euler")
    seed = int(params.get("seed", -1))
    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

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

    out_path = os.path.join(output_dir, f"sd35_t2i_{job_id}.png")
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
            "lowres, bad anatomy, error body, error hands, error fingers, "
            "disfigured, deformed, blurry, cloned face, malformed limbs"
        ),
    }

    out = run_inference_sd35_t2i(job_id, prompt, params)
    print("✅ Output image saved at:", out)
