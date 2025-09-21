# Download the weights from the links below:

# Download sd3.5_large from https://huggingface.co/city96/stable-diffusion-3.5-large-gguf/blob/main/sd3.5_large-Q5_0.gguf
# Download clip_g from https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/clip_g.safetensors
# Download clip_l from https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/clip_l.safetensors
# Download t5xxl from https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/t5xxl_fp16.safetensors


import os
import uuid
from typing import Dict, Any, Optional
from PIL import Image
from stable_diffusion_cpp import StableDiffusion

MODEL_DIR = os.environ.get("HF_HOME", "/models_storage/all_models")
SD35_LOCAL_PATH = os.environ.get(
    "SD35_LOCAL_PATH",
    os.path.join(MODEL_DIR, "hf_models", "second-state_stable-diffusion-3.5-large-GGUF")
)

# Define models
MODEL_FILES = {
    "DIFFUSION_MODEL": "sd3.5_large-Q8_0.gguf",
    "CLIP_L_PATH": "clip_l.safetensors",
    "CLIP_G_PATH": "clip_g-Q8_0.gguf",
    "T5XXL_PATH": "t5xxl-Q8_0.gguf"
}

MODEL_PATHS = {k: os.path.join(SD35_LOCAL_PATH, v) for k, v in MODEL_FILES.items()}

for key, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required model file: {path}")
    else:
        print(f"✓ {key} found at {path}")


# Environment-default model paths (override via env)
MODEL_PATH = MODEL_PATHS["DIFFUSION_MODEL"]
CLIP_L_PATH   = MODEL_PATHS["CLIP_L_PATH"]
CLIP_G_PATH   = MODEL_PATHS["CLIP_G_PATH"]
T5XXL_PATH   = MODEL_PATHS["T5XXL_PATH"]

stable_diffusion = StableDiffusion(
    model_path=MODEL_PATH,
    clip_l_path=CLIP_L_PATH,
    clip_g_path=CLIP_G_PATH,
    t5xxl_path=T5XXL_PATH,
    vae_decode_only=True, # Can be True if we are not generating image to image
)


def run_inference_sd35_t2i(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run SD 3.5 Txt2Img inference and save the result locally.
    Returns the path to the saved PNG file.
    """
    negative_prompt = params.get("negative_prompt", None)
    width           = params.get("width", 512)
    height          = params.get("height", 512)
    steps           = params.get("sample_steps", 25)
    cfg             = params.get("cfg_scale", 4.5)
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
    out_path = os.path.join(output_dir, f"sd_35_t2i_{job_id}.png")

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
        "output_dir": "./outputs",
        "negative_prompt": "lowres, bad anatomy, error body, error hands, error fingers, error legs, error arms, disfigured, deformed, blurry, fuzzy, out of focus, extra limbs, cloned face, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands and fingers, poorly drawn hands and fingers, fused fingers, too many fingers, long neck",

    }

    out = run_inference_sd35_t2i(job_id, prompt, params)
    print("✅ Output image saved at:", out)
