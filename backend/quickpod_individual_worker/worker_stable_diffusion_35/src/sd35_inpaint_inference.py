# Download the weights from the links below:

# Download sd3.5_large from https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/sd3.5_large.safetensors
# Download clip_g from https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/clip_g.safetensors
# Download clip_l from https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/clip_l.safetensors
# Download t5xxl from https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/t5xxl_fp16.safetensors

#sd35 inpaint_inference.py
import os
import base64
from io import BytesIO
from typing import Any, Dict, Optional, Union
from PIL import Image, ImageOps
import numpy as np
from stable_diffusion_cpp import StableDiffusion
from huggingface_hub import hf_hub_download

# Always use /quickpod/all_models inside cloud GPU
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

# -----------------------
# Initialize once
# -----------------------
sd = StableDiffusion(
    model_path=MODEL_PATH,
    clip_l_path=CLIP_L_PATH,
    clip_g_path=CLIP_G_PATH,
    t5xxl_path=T5XXL_PATH
)

# -----------------------
# Helper: load image from path, PIL.Image, or base64
# -----------------------
def _load_image(img: Optional[Union[str, Image.Image, bytes]]) -> Optional[Image.Image]:
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, bytes):
        return Image.open(BytesIO(img)).convert("RGB")
    # If string, could be file path
    if os.path.exists(img):
        return Image.open(img).convert("RGB")
    # Otherwise assume base64 string
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
    new_w = (new_w // 64) * 64
    new_h = (new_h // 64) * 64
    new_w = max(64, new_w)
    new_h = max(64, new_h)
    return new_w, new_h

def _process_mask_for_diffusers(
    mask: Image.Image,
    target_size: Optional[tuple] = None,
    binarize: bool = True,
    threshold: int = 127,
    invert: bool = False,
    resize_nearest: bool = False,
    debug_save_path: Optional[str] = None
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


# -----------------------
# Inference function
# -----------------------
def run_inference_sd35_inpaint(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run SD3.5 inpainting and save output.
    Accepts init_image and mask_image as base64 strings, paths, or PIL.Image.
    """
    init_image = _load_image(params.get("init_image"))
    raw_mask = _load_image(params.get("mask_image"))

    binarize = params.get("binarize_mask", True)
    invert_mask = params.get("invert_mask", False)
    mask_threshold = params.get("mask_threshold", 127)
    debug_save_mask = params.get("debug_save_mask", False)

    # Dynamic resolution
    width, height = _get_dynamic_size(init_image, max_dim=1152)
    # init_image = init_image.resize((width, height), resample=Image.LANCZOS)

    mask_image = None
    if raw_mask is not None:
        mask_image = _process_mask_for_diffusers(
            raw_mask,
            target_size=(width, height),
            binarize=binarize,
            threshold=mask_threshold,
            invert=invert_mask,
            resize_nearest=True,
            debug_save_path=(os.path.join(params.get("output_dir", "/quickpod"),
                                          f"debug_{job_id}_processed_mask.png")
                             if debug_save_mask else None)
        )
        arr = np.array(mask_image)
        white = int((arr >= 128).sum())
        total = arr.size
        print(f"[mask diagnostics] white pixels (to be inpainted): {white}/{total} "
              f"({white/total*100:.2f}%) ; invert_mask={invert_mask} ; binarize={binarize}")
        if white == 0:
            raise ValueError("Processed mask has zero white pixels (nothing to inpaint). "
                             "Check invert_mask/binarize settings and mask orientation.")

    strength = params.get("strength", 0.4)
    steps = params.get("sample_steps", 30)
    cfg = params.get("cfg_scale", 7.5)
    method = params.get("sample_method", "euler")
    negative = params.get("negative_prompt", None)
    seed = params.get("seed", -1)
    output_dir = params.get("output_dir", "/quickpod")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"sd3_5_inpaint_{job_id}.png")
    
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

    # Save defensively (list or single PIL.Image)
    img = out[0] if isinstance(out, list) else out
    if isinstance(img, Image.Image):
        img.save(out_path)
    else:
        Image.fromarray(img).save(out_path)

    return out_path


if __name__ == "__main__":
    import json
    
    job_id = "test2"
    prompt = "make the masked area yellow"
    input_path = "/quickpod/quickpod_workers/worker_stable_diffusion_35/input_image.jpg"

    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    mask_image_path = "/quickpod/quickpod_workers/worker_stable_diffusion_35/mask.png"

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
        "negative_prompt": "lowres, bad anatomy, error body, error hands, error fingers, error legs, error arms, disfigured, deformed, blurry, fuzzy, out of focus, extra limbs, cloned face, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands and fingers, poorly drawn hands and fingers, fused fingers, too many fingers, long neck",

    }

    out = run_inference_sd35_inpaint(job_id, prompt, params)
    print("✅ Output image saved at:", out)
