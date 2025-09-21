# flux_kontext_diffusers.py
# Usage: python flux_kontext_diffusers.py
# Dynamic resolution (keeps aspect ratio, multiple-of-64), seed, CFG, steps

import os
import base64
from io import BytesIO
from typing import Dict, Any
from PIL import Image
import torch
torch.cuda.empty_cache()

from diffusers import FluxKontextPipeline
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

# -------------------------
# Paths / Model config
# -------------------------
FLUX_LOCAL_PATH = os.environ.get("FLUX_KONTEXT_LOCAL_PATH")
NUNCHAKU_LOCAL_PATH = os.environ.get("NUNCHAKU_LOCAL_PATH")

if not FLUX_LOCAL_PATH or not os.path.exists(FLUX_LOCAL_PATH):
    raise RuntimeError("Flux-Kontext repo not found. Did worker_flux.sh download it?")
if not NUNCHAKU_LOCAL_PATH or not os.path.exists(NUNCHAKU_LOCAL_PATH):
    raise RuntimeError("Nunchaku repo not found. Did worker_flux.sh download it?")

# Pick correct transformer file from Nunchaku
precision = get_precision()
transformer_file = f"svdq-{precision}_r32-flux.1-kontext-dev.safetensors"
TRANSFORMER_PATH = os.path.join(NUNCHAKU_LOCAL_PATH, transformer_file)

if not os.path.exists(TRANSFORMER_PATH):
    raise FileNotFoundError(f"Transformer file not found at {TRANSFORMER_PATH}")

# -------------------------
# Initialize Diffusers pipeline
# -------------------------
transformer = NunchakuFluxTransformer2dModel.from_pretrained(TRANSFORMER_PATH)

pipeline = FluxKontextPipeline.from_pretrained(
    FLUX_LOCAL_PATH,
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")

# -------------------------
# LoRA Setup (optional)
# -------------------------
def maybe_apply_lora(transformer, params: Dict[str, Any]):
    """
    Enable LoRA only if params contain lora_path (and optionally lora_strength).
    """
    apply_lora = params.get("apply_lora", False)
    if not apply_lora:
        print("[LoRA] Skipped (apply_lora is False)")
        return
    lora_path = "/models_storage/all_models/nunchaku-flux_kontext_nsfw_lora/clothes_remover_v0_flux_kontext.safetensors"

    if lora_path and os.path.exists(lora_path):
        print(f"[LoRA] Loading LoRA from {lora_path}")
        transformer.update_lora_params(lora_path)
        lora_strength = params.get("lora_strength", 1.0)
        transformer.set_lora_strength(float(lora_strength))
        print(f"[LoRA] Strength set to {lora_strength}")
    else:
        print("[LoRA] Skipped (LoRA file not found)")

# -------------------------
# Helpers
# -------------------------
def _load_image(img_b64: str) -> Image.Image:
    decoded_bytes = base64.b64decode(img_b64)
    return Image.open(BytesIO(decoded_bytes)).convert("RGB")

def _get_dynamic_size(img: Image.Image, max_dim: int = 1152) -> tuple[int, int]:
    """Return width/height close to input, within max_dim, multiple-of-64."""
    w, h = img.size
    aspect = w / h
    if w > h:
        new_w = min(w, max_dim)
        new_h = int(new_w / aspect)
    else:
        new_h = min(h, max_dim)
        new_w = int(new_h * aspect)
    # Round to multiple-of-64
    new_w = max(64, (new_w // 64) * 64)
    new_h = max(64, (new_h // 64) * 64)
    return new_w, new_h

# -------------------------
# Main inference
# -------------------------
def run_inference_flux_kontext_dev(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run Flux-Kontext inference (image editing) with dynamic resolution using Diffusers.
    Returns the local path to the saved PNG file.
    """
    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    init_images = params.get("images")
    if init_images is None:
        raise ValueError("init_images (base64) is required")
    image = _load_image(init_images)

    # Decide output resolution (close to input, multiple-of-64)
    width, height = _get_dynamic_size(image, max_dim=1152)

    # Inference parameters
    steps = params.get("sample_steps", 28)
    cfg = params.get("cfg_scale", 1.5)

    # Apply LoRA if requested
    maybe_apply_lora(transformer, params)

    # Run Diffusers pipeline
    output = pipeline(
        image=image,          # keep original image as input
        prompt=prompt,
        guidance_scale=cfg,
        num_inference_steps=steps,
        width=width,
        height=height
    ).images[0]

    out_path = os.path.join(output_dir, f"flux_kontext_{job_id}.png")
    output.save(out_path)
    return out_path

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    job_id = "bikini_dynamic"
    prompt = "A cinematic portrait with altered style"
    input_path = "/quickpod/quickpod_workers/worker_flux_kontext_dev/input_image.jpg"

    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    params = {
        "sample_steps": 30,
        "cfg_scale": 2.5,
        "output_dir": "./outputs",
        "images": img_b64,
        "apply_lora": True,
        "lora_strength": 1.0
    }

    out = run_inference_flux_kontext_dev(job_id, prompt, params)
    print("✅ Output image saved at:", out)
