# flux_kontext_dev_inference.py
# Dynamic resolution (keeps aspect ratio, multiple-of-64), seed, CFG, steps
# ✅ Now with Lazy Loading for model and pipeline

import os
import base64
from io import BytesIO
from typing import Dict, Any
from PIL import Image
import httpx
import torch

torch.cuda.empty_cache()

from diffusers import FluxKontextPipeline
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

from upscaler import upscale_image

# --- Auto-merge split safetensors if merge.py exists ---
try:
    from merge import main as merge_main
    print("🔍 Scanning for .safetensors.part-* files...")
    merge_main()
except Exception as e:
    print(f"⚠️ merge.py auto-merge skipped or failed: {e}")

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
# Lazy loader cache
# -------------------------
_model_cache = {
    "transformer": None,
    "pipeline": None
}

# -------------------------
# Lazy load functions
# -------------------------
def get_transformer():
    """
    Lazily load and return the Nunchaku transformer.
    Cached after first load.
    """
    global _model_cache
    if _model_cache["transformer"] is not None:
        return _model_cache["transformer"]

    print("🧠 Loading Nunchaku Transformer...")
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(TRANSFORMER_PATH)
    _model_cache["transformer"] = transformer
    print("✅ Transformer loaded and cached.")
    return transformer


def get_pipeline():
    """
    Lazily load and return the Flux-Kontext Diffusers pipeline.
    Cached after first load.
    """
    global _model_cache
    if _model_cache["pipeline"] is not None:
        return _model_cache["pipeline"]

    transformer = get_transformer()
    print("🚀 Loading Flux-Kontext pipeline...")
    pipe = FluxKontextPipeline.from_pretrained(
        FLUX_LOCAL_PATH,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    _model_cache["pipeline"] = pipe
    print("✅ Pipeline loaded and cached.")
    return pipe


def unload_models():
    """
    Frees up GPU memory by unloading models from cache (optional manual call).
    """
    global _model_cache
    _model_cache["transformer"] = None
    _model_cache["pipeline"] = None
    torch.cuda.empty_cache()
    print("🧹 Model cache cleared and GPU memory freed.")


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
    print(f"🧩 Starting inference job: {job_id}")
    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    init_image = params.get("init_image")
    if init_image is None:
        raise ValueError("init_image (base64) is required")

    image = _load_image(init_image)
    width, height = _get_dynamic_size(image, max_dim=1152)

    steps = params.get("sample_steps", 28)
    cfg = params.get("cfg_scale", 1.5)

    transformer = get_transformer()
    maybe_apply_lora(transformer, params)
    pipeline = get_pipeline()

    print(f"⚙️ Running pipeline at {width}x{height} | Steps={steps} | CFG={cfg}")

    output = pipeline(
        image=image,
        prompt=prompt,
        guidance_scale=cfg,
        num_inference_steps=steps,
        width=width,
        height=height
    ).images[0]

    out_path = os.path.join(output_dir, f"flux_kontext_{job_id}.png")
    output.save(out_path)
    print(f"🖼️ Saved generated image to {out_path}")

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
# Example usage
# -------------------------
if __name__ == "__main__":
    job_id = "flux_test_001"
    prompt = "Change her body orientation so she is facing the camera directly."
    url = "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/Snapinsta.app_262322643_4937545659591345_7487236998017046274_n_1080.jpg"

    with httpx.Client(timeout=60) as c:
        img_bytes = c.get(url).content
    img_b64 = base64.b64encode(img_bytes).decode()


    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

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
