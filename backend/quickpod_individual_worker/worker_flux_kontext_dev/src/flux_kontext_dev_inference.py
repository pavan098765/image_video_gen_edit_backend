# flux_kontext_dev_inference.py
# ✅ Lazy Loading for model and pipeline
# ✅ GPU memory cleanup before/after inference (OOM prevention)
# Dynamic resolution (keeps aspect ratio, multiple-of-64), seed, CFG, steps

import os
import base64
import gc
from io import BytesIO
from typing import Dict, Any
from PIL import Image
import httpx
import torch

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
# Lazy Pipeline Global
# -------------------------
_pipeline = None
_transformer = None


def get_pipeline():
    """Lazy-initialize the pipeline only once to prevent duplicate GPU memory usage."""
    global _pipeline, _transformer
    if _pipeline is not None:
        return _pipeline, _transformer

    import time
    total_start = time.time()

    # CRITICAL: Clear GPU cache before loading to maximize free memory
    gc.collect()
    torch.cuda.empty_cache()

    print(f"🧠 Loading Nunchaku Transformer from {TRANSFORMER_PATH}...")
    transformer_start = time.time()
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(TRANSFORMER_PATH)
    transformer_time = time.time() - transformer_start
    _transformer = transformer
    print(f"✅ Transformer loaded in {transformer_time:.1f}s")

    print(f"🚀 Loading Flux-Kontext pipeline from {FLUX_LOCAL_PATH}...")
    pipeline_start = time.time()
    pipe = FluxKontextPipeline.from_pretrained(
        FLUX_LOCAL_PATH,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    pipeline_time = time.time() - pipeline_start

    _pipeline = pipe
    total_time = time.time() - total_start
    print(f"✅ Pipeline initialized in {total_time:.1f}s (transformer: {transformer_time:.1f}s, pipeline: {pipeline_time:.1f}s)")
    return _pipeline, _transformer


def unload_models():
    """Frees up GPU memory by unloading models from cache."""
    global _pipeline, _transformer
    _pipeline = None
    _transformer = None
    gc.collect()
    torch.cuda.empty_cache()
    print("🧹 Model cache cleared and GPU memory freed.")


# -------------------------
# LoRA Setup (optional)
# -------------------------
def maybe_apply_lora(transformer, params: Dict[str, Any]):
    """Enable LoRA only if params contain lora_path (and optionally lora_strength)."""
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

    pipeline, transformer = get_pipeline()
    maybe_apply_lora(transformer, params)

    # CRITICAL: Clear GPU memory before inference to maximize available VRAM
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Log memory state before inference
    try:
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        print(f"🔍 GPU memory before inference: {free_mem / 1024**3:.2f} GB free")
    except Exception:
        pass

    print(f"⚙️ Running pipeline at {width}x{height} | Steps={steps} | CFG={cfg}")

    try:
        with torch.inference_mode():
            output = pipeline(
                image=image,
                prompt=prompt,
                guidance_scale=cfg,
                num_inference_steps=steps,
                width=width,
                height=height
            ).images[0]
    finally:
        # CRITICAL: Clean up GPU memory after inference
        gc.collect()
        torch.cuda.empty_cache()

    out_path = os.path.join(output_dir, f"flux_kontext_{job_id}.png")
    output.save(out_path)
    print(f"🖼️ Saved generated image to {out_path}")

    # -------------------------
    # Optional Upscaling and compulsory metadata removal
    # -------------------------
    scale_image = params.get("scale_image", True)
    scale_factor = params.get("scale_factor", 2) if scale_image else 0
    model_name = "RealESRGAN_x2plus" if scale_factor == 2 else "RealESRGAN_x4plus"

    upscaled = upscale_image(out_path, model_name=model_name, scale_factor=scale_factor)

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
