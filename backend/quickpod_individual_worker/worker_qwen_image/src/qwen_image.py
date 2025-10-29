# qwen_image_diffusers.py
# ✅ Qwen Image (Lightning LoRA fused) Lazy Inference
# - Based on official Nunchaku repo logic (FlowMatchEulerDiscreteScheduler + QwenImagePipeline)
# - Keeps lazy loading, caching, offload, and upscaler integration

import os
import math
import base64
from io import BytesIO
from typing import Dict, Any
from PIL import Image
import httpx
import torch

from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline
from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision
from upscaler import upscale_image

torch.cuda.empty_cache()

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
QWEN_LOCAL_PATH = os.environ.get("QWEN_LOCAL_PATH")
NUNCHAKU_QWEN_LOCAL_PATH = os.environ.get("NUNCHAKU_QWEN_LOCAL_PATH")

if not QWEN_LOCAL_PATH or not os.path.exists(QWEN_LOCAL_PATH):
    raise RuntimeError("❌ Qwen repo not found. Ensure it is baked and available at QWEN_LOCAL_PATH.")
if not NUNCHAKU_QWEN_LOCAL_PATH or not os.path.exists(NUNCHAKU_QWEN_LOCAL_PATH):
    raise RuntimeError("❌ Nunchaku repo not found. Ensure it is baked and available at NUNCHAKU_QWEN_LOCAL_PATH.")

# Fixed rank/steps (Lightning fused)
NUM_INFERENCE_STEPS = 8
RANK = 128

precision = get_precision()
transformer_file = f"svdq-{precision}_r{RANK}-qwen-image-lightningv1.1-{NUM_INFERENCE_STEPS}steps.safetensors"
TRANSFORMER_PATH = os.path.join(NUNCHAKU_QWEN_LOCAL_PATH, transformer_file)

if not os.path.exists(TRANSFORMER_PATH):
    raise FileNotFoundError(f"❌ Transformer file not found at {TRANSFORMER_PATH}")

# -------------------------
# Scheduler (official config)
# -------------------------
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

# -------------------------
# Lazy Loader Cache
# -------------------------
_model_cache = {
    "transformer": None,
    "pipeline": None,
    "scheduler": None
}

# -------------------------
# Lazy Load Functions
# -------------------------
def get_scheduler():
    """Return cached FlowMatchEulerDiscreteScheduler."""
    global _model_cache
    if _model_cache["scheduler"] is not None:
        return _model_cache["scheduler"]

    print("🧩 Initializing Qwen Image scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    _model_cache["scheduler"] = scheduler
    return scheduler


def get_transformer():
    """Lazily load Nunchaku Qwen Image Transformer."""
    global _model_cache
    if _model_cache["transformer"] is not None:
        return _model_cache["transformer"]

    print("🧠 Loading Nunchaku Qwen Image Transformer model...")
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(TRANSFORMER_PATH)
    _model_cache["transformer"] = transformer
    print("✅ Transformer loaded and cached.")
    return transformer


def get_pipeline():
    """Lazily load Qwen Image pipeline with proper scheduler and offload."""
    global _model_cache
    if _model_cache["pipeline"] is not None:
        return _model_cache["pipeline"]

    print("🚀 Loading Qwen Image pipeline...")
    transformer = get_transformer()
    scheduler = get_scheduler()

    pipeline = QwenImagePipeline.from_pretrained(
        QWEN_LOCAL_PATH,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
    )

    gpu_mem = get_gpu_memory()
    print(f"💾 GPU memory detected: {gpu_mem} GB")

    if gpu_mem > 18:
        print("⚙️ Enabling model CPU offload...")
        pipeline.enable_model_cpu_offload()
    else:
        print("⚙️ Using sequential CPU offload (low VRAM mode)...")
        transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
        pipeline._exclude_from_cpu_offload.append("transformer")
        pipeline.enable_sequential_cpu_offload()

    _model_cache["pipeline"] = pipeline
    print("✅ Pipeline loaded and cached.")
    return pipeline


def unload_models():
    """Manually clear all cached models and free VRAM."""
    global _model_cache
    _model_cache["transformer"] = None
    _model_cache["pipeline"] = None
    _model_cache["scheduler"] = None
    torch.cuda.empty_cache()
    print("🧹 Cleared Qwen models and freed GPU memory.")


# -------------------------
# Helpers
# -------------------------
def _get_dynamic_size(max_dim: int = 1024) -> tuple[int, int]:
    """Return a dynamic width/height within max_dim, multiple-of-64 (square default)."""
    size = max_dim
    size = (size // 64) * 64
    return size, size


# -------------------------
# Main Inference Function
# -------------------------
def run_inference_qwen_image(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run Qwen Image pipeline (Diffusers) using the official Nunchaku config.
    Returns local path to upscaled PNG output.
    """
    print(f"🧩 Starting Qwen Image inference job: {job_id}")
    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    inference_steps = params.get("num_inference_steps", NUM_INFERENCE_STEPS)
    cfg = params.get("cfg_scale", 1.0)
    width, height = _get_dynamic_size(1024)

    pipeline = get_pipeline()

    print(f"⚙️ Running Qwen Image | Steps={inference_steps}, CFG={cfg}, Size={width}x{height}")
    print(f"🪶 Prompt: {prompt}")

    output = pipeline(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=inference_steps,
        true_cfg_scale=cfg,
    ).images[0]

    out_path = os.path.join(output_dir, f"qwen_image_{job_id}_lightning.png")
    output.save(out_path)
    print(f"🖼️ Saved generated image to {out_path}")

    # -------------------------
    # Optional Upscaling
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
    job_id = "bookstore_test"
    prompt = """Bookstore window display. A sign displays “New Arrivals This Week”.
    Below, a shelf tag with the text “Best-Selling Novels Here”. To the side,
    a colorful poster advertises “Author Meet And Greet on Saturday” with a
    central portrait of the author. There are four books on the bookshelf,
    namely “The light between worlds” “When stars are scattered” “The silent patient”
    “The night circus”."""

    params = {
        "cfg_scale": 1.0,
        "output_dir": "./outputs",
        "num_inference_steps": 8,
        "scale_image": True,
        "scale_factor": 2,
    }

    out = run_inference_qwen_image(job_id, prompt, params)
    print("✅ Output image saved at:", out)
