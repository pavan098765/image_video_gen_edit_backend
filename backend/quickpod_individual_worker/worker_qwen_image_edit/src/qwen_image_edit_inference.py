# qwen_image_edit_diffusers.py
# ✅ Lazy-loading + caching implementation
# Dynamic resolution (keeps aspect ratio, multiple-of-64)
# Fixed rank=128 and 8 steps (Lightning LoRA fused)

import os
import math
import base64
from io import BytesIO
from typing import Dict, Any
from PIL import Image
import httpx
import torch

from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline
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

# Build transformer file path
precision = get_precision()
transformer_file = f"svdq-{precision}_r{RANK}-qwen-image-edit-lightningv1.0-{NUM_INFERENCE_STEPS}steps.safetensors"
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

    print("🧩 Initializing Qwen scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    _model_cache["scheduler"] = scheduler
    return scheduler


def get_transformer():
    """Lazily load Nunchaku Qwen Transformer."""
    global _model_cache
    if _model_cache["transformer"] is not None:
        return _model_cache["transformer"]

    print("🧠 Loading Nunchaku Qwen Transformer model...")
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(TRANSFORMER_PATH)
    _model_cache["transformer"] = transformer
    print("✅ Transformer loaded and cached.")
    return transformer


def get_pipeline():
    """Lazily load Qwen Image Edit pipeline with proper scheduler and offload."""
    global _model_cache
    if _model_cache["pipeline"] is not None:
        return _model_cache["pipeline"]

    print("🚀 Loading Qwen Image Edit pipeline...")
    transformer = get_transformer()
    scheduler = get_scheduler()

    pipeline = QwenImageEditPipeline.from_pretrained(
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
def _load_image(img_b64: str) -> Image.Image:
    decoded_bytes = base64.b64decode(img_b64)
    return Image.open(BytesIO(decoded_bytes)).convert("RGB")


def _get_dynamic_size(img: Image.Image, max_dim: int = 1152) -> tuple[int, int]:
    """Return width/height close to input, within max_dim, multiple-of-64 (no upscaling)."""
    w, h = img.size
    aspect = w / h
    if max(w, h) > max_dim:
        if w > h:
            new_w = max_dim
            new_h = int(new_w / aspect)
        else:
            new_h = max_dim
            new_w = int(new_h * aspect)
    else:
        new_w, new_h = w, h
    new_w = max(64, (new_w // 64) * 64)
    new_h = max(64, (new_h // 64) * 64)
    return new_w, new_h


# -------------------------
# Main Inference Function
# -------------------------
def run_inference_qwen_image_edit(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run Qwen Image Edit pipeline (Diffusers) with dynamic resolution.
    Returns local path to upscaled PNG output.
    """
    print(f"🧩 Starting inference job: {job_id}")
    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    inference_steps = params.get("num_inference_steps", NUM_INFERENCE_STEPS)
    cfg = params.get("cfg_scale", 1.0)
    negative_prompt = params.get("negative_prompt", " ")

    init_images = params.get("init_image")
    if init_images is None:
        raise ValueError("init_images (base64) is required")

    image = _load_image(init_images)
    width, height = _get_dynamic_size(image, max_dim=1152)

    pipeline = get_pipeline()

    print(f"⚙️ Running Qwen Image Edit | Steps={inference_steps}, CFG={cfg}, Size={width}x{height}")

    output = pipeline(
        image=image,
        prompt=prompt,
        true_cfg_scale=cfg,
        negative_prompt=negative_prompt,
        num_inference_steps=inference_steps,
        width=width,
        height=height,
    ).images[0]

    out_path = os.path.join(output_dir, f"qwen_image_edit_{job_id}_4_step_on_8_step_lora.png")
    output.save(out_path)
    print(f"🖼️ Saved edited image to {out_path}")

    # -------------------------
    # Optional Upscaling + metadata removal
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
    job_id = "tara"
    prompt = "Change her body orientation so she is facing the camera directly."

    url = "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/Snapinsta.app_262322643_4937545659591345_7487236998017046274_n_1080.jpg"

    with httpx.Client(timeout=60) as c:
        img_bytes = c.get(url).content
    img_b64 = base64.b64encode(img_bytes).decode()

    params = {
        "cfg_scale": 1.0,
        "output_dir": "./outputs",
        "images": img_b64,
        "negative_prompt": " ",
        "num_inference_steps": 4,
    }

    out = run_inference_qwen_image_edit(job_id, prompt, params)
    print("✅ Output image saved at:", out)
