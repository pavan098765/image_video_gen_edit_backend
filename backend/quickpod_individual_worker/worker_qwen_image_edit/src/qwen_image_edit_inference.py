# qwen_image_edit_diffusers.py
# Usage: python qwen_image_edit_diffusers.py
# Dynamic resolution (keeps aspect ratio, multiple-of-64), fixed rank=128 and 8 steps (Lightning LoRA fused)

import os
import math
import base64
from io import BytesIO
from typing import Dict, Any
from PIL import Image
import torch
torch.cuda.empty_cache()

from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline
from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

# -------------------------
# Paths / Model config
# -------------------------
QWEN_LOCAL_PATH = os.environ.get("QWEN_LOCAL_PATH")
NUNCHAKU_QWEN_LOCAL_PATH = os.environ.get("NUNCHAKU_QWEN_LOCAL_PATH")

if not QWEN_LOCAL_PATH or not os.path.exists(QWEN_LOCAL_PATH):
    raise RuntimeError("Qwen repo not found. Did worker_qwen.sh download it?")
if not NUNCHAKU_QWEN_LOCAL_PATH or not os.path.exists(NUNCHAKU_QWEN_LOCAL_PATH):
    raise RuntimeError("Nunchaku repo not found. Did worker_qwen.sh download it?")

# Fixed rank/steps (Lightning fused)
NUM_INFERENCE_STEPS = 8
RANK = 128

# Build transformer file path
precision = get_precision()
transformer_file = f"svdq-{precision}_r{RANK}-qwen-image-edit-lightningv1.0-{NUM_INFERENCE_STEPS}steps.safetensors"
TRANSFORMER_PATH = os.path.join(NUNCHAKU_QWEN_LOCAL_PATH, transformer_file)

if not os.path.exists(TRANSFORMER_PATH):
    raise FileNotFoundError(f"Transformer file not found at {TRANSFORMER_PATH}")

# -------------------------
# Scheduler (official config)
# -------------------------
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),  # We use shift=3 in distillation
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),  # We use shift=3 in distillation
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
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

# -------------------------
# Initialize Diffusers pipeline
# -------------------------
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(TRANSFORMER_PATH)

pipeline = QwenImageEditPipeline.from_pretrained(
    QWEN_LOCAL_PATH,
    transformer=transformer,
    scheduler=scheduler,
    torch_dtype=torch.bfloat16
)

# VRAM / offload handling
if get_gpu_memory() > 18:
    pipeline.enable_model_cpu_offload()
else:
    transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
    pipeline._exclude_from_cpu_offload.append("transformer")
    pipeline.enable_sequential_cpu_offload()

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
# Main inference
# -------------------------
def run_inference_qwen_image_edit(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    output_dir = params.get("output_dir", "./outputs")
    inference_steps = params.get("num_inference_steps", NUM_INFERENCE_STEPS)
    os.makedirs(output_dir, exist_ok=True)

    init_images = params.get("images")
    if init_images is None:
        raise ValueError("init_images (base64) is required")
    image = _load_image(init_images)

    width, height = _get_dynamic_size(image, max_dim=1152)

    cfg = params.get("cfg_scale", 1.0)
    negative_prompt = params.get("negative_prompt", " ")

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
    return out_path

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    job_id = "tara"
    prompt = "Change her body orientation so she is facing the camera directly."
    input_path = "/quickpod/quickpod_workers/worker_qwen_image_edit/tara.jpg"

    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    params = {
        "cfg_scale": 1.0,
        "output_dir": "./outputs",
        "images": img_b64,
        "negative_prompt": " ",
        "num_inference_steps": 4
    }

    out = run_inference_qwen_image_edit(job_id, prompt, params)
    print("✅ Output image saved at:", out)
