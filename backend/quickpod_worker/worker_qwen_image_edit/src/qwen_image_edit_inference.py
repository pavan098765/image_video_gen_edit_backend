# qwen_image_edit_diffusers.py
# Usage: python qwen_image_edit_diffusers.py
# Dynamic resolution (keeps aspect ratio, multiple-of-64), fixed rank=128 and 8 steps (Lightning LoRA fused)

import os
os.environ["HF_HOME"] = "/models_storage/all_models/qwen_image_edit"
os.environ["HF_HUB_CACHE"] = "/models_storage/all_models/qwen_image_edit"

import math
import base64
from io import BytesIO
from typing import Dict, Any
from PIL import Image
import torch
torch.cuda.empty_cache()

from huggingface_hub import hf_hub_download
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline
from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

# -------------------------
# Paths / Model config
# -------------------------

MODEL_DIR = os.path.join("/models_storage", "all_models")
os.makedirs(MODEL_DIR, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "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")

# Fixed rank/steps (user requested)
NUM_INFERENCE_STEPS = 8
RANK = 128

MODEL_FILES = {
    "DIFFUSION_MODEL": (
        "nunchaku-tech/nunchaku-qwen-image-edit",
        # This is the Lightning-fused safetensors (contains Lightning LoRA fusion)
        f"svdq-{get_precision()}_r{RANK}-qwen-image-edit-lightningv1.0-{NUM_INFERENCE_STEPS}steps.safetensors"
    ),
}

def download_models():
    local_paths = {}
    for key, (repo_id, filename) in MODEL_FILES.items():
        print(f"[Download] {key} → {repo_id}/{filename}")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=HF_TOKEN,
            cache_dir=MODEL_DIR,
            resume_download=True
        )
        local_paths[key] = path
    return local_paths

MODEL_PATHS = download_models()
TRANSFORMER_PATH = MODEL_PATHS["DIFFUSION_MODEL"]

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
    "shift_terminal": None,  # set shift_terminal to None
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

# -------------------------
# Initialize Diffusers pipeline (transformer comes from the fused Lightning safetensors)
# -------------------------
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(TRANSFORMER_PATH)

pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    transformer=transformer,
    scheduler=scheduler,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN
)

# VRAM / offload handling (official pattern)
if get_gpu_memory() > 18:
    # Plenty of GPU memory: let the pipeline handle CPU offload when idle
    pipeline.enable_model_cpu_offload()
else:
    # Low VRAM: per-layer offload for transformer + sequential pipeline offload
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

    # If either side exceeds max_dim, downscale proportionally; otherwise keep original
    if max(w, h) > max_dim:
        if w > h:
            new_w = max_dim
            new_h = int(new_w / aspect)
        else:
            new_h = max_dim
            new_w = int(new_h * aspect)
    else:
        new_w, new_h = w, h

    # Round down to multiple-of-64
    new_w = max(64, (new_w // 64) * 64)
    new_h = max(64, (new_h // 64) * 64)
    return new_w, new_h

# -------------------------
# Main inference
# -------------------------
def run_inference_qwen_image_edit(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run Qwen Image Edit inference with dynamic resolution.
    Fixed rank=128 and 8 steps (Lightning LoRA fused).
    Returns the local path to the saved PNG file.
    """
    output_dir = params.get("output_dir", "./outputs")
    inference_steps = params.get("num_inference_steps", NUM_INFERENCE_STEPS)
    os.makedirs(output_dir, exist_ok=True)

    init_images = params.get("images")
    if init_images is None:
        raise ValueError("init_images (base64) is required")
    image = _load_image(init_images)

    # Decide output resolution (close to input, multiple-of-64)
    width, height = _get_dynamic_size(image, max_dim=1152)

    # Inference parameters (user-visible)
    cfg = params.get("cfg_scale", 1.0)
    negative_prompt = params.get("negative_prompt", " ")

    # Run pipeline (NUM_INFERENCE_STEPS is fixed to 8)
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
    prompt = "Change her body orientation so that she is facing the camera directly, with her upper body visible completely from the front.'"
    input_path = "/quickpod/quickpod_workers/worker_qwen_image_edit/tara.jpg"

    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    params = {
        "cfg_scale": 1.0,
        "output_dir": "./outputs",
        "images": img_b64,
        "negative_prompt": " ",
        "num_inference_steps": 4   # User can request fewer steps than the fixed 8-step LoRA
    }

    out = run_inference_qwen_image_edit(job_id, prompt, params)
    print("✅ Output image saved at:", out)
