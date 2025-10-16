# qwen_image_edit_diffusers.py
# Usage: python qwen_image_edit_diffusers.py
# Dynamic resolution (keeps aspect ratio, multiple-of-64), fixed rank=128 and 8 steps (Lightning LoRA fused)

import os
import math
import base64
from io import BytesIO
from typing import Dict, Any, Optional, Union
from PIL import Image
import torch
torch.cuda.empty_cache()

from huggingface_hub import hf_hub_download
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline
from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

# -------------------------
# Config / paths
# -------------------------
os.environ["HF_HOME"] = "/models_storage/all_models"
os.environ["HF_HUB_CACHE"] = "/models_storage/all_models"

MODEL_DIR = os.path.join("/models_storage", "all_models")
os.makedirs(MODEL_DIR, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "hf_xxx")  # replace with your token

# Fixed rank/steps (Lightning LoRA fused model)
NUM_INFERENCE_STEPS = 8
RANK = 128

MODEL_FILES = {
    "DIFFUSION_MODEL": (
        "nunchaku-tech/nunchaku-qwen-image-edit",
        f"svdq-{get_precision()}_r{RANK}-qwen-image-edit-lightningv1.0-{NUM_INFERENCE_STEPS}steps.safetensors"
    ),
}

# Cache loaded pipeline to avoid re-init
PIPELINE_CACHE: Optional[QwenImageEditPipeline] = None


# -------------------------
# Model Download
# -------------------------
def _download_models():
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


# -------------------------
# Scheduler (official config)
# -------------------------
def _make_scheduler():
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
    return FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)


# -------------------------
# Pipeline Init (lazy)
# -------------------------
def _get_pipeline() -> QwenImageEditPipeline:
    global PIPELINE_CACHE
    if PIPELINE_CACHE is not None:
        return PIPELINE_CACHE

    model_paths = _download_models()
    transformer_path = model_paths["DIFFUSION_MODEL"]

    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(transformer_path)
    scheduler = _make_scheduler()

    pipe = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )

    # VRAM / offload handling
    if get_gpu_memory() > 18:
        pipe.enable_model_cpu_offload()
    else:
        transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
        pipe._exclude_from_cpu_offload.append("transformer")
        pipe.enable_sequential_cpu_offload()

    PIPELINE_CACHE = pipe
    print("[Init] Qwen Image Edit pipeline loaded")
    return PIPELINE_CACHE


# -------------------------
# Helpers
# -------------------------
def _load_image(img: Union[str, bytes]) -> Image.Image:
    """Decode base64 or path → PIL.Image."""
    if os.path.exists(str(img)):
        return Image.open(str(img)).convert("RGB")
    if isinstance(img, bytes):
        return Image.open(BytesIO(img)).convert("RGB")
    decoded_bytes = base64.b64decode(img)
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

    # Round down to multiple-of-64
    new_w = max(64, (new_w // 64) * 64)
    new_h = max(64, (new_h // 64) * 64)
    return new_w, new_h


# -------------------------
# Main Inference
# -------------------------
def run_inference_qwen_image_edit(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Run Qwen Image Edit inference with dynamic resolution.
    Fixed rank=128 and 8 steps (Lightning LoRA fused).
    Returns path to saved PNG.
    """
    pipe = _get_pipeline()

    output_dir = params.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    init_image = params.get("init_image") or params.get("images")
    if init_image is None:
        raise ValueError("init_image (base64 or path) is required")

    image = _load_image(init_image)
    width, height = _get_dynamic_size(image, max_dim=1152)

    steps = params.get("num_inference_steps", NUM_INFERENCE_STEPS)
    cfg = params.get("cfg_scale", 1.0)
    negative_prompt = params.get("negative_prompt", " ")

    output = pipe(
        image=image,
        prompt=prompt,
        true_cfg_scale=cfg,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        width=width,
        height=height,
    ).images[0]

    out_path = os.path.join(output_dir, f"qwen_image_edit_{job_id}.png")
    output.save(out_path)
    return out_path


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    job_id = "tara"
    prompt = "Change her body orientation so she faces the camera frontally."

    input_path = "./tara.jpg"
    with open(input_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    params = {
        "output_dir": "./outputs",
        "init_image": img_b64,
        "cfg_scale": 1.0,
        "negative_prompt": " ",
        "num_inference_steps": 4,  # user can override fewer than 8
    }

    out = run_inference_qwen_image_edit(job_id, prompt, params)
    print("✅ Output saved at:", out)
