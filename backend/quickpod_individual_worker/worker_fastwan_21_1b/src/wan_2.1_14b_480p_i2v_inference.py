import os
os.environ["HF_HOME"] = "/models_storage/all_models/fastwan"
os.environ["HF_HUB_CACHE"] = "/models_storage/all_models/fastwan"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import uuid
import tempfile
import random
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, UniPCMultistepScheduler
from transformers import CLIPVisionModel
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download

# -----------------------------
# Model paths
# -----------------------------
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
LORA_REPO_ID = "Kijai/WanVideo_comfy"
LORA_FILENAME = "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"

# -----------------------------
# Load models & LoRA
# -----------------------------
image_encoder = CLIPVisionModel.from_pretrained(MODEL_ID, subfolder="image_encoder", torch_dtype=torch.float16)
vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float16)
pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID, vae=vae, image_encoder=image_encoder, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)

# Memory optimizations
pipe.enable_model_cpu_offload()  # offload layers to CPU to prevent OOM
pipe.enable_attention_slicing()
# pipe.enable_xformers_memory_efficient_attention()  # optional if xformers installed

# Load LoRA
causvid_path = hf_hub_download(repo_id=LORA_REPO_ID, filename=LORA_FILENAME)
pipe.load_lora_weights(causvid_path, adapter_name="causvid_lora")
pipe.set_adapters(["causvid_lora"], adapter_weights=[0.95])
pipe.fuse_lora()

# -----------------------------
# Constants
# -----------------------------
MOD_VALUE = 32
FIXED_FPS = 24
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 81
MAX_SEED = np.iinfo(np.int32).max

DEFAULT_H_SLIDER_VALUE = 512
DEFAULT_W_SLIDER_VALUE = 896
NEW_FORMULA_MAX_AREA = 480.0 * 832.0 

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards, watermark, text, signature"
)

# -----------------------------
# Utilities
# -----------------------------
def _calculate_new_dimensions_wan(pil_image, mod_val, calculation_max_area, min_h, max_h, min_w, max_w, default_h, default_w):
    orig_w, orig_h = pil_image.size
    if orig_w <= 0 or orig_h <= 0:
        return default_h, default_w
    aspect_ratio = orig_h / orig_w
    calc_h = round(np.sqrt(calculation_max_area * aspect_ratio))
    calc_w = round(np.sqrt(calculation_max_area / aspect_ratio))
    calc_h = max(mod_val, (calc_h // mod_val) * mod_val)
    calc_w = max(mod_val, (calc_w // mod_val) * mod_val)
    new_h = int(np.clip(calc_h, min_h, (max_h // mod_val) * mod_val))
    new_w = int(np.clip(calc_w, min_w, (max_w // mod_val) * mod_val))
    return new_h, new_w

# -----------------------------
# Inference function
# -----------------------------
def run_inference_wan2_1_i2v(job_id: str, input_image: Image.Image, prompt: str, params: Dict[str, Any]) -> str:
    """
    WAN2.1 I2V inference, HF Space style
    """
    steps = params.get("steps", 4)
    cfg = params.get("guidance_scale", 1.0)
    seed = params.get("seed", 42)
    randomize_seed = params.get("randomize_seed", False)
    height = params.get("height", DEFAULT_H_SLIDER_VALUE)
    width = params.get("width", DEFAULT_W_SLIDER_VALUE)
    negative_prompt = params.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    output_dir = params.get("output_dir", "./outputs")
    duration_seconds = params.get("duration_seconds", 2)

    # Resize to multiple of MOD_VALUE
    target_h = max(MOD_VALUE, (int(height) // MOD_VALUE) * MOD_VALUE)
    target_w = max(MOD_VALUE, (int(width) // MOD_VALUE) * MOD_VALUE)

    num_frames = np.clip(int(round(duration_seconds * FIXED_FPS)), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)

    resized_image = input_image.resize((target_w, target_h))

    with torch.inference_mode():
        output_frames_list = pipe(
            image=resized_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=target_h,
            width=target_w,
            num_frames=num_frames,
            guidance_scale=float(cfg),
            num_inference_steps=int(steps),
            generator=torch.Generator(device="cuda").manual_seed(current_seed)
        ).frames[0]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"wan21_i2v_{job_id}.mp4")
    export_to_video(output_frames_list, out_path, fps=FIXED_FPS)
    return out_path

# -----------------------------
# Main usage
# -----------------------------
if __name__ == "__main__":
    job_id = str(uuid.uuid4())
    input_image_path = "input_example.jpg"  # Replace with your image
    input_img = Image.open(input_image_path).convert("RGB")

    params = {
        "steps": 4,
        "guidance_scale": 1.0,
        "seed": 42,
        "randomize_seed": True,
        "height": 512,
        "width": 896,
        "duration_seconds": 3,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "output_dir": "./outputs"
    }

    output_video = run_inference_wan2_1_i2v(job_id, input_img, "Animate this image with cinematic motion, smooth animation", params)
    print(f"[WAN2.1 I2V] Saved video at: {output_video}")
