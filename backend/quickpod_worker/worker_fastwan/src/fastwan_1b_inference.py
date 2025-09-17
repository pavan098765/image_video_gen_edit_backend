import os

os.environ["HF_HOME"] = "/models_storage/all_models/fastwan"
os.environ["HF_HUB_CACHE"] = "/models_storage/all_models/fastwan"
import uuid
from typing import Dict, Any, Optional
from PIL import Image
import torch
import numpy as np
import random
from diffusers import (
    AutoencoderKLWan,
    WanPipeline,
    WanImageToVideoPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import export_to_video

# -----------------------------
# Environment
# -----------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------
# Model ID
# -----------------------------
MODEL_ID_1B = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
PIPELINE_CACHE = {}

# -----------------------------
# Constants
# -----------------------------
MOD_VALUE = 32
FIXED_FPS = 24
MIN_FRAMES_MODEL = 25
MAX_FRAMES_MODEL = 193
MAX_SEED = np.iinfo(np.int32).max

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards, watermark, text, signature"
)


# -----------------------------
# Helpers
# -----------------------------


def get_pipelines():
    if MODEL_ID_1B in PIPELINE_CACHE:
        return PIPELINE_CACHE[MODEL_ID_1B]

    print(f"[WAN 1B] Loading pipelines...")
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID_1B, subfolder="vae", torch_dtype=torch.float32
    )
    t2v_pipe = WanPipeline.from_pretrained(
        MODEL_ID_1B, vae=vae, torch_dtype=torch.bfloat16
    )
    i2v_pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID_1B, vae=vae, torch_dtype=torch.bfloat16
    )

    for pipe in [t2v_pipe, i2v_pipe]:
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=8.0
        )
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    PIPELINE_CACHE[MODEL_ID_1B] = {"t2v": t2v_pipe, "i2v": i2v_pipe}
    return PIPELINE_CACHE[MODEL_ID_1B]


# -----------------------------
# Inference
# -----------------------------
def run_inference_wan_1b_t2i(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    pipes = get_pipelines()
    t2v_pipe = pipes["t2v"]

    steps = params.get("steps", 4)
    cfg = params.get("guidance_scale", 0.0)
    seed = params.get("seed", 44)
    randomize_seed = params.get("randomize_seed", False)
    height = params.get("height", 896)
    width = params.get("width", 896)
    negative_prompt = params.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    output_dir = params.get("output_dir", "./outputs")

    target_h = max(MOD_VALUE, (int(height) // MOD_VALUE) * MOD_VALUE)
    target_w = max(MOD_VALUE, (int(width) // MOD_VALUE) * MOD_VALUE)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)

    with torch.inference_mode():
        frame = t2v_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=target_h,
            width=target_w,
            num_frames=1,
            guidance_scale=float(cfg),
            num_inference_steps=int(steps),
            generator=torch.Generator(device="cuda").manual_seed(current_seed),
        ).frames[0][0]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"wan_t2i_1B_{job_id}.png")
    if isinstance(frame, np.ndarray):
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        frame = frame.squeeze()
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        Image.fromarray(frame).save(out_path)
    elif isinstance(frame, Image.Image):
        frame.save(out_path)
    return out_path


def run_inference_wan_1b_t2v(
    job_id: str,
    prompt: str,
    params: Dict[str, Any],
    input_image: Optional[Image.Image] = None,
) -> str:
    pipes = get_pipelines()
    t2v_pipe = pipes["t2v"]
    i2v_pipe = pipes["i2v"]

    steps = params.get("steps", 4)
    cfg = params.get("guidance_scale", 0.0)
    seed = params.get("seed", 44)
    randomize_seed = params.get("randomize_seed", False)
    height = params.get("height", 896)
    width = params.get("width", 896)
    duration_seconds = params.get("duration_seconds", 2)
    negative_prompt = params.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    output_dir = params.get("output_dir", "./outputs")

    target_h = max(MOD_VALUE, (int(height) // MOD_VALUE) * MOD_VALUE)
    target_w = max(MOD_VALUE, (int(width) // MOD_VALUE) * MOD_VALUE)
    num_frames = np.clip(
        int(round(duration_seconds * FIXED_FPS)), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL
    )
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)

    with torch.inference_mode():
        frames_list = t2v_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=target_h,
            width=target_w,
            num_frames=num_frames,
            guidance_scale=float(cfg),
            num_inference_steps=int(steps),
            generator=torch.Generator(device="cuda").manual_seed(current_seed),
        ).frames[0]
    file_name = f"wan_t2v_1B_{job_id}.mp4"

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, file_name)
    export_to_video(frames_list, out_path, fps=FIXED_FPS)
    return out_path


# -----------------------------
# Main Example for WAN 1.3B
# -----------------------------
if __name__ == "__main__":
    job_id = str(uuid.uuid4())

    # -----------------------------
    # Example 1: T2I (Text-to-Image)
    # -----------------------------
    print("[*] Running WAN 1.3B T2I inference ...")
    t2i_params = {
        "steps": 10,
        "guidance_scale": 0.0,
        "seed": 42,
        "randomize_seed": True,
        "height": 896,
        "width": 896,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "output_dir": "./outputs",
    }
    t2i_path = run_inference_wan_1b_t2i(
        job_id,
        "A cinematic portrait of a indian girl in bikini, ultra realistic, 8k lighting",
        t2i_params,
    )
    print(f"[T2I 1.3B] Saved image at: {t2i_path}")

    # -----------------------------
    # Example 2: T2V (Text-to-Video)
    # -----------------------------
    print("[*] Running WAN 1.3B T2V inference ...")
    t2v_params = {
        "steps": 8,
        "guidance_scale": 0.0,
        "seed": 42,
        "randomize_seed": True,
        "height": 896,
        "width": 896,
        "duration_seconds": 5,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "output_dir": "./outputs",
    }
    t2v_path = run_inference_wan_1b_t2v(
        job_id,
        "A cinematic shot of a beach at sunset filled with people, smooth motion",
        t2v_params,
    )
    print(f"[T2V 1.3B] Saved video at: {t2v_path}")
