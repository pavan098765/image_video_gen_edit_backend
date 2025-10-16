import os
import uuid
import random
from typing import Dict, Any, Optional
from PIL import Image
import torch
import numpy as np
from diffusers import (
    AutoencoderKLWan,
    WanPipeline,
    WanImageToVideoPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import export_to_video
from upscaler import upscale_image

# --- Auto-merge split safetensors if merge.py exists ---
try:
    from merge import main as merge_main
    print("🔍 Scanning for .safetensors.part-* files...")
    merge_main()
except Exception as e:
    print(f"⚠️ merge.py auto-merge skipped or failed: {e}")

# -----------------------------
# Model ID / Path
# -----------------------------
MODEL_ID_1B = os.environ.get(
    "FASTWAN_1B_LOCAL_PATH",
    "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",  # fallback if not baked
)
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
    """Load and cache WAN pipelines."""
    if MODEL_ID_1B in PIPELINE_CACHE:
        return PIPELINE_CACHE[MODEL_ID_1B]

    print(f"[WAN] Loading pipelines from {MODEL_ID_1B} ...")
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID_1B, subfolder="vae", torch_dtype=torch.float32)
    t2v_pipe = WanPipeline.from_pretrained(MODEL_ID_1B, vae=vae, torch_dtype=torch.bfloat16)
    i2v_pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID_1B, vae=vae, torch_dtype=torch.bfloat16)

    for pipe in [t2v_pipe, i2v_pipe]:
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)
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
    """Text-to-Image using WAN 1B."""
    pipes = get_pipelines()
    t2v_pipe = pipes["t2v"]

    steps = int(params.get("steps", 4))
    cfg = float(params.get("guidance_scale", 0.0))
    seed = int(params.get("seed", 44))
    randomize_seed = params.get("randomize_seed", False)
    height = int(params.get("height", 896))
    width = int(params.get("width", 896))
    negative_prompt = params.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    output_dir = params.get("output_dir", "./outputs")

    target_h = max(MOD_VALUE, (height // MOD_VALUE) * MOD_VALUE)
    target_w = max(MOD_VALUE, (width // MOD_VALUE) * MOD_VALUE)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else seed

    with torch.inference_mode():
        frame = t2v_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=target_h,
            width=target_w,
            num_frames=1,
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=torch.Generator(device="cuda").manual_seed(current_seed),
        ).frames[0][0]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"wan_t2i_1B_{job_id}.png")

    if isinstance(frame, np.ndarray):
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8).squeeze()
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        Image.fromarray(frame).save(out_path)
    elif isinstance(frame, Image.Image):
        frame.save(out_path)

    # -------------------------
    # Optional Upscaling and compulsory metadata removal
    # -------------------------
    
    scale_image = params.get("scale_image", True)
    scale_factor = params.get("scale_factor", 2) if scale_image else 0
    model_name = "RealESRGAN_x2plus" if scale_factor == 2 else "RealESRGAN_x4plus"

    upscaled = upscale_image(out_path,  model_name=model_name, scale_factor=scale_factor)

    upscaled_path = os.path.join(output_dir, f"{job_id}_upscaled.png")
    upscaled.save(upscaled_path)
    return os.path.abspath(upscaled_path)


def run_inference_wan_1b_t2v(
    job_id: str,
    prompt: str,
    params: Dict[str, Any],
    input_image: Optional[Image.Image] = None,
) -> str:
    """Text-to-Video using WAN 1B."""
    pipes = get_pipelines()
    t2v_pipe = pipes["t2v"]

    steps = int(params.get("steps", 4))
    cfg = float(params.get("guidance_scale", 0.0))
    seed = int(params.get("seed", 44))
    randomize_seed = params.get("randomize_seed", False)
    height = int(params.get("height", 896))
    width = int(params.get("width", 896))
    duration_seconds = float(params.get("duration_seconds", 2))
    negative_prompt = params.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    output_dir = params.get("output_dir", "./outputs")

    target_h = max(MOD_VALUE, (height // MOD_VALUE) * MOD_VALUE)
    target_w = max(MOD_VALUE, (width // MOD_VALUE) * MOD_VALUE)
    num_frames = np.clip(int(round(duration_seconds * FIXED_FPS)), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else seed

    with torch.inference_mode():
        frames_list = t2v_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=target_h,
            width=target_w,
            num_frames=num_frames,
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=torch.Generator(device="cuda").manual_seed(current_seed),
        ).frames[0]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"wan_t2v_1B_{job_id}.mp4")
    export_to_video(frames_list, out_path, fps=FIXED_FPS)
    return out_path

# -----------------------------
# Main Example
# -----------------------------
if __name__ == "__main__":
    job_id = str(uuid.uuid4())

    print("[*] Running WAN T2I inference ...")
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
        "A cinematic portrait of a woman, ultra realistic, 8k lighting",
        t2i_params,
    )
    print(f"[T2I] Saved image at: {t2i_path}")

    print("[*] Running WAN T2V inference ...")
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
    print(f"[T2V] Saved video at: {t2v_path}")
