import os

os.environ["HF_HOME"] = "/models_storage/all_models/fastwan"
os.environ["HF_HUB_CACHE"] = "/models_storage/all_models/fastwan"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
# Model paths
# -----------------------------
MODEL_ID_5B = os.getenv(
    "WAN_MODEL_ID", "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
)
MODEL_ID_1B = os.getenv("WAN_MODEL_ID_1B", "FastVideo/FastWan2.1-T2V-1.3B-Diffusers")

WAN_MODELS = {"2.2": MODEL_ID_5B, "2.1": MODEL_ID_1B}
PIPELINES_CACHE: Dict[str, Any] = {}

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
# Helper functions
# -----------------------------
def _calculate_new_dimensions_wan(
    pil_image,
    mod_val=32,
    max_area=720 * 1024,
    min_h=256,
    max_h=1024,
    min_w=256,
    max_w=1024,
    default_h=896,
    default_w=896,
):
    if pil_image is None:
        return default_h, default_w
    orig_w, orig_h = pil_image.size
    if orig_w <= 0 or orig_h <= 0:
        return default_h, default_w
    aspect_ratio = orig_h / orig_w

    calc_h = round(np.sqrt(max_area * aspect_ratio))
    calc_w = round(np.sqrt(max_area / aspect_ratio))
    calc_h = max(mod_val, (calc_h // mod_val) * mod_val)
    calc_w = max(mod_val, (calc_w // mod_val) * mod_val)

    new_h = int(np.clip(calc_h, min_h, (max_h // mod_val) * mod_val))
    new_w = int(np.clip(calc_w, min_w, (max_w // mod_val) * mod_val))
    return new_h, new_w


def get_pipelines(model_version: str):
    """Lazy load WAN pipelines (T2V/I2V) for 2.1 or 2.2"""
    model_id = WAN_MODELS[model_version]
    if model_id in PIPELINES_CACHE:
        return PIPELINES_CACHE[model_id]

    print(f"Loading WAN pipelines for {model_id} ...")

    vae = AutoencoderKLWan.from_pretrained(
        model_id, 
        subfolder="vae", 
        torch_dtype=torch.float32
    )

    t2v_pipe = WanPipeline.from_pretrained(
        model_id, 
        vae=vae, 
        torch_dtype=torch.bfloat16
    )

    i2v_pipe = WanImageToVideoPipeline.from_pretrained(
        model_id, 
        vae=vae, 
        torch_dtype=torch.bfloat16
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

    PIPELINES_CACHE[model_id] = {"t2v": t2v_pipe, "i2v": i2v_pipe}
    return PIPELINES_CACHE[model_id]

# -----------------------------
# FASTWAN Inference
# -----------------------------
def run_inference_wan_t2i(
    job_id: str, prompt: str, params: Dict[str, Any], model_version: str = "2.2"
) -> str:
    """Text-to-Image (T2I) - HF Space style"""
    pipes = get_pipelines(model_version)
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
        output_frame = t2v_pipe(
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
    if model_version == "2.1":
        file_name = f"wan_t2i_1B_{job_id}.png"
    else:
        file_name = f"wan_t2i_5B_{job_id}.png"

    out_path = os.path.join(output_dir, file_name)

    if isinstance(output_frame, np.ndarray):
        if output_frame.dtype in [np.float32, np.float64]:
            output_frame = np.clip(output_frame * 255, 0, 255).astype(np.uint8)
        output_frame = output_frame.squeeze()
        if output_frame.ndim == 2:
            output_frame = np.stack([output_frame] * 3, axis=-1)
        Image.fromarray(output_frame).save(out_path)
    elif isinstance(output_frame, Image.Image):
        output_frame.save(out_path)
    else:
        raise TypeError(f"Unexpected output type: {type(output_frame)}")

    return out_path


def run_inference_wan_t2v(
    job_id: str,
    prompt: str,
    params: Dict[str, Any],
    input_image: Optional[Image.Image] = None,
    model_version: str = "2.2",
) -> str:
    """Text-to-Video (T2V) or Image-to-Video (I2V) - HF Space style"""
    pipes = get_pipelines(model_version)
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

    if input_image is not None:
        target_h, target_w = _calculate_new_dimensions_wan(input_image)
        resized_image = input_image.resize((target_w, target_h))
        with torch.inference_mode():
            output_frames_list = i2v_pipe(
                image=resized_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=target_h,
                width=target_w,
                num_frames=num_frames,
                guidance_scale=float(cfg),
                num_inference_steps=int(steps),
                generator=torch.Generator(device="cuda").manual_seed(current_seed),
            ).frames[0]
            file_name = f"wan_i2v_{job_id}.mp4"
    else:
        with torch.inference_mode():
            output_frames_list = t2v_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=target_h,
                width=target_w,
                num_frames=num_frames,
                guidance_scale=float(cfg),
                num_inference_steps=int(steps),
                generator=torch.Generator(device="cuda").manual_seed(current_seed),
            ).frames[0]
            if model_version == "2.1":
                file_name = f"wan_t2v_1.3B_{job_id}.mp4"
            else:
                file_name = f"wan_t2v_5B_{job_id}.mp4"

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, file_name)
    export_to_video(output_frames_list, out_path, fps=FIXED_FPS)
    return out_path


# -----------------------------
# Main Example
# -----------------------------
if __name__ == "__main__":
    job_id = str(uuid.uuid4())

    # Example 1: T2I
    print("[*] Running WAN 2.2 T2I  5B inference ...")
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
    t2i_path = run_inference_wan_t2i(
        job_id,
        "A cinematic portrait of a monk, ultra realistic, 8k lighting",
        t2i_params, model_version="2.2",
    )
    print(f"[T2I] Saved image at: {t2i_path}")

    # Example 2: T2I (WAN2.1, 1.3B)
    print("[*] Running WAN 2.1 T2I 1.3B inference ...")
    t2v_path_1b = run_inference_wan_t2i(
        job_id,
        "A cinematic shot of a futuristic city at sunset, smooth motion",
        t2i_params,
        model_version="2.1",
    )

    # Example 3: T2V (WAN2.2 default)
    print("[*] Running WAN 2.2 T2V  5B inference ...")
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
    t2v_path = run_inference_wan_t2v(
        job_id,
        "A cinematic shot of a futuristic city at sunset, smooth motion",
        t2v_params,
        model_version="2.2",
    )
    print(f"[T2V 5B] Saved video at: {t2v_path}")

    # Example 4: T2V (WAN2.1, 1.3B)
    print("[*] Running WAN 2.1 T2V 1.3B inference ...")
    t2v_path_1b = run_inference_wan_t2v(
        job_id,
        "A cinematic shot of a futuristic city at sunset, smooth motion",
        t2v_params,
        model_version="2.1",
    )
    print(f"[T2V 1.3B] Saved video at: {t2v_path_1b}")

    # Example 5: I2V
    print("[*] Running WAN 2.2 I2V  5B inference ...")
    input_img = Image.open("input_example.jpg").convert("RGB")
    i2v_params = {
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
    i2v_path = run_inference_wan_t2v(
        job_id,
        "Animate this image with cinematic motion, smooth and realistic",
        i2v_params,
        input_image=input_img,
    )
    print(f"[I2V] Saved video at: {i2v_path}")


