# ltx_video_I2V_inference.py
# ✅ Lazy-loading implementation with GPU memory optimizations

import gc
import os
import base64
import random
import tempfile
from io import BytesIO
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np
import torch
import imageio
import yaml

# Aggressive GPU cleanup at import to prevent OOM
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# --- Auto-merge split safetensors if merge.py exists ---
try:
    from merge import main as merge_main
    print("🔍 Scanning for .safetensors.part-* files...")
    merge_main()
except Exception as e:
    print(f"⚠️ merge.py auto-merge skipped or failed: {e}")

# -----------------------------
# Model Configuration
# -----------------------------
MODEL_DIR = os.environ.get("LTX_VIDEO_LOCAL_PATH", "/app/models/ltx_video")
CONFIG_PATH = os.environ.get("LTX_CONFIG_PATH", "configs/ltxv-13b-0.9.8-distilled.yaml")

PIPELINE_CACHE = {}
FPS = 30
MAX_SEED = np.iinfo(np.int32).max

# -----------------------------
# Core Imports
# -----------------------------
from inference import (
    create_ltx_video_pipeline,
    create_latent_upsampler,
    seed_everething,
    calculate_padding,
    load_image_to_tensor_with_resize_and_crop,
    load_media_file,
)
from ltx_video.pipelines.pipeline_ltx_video import LTXMultiScalePipeline, ConditioningItem


# -----------------------------
# Helpers
# -----------------------------
def _calculate_frame_params(duration_sec: float, fps: int = FPS) -> int:
    target_frames = round(max(1, min(257, duration_sec * fps)))
    n_val = round((target_frames - 1) / 8.0)
    return int(n_val * 8 + 1)


def _pad_dimensions(h: int, w: int) -> tuple[int, int]:
    h_pad = ((h - 1) // 32 + 1) * 32
    w_pad = ((w - 1) // 32 + 1) * 32
    return h_pad, w_pad


def get_pipelines() -> dict:
    """Load and cache the LTX Video pipeline + upscaler."""
    if "ltx" in PIPELINE_CACHE:
        return PIPELINE_CACHE["ltx"]

    print(f"[LTX] Loading from local path: {MODEL_DIR}")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    ckpt_path = os.path.join(MODEL_DIR, "ltxv-13b-0.9.8-distilled.safetensors")
    upscaler_path = os.path.join(MODEL_DIR, "spatial_upscaler.safetensors")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = create_ltx_video_pipeline(
        ckpt_path=ckpt_path,
        precision=config["precision"],
        text_encoder_model_name_or_path=config["text_encoder_model_name_or_path"],
        sampler=config["sampler"],
        device=device,
        enhance_prompt=False,
        prompt_enhancer_image_caption_model_name_or_path=config["prompt_enhancer_image_caption_model_name_or_path"],
        prompt_enhancer_llm_model_name_or_path=config["prompt_enhancer_llm_model_name_or_path"],
    )

    upsampler = None
    if os.path.exists(upscaler_path):
        upsampler = create_latent_upsampler(upscaler_path, device)

    PIPELINE_CACHE["ltx"] = {"pipe": pipe, "upsampler": upsampler, "device": device, "config": config}
    print("✅ LTX Video pipeline cached and ready.")
    return PIPELINE_CACHE["ltx"]


def unload_ltx_pipelines():
    """
    Optional cleanup: unload LTX Video models and free VRAM/CPU cache.
    """
    global PIPELINE_CACHE
    PIPELINE_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("🧹 LTX Video model cache cleared and GPU memory freed.")


# -----------------------------
# Inference Logic
# -----------------------------
def run_inference_ltx_t2v_i2v_v2v(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """
    Unified Text-to-Video, Image-to-Video, Video-to-Video inference for LTX Video.
    """
    print(f"🧩 Starting LTX Video job: {job_id}")
    
    # Pre-inference GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    data = get_pipelines()
    pipe = data["pipe"]
    upsampler = data["upsampler"]
    device = data["device"]
    config = data["config"]

    mode = params.get("mode", "text-to-video")
    negative_prompt = params.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted")
    height = int(params.get("height", 512))
    width = int(params.get("width", 704))
    duration = float(params.get("duration", 2.0))
    seed = int(params.get("seed", 42))
    randomize_seed = bool(params.get("randomize_seed", True))
    guidance_scale = float(params.get("guidance_scale", 3.0))
    improve_texture = bool(params.get("improve_texture_flag", True))
    frames_to_use = int(params.get("frames_to_use", 9))
    init_image = params.get("init_image")
    init_video = params.get("init_video")

    # Seed
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    seed_everething(seed)

    # Dimensions & padding
    num_frames = _calculate_frame_params(duration, FPS)
    h_pad, w_pad = _pad_dimensions(height, width)
    padding = calculate_padding(height, width, h_pad, w_pad)

    # Call parameters
    call_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": h_pad,
        "width": w_pad,
        "num_frames": num_frames,
        "frame_rate": FPS,
        "generator": torch.Generator(device=device).manual_seed(seed),
        "output_type": "pt",
        "conditioning_items": None,
        "media_items": None,
        "decode_timestep": config["decode_timestep"],
        "decode_noise_scale": config["decode_noise_scale"],
        "stochastic_sampling": config["stochastic_sampling"],
        "is_video": True,
        "vae_per_channel_normalize": True,
    }

    # Mode-specific setup
    if mode == "image-to-video" and init_image:
        print("[LTX] Running Image-to-Video...")
        img = Image.open(BytesIO(base64.b64decode(init_image))).convert("RGB")
        img_path = f"/tmp/{job_id}_init.png"
        img.save(img_path)
        tensor = load_image_to_tensor_with_resize_and_crop(img_path, height, width)
        tensor = torch.nn.functional.pad(tensor, padding)
        call_kwargs["conditioning_items"] = [ConditioningItem(tensor.to(device), 0, 1.0)]

    elif mode == "video-to-video" and init_video:
        print("[LTX] Running Video-to-Video...")
        tmp_video = f"/tmp/{job_id}_input.mp4"
        with open(tmp_video, "wb") as f:
            f.write(base64.b64decode(init_video))
        call_kwargs["media_items"] = load_media_file(
            media_path=tmp_video,
            height=height,
            width=width,
            max_frames=frames_to_use,
            padding=padding
        ).to(device)
    else:
        print("[LTX] Running Text-to-Video...")

    # Run pipeline
    if improve_texture and upsampler:
        final_pipe = LTXMultiScalePipeline(pipe, upsampler)
    else:
        final_pipe = pipe

    print(f"[LTX] Generating with seed {seed}, {num_frames} frames at {FPS} fps...")
    with torch.inference_mode():
        result = final_pipe(**call_kwargs)
    tensor_out = result.images

    if tensor_out is None:
        raise RuntimeError("❌ LTX Video generation failed: no frames returned.")
    
    # Post-inference GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    video_np = tensor_out[0].permute(1, 2, 3, 0).cpu().float().numpy()
    video_np = np.clip(video_np, 0, 1)
    video_np = (video_np * 255).astype(np.uint8)

    tmp_dir = tempfile.mkdtemp()
    out_path = os.path.join(tmp_dir, f"{job_id}_{mode}.mp4")

    with imageio.get_writer(out_path, fps=FPS, macro_block_size=1) as writer:
        for f in range(video_np.shape[0]):
            writer.append_data(video_np[f])

    print(f"[LTX] ✅ Video saved at {out_path}")
    return os.path.abspath(out_path)


# -----------------------------
# Example CLI usage
# -----------------------------
if __name__ == "__main__":
    job_id = "demo_" + str(random.randint(1000, 9999))
    prompt = "A cinematic aerial drone shot of snowy mountains at sunrise"
    params = {
        "mode": "text-to-video",
        "height": 512,
        "width": 704,
        "duration": 2.0,
        "guidance_scale": 3.0,
        "randomize_seed": True,
    }
    out = run_inference_ltx_t2v_i2v_v2v(job_id, prompt, params)
    print("✅ Demo complete:", out)
