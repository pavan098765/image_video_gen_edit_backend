# fastwan_5b_inference.py
# ✅ Lazy-loading implementation with GPU memory optimizations

import base64
import gc
from io import BytesIO
import os
import uuid
import random
from typing import Dict, Any, Optional
from PIL import Image
import httpx
import torch
import numpy as np
from diffusers import AutoencoderKLWan, WanPipeline, WanImageToVideoPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from upscaler import upscale_image

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
# Model ID / Path
# -----------------------------
MODEL_ID_5B = os.environ.get(
    "FASTWAN_5B_LOCAL_PATH",
    "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",  # fallback if not baked
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
def _calculate_new_dimensions(
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
    """Resize input image to valid multiples of mod_val while preserving aspect ratio."""
    if pil_image is None:
        return default_h, default_w
    orig_w, orig_h = pil_image.size
    if orig_w <= 0 or orig_h <= 0:
        return default_h, default_w
    aspect_ratio = orig_h / orig_w
    calc_h = max(mod_val, (round(np.sqrt(max_area * aspect_ratio)) // mod_val) * mod_val)
    calc_w = max(mod_val, (round(np.sqrt(max_area / aspect_ratio)) // mod_val) * mod_val)
    new_h = int(np.clip(calc_h, min_h, (max_h // mod_val) * mod_val))
    new_w = int(np.clip(calc_w, min_w, (max_w // mod_val) * mod_val))
    return new_h, new_w


def get_pipelines():
    """Load and cache WAN 5B pipelines."""
    if MODEL_ID_5B in PIPELINE_CACHE:
        return PIPELINE_CACHE[MODEL_ID_5B]

    print(f"[WAN 5B] Loading pipelines from {MODEL_ID_5B} ...")
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID_5B, subfolder="vae", torch_dtype=torch.float32)
    t2v_pipe = WanPipeline.from_pretrained(MODEL_ID_5B, vae=vae, torch_dtype=torch.bfloat16)
    i2v_pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID_5B, vae=vae, torch_dtype=torch.bfloat16)

    for pipe in [t2v_pipe, i2v_pipe]:
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    PIPELINE_CACHE[MODEL_ID_5B] = {"t2v": t2v_pipe, "i2v": i2v_pipe}
    print("✅ WAN 5B pipelines loaded and cached.")
    return PIPELINE_CACHE[MODEL_ID_5B]


def unload_wan_5b_pipelines():
    """
    Optional cleanup: unload WAN 5B models and free VRAM/CPU cache.
    """
    global PIPELINE_CACHE
    if MODEL_ID_5B in PIPELINE_CACHE:
        del PIPELINE_CACHE[MODEL_ID_5B]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("🧹 WAN 5B model cache cleared and GPU memory freed.")

# -----------------------------
# Inference
# -----------------------------
def run_inference_wan_5b_t2i(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """Text-to-Image with WAN 5B."""
    print(f"🧩 Starting WAN 5B T2I job: {job_id}")
    
    # Pre-inference GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
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
    out_path = os.path.join(output_dir, f"wan_t2i_5B_{job_id}.png")

    if isinstance(frame, np.ndarray):
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8).squeeze()
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        Image.fromarray(frame).save(out_path)
    elif isinstance(frame, Image.Image):
        frame.save(out_path)

    # Post-inference GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

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


def run_inference_wan_5b_t2v(job_id: str, prompt: str, params: Dict[str, Any], init_image: Optional[Image.Image] = None) -> str:
    """Text-to-Video or Image-to-Video with WAN 5B (init_image is a PIL.Image if provided)."""
    print(f"🧩 Starting WAN 5B T2V/I2V job: {job_id}")
    
    # Pre-inference GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    pipes = get_pipelines()
    t2v_pipe = pipes["t2v"]
    i2v_pipe = pipes["i2v"]

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

    if init_image is not None:
        target_h, target_w = _calculate_new_dimensions(init_image)
        resized_image = init_image.resize((target_w, target_h))
        with torch.inference_mode():
            frames_list = i2v_pipe(
                image=resized_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=target_h,
                width=target_w,
                num_frames=num_frames,
                guidance_scale=cfg,
                num_inference_steps=steps,
                generator=torch.Generator(device="cuda").manual_seed(current_seed),
            ).frames[0]
        prefix = "i2v"
    else:
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
        prefix = "t2v"

    # Post-inference GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"wan_{prefix}_5B_{job_id}.mp4")

    try:
        export_to_video(frames_list, out_path, fps=FIXED_FPS)
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"Video export failed: {out_path}")
        print(f"[{prefix.upper()}] Exported successfully to {out_path}")
    except Exception as e:
        print(f"[{prefix.upper()} ERROR] Video export failed: {e}")
        import traceback
        traceback.print_exc()
    return out_path


# -----------------------------
# Main Example
# -----------------------------
if __name__ == "__main__":
    # --- T2I ---
    t2i_job_id = str(uuid.uuid4())
    print("[*] Running WAN 5B T2I inference ...")
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
    t2i_path = run_inference_wan_5b_t2i(t2i_job_id, "A cinematic portrait of a woman, ultra realistic, 8k lighting", t2i_params)
    print(f"[T2I 5B] Saved image at: {t2i_path}\n")

    # --- T2V ---
    t2v_job_id = str(uuid.uuid4())
    print("[*] Running WAN 5B T2V inference ...")
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
    t2v_path = run_inference_wan_5b_t2v(t2v_job_id, "A cinematic shot of a beach at sunset filled with people, smooth motion", t2v_params)
    print(f"[T2V 5B] Saved video at: {t2v_path}\n")

    # --- I2V ---
    i2v_job_id = str(uuid.uuid4())
    print("[*] Running WAN 5B I2V inference ...")
    url = "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/Snapinsta.app_262322643_4937545659591345_7487236998017046274_n_1080.jpg"
    with httpx.Client(timeout=60) as c:
        img_bytes = c.get(url).content
    input_img = Image.open(BytesIO(img_bytes)).convert("RGB")

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
    i2v_path = run_inference_wan_5b_t2v(i2v_job_id, "Animate this image with cinematic motion, smooth and realistic", i2v_params, init_image=input_img)
    print(f"[I2V 5B] Saved video at: {i2v_path}")
