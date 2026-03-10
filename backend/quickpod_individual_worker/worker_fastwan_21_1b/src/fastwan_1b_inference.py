# fastwan_1b_inference.py
# ✅ Lazy-loading implementation with GPU memory optimizations

import gc
import os
import uuid
import random
import time
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
from diffusers.utils import export_to_video, load_image
from upscaler import upscale_image

# Video post-processing imports
try:
    from video_postprocessing import (
        load_rife_model,
        load_sd_upscale,
        apply_rife_interpolation,
        apply_realesrgan_upscaling,
        check_rife_code_available,
    )
    POSTPROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Video post-processing not available: {e}")
    POSTPROCESSING_AVAILABLE = False

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
MODEL_ID_1B = os.environ.get(
    "FASTWAN_1B_LOCAL_PATH",
    "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",  # fallback if not baked
)
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
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
    print("✅ WAN 1B pipelines loaded and cached.")
    return PIPELINE_CACHE[MODEL_ID_1B]


def unload_wan_1b_pipelines():
    """
    Optional cleanup: unload WAN 1B models and free VRAM/CPU cache.
    """
    global PIPELINE_CACHE
    if MODEL_ID_1B in PIPELINE_CACHE:
        del PIPELINE_CACHE[MODEL_ID_1B]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("🧹 WAN 1B model cache cleared and GPU memory freed.")

# -----------------------------
# Inference
# -----------------------------
def run_inference_wan_1b_t2i(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """Text-to-Image using WAN 1B."""
    print(f"🧩 Starting WAN 1B T2I job: {job_id}")
    
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
    out_path = os.path.join(output_dir, f"wan_t2i_1B_{job_id}.png")

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


def run_inference_wan_1b_t2v(
    job_id: str,
    prompt: str,
    params: Dict[str, Any],
    input_image: Optional[Image.Image] = None,
) -> str:
    """Text-to-Video using WAN 1B."""
    print(f"🧩 Starting WAN 1B T2V job: {job_id}")
    
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

    # Post-processing options
    enable_rife = params.get("enable_rife", False)
    rife_target_fps = params.get("rife_target_fps", 30)
    enable_upscaling = params.get("enable_upscaling", False)
    upscale_factor = int(params.get("upscale_factor", 2))
    postprocess_interpolate_first = params.get("postprocess_interpolate_first", False)
    
    # Post-inference GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # Post-processing (RIFE interpolation and/or Real-ESRGAN upscaling)
    postprocess_time = 0.0
    final_fps = FIXED_FPS
    final_output = frames_list
    final_height = target_h
    final_width = target_w
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if POSTPROCESSING_AVAILABLE and (enable_rife or enable_upscaling):
        postprocess_start = time.time()
        
        # Load post-processing models from local paths
        rife_model = None
        upscale_model = None
        
        if enable_rife:
            if check_rife_code_available():
                rife_path = os.path.join(MODELS_DIR, "rife")
                if os.path.exists(rife_path) and os.listdir(rife_path):
                    rife_model = load_rife_model(rife_path)
                else:
                    print(f"⚠️  RIFE model not found at {rife_path}, frame interpolation disabled")
            else:
                print(f"⚠️  RIFE code not available, frame interpolation disabled")
        
        if enable_upscaling:
            realesrgan_model_file = os.path.join(MODELS_DIR, "realesrgan", f"RealESRGAN_x{upscale_factor}.pth")
            if os.path.exists(realesrgan_model_file):
                try:
                    upscale_model = load_sd_upscale(realesrgan_model_file, device)
                except Exception as e:
                    print(f"⚠️  Real-ESRGAN model failed to load: {str(e)[:100]}")
                    upscale_model = None
            else:
                print(f"⚠️  Real-ESRGAN {upscale_factor}x model not found at {realesrgan_model_file}, upscaling disabled")
        
        # Apply post-processing based on order
        if postprocess_interpolate_first:
            # Interpolate → Upscale
            if enable_rife and rife_model:
                try:
                    final_output, actual_fps = apply_rife_interpolation(
                        rife_model, final_output, device, original_fps=FIXED_FPS, target_fps=rife_target_fps
                    )
                    final_fps = actual_fps
                except Exception as e:
                    print(f"⚠️  RIFE interpolation failed: {str(e)[:100]}")
            
            if enable_upscaling and upscale_model:
                try:
                    final_output = apply_realesrgan_upscaling(upscale_model, final_output, device)
                    final_height = target_h * upscale_factor
                    final_width = target_w * upscale_factor
                except Exception as e:
                    print(f"⚠️  Upscaling failed: {str(e)[:100]}")
        else:
            # Upscale → Interpolate (default)
            if enable_upscaling and upscale_model:
                try:
                    final_output = apply_realesrgan_upscaling(upscale_model, final_output, device)
                    final_height = target_h * upscale_factor
                    final_width = target_w * upscale_factor
                except Exception as e:
                    print(f"⚠️  Upscaling failed: {str(e)[:100]}")
            
            if enable_rife and rife_model:
                try:
                    final_output, actual_fps = apply_rife_interpolation(
                        rife_model, final_output, device, original_fps=FIXED_FPS, target_fps=rife_target_fps
                    )
                    final_fps = actual_fps
                except Exception as e:
                    print(f"⚠️  RIFE interpolation failed: {str(e)[:100]}")
        
        postprocess_time = time.time() - postprocess_start
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"wan_t2v_1B_{job_id}.mp4")
    export_to_video(final_output, out_path, fps=final_fps)
    print(f"[T2V 1B] ✅ Video saved at {out_path} | {final_width}x{final_height} @ {final_fps:.2f}fps")
    if postprocess_time > 0:
        print(f"[T2V 1B] Post-processing time: {postprocess_time:.1f}s")
    return out_path


def run_inference_wan_1b_i2v(
    job_id: str,
    prompt: str,
    params: Dict[str, Any],
    input_image: Optional[Image.Image] = None,
) -> str:
    """Image-to-Video using WAN 1B."""
    print(f"🧩 Starting WAN 1B I2V job: {job_id}")
    
    # Pre-inference GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    pipes = get_pipelines()
    i2v_pipe = pipes["i2v"]
    
    # Load input image
    if input_image is None:
        init_image_path = params.get("init_image")
        if init_image_path:
            if isinstance(init_image_path, str) and os.path.exists(init_image_path):
                input_image = load_image(init_image_path)
            elif isinstance(init_image_path, str) and init_image_path.startswith("data:"):
                # Base64 encoded image
                import base64
                from io import BytesIO
                header, encoded = init_image_path.split(",", 1)
                image_data = base64.b64decode(encoded)
                input_image = Image.open(BytesIO(image_data)).convert("RGB")
            else:
                raise ValueError(f"Invalid init_image: {init_image_path}")
        else:
            raise ValueError("init_image is required for I2V")
    
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
    
    # Resize image to target dimensions
    input_image = input_image.resize((target_w, target_h), Image.LANCZOS)
    
    with torch.inference_mode():
        frames_list = i2v_pipe(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=target_h,
            width=target_w,
            num_frames=num_frames,
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=torch.Generator(device="cuda").manual_seed(current_seed),
        ).frames[0]
    
    # Post-processing options
    enable_rife = params.get("enable_rife", False)
    rife_target_fps = params.get("rife_target_fps", 30)
    enable_upscaling = params.get("enable_upscaling", False)
    upscale_factor = int(params.get("upscale_factor", 2))
    postprocess_interpolate_first = params.get("postprocess_interpolate_first", False)
    
    # Post-inference GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # Post-processing (RIFE interpolation and/or Real-ESRGAN upscaling)
    postprocess_time = 0.0
    final_fps = FIXED_FPS
    final_output = frames_list
    final_height = target_h
    final_width = target_w
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if POSTPROCESSING_AVAILABLE and (enable_rife or enable_upscaling):
        postprocess_start = time.time()
        
        # Load post-processing models from local paths
        rife_model = None
        upscale_model = None
        
        if enable_rife:
            if check_rife_code_available():
                rife_path = os.path.join(MODELS_DIR, "rife")
                if os.path.exists(rife_path) and os.listdir(rife_path):
                    rife_model = load_rife_model(rife_path)
                else:
                    print(f"⚠️  RIFE model not found at {rife_path}, frame interpolation disabled")
            else:
                print(f"⚠️  RIFE code not available, frame interpolation disabled")
        
        if enable_upscaling:
            realesrgan_model_file = os.path.join(MODELS_DIR, "realesrgan", f"RealESRGAN_x{upscale_factor}.pth")
            if os.path.exists(realesrgan_model_file):
                try:
                    upscale_model = load_sd_upscale(realesrgan_model_file, device)
                except Exception as e:
                    print(f"⚠️  Real-ESRGAN model failed to load: {str(e)[:100]}")
                    upscale_model = None
            else:
                print(f"⚠️  Real-ESRGAN {upscale_factor}x model not found at {realesrgan_model_file}, upscaling disabled")
        
        # Apply post-processing based on order
        if postprocess_interpolate_first:
            # Interpolate → Upscale
            if enable_rife and rife_model:
                try:
                    final_output, actual_fps = apply_rife_interpolation(
                        rife_model, final_output, device, original_fps=FIXED_FPS, target_fps=rife_target_fps
                    )
                    final_fps = actual_fps
                except Exception as e:
                    print(f"⚠️  RIFE interpolation failed: {str(e)[:100]}")
            
            if enable_upscaling and upscale_model:
                try:
                    final_output = apply_realesrgan_upscaling(upscale_model, final_output, device)
                    final_height = target_h * upscale_factor
                    final_width = target_w * upscale_factor
                except Exception as e:
                    print(f"⚠️  Upscaling failed: {str(e)[:100]}")
        else:
            # Upscale → Interpolate (default)
            if enable_upscaling and upscale_model:
                try:
                    final_output = apply_realesrgan_upscaling(upscale_model, final_output, device)
                    final_height = target_h * upscale_factor
                    final_width = target_w * upscale_factor
                except Exception as e:
                    print(f"⚠️  Upscaling failed: {str(e)[:100]}")
            
            if enable_rife and rife_model:
                try:
                    final_output, actual_fps = apply_rife_interpolation(
                        rife_model, final_output, device, original_fps=FIXED_FPS, target_fps=rife_target_fps
                    )
                    final_fps = actual_fps
                except Exception as e:
                    print(f"⚠️  RIFE interpolation failed: {str(e)[:100]}")
        
        postprocess_time = time.time() - postprocess_start
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"wan_i2v_1B_{job_id}.mp4")
    export_to_video(final_output, out_path, fps=final_fps)
    print(f"[I2V 1B] ✅ Video saved at {out_path} | {final_width}x{final_height} @ {final_fps:.2f}fps")
    if postprocess_time > 0:
        print(f"[I2V 1B] Post-processing time: {postprocess_time:.1f}s")
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
