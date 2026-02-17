#!/usr/bin/env python3
"""
Inference module for WAN 2.2 I2V 14B SDNQ FP8 FastDM merged Lightning worker.
Uses FastDM merged Lightning model where Lightning distillation is already fused into the base model weights,
then quantized. No LoRA loading needed.
Based on wan_22_14b_fastdm_i2v_inference.py from work_space.
"""

from io import BytesIO
import os
import random
import time
import torch
import traceback
import httpx
import numpy as np
import diffusers
from diffusers.utils import export_to_video, load_image
from sdnq import SDNQConfig  # import sdnq to register it into diffusers and transformers
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
from typing import Dict, Any, Optional
from PIL import Image

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

# Set PyTorch optimizations for speed
torch.set_float32_matmul_precision('medium')  # Faster matmul with slight precision trade-off
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed

# Configuration
# FastDM merged Lightning model: Lightning distillation is already fused into base model, then quantized
MODEL_REPO = "rockapaper/wan22-14b-i2v-lightning-sdnq-fp8-svd-r32"
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
MODEL_LOCAL_DIR = os.path.join(MODELS_DIR, "fastdm_wan22_i2v_lightning_sdnq_fp8_svd_r32")

# Pipeline cache
PIPELINE_CACHE = {}
PIPELINE_LOADED = False

# Default inference parameters (Lightning-merged model works with 4-8 steps and guidance_scale=1.0)

# Default inference parameters
DEFAULT_NEGATIVE_PROMPT = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景, 三条腿, 背景人很多, 倒着走, 阴道变形, 乳房变形, 乳头畸形, 变形的皮肤, 大腿畸形，臀部畸, 面部畸形，眼睛畸形, 膝盖畸形，髋关节畸形"

DEFAULT_FPS = 16
DEFAULT_NUM_FRAMES = 81
DEFAULT_NUM_INFERENCE_STEPS = 8
DEFAULT_GUIDANCE_SCALE = 1.0


def load_pipeline():
    """Load and cache the I2V pipeline."""
    global PIPELINE_CACHE, PIPELINE_LOADED
    
    if PIPELINE_LOADED and MODEL_LOCAL_DIR in PIPELINE_CACHE:
        return PIPELINE_CACHE[MODEL_LOCAL_DIR]
    
    load_start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Verify model directory exists and has required files
    if not os.path.exists(MODEL_LOCAL_DIR):
        raise FileNotFoundError(f"Model directory not found: {MODEL_LOCAL_DIR}")
    
    # Check for text_encoder directory
    text_encoder_dir = os.path.join(MODEL_LOCAL_DIR, "text_encoder")
    if not os.path.exists(text_encoder_dir):
        raise FileNotFoundError(f"Text encoder directory not found: {text_encoder_dir}. Model may be incomplete.")
    
    # Check for text encoder config
    text_encoder_config = os.path.join(text_encoder_dir, "config.json")
    if not os.path.exists(text_encoder_config):
        print(f"⚠️  Warning: Text encoder config not found at {text_encoder_config}")
    else:
        print(f"✓ Text encoder directory verified: {text_encoder_dir}")
    
    # Check for text encoder model file and verify it's not suspiciously small
    text_encoder_model = os.path.join(text_encoder_dir, "model.safetensors")
    if os.path.exists(text_encoder_model):
        model_size = os.path.getsize(text_encoder_model)
        model_size_gb = model_size / (1024 ** 3)
        # Expected size is ~8.15GB, warn if significantly smaller (e.g., < 7.5GB)
        if model_size_gb < 7.5:
            print(f"⚠️  WARNING: Text encoder model file is suspiciously small!")
            print(f"   Size: {model_size_gb:.2f} GB (expected ~8.15 GB)")
            print(f"   This may indicate an incomplete merge or corrupted download.")
            print(f"   Check if merge.py ran correctly and all .part-* files were merged.")
        else:
            print(f"✓ Text encoder model file size verified: {model_size_gb:.2f} GB")
    else:
        # Check for split parts
        import glob
        part_files = glob.glob(text_encoder_model + ".part-*")
        if part_files:
            print(f"⚠️  WARNING: Text encoder model is still split into {len(part_files)} parts!")
            print(f"   Merge.py should have merged these. Parts found:")
            for part in sorted(part_files)[:5]:  # Show first 5
                print(f"     - {os.path.basename(part)}")
            if len(part_files) > 5:
                print(f"     ... and {len(part_files) - 5} more")
            raise FileNotFoundError(f"Text encoder model not merged. Found {len(part_files)} part files. Run merge.py first.")
        else:
            print(f"⚠️  Warning: Text encoder model.safetensors not found at {text_encoder_model}")
    
    print(f"📦 Loading pipeline from {MODEL_LOCAL_DIR}...")
    print(f"   Loading SDNQ FP8+SVD quantized model (FastDM merged Lightning)...")
    pipe = diffusers.WanImageToVideoPipeline.from_pretrained(
        MODEL_LOCAL_DIR,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,  # Load components on CPU first
        local_files_only=True  # Ensure we only use local files, prevent partial downloads
    )
    
    # Fix SDNQ tied weights: shared.weight -> encoder.embed_tokens.weight
    # SDNQ's load_sdnq_model handles this (loader.py:128-129), but
    # diffusers' from_pretrained does not. Re-tie explicitly.
    if hasattr(pipe, 'text_encoder'):
        te = pipe.text_encoder
        if hasattr(te, 'shared') and hasattr(te, 'encoder') and hasattr(te.encoder, 'embed_tokens'):
            shared_ptr = te.shared.weight.data_ptr()
            embed_ptr = te.encoder.embed_tokens.weight.data_ptr()
            if shared_ptr != embed_ptr:
                te.encoder.embed_tokens.weight = te.shared.weight
                print("✓ Fixed: Re-tied shared.weight -> encoder.embed_tokens.weight")
            else:
                print("✓ Text encoder weights already tied correctly")
        # Also call tie_weights() as a safety net
        if hasattr(te, 'tie_weights'):
            te.tie_weights()
    
    # Apply memory management (sequential CPU offload for better VRAM management)
    # FP8 models (~20-28GB) use sequential CPU offload which is more aggressive
    print("⚡ Applying memory and speed optimizations...")
    if device == "cuda":
        try:
            pipe.enable_sequential_cpu_offload()
            print("✓ Sequential CPU offload enabled (aggressive memory management)")
        except Exception as e1:
            print(f"   ⚠️  Sequential CPU offload failed: {e1}")
            try:
                # Fallback to regular CPU offload
                pipe.enable_model_cpu_offload()
                print("✓ Model CPU offload enabled (fallback)")
            except Exception as e2:
                print(f"   ⚠️  CPU offload failed: {e2}")
                # Last resort: move to device directly
                pipe = pipe.to(device)
                print(f"   ✓ Model moved to {device} directly")
    else:
        pipe = pipe.to(device)
        print(f"✓ Model moved to {device}")
    
    # Try Flash Attention first, fallback to XFormers
    flash_attn_available = False
    try:
        import flash_attn
        flash_attn_available = True
        print(f"   ✓ Flash Attention detected (version: {getattr(flash_attn, '__version__', 'unknown')})")
    except ImportError:
        pass
    
    if not flash_attn_available:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("   ✓ XFormers memory-efficient attention enabled")
        except Exception:
            pass
    
    # Try to enable VAE tiling if available (reduces VRAM for VAE)
    try:
        if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_tiling'):
            pipe.vae.enable_tiling()
            print("   ✓ VAE tiling enabled (reduces VRAM usage)")
    except Exception:
        pass  # VAE tiling not available, skip silently
    
    # Try to enable attention slicing if available (reduces VRAM for attention)
    try:
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing(slice_size="max")
            print("   ✓ Attention slicing enabled (reduces VRAM usage)")
    except Exception:
        pass  # Attention slicing not available, skip silently
    
    load_time = time.time() - load_start_time
    print(f"✓ I2V Pipeline loaded ({load_time:.1f}s)")
    
    # Verify text encoder weights are correct after all loading
    try:
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
            te = pipe.text_encoder
            if hasattr(te, 'shared') and hasattr(te, 'encoder') and hasattr(te.encoder, 'embed_tokens'):
                shared_ptr = te.shared.weight.data_ptr()
                embed_ptr = te.encoder.embed_tokens.weight.data_ptr()
                if shared_ptr == embed_ptr:
                    print(f"✓ Text encoder verified: shared.weight and embed_tokens.weight are tied (shape: {te.shared.weight.shape})")
                else:
                    print("⚠️  WARNING: Text encoder weights are NOT tied after full pipeline load!")
        else:
            print("⚠️  WARNING: Text encoder not found in pipeline!")
    except Exception as e:
        print(f"⚠️  Warning: Could not verify text encoder: {str(e)[:100]}")
    
    PIPELINE_CACHE[MODEL_LOCAL_DIR] = {
        "pipe": pipe,
        "flash_attn_available": flash_attn_available
    }
    PIPELINE_LOADED = True
    
    return PIPELINE_CACHE[MODEL_LOCAL_DIR]


def run_inference_wan_22_i2v(
    job_id: str,
    prompt: str,
    params: Dict[str, Any],
    init_image: Optional[Image.Image] = None
) -> str:
    """
    Run image-to-video inference.
    
    Args:
        job_id: Job identifier
        prompt: Text prompt
        params: Inference parameters (height, width, num_frames, steps, guidance_scale, seed, etc.)
        init_image: Input PIL Image (required for I2V)
    
    Returns:
        Path to output video file
    """
    if init_image is None:
        raise ValueError("init_image is required for I2V inference")
    
    # Pre-inference GPU cleanup
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # Load pipeline
    pipeline_data = load_pipeline()
    pipe = pipeline_data["pipe"]
    
    # Clear CUDA cache before inference to maximize available VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Extract parameters
    num_frames = int(params.get("num_frames", DEFAULT_NUM_FRAMES))
    guidance_scale = float(params.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
    num_inference_steps = int(params.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS))
    seed = params.get("seed", None)
    if seed is None:
        seed = random.randint(0, 1000)
    else:
        seed = int(seed)
    
    negative_prompt = params.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    output_dir = params.get("output_dir", "/workspace")
    os.makedirs(output_dir, exist_ok=True)
    
    # Image processing
    resize_image = params.get("resize_image", True)
    
    # Calculate dimensions
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    
    if resize_image:
        max_area = 480 * 832
        aspect_ratio = init_image.height / init_image.width
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = init_image.resize((width, height))
    else:
        original_width = init_image.width
        original_height = init_image.height
        width = (original_width // mod_value) * mod_value
        height = (original_height // mod_value) * mod_value
        
        if width != original_width or height != original_height:
            image = init_image.resize((width, height))
        else:
            image = init_image
    
    # Post-processing options
    enable_rife = params.get("enable_rife", False)
    rife_target_fps = params.get("rife_target_fps", 30)
    enable_upscaling = params.get("enable_upscaling", False)
    upscale_factor = int(params.get("upscale_factor", 2))
    postprocess_interpolate_first = params.get("postprocess_interpolate_first", False)
    
    fps = params.get("fps", DEFAULT_FPS)
    
    # Generate video
    print(f"[{job_id}] Generating I2V: {width}x{height}, {num_frames}f @ {fps}fps, {num_inference_steps} steps")
    
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    
    generation_start_time = time.time()
    
    with torch.inference_mode():
        output = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).frames[0]
    
    generation_time = time.time() - generation_start_time
    
    # Post-processing (RIFE interpolation and/or Real-ESRGAN upscaling)
    postprocess_time = 0.0
    final_fps = fps
    final_output = output
    final_height = height
    final_width = width
    
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
                    print(f"⚠️  RIFE model not found, interpolation disabled")
            else:
                print(f"⚠️  RIFE code not available, interpolation disabled")
        
        if enable_upscaling:
            realesrgan_model_file = os.path.join(MODELS_DIR, "realesrgan", f"RealESRGAN_x{upscale_factor}.pth")
            if os.path.exists(realesrgan_model_file):
                try:
                    upscale_model = load_sd_upscale(realesrgan_model_file, device)
                except Exception as e:
                    print(f"⚠️  Real-ESRGAN load failed: {str(e)[:100]}")
                    upscale_model = None
            else:
                print(f"⚠️  Real-ESRGAN {upscale_factor}x model not found")
        
        # Apply post-processing based on order
        if postprocess_interpolate_first:
            # Interpolate → Upscale
            if enable_rife and rife_model:
                try:
                    final_output, actual_fps = apply_rife_interpolation(
                        rife_model, final_output, device, original_fps=fps, target_fps=rife_target_fps
                    )
                    final_fps = actual_fps
                except Exception as e:
                    print(f"⚠️  RIFE interpolation failed: {str(e)[:100]}")
            
            if enable_upscaling and upscale_model:
                try:
                    final_output = apply_realesrgan_upscaling(upscale_model, final_output, device)
                    final_height = height * upscale_factor
                    final_width = width * upscale_factor
                except Exception as e:
                    print(f"⚠️  Upscaling failed: {str(e)[:100]}")
        else:
            # Upscale → Interpolate (default)
            if enable_upscaling and upscale_model:
                try:
                    final_output = apply_realesrgan_upscaling(upscale_model, final_output, device)
                    final_height = height * upscale_factor
                    final_width = width * upscale_factor
                except Exception as e:
                    print(f"⚠️  Upscaling failed: {str(e)[:100]}")
            
            if enable_rife and rife_model:
                try:
                    final_output, actual_fps = apply_rife_interpolation(
                        rife_model, final_output, device, original_fps=fps, target_fps=rife_target_fps
                    )
                    final_fps = actual_fps
                except Exception as e:
                    print(f"⚠️  RIFE interpolation failed: {str(e)[:100]}")
        
        postprocess_time = time.time() - postprocess_start
    
    # Export video
    out_path = os.path.join(output_dir, f"wan_i2v_{job_id}.mp4")
    export_start_time = time.time()
    export_to_video(final_output, out_path, fps=final_fps)
    export_time = time.time() - export_start_time
    
    total_time = generation_time + postprocess_time + export_time
    print(f"[{job_id}] ✓ Complete: {total_time:.1f}s (gen: {generation_time:.1f}s, post: {postprocess_time:.1f}s, export: {export_time:.1f}s) | {final_width}x{final_height} @ {final_fps}fps")
    
    # Post-inference GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    return os.path.abspath(out_path)

if __name__ == "__main__":
    job_id = "test"
    prompt = "animate this image with cinematic motion, smooth animation"
    params = {
        "num_frames": 81,
        "guidance_scale": 1.0,
        "num_inference_steps": 8,
        "seed": 42,
        "enable_rife": True,
        "enable_upscaling": True,
        "upscale_factor": 2,
        "rife_target_fps": 30,
        "resize_image": True,
    }
    # Toggle: Set to True to use local image, False to use remote image
    USE_LOCAL_IMAGE = True

    if USE_LOCAL_IMAGE:
        # Load image from local file "testimg.jpeg" in the same folder as this script
        local_image_path = os.path.join(os.path.dirname(__file__), "testimg.jpeg")
        input_img = Image.open(local_image_path).convert("RGB")
    else:
        url = "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/Snapinsta.app_262322643_4937545659591345_7487236998017046274_n_1080.jpg"
        with httpx.Client(timeout=60) as c:
            img_bytes = c.get(url).content
        input_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    out_path = run_inference_wan_22_i2v(job_id, prompt, params, input_img)
    print(f"Output video saved to: {out_path}")
