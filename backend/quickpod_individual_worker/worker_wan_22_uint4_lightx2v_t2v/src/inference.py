#!/usr/bin/env python3
"""
Inference module for WAN 2.2 T2V 14B SDNQ UINT4 Lightx2v worker.
Based on wan_22_14b_lightx2v_t2v_sdnq_inference.py from work_space.
"""

import os
import time
import torch
import traceback
import numpy as np
import diffusers
from diffusers.utils import export_to_video
from sdnq import SDNQConfig  # import sdnq to register it into diffusers and transformers
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
from typing import Dict, Any, Optional

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
    
    # Fix CUSOLVER errors by setting preferred linear algebra library
    try:
        if hasattr(torch.backends.cuda, 'preferred_linalg_library'):
            try:
                torch.backends.cuda.preferred_linalg_library('cusolver')
            except (ValueError, RuntimeError):
                try:
                    torch.backends.cuda.preferred_linalg_library('magma')
                except (ValueError, RuntimeError):
                    try:
                        torch.backends.cuda.preferred_linalg_library('cublaslt')
                    except (ValueError, RuntimeError):
                        pass
    except AttributeError:
        pass

# Configuration
MODEL_REPO = "Disty0/Wan2.2-T2V-A14B-SDNQ-uint4-svd-r32"
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
MODEL_LOCAL_DIR = os.path.join(MODELS_DIR, "wan2.2_t2v_sdnq_uint4_svd_r32")

# LoRA Configuration
LORA_REPO = "Kijai/WanVideo_comfy"
LORA_WEIGHT_NAME = "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors"
LORA_LOCAL_DIR = os.path.join(MODELS_DIR, "lora", "Kijai_WanVideo_comfy")

# Pipeline cache
PIPELINE_CACHE = {}
PIPELINE_LOADED = False

# Default inference parameters
DEFAULT_NEGATIVE_PROMPT = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景, 三条腿, 背景人很多, 倒着走, 阴道变形, 乳房变形, 乳头畸形, 变形的皮肤, 大腿畸形，臀部畸, 面部畸形，眼睛畸形, 膝盖畸形，髋关节畸形"

DEFAULT_FPS = 16
DEFAULT_NUM_FRAMES = 81
DEFAULT_NUM_INFERENCE_STEPS = 8
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_GUIDANCE_SCALE_2 = 1.0


def load_pipeline():
    """Load and cache the T2V pipeline."""
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
    
    # Load VAE separately (required for T2V pipeline)
    vae = diffusers.AutoencoderKLWan.from_pretrained(
        MODEL_LOCAL_DIR,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True  # Ensure we only use local files
    )
    
    # Load pipeline
    pipe = diffusers.WanPipeline.from_pretrained(
        MODEL_LOCAL_DIR,
        vae=vae,
        torch_dtype=torch.bfloat16,
        local_files_only=True  # Ensure we only use local files
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
    
    pipe.to(device)
    
    # Load Lightning LoRA
    lora_path = os.path.join(LORA_LOCAL_DIR, LORA_WEIGHT_NAME)
    lora_loaded = False
    
    if os.path.exists(lora_path):
        try:
            lora_dir = os.path.dirname(lora_path)
            lora_filename = os.path.basename(lora_path)
            
            if "Lightx2v" in lora_path or "lightx2v" in os.path.dirname(lora_path):
                parent_dir = os.path.dirname(lora_dir)
                subdir_name = os.path.basename(lora_dir)
                weight_name_with_subdir = os.path.join(subdir_name, lora_filename)
                
                pipe.load_lora_weights(
                    parent_dir,
                    weight_name=weight_name_with_subdir,
                    adapter_name="lightx2v"
                )
                
                kwargs_lora = {"load_into_transformer_2": True}
                pipe.load_lora_weights(
                    parent_dir,
                    weight_name=weight_name_with_subdir,
                    adapter_name="lightx2v_2",
                    **kwargs_lora
                )
            else:
                pipe.load_lora_weights(
                    lora_dir,
                    weight_name=lora_filename,
                    adapter_name="lightx2v"
                )
                
                kwargs_lora = {"load_into_transformer_2": True}
                pipe.load_lora_weights(
                    lora_dir,
                    weight_name=lora_filename,
                    adapter_name="lightx2v_2",
                    **kwargs_lora
                )
            
            pipe.set_adapters(["lightx2v", "lightx2v_2"], adapter_weights=[1., 1.])
            lora_loaded = True
        except Exception as e:
            print(f"⚠️  Failed to load LoRA: {str(e)[:200]}")
            try:
                pipe.unload_lora_weights()
            except:
                pass
    
    # Apply speed optimizations
    pipe.enable_model_cpu_offload()
    
    flash_attn_available = False
    try:
        import flash_attn
        flash_attn_available = True
    except ImportError:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    
    load_time = time.time() - load_start_time
    print(f"✓ T2V Pipeline loaded ({load_time:.1f}s)")
    
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
        "lora_loaded": lora_loaded,
        "flash_attn_available": flash_attn_available
    }
    PIPELINE_LOADED = True
    
    return PIPELINE_CACHE[MODEL_LOCAL_DIR]


def run_inference_wan_22_t2v(
    job_id: str,
    prompt: str,
    params: Dict[str, Any]
) -> str:
    """
    Run text-to-video inference.
    
    Args:
        job_id: Job identifier
        prompt: Text prompt
        params: Inference parameters (height, width, num_frames, steps, guidance_scale, seed, etc.)
    
    Returns:
        Path to output video file
    """
    # Pre-inference GPU cleanup
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # Load pipeline
    pipeline_data = load_pipeline()
    pipe = pipeline_data["pipe"]
    
    # Extract parameters
    num_frames = int(params.get("num_frames", DEFAULT_NUM_FRAMES))
    guidance_scale = float(params.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
    guidance_scale_2 = float(params.get("guidance_scale_2", DEFAULT_GUIDANCE_SCALE_2))
    num_inference_steps = int(params.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS))
    seed = params.get("seed", None)
    if seed is None:
        seed = 0
    else:
        seed = int(seed)
    
    height = int(params.get("height", 480))
    width = int(params.get("width", 832))
    negative_prompt = params.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    output_dir = params.get("output_dir", "/workspace")
    os.makedirs(output_dir, exist_ok=True)
    
    # Post-processing options
    enable_rife = params.get("enable_rife", False)
    rife_target_fps = params.get("rife_target_fps", 30)
    enable_upscaling = params.get("enable_upscaling", False)
    upscale_factor = int(params.get("upscale_factor", 2))
    postprocess_interpolate_first = params.get("postprocess_interpolate_first", False)
    
    fps = params.get("fps", DEFAULT_FPS)
    
    # Generate video
    print(f"[{job_id}] Generating T2V: {width}x{height}, {num_frames}f @ {fps}fps, {num_inference_steps} steps")
    
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    
    generation_start_time = time.time()
    
    # Ensure CUDA is fully initialized and warm up linear algebra operations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            _ = torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
        except:
            pass
        
        # Warm up torch.linalg.solve to initialize CUSOLVER before inference
        try:
            warmup_A = torch.randn(8, 8, device="cuda", dtype=torch.float32)
            warmup_b = torch.randn(8, device="cuda", dtype=torch.float32)
            warmup_A = warmup_A @ warmup_A.T + torch.eye(8, device="cuda", dtype=torch.float32) * 0.1
            with torch.inference_mode():
                _ = torch.linalg.solve(warmup_A, warmup_b)
            torch.cuda.synchronize()
        except Exception:
            pass
    
    # Retry logic for CUSOLVER errors
    max_retries = 3
    backends_to_try = ['cusolver', 'magma', 'cublaslt']
    
    for attempt in range(max_retries):
        try:
            with torch.inference_mode():
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    guidance_scale_2=guidance_scale_2,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                ).frames[0]
            # Success - break out of retry loop
            break
        except RuntimeError as e:
            error_str = str(e)
            if "CUSOLVER" in error_str or "cusolver" in error_str.lower():
                if attempt < max_retries - 1 and device == "cuda" and hasattr(torch.backends.cuda, 'preferred_linalg_library'):
                    # Try switching to a different backend
                    backend_switched = False
                    for backend in backends_to_try:
                        try:
                            torch.backends.cuda.preferred_linalg_library(backend)
                            backend_switched = True
                            break
                        except (ValueError, RuntimeError):
                            continue
                    
                    if not backend_switched:
                        raise RuntimeError(
                            "CUSOLVER error persists with all backends. "
                            "This may be a GPU driver or CUDA installation issue."
                        ) from e
                else:
                    raise
            else:
                raise
    
    generation_time = time.time() - generation_start_time
    
    # Post-processing (RIFE interpolation and/or Real-ESRGAN upscaling)
    postprocess_time = 0.0
    final_fps = fps
    final_output = output
    final_height = height
    final_width = width
    
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
    out_path = os.path.join(output_dir, f"wan_t2v_{job_id}.mp4")
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
