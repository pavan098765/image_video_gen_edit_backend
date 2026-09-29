# hunyan15_video_inference.py
# ✅ Hunyan15 480p Text-to-Video Inference Worker
# Model: hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled
# Works safely on RTX 4090 (24GB) and RTX 5090 (32GB)
# Specialized for Text-to-Video generation with step distillation
# Requires: kernels library for efficient attention backend

import gc
import os
import sys
import base64
import random
import tempfile
import time
from io import BytesIO
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import numpy as np
import torch
import imageio
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"


def _realesrgan_video_ckpt_name(upscale_factor: Any) -> str:
    """Align with rockapaper/realesrgan baked weights (x2plus / x4plus)."""
    try:
        u = int(upscale_factor)
    except (TypeError, ValueError):
        u = 4
    return "RealESRGAN_x2plus.pth" if u <= 2 else "RealESRGAN_x4plus.pth"

# # Disable PyTorch's automatic SDPA optimizations to prevent cutlassF errors
# # These settings force PyTorch to use eager attention instead of cutlassF kernels
# if hasattr(torch.backends.cuda, "enable_flash_sdp"):
#     torch.backends.cuda.enable_flash_sdp(False)
# if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
#     torch.backends.cuda.enable_mem_efficient_sdp(False)
# if hasattr(torch.backends.cuda, "enable_math_sdp"):
#     # Keep math SDP enabled as fallback (it's the default eager implementation)
#     torch.backends.cuda.enable_math_sdp(True)

# Aggressive GPU cleanup at import to prevent OOM
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# --- Auto-merge split safetensors if merge.py exists ---
try:
    from merge import main as merge_main
    print("🔍 Scanning for .safetensors.part-* files...")
    merge_start = time.time()
    merge_main()
    merge_time = time.time() - merge_start
    print(f"✅ Merge completed in {merge_time:.1f}s")
except Exception as e:
    print(f"⚠️ merge.py auto-merge skipped or failed: {e}")


# ==============================
# Environment Configuration
# ==============================
# HuggingFace repo ID for Hunyan15 model (for reference only - not used for loading)
HUNYAN15_REPO_ID = os.environ.get(
    "HUNYAN15_REPO_ID",
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled"
)

# Local model paths (REQUIRED - models must be pre-downloaded, no HuggingFace downloads)
MODEL_DIR = os.environ.get("HUNYAN15_T2V_LOCAL_PATH", "/app/models/hunyan15_t2v")
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")

# Inference defaults (480p model)
DEFAULT_FPS = 24
MAX_SEED = np.iinfo(np.int32).max
MAX_HEIGHT = 480   # 480p vertical resolution
MAX_WIDTH = 864    # 480p 16:9 width (mod 16)
MAX_FRAMES = 240   # max 8 seconds at 24fps
MIN_DURATION = 4   # seconds
MAX_DURATION = 10   # seconds

# Step distilled model uses only 50 steps
NUM_INFERENCE_STEPS = 50

# ==============================
# Pipeline Cache
# ==============================
PIPELINE_CACHE: Dict[str, Any] = {}


# ==============================
# Helper Functions
# ==============================
def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"available": False, "vram_gb": 0, "name": "CPU"}
    
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    
    return {
        "available": True,
        "vram_gb": vram_gb,
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
    }


def cleanup_gpu_memory(verbose: bool = False):
    """Thorough GPU memory cleanup."""
    try:
        if verbose:
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"[GPU] Before cleanup: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()

        if verbose:
            print(f"[GPU] After cleanup: {torch.cuda.memory_allocated()/1024**2:.2f} MB allocated")
    except Exception as e:
        print(f"[GPU] Cleanup warning: {e}")


def calculate_optimal_frames(duration_sec: float, fps: int = DEFAULT_FPS) -> int:
    """
    Calculate optimal frame count for Hunyan15.
    Standard frame calculation: duration * fps
    """
    target_frames = max(1, min(MAX_FRAMES, int(duration_sec * fps)))
    return target_frames


def pad_dimensions_to_8(height: int, width: int) -> Tuple[int, int]:
    """Pad dimensions to be divisible by 8."""
    h_pad = ((height - 1) // 8 + 1) * 8
    w_pad = ((width - 1) // 8 + 1) * 8
    return h_pad, w_pad


def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def export_video(
    frames: np.ndarray, 
    output_path: str, 
    fps: int = DEFAULT_FPS
) -> str:
    """Export numpy array of frames to video file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    with imageio.get_writer(output_path, fps=fps, macro_block_size=1) as writer:
        for i in range(frames.shape[0]):
            writer.append_data(frames[i])
    
    return output_path


# ==============================
# Pipeline Loading
# ==============================
def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_hunyan15_pipeline(cache_key: str = "hunyan15_t2v_distilled"):
    """
    Load or retrieve cached Hunyan15 Text-to-Video pipeline.
    
    Returns:
        Dict with keys: pipeline, device, gpu_info
    """
    global PIPELINE_CACHE
    
    if cache_key in PIPELINE_CACHE:
        print(f"📦 Using cached pipeline: {cache_key}")
        sys.stdout.flush()
        return PIPELINE_CACHE[cache_key]
    
    print(f"🚀 Loading Hunyan15 t2v pipeline from local files only")
    print(f"   Model directory: {MODEL_DIR}")
    print(f"   (Reference repo: {HUNYAN15_REPO_ID})")
    sys.stdout.flush()
    
    device = get_device()
    gpu_info = get_gpu_info()
    dtype = torch.bfloat16
    
    print(f"   Device: {device}")
    print(f"   Data type: {dtype}")
    print(f"   GPU: {gpu_info}")
    sys.stdout.flush()
    
    start_time = time.time()
    print(f"⏳ Starting GPU cleanup...")
    sys.stdout.flush()
    cleanup_gpu_memory(verbose=True)
    
    try:
        print(f"📦 Importing diffusers modules...")
        sys.stdout.flush()
        from diffusers import HunyuanVideo15Pipeline, attention_backend
        from diffusers.utils import export_to_video
        print(f"✅ Diffusers imported successfully")
        sys.stdout.flush()
        
        # CRITICAL: Only use local files - never download from HuggingFace
        if not os.path.exists(MODEL_DIR):
            error_msg = (
                f"❌ Model directory not found: {MODEL_DIR}\n"
                f"   Models must be pre-downloaded to this path.\n"
                f"   Run prep_model.py to download models first."
            )
            print(error_msg)
            sys.stdout.flush()
            raise FileNotFoundError(error_msg)
        
        try:
            files = os.listdir(MODEL_DIR)
            if not files:
                error_msg = (
                    f"❌ Model directory is empty: {MODEL_DIR}\n"
                    f"   Models must be pre-downloaded to this path.\n"
                    f"   Run prep_model.py to download models first."
                )
                print(error_msg)
                sys.stdout.flush()
                raise FileNotFoundError(error_msg)
        except Exception as e:
            error_msg = f"❌ Error accessing model directory {MODEL_DIR}: {e}"
            print(error_msg)
            sys.stdout.flush()
            raise
        
        print(f"📁 Using LOCAL model files only (no HuggingFace download)")
        print(f"   Model directory: {MODEL_DIR}")
        print(f"   Files found: {len(files)} items")
        sys.stdout.flush()
        
        # Load pipeline from LOCAL files only
        print(f"⏳ Loading pipeline from local files (this may take 1-3 minutes)...")
        print(f"   Path: {MODEL_DIR}")
        print(f"   ⚠️  This is a blocking operation - logs may pause here...")
        sys.stdout.flush()
        
        load_start = time.time()
        
        # Wrap the loading in a try-except to catch any errors
        try:
            # ALWAYS use local_files_only=True to prevent any downloads
            pipeline = HunyuanVideo15Pipeline.from_pretrained(
                MODEL_DIR,
                torch_dtype=dtype,
                local_files_only=True  # CRITICAL: Never download
            )
        except Exception as load_error:
            elapsed = time.time() - load_start
            error_msg = (
                f"❌ Pipeline loading failed after {elapsed:.1f}s: {load_error}\n"
                f"   Model path: {MODEL_DIR}\n"
                f"   Make sure models are properly downloaded via prep_model.py"
            )
            print(error_msg)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise FileNotFoundError(error_msg) from load_error
        
        load_elapsed = time.time() - load_start
        print(f"✅ Pipeline.from_pretrained() completed in {load_elapsed:.1f}s")
        print(f"   Source: local files only (no download)")
        sys.stdout.flush()
        
        print(f"⚙️  Enabling memory-efficient operations...")
        sys.stdout.flush()
        pipeline.enable_model_cpu_offload()
        print(f"   ✅ CPU offload enabled")
        sys.stdout.flush()
        
        pipeline.vae.enable_tiling()
        print(f"   ✅ VAE tiling enabled")
        sys.stdout.flush()
        
        elapsed = time.time() - start_time
        print(f"✅ Hunyan15 t2v pipeline fully loaded in {elapsed:.1f}s")
        sys.stdout.flush()
        
        PIPELINE_CACHE[cache_key] = {
            "pipeline": pipeline,
            "device": device,
            "gpu_info": gpu_info,
            "dtype": dtype,
        }
        
        return PIPELINE_CACHE[cache_key]
        
    except ImportError as e:
        print(f"❌ diffusers not installed! Error: {e}")
        sys.stdout.flush()
        raise ImportError(
            f"Hunyan15 requires diffusers>=0.31.0. "
            f"Install with: pip install 'diffusers[torch]>=0.31.0'. "
            f"Original error: {e}"
        )
    except Exception as e:
        print(f"❌ Failed to load pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise


def preload_pipeline(pipeline_type: Optional[str] = None):
    """
    Preload pipeline in background.
    
    Args:
        pipeline_type: Optional pipeline type (for compatibility, not used for Hunyan15)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print("=" * 80)
        print("PIPELINE PRELOAD: Starting pipeline preload...")
        print("=" * 80)
        sys.stdout.flush()
        
        start_time = time.time()
        result = get_hunyan15_pipeline()
        
        elapsed = time.time() - start_time
        print("=" * 80)
        print(f"✅ Pipeline preloaded successfully in {elapsed:.1f}s")
        print("=" * 80)
        sys.stdout.flush()
        return True
    except Exception as e:
        print("=" * 80)
        print(f"❌ Pipeline preload failed: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return False


def unload_pipeline():
    """Unload pipeline(s) and free GPU memory."""
    global PIPELINE_CACHE
    PIPELINE_CACHE.clear()
    print("🧹 Unloaded all pipelines")
    cleanup_gpu_memory(verbose=True)


# ==============================
# Inference Functions
# ==============================
def run_inference_hunyan15_t2v(
    job_id: str,
    prompt: str,
    params: Dict[str, Any]
) -> str:
    """
    Text-to-Video generation using Hunyan15 480p step distilled model.
    
    Args:
        job_id: Unique job identifier
        prompt: Text prompt describing the video
        params:
            - duration (float): Video duration in seconds (4-8). If set, frames = duration * fps.
            - num_frames (int): Number of frames (default: 121, used when duration not set)
            - num_inference_steps (int): Denoising steps (default: 50)
            - seed (int): Random seed
            - fps (int): Frames per second (default: 24)
            - output_dir (str): Output directory
            Note: guidance_scale is not supported by HunyuanVideo15Pipeline
    
    Returns:
        Absolute path to generated video
    """
    import logging
    logger = logging.getLogger("worker")
    
    logger.info(f"[{job_id}] 🎬 Starting Text-to-Video inference")
    print(f"🎬 Starting Text-to-Video job: {job_id}")
    sys.stdout.flush()
    
    cleanup_gpu_memory()
    
    # Parameters
    fps = int(params.get("fps", DEFAULT_FPS))
    duration = params.get("duration")
    if duration is not None:
        # Duration-based: clamp to [4, 8] sec, compute frames at fps (max 240 at 24fps)
        duration_sec = max(MIN_DURATION, min(MAX_DURATION, float(duration)))
        num_frames = max(1, min(MAX_FRAMES, int(duration_sec * fps)))
    else:
        num_frames = int(params.get("num_frames", 121))
    num_inference_steps = int(params.get("num_inference_steps", NUM_INFERENCE_STEPS))
    # Note: guidance_scale is not supported by HunyuanVideo15Pipeline
    seed = int(params.get("seed", random.randint(0, MAX_SEED)))
    output_dir = params.get("output_dir", "/app/worker/outputs")
    
    # Validate parameters
    num_frames = max(1, min(num_frames, MAX_FRAMES))
    num_inference_steps = max(1, num_inference_steps)
    
    logger.info(f"[{job_id}] 📐 Configuration: frames={num_frames}, steps={num_inference_steps}, fps={fps}, seed={seed}")
    print(f"📐 Configuration:")
    print(f"   Frames: {num_frames}")
    print(f"   Steps: {num_inference_steps}")
    print(f"   FPS: {fps}")
    print(f"   Seed: {seed}")
    print(f"   Output directory: {output_dir}")
    sys.stdout.flush()
    
    try:
        seed_everything(seed)
        
        # Load pipeline
        logger.info(f"[{job_id}] Loading pipeline...")
        data = get_hunyan15_pipeline()
        pipeline = data["pipeline"]
        device = data["device"]
        dtype = data["dtype"]
        logger.info(f"[{job_id}] Pipeline loaded successfully")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"t2v_{job_id}.mp4")
        logger.info(f"[{job_id}] Output path: {output_path}")
        print(f"📁 Output path: {output_path}")
        print(f"   Directory exists: {os.path.exists(output_dir)}")
        sys.stdout.flush()
        
        logger.info(f"[{job_id}] 🚀 Starting video generation (seed={seed})...")
        print(f"🚀 Generating video with seed={seed}...")
        print(f"   Prompt: {prompt[:100]}...")
        sys.stdout.flush()
        
        from diffusers import attention_backend
        
        # Generate video using attention backend (use flash_hub for non-H100/H800)
        logger.info(f"[{job_id}] 🎥 Starting video generation (this may take 1-3 minutes)...")
        print(f"🎥 Starting video generation (this may take 1-3 minutes)...")
        sys.stdout.flush()
        
        try:
            with attention_backend("flash_hub"):  # Use flash_hub for RTX 4090/5090
                generator = torch.Generator(device=device).manual_seed(seed)
                
                logger.info(f"[{job_id}] Calling pipeline.generate()...")
                print(f"   Calling pipeline.generate()...")
                print(f"   Inference will run for {num_inference_steps} steps (this may take 2-5 minutes)...")
                sys.stdout.flush()
                
                # Progress callback to log inference steps
                inference_start_time = time.time()
                last_log_time = inference_start_time
                log_interval = 10  # Log every 10 seconds
                
                def progress_callback(step: int, timestep: int, callback_kwargs: dict):
                    """Callback to log inference progress."""
                    nonlocal last_log_time
                    current_time = time.time()
                    elapsed = current_time - inference_start_time
                    progress_pct = (step + 1) / num_inference_steps * 100
                    
                    # Log every 10 seconds or every 10% progress
                    if (current_time - last_log_time >= log_interval) or (step % max(1, num_inference_steps // 10) == 0):
                        elapsed_str = f"{elapsed:.1f}s"
                        eta = (elapsed / (step + 1)) * (num_inference_steps - step - 1) if step > 0 else 0
                        eta_str = f"{eta:.1f}s" if eta > 0 else "calculating..."
                        logger.info(f"[{job_id}] 🔄 Inference progress: {step + 1}/{num_inference_steps} steps ({progress_pct:.1f}%) | Elapsed: {elapsed_str} | ETA: {eta_str}")
                        print(f"   🔄 Step {step + 1}/{num_inference_steps} ({progress_pct:.1f}%) | Elapsed: {elapsed_str} | ETA: {eta_str}")
                        sys.stdout.flush()
                        last_log_time = current_time
                    
                    return callback_kwargs
                
                # Try to use callback if supported, otherwise just log before/after
                try:
                    result = pipeline(
                        prompt=prompt,
                        generator=generator,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        callback_on_step_end=progress_callback,
                    )
                except (TypeError, AttributeError) as e:
                    # Callback not supported, use periodic logging instead
                    logger.info(f"[{job_id}] Pipeline doesn't support callback (reason: {type(e).__name__}), using time-based estimation...")
                    print(f"   ⚠️ Note: Progress shown is ESTIMATED based on elapsed time, not actual step count")
                    print(f"   Starting inference (will log estimated progress every 10 seconds)...")
                    sys.stdout.flush()
                    
                    # Start a background thread to log progress periodically
                    import threading
                    progress_logging_active = threading.Event()
                    progress_logging_active.set()
                    
                    # Adaptive estimation: start with conservative estimate, adjust based on typical times
                    # Typical: 50 steps takes ~100-150s on RTX 4090/5090
                    estimated_total_time = 120.0  # Initial estimate: 2 minutes for 50 steps
                    
                    def periodic_progress_logger():
                        step_count = 0
                        while progress_logging_active.is_set() and step_count < num_inference_steps:
                            time.sleep(log_interval)
                            if progress_logging_active.is_set():
                                elapsed = time.time() - inference_start_time
                                # Estimate current step based on elapsed time
                                # Use linear interpolation: if we're X% through time, assume X% through steps
                                estimated_step = min(int((elapsed / estimated_total_time) * num_inference_steps), num_inference_steps - 1)
                                progress_pct = (estimated_step + 1) / num_inference_steps * 100
                                
                                # Calculate ETA based on current rate
                                if elapsed > 5:  # After 5 seconds, we can estimate
                                    steps_per_second = estimated_step / elapsed if elapsed > 0 else 0
                                    if steps_per_second > 0:
                                        remaining_steps = num_inference_steps - estimated_step
                                        eta_seconds = remaining_steps / steps_per_second
                                        eta_str = f"{eta_seconds:.1f}s"
                                    else:
                                        eta_str = "calculating..."
                                else:
                                    eta_str = "calculating..."
                                
                                logger.info(f"[{job_id}] 🔄 Inference in progress (ESTIMATED): ~{estimated_step + 1}/{num_inference_steps} steps ({progress_pct:.1f}%) | Elapsed: {elapsed:.1f}s | ETA: {eta_str}")
                                print(f"   🔄 Inference in progress (ESTIMATED): ~{estimated_step + 1}/{num_inference_steps} steps ({progress_pct:.1f}%) | Elapsed: {elapsed:.1f}s | ETA: {eta_str}")
                                sys.stdout.flush()
                    
                    progress_thread = threading.Thread(target=periodic_progress_logger, daemon=True)
                    progress_thread.start()
                    
                    result = pipeline(
                        prompt=prompt,
                        generator=generator,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                    )
                    
                    progress_logging_active.clear()
                    
                    # Update estimated total time for next run (adaptive learning)
                    actual_time = time.time() - inference_start_time
                    if actual_time > 0:
                        estimated_total_time = actual_time  # Use actual time for future estimates
                
                inference_elapsed = time.time() - inference_start_time
                logger.info(f"[{job_id}] ✅ Pipeline generation completed in {inference_elapsed:.1f}s, extracting frames...")
                print(f"   ✅ Pipeline generation completed in {inference_elapsed:.1f}s, extracting frames...")
                sys.stdout.flush()
                
                video_frames = result.frames[0]
                num_frames_generated = len(video_frames) if hasattr(video_frames, '__len__') else 'N/A'
                logger.info(f"[{job_id}] Frames extracted: {num_frames_generated} frames")
                print(f"   Frames extracted: {num_frames_generated} frames")
                sys.stdout.flush()
            
            # Post-processing (RIFE interpolation and/or Real-ESRGAN upscaling)
            postprocess_time = 0.0
            final_fps = fps
            final_output = video_frames
            final_height = None
            final_width = None
            
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Coerce bools so "true"/"false" from JSON work; support enable_upscale alias
            def _bool_param(p, key, default=False):
                v = p.get(key, default)
                if isinstance(v, bool):
                    return v
                if isinstance(v, str):
                    return v.strip().lower() in ("true", "1", "yes")
                return bool(v)

            enable_rife = _bool_param(params, "enable_rife", False)
            rife_target_fps = int(params.get("rife_target_fps", 30))
            force_target_fps = _bool_param(params, "force_target_fps", True)
            enable_upscaling = _bool_param(params, "enable_upscaling", False) or _bool_param(params, "enable_upscale", False)
            upscale_factor = int(params.get("upscale_factor", 2))
            postprocess_interpolate_first = _bool_param(params, "postprocess_interpolate_first", False)

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
                    realesrgan_model_file = os.path.join(
                        MODELS_DIR,
                        "realesrgan",
                        _realesrgan_video_ckpt_name(upscale_factor),
                    )
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
                                rife_model, final_output, device, original_fps=fps, target_fps=rife_target_fps, force_target_fps=force_target_fps
                            )
                            final_fps = actual_fps
                        except Exception as e:
                            print(f"⚠️  RIFE interpolation failed: {str(e)[:100]}")
                    
                    if enable_upscaling and upscale_model:
                        try:
                            final_output = apply_realesrgan_upscaling(upscale_model, final_output, device)
                        except Exception as e:
                            print(f"⚠️  Upscaling failed: {str(e)[:100]}")
                else:
                    # Upscale → Interpolate (default)
                    if enable_upscaling and upscale_model:
                        try:
                            final_output = apply_realesrgan_upscaling(upscale_model, final_output, device)
                        except Exception as e:
                            print(f"⚠️  Upscaling failed: {str(e)[:100]}")
                    
                    if enable_rife and rife_model:
                        try:
                            final_output, actual_fps = apply_rife_interpolation(
                                rife_model, final_output, device, original_fps=fps, target_fps=rife_target_fps, force_target_fps=force_target_fps
                            )
                            final_fps = actual_fps
                        except Exception as e:
                            print(f"⚠️  RIFE interpolation failed: {str(e)[:100]}")
                
                postprocess_time = time.time() - postprocess_start
            
            # Export video
            logger.info(f"[{job_id}] 💾 Exporting video to {output_path}...")
            print(f"💾 Exporting video to {output_path}...")
            sys.stdout.flush()
            
            from diffusers.utils import export_to_video
            export_to_video(final_output, output_path, fps=final_fps)
            
            # Verify file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / 1024 / 1024  # MB
                logger.info(f"[{job_id}] ✅ Video generated successfully: {output_path} ({file_size:.2f} MB)")
                print(f"✅ Video generated successfully: {output_path} ({file_size:.2f} MB)")
            else:
                error_msg = f"Video file not found at expected path: {output_path}"
                logger.error(f"[{job_id}] ⚠️  WARNING: {error_msg}")
                print(f"⚠️  WARNING: {error_msg}")
            sys.stdout.flush()
            
            return output_path
            
        except Exception as pipeline_error:
            error_type = type(pipeline_error).__name__
            error_msg = f"Pipeline generation failed: {error_type}: {pipeline_error}"
            logger.error(f"[{job_id}] ❌ {error_msg}")
            logger.exception(f"[{job_id}] Full traceback:")
            print(f"❌ {error_msg}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            sys.stdout.flush()
            raise
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = f"Inference failed: {error_type}: {e}"
        logger.error(f"[{job_id}] ❌ {error_msg}")
        logger.exception(f"[{job_id}] Full traceback:")
        print(f"❌ {error_msg}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        logger.info(f"[{job_id}] 🧹 Cleaning up GPU memory...")
        print(f"🧹 Cleaning up GPU memory...")
        sys.stdout.flush()
        cleanup_gpu_memory(verbose=True)
