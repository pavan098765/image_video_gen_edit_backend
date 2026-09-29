# hunyan15_video_inference.py
# ✅ Hunyan15 480p Image-to-Video Inference Worker
# Model: hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled
# Works safely on RTX 4090 (24GB) and RTX 5090 (32GB)
# Specialized for Image-to-Video generation with step distillation

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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"


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
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled"
)

# Local model paths (REQUIRED - models must be pre-downloaded, no HuggingFace downloads)
MODEL_DIR = os.environ.get("HUNYAN15_MODEL_LOCAL_PATH", "/app/models/hunyan15")
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")

# Inference defaults (480p model)
DEFAULT_FPS = 24
MAX_SEED = np.iinfo(np.int32).max
MAX_HEIGHT = 480   # 480p vertical resolution
MAX_WIDTH = 864   # 480p 16:9 width (mod 16)
MAX_FRAMES = 240  # max 8 seconds at 24fps
MIN_DURATION = 4  # seconds
MAX_DURATION = 10  # seconds

# Step distilled model uses only 12 steps
NUM_INFERENCE_STEPS = 12

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


def load_image_from_base64(img_b64: str) -> Image.Image:
    """Load PIL Image from base64 string."""
    return Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")


def load_image_from_url(image_url: str) -> Image.Image:
    """Load PIL Image from URL."""
    from diffusers.utils import load_image
    return load_image(image_url).convert("RGB")


def load_video_from_base64(video_b64: str, max_frames: int = 121) -> List[Image.Image]:
    """Load video frames from base64 encoded video."""
    video_bytes = base64.b64decode(video_b64)
    tmp_path = f"/tmp/temp_video_{random.randint(0, 99999)}.mp4"
    
    try:
        with open(tmp_path, "wb") as f:
            f.write(video_bytes)
        
        reader = imageio.get_reader(tmp_path)
        frames = []
        for i, frame in enumerate(reader):
            if i >= max_frames:
                break
            frames.append(Image.fromarray(frame).convert("RGB"))
        reader.close()
        return frames
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def resize_image_keep_aspect(
    image: Image.Image,
    target_height: int,
    target_width: int
) -> Image.Image:
    """
    Resize image to fit within target dimensions maintaining aspect ratio.
    No cropping, no padding. Output dimensions may differ from target
    but will always fit within target_width x target_height and be divisible by 8.
    """
    input_width, input_height = image.size

    # Scale to fit within target box
    scale = min(target_width / input_width, target_height / input_height)
    
    new_width = int(input_width * scale)
    new_height = int(input_height * scale)

    # Align to 8 for model compatibility
    new_width = max(8, (new_width // 8) * 8)
    new_height = max(8, (new_height // 8) * 8)

    return image.resize((new_width, new_height), Image.LANCZOS)

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


def get_hunyan15_pipeline(cache_key: str = "hunyan15_480p_i2v_step_distilled"):
    """
    Load or retrieve cached Hunyan15 pipeline.
    
    Returns:
        Dict with keys: pipeline, device, gpu_info
    """
    global PIPELINE_CACHE
    
    if cache_key in PIPELINE_CACHE:
        print(f"📦 Using cached pipeline: {cache_key}")
        sys.stdout.flush()
        return PIPELINE_CACHE[cache_key]
    
    print(f"🚀 Loading Hunyan15 i2v pipeline from local files only")
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
        from diffusers import HunyuanVideo15ImageToVideoPipeline, attention_backend
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
            # Use eager attention to avoid cutlassF kernel errors
            pipeline = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
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
        print(f"✅ Hunyan15 i2v pipeline fully loaded in {elapsed:.1f}s")
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
def run_inference_hunyan15_i2v(
    job_id: str,
    prompt: str,
    image_input: str,
    params: Dict[str, Any]
) -> str:
    """
    Image-to-Video generation using Hunyan15 480p step distilled model.
    
    Args:
        job_id: Unique job identifier
        prompt: Text prompt describing the animation
        image_input: Presigned https URL to the input image (or local file path for dev)
        params:
            - duration (float): Video duration in seconds (4-8). If set, frames = duration * fps.
            - num_frames (int): Number of frames (default: 121, used when duration not set)
            - num_inference_steps (int): Denoising steps (default: 12)
            - seed (int): Random seed
            - fps (int): Frames per second (default: 24)
            - output_dir (str): Output directory
            Note: guidance_scale is not supported by HunyuanVideo15ImageToVideoPipeline
    
    Returns:
        Absolute path to generated video
    """
    import logging
    logger = logging.getLogger("worker")
    
    logger.info(f"[{job_id}] 🎬 Starting Image-to-Video inference")
    print(f"🎬 Starting Image-to-Video job: {job_id}")
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
    # Note: guidance_scale is not supported by HunyuanVideo15ImageToVideoPipeline
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
        output_path = os.path.join(output_dir, f"i2v_{job_id}.mp4")
        logger.info(f"[{job_id}] Output path: {output_path}")
        print(f"📁 Output path: {output_path}")
        print(f"   Directory exists: {os.path.exists(output_dir)}")
        sys.stdout.flush()
        
        # Load and prepare input image
        logger.info(f"[{job_id}] Loading input image...")
        print(f"🖼️  Loading input image...")
        sys.stdout.flush()
        
        try:
            if image_input.startswith("http://") or image_input.startswith("https://"):
                image = load_image_from_url(image_input)
            elif os.path.isfile(image_input):
                image = Image.open(image_input).convert("RGB")
            else:
                raise ValueError(
                    "image_input must be an http(s) presigned URL or an existing file path"
                )
            logger.info(f"[{job_id}] Image loaded successfully: {image.size}")
        except Exception as e:
            error_msg = f"Failed to load input image: {e}"
            logger.error(f"[{job_id}] ❌ {error_msg}")
            print(f"❌ {error_msg}")
            sys.stdout.flush()
            raise
        
        print(f"   Image size: {image.size}")
        image = resize_image_keep_aspect(image, target_height=MAX_HEIGHT, target_width=MAX_WIDTH)
        print(f"   Resized to: {image.size}")  # Will always be 864×480
        sys.stdout.flush()
        
        logger.info(f"[{job_id}] 🚀 Starting video generation (seed={seed})...")
        print(f"🚀 Generating video with seed={seed}...")
        print(f"   Prompt: {prompt[:100]}...")
        sys.stdout.flush()
        
        from diffusers import attention_backend
        
        # Generate video using attention backend
        # Note: HunyuanVideo15ImageToVideoPipeline doesn't support guidance_scale parameter
        logger.info(f"[{job_id}] 🎥 Starting video generation (this may take 1-3 minutes)...")
        print(f"🎥 Starting video generation (this may take 1-3 minutes)...")
        sys.stdout.flush()
        
        try:
            with attention_backend("flash_hub"):  # or "flash_hub"/"flash_varlen_hub" if not on H100/H800
                generator = torch.Generator(device=device).manual_seed(seed)
                
                logger.info(f"[{job_id}] Calling pipeline.generate()...")
                print(f"   Calling pipeline.generate()...")
                sys.stdout.flush()
                
                result = pipeline(
                    prompt=prompt,
                    image=image,
                    generator=generator,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                )
                
                logger.info(f"[{job_id}] Pipeline generation completed, extracting frames...")
                print(f"   Pipeline generation completed, extracting frames...")
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


def run_inference_hunyan15_video(
    job_id: str,
    prompt: str,
    params: Dict[str, Any]
) -> str:
    """
    Wrapper for backward compatibility.
    Delegates to run_inference_hunyan15_i2v.
    """
    image_input = params.pop("image_input", None)
    if not image_input:
        raise ValueError("image_input is required in params for Hunyan15 model")
    
    return run_inference_hunyan15_i2v(job_id, prompt, image_input, params)


# ==============================
# Auto-test (run when file is executed directly)
# ==============================
if __name__ == "__main__":
    import requests
    
    print("=" * 80)
    print("Hunyan15 480p I2V Step Distilled - Auto-test")
    print("=" * 80)
    print("\n💡 This test will:")
    print("   1. Download a test image from Unsplash")
    print("   2. Generate a short video (25 frames, ~1 second)")
    print("   3. Verify the output file is created")
    print("   4. Report generation statistics")
    print("\n⏳ This may take 1-3 minutes depending on GPU...")
    print("=" * 80)
    
    # Test configuration
    TEST_IMAGE_URL = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=768&h=512&fit=crop"
    TEST_PROMPT = "The mountains come alive with flowing clouds and gentle wind, cinematic movement"
    TEST_JOB_ID = "test-hunyan15-i2v-direct"
    
    print(f"\n📥 Downloading test image from: {TEST_IMAGE_URL}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(TEST_IMAGE_URL, headers=headers, timeout=30)
        response.raise_for_status()
        image_bytes = response.content
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        print(f"✅ Test image downloaded ({len(image_b64)//1024}KB base64)")
    except Exception as e:
        print(f"❌ Failed to download test image: {e}")
        print("⚠️  Skipping test - cannot proceed without test image")
        exit(1)
    
    # Test parameters
    test_params = {
        "num_frames": 25,  # Short test video (~1 second at 24fps)
        "num_inference_steps": 12,  # Step distilled uses 12 steps
        "guidance_scale": 7.5,
        "fps": 24,
        "seed": 42,
        "output_dir": "./test_outputs",
    }
    
    print(f"\n🎬 Starting inference test...")
    print(f"   Job ID: {TEST_JOB_ID}")
    print(f"   Prompt: {TEST_PROMPT}")
    print(f"   Frames: {test_params['num_frames']}")
    print(f"   Steps: {test_params['num_inference_steps']}")
    print(f"   Guidance: {test_params['guidance_scale']}")
    print(f"   FPS: {test_params['fps']}")
    print(f"   Seed: {test_params['seed']}")
    
    try:
        start_time = time.time()
        
        # Run inference
        output_path = run_inference_hunyan15_i2v(
            job_id=TEST_JOB_ID,
            prompt=TEST_PROMPT,
            image_input=image_b64,
            params=test_params
        )
        
        elapsed = time.time() - start_time
        
        # Verify output
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024 / 1024  # MB
            print(f"\n✅ TEST PASSED!")
            print(f"   Output video: {output_path}")
            print(f"   File size: {file_size:.2f} MB")
            print(f"   Generation time: {elapsed:.1f}s")
            print(f"   FPS: {test_params['num_frames']/elapsed:.1f} frames/sec")
            print("\n" + "=" * 80)
            print("✅ Inference code is working correctly!")
            print("=" * 80)
            exit(0)
        else:
            print(f"\n❌ TEST FAILED!")
            print(f"   Expected output file not found: {output_path}")
            exit(1)
            
    except Exception as e:
        print(f"\n❌ TEST FAILED with exception:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)