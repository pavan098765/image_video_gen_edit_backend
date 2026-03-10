# hunyan15_video_inference.py
# ✅ Hunyan15 720p Image-to-Video Inference Worker (FP8)
# Model: hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled
# Transformer: Comfy-Org/HunyuanVideo_1.5_repackaged (FP8 quantized)
# Works safely on RTX 4090 (24GB) and RTX 5090 (32GB)
# Specialized for Image-to-Video generation with FP8 transformer quantization
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

# Set CUDA memory config to improve fragmentation and allocation performance
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# Disable PyTorch's automatic SDPA optimizations to prevent cutlassF errors
# These settings force PyTorch to use eager attention instead of cutlassF kernels
if hasattr(torch.backends.cuda, "enable_flash_sdp"):
    torch.backends.cuda.enable_flash_sdp(False)
if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
    torch.backends.cuda.enable_mem_efficient_sdp(False)
if hasattr(torch.backends.cuda, "enable_math_sdp"):
    # Keep math SDP enabled as fallback (it's the default eager implementation)
    torch.backends.cuda.enable_math_sdp(True)

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
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled",
)

# Local model paths (REQUIRED - models must be pre-downloaded, no HuggingFace downloads)
MODEL_DIR = os.environ.get(
    "HUNYAN15_720P_FP8_LOCAL_PATH", "/app/models/hunyan15_720p_fp8"
)

# Inference defaults
DEFAULT_FPS = 24
MAX_SEED = np.iinfo(np.int32).max
MAX_HEIGHT = 1024
MAX_WIDTH = 1024
MAX_FRAMES = 121  # approximately 5 seconds at 24fps

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
            print(
                f"[GPU] Before cleanup: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved"
            )

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()

        if verbose:
            print(
                f"[GPU] After cleanup: {torch.cuda.memory_allocated()/1024**2:.2f} MB allocated"
            )
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
    image: Image.Image, target_height: int, target_width: int
) -> Image.Image:
    """Resize image maintaining aspect ratio with center crop."""
    input_width, input_height = image.size
    aspect_ratio_target = target_width / target_height
    aspect_ratio_frame = input_width / input_height

    if aspect_ratio_frame > aspect_ratio_target:
        new_width = int(input_height * aspect_ratio_target)
        new_height = input_height
        x_start = (input_width - new_width) // 2
        y_start = 0
    else:
        new_width = input_width
        new_height = int(input_width / aspect_ratio_target)
        x_start = 0
        y_start = (input_height - new_height) // 2

    image = image.crop((x_start, y_start, x_start + new_width, y_start + new_height))
    image = image.resize((target_width, target_height), Image.LANCZOS)
    return image


def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def export_video(frames: np.ndarray, output_path: str, fps: int = DEFAULT_FPS) -> str:
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


def get_hunyan15_pipeline(cache_key: str = "hunyan15_i2v_distilled"):
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

    print(f"🚀 Loading Hunyan15 i2v pipeline (FP8) from local files only")
    print(f"   Model directory: {MODEL_DIR}")
    print(f"   (Reference repo: {HUNYAN15_REPO_ID})")
    print(f"   Transformer: FP8 quantized from Comfy-Org/HunyuanVideo_1.5_repackaged")
    sys.stdout.flush()

    device = get_device()
    gpu_info = get_gpu_info()
    # Use bfloat16 for VAE, text encoders, and other components
    # Transformer is pre-quantized to FP8 and will be loaded as FP8 automatically
    dtype = torch.bfloat16

    print(f"   Device: {device}")
    print(f"   Data type (VAE, text encoders): {dtype}")
    print(f"   Transformer dtype: FP8 (pre-quantized, loaded from Comfy-Org repo)")
    print(f"   GPU: {gpu_info}")
    sys.stdout.flush()

    start_time = time.time()
    print(f"⏳ Starting GPU cleanup...")
    sys.stdout.flush()
    cleanup_gpu_memory(verbose=True)

    try:
        print(f"📦 Importing diffusers modules...")
        sys.stdout.flush()
        from diffusers import HunyuanVideo15ImageToVideoPipeline, HunyuanVideoTransformer3DModel, attention_backend
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
            # Load pipeline with bfloat16 for most components
            # Transformer is already FP8 quantized in the model directory
            print(
                f"   Loading pipeline with bfloat16 (transformer is pre-quantized FP8)..."
            )
            sys.stdout.flush()

            transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                MODEL_DIR,
                subfolder="transformer",
                torch_dtype=torch.float8_e4m3fn,  # Standard FP8 format
                local_files_only=True,
            )

            pipeline = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
                MODEL_DIR,
                transformer=transformer,  
                torch_dtype=dtype,  # bfloat16 for VAE, text encoders, etc.
                local_files_only=True,  # CRITICAL: Never download
            )

            # Verify transformer is using FP8 (it should be loaded from the FP8 file)
            # The transformer weights are already FP8 quantized in the model directory
            transformer_path = os.path.join(
                MODEL_DIR, "transformer", "diffusion_pytorch_model.safetensors"
            )
            if os.path.exists(transformer_path):
                print(f"   ✅ Transformer file found: {transformer_path}")
                print(
                    f"   ✅ Transformer is pre-quantized FP8 (loaded from Comfy-Org repo)"
                )
                sys.stdout.flush()
            else:
                print(f"   ⚠️  Warning: Transformer FP8 file not found at expected path")
                sys.stdout.flush()
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
    job_id: str, prompt: str, image_input: str, params: Dict[str, Any]
) -> str:
    """
    Image-to-Video generation using Hunyan15 720p distilled model.

    Args:
        job_id: Unique job identifier
        prompt: Text prompt describing the animation
        image_input: Base64 encoded image or image URL
        params:
            - num_frames (int): Number of frames (default: 121)
            - num_inference_steps (int): Denoising steps (default: 50)
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
    num_frames = int(params.get("num_frames", 121))
    num_inference_steps = int(params.get("num_inference_steps", NUM_INFERENCE_STEPS))
    # Note: guidance_scale is not supported by HunyuanVideo15ImageToVideoPipeline
    seed = int(params.get("seed", random.randint(0, MAX_SEED)))
    fps = int(params.get("fps", DEFAULT_FPS))
    output_dir = params.get("output_dir", "/app/worker/outputs")

    # Validate parameters
    num_frames = min(num_frames, MAX_FRAMES)
    num_inference_steps = max(1, num_inference_steps)

    logger.info(
        f"[{job_id}] 📐 Configuration: frames={num_frames}, steps={num_inference_steps}, fps={fps}, seed={seed}"
    )
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
            if image_input.startswith("http"):
                image = load_image_from_url(image_input)
            elif image_input.startswith("data:") or len(image_input) > 500:
                image = load_image_from_base64(image_input)
            else:
                # Assume it's a file path
                image = Image.open(image_input).convert("RGB")
            logger.info(f"[{job_id}] Image loaded successfully: {image.size}")
        except Exception as e:
            error_msg = f"Failed to load input image: {e}"
            logger.error(f"[{job_id}] ❌ {error_msg}")
            print(f"❌ {error_msg}")
            sys.stdout.flush()
            raise

        print(f"   Image size: {image.size}")
        sys.stdout.flush()

        logger.info(f"[{job_id}] 🚀 Starting video generation (seed={seed})...")
        print(f"🚀 Generating video with seed={seed}...")
        print(f"   Prompt: {prompt[:100]}...")
        sys.stdout.flush()

        from diffusers import attention_backend

        # Generate video using attention backend
        # Note: HunyuanVideo15ImageToVideoPipeline doesn't support guidance_scale parameter
        logger.info(
            f"[{job_id}] 🎥 Starting video generation (this may take 1-3 minutes)..."
        )
        print(f"🎥 Starting video generation (this may take 1-3 minutes)...")
        sys.stdout.flush()

        try:
            with attention_backend("flash_hub"):  # Use flash_hub for RTX 4090/5090
                generator = torch.Generator(device=device).manual_seed(seed)

                logger.info(f"[{job_id}] Calling pipeline.generate()...")
                print(f"   Calling pipeline.generate()...")
                print(
                    f"   Inference will run for {num_inference_steps} steps (this may take 2-5 minutes)..."
                )
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
                    if (current_time - last_log_time >= log_interval) or (
                        step % max(1, num_inference_steps // 10) == 0
                    ):
                        elapsed_str = f"{elapsed:.1f}s"
                        eta = (
                            (elapsed / (step + 1)) * (num_inference_steps - step - 1)
                            if step > 0
                            else 0
                        )
                        eta_str = f"{eta:.1f}s" if eta > 0 else "calculating..."
                        logger.info(
                            f"[{job_id}] 🔄 Inference progress: {step + 1}/{num_inference_steps} steps ({progress_pct:.1f}%) | Elapsed: {elapsed_str} | ETA: {eta_str}"
                        )
                        print(
                            f"   🔄 Step {step + 1}/{num_inference_steps} ({progress_pct:.1f}%) | Elapsed: {elapsed_str} | ETA: {eta_str}"
                        )
                        sys.stdout.flush()
                        last_log_time = current_time

                    return callback_kwargs

                # Try to use callback if supported, otherwise just log before/after
                try:
                    result = pipeline(
                        prompt=prompt,
                        image=image,
                        generator=generator,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        callback_on_step_end=progress_callback,
                    )
                except (TypeError, AttributeError) as e:
                    # Callback not supported, use periodic logging instead
                    logger.info(
                        f"[{job_id}] Pipeline doesn't support callback (reason: {type(e).__name__}), using time-based estimation..."
                    )
                    print(
                        f"   ⚠️ Note: Progress shown is ESTIMATED based on elapsed time, not actual step count"
                    )
                    print(
                        f"   Starting inference (will log estimated progress every 10 seconds)..."
                    )
                    sys.stdout.flush()

                    # Start a background thread to log progress periodically
                    import threading

                    progress_logging_active = threading.Event()
                    progress_logging_active.set()

                    # Adaptive estimation: start with conservative estimate, adjust based on typical times
                    # Typical: 50 steps takes ~100-150s on RTX 4090/5090
                    estimated_total_time = (
                        120.0  # Initial estimate: 2 minutes for 50 steps
                    )

                    def periodic_progress_logger():
                        step_count = 0
                        while (
                            progress_logging_active.is_set()
                            and step_count < num_inference_steps
                        ):
                            time.sleep(log_interval)
                            if progress_logging_active.is_set():
                                elapsed = time.time() - inference_start_time
                                # Estimate current step based on elapsed time
                                # Use linear interpolation: if we're X% through time, assume X% through steps
                                estimated_step = min(
                                    int(
                                        (elapsed / estimated_total_time)
                                        * num_inference_steps
                                    ),
                                    num_inference_steps - 1,
                                )
                                progress_pct = (
                                    (estimated_step + 1) / num_inference_steps * 100
                                )

                                # Calculate ETA based on current rate
                                if elapsed > 5:  # After 5 seconds, we can estimate
                                    steps_per_second = (
                                        estimated_step / elapsed if elapsed > 0 else 0
                                    )
                                    if steps_per_second > 0:
                                        remaining_steps = (
                                            num_inference_steps - estimated_step
                                        )
                                        eta_seconds = remaining_steps / steps_per_second
                                        eta_str = f"{eta_seconds:.1f}s"
                                    else:
                                        eta_str = "calculating..."
                                else:
                                    eta_str = "calculating..."

                                logger.info(
                                    f"[{job_id}] 🔄 Inference in progress (ESTIMATED): ~{estimated_step + 1}/{num_inference_steps} steps ({progress_pct:.1f}%) | Elapsed: {elapsed:.1f}s | ETA: {eta_str}"
                                )
                                print(
                                    f"   🔄 Inference in progress (ESTIMATED): ~{estimated_step + 1}/{num_inference_steps} steps ({progress_pct:.1f}%) | Elapsed: {elapsed:.1f}s | ETA: {eta_str}"
                                )
                                sys.stdout.flush()

                    progress_thread = threading.Thread(
                        target=periodic_progress_logger, daemon=True
                    )
                    progress_thread.start()

                    result = pipeline(
                        prompt=prompt,
                        image=image,
                        generator=generator,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                    )

                    progress_logging_active.clear()

                    # Update estimated total time for next run (adaptive learning)
                    actual_time = time.time() - inference_start_time
                    if actual_time > 0:
                        estimated_total_time = (
                            actual_time  # Use actual time for future estimates
                        )

                inference_elapsed = time.time() - inference_start_time
                logger.info(
                    f"[{job_id}] ✅ Pipeline generation completed in {inference_elapsed:.1f}s, extracting frames..."
                )
                print(
                    f"   ✅ Pipeline generation completed in {inference_elapsed:.1f}s, extracting frames..."
                )
                sys.stdout.flush()

                video_frames = result.frames[0]
                num_frames_generated = (
                    len(video_frames) if hasattr(video_frames, "__len__") else "N/A"
                )
                logger.info(
                    f"[{job_id}] Frames extracted: {num_frames_generated} frames"
                )
                print(f"   Frames extracted: {num_frames_generated} frames")
                sys.stdout.flush()

            # Export video
            logger.info(f"[{job_id}] 💾 Exporting video to {output_path}...")
            print(f"💾 Exporting video to {output_path}...")
            sys.stdout.flush()

            from diffusers.utils import export_to_video

            export_to_video(video_frames, output_path, fps=fps)

            # Verify file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / 1024 / 1024  # MB
                logger.info(
                    f"[{job_id}] ✅ Video generated successfully: {output_path} ({file_size:.2f} MB)"
                )
                print(
                    f"✅ Video generated successfully: {output_path} ({file_size:.2f} MB)"
                )
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
    job_id: str, prompt: str, params: Dict[str, Any]
) -> str:
    """
    Wrapper for backward compatibility.
    Delegates to run_inference_hunyan15_i2v.
    """
    image_input = params.pop("image_input", None)
    if not image_input:
        raise ValueError("image_input is required in params for Hunyan15 model")

    return run_inference_hunyan15_i2v(job_id, prompt, image_input, params)
