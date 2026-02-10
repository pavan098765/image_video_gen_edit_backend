# wan_21_fast_inference.py
# ✅ Wan 2.1 Fast I2V Inference Worker
# Model: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
# LoRA: Kijai/WanVideo_comfy - Wan21_CausVid_14B_T2V_lora_rank32.safetensors
# Fast 4-8 step generation with CausVid LoRA distillation
# Works safely on RTX 4090 (24GB) and RTX 5090 (32GB)

import os
import gc
import sys
import time
import base64
import random
import tempfile
import numpy as np
from io import BytesIO
from typing import Dict, Any, Optional
from pathlib import Path

import torch
from PIL import Image

# Set CUDA memory config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# Aggressive GPU cleanup at import
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
MODEL_ID = os.environ.get(
    "WAN_21_MODEL_ID",
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
)
LORA_REPO_ID = os.environ.get(
    "WAN_21_LORA_REPO_ID",
    "Kijai/WanVideo_comfy"
)
LORA_FILENAME = os.environ.get(
    "WAN_21_LORA_FILENAME",
    "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"
)

# Local model paths (REQUIRED - models must be pre-downloaded)
MODEL_DIR = os.environ.get("WAN_21_LOCAL_PATH", "/app/models/wan_21_fast")
LORA_DIR = os.environ.get("WAN_21_LORA_LOCAL_PATH", "/app/models/wan_21_fast/lora")

# Inference defaults
MOD_VALUE = 32
DEFAULT_H = 512
DEFAULT_W = 896
NEW_FORMULA_MAX_AREA = 480.0 * 832.0
SLIDER_MIN_H, SLIDER_MAX_H = 128, 896
SLIDER_MIN_W, SLIDER_MAX_W = 128, 896
MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 24
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 81

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards, watermark, text, signature"
)

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


def load_image_from_base64(img_b64: str) -> Image.Image:
    """Load PIL Image from base64 string."""
    return Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")


def load_image_from_url(image_url: str) -> Image.Image:
    """Load PIL Image from URL."""
    from diffusers.utils import load_image
    return load_image(image_url).convert("RGB")


def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _calculate_new_dimensions_wan(pil_image, mod_val, calculation_max_area,
                                 min_slider_h, max_slider_h,
                                 min_slider_w, max_slider_w,
                                 default_h, default_w):
    """Calculate optimal dimensions for Wan model."""
    orig_w, orig_h = pil_image.size
    if orig_w <= 0 or orig_h <= 0:
        return default_h, default_w

    aspect_ratio = orig_h / orig_w
    
    calc_h = round(np.sqrt(calculation_max_area * aspect_ratio))
    calc_w = round(np.sqrt(calculation_max_area / aspect_ratio))

    calc_h = max(mod_val, (calc_h // mod_val) * mod_val)
    calc_w = max(mod_val, (calc_w // mod_val) * mod_val)
    
    new_h = int(np.clip(calc_h, min_slider_h, (max_slider_h // mod_val) * mod_val))
    new_w = int(np.clip(calc_w, min_slider_w, (max_slider_w // mod_val) * mod_val))
    
    return new_h, new_w


# ==============================
# Pipeline Loading
# ==============================
def get_wan_21_pipeline(cache_key: str = "wan_21_fast"):
    """
    Load or retrieve cached Wan 2.1 pipeline with CausVid LoRA.
    
    Returns:
        Dict with keys: pipeline, device, gpu_info
    """
    global PIPELINE_CACHE
    
    if cache_key in PIPELINE_CACHE:
        print(f"📦 Using cached pipeline: {cache_key}")
        sys.stdout.flush()
        return PIPELINE_CACHE[cache_key]
    
    print(f"🚀 Loading Wan 2.1 Fast I2V pipeline from local files only")
    print(f"   Model directory: {MODEL_DIR}")
    print(f"   LoRA directory: {LORA_DIR}")
    sys.stdout.flush()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_info = get_gpu_info()
    
    print(f"   Device: {device}")
    print(f"   GPU: {gpu_info}")
    sys.stdout.flush()
    
    start_time = time.time()
    print(f"⏳ Starting GPU cleanup...")
    sys.stdout.flush()
    cleanup_gpu_memory(verbose=True)
    
    try:
        print(f"📦 Importing diffusers modules...")
        sys.stdout.flush()
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, UniPCMultistepScheduler
        from transformers import CLIPVisionModel
        from huggingface_hub import hf_hub_download
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
        
        # Load components from LOCAL files only
        print(f"⏳ Loading image encoder from local files...")
        sys.stdout.flush()
        load_start = time.time()
        
        try:
            image_encoder = CLIPVisionModel.from_pretrained(
                MODEL_DIR,
                subfolder="image_encoder",
                torch_dtype=torch.float32,
                local_files_only=True
            )
            print(f"✅ Image encoder loaded in {time.time() - load_start:.1f}s")
            sys.stdout.flush()
        except Exception as load_error:
            error_msg = f"❌ Failed to load image encoder: {load_error}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise FileNotFoundError(error_msg) from load_error
        
        print(f"⏳ Loading VAE from local files...")
        sys.stdout.flush()
        load_start = time.time()
        
        try:
            vae = AutoencoderKLWan.from_pretrained(
                MODEL_DIR,
                subfolder="vae",
                torch_dtype=torch.float32,
                local_files_only=True
            )
            print(f"✅ VAE loaded in {time.time() - load_start:.1f}s")
            sys.stdout.flush()
        except Exception as load_error:
            error_msg = f"❌ Failed to load VAE: {load_error}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise FileNotFoundError(error_msg) from load_error
        
        print(f"⏳ Loading pipeline from local files...")
        sys.stdout.flush()
        load_start = time.time()
        
        try:
            pipeline = WanImageToVideoPipeline.from_pretrained(
                MODEL_DIR,
                vae=vae,
                image_encoder=image_encoder,
                torch_dtype=torch.bfloat16,
                local_files_only=True
            )
            print(f"✅ Pipeline loaded in {time.time() - load_start:.1f}s")
            sys.stdout.flush()
        except Exception as load_error:
            error_msg = f"❌ Failed to load pipeline: {load_error}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise FileNotFoundError(error_msg) from load_error
        
        # Configure scheduler
        print(f"⚙️  Configuring UniPCMultistepScheduler...")
        sys.stdout.flush()
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config,
            flow_shift=8.0
        )
        print(f"   ✅ Scheduler configured")
        sys.stdout.flush()
        
        # Move to device
        print(f"⚙️  Moving pipeline to {device}...")
        sys.stdout.flush()
        pipeline.to(device)
        print(f"   ✅ Pipeline moved to {device}")
        sys.stdout.flush()
        
        # Load LoRA (matching Gradio app logic)
        lora_path = os.path.join(LORA_DIR, LORA_FILENAME)
        if not os.path.exists(lora_path):
            # Try downloading from HuggingFace if local file doesn't exist (matching Gradio app)
            print(f"⚠️  LoRA file not found at {lora_path}, attempting to download from HuggingFace...")
            sys.stdout.flush()
            try:
                os.makedirs(LORA_DIR, exist_ok=True)
                lora_path = hf_hub_download(
                    repo_id=LORA_REPO_ID,
                    filename=LORA_FILENAME
                )
                # Copy to our local directory for future use
                import shutil
                local_lora_path = os.path.join(LORA_DIR, LORA_FILENAME)
                shutil.copy2(lora_path, local_lora_path)
                lora_path = local_lora_path
                print(f"✅ LoRA downloaded to {lora_path}")
                sys.stdout.flush()
            except Exception as e:
                error_msg = f"❌ Failed to download LoRA: {e}"
                print(error_msg)
                sys.stdout.flush()
                raise FileNotFoundError(error_msg) from e
        else:
            print(f"📁 Using local LoRA file: {lora_path}")
            sys.stdout.flush()
        
        print(f"⏳ Loading CausVid LoRA (matching Gradio app)...")
        sys.stdout.flush()
        load_start = time.time()
        
        try:
            # Match Gradio app: load_lora_weights, set_adapters, fuse_lora
            pipeline.load_lora_weights(lora_path, adapter_name="causvid_lora")
            pipeline.set_adapters(["causvid_lora"], adapter_weights=[0.95])
            pipeline.fuse_lora()
            print(f"✅ LoRA loaded and fused in {time.time() - load_start:.1f}s")
            sys.stdout.flush()
        except Exception as load_error:
            error_msg = f"❌ Failed to load LoRA: {load_error}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise FileNotFoundError(error_msg) from load_error
        
        elapsed = time.time() - start_time
        print(f"✅ Wan 2.1 Fast I2V pipeline fully loaded in {elapsed:.1f}s")
        sys.stdout.flush()
        
        PIPELINE_CACHE[cache_key] = {
            "pipeline": pipeline,
            "device": device,
            "gpu_info": gpu_info,
        }
        
        return PIPELINE_CACHE[cache_key]
        
    except ImportError as e:
        print(f"❌ diffusers not installed! Error: {e}")
        sys.stdout.flush()
        raise ImportError(
            f"Wan 2.1 requires diffusers>=0.31.0. "
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
        pipeline_type: Optional pipeline type (for compatibility, not used for Wan 2.1)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print("=" * 80)
        print("PIPELINE PRELOAD: Starting pipeline preload...")
        print("=" * 80)
        sys.stdout.flush()
        
        start_time = time.time()
        result = get_wan_21_pipeline()
        
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
def run_inference_wan_21_fast_i2v(
    job_id: str,
    prompt: str,
    image_input: str,
    params: Dict[str, Any]
) -> str:
    """
    Image-to-Video generation using Wan 2.1 14B with CausVid LoRA.
    
    Args:
        job_id: Unique job identifier
        prompt: Text prompt describing the animation
        image_input: Base64 encoded image or image URL
        params:
            - num_frames (int): Number of frames (default: calculated from duration)
            - num_inference_steps (int): Denoising steps (default: 4)
            - guidance_scale (float): Guidance scale (default: 1.0)
            - seed (int): Random seed
            - randomize_seed (bool): Whether to randomize seed
            - fps (int): Frames per second (default: 24)
            - height (int): Output height (default: 512)
            - width (int): Output width (default: 896)
            - duration_seconds (float): Video duration in seconds (default: 2.0)
            - negative_prompt (str): Negative prompt
            - output_dir (str): Output directory
    
    Returns:
        Absolute path to generated video
    """
    import logging
    logger = logging.getLogger("worker")
    
    logger.info(f"[{job_id}] 🎬 Starting Wan 2.1 Fast I2V inference")
    print(f"🎬 Starting Wan 2.1 Fast I2V job: {job_id}")
    sys.stdout.flush()
    
    cleanup_gpu_memory()
    
    # Parameters
    steps = int(params.get("num_inference_steps", 4))
    guidance_scale = float(params.get("guidance_scale", 1.0))
    seed = int(params.get("seed", random.randint(0, MAX_SEED)))
    randomize_seed = params.get("randomize_seed", False)
    height = int(params.get("height", DEFAULT_H))
    width = int(params.get("width", DEFAULT_W))
    duration_seconds = float(params.get("duration_seconds", 2.0))
    negative_prompt = params.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    output_dir = params.get("output_dir", "/app/worker/outputs")
    fps = int(params.get("fps", FIXED_FPS))
    
    # Calculate frames from duration
    num_frames = int(np.clip(
        round(duration_seconds * fps),
        MIN_FRAMES_MODEL,
        MAX_FRAMES_MODEL
    ))
    
    # Adjust dimensions to be multiples of MOD_VALUE
    target_h = max(MOD_VALUE, (height // MOD_VALUE) * MOD_VALUE)
    target_w = max(MOD_VALUE, (width // MOD_VALUE) * MOD_VALUE)
    
    # Randomize seed if requested
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else seed
    
    logger.info(f"[{job_id}] 📐 Configuration: frames={num_frames}, steps={steps}, fps={fps}, seed={current_seed}, size={target_h}x{target_w}")
    print(f"📐 Configuration:")
    print(f"   Frames: {num_frames}")
    print(f"   Steps: {steps}")
    print(f"   Guidance Scale: {guidance_scale}")
    print(f"   FPS: {fps}")
    print(f"   Seed: {current_seed}")
    print(f"   Size: {target_h}x{target_w}")
    print(f"   Output directory: {output_dir}")
    sys.stdout.flush()
    
    try:
        seed_everything(current_seed)
        
        # Load pipeline
        logger.info(f"[{job_id}] Loading pipeline...")
        data = get_wan_21_pipeline()
        pipeline = data["pipeline"]
        device = data["device"]
        logger.info(f"[{job_id}] Pipeline loaded successfully")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"wan21_i2v_{job_id}.mp4")
        logger.info(f"[{job_id}] Output path: {output_path}")
        print(f"📁 Output path: {output_path}")
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
        
        # Resize image to target dimensions
        resized_image = image.resize((target_w, target_h), Image.LANCZOS)
        print(f"   Image size: {image.size} -> {resized_image.size}")
        sys.stdout.flush()
        
        logger.info(f"[{job_id}] 🚀 Starting video generation (seed={current_seed})...")
        print(f"🚀 Generating video with seed={current_seed}...")
        print(f"   Prompt: {prompt[:100]}...")
        sys.stdout.flush()
        
        # Generate video
        logger.info(f"[{job_id}] 🎥 Starting video generation (this may take 30-120 seconds)...")
        print(f"🎥 Starting video generation (this may take 30-120 seconds)...")
        sys.stdout.flush()
        
        inference_start_time = time.time()
        
        try:
            generator = torch.Generator(device=device).manual_seed(current_seed)
            
            with torch.inference_mode():
                result = pipeline(
                    image=resized_image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=target_h,
                    width=target_w,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps,
                    generator=generator
                )
            
            inference_elapsed = time.time() - inference_start_time
            logger.info(f"[{job_id}] ✅ Pipeline generation completed in {inference_elapsed:.1f}s, extracting frames...")
            print(f"   ✅ Pipeline generation completed in {inference_elapsed:.1f}s, extracting frames...")
            sys.stdout.flush()
            
            video_frames = result.frames[0]
            num_frames_generated = len(video_frames) if hasattr(video_frames, '__len__') else 'N/A'
            logger.info(f"[{job_id}] Frames extracted: {num_frames_generated} frames")
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


# ==============================
# Main usage (for testing)
# ==============================
if __name__ == "__main__":
    import uuid
    
    print("=" * 80)
    print("Wan 2.1 Fast I2V - Direct Inference Test")
    print("=" * 80)
    print("This script tests the inference function directly.")
    print("For production use, use the FastAPI app (app.py)")
    print("=" * 80)
    sys.stdout.flush()
    
    job_id = str(uuid.uuid4())
    
    # Example: You would load an image here
    # input_image_path = "input_example.jpg"
    # input_img = Image.open(input_image_path).convert("RGB")
    
    # For testing, create a dummy image
    input_img = Image.new("RGB", (896, 512), color="blue")
    
    params = {
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "seed": 42,
        "randomize_seed": False,
        "height": 512,
        "width": 896,
        "duration_seconds": 2.0,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "output_dir": "./outputs"
    }
    
    # Convert image to base64 for testing
    import io
    buffer = io.BytesIO()
    input_img.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    try:
        output_video = run_inference_wan_21_fast_i2v(
            job_id,
            "make this image come alive, cinematic motion, smooth animation",
            image_b64,
            params
        )
        print(f"\n✅ [Wan 2.1 Fast I2V] Saved video at: {output_video}")
    except Exception as e:
        print(f"\n❌ [Wan 2.1 Fast I2V] Test failed: {e}")
        import traceback
        traceback.print_exc()
