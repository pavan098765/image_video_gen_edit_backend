#!/usr/bin/env python3
"""
Standalone script for 4-bit (UINT4 with SVD rank 32) quantization of 
Wan-AI/Wan2.2-T2V-A14B-Diffusers using SDNQ.

This script:
1. Downloads the model locally using huggingface snapshot_download
2. Runs text-to-video inference using the local model path
"""

import os
import time
import torch
import numpy as np
import diffusers
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download
from sdnq import SDNQConfig  # import sdnq to register it into diffusers and transformers
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:512,expandable_segments:True"

# Video post-processing imports
try:
    from video_postprocessing import (
        download_rife_model,
        download_realesrgan_model,
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
    # Try 'cusolver' first (preferred), fallback to 'magma' or 'cublaslt' if errors occur
    try:
        if hasattr(torch.backends.cuda, 'preferred_linalg_library'):
            # Try to set to cusolver first (preferred)
            try:
                torch.backends.cuda.preferred_linalg_library('cusolver')
                print("✓ Set linear algebra backend to 'cusolver' (preferred)")
            except (ValueError, RuntimeError):
                # Fallback to magma if cusolver not available
                try:
                    torch.backends.cuda.preferred_linalg_library('magma')
                    print("✓ Set linear algebra backend to 'magma' (fallback)")
                except (ValueError, RuntimeError):
                    # Last resort: try cublaslt
                    try:
                        torch.backends.cuda.preferred_linalg_library('cublaslt')
                        print("✓ Set linear algebra backend to 'cublaslt' (fallback)")
                    except (ValueError, RuntimeError):
                        print("⚠️  Could not set preferred linear algebra backend, using default")
    except AttributeError:
        # Method doesn't exist in this PyTorch version
        pass

# Configuration
MODEL_REPO = "Disty0/Wan2.2-T2V-A14B-SDNQ-uint4-svd-r32"
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
MODEL_LOCAL_DIR = os.path.join(MODELS_DIR, "wan2.2_t2v_sdnq_uint4_svd_r32")

# LoRA Configuration
LORA_REPO = "Kijai/WanVideo_comfy"
LORA_WEIGHT_NAME = "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors"
LORA_LOCAL_DIR = os.path.join(MODELS_DIR, "lora", "Kijai_WanVideo_comfy")
USE_LORA = True  # Set to False to disable LoRA

# Output settings
OUTPUT_VIDEO_PATH = os.path.join(WORKSPACE_DIR, "t2v_output.mp4")

# Inference parameters
PROMPT ="A wide shot of a dusty, neon-lit cyberpunk street at night, rain-slicked pavement reflecting blue and pink lights. A mysterious figure in a trench coat walks slowly towards the camera while smoke rises from a manhole. Cinematic, 35mm film style, slow motion."

NEGATIVE_PROMPT = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景, 三条腿, 背景人很多, 倒着走, 阴道变形, 乳房变形, 乳头畸形, 变形的皮肤, 大腿畸形，臀部畸, 面部畸形，眼睛畸形, 膝盖畸形，髋关节畸形"

NUM_FRAMES = 81
GUIDANCE_SCALE = 1.0
GUIDANCE_SCALE_2 = 1.0
NUM_INFERENCE_STEPS = 8  # With Lightning LoRA, can reduce to 4-8 steps for 10x speedup
FPS = 16
SEED = 0

# Resolution
HEIGHT = 480
WIDTH = 832

# Post-processing options
ENABLE_RIFE = True  # Frame interpolation with RIFE
RIFE_TARGET_FPS = 30  # Target FPS for RIFE (None = double original FPS, e.g., 16 -> 32)
                        # Examples: 24, 30, 32, 48, 60 (will use closest achievable)

ENABLE_UPSCALING = True  # Super-resolution with Real-ESRGAN
UPSCALE_FACTOR = 2  # Upscale factor: 2, 4, or 8 (e.g., 480p -> 960p, 1920p, or 3840p)

# Post-processing order
# True = Interpolate first, then upscale (may cause errors, speed benefit not noticeable)
# False = Upscale first, then interpolate (DEFAULT: more stable, better compatibility)
POSTPROCESS_INTERPOLATE_FIRST = False


def format_time(seconds):
    """Format time in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def download_model():
    """Download the model locally if not already present."""
    # Check if model directory exists and has content
    if os.path.exists(MODEL_LOCAL_DIR):
        # Check for key model files (config.json is a good indicator)
        config_file = os.path.join(MODEL_LOCAL_DIR, "config.json")
        if os.path.exists(config_file):
            print(f"✓ Model already exists at {MODEL_LOCAL_DIR}")
            return MODEL_LOCAL_DIR, 0.0
        else:
            print(f"⚠️ Model directory exists but appears incomplete, re-downloading...")
    
    print(f"⬇️ Downloading {MODEL_REPO} to {MODEL_LOCAL_DIR}...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    start_time = time.time()
    try:
        snapshot_download(
            MODEL_REPO,
            local_dir=MODEL_LOCAL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        elapsed_time = time.time() - start_time
        print(f"✓ Model downloaded successfully to {MODEL_LOCAL_DIR}")
        print(f"   ⏱️  Download time: {format_time(elapsed_time)}")
        return MODEL_LOCAL_DIR, elapsed_time
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        raise


def download_lora():
    """Download LoRA weights locally if not already present."""
    if not USE_LORA:
        return None, 0.0
    
    # Check if LoRA file already exists locally
    lora_file_path = os.path.join(LORA_LOCAL_DIR, LORA_WEIGHT_NAME)
    if os.path.exists(lora_file_path):
        print(f"✓ LoRA already exists at {lora_file_path}")
        return lora_file_path, 0.0
    
    print(f"⬇️ Downloading LoRA from {LORA_REPO}...")
    print(f"   Weight: {LORA_WEIGHT_NAME}")
    os.makedirs(LORA_LOCAL_DIR, exist_ok=True)
    
    start_time = time.time()
    try:
        # Download the specific LoRA file
        snapshot_download(
            LORA_REPO,
            local_dir=LORA_LOCAL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=[LORA_WEIGHT_NAME],  # Only download the specific LoRA file
        )
        elapsed_time = time.time() - start_time
        
        # Verify the file exists
        if os.path.exists(lora_file_path):
            file_size_mb = os.path.getsize(lora_file_path) / (1024 * 1024)
            print(f"✓ LoRA downloaded successfully to {lora_file_path}")
            print(f"   ⏱️  Download time: {format_time(elapsed_time)}")
            print(f"   📦 File size: {file_size_mb:.2f} MB")
            return lora_file_path, elapsed_time
        else:
            # File might be in a subdirectory, check for it
            for root, dirs, files in os.walk(LORA_LOCAL_DIR):
                if LORA_WEIGHT_NAME in files:
                    found_path = os.path.join(root, LORA_WEIGHT_NAME)
                    file_size_mb = os.path.getsize(found_path) / (1024 * 1024)
                    print(f"✓ LoRA downloaded successfully to {found_path}")
                    print(f"   ⏱️  Download time: {format_time(elapsed_time)}")
                    print(f"   📦 File size: {file_size_mb:.2f} MB")
                    return found_path, elapsed_time
            
            raise FileNotFoundError(f"LoRA file not found after download: {lora_file_path}")
    except Exception as e:
        print(f"❌ Failed to download LoRA: {e}")
        print(f"   ℹ️  Will try to load from HuggingFace Hub directly during inference")
        return None, 0.0


def download_postprocessing_models():
    """Download RIFE and Real-ESRGAN models if post-processing is enabled."""
    if not POSTPROCESSING_AVAILABLE:
        return None, None, 0.0
    
    total_time = 0.0
    rife_path = None
    realesrgan_path = None
    
    if ENABLE_RIFE:
        start_time = time.time()
        rife_path = download_rife_model(MODELS_DIR)
        total_time += time.time() - start_time
    
    if ENABLE_UPSCALING:
        start_time = time.time()
        realesrgan_path = download_realesrgan_model(MODELS_DIR, UPSCALE_FACTOR)
        total_time += time.time() - start_time
    
    return rife_path, realesrgan_path, total_time


def run_inference():
    """Run text-to-video inference using the local model."""
    total_start_time = time.time()
    
    print("\n" + "="*60)
    print("Starting Text-to-Video Inference")
    print("="*60)
    
    # Download model, LoRA, and post-processing models if needed
    model_path, model_download_time = download_model()
    lora_path, lora_download_time = download_lora()
    rife_path, realesrgan_path, postproc_download_time = download_postprocessing_models()
    download_time = model_download_time + lora_download_time + postproc_download_time
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA: {torch.version.cuda}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load VAE separately (required for T2V pipeline)
    print(f"\n📦 Loading VAE from {model_path}...")
    vae_start_time = time.time()
    vae = diffusers.AutoencoderKLWan.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch.float32
    )
    vae_time = time.time() - vae_start_time
    print(f"✓ VAE loaded successfully")
    print(f"   ⏱️  VAE loading time: {format_time(vae_time)}")
    
    # Load pipeline from local path
    print(f"\n📦 Loading pipeline from {model_path}...")
    load_start_time = time.time()
    pipe = diffusers.WanPipeline.from_pretrained(
        model_path,
        vae=vae,
        torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    
    # Load Lightning LoRA (before other optimizations)
    # IMPORTANT: We use adapters WITHOUT fusion since the model is already quantized with SDNQ
    # Fusion would modify the quantized weights and break SDNQ quantization
    # However, LoRA adapters work perfectly fine with quantized models - they add weight deltas
    # This gives us the speed benefits of Lightning LoRA without breaking quantization
    lora_loaded = False
    if USE_LORA:
        print("   🚀 Loading Lightning LoRA (speeds up inference significantly)...")
        try:
            # Use local LoRA path if available, otherwise use repo
            if lora_path and os.path.exists(lora_path):
                # Local path - extract directory and filename
                lora_dir = os.path.dirname(lora_path)
                lora_filename = os.path.basename(lora_path)
                print(f"   📁 Using local LoRA: {lora_path}")
                
                # Check if file is in a subdirectory (e.g., Lightx2v/lightx2v_*.safetensors)
                # If so, we need to use the parent directory and the subdirectory path
                if "Lightx2v" in lora_path or "lightx2v" in os.path.dirname(lora_path):
                    # File is in a subdirectory, use parent directory and relative path
                    parent_dir = os.path.dirname(lora_dir)  # Get parent of subdirectory
                    subdir_name = os.path.basename(lora_dir)  # e.g., "Lightx2v"
                    weight_name_with_subdir = os.path.join(subdir_name, lora_filename)
                    
                    pipe.load_lora_weights(
                        parent_dir,
                        weight_name=weight_name_with_subdir,
                        adapter_name="lightx2v"
                    )
                else:
                    # File is directly in the directory
                    pipe.load_lora_weights(
                        lora_dir,
                        weight_name=lora_filename,
                        adapter_name="lightx2v"
                    )
                print("   ✓ LoRA loaded for transformer (local)")
                
                # Load LoRA for transformer_2 from local path
                kwargs_lora = {"load_into_transformer_2": True}
                if "Lightx2v" in lora_path or "lightx2v" in os.path.dirname(lora_path):
                    parent_dir = os.path.dirname(lora_dir)
                    subdir_name = os.path.basename(lora_dir)
                    weight_name_with_subdir = os.path.join(subdir_name, lora_filename)
                    
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
                        adapter_name="lightx2v_2",
                        **kwargs_lora
                    )
                print("   ✓ LoRA loaded for transformer_2 (local)")
            else:
                # Fallback to HuggingFace Hub
                print(f"   📡 Loading LoRA from HuggingFace Hub: {LORA_REPO}")
                pipe.load_lora_weights(
                    LORA_REPO,
                    weight_name=LORA_WEIGHT_NAME,
                    adapter_name="lightx2v"
                )
                print("   ✓ LoRA loaded for transformer (from Hub)")
                
                # Load LoRA for transformer_2
                kwargs_lora = {"load_into_transformer_2": True}
                pipe.load_lora_weights(
                    LORA_REPO,
                    weight_name=LORA_WEIGHT_NAME,
                    adapter_name="lightx2v_2",
                    **kwargs_lora
                )
                print("   ✓ LoRA loaded for transformer_2 (from Hub)")
            
            # Activate both adapters
            pipe.set_adapters(["lightx2v", "lightx2v_2"], adapter_weights=[1., 1.])
            print("   ✓ Lightning LoRA adapters activated")
            print("   💡 With Lightning LoRA, you can reduce steps to 4-8 for faster inference!")
            lora_loaded = True
        except Exception as e:
            print(f"   ⚠️  Failed to load LoRA: {str(e)[:200]}")
            print("   ℹ️  Continuing without LoRA (will be slower)")
            # Try to unload any partially loaded adapters
            try:
                pipe.unload_lora_weights()
            except:
                pass
    
    # Apply speed optimizations
    print("   ⚡ Applying speed optimizations...")
    
    # Memory offloading (use model_cpu_offload for speed, sequential is slower)
    pipe.enable_model_cpu_offload()
    print("   ✓ Model CPU offload enabled")
    
    # Flash Attention (you have it installed - this will use it automatically)
    flash_attn_available = False
    try:
        import flash_attn
        flash_attn_available = True
        print(f"   ✓ Flash Attention detected (version: {getattr(flash_attn, '__version__', 'unknown')})")
        print("   ✓ Flash Attention will be used automatically by diffusers")
    except ImportError:
        print("   ⚠️  Flash Attention not found (but you said it's installed)")
    
    # XFormers as fallback (if flash-attn doesn't work)
    if not flash_attn_available:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("   ✓ XFormers memory-efficient attention enabled")
        except Exception as e:
            print(f"   ℹ️  XFormers not available: {str(e)[:100]}")
    
    load_time = time.time() - load_start_time
    print("✓ Pipeline loaded and optimized successfully")
    print(f"   ⏱️  Loading time: {format_time(load_time)}")
    
    # Load post-processing models if enabled
    rife_model = None
    upscale_model = None
    if POSTPROCESSING_AVAILABLE:
        if ENABLE_RIFE:
            # First check if RIFE code is available
            if not check_rife_code_available():
                print(f"   ⚠️  RIFE code not available, frame interpolation disabled")
                print(f"   ℹ️  See instructions above to install RIFE code")
            elif rife_path:
                print(f"\n🎬 Loading RIFE model for frame interpolation...")
                rife_load_start = time.time()
                rife_model = load_rife_model(rife_path)
                if rife_model:
                    # RIFE model is already on device (moved in __init__ via self.device())
                    # No need to call .to(device) as RIFE Model doesn't have that method
                    target_fps_str = f"{RIFE_TARGET_FPS}fps" if RIFE_TARGET_FPS else f"{FPS*2}fps (2x)"
                    print(f"   ✓ RIFE model loaded (target: {target_fps_str})")
                    print(f"   ⏱️  RIFE loading time: {format_time(time.time() - rife_load_start)}")
                else:
                    print(f"   ⚠️  RIFE model failed to load, frame interpolation disabled")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"   ⚠️  RIFE model path not available, frame interpolation disabled")
        
        if ENABLE_UPSCALING and realesrgan_path:
            print(f"\n🔍 Loading Real-ESRGAN model for upscaling...")
            upscale_load_start = time.time()
            try:
                upscale_model = load_sd_upscale(realesrgan_path, device)
                print(f"   ✓ Real-ESRGAN model loaded (will {UPSCALE_FACTOR}x upscale resolution)")
                print(f"   ⏱️  Real-ESRGAN loading time: {format_time(time.time() - upscale_load_start)}")
            except Exception as e:
                print(f"   ⚠️  Real-ESRGAN model failed to load: {str(e)[:100]}")
                upscale_model = None
    
    # Ensure CUDA is fully initialized and warm up linear algebra operations
    # This prevents CUSOLVER errors on first run due to uninitialized CUDA context
    if device == "cuda":
        # Warm up CUDA context with a small operation
        try:
            _ = torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
        except:
            pass
        
        # Set preferred backend right before inference (ensures CUDA is ready)
        try:
            if hasattr(torch.backends.cuda, 'preferred_linalg_library'):
                # Try cusolver first (preferred)
                try:
                    torch.backends.cuda.preferred_linalg_library('cusolver')
                    current_backend = 'cusolver'
                    print("   ✓ Linear algebra backend set to 'cusolver' (ready for inference)")
                except (ValueError, RuntimeError):
                    # Fallback to magma
                    try:
                        torch.backends.cuda.preferred_linalg_library('magma')
                        current_backend = 'magma'
                        print("   ✓ Linear algebra backend set to 'magma' (fallback)")
                    except (ValueError, RuntimeError):
                        # Last resort: cublaslt
                        try:
                            torch.backends.cuda.preferred_linalg_library('cublaslt')
                            current_backend = 'cublaslt'
                            print("   ✓ Linear algebra backend set to 'cublaslt' (fallback)")
                        except (ValueError, RuntimeError):
                            current_backend = None
                            print("   ⚠️  Using default linear algebra backend")
        except AttributeError:
            current_backend = None
        
        # Warm up torch.linalg.solve to initialize CUSOLVER before inference
        # This is what the scheduler uses internally, so we need to initialize it
        print("   🔥 Warming up linear algebra operations (prevents first-run CUSOLVER errors)...")
        try:
            # Create small dummy matrices and solve to initialize CUSOLVER
            warmup_A = torch.randn(8, 8, device="cuda", dtype=torch.float32)
            warmup_b = torch.randn(8, device="cuda", dtype=torch.float32)
            # Make A positive definite to ensure solve works
            warmup_A = warmup_A @ warmup_A.T + torch.eye(8, device="cuda", dtype=torch.float32) * 0.1
            with torch.inference_mode():
                _ = torch.linalg.solve(warmup_A, warmup_b)
            torch.cuda.synchronize()
            print("   ✓ Linear algebra operations warmed up (CUSOLVER initialized)")
        except Exception as e:
            print(f"   ⚠️  Warmup failed (may still work): {str(e)[:100]}")
            # Continue anyway - the retry logic will handle it
    
    # Generate video
    print(f"\n🎬 Generating video...")
    print(f"   Prompt: {PROMPT[:100]}...")
    print(f"   Frames: {NUM_FRAMES}")
    print(f"   Steps: {NUM_INFERENCE_STEPS}")
    print(f"   Guidance scale: {GUIDANCE_SCALE}")
    print(f"   Guidance scale 2: {GUIDANCE_SCALE_2}")
    print(f"   Seed: {SEED}")
    print(f"   Resolution: {WIDTH}x{HEIGHT}")
    print(f"\n   💡 Speed optimizations active:")
    print(f"      • Lightning LoRA: {'✓' if lora_loaded else '✗'}")
    print(f"      • Flash Attention: {'✓' if flash_attn_available else '✗'}")
    print(f"      • Inference Mode: ✓")
    print(f"      • CUDA Optimizations: {'✓' if device == 'cuda' else '✗'}")
    if lora_loaded:
        print(f"      💡 Tip: With Lightning LoRA, try reducing steps to 4-8 for 10x faster inference!")
    print(f"\n   🎨 Post-processing options:")
    print(f"      • RIFE Interpolation: {'✓' if (ENABLE_RIFE and rife_model) else '✗'}")
    print(f"      • Real-ESRGAN Upscaling: {'✓' if (ENABLE_UPSCALING and upscale_model) else '✗'}")
    
    generator = torch.Generator(device=device).manual_seed(SEED)
    
    # Use inference mode for faster inference (disables gradient computation)
    print("   🚀 Starting inference with optimizations...")
    generation_start_time = time.time()
    
    # Retry logic for CUSOLVER errors
    max_retries = 3
    backends_to_try = ['cusolver', 'magma', 'cublaslt']  # Try cusolver first, then fallbacks
    
    # Get current backend (set above before inference)
    if device == "cuda" and hasattr(torch.backends.cuda, 'preferred_linalg_library'):
        try:
            current_backend = torch.backends.cuda.preferred_linalg_library()
        except:
            current_backend = None
    else:
        current_backend = None
    
    for attempt in range(max_retries):
        try:
            with torch.inference_mode():  # Faster than no_grad()
                output = pipe(
                    prompt=PROMPT,
                    negative_prompt=NEGATIVE_PROMPT,
                    height=HEIGHT,
                    width=WIDTH,
                    num_frames=NUM_FRAMES,
                    guidance_scale=GUIDANCE_SCALE,
                    guidance_scale_2=GUIDANCE_SCALE_2,
                    num_inference_steps=NUM_INFERENCE_STEPS,
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
                        if backend == current_backend:
                            continue
                        try:
                            print(f"   ⚠️  CUSOLVER error detected (attempt {attempt + 1}/{max_retries}), switching to '{backend}' backend...")
                            torch.backends.cuda.preferred_linalg_library(backend)
                            current_backend = backend
                            print(f"   ✓ Switched to '{backend}', retrying inference...")
                            backend_switched = True
                            break
                        except (ValueError, RuntimeError):
                            continue
                    
                    if not backend_switched:
                        # All backends failed, re-raise
                        raise RuntimeError(
                            "CUSOLVER error persists with all backends. "
                            "This may be a GPU driver or CUDA installation issue. "
                            "Try updating CUDA drivers or using a different GPU."
                        ) from e
                else:
                    # No more retries or can't switch backend
                    raise
            else:
                # Not a CUSOLVER error, re-raise
                raise
    
    generation_time = time.time() - generation_start_time
    print("✓ Video generation completed")
    print(f"   ⏱️  Generation time: {format_time(generation_time)}")
    
    # Apply post-processing (RIFE interpolation and/or Real-ESRGAN upscaling)
    postprocess_time = 0.0
    final_fps = FPS
    final_output = output
    final_height = HEIGHT
    final_width = WIDTH
    
    if POSTPROCESSING_AVAILABLE and (ENABLE_RIFE or ENABLE_UPSCALING):
        print(f"\n🎨 Applying post-processing...")
        postprocess_start = time.time()
        
        # Determine processing order based on configuration
        # Recommended: Interpolate first (faster, more efficient)
        # Alternative: Upscale first (slower but may have slightly better quality in some cases)
        if POSTPROCESS_INTERPOLATE_FIRST:
            # Order: Interpolate → Upscale (may cause errors)
            print(f"   📋 Processing order: Interpolate → Upscale (may cause errors)")
            
            # Apply RIFE frame interpolation first (if enabled)
            if ENABLE_RIFE and rife_model:
                target_fps_str = f"to {RIFE_TARGET_FPS}fps" if RIFE_TARGET_FPS else "doubling frame rate"
                print(f"   🎬 Interpolating frames with RIFE ({target_fps_str})...")
                try:
                    final_output, actual_fps = apply_rife_interpolation(
                        rife_model, final_output, device, original_fps=FPS, target_fps=RIFE_TARGET_FPS
                    )
                    # Update FPS
                    final_fps = actual_fps
                    if RIFE_TARGET_FPS and actual_fps != RIFE_TARGET_FPS:
                        print(f"   ℹ️  Achieved {actual_fps}fps (target was {RIFE_TARGET_FPS}fps)")
                    print(f"   ✓ Frame interpolation completed (new FPS: {final_fps})")
                except Exception as e:
                    print(f"   ⚠️  Frame interpolation failed: {str(e)[:200]}")
                    print("   ℹ️  Continuing with original frame rate")
            
            # Apply Real-ESRGAN upscaling second (if enabled)
            if ENABLE_UPSCALING and upscale_model:
                print(f"   🔍 Upscaling video with Real-ESRGAN ({UPSCALE_FACTOR}x)...")
                try:
                    final_output = apply_realesrgan_upscaling(upscale_model, final_output, device)
                    # Update resolution info
                    final_height = HEIGHT * UPSCALE_FACTOR
                    final_width = WIDTH * UPSCALE_FACTOR
                    print(f"   ✓ Upscaling completed (new resolution: {final_width}x{final_height})")
                except Exception as e:
                    print(f"   ⚠️  Upscaling failed: {str(e)[:200]}")
                    print("   ℹ️  Continuing with original resolution")
        else:
            # Order: Upscale → Interpolate (DEFAULT: more stable)
            print(f"   📋 Processing order: Upscale → Interpolate (more stable)")
            
            # Apply Real-ESRGAN upscaling first (if enabled)
            if ENABLE_UPSCALING and upscale_model:
                print(f"   🔍 Upscaling video with Real-ESRGAN ({UPSCALE_FACTOR}x)...")
                try:
                    final_output = apply_realesrgan_upscaling(upscale_model, final_output, device)
                    # Update resolution info
                    final_height = HEIGHT * UPSCALE_FACTOR
                    final_width = WIDTH * UPSCALE_FACTOR
                    print(f"   ✓ Upscaling completed (new resolution: {final_width}x{final_height})")
                except Exception as e:
                    print(f"   ⚠️  Upscaling failed: {str(e)[:200]}")
                    print("   ℹ️  Continuing with original resolution")
            
            # Apply RIFE frame interpolation second (if enabled)
            if ENABLE_RIFE and rife_model:
                target_fps_str = f"to {RIFE_TARGET_FPS}fps" if RIFE_TARGET_FPS else "doubling frame rate"
                print(f"   🎬 Interpolating frames with RIFE ({target_fps_str})...")
                try:
                    final_output, actual_fps = apply_rife_interpolation(
                        rife_model, final_output, device, original_fps=FPS, target_fps=RIFE_TARGET_FPS
                    )
                    # Update FPS
                    final_fps = actual_fps
                    if RIFE_TARGET_FPS and actual_fps != RIFE_TARGET_FPS:
                        print(f"   ℹ️  Achieved {actual_fps}fps (target was {RIFE_TARGET_FPS}fps)")
                    print(f"   ✓ Frame interpolation completed (new FPS: {final_fps})")
                except Exception as e:
                    print(f"   ⚠️  Frame interpolation failed: {str(e)[:200]}")
                    print("   ℹ️  Continuing with original frame rate")
        
        postprocess_time = time.time() - postprocess_start
        print(f"   ⏱️  Post-processing time: {format_time(postprocess_time)}")
    
    # Export video
    print(f"\n💾 Exporting video to {OUTPUT_VIDEO_PATH}...")
    export_start_time = time.time()
    export_to_video(final_output, OUTPUT_VIDEO_PATH, fps=final_fps)
    export_time = time.time() - export_start_time
    print(f"✓ Video exported successfully!")
    print(f"   Output: {OUTPUT_VIDEO_PATH}")
    print(f"   Resolution: {final_width}x{final_height}")
    print(f"   FPS: {final_fps}")
    print(f"   ⏱️  Export time: {format_time(export_time)}")
    
    total_time = time.time() - total_start_time
    
    # Print timing summary
    print("\n" + "-"*60)
    print("⏱️  Timing Summary:")
    print("-"*60)
    if download_time > 0:
        print(f"   Model download: {format_time(download_time)}")
    print(f"   VAE loading: {format_time(vae_time)}")
    print(f"   Pipeline loading: {format_time(load_time)}")
    print(f"   Video generation: {format_time(generation_time)}")
    if postprocess_time > 0:
        print(f"   Post-processing: {format_time(postprocess_time)}")
    print(f"   Video export: {format_time(export_time)}")
    print(f"   {'─'*50}")
    print(f"   Total time: {format_time(total_time)}")
    print("-"*60)
    
    return OUTPUT_VIDEO_PATH, {
        'download_time': download_time,
        'vae_time': vae_time,
        'load_time': load_time,
        'generation_time': generation_time,
        'postprocess_time': postprocess_time,
        'export_time': export_time,
        'total_time': total_time
    }


if __name__ == "__main__":
    try:
        output_path, timings = run_inference()
        print("\n" + "="*60)
        print("✅ Inference completed successfully!")
        print(f"📹 Output video: {output_path}")
        print(f"⏱️  Total execution time: {format_time(timings['total_time'])}")
        print("="*60)
    except Exception as e:
        print("\n" + "="*60)
        print("❌ Error during inference:")
        print(f"   {str(e)}")
        print("="*60)
        import traceback
        traceback.print_exc()
        exit(1)
