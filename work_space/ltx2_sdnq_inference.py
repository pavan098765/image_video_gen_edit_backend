#!/usr/bin/env python3
"""
Standalone script for 4-bit dynamic quantization of LTX-2 using SDNQ.

This script:
1. Downloads the model locally using huggingface snapshot_download
2. Runs text-to-video inference using the local model path
"""

import os
import time
import torch
import numpy as np
import diffusers
from diffusers.pipelines.ltx2.export_utils import encode_video
from huggingface_hub import snapshot_download
from sdnq import SDNQConfig  # import sdnq to register it into diffusers and transformers
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:512,expandable_segments:True"

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
MODEL_REPO = "Disty0/LTX-2-SDNQ-8bit-dynamic"
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
MODEL_LOCAL_DIR = os.path.join(MODELS_DIR, "ltx2_sdnq_8bit_dynamic")

# Output settings
OUTPUT_VIDEO_PATH = os.path.join(WORKSPACE_DIR, "ltx2_t2v_sdnq_output.mp4")

# Inference parameters
PROMPT ="A wide shot of a dusty, neon-lit cyberpunk street at night, rain-slicked pavement reflecting blue and pink lights. A mysterious figure in a trench coat walks slowly towards the camera while smoke rises from a manhole. Cinematic, 35mm film style, slow motion."
NEGATIVE_PROMPT = "blurry, low quality, still frame, frames, watermark, overlay, titles, has blurbox, has subtitles"

WIDTH = 768
HEIGHT = 512
NUM_FRAMES = 121
FRAME_RATE = 25.0
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 4.0
SEED = 10

# SDNQ optimizations
USE_QUANTIZED_MATMUL = True  # Enable INT8 MatMul for AMD, Intel ARC and Nvidia GPUs
USE_TORCH_COMPILE = False  # Optional torch.compile for faster speeds (disabled for non-persistent pods)


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


def run_inference():
    """Run text-to-video inference using the local model."""
    total_start_time = time.time()
    
    print("\n" + "="*60)
    print("Starting LTX-2 Text-to-Video Inference")
    print("="*60)
    
    # Download model if needed
    model_path, download_time = download_model()
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA: {torch.version.cuda}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load pipeline from local path
    print(f"\n📦 Loading pipeline from {model_path}...")
    load_start_time = time.time()
    pipe = diffusers.LTX2Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )
    
    # Apply SDNQ optimizations
    print("   ⚡ Applying SDNQ optimizations...")
    if USE_QUANTIZED_MATMUL and triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
        print("   ✓ Enabling INT8 MatMul for transformer and text_encoder...")
        pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
        pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
        print("   ✓ INT8 MatMul enabled (faster inference)")
    else:
        if not triton_is_available:
            print("   ⚠️  Triton not available, skipping quantized MatMul")
        elif not (torch.cuda.is_available() or torch.xpu.is_available()):
            print("   ⚠️  CUDA/XPU not available, skipping quantized MatMul")
    
    # Optional torch.compile (disabled for non-persistent pods)
    if USE_TORCH_COMPILE:
        try:
            pipe.transformer = torch.compile(pipe.transformer)
            print("   ✓ Transformer compiled with torch.compile")
        except Exception as e:
            print(f"   ⚠️  Torch compile failed: {str(e)[:100]}")
    
    # Enable VAE tiling
    pipe.vae.enable_tiling()
    print("   ✓ VAE tiling enabled")
    
    # Enable model CPU offload
    pipe.enable_model_cpu_offload()
    print("   ✓ Model CPU offload enabled")
    
    # Flash Attention detection
    flash_attn_available = False
    try:
        import flash_attn
        flash_attn_available = True
        print(f"   ✓ Flash Attention detected (version: {getattr(flash_attn, '__version__', 'unknown')})")
        print("   ✓ Flash Attention will be used automatically by diffusers")
    except ImportError:
        print("   ⚠️  Flash Attention not found")
    
    # XFormers as fallback
    if not flash_attn_available:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("   ✓ XFormers memory-efficient attention enabled")
        except Exception as e:
            print(f"   ℹ️  XFormers not available: {str(e)[:100]}")
    
    load_time = time.time() - load_start_time
    print("✓ Pipeline loaded and optimized successfully")
    print(f"   ⏱️  Loading time: {format_time(load_time)}")
    
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
    print(f"   Seed: {SEED}")
    print(f"   Resolution: {WIDTH}x{HEIGHT}")
    print(f"   Frame rate: {FRAME_RATE} fps")
    print(f"\n   💡 Speed optimizations active:")
    print(f"      • SDNQ INT8 MatMul: {'✓' if (USE_QUANTIZED_MATMUL and triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available())) else '✗'}")
    print(f"      • Flash Attention: {'✓' if flash_attn_available else '✗'}")
    print(f"      • VAE Tiling: ✓")
    print(f"      • Inference Mode: ✓")
    print(f"      • CUDA Optimizations: {'✓' if device == 'cuda' else '✗'}")
    
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
                video, audio = pipe(
                    prompt=PROMPT,
                    negative_prompt=NEGATIVE_PROMPT,
                    width=WIDTH,
                    height=HEIGHT,
                    num_frames=NUM_FRAMES,
                    frame_rate=FRAME_RATE,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=generator,
                    output_type="np",
                    return_dict=False,
                )
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
    
    # Process video output
    print(f"\n🎞️ Processing video output...")
    process_start_time = time.time()
    video = (video * 255).round().astype("uint8")
    video = torch.from_numpy(video)
    process_time = time.time() - process_start_time
    print(f"   ⏱️  Processing time: {format_time(process_time)}")
    
    # Export video with audio
    print(f"\n💾 Exporting video to {OUTPUT_VIDEO_PATH}...")
    export_start_time = time.time()
    encode_video(
        video[0],
        fps=FRAME_RATE,
        audio=audio[0].float().cpu(),
        audio_sample_rate=pipe.vocoder.config.output_sampling_rate,  # should be 24000
        output_path=OUTPUT_VIDEO_PATH,
    )
    export_time = time.time() - export_start_time
    print(f"✓ Video exported successfully!")
    print(f"   Output: {OUTPUT_VIDEO_PATH}")
    print(f"   ⏱️  Export time: {format_time(export_time)}")
    
    total_time = time.time() - total_start_time
    
    # Print timing summary
    print("\n" + "-"*60)
    print("⏱️  Timing Summary:")
    print("-"*60)
    if download_time > 0:
        print(f"   Model download: {format_time(download_time)}")
    print(f"   Pipeline loading: {format_time(load_time)}")
    print(f"   Video generation: {format_time(generation_time)}")
    print(f"   Video processing: {format_time(process_time)}")
    print(f"   Video export: {format_time(export_time)}")
    print(f"   {'─'*50}")
    print(f"   Total time: {format_time(total_time)}")
    print("-"*60)
    
    return OUTPUT_VIDEO_PATH, {
        'download_time': download_time,
        'load_time': load_time,
        'generation_time': generation_time,
        'process_time': process_time,
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
