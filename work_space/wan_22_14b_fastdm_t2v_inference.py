#!/usr/bin/env python3
"""
Standalone script for Wan2.2 T2V A14B using FastDM merged Lightning model.

Uses FastDM/Wan2.2-T2V-A14B-Merge-Lightning-V1.0-Diffusers which is a
pre-merged model with Lightning distillation baked into the weights.

Supports FOUR quantization backends (controlled by QUANTIZATION_TYPE flag):
  - "int8"       : optimum.quanto INT8 quantization (~32GB on disk)
                   Created by: python quantize_fastdm.py
  - "sdnq_uint4" : SDNQ UINT4 + SVD rank 32 (~10-15GB on disk)
                   Created by: python quantize_fastdm_sdnq.py (with uint4)
  - "sdnq_int8"  : SDNQ INT8 + SVD rank 32 (~20-28GB on disk)
                   Created by: python quantize_fastdm_sdnq.py (with int8)
  - "sdnq_fp8"   : SDNQ FP8 (float8_e4m3fn) + SVD rank 32 (~20-28GB on disk)
                   Created by: python quantize_fastdm_sdnq.py (with fp8)

Key advantages over SDNQ + Lightx2v LoRA approach:
- No LoRA needed (Lightning distillation is merged into model weights)
- No "unexpected LoRA keys" issue
- Complete Lightning distillation (all weights merged, not partial)
- Works with 4-8 steps and guidance_scale=1.0

Requirements:
  pip install diffusers accelerate transformers sentencepiece protobuf
  # For INT8 mode:
  pip install optimum[quanto]
  # For SDNQ mode:
  pip install sdnq   # or git+https://github.com/Disty0/sdnq

This script:
1. Loads pre-quantized model from local path (INT8 or SDNQ)
2. Uses appropriate memory management per quantization type
3. Supports RIFE frame interpolation and Real-ESRGAN upscaling
"""

import os
import sys
import gc
import random
import time
import torch
import traceback
import numpy as np
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download

os.environ["HF_HOME"] = "/models_storage/all_models/fastdm"
# ============================================================================
# QUANTIZATION TYPE FLAG
# ============================================================================
# "int8"       = optimum.quanto INT8 (created by quantize_fastdm.py, ~32GB)
# "sdnq_uint4" = SDNQ UINT4 + SVD rank 32 (created by quantize_fastdm_sdnq.py with uint4, ~10-15GB)
# "sdnq_int8"  = SDNQ INT8 + SVD rank 32 (created by quantize_fastdm_sdnq.py with int8, ~20-28GB)
# "sdnq_fp8"   = SDNQ FP8 (float8_e4m3fn) + SVD rank 32 (created by quantize_fastdm_sdnq.py with fp8, ~20-28GB)
QUANTIZATION_TYPE = "sdnq_uint4"  # <-- Change this to switch between quantization backends

# ============================================================================
# Conditional imports based on quantization type
# ============================================================================
if QUANTIZATION_TYPE in ["sdnq_uint4", "sdnq_int8", "sdnq_fp8"]:
    # Add local SDNQ source to path (if not installed via pip)
    WORKSPACE_DIR_EARLY = os.path.dirname(os.path.abspath(__file__))
    SDNQ_SRC_DIR = os.path.join(WORKSPACE_DIR_EARLY, "sdnq", "src")
    if os.path.exists(SDNQ_SRC_DIR) and SDNQ_SRC_DIR not in sys.path:
        sys.path.insert(0, SDNQ_SRC_DIR)
        print(f"✓ Added local SDNQ source: {SDNQ_SRC_DIR}")

    import diffusers
    from sdnq import SDNQConfig  # Registers SDNQ quantizer with diffusers/transformers
    from sdnq.loader import apply_sdnq_options_to_model

    WanPipeline = diffusers.WanPipeline
    print(f"✓ SDNQ mode: SDNQ quantizer registered with diffusers")
else:
    from diffusers import WanPipeline

    print(f"✓ INT8 mode: using optimum.quanto")

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
torch.set_float32_matmul_precision("medium")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# ============================================================================
# Model Configuration
# ============================================================================
# HuggingFace repos for quantized models (set after uploading with upload_quantized_fastdm.py)
# Set to None to use local model only, or specify repo ID to download from HuggingFace
# Format: "your-username/your-repo-name"
# NOTE: T2V FastDM merged Lightning repos need to be created/uploaded
QUANTIZED_MODEL_REPOS = {
    "int8": None,  # Example: "your-username/wan22-14b-t2v-lightning-int8"
    "sdnq_uint4": None,  # TODO: "rockapaper/wan22-14b-t2v-lightning-sdnq-uint4-svd-r32"
    "sdnq_int8": None,  # TODO: "rockapaper/wan22-14b-t2v-lightning-sdnq-int8-svd-r32"
    "sdnq_fp8": None,  # TODO: "rockapaper/wan22-14b-t2v-lightning-sdnq-fp8-svd-r32"
}

# Original repo (for reference only)
MODEL_REPO = "FastDM/Wan2.2-T2V-A14B-Merge-Lightning-V1.0-Diffusers"

# Local paths
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")

# Model paths per quantization type (local directories)
MODEL_PATHS = {
    "int8": os.path.join(MODELS_DIR, "fastdm_wan22_t2v_lightning_int8"),
    "sdnq_uint4": os.path.join(MODELS_DIR, "fastdm_wan22_t2v_lightning_sdnq_uint4_svd_r32"),
    "sdnq_int8": os.path.join(MODELS_DIR, "fastdm_wan22_t2v_lightning_sdnq_int8_svd_r32"),
    "sdnq_fp8": os.path.join(MODELS_DIR, "fastdm_wan22_t2v_lightning_sdnq_fp8_svd_r32"),
}
MODEL_LOCAL_DIR = MODEL_PATHS[QUANTIZATION_TYPE]
QUANTIZED_MODEL_REPO = QUANTIZED_MODEL_REPOS.get(QUANTIZATION_TYPE)  # Current quantization type's repo

# Output settings
OUTPUT_VIDEO_PATH = os.path.join(WORKSPACE_DIR, "t2v_fastdm_output.mp4")

# ============================================================================
# Inference Parameters
# ============================================================================
# NOTE: FastDM merged Lightning model works with 4-8 steps and guidance_scale=1.0
# since the Lightning distillation is fully merged into the model weights.

PROMPT = "A wide shot of a dusty, neon-lit cyberpunk street at night, rain-slicked pavement reflecting blue and pink lights. A mysterious figure in a trench coat walks slowly towards the camera while smoke rises from a manhole. Cinematic, 35mm film style, slow motion."

NEGATIVE_PROMPT = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景, 三条腿, 背景人很多, 倒着走"

NUM_FRAMES = 81
GUIDANCE_SCALE = 1.0  # Lightning-merged model uses 1.0
GUIDANCE_SCALE_2 = 1.0
NUM_INFERENCE_STEPS = 8  # Lightning-merged model works with 4-8 steps
FPS = 16
SEED = random.randint(0, 1000)

# Resolution
HEIGHT = 480
WIDTH = 832

# Post-processing options
ENABLE_RIFE = True
RIFE_TARGET_FPS = 30
ENABLE_UPSCALING = True
UPSCALE_FACTOR = 2
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


def check_quantized_model():
    """Check if quantized model exists locally or download from HuggingFace."""
    if QUANTIZATION_TYPE == "int8":
        quant_label = "INT8 (optimum.quanto)"
        quant_script = "quantize_fastdm.py"
    elif QUANTIZATION_TYPE == "sdnq_uint4":
        quant_label = "SDNQ UINT4+SVD r32"
        quant_script = "quantize_fastdm_sdnq.py"
    elif QUANTIZATION_TYPE == "sdnq_int8":
        quant_label = "SDNQ INT8+SVD r32"
        quant_script = "quantize_fastdm_sdnq.py"
    elif QUANTIZATION_TYPE == "sdnq_fp8":
        quant_label = "SDNQ FP8+SVD r32"
        quant_script = "quantize_fastdm_sdnq.py"
    else:
        quant_label = "Unknown"
        quant_script = "quantize_fastdm.py"

    # Check if local model exists
    if os.path.exists(MODEL_LOCAL_DIR):
        # Diffusers pipelines use model_index.json; SDNQ models may use config.json
        model_index = os.path.join(MODEL_LOCAL_DIR, "model_index.json")
        config_file = os.path.join(MODEL_LOCAL_DIR, "config.json")
        # SDNQ also creates quantization_config.json in subdirectories
        quant_config = os.path.join(MODEL_LOCAL_DIR, "quantization_config.json")

        if (
            os.path.exists(model_index)
            or os.path.exists(config_file)
            or os.path.exists(quant_config)
        ):
            print(
                f"✓ Quantized model ({quant_label}) found locally at {MODEL_LOCAL_DIR}"
            )
            return MODEL_LOCAL_DIR, 0.0
        else:
            print(f"⚠️ Model directory exists but appears incomplete")
            print(
                f"   Expected model_index.json, config.json, or quantization_config.json"
            )

    # Try downloading from HuggingFace if repo is specified
    if QUANTIZED_MODEL_REPO:
        if QUANTIZATION_TYPE == "int8":
            size_hint = "~32GB"
        elif QUANTIZATION_TYPE == "sdnq_uint4":
            size_hint = "~10-15GB"
        else:  # sdnq_int8 or sdnq_fp8
            size_hint = "~20-28GB"
        print(
            f"\n⬇️ Downloading quantized model ({quant_label}) from HuggingFace: {QUANTIZED_MODEL_REPO}"
        )
        print(f"   This is a one-time download ({size_hint})")
        os.makedirs(MODELS_DIR, exist_ok=True)

        start_time = time.time()
        try:
            snapshot_download(
                QUANTIZED_MODEL_REPO,
                local_dir=MODEL_LOCAL_DIR,
                local_dir_use_symlinks=False,
            )
            elapsed_time = time.time() - start_time
            print(f"✓ Model downloaded successfully to {MODEL_LOCAL_DIR}")
            print(f"   ⏱️  Download time: {format_time(elapsed_time)}")
            return MODEL_LOCAL_DIR, elapsed_time
        except Exception as e:
            print(f"❌ Failed to download from HuggingFace: {e}")
            print(f"   Falling back to local quantization instructions...")

    # Model not found - prompt user
    print(f"\n❌ Quantized model ({quant_label}) not found at {MODEL_LOCAL_DIR}")
    print(f"\n📋 Options to get the quantized model:")
    
    if QUANTIZED_MODEL_REPO:
        print(f"\n   1. Download from HuggingFace (recommended):")
        print(f"      Model will be downloaded from: {QUANTIZED_MODEL_REPO}")
        print(f"      Set QUANTIZED_MODEL_REPOS['{QUANTIZATION_TYPE}'] to change the repo")
    else:
        print(f"\n   1. Download from HuggingFace (if available):")
        print(f"      Set QUANTIZED_MODEL_REPOS['{QUANTIZATION_TYPE}'] in this script to your repo ID")
        print(f"      Example: QUANTIZED_MODEL_REPOS['{QUANTIZATION_TYPE}'] = 'your-username/your-repo-name'")
    
    print(f"\n   2. Create locally (one-time):")
    print(f"      python {quant_script}")
    print(f"      This will:")
    print(f"      - Download the 126GB FastDM model (one-time)")
    print(f"      - Quantize it to {quant_label}")
    print(f"      - Save it to {MODEL_LOCAL_DIR}")
    print(
        f"\n   3. Upload your quantized model to HuggingFace (after creating locally):"
    )
    print(f"      python upload_quantized_fastdm.py --model-type {QUANTIZATION_TYPE}")

    raise FileNotFoundError(
        f"Quantized model not found. Please set QUANTIZED_MODEL_REPOS['{QUANTIZATION_TYPE}'] or run {quant_script}"
    )


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


def fix_tied_weights(pipe):
    """Fix tied weights: shared.weight -> encoder.embed_tokens.weight.

    UMT5EncoderModel ties shared.weight to encoder.embed_tokens.weight,
    but diffusers.from_pretrained() doesn't always restore this.
    """
    if hasattr(pipe, "text_encoder"):
        te = pipe.text_encoder
        if (
            hasattr(te, "shared")
            and hasattr(te, "encoder")
            and hasattr(te.encoder, "embed_tokens")
        ):
            shared_ptr = te.shared.weight.data_ptr()
            embed_ptr = te.encoder.embed_tokens.weight.data_ptr()
            if shared_ptr != embed_ptr:
                te.encoder.embed_tokens.weight = te.shared.weight
                print(
                    "   ✓ Fixed: Re-tied shared.weight -> encoder.embed_tokens.weight"
                )
            else:
                print("   ✓ Text encoder weights already tied correctly")
        if hasattr(te, "tie_weights"):
            te.tie_weights()
    else:
        print("   ⚠️  Text encoder not found in pipeline!")


def load_pipeline_sdnq(model_path, device):
    """Load pipeline using SDNQ quantization backend.

    SDNQ models can be loaded directly with from_pretrained() (SDNQ registers
    itself with diffusers/transformers on import). For larger models (FP8),
    we use more aggressive memory management with sequential CPU offload.
    """
    if QUANTIZATION_TYPE == "sdnq_uint4":
        model_type_str = "UINT4+SVD"
    elif QUANTIZATION_TYPE == "sdnq_int8":
        model_type_str = "INT8+SVD"
    else:  # sdnq_fp8
        model_type_str = "FP8+SVD"
    print(f"   Loading SDNQ {model_type_str} quantized model...")
    
    # Load VAE separately (required for T2V pipeline)
    print(f"   Loading VAE...")
    vae = diffusers.AutoencoderKLWan.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch.float32
    )
    
    pipe = WanPipeline.from_pretrained(
        model_path,
        vae=vae,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,  # Load components on CPU first
    )

    # Fix tied weights before device management
    fix_tied_weights(pipe)

    # For FP8 models (~20-28GB), use sequential CPU offload which is more aggressive
    # than enable_model_cpu_offload(). Sequential offload moves components between
    # CPU and GPU more frequently, reducing peak VRAM usage.
    if device == "cuda":
        try:
            pipe.enable_sequential_cpu_offload()
            print("   ✓ Sequential CPU offload enabled (aggressive memory management)")
        except Exception as e1:
            print(f"   ⚠️  Sequential CPU offload failed: {e1}")
            try:
                # Fallback to regular CPU offload
                pipe.enable_model_cpu_offload()
                print("   ✓ Model CPU offload enabled (fallback)")
            except Exception as e2:
                print(f"   ⚠️  CPU offload failed: {e2}")
                # Last resort: move components individually but keep on CPU longer
                print("   ⚠️  Using manual component management")
                # Don't move to GPU - keep on CPU and let inference move as needed
    else:
        pipe = pipe.to(device)
        print(f"   ✓ Model moved to {device}")

    return pipe


def load_pipeline_int8(model_path, device):
    """Load pipeline using optimum.quanto INT8 quantization backend.

    INT8 quantized models need low_cpu_mem_usage=True and sequential CPU offload
    because enable_model_cpu_offload() fails with meta tensor errors.
    """
    print(f"   Loading INT8 (optimum.quanto) quantized model...")
    
    # Load VAE separately (required for T2V pipeline)
    print(f"   Loading VAE...")
    vae = diffusers.AutoencoderKLWan.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch.float32
    )
    
    pipe = WanPipeline.from_pretrained(
        model_path,
        vae=vae,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,  # Required for quanto quantized models
    )

    # Fix tied weights before device management
    fix_tied_weights(pipe)

    # INT8 quantized models from optimum.quanto may have meta tensors.
    # enable_model_cpu_offload() tries pipe.to("cpu") first which fails on meta tensors.
    # Use sequential CPU offload which is more compatible.
    if device == "cuda":
        try:
            pipe.enable_sequential_cpu_offload()
            print("   ✓ Sequential CPU offload enabled (compatible with quanto INT8)")
        except Exception as e1:
            print(f"   ⚠️  Sequential CPU offload failed: {e1}")
            try:
                pipe.enable_model_cpu_offload()
                print("   ✓ Model CPU offload enabled (fallback)")
            except Exception as e2:
                print(f"   ⚠️  CPU offload failed: {e2}")
                # Last resort: move components individually
                try:
                    for name in ["transformer", "transformer_2", "text_encoder", "vae"]:
                        if hasattr(pipe, name):
                            setattr(pipe, name, getattr(pipe, name).to(device))
                    print(f"   ✓ Model components moved to {device} individually")
                except Exception as e3:
                    print(f"   ⚠️  Direct device move also failed: {e3}")
                    print("   ℹ️  Continuing without explicit device management")
    else:
        pipe = pipe.to(device)
        print(f"   ✓ Model moved to {device}")

    return pipe


def run_inference():
    """Run text-to-video inference using FastDM merged Lightning model."""
    total_start_time = time.time()

    if QUANTIZATION_TYPE == "int8":
        quant_label = "INT8 (optimum.quanto)"
    elif QUANTIZATION_TYPE == "sdnq_uint4":
        quant_label = "SDNQ UINT4+SVD r32"
    elif QUANTIZATION_TYPE == "sdnq_int8":
        quant_label = "SDNQ INT8+SVD r32"
    elif QUANTIZATION_TYPE == "sdnq_fp8":
        quant_label = "SDNQ FP8+SVD r32"
    else:
        quant_label = "Unknown"
    print("\n" + "=" * 60)
    print(f"Starting Text-to-Video Inference (FastDM Lightning - {quant_label})")
    print("=" * 60)

    # ================================================================
    # 1. Check for quantized model and download post-processing models
    # ================================================================
    model_path, model_check_time = check_quantized_model()
    rife_path, realesrgan_path, postproc_download_time = (
        download_postprocessing_models()
    )
    download_time = model_check_time + postproc_download_time

    # ================================================================
    # 2. Determine device
    # ================================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA: {torch.version.cuda}")
        try:
            mem = torch.cuda.get_device_properties(0).total_mem
        except AttributeError:
            mem = torch.cuda.get_device_properties(0).total_memory
        print(f"   Memory: {mem / 1024**3:.1f} GB")

    # ================================================================
    # 3. Load pipeline (quantization-type specific)
    # ================================================================
    print(f"\n📦 Loading pipeline from {model_path} ({quant_label})...")
    load_start_time = time.time()

    if QUANTIZATION_TYPE in ["sdnq_uint4", "sdnq_int8", "sdnq_fp8"]:
        pipe = load_pipeline_sdnq(model_path, device)
    else:
        pipe = load_pipeline_int8(model_path, device)

    # ================================================================
    # 4. Apply memory and speed optimizations
    # ================================================================
    print("   ⚡ Applying memory and speed optimizations...")

    # Clear CUDA cache to free up any memory from loading
    if device == "cuda":
        torch.cuda.empty_cache()
        print("   ✓ CUDA cache cleared")

    flash_attn_available = False
    try:
        import flash_attn

        flash_attn_available = True
        print(
            f"   ✓ Flash Attention detected (version: {getattr(flash_attn, '__version__', 'unknown')})"
        )
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
    print(f"✓ Pipeline loaded and optimized ({format_time(load_time)})")

    # ================================================================
    # 5. Load post-processing models
    # ================================================================
    rife_model = None
    upscale_model = None
    if POSTPROCESSING_AVAILABLE:
        if ENABLE_RIFE:
            if not check_rife_code_available():
                print(f"   ⚠️  RIFE code not available, frame interpolation disabled")
            elif rife_path:
                print(f"\n🎬 Loading RIFE model for frame interpolation...")
                rife_load_start = time.time()
                rife_model = load_rife_model(rife_path)
                if rife_model:
                    target_fps_str = (
                        f"{RIFE_TARGET_FPS}fps"
                        if RIFE_TARGET_FPS
                        else f"{FPS*2}fps (2x)"
                    )
                    print(f"   ✓ RIFE model loaded (target: {target_fps_str})")
                    print(
                        f"   ⏱️  RIFE loading: {format_time(time.time() - rife_load_start)}"
                    )
                else:
                    print(f"   ⚠️  RIFE model failed to load")

        if ENABLE_UPSCALING and realesrgan_path:
            print(f"\n🔍 Loading Real-ESRGAN model for upscaling...")
            upscale_load_start = time.time()
            try:
                upscale_model = load_sd_upscale(realesrgan_path, device)
                print(f"   ✓ Real-ESRGAN model loaded ({UPSCALE_FACTOR}x)")
                print(
                    f"   ⏱️  Real-ESRGAN loading: {format_time(time.time() - upscale_load_start)}"
                )
            except Exception as e:
                print(f"   ⚠️  Real-ESRGAN failed to load: {str(e)[:100]}")
                upscale_model = None

    # ================================================================
    # 6. Generate video
    # ================================================================
    print(f"\n🎬 Generating video...")
    print(f"   Prompt: {PROMPT[:100]}...")
    print(
        f"   Frames: {NUM_FRAMES}, Steps: {NUM_INFERENCE_STEPS}, "
        f"Guidance: {GUIDANCE_SCALE}, Seed: {SEED}"
    )
    print(f"   Resolution: {WIDTH}x{HEIGHT}")

    generator = torch.Generator(device=device).manual_seed(SEED)

    # Clear CUDA cache before inference to maximize available VRAM
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    print("   🚀 Starting inference...")
    generation_start_time = time.time()

    with torch.inference_mode():
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

    generation_time = time.time() - generation_start_time
    print(f"✓ Video generation completed ({format_time(generation_time)})")

    # ================================================================
    # 7. Apply post-processing
    # ================================================================
    postprocess_time = 0.0
    final_fps = FPS
    final_output = output
    final_height = HEIGHT
    final_width = WIDTH

    if POSTPROCESSING_AVAILABLE and (ENABLE_RIFE or ENABLE_UPSCALING):
        print(f"\n🎨 Applying post-processing...")
        postprocess_start = time.time()

        if POSTPROCESS_INTERPOLATE_FIRST:
            print(f"   📋 Order: Interpolate → Upscale")

            if ENABLE_RIFE and rife_model:
                target_fps_str = (
                    f"to {RIFE_TARGET_FPS}fps"
                    if RIFE_TARGET_FPS
                    else "doubling frame rate"
                )
                print(f"   🎬 Interpolating frames with RIFE ({target_fps_str})...")
                try:
                    final_output, actual_fps = apply_rife_interpolation(
                        rife_model,
                        final_output,
                        device,
                        original_fps=FPS,
                        target_fps=RIFE_TARGET_FPS,
                    )
                    final_fps = actual_fps
                    print(f"   ✓ Frame interpolation completed (FPS: {final_fps})")
                except Exception as e:
                    print(f"   ⚠️  Frame interpolation failed: {str(e)[:200]}")

            if ENABLE_UPSCALING and upscale_model:
                print(f"   🔍 Upscaling with Real-ESRGAN ({UPSCALE_FACTOR}x)...")
                try:
                    final_output = apply_realesrgan_upscaling(
                        upscale_model, final_output, device
                    )
                    final_height = HEIGHT * UPSCALE_FACTOR
                    final_width = WIDTH * UPSCALE_FACTOR
                    print(f"   ✓ Upscaling completed ({final_width}x{final_height})")
                except Exception as e:
                    print(f"   ⚠️  Upscaling failed: {str(e)[:200]}")
        else:
            print(f"   📋 Order: Upscale → Interpolate")

            if ENABLE_UPSCALING and upscale_model:
                print(f"   🔍 Upscaling with Real-ESRGAN ({UPSCALE_FACTOR}x)...")
                try:
                    final_output = apply_realesrgan_upscaling(
                        upscale_model, final_output, device
                    )
                    final_height = HEIGHT * UPSCALE_FACTOR
                    final_width = WIDTH * UPSCALE_FACTOR
                    print(f"   ✓ Upscaling completed ({final_width}x{final_height})")
                except Exception as e:
                    print(f"   ⚠️  Upscaling failed: {str(e)[:200]}")

            if ENABLE_RIFE and rife_model:
                target_fps_str = (
                    f"to {RIFE_TARGET_FPS}fps"
                    if RIFE_TARGET_FPS
                    else "doubling frame rate"
                )
                print(f"   🎬 Interpolating frames with RIFE ({target_fps_str})...")
                try:
                    final_output, actual_fps = apply_rife_interpolation(
                        rife_model,
                        final_output,
                        device,
                        original_fps=FPS,
                        target_fps=RIFE_TARGET_FPS,
                    )
                    final_fps = actual_fps
                    print(f"   ✓ Frame interpolation completed (FPS: {final_fps})")
                except Exception as e:
                    print(f"   ⚠️  Frame interpolation failed: {str(e)[:200]}")

        postprocess_time = time.time() - postprocess_start
        print(f"   ⏱️  Post-processing: {format_time(postprocess_time)}")

    # ================================================================
    # 8. Export video
    # ================================================================
    print(f"\n💾 Exporting video to {OUTPUT_VIDEO_PATH}...")
    export_start_time = time.time()
    export_to_video(final_output, OUTPUT_VIDEO_PATH, fps=final_fps)
    export_time = time.time() - export_start_time
    print(f"✓ Video exported!")
    print(f"   Output: {OUTPUT_VIDEO_PATH}")
    print(f"   Resolution: {final_width}x{final_height}")
    print(f"   FPS: {final_fps}")

    total_time = time.time() - total_start_time

    # Print summary
    print("\n" + "=" * 60)
    print("📊 Inference Summary")
    print("=" * 60)
    if download_time > 0:
        print(f"   Model download:    {format_time(download_time)}")
    print(f"   Pipeline loading:  {format_time(load_time)}")
    print(f"   Video generation:  {format_time(generation_time)}")
    if postprocess_time > 0:
        print(f"   Post-processing:   {format_time(postprocess_time)}")
    print(f"   Video export:      {format_time(export_time)}")
    print(f"   {'─'*50}")
    print(f"   Total time:        {format_time(total_time)}")
    print("=" * 60)

    return OUTPUT_VIDEO_PATH, {
        "download_time": download_time,
        "load_time": load_time,
        "generation_time": generation_time,
        "postprocess_time": postprocess_time,
        "export_time": export_time,
        "total_time": total_time,
    }


if __name__ == "__main__":
    try:
        output_path, timings = run_inference()
        print("\n" + "=" * 60)
        print("✅ Inference completed successfully!")
        print(f"📹 Output video: {output_path}")
        print(f"⏱️  Total execution time: {format_time(timings['total_time'])}")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ Error during inference:")
        print(f"   {str(e)}")
        print("=" * 60)
        traceback.print_exc()
        exit(1)
