#!/usr/bin/env python3
"""
Standalone script for Wan2.2 I2V A14B using GGUF quantization.

Uses QuantStack/Wan2.2-I2V-A14B-GGUF with Q6_K quantization for
high-quality inference with reduced VRAM usage.

The HighNoise and LowNoise transformers enable few-step generation:
- HighNoise transformer handles early (high-noise) denoising steps
- LowNoise transformer handles later (low-noise) denoising steps

Key differences from SDNQ + Lightx2v LoRA approach:
- No LoRA needed (distillation may be baked into the model weights)
- No "unexpected LoRA keys" issue
- No embed_tokens weight re-tying fix needed
- GGUF quantization handled natively by diffusers
- Dual-transformer architecture for efficient few-step generation

Requirements:
  pip install diffusers[gguf] accelerate transformers sentencepiece protobuf

This script:
1. Downloads GGUF models locally using hf_hub_download
2. Loads quantized transformers with GGUFQuantizationConfig
3. Creates pipeline with dual transformers for efficient inference
4. Supports RIFE frame interpolation and Real-ESRGAN upscaling
"""

import os
import random
import time
import torch
import traceback
import numpy as np
from diffusers import (
    WanImageToVideoPipeline,
    WanTransformer3DModel,
    UniPCMultistepScheduler,
    GGUFQuantizationConfig,
)
from diffusers.utils import export_to_video, load_image
from huggingface_hub import hf_hub_download

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
# GGUF Model Configuration
# ============================================================================
GGUF_REPO = "QuantStack/Wan2.2-I2V-A14B-GGUF"
HIGH_NOISE_FILENAME = "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q6_K.gguf"
LOW_NOISE_FILENAME = "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q6_K.gguf"

# Base model repo (for pipeline components: text_encoder, VAE, scheduler, tokenizer)
# The transformers are loaded from GGUF files, NOT from this repo.
BASE_MODEL_REPO = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

# Config source for transformer architecture definition.
# Both HighNoise and LowNoise share the same WanTransformer3DModel architecture,
# only the weights differ. We use the base model's transformer config for both.
TRANSFORMER_CONFIG_REPO = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
TRANSFORMER_CONFIG_SUBFOLDER = "transformer"

# Local paths
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
GGUF_LOCAL_DIR = os.path.join(MODELS_DIR, "gguf_wan22_i2v_q6k")

# Input image path
INPUT_IMAGE_PATH = os.path.join(WORKSPACE_DIR, "testimg.jpeg")

# Output settings
OUTPUT_VIDEO_PATH = os.path.join(WORKSPACE_DIR, "i2v_gguf_output.mp4")

# ============================================================================
# Inference Parameters
# ============================================================================
# NOTE: The HighNoise/LowNoise dual-transformer setup suggests Lightning-style
# distillation. If the GGUF was quantized from a Lightning-merged model,
# 4-8 steps with guidance_scale=1.0 should work well.
# If quality is poor, try increasing steps to 14-20 and guidance_scale to 5.0.

PROMPT = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

NEGATIVE_PROMPT = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景, 三条腿, 背景人很多, 倒着走"

NUM_FRAMES = 81
GUIDANCE_SCALE = 1.0  # Use 1.0 for Lightning-distilled, 5.0 for non-distilled
NUM_INFERENCE_STEPS = 8  # Use 4-8 for Lightning-distilled, 20-30 for non-distilled
FPS = 16
SEED = random.randint(0, 1000)

# Scheduler options
# flow_shift controls noise schedule - higher values shift more noise to early steps.
# 8.0 is recommended for the dual-transformer setup.
FLOW_SHIFT = 8.0

# Image processing options
RESIZE_IMAGE = True

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


def download_gguf_models():
    """Download GGUF model files locally if not already present."""
    os.makedirs(GGUF_LOCAL_DIR, exist_ok=True)

    high_noise_local = os.path.join(GGUF_LOCAL_DIR, os.path.basename(HIGH_NOISE_FILENAME))
    low_noise_local = os.path.join(GGUF_LOCAL_DIR, os.path.basename(LOW_NOISE_FILENAME))

    total_time = 0.0

    # Download HighNoise GGUF
    if os.path.exists(high_noise_local):
        size_gb = os.path.getsize(high_noise_local) / (1024**3)
        print(f"✓ HighNoise GGUF already exists at {high_noise_local} ({size_gb:.2f} GB)")
    else:
        print(f"⬇️ Downloading HighNoise GGUF from {GGUF_REPO}...")
        print(f"   File: {HIGH_NOISE_FILENAME}")
        start_time = time.time()
        try:
            downloaded_path = hf_hub_download(
                repo_id=GGUF_REPO,
                filename=HIGH_NOISE_FILENAME,
                local_dir=GGUF_LOCAL_DIR,
            )
            elapsed = time.time() - start_time
            total_time += elapsed
            # hf_hub_download with local_dir stores in subdirectory matching the repo path
            # Move to flat structure if needed
            actual_path = os.path.join(GGUF_LOCAL_DIR, HIGH_NOISE_FILENAME)
            if os.path.exists(actual_path) and actual_path != high_noise_local:
                import shutil
                shutil.move(actual_path, high_noise_local)
                # Clean up empty subdirectory
                subdir = os.path.join(GGUF_LOCAL_DIR, os.path.dirname(HIGH_NOISE_FILENAME))
                if os.path.isdir(subdir) and not os.listdir(subdir):
                    os.rmdir(subdir)
            elif downloaded_path != high_noise_local and os.path.exists(downloaded_path):
                import shutil
                shutil.copy2(downloaded_path, high_noise_local)

            size_gb = os.path.getsize(high_noise_local) / (1024**3)
            print(f"   ✓ Downloaded ({size_gb:.2f} GB, {format_time(elapsed)})")
        except Exception as e:
            print(f"   ❌ Failed to download HighNoise GGUF: {e}")
            raise

    # Download LowNoise GGUF
    if os.path.exists(low_noise_local):
        size_gb = os.path.getsize(low_noise_local) / (1024**3)
        print(f"✓ LowNoise GGUF already exists at {low_noise_local} ({size_gb:.2f} GB)")
    else:
        print(f"⬇️ Downloading LowNoise GGUF from {GGUF_REPO}...")
        print(f"   File: {LOW_NOISE_FILENAME}")
        start_time = time.time()
        try:
            downloaded_path = hf_hub_download(
                repo_id=GGUF_REPO,
                filename=LOW_NOISE_FILENAME,
                local_dir=GGUF_LOCAL_DIR,
            )
            elapsed = time.time() - start_time
            total_time += elapsed
            actual_path = os.path.join(GGUF_LOCAL_DIR, LOW_NOISE_FILENAME)
            if os.path.exists(actual_path) and actual_path != low_noise_local:
                import shutil
                shutil.move(actual_path, low_noise_local)
                subdir = os.path.join(GGUF_LOCAL_DIR, os.path.dirname(LOW_NOISE_FILENAME))
                if os.path.isdir(subdir) and not os.listdir(subdir):
                    os.rmdir(subdir)
            elif downloaded_path != low_noise_local and os.path.exists(downloaded_path):
                import shutil
                shutil.copy2(downloaded_path, low_noise_local)

            size_gb = os.path.getsize(low_noise_local) / (1024**3)
            print(f"   ✓ Downloaded ({size_gb:.2f} GB, {format_time(elapsed)})")
        except Exception as e:
            print(f"   ❌ Failed to download LowNoise GGUF: {e}")
            raise

    return high_noise_local, low_noise_local, total_time


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
    """Run image-to-video inference using GGUF quantized models."""
    total_start_time = time.time()

    print("\n" + "=" * 60)
    print("Starting Image-to-Video Inference (GGUF Q6_K)")
    print("=" * 60)

    # ================================================================
    # 1. Download models
    # ================================================================
    high_noise_path, low_noise_path, gguf_download_time = download_gguf_models()
    rife_path, realesrgan_path, postproc_download_time = download_postprocessing_models()
    download_time = gguf_download_time + postproc_download_time

    # ================================================================
    # 2. Determine device
    # ================================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA: {torch.version.cuda}")
        print(
            f"   Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB"
            if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
            else f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    # ================================================================
    # 3. Load GGUF quantized transformers
    # ================================================================
    print(f"\n📦 Loading GGUF quantized transformers...")
    load_start_time = time.time()

    gguf_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)

    print(f"   Loading HighNoise transformer from {os.path.basename(high_noise_path)}...")
    transformer_high_noise = WanTransformer3DModel.from_single_file(
        high_noise_path,
        quantization_config=gguf_config,
        config=TRANSFORMER_CONFIG_REPO,
        subfolder=TRANSFORMER_CONFIG_SUBFOLDER,
        torch_dtype=torch.bfloat16,
    )
    print(f"   ✓ HighNoise transformer loaded")

    print(f"   Loading LowNoise transformer from {os.path.basename(low_noise_path)}...")
    transformer_low_noise = WanTransformer3DModel.from_single_file(
        low_noise_path,
        quantization_config=gguf_config,
        config=TRANSFORMER_CONFIG_REPO,
        subfolder=TRANSFORMER_CONFIG_SUBFOLDER,
        torch_dtype=torch.bfloat16,
    )
    print(f"   ✓ LowNoise transformer loaded")

    # ================================================================
    # 4. Create pipeline
    # ================================================================
    # Load the pipeline from the base model repo. The transformer and transformer_2
    # are overridden with our GGUF-quantized versions. Only text_encoder, VAE,
    # scheduler, and tokenizer are loaded from the base repo.
    print(f"\n📦 Creating pipeline from {BASE_MODEL_REPO}...")
    print(f"   (Only loading text_encoder, VAE, scheduler from base repo)")
    pipe = WanImageToVideoPipeline.from_pretrained(
        BASE_MODEL_REPO,
        transformer=transformer_high_noise,
        transformer_2=transformer_low_noise,
        torch_dtype=torch.bfloat16,
    )

    # Replace scheduler with UniPCMultistepScheduler for the dual-transformer setup
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=FLOW_SHIFT
    )
    print(f"   ✓ Scheduler set to UniPCMultistepScheduler (flow_shift={FLOW_SHIFT})")

    # ================================================================
    # 5. Apply optimizations
    # ================================================================
    print("   ⚡ Applying speed optimizations...")

    pipe.enable_model_cpu_offload()
    print("   ✓ Model CPU offload enabled")

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

    load_time = time.time() - load_start_time
    print(f"✓ Pipeline loaded and optimized ({format_time(load_time)})")

    # ================================================================
    # 6. Load post-processing models
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
                    print(f"   ⏱️  RIFE loading: {format_time(time.time() - rife_load_start)}")
                else:
                    print(f"   ⚠️  RIFE model failed to load")

        if ENABLE_UPSCALING and realesrgan_path:
            print(f"\n🔍 Loading Real-ESRGAN model for upscaling...")
            upscale_load_start = time.time()
            try:
                upscale_model = load_sd_upscale(realesrgan_path, device)
                print(f"   ✓ Real-ESRGAN model loaded ({UPSCALE_FACTOR}x)")
                print(f"   ⏱️  Real-ESRGAN loading: {format_time(time.time() - upscale_load_start)}")
            except Exception as e:
                print(f"   ⚠️  Real-ESRGAN failed to load: {str(e)[:100]}")
                upscale_model = None

    # ================================================================
    # 7. Load and prepare input image
    # ================================================================
    print(f"\n🖼️ Loading input image from {INPUT_IMAGE_PATH}...")
    image_start_time = time.time()
    if not os.path.exists(INPUT_IMAGE_PATH):
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE_PATH}")
    image = load_image(INPUT_IMAGE_PATH)
    image_time = time.time() - image_start_time
    print(f"✓ Original image size: {image.width}x{image.height}")

    # Calculate dimensions
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]

    if RESIZE_IMAGE:
        max_area = 480 * 832
        aspect_ratio = image.height / image.width
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        print(f"📐 Resizing to: {width}x{height} (mod_value: {mod_value})")
        image = image.resize((width, height))
    else:
        original_width = image.width
        original_height = image.height
        width = (original_width // mod_value) * mod_value
        height = (original_height // mod_value) * mod_value
        if width != original_width or height != original_height:
            print(f"📐 Rounding to: {width}x{height} (mod_value: {mod_value})")
            image = image.resize((width, height))
        else:
            print(f"📐 Using original dimensions: {width}x{height}")

    # ================================================================
    # 8. Generate video
    # ================================================================
    print(f"\n🎬 Generating video...")
    print(f"   Prompt: {PROMPT[:100]}...")
    print(f"   Frames: {NUM_FRAMES}, Steps: {NUM_INFERENCE_STEPS}, "
          f"Guidance: {GUIDANCE_SCALE}, Seed: {SEED}")
    print(f"   Resolution: {width}x{height}")
    print(f"   Quantization: GGUF Q6_K (dual-transformer)")
    print(f"   Scheduler: UniPCMultistepScheduler (flow_shift={FLOW_SHIFT})")
    print(f"   Post-processing: RIFE={'✓' if rife_model else '✗'}, "
          f"Real-ESRGAN={'✓' if upscale_model else '✗'}")

    generator = torch.Generator(device=device).manual_seed(SEED)

    print("   🚀 Starting inference...")
    generation_start_time = time.time()

    with torch.inference_mode():
        output = pipe(
            image=image,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            height=height,
            width=width,
            num_frames=NUM_FRAMES,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=generator,
        ).frames[0]

    generation_time = time.time() - generation_start_time
    print(f"✓ Video generation completed ({format_time(generation_time)})")

    # ================================================================
    # 9. Apply post-processing
    # ================================================================
    postprocess_time = 0.0
    final_fps = FPS
    final_output = output
    final_height = height
    final_width = width

    if POSTPROCESSING_AVAILABLE and (ENABLE_RIFE or ENABLE_UPSCALING):
        print(f"\n🎨 Applying post-processing...")
        postprocess_start = time.time()

        if POSTPROCESS_INTERPOLATE_FIRST:
            print(f"   📋 Order: Interpolate → Upscale")

            if ENABLE_RIFE and rife_model:
                target_fps_str = (
                    f"to {RIFE_TARGET_FPS}fps" if RIFE_TARGET_FPS else "doubling frame rate"
                )
                print(f"   🎬 Interpolating frames with RIFE ({target_fps_str})...")
                try:
                    final_output, actual_fps = apply_rife_interpolation(
                        rife_model, final_output, device,
                        original_fps=FPS, target_fps=RIFE_TARGET_FPS,
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
                    final_height = height * UPSCALE_FACTOR
                    final_width = width * UPSCALE_FACTOR
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
                    final_height = height * UPSCALE_FACTOR
                    final_width = width * UPSCALE_FACTOR
                    print(f"   ✓ Upscaling completed ({final_width}x{final_height})")
                except Exception as e:
                    print(f"   ⚠️  Upscaling failed: {str(e)[:200]}")

            if ENABLE_RIFE and rife_model:
                target_fps_str = (
                    f"to {RIFE_TARGET_FPS}fps" if RIFE_TARGET_FPS else "doubling frame rate"
                )
                print(f"   🎬 Interpolating frames with RIFE ({target_fps_str})...")
                try:
                    final_output, actual_fps = apply_rife_interpolation(
                        rife_model, final_output, device,
                        original_fps=FPS, target_fps=RIFE_TARGET_FPS,
                    )
                    final_fps = actual_fps
                    print(f"   ✓ Frame interpolation completed (FPS: {final_fps})")
                except Exception as e:
                    print(f"   ⚠️  Frame interpolation failed: {str(e)[:200]}")

        postprocess_time = time.time() - postprocess_start
        print(f"   ⏱️  Post-processing: {format_time(postprocess_time)}")

    # ================================================================
    # 10. Export video
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

    # Print timing summary
    print("\n" + "-" * 60)
    print("⏱️  Timing Summary:")
    print("-" * 60)
    if download_time > 0:
        print(f"   Model download:    {format_time(download_time)}")
    print(f"   Pipeline loading:  {format_time(load_time)}")
    print(f"   Image loading:     {format_time(image_time)}")
    print(f"   Video generation:  {format_time(generation_time)}")
    if postprocess_time > 0:
        print(f"   Post-processing:   {format_time(postprocess_time)}")
    print(f"   Video export:      {format_time(export_time)}")
    print(f"   {'─'*50}")
    print(f"   Total time:        {format_time(total_time)}")
    print("-" * 60)

    return OUTPUT_VIDEO_PATH, {
        "download_time": download_time,
        "load_time": load_time,
        "image_time": image_time,
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
