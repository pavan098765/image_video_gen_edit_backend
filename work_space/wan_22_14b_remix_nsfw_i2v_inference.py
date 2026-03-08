#!/usr/bin/env python3
"""
Wan2.2 I2V 480p Lightning NSFW inference script.

Uses SDNQ quantized model with LoRA baked into weights.
Downloads from HuggingFace if not found locally.
Retains RIFE frame interpolation and Real-ESRGAN upscaling.
"""

import os
import sys
import random
import time
import torch
import traceback
import numpy as np
import diffusers
from huggingface_hub import snapshot_download
from diffusers.utils import export_to_video, load_image

# Add local SDNQ source to path (if not installed via pip)
WORKSPACE_DIR_EARLY = os.path.dirname(os.path.abspath(__file__))
SDNQ_SRC_DIR = os.path.join(WORKSPACE_DIR_EARLY, "sdnq", "src")
if os.path.exists(SDNQ_SRC_DIR) and SDNQ_SRC_DIR not in sys.path:
    sys.path.insert(0, SDNQ_SRC_DIR)
    print(f"Added local SDNQ source: {SDNQ_SRC_DIR}")

from sdnq import SDNQConfig  # Registers SDNQ quantizer with diffusers/transformers

print("✓ SDNQ mode: SDNQ quantizer registered with diffusers")

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
    print(f"Video post-processing not available: {e}")
    POSTPROCESSING_AVAILABLE = False

torch.set_float32_matmul_precision("medium")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")

# MODEL_LOCAL_DIR = os.path.join(MODELS_DIR, "wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_DR34ML4Y_sdnq_fp8_svd_r64")
MODEL_LOCAL_DIR = os.path.join(MODELS_DIR, "wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_Multiscene_softcore_sdnq_fp8_svd_r64")
# MODEL_LOCAL_DIR = os.path.join(MODELS_DIR, "wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_Pornmaster_sdnq_fp8_svd_r64")
# MODEL_LOCAL_DIR = os.path.join(MODELS_DIR, "wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_GENERAL_NSFW_sdnq_fp8_svd_r64")
# MODEL_LOCAL_DIR = os.path.join(MODELS_DIR, "wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_NSFW_POSING_sdnq_fp8_svd_r64")
# MODEL_LOCAL_DIR = os.path.join(MODELS_DIR, "wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_FM_BDY_ENHCR_sdnq_fp8_svd_r64")


# MODEL_HF_REPO = "rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_DR34ML4Y_sdnq_fp8_svd_r64"
MODEL_HF_REPO = "rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_Multiscene_softcore_sdnq_fp8_svd_r64"
# MODEL_HF_REPO = "rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_Pornmaster_sdnq_fp8_svd_r64"
# MODEL_HF_REPO = "rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_GENERAL_NSFW_sdnq_fp8_svd_r64"
# MODEL_HF_REPO = "rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_NSFW_POSING_sdnq_fp8_svd_r64"
# MODEL_HF_REPO = "rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_FM_BDY_ENHCR_sdnq_fp8_svd_r64"


INPUT_IMAGE_NAMES = [
    "an1.jpeg",
    "d1.jpeg",
    # "d2.jpeg",
    # "d3.jpeg",
    "d4.jpeg",
    # "d5.jpeg",
    # "d6.jpeg",
    # "d7.jpeg",
    "d8.jpeg",
    "d9.jpeg",
    "d10.jpeg",
    "h1.jpeg",
    # "h2.jpeg",
    "h3.jpeg",
    "h4.jpeg",
    #"h5.jpeg",
    "h6.jpeg",
    "h7.jpeg",
    #"h8.jpeg",
    "n1.jpeg",
    "p1.jpeg",
    "s1.jpeg",
]

model_keyword = ""
if MODEL_LOCAL_DIR == os.path.join(
    MODELS_DIR, "wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_DR34ML4Y_sdnq_fp8_svd_r64"
):
    model_keyword = "DR34ML4Y"
elif MODEL_LOCAL_DIR == os.path.join(
    MODELS_DIR,
    "wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_Multiscene_softcore_sdnq_fp8_svd_r64",
):
    model_keyword = "Multiscene_softcore"
elif MODEL_LOCAL_DIR == os.path.join(
    MODELS_DIR,
    "wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_Pornmaster_sdnq_fp8_svd_r64",
):
    model_keyword = "Pmaster"
else:
    model_keyword = ""


_nsfw_pr = """
Starting from the original image:  
Over the first 1.5-2 seconds her outfit gently dissolves and completely vanishes, naturally uncovering her nude body.  
Full nudity is revealed — bare breasts and exposed vagina / pubic area.  
Maintain identical:  
• camera angle  
• framing & composition  
• perspective  
• facial features & expression  
• skin tone & skin texture  
• lighting  
• hairstyle  
• background & environment  

Only the clothing is removed. No other changes to pose, body shape, face, or scene.
"""
PROMPT = "Make the image come alive, cinematic motion and smooth animation."

PROMPT = _nsfw_pr

NEGATIVE_PROMPT = (
    "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, "
    "JPEG压缩残留, 丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景, 三条腿, 背景人很多, 倒着走"
)

NUM_FRAMES = 257 # 81
GUIDANCE_SCALE = 1.0
NUM_INFERENCE_STEPS = 8
FPS = 16
SEED = random.randint(0, 1000)
RESIZE_IMAGE = True
ENABLE_RIFE = True
RIFE_TARGET_FPS = 30
ENABLE_UPSCALING = True
UPSCALE_FACTOR = 2
POSTPROCESS_INTERPOLATE_FIRST = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {seconds % 60:.2f}s"
    else:
        h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), seconds % 60
        return f"{h}h {m}m {s:.2f}s"


def ensure_model_available():
    """Check if model exists locally; if not, download from HuggingFace."""
    if os.path.exists(MODEL_LOCAL_DIR):
        model_index = os.path.join(MODEL_LOCAL_DIR, "model_index.json")
        config_file = os.path.join(MODEL_LOCAL_DIR, "config.json")
        quant_config = os.path.join(MODEL_LOCAL_DIR, "quantization_config.json")
        if os.path.exists(model_index) or os.path.exists(config_file) or os.path.exists(quant_config):
            print(f"✓ Model found locally at {MODEL_LOCAL_DIR}")
            return
        print("⚠ Model directory exists but appears incomplete")

    print(f"\n⬇ Downloading model from HuggingFace: {MODEL_HF_REPO}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    try:
        snapshot_download(
            MODEL_HF_REPO, local_dir=MODEL_LOCAL_DIR, local_dir_use_symlinks=False
        )
        print(f"✓ Model downloaded to {MODEL_LOCAL_DIR}")
    except Exception as e:
        raise FileNotFoundError(
            f"Model not found at {MODEL_LOCAL_DIR} and download failed: {e}"
        ) from e


def download_postprocessing_models():
    """Download RIFE and Real-ESRGAN models if post-processing is enabled."""
    if not POSTPROCESSING_AVAILABLE:
        return None, None, 0.0
    t, rife_path, realesrgan_path = 0.0, None, None
    if ENABLE_RIFE:
        s = time.time()
        rife_path = download_rife_model(MODELS_DIR)
        t += time.time() - s
    if ENABLE_UPSCALING:
        s = time.time()
        realesrgan_path = download_realesrgan_model(MODELS_DIR, UPSCALE_FACTOR)
        t += time.time() - s
    return rife_path, realesrgan_path, t


# ---------------------------------------------------------------------------
# Single image inference
# ---------------------------------------------------------------------------
def process_single_image(
    pipe,
    device,
    generator_device,
    rife_model,
    upscale_model,
    input_image_path,
    output_video_path,
    seed=None,
):
    if seed is None:
        seed = SEED
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    image = load_image(input_image_path)
    print(f"Original size: {image.width}x{image.height}")

    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]

    if RESIZE_IMAGE:
        max_area = 480 * 832
        ar = image.height / image.width
        height = int(np.sqrt(max_area * ar)) // mod_value * mod_value
        width = int(np.sqrt(max_area / ar)) // mod_value * mod_value
        print(f"Resizing to: {width}x{height}")
        image = image.resize((width, height))
    else:
        width = (image.width // mod_value) * mod_value
        height = (image.height // mod_value) * mod_value
        if width != image.width or height != image.height:
            image = image.resize((width, height))
        print(f"Dimensions: {width}x{height}")

    print(f"\nGenerating (seed={seed}, steps={NUM_INFERENCE_STEPS})...")
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    t0 = time.time()

    def _cb(pipe, step_index, timestep, callback_kwargs):
        print(
            f"   Step {step_index+1}/{NUM_INFERENCE_STEPS} "
            f"({(step_index+1)/NUM_INFERENCE_STEPS*100:.0f}%)",
            end="\r",
            flush=True,
        )
        return callback_kwargs

    kwargs = dict(
        image=image,
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=height,
        width=width,
        num_frames=NUM_FRAMES,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
        callback_on_step_end=_cb,
    )
    print("   About to run pipeline...", flush=True)
    print(f"   VRAM before: {torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)
    torch.cuda.synchronize()

    with torch.inference_mode():
        try:
            output = pipe(**kwargs).frames[0]
        except TypeError as e:
            if "callback_on_step_end" in str(e):
                kwargs.pop("callback_on_step_end")
                output = pipe(**kwargs).frames[0]
            else:
                raise
    print()

    gen_time = time.time() - t0
    print(f"Generated in {format_time(gen_time)}")

    pp_time = 0.0
    final_fps, final_output, final_h, final_w = FPS, output, height, width

    if POSTPROCESSING_AVAILABLE and (ENABLE_RIFE or ENABLE_UPSCALING):
        pp_start = time.time()
        if not POSTPROCESS_INTERPOLATE_FIRST and ENABLE_UPSCALING and upscale_model:
            final_output = apply_realesrgan_upscaling(
                upscale_model, final_output, device
            )
            final_h, final_w = height * UPSCALE_FACTOR, width * UPSCALE_FACTOR
            print(f"   Upscaled to {final_w}x{final_h}")
        if ENABLE_RIFE and rife_model:
            final_output, final_fps = apply_rife_interpolation(
                rife_model,
                final_output,
                device,
                original_fps=FPS,
                target_fps=RIFE_TARGET_FPS,
            )
            print(f"   Interpolated to {final_fps}fps")
        if POSTPROCESS_INTERPOLATE_FIRST and ENABLE_UPSCALING and upscale_model:
            final_output = apply_realesrgan_upscaling(
                upscale_model, final_output, device
            )
            final_h, final_w = height * UPSCALE_FACTOR, width * UPSCALE_FACTOR
            print(f"   Upscaled to {final_w}x{final_h}")
        pp_time = time.time() - pp_start

    final_output = np.clip(
        np.nan_to_num(np.asarray(final_output, np.float32), 0, 1, 0), 0, 1
    )
    export_to_video(final_output, output_video_path, fps=final_fps)
    print(
        f"   Exported: {os.path.basename(output_video_path)} ({final_w}x{final_h}, {final_fps}fps)"
    )
    return {"generation_time": gen_time, "postprocess_time": pp_time}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_inference():
    t_start = time.time()
    print("\n" + "=" * 60)
    print("Wan2.2 I2V 480p Lightning NSFW - Inference")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"   GPU:  {torch.cuda.get_device_name(0)}")
        print(
            f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f} GB"
        )

    rife_path, realesrgan_path, _ = download_postprocessing_models()

    ensure_model_available()

    print(f"\nLoading pipeline from {MODEL_LOCAL_DIR}...")
    load_start = time.time()
    pipe = diffusers.WanImageToVideoPipeline.from_pretrained(
        MODEL_LOCAL_DIR,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Fix SDNQ tied weights: shared.weight -> encoder.embed_tokens.weight
    # SDNQ's load_sdnq_model handles this, but diffusers' from_pretrained does not. Re-tie explicitly.
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
                print("   ✓ Fixed: Re-tied shared.weight -> encoder.embed_tokens.weight")
            else:
                print("   ✓ Text encoder weights already tied correctly")
            if hasattr(te, "tie_weights"):
                te.tie_weights()
    else:
        print("   ⚠ Text encoder not found in pipeline")

    # print("\nEnabling sequential CPU offload...")
    # pipe.enable_sequential_cpu_offload()

    print("Enable model cpu offload...")
    pipe.enable_model_cpu_offload()

    generator_device = "cpu"

    try:
        import flash_attn

        print("   ✓ Flash Attention available")
    except ImportError:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("   ✓ XFormers enabled")
        except Exception:
            pass

    load_time = time.time() - load_start
    print(f"Pipeline ready in {format_time(load_time)}")

    rife_model = upscale_model = None
    if POSTPROCESSING_AVAILABLE:
        if ENABLE_RIFE and rife_path and check_rife_code_available():
            rife_model = load_rife_model(rife_path)
        if ENABLE_UPSCALING and realesrgan_path:
            try:
                upscale_model = load_sd_upscale(realesrgan_path, device)
            except Exception as e:
                print(f"   Real-ESRGAN load failed: {e}")

    if not INPUT_IMAGE_NAMES:
        raise ValueError("INPUT_IMAGE_NAMES is empty")

    image_paths = [os.path.join(WORKSPACE_DIR, n) for n in INPUT_IMAGE_NAMES]
    print(f"\nInference: {len(image_paths)} images")
    for i, p in enumerate(image_paths, 1):
        print(
            f"   {i:2d}. {os.path.basename(p)} [{'OK' if os.path.exists(p) else 'MISSING'}]"
        )

    output_paths, all_timings = [], []
    for idx, img_path in enumerate(image_paths, 1):
        print("\n" + "=" * 60)
        print(f"Image {idx}/{len(image_paths)}: {os.path.basename(img_path)}")
        print("=" * 60)
        base = os.path.splitext(os.path.basename(img_path))[0]

        if MODEL_LOCAL_DIR == os.path.join(
            MODELS_DIR,
            "wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_DR34ML4Y_sdnq_fp8_svd_r64",
        ):
            base = base + "_DR34ML4Y"
        elif MODEL_LOCAL_DIR == os.path.join(
            MODELS_DIR,
            "wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_Multiscene_softcore_sdnq_fp8_svd_r64",
        ):
            base = base + "_Multiscene_softcore"
        elif MODEL_LOCAL_DIR == os.path.join(
            MODELS_DIR,
            "wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_Pornmaster_sdnq_fp8_svd_r64",
        ):
            base = base + "_Pmaster"
        elif MODEL_LOCAL_DIR == os.path.join(
            MODELS_DIR,
            "wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_GENERAL_NSFW_sdnq_fp8_svd_r64",
        ):
            base = base + "_GENERAL_NSFW"
        elif MODEL_LOCAL_DIR == os.path.join(
            MODELS_DIR,
            "wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_NSFW_POSING_sdnq_fp8_svd_r64",
        ):
            base = base + "_NSFW_POSING"
        elif MODEL_LOCAL_DIR == os.path.join(
            MODELS_DIR,
            "wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_lora_FM_BDY_ENHCR_sdnq_fp8_svd_r64",
        ):
            base = base + "_FM_BDY_ENHCR"
        else:
            base = base + "_UNKNOWN"

        out_path = os.path.join(WORKSPACE_DIR, f"{base}_i2v_remix_nsfw.mp4")
        try:
            t = process_single_image(
                pipe,
                device,
                generator_device,
                rife_model,
                upscale_model,
                img_path,
                out_path,
                seed=SEED + idx - 1,
            )
            output_paths.append(out_path)
            all_timings.append(t)
            print(f"Done: {os.path.basename(img_path)}")
        except Exception as e:
            print(f"FAILED {os.path.basename(img_path)}: {e}")
            traceback.print_exc()

    total_time = time.time() - t_start
    print("\n" + "=" * 60 + "\nSummary\n" + "=" * 60)
    print(f"   Processed : {len(output_paths)}/{len(image_paths)}")
    print(f"   Load time : {format_time(load_time)}")
    if all_timings:
        print(
            f"   Avg gen   : {format_time(sum(t['generation_time'] for t in all_timings)/len(all_timings))}"
        )
        print(
            f"   Total gen : {format_time(sum(t['generation_time'] for t in all_timings))}"
        )
    print(f"   Total     : {format_time(total_time)}")
    for p in output_paths:
        print(f"   {os.path.basename(p)}")
    print("=" * 60)
    return output_paths, {
        "load_time": load_time,
        "total_time": total_time,
        "output_paths": output_paths,
    }


if __name__ == "__main__":
    try:
        output_paths, timings = run_inference()
        print(f"\nDone. Total: {format_time(timings['total_time'])}")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        exit(1)
