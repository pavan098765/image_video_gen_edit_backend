#!/usr/bin/env python3
"""
Wan2.2 I2V Long Video Inference V2 - SVI-style Autoregressive Clip Chaining
============================================================================

Improvements over v1 (aligned with official SVI inference_svi_2.0_pro.py):
  - NUM_OVERLAP_FRAME=5: Drop first 5 frames of each continuation clip (repo uses 4-5).
    The first few frames are "warm-up" and least consistent; hiding them reduces visible transitions.
  - NOISE_AUG_STRENGTH=0.0: Repo does NOT add noise to handoff. Without latent bridging,
    pixel-space noise may introduce extra inconsistency at boundaries.
  - NUM_MOTION_FRAME: Temporal averaging of last N frames for handoff (repo uses multi-frame
    conditioning; we approximate by blending since diffusers expects single image).
  - Optional cross-fade blend at stitch points to soften visible cuts.

SVI LoRAs are PRE-MERGED into the base model weights. No runtime LoRA loading needed.

Inference count: Same as v1 and repo — N separate inference calls for N clips.
"""

import os
import sys
import random
import time
import torch
import traceback
import numpy as np
import diffusers
from PIL import Image
from huggingface_hub import snapshot_download
from diffusers.utils import export_to_video, load_image

# ---------------------------------------------------------------------------
# SDNQ setup
# ---------------------------------------------------------------------------
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
SDNQ_SRC_DIR = os.path.join(WORKSPACE_DIR, "sdnq", "src")
if os.path.exists(SDNQ_SRC_DIR) and SDNQ_SRC_DIR not in sys.path:
    sys.path.insert(0, SDNQ_SRC_DIR)
    print(f"Added local SDNQ source: {SDNQ_SRC_DIR}")

try:
    from sdnq import SDNQConfig
    print("✓ SDNQ registered")
except ImportError:
    print("⚠ SDNQ not found, continuing without it")

try:
    from video_postprocessing import (
        download_rife_model, download_realesrgan_model,
        load_rife_model, load_sd_upscale,
        apply_rife_interpolation, apply_realesrgan_upscaling,
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
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")

# SVI LoRAs are already merged into this model - no runtime LoRA loading needed
MODEL_LOCAL_DIR = os.path.join(
    MODELS_DIR,
    "wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_Multiscene_SVI_sdnq_fp8_svd_r64"
)
MODEL_HF_REPO = "rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_Multiscene_SVI_sdnq_fp8_svd_r64"

# --- Input ---
# Single image (str) or batch list. Each image = separate request (NUM_CLIPS clips per image).
INPUT_IMAGE_PATHS = [
    os.path.join(WORKSPACE_DIR, "an1.jpeg"),
    os.path.join(WORKSPACE_DIR, "d9.jpeg"),
    os.path.join(WORKSPACE_DIR, "d10.jpeg"),
    os.path.join(WORKSPACE_DIR, "n1.jpeg"),
    os.path.join(WORKSPACE_DIR, "h1.jpeg"),
    os.path.join(WORKSPACE_DIR, "h3.jpeg"),
    os.path.join(WORKSPACE_DIR, "h4.jpeg"),
    os.path.join(WORKSPACE_DIR, "h6.jpeg"),
    os.path.join(WORKSPACE_DIR, "h7.jpeg"),
    os.path.join(WORKSPACE_DIR, "p1.jpeg"),
    os.path.join(WORKSPACE_DIR, "s1.jpeg"),
    # os.path.join(WORKSPACE_DIR, "img2.jpeg"),
]
# Backward compat: use single path as str for one image
# INPUT_IMAGE_PATHS = os.path.join(WORKSPACE_DIR, "an1.jpeg")

# --- Long video settings ---
NUM_CLIPS      = 4    # 4 clips × 5s = ~20s total (after overlap removal)
NUM_FRAMES     = 81   # max for Wan2.2 = 5s at 16fps
FPS            = 16
CONTEXT_FRAMES = 5    # tail frames stored per clip (informational; used for future VACE support)

# V2: Repo does NOT add noise to handoff. Without latent bridging, noise may hurt transitions.
NOISE_AUG_STRENGTH = 0.0

# V2: Repo uses num_overlap_frame=4-5. Drop first N frames of each continuation clip.
# Those frames are warm-up and least consistent; hiding them reduces visible transitions.
NUM_OVERLAP_FRAME = 5

# V2: Repo uses num_motion_frame=4 (last 4 frames as input). We approximate by blending
# last N frames into one handoff image since diffusers expects single image.
NUM_MOTION_FRAME = 2  # 1 = last frame only (v1 behavior), 2+ = weighted average

# V2: Optional cross-fade at stitch points (experimental). 0 = disabled.
BLEND_AT_JOIN_FRAMES = 3

# --- Per-clip prompts ---
# Tip: keep clip 1 close to your original prompt, evolve gently from clip 2 onward.
# The model will naturally continue motion; these prompts guide the direction.
CLIP_PROMPTS = [
    "Make the image come alive, cinematic motion and smooth animation.",
    "Continue the scene with fluid natural motion, maintaining consistent lighting and character appearance.",
    "The subject continues moving naturally, camera holds steady, seamless continuation of the previous action.",
    "Final continuation, natural organic motion, temporally consistent with previous frames.",
]

NEGATIVE_PROMPT = (
    "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, "
    "JPEG压缩残留, 丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景, 三条腿, 背景人很多, 倒着走"
)

# --- Inference settings ---
# workflow uses steps=6, cfg=1.5 with LightX2V baked in
# your original NSFW model used steps=8, cfg=1.0 — try both, 6/1.5 is recommended with SVI merged
NUM_INFERENCE_STEPS = 6
GUIDANCE_SCALE      = 1.5
SEED                = random.randint(0, 10000)

# --- Post-processing ---
ENABLE_RIFE      = True
RIFE_TARGET_FPS  = 30
ENABLE_UPSCALING = True
UPSCALE_FACTOR   = 2

# --- Output ---
OUTPUT_VIDEO_BASENAME = f"long_video_svi_v2_{NUM_CLIPS}clips"  # per-image: {basename}_{base}.mp4
SAVE_INDIVIDUAL_CLIPS = True  # save each raw clip for debugging/review


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_time(s):
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s // 60)}m {s % 60:.1f}s"


def tensor_to_pil(t) -> Image.Image:
    """float32 array [H, W, 3] in [0,1] → PIL Image"""
    return Image.fromarray((np.clip(t, 0, 1) * 255).astype(np.uint8))


def apply_noise_aug(frame_pil: Image.Image, strength: float) -> Image.Image:
    """
    Add tiny Gaussian noise to the handoff frame.
    SVI trains the model to recover from slightly imperfect conditioning,
    so this small perturbation actually improves temporal consistency.
    """
    if strength <= 0.0:
        return frame_pil
    arr = np.array(frame_pil).astype(np.float32) / 255.0
    noise = np.random.randn(*arr.shape).astype(np.float32) * strength
    return Image.fromarray((np.clip(arr + noise, 0, 1) * 255).astype(np.uint8))


def resize_for_wan(image: Image.Image, max_area: int = 480 * 832, mod: int = 16):
    """Resize preserving aspect ratio, quantized to mod, within max_area."""
    ar = image.height / image.width
    height = max(int(np.sqrt(max_area * ar)) // mod * mod, mod)
    width  = max(int(np.sqrt(max_area / ar)) // mod * mod, mod)
    return image.resize((width, height), Image.LANCZOS), width, height


def frames_to_handoff_image(frames: list, width: int, height: int, num_motion_frame: int) -> Image.Image:
    """
    V2: Repo uses num_motion_frame (last N frames). We approximate by weighted average
    since diffusers expects single image. More weight on last frame.
    """
    if num_motion_frame <= 1 or len(frames) < 2:
        return tensor_to_pil(frames[-1]).resize((width, height), Image.LANCZOS)
    n = min(num_motion_frame, len(frames))
    tail = frames[-n:]
    # Weights: last frame gets most weight (linear decay)
    weights = np.array([i + 1 for i in range(n)], dtype=np.float32)
    weights /= weights.sum()
    arr = np.zeros_like(np.asarray(tail[0], np.float32))
    for i, f in enumerate(tail):
        a = np.asarray(f, np.float32)
        if a.max() > 1.0:
            a = a / 255.0
        arr += weights[i] * np.clip(a, 0, 1)
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((width, height), Image.LANCZOS)


def blend_at_join(frames: list, blend_frames: int) -> list:
    """
    V2: Cross-fade the last blend_frames of prev clip with first blend_frames of next.
    frames = [last blend_frames of prev] + [first blend_frames of next]. Returns blend_frames.
    """
    if blend_frames <= 0 or len(frames) < 2 * blend_frames:
        return list(frames[:blend_frames]) if blend_frames > 0 and frames else []
    before = np.asarray(frames[:blend_frames], np.float32)
    after = np.asarray(frames[blend_frames : 2 * blend_frames], np.float32)
    if before.max() > 1.0:
        before = before / 255.0
    if after.max() > 1.0:
        after = after / 255.0
    blended = []
    for i in range(blend_frames):
        alpha = (i + 1) / (blend_frames + 1)
        b = np.clip(before[i] * (1 - alpha) + after[i] * alpha, 0, 1)
        blended.append(b)
    return blended


def stitch_clips(all_clips: list, overlap: int = 1, blend_frames: int = 0) -> list:
    """
    Concatenate clips, dropping the first `overlap` frames of each clip
    after the first. Frame[0] of clip N ≈ frame[-1] of clip N-1 (same
    start_image was used), so dropping it avoids a freeze/duplicate frame
    at each join point.
    V2: Optional blend_frames for cross-fade at each join.
    """
    if not all_clips:
        return []
    result = list(all_clips[0])
    for clip in all_clips[1:]:
        tail_len = overlap
        if blend_frames > 0 and len(result) >= blend_frames and len(clip) > overlap + blend_frames:
            join_region = list(result[-blend_frames:]) + list(clip[overlap:overlap + blend_frames])
            blended = blend_at_join(join_region, blend_frames)
            result = result[:-blend_frames] + blended[:blend_frames]
            result.extend(list(clip[overlap + blend_frames:]))
        else:
            result.extend(list(clip[overlap:]))
    return result


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def ensure_model_available():
    if os.path.exists(MODEL_LOCAL_DIR):
        markers = ["model_index.json", "config.json", "quantization_config.json"]
        if any(os.path.exists(os.path.join(MODEL_LOCAL_DIR, m)) for m in markers):
            print(f"✓ Model found: {MODEL_LOCAL_DIR}")
            return
        print("⚠ Model directory exists but appears incomplete, re-downloading...")
    print(f"\n⬇ Downloading model: {MODEL_HF_REPO}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    snapshot_download(MODEL_HF_REPO, local_dir=MODEL_LOCAL_DIR, local_dir_use_symlinks=False)
    print(f"✓ Downloaded to {MODEL_LOCAL_DIR}")


def load_pipeline():
    print(f"\nLoading pipeline from {MODEL_LOCAL_DIR}...")
    t0 = time.time()

    pipe = diffusers.WanImageToVideoPipeline.from_pretrained(
        MODEL_LOCAL_DIR,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Fix SDNQ tied weights: encoder.embed_tokens.weight is missing in checkpoint
    # because SDNQ's loader handles it, but diffusers' from_pretrained does not.
    if hasattr(pipe, "text_encoder"):
        te = pipe.text_encoder
        if (hasattr(te, "shared") and hasattr(te, "encoder")
                and hasattr(te.encoder, "embed_tokens")):
            if te.shared.weight.data_ptr() != te.encoder.embed_tokens.weight.data_ptr():
                te.encoder.embed_tokens.weight = te.shared.weight
                print("   ✓ Fixed: Re-tied shared.weight -> encoder.embed_tokens.weight")
            else:
                print("   ✓ Text encoder weights already tied correctly")
            if hasattr(te, "tie_weights"):
                te.tie_weights()
    else:
        print("   ⚠ Text encoder not found in pipeline")

    pipe.enable_model_cpu_offload()

    try:
        import flash_attn
        print("   ✓ Flash Attention available")
    except ImportError:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("   ✓ XFormers enabled")
        except Exception:
            pass

    load_time = time.time() - t0
    print(f"Pipeline ready in {format_time(load_time)}")
    return pipe, load_time


# ---------------------------------------------------------------------------
# Single clip generation
# ---------------------------------------------------------------------------
def generate_clip(pipe, start_image: Image.Image, prompt: str,
                  width: int, height: int, seed: int, clip_idx: int):
    """
    Generate one 81-frame clip conditioned on start_image.

    For clip 0: start_image = original input photo (clean, no noise)
    For clip N: start_image = last frame of clip N-1 + tiny noise aug
                (SVI error-recycling: model handles slightly imperfect
                 conditioning and continues naturally)
    """
    if clip_idx > 0 and NOISE_AUG_STRENGTH > 0:
        start_image = apply_noise_aug(start_image, NOISE_AUG_STRENGTH)

    generator = torch.Generator(device="cpu").manual_seed(seed)

    print(f"   seed={seed}  size={width}x{height}  steps={NUM_INFERENCE_STEPS}  cfg={GUIDANCE_SCALE}", flush=True)
    if torch.cuda.is_available():
        print(f"   VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB / "
              f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB", flush=True)

    def _cb(pipe, step_index, timestep, callback_kwargs):
        print(f"   Step {step_index+1}/{NUM_INFERENCE_STEPS} "
              f"({(step_index+1)/NUM_INFERENCE_STEPS*100:.0f}%)", flush=True)
        return callback_kwargs

    kwargs = dict(
        image=start_image,
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        height=height,
        width=width,
        num_frames=NUM_FRAMES,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
    )

    t0 = time.time()
    with torch.inference_mode():
        try:
            output = pipe(**kwargs, callback_on_step_end=_cb).frames[0]
        except TypeError:
            # Some pipeline versions don't support callback_on_step_end
            output = pipe(**kwargs).frames[0]

    gen_time = time.time() - t0
    print(f"\n   Generated in {format_time(gen_time)}", flush=True)
    return output, gen_time


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_long_video():
    t_start = time.time()
    print("\n" + "=" * 60)
    print("Wan2.2 Long Video V2 — SVI Autoregressive Clip Chaining")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")

    ensure_model_available()

    # Download post-processing models before loading the heavy pipeline
    rife_path = realesrgan_path = None
    if POSTPROCESSING_AVAILABLE:
        if ENABLE_RIFE:
            rife_path = download_rife_model(MODELS_DIR)
        if ENABLE_UPSCALING:
            realesrgan_path = download_realesrgan_model(MODELS_DIR, UPSCALE_FACTOR)

    pipe, load_time = load_pipeline()

    rife_model = upscale_model = None
    if POSTPROCESSING_AVAILABLE:
        if ENABLE_RIFE and rife_path and check_rife_code_available():
            rife_model = load_rife_model(rife_path)
        if ENABLE_UPSCALING and realesrgan_path:
            try:
                upscale_model = load_sd_upscale(realesrgan_path, device)
            except Exception as e:
                print(f"Real-ESRGAN load failed: {e}")

    # Normalize input: str -> single-item list, ensure full paths
    paths = INPUT_IMAGE_PATHS if isinstance(INPUT_IMAGE_PATHS, (list, tuple)) else [INPUT_IMAGE_PATHS]
    image_paths = [os.path.join(WORKSPACE_DIR, p) if not os.path.isabs(p) else p for p in paths]
    image_paths = [p for p in image_paths if os.path.exists(p)]
    if not image_paths:
        raise FileNotFoundError(f"No input images found. Checked: {paths}")

    output_paths = []
    for img_idx, input_path in enumerate(image_paths):
        print("\n" + "#" * 60)
        print(f"BATCH {img_idx + 1}/{len(image_paths)}: {os.path.basename(input_path)}")
        print("#" * 60)

        # Per-image output path
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(WORKSPACE_DIR, f"{OUTPUT_VIDEO_BASENAME}_{base}.mp4")

        # Load and resize input image
        source_image = load_image(input_path)
        print(f"\nInput : {source_image.width}x{source_image.height}  →  ", end="")
        source_image, width, height = resize_for_wan(source_image)
        print(f"{width}x{height}")

        # Pad prompt list to cover all clips
        prompts = list(CLIP_PROMPTS)
        while len(prompts) < NUM_CLIPS:
            prompts.append(prompts[-1])

        total_output_frames = NUM_FRAMES + (NUM_CLIPS - 1) * (NUM_FRAMES - NUM_OVERLAP_FRAME)
        print(f"\nPlan  : {NUM_CLIPS} clips × {NUM_FRAMES} frames, overlap={NUM_OVERLAP_FRAME}")
        print(f"Output: ~{total_output_frames} frames  ({total_output_frames / FPS:.1f}s at {FPS}fps)")
        print(f"SVI   : pre-merged into model weights ✓\n")

        # Autoregressive clip generation
        all_clips     = []
        all_gen_times = []
        current_start = source_image

        for clip_idx in range(NUM_CLIPS):
            print("\n" + "=" * 60)
            print(f"CLIP {clip_idx + 1} / {NUM_CLIPS}")
            print(f"Prompt: {prompts[clip_idx][:90]}")
            print("=" * 60)

            frames, gen_time = generate_clip(
            pipe=pipe,
            start_image=current_start,
            prompt=prompts[clip_idx],
            width=width,
            height=height,
            seed=SEED + clip_idx,
            clip_idx=clip_idx,
            )
            all_clips.append(frames)
            all_gen_times.append(gen_time)

            # Save individual clip for review
            if SAVE_INDIVIDUAL_CLIPS:
                clip_path = output_path.replace(".mp4", f"_clip{clip_idx + 1:02d}.mp4")
                clip_arr = np.clip(np.nan_to_num(np.asarray(frames, np.float32), 0, 1, 0), 0, 1)
                export_to_video(clip_arr, clip_path, fps=FPS)
                print(f"   Saved: {os.path.basename(clip_path)}")

            # --- Handoff: V2 uses temporal averaging (num_motion_frame) for smoother continuity ---
            current_start = frames_to_handoff_image(frames, width, height, NUM_MOTION_FRAME)
            if clip_idx < NUM_CLIPS - 1:
                print(f"   ✓ Handoff image (last {NUM_MOTION_FRAME} frames avg) → start_image for clip {clip_idx + 2}")

        # Stitch (V2: overlap=NUM_OVERLAP_FRAME, optional cross-fade)
        print("\n" + "=" * 60)
        print(f"Stitching clips (overlap={NUM_OVERLAP_FRAME}, blend={BLEND_AT_JOIN_FRAMES})...")
        final_frames = stitch_clips(all_clips, overlap=NUM_OVERLAP_FRAME, blend_frames=BLEND_AT_JOIN_FRAMES)
        print(f"Stitched: {len(final_frames)} frames ({len(final_frames) / FPS:.1f}s)")

        # Post-processing
        final_fps = FPS
        final_h, final_w = height, width

        if POSTPROCESSING_AVAILABLE and (ENABLE_UPSCALING or ENABLE_RIFE):
            print("\nPost-processing...")
            if ENABLE_UPSCALING and upscale_model:
                final_frames = apply_realesrgan_upscaling(upscale_model, final_frames, device)
                final_h, final_w = height * UPSCALE_FACTOR, width * UPSCALE_FACTOR
                print(f"   Upscaled  : {final_w}x{final_h}")
            if ENABLE_RIFE and rife_model:
                final_frames, final_fps = apply_rife_interpolation(
                    rife_model, final_frames, device,
                    original_fps=FPS, target_fps=RIFE_TARGET_FPS,
                )
                print(f"   Interpolated: {final_fps}fps")

        # Export
        final_arr = np.clip(
            np.nan_to_num(np.asarray(final_frames, np.float32), 0, 1, 0), 0, 1
        )
        export_to_video(final_arr, output_path, fps=final_fps)
        output_paths.append(output_path)
        print(f"   → {output_path}")

    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"   Processed  : {len(output_paths)} image(s)")
    print(f"   Load time  : {format_time(load_time)}")
    print(f"   Total time : {format_time(total_time)}")
    for p in output_paths:
        print(f"   → {p}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_long_video()
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        exit(1)
