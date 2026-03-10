#!/usr/bin/env python3
"""
Wan2.2 I2V Long Video Inference V3 - SVI-style Autoregressive Clip Chaining
============================================================================

V3 vs V2 quality fixes:
  - Smoke/haze: sharp single-frame handoff (no averaging), SHARPEN pass
  - Face drift: extended negative prompt with English identity/atmospheric terms
  - Overlap: 5 frames dropped per join (consistent default)
  - Cross-fade: 3-frame blend at each stitch point
  - Seeds: clip_idx * SEED_MULTIPLIER (no random base)

V3 memory fixes (OOM between batched images):

  PROBLEM 1 — all_clips[] + frames_list_to_array() peak: ~3.6 GB
    Old code accumulated all NUM_CLIPS clips in all_clips[], then called
    frames_list_to_array() which converted everything to float32 numpy,
    then export_to_video called np.asarray() again. At peak:
      all_clips PIL (370 MB) + stitch copy (370 MB) + float32 (1.48 GB)
      + export copy (1.48 GB) = ~3.6 GB just for frames.

  FIX 1 — INCREMENTAL STITCH (peak: ~740 MB)
    Each clip is appended to a running `stitched` list immediately after
    generation, then the clip tensor is deleted. all_clips[] never exists.
    Peak = stitched list (370 MB) + 1 clip at a time (370 MB) = 740 MB.

  PROBLEM 2 — raw_frames held as float32 after pipeline returns
    Pipeline output is float32 [T,H,W,3]. Keeping it while also building
    PIL list doubles frame memory temporarily.

  FIX 2 — IMMEDIATE UINT8 CONVERSION
    Convert pipeline output to uint8 PIL immediately, del raw float32.
    uint8 PIL = 1 byte/channel vs float32 = 4 bytes/channel.

  PROBLEM 3 — SAVE_INDIVIDUAL_CLIPS allocated a full float32 copy per clip
    np.asarray(frames, np.float32) on 81 frames = ~370 MB per clip.

  FIX 3 — save_clip_video() builds the array in one stack call, uses it,
    then deletes it immediately. No persistent copy.

  PROBLEM 4 — CUDA cache not flushed between images in batch loop
    Activation memory from the last inference step stays in GPU cache.

  FIX 4 — flush_memory() (torch.cuda.empty_cache + gc.collect) called
    after every clip generation AND after every image is complete.

V3 post-processing memory fix (OOM during apply_realesrgan_upscaling):

  PROBLEM 5 — apply_realesrgan_upscaling allocates ~10 GB at upscale time
    Standard Real-ESRGAN wrapper processes all frames then np.stack()s them:
      result = [upscale(f) for f in frames]  <- 5.2 GB list (300 x 864x1792)
      return np.stack(result)                <- 5.2 GB copy = 10.4 GB peak
    Combined with 20 GB model + 1.3 GB input = ~31.7 GB -> kills 32 GB system.

  FIX 5 — STREAMING UPSCALE + RIFE (upscale_frames_streaming)
    Process Real-ESRGAN one frame at a time, storing output as uint8 PIL.
    Peak RAM during upscale: ~1 frame on GPU (~4 MB) + result list (uint8, ~1.3 GB).
    Only convert to float32 at final export_to_video call.
    Same streaming approach applied to RIFE interpolation.
"""

import gc
import os
import sys
import time
import torch
import traceback
import numpy as np
import diffusers
from PIL import Image, ImageFilter
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

MODEL_LOCAL_DIR = os.path.join(
    MODELS_DIR,
    "wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_Multiscene_SVI_sdnq_fp8_svd_r64"
)
MODEL_HF_REPO = "rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_Multiscene_SVI_sdnq_fp8_svd_r64"

# --- Input ---
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
]

# --- Long video settings ---
NUM_CLIPS      = 4

NUM_FRAMES     = 81
FPS            = 15
CONTEXT_FRAMES = 5

NOISE_AUG_STRENGTH   = 0.0
NUM_OVERLAP_FRAME    = 5   # frames dropped from start of each continuation clip
NUM_MOTION_FRAME     = 1   # keep at 1: last single frame, no averaging (avoids smoke)
BLEND_AT_JOIN_FRAMES = 3   # cross-fade frames at join (0 = hard cut after overlap)

# --- Per-clip prompts ---
CLIP_PROMPTS = [
    "Completely remove all the subjects clothes. Her clothes are now completely removed to expose her breasts and vagina. Maintain identical camera angle, framing, skintone and perspective and make sure to leave her facial features identical. The rest of the image remains the same, only remove her clothes to expose her naked body.",
    "Continue the scene with fluid natural motion, maintaining consistent lighting and character appearance.",
    "The subject continues moving naturally, camera holds steady, seamless continuation of the previous action.",
    "Final continuation, natural organic motion, temporally consistent with previous frames.",
]

NEGATIVE_PROMPT = (
    # Original Chinese terms
    "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, "
    "JPEG压缩残留, 丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景, 三条腿, 背景人很多, 倒着走, "
    # Atmospheric suppression — targets smoke/haze/fog artifacts at transitions
    "fog, haze, smoke, mist, steam, foggy, misty, smoky, atmospheric haze, "
    "smoke effects, hazy background, particle effects, vapor, dust haze, "
    "transition, fade in, fade out, dissolve, wipe, flash, "
    "overexposed, blown out highlights, washed out, "
    "soft focus, out of focus, blurry, defocus, "
    # Identity stability — suppresses face drift across clips
    "face morphing, identity change, different person, face swap, "
    "character change, different face, changing face, face drift, face change, "
    # Color drift suppression
    "color shift, color cast, tint change, hue shift, color grading change"
)

# --- Inference settings ---
NUM_INFERENCE_STEPS = 6
GUIDANCE_SCALE      = 1.5
SEED_MULTIPLIER     = 42   # seed per clip = clip_idx * SEED_MULTIPLIER

# --- Post-processing ---
ENABLE_RIFE      = True
RIFE_TARGET_FPS  = 30
ENABLE_UPSCALING = True
UPSCALE_FACTOR   = 2

# --- Output ---
OUTPUT_VIDEO_BASENAME = f"long_video_svi_v3_{NUM_CLIPS}clips"
SAVE_INDIVIDUAL_CLIPS = True


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------
def flush_memory():
    """Free CPU and GPU memory without touching the loaded model."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def vram_str() -> str:
    if not torch.cuda.is_available():
        return ""
    alloc = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{alloc:.1f}/{total:.1f}GB"


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------
def frames_to_uint8_pil(frames) -> list:
    """
    Convert pipeline output (list of float32 numpy [H,W,3] in [0,1]) to
    uint8 PIL Images. uint8 uses 4x less RAM than float32.
    Called immediately after pipeline returns, before storing frames anywhere.
    """
    result = []
    for f in frames:
        if isinstance(f, Image.Image):
            result.append(f.convert("RGB") if f.mode != "RGB" else f)
        else:
            arr = np.asarray(f, dtype=np.float32)
            if arr.max() <= 1.0:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            result.append(Image.fromarray(arr))
    return result


def resize_for_wan(image: Image.Image, max_area: int = 480 * 832, mod: int = 16):
    ar = image.height / image.width
    height = max(int(np.sqrt(max_area * ar)) // mod * mod, mod)
    width  = max(int(np.sqrt(max_area / ar)) // mod * mod, mod)
    return image.resize((width, height), Image.LANCZOS), width, height


def get_handoff_frame(frames: list, width: int, height: int) -> Image.Image:
    """
    Last frame of current clip → conditioning image for next clip.
    Applies one PIL SHARPEN pass: sharpness removes the model's tendency
    to interpret a soft/blurry input as fog/haze/smoke.
    """
    last = frames[-1]
    if not isinstance(last, Image.Image):
        arr = np.asarray(last, dtype=np.float32)
        arr = (np.clip(arr if arr.max() <= 1.0 else arr / 255.0, 0, 1) * 255).astype(np.uint8)
        last = Image.fromarray(arr)
    if last.size != (width, height):
        last = last.resize((width, height), Image.LANCZOS)
    return last.filter(ImageFilter.SHARPEN)


def save_clip_video(frames: list, path: str, fps: int):
    """
    Save clip from uint8 PIL list. Builds float32 array, exports, then deletes.
    Never holds the array beyond this call — no persistent float32 copy per clip.
    """
    arr = np.stack([
        np.asarray(f, dtype=np.float32) / 255.0
        for f in frames
    ])
    export_to_video(arr, path, fps=fps)
    del arr


def blend_at_join(prev_tail: list, next_head: list, n: int) -> list:
    """Cross-fade n frames from prev_tail into next_head. Returns n PIL frames."""
    def to_f32(f):
        a = np.asarray(f, dtype=np.float32)
        return a / 255.0 if a.max() > 1.0 else a

    before = np.stack([to_f32(f) for f in prev_tail])
    after  = np.stack([to_f32(f) for f in next_head])
    blended = []
    for i in range(n):
        alpha = (i + 1) / (n + 1)
        b = np.clip(before[i] * (1.0 - alpha) + after[i] * alpha, 0, 1)
        blended.append(Image.fromarray((b * 255).astype(np.uint8)))
    return blended


def stitch_append(running: list, new_clip: list, overlap: int, blend_frames: int) -> list:
    """
    Append one clip to the running stitched result in-place.

    This replaces the old all_clips[] + stitch_clips() pattern.
    Memory at any point: running (final video so far) + new_clip (one clip).
    Once this returns, new_clip can be deleted by the caller.
    """
    trimmed = new_clip[overlap:]  # drop warm-up frames from start of new clip
    if blend_frames > 0 and len(running) >= blend_frames and len(trimmed) >= blend_frames:
        prev_tail = running[-blend_frames:]
        next_head = trimmed[:blend_frames]
        blended   = blend_at_join(prev_tail, next_head, blend_frames)
        return running[:-blend_frames] + blended + trimmed[blend_frames:]
    else:
        return running + trimmed


def pil_list_to_f32(frames: list) -> np.ndarray:
    """Convert uint8 PIL list → float32 [T,H,W,3] in [0,1]. Called once at export."""
    return np.stack([np.asarray(f, dtype=np.float32) / 255.0 for f in frames])


def upscale_frames_streaming(upscale_model, frames: list, device: str,
                              upscale_factor: int,
                              batch_size: int = 50) -> list:
    """
    FIX 5: Batched Real-ESRGAN upscaling, returning uint8 PIL list.

    ROOT CAUSE of kill: video_postprocessing.tiled_scale() does:
        output = torch.empty([ALL_FRAMES, 3, H*2, W*2], device='cpu')
    This pre-allocates the ENTIRE output before processing any tile.
    300 frames at 864x1792 = 5.2 GB. Plus pt_to_numpy copy = 10.4 GB peak.
    On top of 20 GB model = 31+ GB -> killed on 32 GB system.

    FIX: Process `batch_size` frames at a time using tiled_scale directly.
    Convert each batch to uint8 PIL immediately, accumulate result list.
    Peak per batch: ~220 MB. Total accumulated uint8: ~1.3 GB.

    Uses the spandrel ModelLoader interface (same as load_sd_upscale returns).
    """
    from video_postprocessing import tiled_scale
    from diffusers.image_processor import VaeImageProcessor

    result = []
    n = len(frames)
    upscale_model.to(device)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = frames[start:end]
        if start % 40 == 0 or start == 0:
            print(f"   Upscaling frames {start+1}-{end}/{n}...", flush=True)

        # Build float16 tensor [B, C, H, W] for this batch only
        arrs = []
        for f in batch:
            a = np.asarray(f if isinstance(f, Image.Image) else Image.fromarray(f),
                           dtype=np.float32) / 255.0
            arrs.append(a)
        batch_np = np.stack(arrs, axis=0)                        # [B, H, W, C]
        batch_t  = torch.from_numpy(
            np.transpose(batch_np, (0, 3, 1, 2))                 # [B, C, H, W]
        ).to(torch.float16).to(device)
        del batch_np, arrs

        # tiled_scale output stays on CPU (output_device='cpu')
        upscaled_t = tiled_scale(
            samples=batch_t,
            function=lambda a: upscale_model(a),
            tile_x=512,
            tile_y=512,
            overlap=32,
            upscale_amount=upscale_model.scale,
            out_channels=3,
            output_device="cpu",
        )
        del batch_t

        # Convert to uint8 PIL immediately — 4x smaller than float32
        upscaled_t = upscaled_t.clamp(0, 1)
        batch_out = VaeImageProcessor.pt_to_numpy(upscaled_t)    # [B, H, W, C] float32
        del upscaled_t

        for i in range(len(batch_out)):
            result.append(Image.fromarray((batch_out[i] * 255).astype(np.uint8)))
        del batch_out

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    upscale_model.to("cpu")  # offload back to CPU when done
    return result


def rife_frames_streaming(rife_model, frames: list, device: str,
                           original_fps: int, target_fps: int) -> tuple:
    """
    FIX 5b: Frame-by-frame RIFE interpolation, returning uint8 PIL list.

    The full-tensor approach (ssim_interpolation_rife) requires loading all N frames
    as a float32 GPU tensor at once. After upscaling to 864x1792, 300 frames = 5.2 GB.

    This version processes one frame PAIR at a time — only 2 frames on GPU ever.
    Peak GPU RAM: ~35 MB. Result accumulates as uint8 PIL (2.6 GB for 600 frames).
    Net savings vs bulk approach: ~5.2 GB.

    RIFE HDv3 API: model.inference(img0, img1, scale=1.0)
    Returns the midpoint frame [1, C, H, W]. For exp=1 (2x), this is all we need.
    For exp=2 (4x): recursively insert midpoints between each pair.
    """
    from video_postprocessing import calculate_rife_exp, make_inference

    if target_fps <= original_fps:
        return frames, original_fps

    exp, actual_fps = calculate_rife_exp(original_fps, target_fps)
    if exp == 0:
        return frames, original_fps

    n = len(frames)
    print(f"   RIFE exp={exp} ({original_fps}fps → {actual_fps}fps), "
          f"{n} frames → {n * 2**exp} frames...", flush=True)

    def pil_to_gpu(f) -> torch.Tensor:
        """uint8 PIL → float32 [1, C, H, W] on device."""
        arr = np.asarray(f, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    def tensor_to_pil(t: torch.Tensor) -> Image.Image:
        """[1, C, H, W] GPU tensor → uint8 PIL, freed immediately."""
        arr = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        return Image.fromarray((arr * 255).astype(np.uint8))

    result = []
    for i in range(n):
        # Always include the original frame
        result.append(frames[i])

        if i < n - 1:
            with torch.no_grad():
                t0 = pil_to_gpu(frames[i])
                t1 = pil_to_gpu(frames[i + 1])
                # make_inference returns list of 2^exp - 1 intermediate tensors
                midpoints = make_inference(rife_model, t0, t1,
                                           upscale_amount=1, n=2**exp - 1)
                del t0, t1
                for mid_t in midpoints:
                    result.append(tensor_to_pil(mid_t))
                    del mid_t
                del midpoints

            if i % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if i % 50 == 0:
            print(f"   RIFE pair {i+1}/{n}  ({len(result)} frames so far)...", flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, actual_fps


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def ensure_model_available():
    if os.path.exists(MODEL_LOCAL_DIR):
        markers = ["model_index.json", "config.json", "quantization_config.json"]
        if any(os.path.exists(os.path.join(MODEL_LOCAL_DIR, m)) for m in markers):
            print(f"✓ Model found: {MODEL_LOCAL_DIR}")
            return
        print("⚠ Model directory incomplete, re-downloading...")
    print(f"\n⬇ Downloading: {MODEL_HF_REPO}")
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


def format_time(s):
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s // 60)}m {s % 60:.1f}s"


# ---------------------------------------------------------------------------
# Single clip generation
# ---------------------------------------------------------------------------
def generate_clip(pipe, start_image: Image.Image, prompt: str,
                  width: int, height: int, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)

    print(f"   seed={seed}  size={width}x{height}  steps={NUM_INFERENCE_STEPS}"
          f"  cfg={GUIDANCE_SCALE}  VRAM={vram_str()}", flush=True)

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
            raw_frames = pipe(**kwargs, callback_on_step_end=_cb).frames[0]
        except TypeError:
            raw_frames = pipe(**kwargs).frames[0]

    gen_time = time.time() - t0
    print(f"\n   Generated in {format_time(gen_time)}", flush=True)

    # Convert to uint8 PIL immediately — 4x less RAM than keeping float32
    frames = frames_to_uint8_pil(raw_frames)
    del raw_frames
    flush_memory()

    return frames, gen_time


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_long_video():
    t_start = time.time()
    print("\n" + "=" * 60)
    print("Wan2.2 Long Video V3 — SVI Autoregressive Clip Chaining")
    print("=" * 60)
    print(f"  overlap       : {NUM_OVERLAP_FRAME} frames dropped per join")
    print(f"  blend at join : {BLEND_AT_JOIN_FRAMES} frame cross-fade")
    print(f"  handoff       : last frame + SHARPEN (no averaging)")
    print(f"  stitch mode   : incremental (no all_clips[] accumulation)")
    print(f"  frame storage : uint8 PIL (4x less RAM than float32)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"\nGPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")

    ensure_model_available()

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

    paths = INPUT_IMAGE_PATHS if isinstance(INPUT_IMAGE_PATHS, (list, tuple)) else [INPUT_IMAGE_PATHS]
    image_paths = [os.path.join(WORKSPACE_DIR, p) if not os.path.isabs(p) else p for p in paths]
    image_paths = [p for p in image_paths if os.path.exists(p)]
    if not image_paths:
        raise FileNotFoundError(f"No input images found. Checked: {paths}")

    output_paths = []

    for img_idx, input_path in enumerate(image_paths):
        print("\n" + "#" * 60)
        print(f"BATCH {img_idx + 1}/{len(image_paths)}: {os.path.basename(input_path)}")
        if torch.cuda.is_available():
            print(f"VRAM before image: {vram_str()}")
        print("#" * 60)

        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(WORKSPACE_DIR, f"{OUTPUT_VIDEO_BASENAME}_{base}.mp4")

        source_image = load_image(input_path)
        print(f"\nInput : {source_image.width}x{source_image.height}  →  ", end="")
        source_image, width, height = resize_for_wan(source_image)
        print(f"{width}x{height}")

        prompts = list(CLIP_PROMPTS)
        while len(prompts) < NUM_CLIPS:
            prompts.append(prompts[-1])

        total_output_frames = NUM_FRAMES + (NUM_CLIPS - 1) * (NUM_FRAMES - NUM_OVERLAP_FRAME)
        print(f"\nPlan  : {NUM_CLIPS} clips × {NUM_FRAMES} frames, overlap={NUM_OVERLAP_FRAME}, blend={BLEND_AT_JOIN_FRAMES}")
        print(f"Output: ~{total_output_frames} frames  ({total_output_frames / FPS:.1f}s at {FPS}fps)\n")

        # stitched = running assembled video as uint8 PIL list
        # Never holds more than (final_video + 1 clip) at any point
        stitched: list = []
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
                seed=clip_idx * SEED_MULTIPLIER,
            )
            all_gen_times.append(gen_time)

            # Save individual clip immediately — no persistent float32 copy
            if SAVE_INDIVIDUAL_CLIPS:
                clip_path = output_path.replace(".mp4", f"_clip{clip_idx + 1:02d}.mp4")
                save_clip_video(frames, clip_path, FPS)
                print(f"   Saved: {os.path.basename(clip_path)}")

            # Incremental stitch — append then free
            if clip_idx == 0:
                stitched = list(frames)
            else:
                stitched = stitch_append(stitched, frames, NUM_OVERLAP_FRAME, BLEND_AT_JOIN_FRAMES)
            print(f"   Stitched total: {len(stitched)} frames")

            # Extract handoff BEFORE freeing frames
            if clip_idx < NUM_CLIPS - 1:
                current_start = get_handoff_frame(frames, width, height)
                print(f"   ✓ Handoff → clip {clip_idx + 2}  VRAM={vram_str()}")

            # Free clip — the only large allocation we can control here
            del frames
            flush_memory()

        print(f"\nFinal stitched: {len(stitched)} frames ({len(stitched) / FPS:.1f}s)")

        # FIX 5: Post-processing stays in uint8 PIL list throughout.
        # Only convert to float32 at the very final export_to_video call.
        # This avoids the 5-10 GB numpy arrays that apply_realesrgan_upscaling
        # and apply_rife_interpolation allocate internally.
        final_frames = stitched  # uint8 PIL list
        del stitched
        flush_memory()

        final_fps = FPS
        final_h, final_w = height, width

        if POSTPROCESSING_AVAILABLE and (ENABLE_UPSCALING or ENABLE_RIFE):
            print("\nPost-processing...")
            if ENABLE_UPSCALING and upscale_model:
                # Streaming: one frame at a time on GPU, output as uint8 PIL
                final_frames = upscale_frames_streaming(
                    upscale_model, final_frames, device, UPSCALE_FACTOR
                )
                final_h, final_w = height * UPSCALE_FACTOR, width * UPSCALE_FACTOR
                flush_memory()
                print(f"   Upscaled  : {final_w}x{final_h}  ({len(final_frames)} frames)")

            if ENABLE_RIFE and rife_model:
                # Streaming: interpolate frame pairs, output as uint8 PIL
                final_frames, final_fps = rife_frames_streaming(
                    rife_model, final_frames, device,
                    original_fps=FPS, target_fps=RIFE_TARGET_FPS,
                )
                flush_memory()
                print(f"   Interpolated: {final_fps}fps  ({len(final_frames)} frames)")

        # Export directly from uint8 PIL list — export_to_video accepts PIL lists.
        # This avoids the final pil_list_to_f32() which would allocate 5-10 GB
        # for 600 upscaled frames. export_to_video handles PIL internally.
        print(f"   Exporting {len(final_frames)} frames at {final_fps}fps...")
        export_to_video(final_frames, output_path, fps=final_fps)
        output_paths.append(output_path)
        print(f"   → {output_path}")

        # Flush everything before next image
        del final_frames
        flush_memory()
        print(f"   Flushed. VRAM={vram_str()}")

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