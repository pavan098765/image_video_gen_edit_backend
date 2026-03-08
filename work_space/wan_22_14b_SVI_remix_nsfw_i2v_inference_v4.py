#!/usr/bin/env python3
"""
Wan2.2 SVI Pro Inference — Official Pipeline + Custom Baked Model
==================================================================

ENVIRONMENT (must use SVI fork, NOT pip install diffsynth):
    cd /work_space                          # already done — diffsynth here is the SVI fork
    pip install -e . --force-reinstall      # reinstall from /work_space (SVI fork root)
    # Verify:
    python3 -c "import diffsynth; print(diffsynth.__file__)"
    # Must print: /work_space/diffsynth/__init__.py  ← this is already the case

WHAT THIS SCRIPT DOES DIFFERENTLY FROM YOUR V3:
    - Uses WanVideoSviProPipeline (dual DiT: high-noise + low-noise)
    - Passes prev_last_latent between clips (raw VAE latent, no decode/encode cycle)
    - Passes anchor image every clip (frozen identity reference)
    - Uses official num_overlap_frame=5, num_motion_latent=1 defaults

YOUR CUSTOM MODEL vs OFFICIAL MODEL:
    Official flow:
        Wan2.2-I2V-A14B base weights  +  SVI Pro LoRA loaded on top

    Your custom model (rockapaper/...):
        transformer/        <- high-noise DiT WITH SVI LoRA already baked in
        transformer_2/      <- low-noise DiT WITH SVI LoRA already baked in
        text_encoder/       <- T5 encoder (standard)
        vae/                <- Wan VAE (standard)

    So you:
        - Skip pipe.load_lora() calls entirely
        - Point ModelConfig at your baked transformer shards directly
        - Still need T5 + VAE (use yours from text_encoder/ and vae/)

MODEL STRUCTURE REQUIRED BY WanVideoSviProPipeline (ModelConfig order):
    [0] high-noise DiT shards  -> becomes pipe.dit
    [1] low-noise DiT shards   -> becomes pipe.dit2
    [2] T5 text encoder        -> becomes pipe.prompter
    [3] VAE                    -> becomes pipe.vae

    DiffSynth identifies model type automatically by inspecting tensor keys.
    For sharded models, pass a list of shard paths as a single ModelConfig.

HOW TO FIX THE IMPORT ERROR (if you still see it):
    The problem is /work_space/diffsynth/ is the SVI fork but pip installed
    an old editable entry pointing elsewhere. Fix:
        pip uninstall diffsynth -y
        cd /work_space
        pip install -e .
    Then verify:
        python3 -c "from diffsynth.pipelines.wan_video_svi_pro import WanVideoSviProPipeline; print('OK')"
"""

import sys
import os

# Ensure the SVI diffsynth is the one being used.
# /work_space IS the SVI fork root (diffsynth/ folder lives there directly).
_WORKSPACE = os.path.dirname(os.path.abspath(__file__))  # /work_space
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

import gc
import time
import torch
import numpy as np
import argparse
from PIL import Image

try:
    from diffsynth.pipelines.wan_video_svi_pro import WanVideoSviProPipeline, ModelConfig
    from diffsynth.utils.data import save_video
    print(f"OK WanVideoSviProPipeline loaded from: {WanVideoSviProPipeline.__module__}")
except ImportError as e:
    print(f"\n ERROR: {e}")
    print(f"\n sys.path = {sys.path}")
    print(
        "\n Fix:\n"
        "   pip uninstall diffsynth -y\n"
        "   cd /work_space && pip install -e .\n"
        "   python3 -c \"from diffsynth.pipelines.wan_video_svi_pro import WanVideoSviProPipeline; print('OK')\"\n"
    )
    sys.exit(1)
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))

# Your custom model root (downloaded from HF)
# rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_Multiscene_SVI_sdnq_fp8_svd_r64
CUSTOM_MODEL_DIR = os.path.join(
    WORKSPACE_DIR, "models",
    "wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_Multiscene_SVI_sdnq_fp8_svd_r64"
)

# Subfolders inside your custom model
HIGH_NOISE_DIR = os.path.join(CUSTOM_MODEL_DIR, "transformer")    # high-noise DiT shards
LOW_NOISE_DIR  = os.path.join(CUSTOM_MODEL_DIR, "transformer_2")  # low-noise DiT shards
TEXT_ENC_DIR   = os.path.join(CUSTOM_MODEL_DIR, "text_encoder")   # T5
VAE_DIR        = os.path.join(CUSTOM_MODEL_DIR, "vae")            # VAE

# Input images
INPUT_IMAGE_PATHS = [
    os.path.join(WORKSPACE_DIR, "an1.jpeg"),
    os.path.join(WORKSPACE_DIR, "d9.jpeg"),
    os.path.join(WORKSPACE_DIR, "d10.jpeg"),
    os.path.join(WORKSPACE_DIR, "n1.jpeg"),
    os.path.join(WORKSPACE_DIR, "h1.jpeg"),
]

# Video settings
HEIGHT     = 480
WIDTH      = 832
FPS        = 15
NUM_FRAMES = 81  # must be 4k+1 for Wan VAE

# SVI Pro official parameters (from README command)
NUM_CLIPS         = 4
NUM_MOTION_LATENT = 1    # latent frames handed off — prevents smoke/haze
NUM_OVERLAP_FRAME = 5    # frames trimmed at stitch (official default)
NUM_MOTION_FRAME  = 1    # last N decoded frames -> next clip input_image
CFG_SCALE         = 4.0  # official default

# Seeds: large spacing = uncorrelated denoising trajectories per clip
SEED_MULTIPLIER = 1000   # clip 0->0, 1->1000, 2->2000

# Prompts
CLIP_PROMPTS = [
    "Subject moves naturally, fluid cinematic motion, consistent lighting.",
    "Continue the scene naturally, maintaining identical character appearance.",
    "Subject continues moving, camera steady, seamless continuation.",
    "Final continuation, organic motion, temporally consistent.",
]

NEGATIVE_PROMPT = (
    "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, "
    "JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, "
    "形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走, "
    "fog, haze, smoke, mist, steam, foggy, misty, smoky, atmospheric haze, "
    "face morphing, identity change, different person, face drift, "
    "color shift, color cast, tint change"
)

OUTPUT_DIR            = os.path.join(WORKSPACE_DIR, "outputs")
SAVE_INDIVIDUAL_CLIPS = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def format_time(s: float) -> str:
    return f"{int(s//60)}m {s%60:.1f}s" if s >= 60 else f"{s:.1f}s"


def vram_str() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    a = torch.cuda.memory_allocated() / 1024**3
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{a:.1f}/{t:.1f}GB"


def get_shard_paths(folder: str) -> list:
    """
    Return sorted list of all .safetensors shard paths in a folder.
    DiffSynth ModelConfig accepts a list of shards for a single model component.
    """
    shards = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".safetensors")
    ])
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found in: {folder}")
    print(f"  {os.path.basename(folder)}/  ->  {len(shards)} shard(s)")
    return shards


def video_to_pil_list(video_clip) -> list:
    """Convert pipeline video output (tensor or list) to uint8 PIL list."""
    if isinstance(video_clip, torch.Tensor):
        frames = []
        for f in video_clip:
            arr = f.cpu().numpy()
            arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
            frames.append(Image.fromarray(arr))
        return frames
    elif isinstance(video_clip, list):
        result = []
        for f in video_clip:
            if isinstance(f, Image.Image):
                result.append(f.convert("RGB"))
            else:
                arr = np.asarray(f, dtype=np.float32)
                arr = (np.clip(arr if arr.max() <= 1.0 else arr/255.0, 0, 1) * 255).astype(np.uint8)
                result.append(Image.fromarray(arr))
        return result
    raise TypeError(f"Unexpected video type: {type(video_clip)}")


def resize_for_wan(image: Image.Image) -> Image.Image:
    return image.resize((WIDTH, HEIGHT), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------
def load_pipeline() -> WanVideoSviProPipeline:
    """
    Load WanVideoSviProPipeline with your custom baked model.

    ModelConfig(path=...) accepts:
        - A single .safetensors file path (string)
        - A list of shard paths for sharded models

    DiffSynth auto-identifies which component each file is by inspecting
    tensor key patterns — no need to specify model type manually.

    Since your SVI LoRA is already baked into the weights, we do NOT call
    pipe.load_lora() at all. The pipeline is ready to use immediately.
    """
    print("\n" + "=" * 60)
    print("Loading WanVideoSviProPipeline")
    print("=" * 60)

    high_noise_shards = get_shard_paths(HIGH_NOISE_DIR)
    low_noise_shards  = get_shard_paths(LOW_NOISE_DIR)
    text_enc_shards   = get_shard_paths(TEXT_ENC_DIR)
    vae_shards        = get_shard_paths(VAE_DIR)

    # Unwrap single-shard lists to plain string (cleaner, both work)
    text_enc_path = text_enc_shards[0] if len(text_enc_shards) == 1 else text_enc_shards
    vae_path      = vae_shards[0]      if len(vae_shards) == 1      else vae_shards

    t0 = time.time()
    pipe = WanVideoSviProPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            # [0] high-noise DiT -> pipe.dit
            ModelConfig(
                path=high_noise_shards,
                offload_device="cpu",
            ),
            # [1] low-noise DiT -> pipe.dit2
            ModelConfig(
                path=low_noise_shards,
                offload_device="cpu",
            ),
            # [2] T5 text encoder -> pipe.prompter
            ModelConfig(
                path=text_enc_path,
                offload_device="cpu",
            ),
            # [3] VAE -> pipe.vae
            ModelConfig(
                path=vae_path,
                offload_device="cpu",
            ),
        ],
    )

    # SVI LoRA is ALREADY BAKED IN — do NOT call pipe.load_lora() here.
    # If you were using official base Wan2.2 weights you would instead do:
    #   pipe.load_lora(pipe.dit,  lora_path_high, alpha=1)
    #   pipe.load_lora(pipe.dit2, lora_path_low,  alpha=1)

    print(f"Pipeline loaded in {format_time(time.time() - t0)}  VRAM={vram_str()}")
    return pipe


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def run_long_video(pipe: WanVideoSviProPipeline, input_path: str, prompts: list):
    """
    Generate a long video from a single input image using SVI Pro clip chaining.

    What fixes smoke/haze + face drift:

    1. prev_last_latent:
       The pipeline returns the raw VAE latent of the last num_motion_latent
       frames. This is passed directly into the next clip — NO decode->PIL->encode.
       The VAE encode/decode cycle produces a soft-blur that the model reads as
       atmospheric haze/fog. Staying in latent space eliminates this entirely.

    2. anchor (original source image):
       Passed every single clip unchanged. The model uses this as a frozen
       identity reference to prevent face/appearance drift across clips.
       This is the "error recycling" core of the SVI paper.

    3. input_image (visual handoff):
       Last num_motion_frame decoded PIL frames from previous clip.
       Provides spatial/visual context for where the scene is now.
    """
    base       = os.path.splitext(os.path.basename(input_path))[0]
    out_prefix = os.path.join(OUTPUT_DIR, f"svi_pro_{base}")

    source_image = Image.open(input_path).convert("RGB")
    source_image = resize_for_wan(source_image)
    print(f"Input: {os.path.basename(input_path)} -> {WIDTH}x{HEIGHT}")

    # anchor: NEVER changes across clips — frozen identity reference
    anchor_image = source_image.copy()

    # State
    all_frames          = []
    prev_last_latent    = None         # raw VAE latent handoff (None for first clip)
    current_input_image = source_image

    expected = NUM_FRAMES + (NUM_CLIPS - 1) * (NUM_FRAMES - NUM_OVERLAP_FRAME)
    print(f"Expected: ~{expected} frames ({expected/FPS:.1f}s @ {FPS}fps)\n")

    for clip_idx in range(NUM_CLIPS):
        prompt = prompts[clip_idx] if clip_idx < len(prompts) else prompts[-1]

        print(f"{'='*60}")
        print(f"CLIP {clip_idx+1}/{NUM_CLIPS}")
        print(f"Prompt : {prompt[:80]}")
        print(f"Seed   : {clip_idx * SEED_MULTIPLIER}  VRAM={vram_str()}")
        print(f"Latent : {'handoff from prev clip' if prev_last_latent is not None else 'none (first clip)'}")
        print("=" * 60)

        t_clip = time.time()

        # Run inference
        # WanVideoSviProPipeline returns dict:
        #   {"video": <frames>, "prev_last_latent": <tensor>}
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            seed=clip_idx * SEED_MULTIPLIER,
            tiled=False,
            height=HEIGHT,
            width=WIDTH,
            input_image=current_input_image,
            num_frames=NUM_FRAMES,
            anchor=anchor_image,           # frozen identity anchor — never changes
            prev_last_latent=prev_last_latent,  # raw latent handoff
            num_motion_latent=NUM_MOTION_LATENT,
            cfg_scale=CFG_SCALE,
        )

        # Capture latent handoff for next clip BEFORE converting video
        if NUM_MOTION_LATENT > 0:
            prev_last_latent = result["prev_last_latent"]

        clip_frames = video_to_pil_list(result["video"])
        del result
        flush_memory()

        print(f"  Generated {len(clip_frames)} frames in {format_time(time.time()-t_clip)}")

        # Save individual clip
        if SAVE_INDIVIDUAL_CLIPS:
            clip_path = f"{out_prefix}_clip{clip_idx+1:02d}.mp4"
            save_video(clip_frames, clip_path, fps=FPS, quality=7)
            print(f"  Saved: {os.path.basename(clip_path)}")

        # Stitch: drop NUM_OVERLAP_FRAME from start of clips 2+
        # These frames are the motion-conditioning frames that duplicate
        # the tail of the previous clip.
        if clip_idx == 0:
            all_frames = list(clip_frames)
        else:
            all_frames = all_frames + clip_frames[NUM_OVERLAP_FRAME:]

        print(f"  Running total: {len(all_frames)} frames ({len(all_frames)/FPS:.1f}s)")

        # Set next clip's visual input
        if clip_idx < NUM_CLIPS - 1:
            current_input_image = clip_frames[-NUM_MOTION_FRAME]

        del clip_frames
        flush_memory()

    # Save final video
    final_path = f"{out_prefix}_{NUM_CLIPS}clips_final.mp4"
    print(f"\nSaving {len(all_frames)} frames -> {os.path.basename(final_path)}")
    save_video(all_frames, final_path, fps=FPS, quality=5)
    print(f"Saved: {final_path}")

    del all_frames
    flush_memory()
    return final_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    global NUM_CLIPS, CFG_SCALE, HEIGHT, WIDTH, FPS, OUTPUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, default=None)
    parser.add_argument("--prompt_path", type=str, default=None)
    parser.add_argument("--num_clips",   type=int, default=NUM_CLIPS)
    parser.add_argument("--cfg_scale",   type=float, default=CFG_SCALE)
    parser.add_argument("--height",      type=int, default=HEIGHT)
    parser.add_argument("--width",       type=int, default=WIDTH)
    parser.add_argument("--fps",         type=int, default=FPS)
    args = parser.parse_args()

    NUM_CLIPS = args.num_clips
    CFG_SCALE = args.cfg_scale
    HEIGHT    = args.height
    WIDTH     = args.width
    FPS       = args.fps
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Resolve prompts
    prompts = CLIP_PROMPTS
    if args.prompt_path and os.path.exists(args.prompt_path):
        with open(args.prompt_path) as f:
            content = f.read()
        local_vars = {}
        exec(content, {}, local_vars)
        prompts = local_vars.get("prompts", CLIP_PROMPTS)
    while len(prompts) < NUM_CLIPS:
        prompts.append(prompts[-1])

    # Resolve input images
    image_paths = [args.input_image] if args.input_image else [
        p for p in INPUT_IMAGE_PATHS if os.path.exists(p)
    ]
    if not image_paths:
        raise FileNotFoundError("No input images found.")

    print("\n" + "=" * 60)
    print("Wan2.2 SVI Pro — Official Pipeline + Custom Baked Model")
    print("=" * 60)
    print(f"  num_motion_latent : {NUM_MOTION_LATENT}  (latent handoff — no smoke)")
    print(f"  num_overlap_frame : {NUM_OVERLAP_FRAME}  (official default)")
    print(f"  anchor            : enabled (identity lock)")
    print(f"  cfg_scale         : {CFG_SCALE}")
    print(f"  seed pattern      : clip_idx x {SEED_MULTIPLIER}")
    print(f"  images            : {len(image_paths)}")
    if torch.cuda.is_available():
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    pipe    = load_pipeline()
    t_total = time.time()

    for idx, path in enumerate(image_paths):
        print(f"\n{'#'*60}")
        print(f"IMAGE {idx+1}/{len(image_paths)}: {os.path.basename(path)}")
        print("#" * 60)
        run_long_video(pipe, path, prompts)
        flush_memory()

    print(f"\n{'='*60}")
    print(f"DONE  {len(image_paths)} image(s) in {format_time(time.time()-t_total)}")
    print("=" * 60)


if __name__ == "__main__":
    main()