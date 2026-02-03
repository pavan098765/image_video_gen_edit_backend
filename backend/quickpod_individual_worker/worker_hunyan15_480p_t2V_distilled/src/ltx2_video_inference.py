# ltx2_video_inference.py
# ✅ LTX-2 Video Generation Worker - FP4 Model
# Model: ltx-2-19b-dev-fp4.safetensors (20 GB)
# Works safely on RTX 4090 (24GB) and RTX 5090 (32GB) - No OOM risk
# Supports: Text-to-Video, Image-to-Video, Video-to-Video, Keyframe Interpolation

import gc
import os
import base64
import random
import tempfile
import time
from io import BytesIO
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import numpy as np
import torch
import imageio
from PIL import Image

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


# -----------------------------
# Environment Configuration
# -----------------------------
# HuggingFace repo ID for LTX-2 models
LTX2_REPO_ID = os.environ.get("LTX2_REPO_ID", "Lightricks/LTX-2")
GEMMA_REPO_ID = os.environ.get("LTX2_GEMMA_REPO_ID", "google/gemma-3-12b-it")

# Local model paths
MODEL_DIR = os.environ.get("LTX2_VIDEO_LOCAL_PATH", "/app/models/ltx2_video")
GEMMA_ROOT = os.environ.get("LTX2_GEMMA_ROOT", "/app/models/gemma")

# We ONLY use FP4 model - works safely on both RTX 4090 (24GB) and RTX 5090 (32GB)
# No risk of OOM with this checkpoint (20 GB)
CHECKPOINT_FILE = "ltx-2-19b-dev-fp4.safetensors"

# Additional files
SPATIAL_UPSAMPLER_FILE = "ltx-2-spatial-upscaler-x2-1.0.safetensors"
DISTILLED_LORA_FILE = "ltx-2-19b-distilled-lora-384.safetensors"

# Performance settings - FP4 doesn't need additional FP8 flag
# The model is already quantized
ENABLE_FP8 = os.environ.get("LTX2_ENABLE_FP8", "1") == "1"
MEMORY_CLEANUP_BETWEEN_STAGES = os.environ.get("LTX2_MEMORY_CLEANUP", "1") == "1"

# Inference defaults
DEFAULT_FPS = 25
MAX_SEED = np.iinfo(np.int32).max
MAX_HEIGHT = 1024
MAX_WIDTH = 1024
MAX_FRAMES = 121  # (15 * 8 + 1), approximately 5 seconds at 25fps

# Distilled model uses only 8 steps (stage 1: 8, stage 2: 4)
DISTILLED_STEPS_STAGE1 = 8
DISTILLED_STEPS_STAGE2 = 4

# -----------------------------
# Pipeline Cache
# -----------------------------
PIPELINE_CACHE: Dict[str, Any] = {}


# -----------------------------
# Multimodal Guidance Parameters
# -----------------------------
@dataclass
class GuidanceParams:
    """
    Guidance parameters for video generation.
    
    - cfg_scale: Classifier-Free Guidance (2.0-5.0 typical, 1.0 to disable)
    - stg_scale: Spatio-Temporal Guidance (0.5-1.5 typical, 0.0 to disable)
    - stg_blocks: Transformer blocks to perturb ([29] for last block)
    - rescale_scale: Variance rescaling (0.5-0.7, prevents over-saturation)
    - modality_scale: Audio-video sync (3.0 for A/V, 1.0 for video-only)
    """
    cfg_scale: float = 3.0
    stg_scale: float = 1.0
    rescale_scale: float = 0.7
    modality_scale: float = 1.0
    skip_step: int = 0
    stg_blocks: List[int] = field(default_factory=lambda: [29])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cfg_scale": self.cfg_scale,
            "stg_scale": self.stg_scale,
            "rescale_scale": self.rescale_scale,
            "modality_scale": self.modality_scale,
            "skip_step": self.skip_step,
            "stg_blocks": self.stg_blocks,
        }


# -----------------------------
# Helper Functions
# -----------------------------
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
        "is_ada": props.major == 9,  # RTX 40xx
        "is_blackwell": "5090" in props.name or "5080" in props.name,  # RTX 50xx
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


def calculate_optimal_frames(duration_sec: float, fps: int = DEFAULT_FPS) -> int:
    """
    Calculate optimal frame count for LTX-2.
    Frame count must satisfy: (num_frames - 1) % 8 == 0
    """
    target_frames = max(1, min(MAX_FRAMES, int(duration_sec * fps)))
    n_val = round((target_frames - 1) / 8.0)
    return int(n_val * 8 + 1)


def pad_dimensions_to_32(height: int, width: int) -> Tuple[int, int]:
    """Pad dimensions to be divisible by 32."""
    h_pad = ((height - 1) // 32 + 1) * 32
    w_pad = ((width - 1) // 32 + 1) * 32
    return h_pad, w_pad


def calculate_padding(
    source_height: int, source_width: int, 
    target_height: int, target_width: int
) -> Tuple[int, int, int, int]:
    """Calculate padding (left, right, top, bottom)."""
    pad_height = target_height - source_height
    pad_width = target_width - source_width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    return (pad_left, pad_right, pad_top, pad_bottom)


def load_image_from_base64(img_b64: str) -> Image.Image:
    """Load PIL Image from base64 string."""
    return Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")


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
    image: Image.Image, 
    target_height: int, 
    target_width: int
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


def export_video(
    frames: np.ndarray, 
    output_path: str, 
    fps: int = DEFAULT_FPS
) -> str:
    """Export numpy array of frames to video file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    with imageio.get_writer(output_path, fps=fps, macro_block_size=1) as writer:
        for i in range(frames.shape[0]):
            writer.append_data(frames[i])
    
    return output_path


# -----------------------------
# Pipeline Loading
# -----------------------------
def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model_paths() -> Dict[str, str]:
    """Get paths to model files."""
    model_paths = {}
    
    # Main checkpoint (FP4 - 20 GB)
    checkpoint_path = os.path.join(MODEL_DIR, CHECKPOINT_FILE)
    if os.path.exists(checkpoint_path):
        model_paths["checkpoint"] = checkpoint_path
    else:
        print(f"⚠️ Checkpoint not found: {checkpoint_path}")
    
    # Spatial upsampler
    upsampler_path = os.path.join(MODEL_DIR, SPATIAL_UPSAMPLER_FILE)
    if os.path.exists(upsampler_path):
        model_paths["spatial_upsampler"] = upsampler_path
    
    # Distilled LoRA
    lora_path = os.path.join(MODEL_DIR, DISTILLED_LORA_FILE)
    if os.path.exists(lora_path):
        model_paths["distilled_lora"] = lora_path
    
    # Gemma
    if os.path.exists(GEMMA_ROOT):
        model_paths["gemma_root"] = GEMMA_ROOT
    
    return model_paths


def get_ltx2_pipeline(pipeline_type: str = "distilled") -> Dict[str, Any]:
    """
    Load and cache LTX-2 pipeline.
    
    Using FP4 checkpoint: ltx-2-19b-dev-fp4.safetensors (20 GB)
    Works on RTX 4090 (24GB) and RTX 5090 (32GB) with no OOM risk.
    
    Pipeline types:
    - "distilled": DistilledPipeline (fastest, 8 steps)
    - "two_stage": TI2VidTwoStagesPipeline (best quality)
    - "ic_lora": ICLoraPipeline (video-to-video)
    - "keyframe": KeyframeInterpolationPipeline
    """
    cache_key = f"ltx2_{pipeline_type}"
    
    if cache_key in PIPELINE_CACHE:
        return PIPELINE_CACHE[cache_key]
    
    print(f"🔧 Loading LTX-2 {pipeline_type} pipeline (FP4 model)...")
    start_time = time.time()
    
    device = get_device()
    gpu_info = get_gpu_info()
    
    print(f"📊 GPU: {gpu_info['name']}, VRAM: {gpu_info['vram_gb']:.1f}GB")
    print(f"   Model: {CHECKPOINT_FILE} (20 GB) - No OOM risk")
    
    model_paths = get_model_paths()
    
    if "checkpoint" not in model_paths:
        raise FileNotFoundError(
            f"LTX-2 FP4 checkpoint not found at {MODEL_DIR}/{CHECKPOINT_FILE}. "
            f"Run prep_model.py first."
        )
    
    try:
        # Try to import ltx-pipelines
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
        from ltx_core.components.guiders import MultiModalGuiderParams
        
        # Build distilled LoRA config if available
        # IMPORTANT: distilled_lora must be an empty list (not None) when LoRA is not available
        distilled_lora = []
        if model_paths.get("distilled_lora"):
            distilled_lora = [
                LoraPathStrengthAndSDOps(
                    model_paths["distilled_lora"],
                    0.6 if pipeline_type == "two_stage" else 0.8,
                    LTXV_LORA_COMFY_RENAMING_MAP
                ),
            ]
        else:
            print("⚠️ Distilled LoRA not found - using empty lora list")
        
        pipeline = None
        
        if pipeline_type == "distilled":
            from ltx_pipelines.distilled import DistilledPipeline
            
            print(f"🚀 Loading DistilledPipeline (fastest, 8+4 steps)...")
            pipeline = DistilledPipeline(
                checkpoint_path=model_paths["checkpoint"],
                spatial_upsampler_path=model_paths.get("spatial_upsampler"),
                gemma_root=model_paths.get("gemma_root"),
                loras=[],
                fp8transformer=False,  # FP4 model, no need for FP8
            )
            
        elif pipeline_type == "two_stage":
            from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
            
            print(f"🚀 Loading TI2VidTwoStagesPipeline (production quality)...")
            pipeline = TI2VidTwoStagesPipeline(
                checkpoint_path=model_paths["checkpoint"],
                distilled_lora=distilled_lora,
                spatial_upsampler_path=model_paths.get("spatial_upsampler"),
                gemma_root=model_paths.get("gemma_root"),
                loras=[],
                fp8transformer=False,
            )
            
        elif pipeline_type == "ic_lora":
            from ltx_pipelines.ic_lora import ICLoraPipeline
            
            print(f"🚀 Loading ICLoraPipeline (video-to-video)...")
            # Note: ICLoraPipeline doesn't take distilled_lora in __init__
            pipeline = ICLoraPipeline(
                checkpoint_path=model_paths["checkpoint"],
                spatial_upsampler_path=model_paths.get("spatial_upsampler"),
                gemma_root=model_paths.get("gemma_root"),
                loras=[],
                fp8transformer=False,
            )
            
        elif pipeline_type == "keyframe":
            from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
            
            print(f"🚀 Loading KeyframeInterpolationPipeline...")
            pipeline = KeyframeInterpolationPipeline(
                checkpoint_path=model_paths["checkpoint"],
                distilled_lora=distilled_lora,
                spatial_upsampler_path=model_paths.get("spatial_upsampler"),
                gemma_root=model_paths.get("gemma_root"),
                loras=[],
                fp8transformer=False,
            )
            
        elif pipeline_type == "one_stage":
            from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
            
            print(f"🚀 Loading TI2VidOneStagePipeline (prototyping)...")
            pipeline = TI2VidOneStagePipeline(
                checkpoint_path=model_paths["checkpoint"],
                gemma_root=model_paths.get("gemma_root"),
                loras=[],
                fp8transformer=False,
            )
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        
        elapsed = time.time() - start_time
        print(f"✅ LTX-2 {pipeline_type} pipeline loaded in {elapsed:.1f}s")
        
        PIPELINE_CACHE[cache_key] = {
            "pipeline": pipeline,
            "device": device,
            "gpu_info": gpu_info,
            "pipeline_type": pipeline_type,
            "model_paths": model_paths,
        }
        
        return PIPELINE_CACHE[cache_key]
        
    except ImportError as e:
        print(f"❌ ltx-pipelines not installed! Error: {e}")
        print()
        print("=" * 60)
        print("INSTALLATION REQUIRED")
        print("=" * 60)
        print("LTX-2 requires the ltx-core and ltx-pipelines packages.")
        print("Install them with:")
        print()
        print('  pip install "git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-core"')
        print('  pip install "git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-pipelines"')
        print()
        print("=" * 60)
        raise ImportError(
            f"LTX-2 packages not installed. Install ltx-core and ltx-pipelines from "
            f"https://github.com/Lightricks/LTX-2.git (packages/ subdirectory). "
            f"Original error: {e}"
        )


def unload_pipeline(pipeline_type: str = None):
    """Unload pipeline(s) and free GPU memory."""
    global PIPELINE_CACHE
    
    if pipeline_type:
        cache_key = f"ltx2_{pipeline_type}"
        if cache_key in PIPELINE_CACHE:
            del PIPELINE_CACHE[cache_key]
            print(f"🧹 Unloaded {pipeline_type} pipeline")
    else:
        PIPELINE_CACHE.clear()
        print("🧹 Unloaded all pipelines")
    
    cleanup_gpu_memory(verbose=True)


# -----------------------------
# Inference Functions
# -----------------------------
def run_inference_text_to_video(
    job_id: str,
    prompt: str,
    params: Dict[str, Any]
) -> str:
    """
    Text-to-Video generation using LTX-2 FP4 model.
    
    Args:
        job_id: Unique job identifier
        prompt: Text prompt (keep under 200 words)
        params:
            - height (int): Output height (default: 512)
            - width (int): Output width (default: 768)
            - duration (float): Video duration in seconds (default: 5.0)
            - fps (int): Frames per second (default: 25)
            - num_inference_steps (int): Denoising steps (default: 8 for distilled)
            - seed (int): Random seed
            - guidance_scale (float): CFG scale (default: 1.0 for distilled)
            - pipeline_type (str): "distilled" (fast), "two_stage" (quality)
            - output_dir (str): Output directory
    
    Returns:
        Absolute path to generated video
    """
    print(f"🎬 Starting Text-to-Video job: {job_id}")
    
    cleanup_gpu_memory()
    
    # Parameters
    height = int(params.get("height", 512))
    width = int(params.get("width", 768))
    duration = float(params.get("duration", 5.0))
    fps = int(params.get("fps", DEFAULT_FPS))
    seed = int(params.get("seed", random.randint(0, MAX_SEED)))
    pipeline_type = params.get("pipeline_type", "distilled")
    output_dir = params.get("output_dir", "/app/worker/outputs")
    
    # Guidance - distilled model uses CFG=1
    is_distilled = "distilled" in pipeline_type
    guidance_scale = float(params.get("guidance_scale", 1.0 if is_distilled else 3.0))
    stg_scale = float(params.get("stg_scale", 0.0 if is_distilled else 1.0))
    num_steps = int(params.get("num_inference_steps", 
                               DISTILLED_STEPS_STAGE1 if is_distilled else 40))
    
    negative_prompt = params.get("negative_prompt", 
        "worst quality, inconsistent motion, blurry, jittery, distorted")
    
    # Prepare dimensions
    height = min(height, MAX_HEIGHT)
    width = min(width, MAX_WIDTH)
    height_padded, width_padded = pad_dimensions_to_32(height, width)
    num_frames = calculate_optimal_frames(duration, fps)
    
    print(f"📐 Resolution: {width}x{height} → {width_padded}x{height_padded}")
    print(f"   Frames: {num_frames} @ {fps}fps = {num_frames/fps:.1f}s")
    print(f"   Pipeline: {pipeline_type}, Steps: {num_steps}, CFG: {guidance_scale}")
    
    seed_everything(seed)
    
    data = get_ltx2_pipeline(pipeline_type)
    pipeline = data["pipeline"]
    device = data["device"]
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"t2v_{job_id}.mp4")
    
    # Import ltx utilities
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
    from ltx_core.model.video_vae import get_video_chunks_number, TilingConfig
    
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
    
    print(f"🚀 Generating video with seed={seed}...")
    
    if is_distilled:
        # DistilledPipeline: simpler API (no guider params, no negative_prompt)
        video, audio = pipeline(
            prompt=prompt,
            seed=seed,
            height=height_padded,
            width=width_padded,
            num_frames=num_frames,
            frame_rate=float(fps),
            images=[],  # No image conditioning for T2V
            tiling_config=tiling_config,
        )
    else:
        # TI2VidTwoStagesPipeline: requires both video and audio guider params
        from ltx_core.components.guiders import MultiModalGuiderParams
        
        video_guider_params = MultiModalGuiderParams(
            cfg_scale=guidance_scale,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=1.0,
            skip_step=0,
            stg_blocks=[29],
        )
        
        # Audio guider params (using video-only mode with modality_scale=1.0)
        audio_guider_params = MultiModalGuiderParams(
            cfg_scale=guidance_scale,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=1.0,
            skip_step=0,
            stg_blocks=[29],
        )
        
        video, audio = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height_padded,
            width=width_padded,
            num_frames=num_frames,
            frame_rate=float(fps),
            num_inference_steps=num_steps,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            images=[],  # No image conditioning for T2V
            tiling_config=tiling_config,
        )
    
    # Encode video to file using ltx-pipelines utility
    encode_video(
        video=video,
        fps=fps,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=output_path,
        video_chunks_number=video_chunks_number,
    )
    
    if MEMORY_CLEANUP_BETWEEN_STAGES:
        cleanup_gpu_memory()
    
    print(f"✅ Text-to-Video complete: {output_path}")
    return os.path.abspath(output_path)


def run_inference_image_to_video(
    job_id: str,
    prompt: str,
    params: Dict[str, Any]
) -> str:
    """
    Image-to-Video generation using LTX-2 FP4 model.
    """
    print(f"🎬 Starting Image-to-Video job: {job_id}")
    
    cleanup_gpu_memory()
    
    # Load input image
    init_image_b64 = params.get("init_image")
    if not init_image_b64:
        raise ValueError("init_image is required for Image-to-Video")
    
    input_image = load_image_from_base64(init_image_b64)
    img_width, img_height = input_image.size
    
    # Parameters
    height = int(params.get("height", img_height))
    width = int(params.get("width", img_width))
    duration = float(params.get("duration", 5.0))
    fps = int(params.get("fps", DEFAULT_FPS))
    seed = int(params.get("seed", random.randint(0, MAX_SEED)))
    pipeline_type = params.get("pipeline_type", "two_stage")
    output_dir = params.get("output_dir", "/app/worker/outputs")
    
    conditioning_strength = float(params.get("conditioning_strength", 1.0))
    
    is_distilled = "distilled" in pipeline_type
    guidance_scale = float(params.get("guidance_scale", 1.0 if is_distilled else 3.0))
    stg_scale = float(params.get("stg_scale", 0.0 if is_distilled else 1.0))
    num_steps = int(params.get("num_inference_steps", 
                               DISTILLED_STEPS_STAGE1 if is_distilled else 40))
    
    negative_prompt = params.get("negative_prompt",
        "worst quality, inconsistent motion, blurry, jittery, distorted")
    
    # Prepare
    height = min(height, MAX_HEIGHT)
    width = min(width, MAX_WIDTH)
    height_padded, width_padded = pad_dimensions_to_32(height, width)
    num_frames = calculate_optimal_frames(duration, fps)
    
    input_image = resize_image_keep_aspect(input_image, height_padded, width_padded)
    
    print(f"📐 Resolution: {width_padded}x{height_padded}, Frames: {num_frames}")
    
    seed_everything(seed)
    
    data = get_ltx2_pipeline(pipeline_type)
    pipeline = data["pipeline"]
    device = data["device"]
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"i2v_{job_id}.mp4")
    
    # Save input image to temp file (ltx-pipelines expects file paths)
    tmp_img_path = f"/tmp/{job_id}_input.png"
    input_image.save(tmp_img_path)
    
    try:
        # Import ltx utilities
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
        from ltx_core.model.video_vae import get_video_chunks_number, TilingConfig
        
        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
        
        print(f"🚀 Generating I2V with seed={seed}...")
        
        # Image conditioning: (path, frame_index, strength)
        images = [(tmp_img_path, 0, conditioning_strength)]
        
        if is_distilled:
            # DistilledPipeline: simpler API
            video, audio = pipeline(
                prompt=prompt,
                seed=seed,
                height=height_padded,
                width=width_padded,
                num_frames=num_frames,
                frame_rate=float(fps),
                images=images,
                tiling_config=tiling_config,
            )
        else:
            # TI2VidTwoStagesPipeline: requires both guider params
            from ltx_core.components.guiders import MultiModalGuiderParams
            
            video_guider_params = MultiModalGuiderParams(
                cfg_scale=guidance_scale,
                stg_scale=stg_scale,
                rescale_scale=0.7,
                modality_scale=1.0,
                skip_step=0,
                stg_blocks=[29],
            )
            
            audio_guider_params = MultiModalGuiderParams(
                cfg_scale=guidance_scale,
                stg_scale=stg_scale,
                rescale_scale=0.7,
                modality_scale=1.0,
                skip_step=0,
                stg_blocks=[29],
            )
            
            video, audio = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                height=height_padded,
                width=width_padded,
                num_frames=num_frames,
                frame_rate=float(fps),
                num_inference_steps=num_steps,
                video_guider_params=video_guider_params,
                audio_guider_params=audio_guider_params,
                images=images,
                tiling_config=tiling_config,
            )
        
        # Encode video to file
        encode_video(
            video=video,
            fps=fps,
            audio=audio,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )
        
    finally:
        if os.path.exists(tmp_img_path):
            os.remove(tmp_img_path)
    
    if MEMORY_CLEANUP_BETWEEN_STAGES:
        cleanup_gpu_memory()
    
    print(f"✅ Image-to-Video complete: {output_path}")
    return os.path.abspath(output_path)


def run_inference_video_to_video(
    job_id: str,
    prompt: str,
    params: Dict[str, Any]
) -> str:
    """Video-to-Video transformation using LTX-2 ICLoraPipeline."""
    print(f"🎬 Starting Video-to-Video job: {job_id}")
    
    cleanup_gpu_memory()
    
    init_video_b64 = params.get("init_video")
    if not init_video_b64:
        raise ValueError("init_video is required for Video-to-Video")
    
    fps = int(params.get("fps", DEFAULT_FPS))
    input_frames = load_video_from_base64(init_video_b64, max_frames=MAX_FRAMES)
    
    if not input_frames:
        raise ValueError("Failed to load input video")
    
    first_frame = input_frames[0]
    img_width, img_height = first_frame.size
    
    height = int(params.get("height", img_height))
    width = int(params.get("width", img_width))
    seed = int(params.get("seed", random.randint(0, MAX_SEED)))
    output_dir = params.get("output_dir", "/app/worker/outputs")
    
    conditioning_strength = float(params.get("conditioning_strength", 0.8))
    
    height = min(height, MAX_HEIGHT)
    width = min(width, MAX_WIDTH)
    height_padded, width_padded = pad_dimensions_to_32(height, width)
    num_frames = calculate_optimal_frames(len(input_frames) / fps, fps)
    
    input_frames = [resize_image_keep_aspect(f, height_padded, width_padded) for f in input_frames]
    
    print(f"📐 Resolution: {width_padded}x{height_padded}, Frames: {num_frames}")
    
    seed_everything(seed)
    
    # ICLoraPipeline for video-to-video
    data = get_ltx2_pipeline("ic_lora")
    pipeline = data["pipeline"]
    
    # Save temp video for video_conditioning
    tmp_video_path = f"/tmp/{job_id}_input.mp4"
    frames_np = np.stack([np.array(f) for f in input_frames])
    export_video(frames_np, tmp_video_path, fps)
    
    # Save first frame for image conditioning
    tmp_img_path = f"/tmp/{job_id}_first_frame.png"
    input_frames[0].save(tmp_img_path)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"v2v_{job_id}.mp4")
    
    try:
        # Import ltx utilities
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
        from ltx_core.model.video_vae import get_video_chunks_number, TilingConfig
        
        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
        
        print(f"🚀 Generating V2V with seed={seed}...")
        
        # ICLoraPipeline.__call__ signature:
        # prompt, seed, height, width, num_frames, frame_rate, 
        # images (list[tuple[str, int, float]]),
        # video_conditioning (list[tuple[str, float]]),
        # enhance_prompt, tiling_config
        video, audio = pipeline(
            prompt=prompt,
            seed=seed,
            height=height_padded,
            width=width_padded,
            num_frames=num_frames,
            frame_rate=float(fps),
            images=[(tmp_img_path, 0, conditioning_strength)],  # First frame as image conditioning
            video_conditioning=[(tmp_video_path, conditioning_strength)],  # Reference video
            tiling_config=tiling_config,
        )
        
        # Encode video to file
        encode_video(
            video=video,
            fps=fps,
            audio=audio,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )
        
    finally:
        if os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)
        if os.path.exists(tmp_img_path):
            os.remove(tmp_img_path)
    
    if MEMORY_CLEANUP_BETWEEN_STAGES:
        cleanup_gpu_memory()
    
    print(f"✅ Video-to-Video complete: {output_path}")
    return os.path.abspath(output_path)


def run_inference_keyframe_interpolation(
    job_id: str,
    prompt: str,
    params: Dict[str, Any]
) -> str:
    """Keyframe interpolation using LTX-2 KeyframeInterpolationPipeline."""
    print(f"🎬 Starting Keyframe Interpolation job: {job_id}")
    
    cleanup_gpu_memory()
    
    keyframes_data = params.get("keyframes", [])
    if len(keyframes_data) < 2:
        raise ValueError("At least 2 keyframes required")
    
    keyframes = []
    for kf in keyframes_data:
        img = load_image_from_base64(kf["image"])
        frame_idx = int(kf.get("frame_index", 0))
        strength = float(kf.get("strength", 1.0))
        keyframes.append({"image": img, "frame_index": frame_idx, "strength": strength})
    
    keyframes.sort(key=lambda x: x["frame_index"])
    
    first_img = keyframes[0]["image"]
    img_width, img_height = first_img.size
    
    height = int(params.get("height", img_height))
    width = int(params.get("width", img_width))
    num_frames = int(params.get("num_frames", 121))
    fps = int(params.get("fps", DEFAULT_FPS))
    seed = int(params.get("seed", random.randint(0, MAX_SEED)))
    output_dir = params.get("output_dir", "/app/worker/outputs")
    
    guidance_scale = float(params.get("guidance_scale", 3.0))
    stg_scale = float(params.get("stg_scale", 1.0))
    num_steps = int(params.get("num_inference_steps", 40))
    
    negative_prompt = params.get("negative_prompt",
        "worst quality, inconsistent motion, blurry, jittery, distorted")
    
    height = min(height, MAX_HEIGHT)
    width = min(width, MAX_WIDTH)
    height_padded, width_padded = pad_dimensions_to_32(height, width)
    num_frames = calculate_optimal_frames(num_frames / fps, fps)
    
    print(f"📐 Resolution: {width_padded}x{height_padded}, Frames: {num_frames}")
    print(f"🖼️ Keyframes: {len(keyframes)} images")
    
    seed_everything(seed)
    
    data = get_ltx2_pipeline("keyframe")
    pipeline = data["pipeline"]
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"kf_{job_id}.mp4")
    
    # Save temp keyframes as files (ltx-pipelines expects file paths)
    keyframe_inputs = []
    for i, kf in enumerate(keyframes):
        tmp_path = f"/tmp/{job_id}_kf_{i}.png"
        resized = resize_image_keep_aspect(kf["image"], height_padded, width_padded)
        resized.save(tmp_path)
        keyframe_inputs.append((tmp_path, kf["frame_index"], kf["strength"]))
    
    try:
        # Import ltx utilities
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
        from ltx_core.model.video_vae import get_video_chunks_number, TilingConfig
        from ltx_core.components.guiders import MultiModalGuiderParams
        
        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
        
        # KeyframeInterpolationPipeline requires both video and audio guider params
        video_guider_params = MultiModalGuiderParams(
            cfg_scale=guidance_scale,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=1.0,
            skip_step=0,
            stg_blocks=[29],
        )
        
        audio_guider_params = MultiModalGuiderParams(
            cfg_scale=guidance_scale,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=1.0,
            skip_step=0,
            stg_blocks=[29],
        )
        
        print(f"🚀 Generating keyframe interpolation with seed={seed}...")
        
        # KeyframeInterpolationPipeline uses "images" parameter (not "keyframe_images")
        video, audio = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height_padded,
            width=width_padded,
            num_frames=num_frames,
            frame_rate=float(fps),
            num_inference_steps=num_steps,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            images=keyframe_inputs,  # List of (path, frame_index, strength) tuples
            tiling_config=tiling_config,
        )
        
        # Encode video to file
        encode_video(
            video=video,
            fps=fps,
            audio=audio,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )
        
    finally:
        for kf_path, _, _ in keyframe_inputs:
            if os.path.exists(kf_path):
                os.remove(kf_path)
    
    if MEMORY_CLEANUP_BETWEEN_STAGES:
        cleanup_gpu_memory()
    
    print(f"✅ Keyframe Interpolation complete: {output_path}")
    return os.path.abspath(output_path)


# -----------------------------
# Unified Entry Point
# -----------------------------
def run_inference_ltx2_video(
    job_id: str,
    prompt: str,
    params: Dict[str, Any]
) -> str:
    """
    Unified entry point for all LTX-2 video generation modes.
    
    Using FP4 model: ltx-2-19b-dev-fp4.safetensors (20 GB)
    Works safely on RTX 4090 (24GB) and RTX 5090 (32GB) - No OOM risk
    
    Args:
        job_id: Unique job identifier
        prompt: Text prompt (keep under 200 words)
        params: Parameters including 'mode'
            - mode: "text-to-video", "image-to-video", "video-to-video", "keyframe-interpolation"
    
    Returns:
        Absolute path to generated video
    """
    mode = params.get("mode", "text-to-video").lower().replace("_", "-")
    
    print(f"🎬 LTX-2 Job {job_id} | Mode: {mode}")
    print(f"   Model: {CHECKPOINT_FILE} (20 GB FP4)")
    
    if mode in ("text-to-video", "t2v"):
        return run_inference_text_to_video(job_id, prompt, params)
    elif mode in ("image-to-video", "i2v"):
        return run_inference_image_to_video(job_id, prompt, params)
    elif mode in ("video-to-video", "v2v"):
        return run_inference_video_to_video(job_id, prompt, params)
    elif mode in ("keyframe-interpolation", "keyframe", "kf"):
        return run_inference_keyframe_interpolation(job_id, prompt, params)
    else:
        raise ValueError(
            f"Unknown mode: {mode}. "
            f"Supported: text-to-video, image-to-video, video-to-video, keyframe-interpolation"
        )


# -----------------------------
# Pipeline Preloading
# -----------------------------
def preload_pipeline(pipeline_type: str = "distilled") -> bool:
    """Preload pipeline for faster first inference."""
    print(f"🔄 Preloading LTX-2 {pipeline_type} pipeline (FP4 model)...")
    start = time.time()
    
    try:
        get_ltx2_pipeline(pipeline_type)
        elapsed = time.time() - start
        print(f"✅ Pipeline preloaded in {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"❌ Pipeline preload failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# -----------------------------
# CLI Test
# -----------------------------
if __name__ == "__main__":
    import uuid
    
    job_id = str(uuid.uuid4())[:8]
    
    prompt = """A cinematic drone shot slowly ascending over misty mountains at golden hour. 
    The camera moves smoothly upward, revealing snow-capped peaks emerging through layers 
    of clouds. Warm sunlight filters through, casting long shadows across the valleys below."""
    
    params = {
        "mode": "text-to-video",
        "height": 512,
        "width": 768,
        "duration": 3.0,
        "pipeline_type": "distilled",
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "output_dir": "./outputs",
    }
    
    print("🚀 Running LTX-2 Text-to-Video test (FP4 model)...")
    try:
        output = run_inference_ltx2_video(job_id, prompt, params)
        print(f"✅ Test complete: {output}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
