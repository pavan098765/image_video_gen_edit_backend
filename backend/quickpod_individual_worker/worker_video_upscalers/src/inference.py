#!/usr/bin/env python3
"""
Inference module for Video Upscaler worker.
Supports RIFE frame interpolation and Real-ESRGAN upscaling.
"""

import os
import time
import torch
import traceback
import numpy as np
import cv2
from typing import Dict, Any, Optional
from PIL import Image
from diffusers.utils import export_to_video

# Video post-processing imports
try:
    from video_postprocessing import (
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
torch.set_float32_matmul_precision('medium')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# Configuration
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")


def _realesrgan_video_ckpt_name(upscale_factor: Any) -> str:
    """rockapaper/realesrgan baked weights (x2plus / x4plus)."""
    try:
        u = int(upscale_factor)
    except (TypeError, ValueError):
        u = 4
    return "RealESRGAN_x2plus.pth" if u <= 2 else "RealESRGAN_x4plus.pth"


# Default parameters
DEFAULT_MAX_FPS = 60
DEFAULT_MAX_UPSCALE = 4


def load_video_frames(video_path: str):
    """Load all frames from a video file.
    
    Returns:
        frames: List of numpy arrays [h, w, c] in RGB format
        fps: Original FPS
        width: Original width
        height: Original height
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    return frames, fps, width, height


def unload_models():
    """Free any cached models and reclaim GPU memory."""
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_inference_video_upscaler(
    job_id: str,
    prompt: str,
    params: Dict[str, Any]
) -> str:
    """
    Run video upscaling and/or interpolation inference.
    
    Args:
        job_id: Job identifier
        prompt: Not used for video upscaling (kept for API compatibility)
        params: Inference parameters:
            - input_video: Path to input video file (required)
            - enable_interpolation: Enable RIFE frame interpolation (default: False)
            - target_fps: Target FPS for interpolation (max 60, default: None = double original)
            - enable_upscaling: Enable Real-ESRGAN upscaling (default: False)
            - upscale_factor: Upscale factor 2, 4 (max 4, default: 2)
            - postprocess_interpolate_first: If True, interpolate first then upscale (default: False)
            - output_dir: Output directory (default: /workspace)
    
    Returns:
        Path to output video file
    """
    print(f"[{job_id}] Starting video upscaling/interpolation job")
    
    # Pre-inference GPU cleanup
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # Extract parameters
    input_video = params.get("input_video")
    if not input_video:
        raise ValueError("input_video parameter is required")

    # Accept presigned http(s) URL OR a local path. Controller ships URLs; workers must
    # download to a temp file before the rest of the pipeline (which assumes a local path).
    if isinstance(input_video, str) and (
        input_video.startswith("http://") or input_video.startswith("https://")
    ):
        import tempfile
        import requests
        url = input_video
        fd, tmp_path = tempfile.mkstemp(prefix=f"upsc_{job_id}_", suffix=".mp4")
        os.close(fd)
        print(f"[{job_id}] Downloading input video {url} -> {tmp_path}")
        with requests.get(url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(tmp_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
        input_video = tmp_path

    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")
    
    enable_interpolation = params.get("enable_interpolation", False)
    target_fps = params.get("target_fps", None)
    if target_fps is not None:
        target_fps = min(float(target_fps), DEFAULT_MAX_FPS)  # Cap at 60 FPS
    
    enable_upscaling = params.get("enable_upscaling", False)
    upscale_factor = int(params.get("upscale_factor", 2))
    upscale_factor = min(upscale_factor, DEFAULT_MAX_UPSCALE)  # Cap at 4x
    if upscale_factor not in [2, 4]:
        raise ValueError(f"upscale_factor must be 2 or 4, got {upscale_factor}")
    
    postprocess_interpolate_first = params.get("postprocess_interpolate_first", False)
    output_dir = params.get("output_dir", "/workspace")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load video
    print(f"[{job_id}] Loading video: {input_video}")
    frames, original_fps, width, height = load_video_frames(input_video)
    print(f"[{job_id}] Loaded {len(frames)} frames, {width}x{height} @ {original_fps:.2f}fps")
    
    # Convert to numpy array [f, h, w, c]
    frames_np = np.array(frames, dtype=np.uint8)
    
    # Post-processing
    postprocess_time = 0.0
    final_fps = original_fps
    final_output = frames_np
    final_height = height
    final_width = width
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if POSTPROCESSING_AVAILABLE and (enable_interpolation or enable_upscaling):
        postprocess_start = time.time()
        
        # Load post-processing models from local paths
        rife_model = None
        upscale_model = None
        
        if enable_interpolation:
            if check_rife_code_available():
                rife_path = os.path.join(MODELS_DIR, "rife")
                if os.path.exists(rife_path) and os.listdir(rife_path):
                    rife_model = load_rife_model(rife_path)
                else:
                    print(f"⚠️  RIFE model not found at {rife_path}, interpolation disabled")
            else:
                print(f"⚠️  RIFE code not available, interpolation disabled")
        
        if enable_upscaling:
            realesrgan_model_file = os.path.join(
                MODELS_DIR,
                "realesrgan",
                _realesrgan_video_ckpt_name(upscale_factor),
            )
            if os.path.exists(realesrgan_model_file):
                try:
                    upscale_model = load_sd_upscale(realesrgan_model_file, device)
                except Exception as e:
                    print(f"⚠️  Real-ESRGAN model failed to load: {str(e)[:100]}")
                    upscale_model = None
            else:
                print(f"⚠️  Real-ESRGAN {upscale_factor}x model not found at {realesrgan_model_file}, upscaling disabled")
        
        # Apply post-processing based on order
        if postprocess_interpolate_first:
            # Interpolate → Upscale
            if enable_interpolation and rife_model:
                print(f"[{job_id}] Interpolating frames with RIFE...")
                try:
                    final_output, actual_fps = apply_rife_interpolation(
                        rife_model, final_output, device, original_fps=original_fps, target_fps=target_fps
                    )
                    final_fps = actual_fps
                except Exception as e:
                    print(f"⚠️  RIFE interpolation failed: {str(e)[:100]}")
            
            if enable_upscaling and upscale_model:
                print(f"[{job_id}] Upscaling video with Real-ESRGAN ({upscale_factor}x)...")
                try:
                    final_output = apply_realesrgan_upscaling(upscale_model, final_output, device)
                    final_height = height * upscale_factor
                    final_width = width * upscale_factor
                except Exception as e:
                    print(f"⚠️  Upscaling failed: {str(e)[:100]}")
        else:
            # Upscale → Interpolate (default)
            if enable_upscaling and upscale_model:
                print(f"[{job_id}] Upscaling video with Real-ESRGAN ({upscale_factor}x)...")
                try:
                    final_output = apply_realesrgan_upscaling(upscale_model, final_output, device)
                    final_height = height * upscale_factor
                    final_width = width * upscale_factor
                except Exception as e:
                    print(f"⚠️  Upscaling failed: {str(e)[:100]}")
            
            if enable_interpolation and rife_model:
                print(f"[{job_id}] Interpolating frames with RIFE...")
                try:
                    final_output, actual_fps = apply_rife_interpolation(
                        rife_model, final_output, device, original_fps=original_fps, target_fps=target_fps
                    )
                    final_fps = actual_fps
                except Exception as e:
                    print(f"⚠️  RIFE interpolation failed: {str(e)[:100]}")
        
        postprocess_time = time.time() - postprocess_start
    
    # Convert numpy frames to PIL Images for export
    # Post-processing functions return float32 [0, 1] arrays, need to convert to uint8 [0, 255]
    print(f"[{job_id}] Converting {len(final_output) if isinstance(final_output, (list, np.ndarray)) else 'unknown'} frames to PIL Images...")
    
    if isinstance(final_output, np.ndarray):
        print(f"[{job_id}] final_output shape: {final_output.shape}, dtype: {final_output.dtype}")
        
        # Ensure correct dtype and range
        if final_output.dtype == np.float32 or final_output.dtype == np.float64:
            # Clamp to [0, 1] and convert to uint8
            final_output = np.clip(final_output, 0, 1)
            final_output = (final_output * 255).astype(np.uint8)
        elif final_output.dtype != np.uint8:
            # Convert other dtypes to uint8
            final_output = final_output.astype(np.uint8)
        
        # Ensure correct shape: [f, h, w, c]
        if len(final_output.shape) == 3:
            # Single frame [h, w, c] - add frame dimension
            final_output = final_output[np.newaxis, ...]
        elif len(final_output.shape) == 2:
            # Single grayscale frame [h, w] - add frame and channel dimensions
            final_output = final_output[np.newaxis, ..., np.newaxis]  # [1, h, w, 1]
            # Convert to RGB
            final_output = np.repeat(final_output, 3, axis=-1)  # [1, h, w, 3]
        elif len(final_output.shape) == 4 and final_output.shape[0] == 0:
            # Empty array - this shouldn't happen but handle gracefully
            raise ValueError(f"Empty frame array with shape {final_output.shape}")
        elif len(final_output.shape) not in [3, 4]:
            raise ValueError(f"Unexpected final_output shape: {final_output.shape}, expected [f, h, w, c] or [h, w, c]")
        
        # Convert each frame to PIL Image
        pil_frames = []
        for i in range(final_output.shape[0]):
            frame = final_output[i]  # [h, w, c] or [h, w]
            
            # Handle edge case: if frame is too small (like 1x1x3), skip it with warning
            if frame.size == 0:
                print(f"[{job_id}] ⚠️  Skipping empty frame {i}")
                continue
            
            # Ensure frame is 2D or 3D with correct shape
            if len(frame.shape) == 3:
                if frame.shape[2] == 3:
                    # RGB frame [h, w, 3]
                    if frame.shape[0] < 1 or frame.shape[1] < 1:
                        print(f"[{job_id}] ⚠️  Skipping invalid frame {i} with shape {frame.shape}")
                        continue
                    pil_frames.append(Image.fromarray(frame, mode='RGB'))
                elif frame.shape[2] == 1:
                    # Single channel [h, w, 1] - convert to RGB
                    frame = frame.squeeze(2)  # [h, w]
                    frame_rgb = np.stack([frame] * 3, axis=-1)  # [h, w, 3]
                    pil_frames.append(Image.fromarray(frame_rgb, mode='RGB'))
                else:
                    raise ValueError(f"Unexpected frame shape: {frame.shape}, expected [h, w, 3] or [h, w, 1]")
            elif len(frame.shape) == 2:
                # Grayscale frame [h, w] - convert to RGB
                if frame.shape[0] < 1 or frame.shape[1] < 1:
                    print(f"[{job_id}] ⚠️  Skipping invalid frame {i} with shape {frame.shape}")
                    continue
                frame_rgb = np.stack([frame] * 3, axis=-1)  # [h, w, 3]
                pil_frames.append(Image.fromarray(frame_rgb, mode='RGB'))
            else:
                raise ValueError(f"Unexpected frame shape: {frame.shape}, expected [h, w, 3] or [h, w]")
        
        print(f"[{job_id}] Converted {len(pil_frames)} frames to PIL Images")
    else:
        # Fallback: assume it's already a list or handle other types
        pil_frames = []
        for i, frame in enumerate(final_output):
            if isinstance(frame, np.ndarray):
                # Convert numpy array to PIL
                if frame.dtype == np.float32 or frame.dtype == np.float64:
                    frame = np.clip(frame, 0, 1)
                    frame = (frame * 255).astype(np.uint8)
                elif frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                
                # Handle shape
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    pil_frames.append(Image.fromarray(frame, mode='RGB'))
                elif len(frame.shape) == 2:
                    frame_rgb = np.stack([frame] * 3, axis=-1)
                    pil_frames.append(Image.fromarray(frame_rgb, mode='RGB'))
                else:
                    print(f"[{job_id}] ⚠️  Skipping frame {i} with unexpected shape: {frame.shape}")
            else:
                # Assume it's already a PIL Image
                pil_frames.append(frame)
    
    # Export video
    out_path = os.path.join(output_dir, f"upscaled_{job_id}.mp4")
    export_start_time = time.time()
    export_to_video(pil_frames, out_path, fps=final_fps)
    export_time = time.time() - export_start_time
    
    total_time = postprocess_time + export_time
    print(f"[{job_id}] ✓ Complete: {total_time:.1f}s (post: {postprocess_time:.1f}s, export: {export_time:.1f}s) | {final_width}x{final_height} @ {final_fps:.2f}fps")
    
    # Post-inference GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    return os.path.abspath(out_path)
