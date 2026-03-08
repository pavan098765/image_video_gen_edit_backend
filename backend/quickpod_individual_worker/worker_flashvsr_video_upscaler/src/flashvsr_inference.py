#!/usr/bin/env python3
"""
FlashVSR-Pro inference wrapper.
Calls infer.py via subprocess; uses sys.executable (same Python as our app, flashvsr env).
"""

import os
import sys
import subprocess
from typing import Dict, Any

WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", "/workspace")
FLASHVSR_PRO_DIR = os.environ.get("FLASHVSR_PRO_DIR", "/workspace/FlashVSR-Pro")


def run_inference_flashvsr_video_upscaler(
    job_id: str,
    prompt: str,
    params: Dict[str, Any],
) -> str:
    """
    Run FlashVSR-Pro video super-resolution via subprocess.

    Args:
        job_id: Job identifier
        prompt: Not used (kept for API compatibility)
        params: Must include input_video (path). Optional: mode, scale, keep_audio, tile_dit, etc.

    Returns:
        Absolute path to output video file
    """
    input_video = params.get("input_video")
    if not input_video:
        raise ValueError("input_video parameter is required")
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

    output_dir = params.get("output_dir", WORKSPACE_DIR)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"upscaled_{job_id}.mp4")

    cmd = [
        sys.executable,
        "infer.py",
        "-i", input_video,
        "-o", out_path,
        "--mode", str(params.get("mode", "tiny")),
        "--scale", str(params.get("scale", 2.0)),
    ]
    if params.get("keep_audio", True):
        cmd.append("--keep-audio")
    if params.get("tile_dit"):
        cmd.append("--tile-dit")
    if params.get("tile_vae"):
        cmd.append("--tile-vae")
    if params.get("color_fix", True):
        cmd.append("--color-fix")
    if params.get("tile_size") is not None:
        cmd.extend(["--tile-size", str(params["tile_size"])])
    if params.get("overlap") is not None:
        cmd.extend(["--overlap", str(params["overlap"])])
    if params.get("quality") is not None:
        cmd.extend(["--quality", str(params["quality"])])
    if params.get("fps") is not None:
        cmd.extend(["--fps", str(params["fps"])])
    if params.get("seed") is not None:
        cmd.extend(["--seed", str(params["seed"])])

    subprocess.run(cmd, cwd=FLASHVSR_PRO_DIR, check=True)

    if not os.path.exists(out_path):
        raise FileNotFoundError(f"FlashVSR output not found: {out_path}")
    return os.path.abspath(out_path)
