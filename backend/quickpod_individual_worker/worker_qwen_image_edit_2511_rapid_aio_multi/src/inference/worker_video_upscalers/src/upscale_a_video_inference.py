# upscale_video.py

import os
import base64
from typing import Dict, Any
from pathlib import Path
import shutil
import subprocess

# -------------------------
# Config / paths
# -------------------------
REPO_DIR = os.path.join("/workspace/Upscale-A-Video")  # Path to cloned repo
DEFAULT_OUTPUT_DIR = "./results"
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def _save_video_from_base64(video_b64: str, tmp_path: str) -> str:
    """Decode base64 video and save as temporary file."""
    decoded = base64.b64decode(video_b64)
    with open(tmp_path, "wb") as f:
        f.write(decoded)
    return tmp_path

# -------------------------
# Main upscale function
# -------------------------
def run_video_upscale(job_id: str, params: Dict[str, Any]) -> str:
    """
    params:
        - image: str (base64) or video path
        - output_dir: str
        - optional args: color_fix, use_video_vae, n, g, s, p
    """
    input_video = params.get("video") or params.get("image")
    if input_video is None:
        raise ValueError("video or image parameter is required")

    output_dir = params.get("output_dir", DEFAULT_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"upscaled_{job_id}.mp4")

    # -------------------------
    # Prepare input video
    # -------------------------
    if isinstance(input_video, str) and Path(input_video).exists():
        input_path = input_video
    else:
        tmp_path = os.path.join(output_dir, f"tmp_input_{job_id}.mp4")
        input_path = _save_video_from_base64(input_video, tmp_path)

    # -------------------------
    # Construct command
    # -------------------------
    cmd = [
        "python", os.path.join(REPO_DIR, "inference_upscale_a_video.py"),
        "-i", str(input_path),
        "-o", str(output_dir),
    ]

    # Optional Upscale-A-Video params
    if "n" in params:
        cmd.extend(["-n", str(params["n"])])
    if "g" in params:
        cmd.extend(["-g", str(params["g"])])
    if "s" in params:
        cmd.extend(["-s", str(params["s"])])
    if "p" in params:
        cmd.extend(["-p", ",".join(map(str, params["p"]))])
    if params.get("use_video_vae", False):
        cmd.append("--use_video_vae")
    if "color_fix" in params:
        cmd.extend(["--color_fix", str(params["color_fix"])])

    # -------------------------
    # Execute Upscale-A-Video
    # -------------------------
    print(f"🔹 Running Upscale-A-Video: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # The script saves output as <input_name>_upscaled.mp4
    input_filename = Path(input_path).stem
    expected_output = os.path.join(output_dir, f"{input_filename}_upscaled.mp4")

    # Rename to standard output path
    shutil.move(expected_output, out_path)

    # Remove temp input if created
    if 'tmp_path' in locals() and Path(tmp_path).exists():
        os.remove(tmp_path)

    return out_path

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    job_id = "test_video"
    input_path = "./inputs/old_video_1.mp4"

    with open(input_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")

    params = {
        "video": video_b64,      # can also pass path
        "output_dir": "./results",
        "n": 150,
        "g": 6,
        "s": 30,
        "use_video_vae": True,
        "color_fix": "AdaIn",
    }

    out = run_video_upscale(job_id, params)
    print("✅ Upscaled video saved at:", out)
