"""
SkyReels-V3 — video extension only (single-shot + shot-switching).

Uses the cloned repo at work_space/SkyReels-V3 (or SKYREELS_REPO_ROOT).
Weights: Hugging Face `Skywork/SkyReels-V3-Video-Extension` via `download_model`,
or a local path via --model_id / SKYREELS_MODEL_ID.

Install (in repo): pip install -r requirements.txt
Recommended: Python 3.12+, CUDA 12.8+ per upstream.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from typing import Literal, Optional, Union

os.environ["DIFFUSERS_DISABLE_TORCHAO"] = "1"
# ---------------------------------------------------------------------------
# Repo on sys.path (package root = SkyReels-V3, not skyreels_v3/)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_REPO = os.path.join(_SCRIPT_DIR, "SkyReels-V3")
REPO_ROOT = os.environ.get("SKYREELS_REPO_ROOT", _DEFAULT_REPO)
SKYREELS_GIT_URL = "https://github.com/SkyworkAI/SkyReels-V3.git"
SKYREELS_AUTO_CLONE = os.environ.get("SKYREELS_AUTO_CLONE", "true").lower() in (
    "1",
    "true",
    "yes",
)


def ensure_repo(repo_path: str = REPO_ROOT) -> None:
    abs_repo = os.path.abspath(repo_path)
    if os.path.isdir(abs_repo) and os.path.isdir(os.path.join(abs_repo, "skyreels_v3")):
        return
    parent = os.path.dirname(abs_repo)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    if not SKYREELS_AUTO_CLONE:
        raise RuntimeError(
            f"SkyReels-V3 not found at {abs_repo}. Clone:\n"
            f"  git clone {SKYREELS_GIT_URL} {abs_repo}\n"
            "Or set SKYREELS_AUTO_CLONE=1."
        )
    print(f"[SkyReels] Cloning {SKYREELS_GIT_URL} -> {abs_repo} ...")
    try:
        subprocess.run(
            ["git", "clone", SKYREELS_GIT_URL, abs_repo],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError("git not on PATH; clone the repo manually.") from e
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or "").strip()
        raise RuntimeError(f"git clone failed: {err or e}") from e
    print("[SkyReels] Clone finished.")


ensure_repo()

_repo_abs = os.path.abspath(REPO_ROOT)
if _repo_abs not in sys.path:
    sys.path.insert(0, _repo_abs)

import imageio
import numpy as np
import torch
import wget

from skyreels_v3.config import SHOT_NUM_CONDITION_FRAMES_MAP
from skyreels_v3.modules import download_model
from skyreels_v3.pipelines import (
    ShotSwitchingExtensionPipeline,
    SingleShotExtensionPipeline,
)

DEFAULT_VIDEO_EXTENSION_MODEL_ID = "Skywork/SkyReels-V3-Video-Extension"

ExtensionTaskType = Literal["single_shot_extension", "shot_switching_extension"]


def maybe_download(path_or_url: str, save_dir: str = "input_video") -> str:
    """Local path unchanged; URLs downloaded under save_dir."""
    if os.path.exists(path_or_url):
        return path_or_url
    filename = path_or_url.split("/")[-1]
    local_path = os.path.join(save_dir, filename)
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    if os.path.exists(local_path):
        return local_path
    logging.info("Downloading input to %s", local_path)
    wget.download(path_or_url, local_path)
    if not os.path.exists(local_path):
        raise RuntimeError(f"Failed to download: {path_or_url}")
    logging.info("Download finished: %s", local_path)
    return local_path


def save_video_np(
    video_frames: np.ndarray,
    output_path: str,
    *,
    fps: int = 24,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    imageio.mimwrite(
        output_path,
        video_frames,
        fps=fps,
        quality=8,
        output_params=["-loglevel", "error"],
    )
    return output_path


def resolve_model_path(model_id: Optional[str]) -> str:
    mid = model_id or os.environ.get(
        "SKYREELS_MODEL_ID", DEFAULT_VIDEO_EXTENSION_MODEL_ID
    )
    return download_model(mid)


def build_extension_pipeline(
    task_type: ExtensionTaskType,
    model_path: str,
    *,
    device: str = "cuda",
    offload: bool = True,
    low_vram: bool = False,
    use_usp: bool = False,
    weight_dtype: torch.dtype = torch.bfloat16,
) -> Union[SingleShotExtensionPipeline, ShotSwitchingExtensionPipeline]:
    if task_type == "single_shot_extension":
        return SingleShotExtensionPipeline(
            model_path=model_path,
            device=device,
            weight_dtype=weight_dtype,
            use_usp=use_usp,
            offload=offload,
            low_vram=low_vram,
        )
    if task_type == "shot_switching_extension":
        return ShotSwitchingExtensionPipeline(
            model_path=model_path,
            device=device,
            weight_dtype=weight_dtype,
            use_usp=use_usp,
            offload=offload,
            low_vram=low_vram,
        )
    raise ValueError(f"Unknown task_type: {task_type}")


def validate_duration(task_type: ExtensionTaskType, duration: int) -> None:
    if task_type == "single_shot_extension":
        if not 5 <= duration <= 30:
            raise ValueError(
                "single_shot_extension: duration must be between 5 and 30 (seconds)."
            )
    else:
        if duration not in SHOT_NUM_CONDITION_FRAMES_MAP:
            raise ValueError(
                "shot_switching_extension: duration must be one of "
                f"{sorted(SHOT_NUM_CONDITION_FRAMES_MAP.keys())} (seconds, max 5)."
            )


def run_video_extension(
    input_video: str,
    prompt: str,
    *,
    task_type: ExtensionTaskType = "single_shot_extension",
    duration: int = 5,
    seed: int = 42,
    resolution: str = "720P",
    model_id: Optional[str] = None,
    output_path: Optional[str] = None,
    device: str = "cuda",
    offload: bool = True,
    low_vram: bool = False,
    use_usp: bool = False,
    input_download_dir: str = "input_video",
) -> tuple[str, np.ndarray]:
    """
    End-to-end video extension: download inputs if needed, load model, extend, save MP4.

    Returns (output_path, video_frames_numpy).
    """
    validate_duration(task_type, duration)

    local_video = maybe_download(input_video, input_download_dir)
    model_path = resolve_model_path(model_id)

    pipe = build_extension_pipeline(
        task_type,
        model_path,
        device=device,
        offload=offload,
        low_vram=low_vram,
        use_usp=use_usp,
    )

    video_out = pipe.extend_video(
        local_video,
        prompt,
        duration,
        seed,
        fps=24,
        resolution=resolution,
    )

    out = output_path or os.path.join(
        "result",
        task_type,
        f"{seed}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.mp4",
    )
    save_video_np(video_out, out, fps=24)
    return out, video_out


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )


def main() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser(
        description="SkyReels V3 — video extension (single-shot / shot-switching)"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="single_shot_extension",
        choices=["single_shot_extension", "shot_switching_extension"],
        help="single_shot_extension: 5–30s; shot_switching_extension: cinematic cuts, duration 2–5 only",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="HF repo id or local checkpoint path (default: Skywork/SkyReels-V3-Video-Extension)",
    )
    parser.add_argument(
        "--input_video",
        type=str,
        default="https://skyreels-api.oss-accelerate.aliyuncs.com/examples/video_extension/test.mp4",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A man is making his way forward slowly, leaning on a white cane to prop himself up.",
    )
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resolution",
        type=str,
        default="720P",
        choices=["480P", "540P", "720P"],
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .mp4 path (default: result/<task_type>/...)",
    )
    parser.add_argument(
        "--no_offload",
        action="store_true",
        help="Keep weights on GPU (default: offload enabled for 14B extension model)",
    )
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument(
        "--use_usp",
        action="store_true",
        help="Multi-GPU xDiT USP; launch with torchrun --nproc_per_node=N",
    )
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    if args.use_usp and args.low_vram:
        parser.error("Cannot combine --use_usp and --low_vram")
    # Pipeline sets offload |= low_vram internally; we mirror intent here.
    offload = args.low_vram or not args.no_offload

    t0 = time.perf_counter()
    out_path, _ = run_video_extension(
        args.input_video,
        args.prompt,
        task_type=args.task_type,
        duration=args.duration,
        seed=args.seed,
        resolution=args.resolution,
        model_id=args.model_id,
        output_path=args.output,
        device=args.device,
        offload=offload,
        low_vram=args.low_vram,
        use_usp=args.use_usp,
    )
    logging.info("Saved %s in %.2fs", out_path, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
