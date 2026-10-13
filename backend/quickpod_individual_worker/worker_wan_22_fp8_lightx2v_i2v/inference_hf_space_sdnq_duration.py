#!/usr/bin/env python3
"""
WAN 2.2 I2V — SDNQ + HF Space–aligned scheduler, flow shift, and duration→num_frames.

- Model: ``rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_Multiscene_SVI_sdnq_fp8_svd_r64``
  (override with env ``HF_REPO_WAN_I2V`` or ``REPO_ID``).
- Duration logic matches ``wan2-2-fp8da-aoti-preview2/app.py``: FIXED_FPS=16,
  inner frame count clamped to [MIN_FRAMES_MODEL, MAX_FRAMES_MODEL], then +1
  → up to **161** latent frames (vs a fixed 81).
- Scheduler rebuild matches the Space pattern (``FlowMatchEulerDiscrete`` uses
  ``shift``; others use ``flow_shift``), then scheduler is restored after each run.

Install (minimal): pip install git+https://github.com/Disty0/sdnq
Plus diffusers / torch / etc. from this worker's requirements.txt.

Run: ``python inference_hf_space_sdnq_duration.py`` (expects ``d1.jpeg`` next to this file).

**Tokenizer:** the rockapaper Hub tree has ``tokenizer/tokenizer.json`` but no ``spiece.model``.
Diffusers would load slow ``T5Tokenizer`` per ``model_index.json`` and SentencePiece fails with
``TypeError: not a string``. This script passes ``T5TokenizerFast`` into ``from_pretrained``.
"""

from __future__ import annotations

import copy
import gc
import inspect
import os
import types
from typing import Any, Dict, Optional

import diffusers
import numpy as np
import torch
from diffusers import (
    DEISMultistepScheduler,
    DPMSolverMultistepInverseScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    FlowMatchEulerDiscreteScheduler,
    SASolverScheduler,
    UniPCMultistepScheduler,
)
from diffusers.utils import export_to_video, load_image
from PIL import Image
from transformers import T5TokenizerFast

from sdnq import SDNQConfig  # noqa: F401 — registers SDNQ with diffusers

# --- HF Space constants (wan2-2-fp8da-aoti-preview2 / app.py) ---

MODEL_REPO = os.getenv(
    "HF_REPO_WAN_I2V",
    os.getenv(
        "REPO_ID",
        "rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_Multiscene_SVI_sdnq_fp8_svd_r64",
    ),
)

FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 160

SCHEDULER_MAP: Dict[str, type] = {
    "FlowMatchEulerDiscrete": FlowMatchEulerDiscreteScheduler,
    "SASolver": SASolverScheduler,
    "DEISMultistep": DEISMultistepScheduler,
    "DPMSolverMultistepInverse": DPMSolverMultistepInverseScheduler,
    "UniPCMultistep": UniPCMultistepScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "DPMSolverSinglestep": DPMSolverSinglestepScheduler,
}

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def get_num_frames_from_duration(duration_seconds: float) -> int:
    """Match HF Space ``get_num_frames``: 1 + clip(round(duration * 16), 8..160)."""
    inner = int(round(float(duration_seconds) * FIXED_FPS))
    inner = int(np.clip(inner, MIN_FRAMES_MODEL, MAX_FRAMES_MODEL))
    return 1 + inner


def resize_area_480p_like_sample(
    image: Image.Image,
    pipe: diffusers.WanImageToVideoPipeline,
) -> Image.Image:
    """SDNQ sample style: max area 480×832, dimensions aligned to VAE patch grid."""
    max_area = 480 * 832
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    return image.resize((width, height), Image.LANCZOS)


def _patch_wan_i2v_prepare_latents_device_alignment(pipe: Any) -> None:
    """Avoid CPU/GPU cat mismatch when ``enable_model_cpu_offload`` is on."""
    if getattr(pipe, "_wan_prepare_latents_device_patch", False):
        return
    _orig = pipe.prepare_latents

    def _fixed_prepare_latents(self: Any, *args: Any, **kwargs: Any):
        result = _orig(*args, **kwargs)
        if len(result) == 3:
            latents, condition, first_frame_mask = result
            if condition.device != latents.device:
                condition = condition.to(latents.device)
            if first_frame_mask.device != latents.device:
                first_frame_mask = first_frame_mask.to(latents.device)
            return latents, condition, first_frame_mask
        latents, condition = result
        if condition.device != latents.device:
            condition = condition.to(latents.device)
        return latents, condition

    pipe.prepare_latents = types.MethodType(_fixed_prepare_latents, pipe)
    pipe._wan_prepare_latents_device_patch = True


def _apply_space_scheduler(
    pipe: diffusers.WanImageToVideoPipeline,
    original_scheduler: Any,
    scheduler_name: str,
    flow_shift: float,
) -> None:
    """Mirror HF Space ``run_inference`` (app.py): swap scheduler + flow shift when needed."""
    scheduler_class = SCHEDULER_MAP.get(scheduler_name)
    if scheduler_class is None:
        raise ValueError(
            f"Unknown scheduler {scheduler_name!r}. Choices: {list(SCHEDULER_MAP.keys())}"
        )
    cfg = pipe.scheduler.config
    class_name = getattr(cfg, "_class_name", type(pipe.scheduler).__name__)
    # Same predicate as the Space (``flow_shift`` vs get("flow_shift", "shift") quirks included).
    if (
        scheduler_class.__name__ != class_name
        or flow_shift != cfg.get("flow_shift", "shift")
    ):
        config = copy.deepcopy(original_scheduler.config)
        if scheduler_class == FlowMatchEulerDiscreteScheduler:
            config["shift"] = flow_shift
        else:
            config["flow_shift"] = flow_shift
        pipe.scheduler = scheduler_class.from_config(config)


def _build_pipe_call_kwargs(
    pipe: diffusers.WanImageToVideoPipeline,
    *,
    image: Image.Image,
    last_image: Optional[Image.Image],
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_frames: int,
    guidance_scale: float,
    guidance_scale_2: float,
    num_inference_steps: int,
    generator: torch.Generator,
) -> Dict[str, Any]:
    sig = inspect.signature(pipe.__call__)
    params = set(sig.parameters.keys())
    kw: Dict[str, Any] = {
        "image": image,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "generator": generator,
    }
    if "guidance_scale_2" in params:
        kw["guidance_scale_2"] = guidance_scale_2
    if last_image is not None and "last_image" in params:
        kw["last_image"] = last_image
    return kw


def load_pipe() -> tuple[diffusers.WanImageToVideoPipeline, Any]:
    print(f"Loading WanImageToVideoPipeline from {MODEL_REPO} …", flush=True)
    # This repo ships ``tokenizer.json`` (Rust "tokenizers" backend) but no ``spiece.model``.
    # ``model_index.json`` still names ``T5Tokenizer`` (slow); that path calls SentencePiece with a
    # missing file → ``TypeError: not a string``. Load ``T5TokenizerFast`` from ``tokenizer.json``.
    print("Loading T5TokenizerFast from tokenizer/ …", flush=True)
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_REPO, subfolder="tokenizer")
    pipe = diffusers.WanImageToVideoPipeline.from_pretrained(
        MODEL_REPO,
        torch_dtype=torch.bfloat16,
        tokenizer=tokenizer,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        pipe.enable_model_cpu_offload()
        _patch_wan_i2v_prepare_latents_device_alignment(pipe)
    else:
        pipe = pipe.to(device)

    original_scheduler = copy.deepcopy(pipe.scheduler)
    return pipe, original_scheduler


def run_i2v(
    *,
    init_image: Image.Image,
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    duration_seconds: float = 5.0,
    num_inference_steps: int = 40,
    guidance_scale: float = 3.5,
    guidance_scale_2: float = 1.0,
    seed: int = 0,
    scheduler_name: str = "UniPCMultistep",
    flow_shift: float = 6.0,
    last_image: Optional[Image.Image] = None,
    pipe: Optional[diffusers.WanImageToVideoPipeline] = None,
    original_scheduler: Any = None,
    output_path: str = "i2v_output.mp4",
    export_fps: Optional[int] = None,
) -> str:
    """
    Run one I2V job. If ``pipe`` is None, loads from hub (slow). Pass cached pipe for repeats.

    ``duration_seconds`` maps to ``num_frames`` like the HF Space (max 161 frames).
    ``export_fps`` defaults to FIXED_FPS (16). Space RIFE fluidity is not included here.
    """
    own_pipe = pipe is None
    if own_pipe:
        pipe, original_scheduler = load_pipe()
    assert original_scheduler is not None

    num_frames = get_num_frames_from_duration(duration_seconds)
    image = resize_area_480p_like_sample(init_image, pipe)
    width, height = image.size

    gen_dev = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=gen_dev).manual_seed(int(seed))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"Generating: {width}x{height}, num_frames={num_frames} "
        f"(duration={duration_seconds}s @ {FIXED_FPS}fps cap), "
        f"scheduler={scheduler_name!r}, flow_shift={flow_shift}, steps={num_inference_steps}",
        flush=True,
    )

    try:
        _apply_space_scheduler(pipe, original_scheduler, scheduler_name, flow_shift)
        call_kw = _build_pipe_call_kwargs(
            pipe,
            image=image,
            last_image=last_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=float(guidance_scale),
            guidance_scale_2=float(guidance_scale_2),
            num_inference_steps=int(num_inference_steps),
            generator=generator,
        )
        with torch.inference_mode():
            out = pipe(**call_kw)
        frames = out.frames[0]
    finally:
        pipe.scheduler = original_scheduler

    fps = int(export_fps) if export_fps is not None else FIXED_FPS
    export_to_video(frames, output_path, fps=fps)
    print(f"Wrote {output_path}", flush=True)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    _input_path = os.path.join(_here, "d1.jpeg")
    if not os.path.isfile(_input_path):
        raise FileNotFoundError(
            f"Expected test image at {_input_path}. Place d1.jpeg beside this script."
        )

    _img = load_image(_input_path)
    _prompt = (
        "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
        "The fluffy-furred feline gazes directly at the camera with a relaxed expression."
    )

    _pipe, _orig_sched = load_pipe()
    _out = os.path.join(_here, "i2v_output_main.mp4")
    run_i2v(
        init_image=_img,
        prompt=_prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        duration_seconds=10.0,
        num_inference_steps=6,
        guidance_scale=1.0,
        guidance_scale_2=1.0,
        seed=0,
        scheduler_name="UniPCMultistep",
        flow_shift=3.0,
        last_image=None,
        pipe=_pipe,
        original_scheduler=_orig_sched,
        output_path=_out,
    )
