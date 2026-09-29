#!/usr/bin/env python3
"""
WAN 2.2 I2V — SDNQ Space-behavior parity (V2). Single-pass generation, Space-aligned
schedulers, dual CFG, optional last_image, duration→num_frames, export quality,
frame_multiplier via existing video_postprocessing RIFE.

Does not use torchao/AoTI. See inference_hub_space_torchao.py for Hub runtime parity.
"""

from __future__ import annotations

import copy
import gc
import inspect
import json
import os
import random
import time
import types
from io import BytesIO
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import diffusers
import httpx
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
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download
from PIL import Image
from sdnq import SDNQConfig  # noqa: F401 — register sdnq

try:
    from video_postprocessing import (
        calculate_rife_exp,
        check_rife_code_available,
        download_realesrgan_model,
        download_rife_model,
        load_rife_model,
        load_sd_upscale,
        make_inference,
        tiled_scale,
    )

    POSTPROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Video post-processing not available: {e}")
    POSTPROCESSING_AVAILABLE = False
    tiled_scale = calculate_rife_exp = make_inference = None
    download_rife_model = download_realesrgan_model = None

torch.set_float32_matmul_precision("medium")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

_DEFAULT_HF_REPO = "rockapaper/wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_Multiscene_SVI_sdnq_fp8_svd_r64"
_DEFAULT_LOCAL_LEAF = (
    "wan2.2_14b_i2v_480p_lightning_nsfw_diffusers_Multiscene_SVI_sdnq_fp8_svd_r64"
)

MODEL_REPO = os.getenv("HF_REPO_WAN_I2V", os.getenv("REPO_ID", _DEFAULT_HF_REPO))
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
MODEL_LOCAL_DIR = os.getenv(
    "MODEL_LOCAL_DIR", os.path.join(MODELS_DIR, _DEFAULT_LOCAL_LEAF)
)

SCHEDULER_MAP = {
    #"FlowMatchEulerDiscrete": FlowMatchEulerDiscreteScheduler,
    "UniPCMultistep": UniPCMultistepScheduler,
    # "DPMSolverMultistep": DPMSolverMultistepScheduler,
    #"DPMSolverSinglestep": DPMSolverSinglestepScheduler,
}

MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 160
FIXED_FPS = 16
MAX_DIM = 832
MIN_DIM = 480
SQUARE_DIM = 640
MULTIPLE_OF = 16

DEFAULT_NEGATIVE_PROMPT = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"

DEFAULT_FPS = 16
DEFAULT_NUM_INFERENCE_STEPS = 4
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_GUIDANCE_SCALE_2 = 1.0

PIPELINE_CACHE: Dict[str, Any] = {}
PIPELINE_LOADED = False


def _auto_download_disabled() -> bool:
    return os.getenv("WAN_I2V_DISABLE_AUTO_DOWNLOAD", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _wan_diffusers_dir_ready(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "model_index.json")):
        return True
    if os.path.isfile(os.path.join(path, "text_encoder", "config.json")):
        return True
    if os.path.isfile(os.path.join(path, "transformer", "config.json")):
        return True
    try:
        return len(os.listdir(path)) > 0
    except OSError:
        return False


def ensure_sdnq_wan_model_downloaded() -> None:
    """Snapshot WAN I2V SDNQ repo into MODEL_LOCAL_DIR if missing or incomplete."""
    if _auto_download_disabled():
        if not _wan_diffusers_dir_ready(MODEL_LOCAL_DIR):
            raise FileNotFoundError(
                f"Model not present at {MODEL_LOCAL_DIR} and WAN_I2V_DISABLE_AUTO_DOWNLOAD is set"
            )
        return
    if _wan_diffusers_dir_ready(MODEL_LOCAL_DIR):
        return
    os.makedirs(MODEL_LOCAL_DIR, exist_ok=True)
    print(f"⬇️  [V2] Downloading {MODEL_REPO} → {MODEL_LOCAL_DIR} …", flush=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=MODEL_LOCAL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    if not _wan_diffusers_dir_ready(MODEL_LOCAL_DIR):
        raise RuntimeError(
            f"Download finished but {MODEL_LOCAL_DIR} still looks incomplete"
        )
    print(f"✓ [V2] WAN weights ready at {MODEL_LOCAL_DIR}", flush=True)


def _realesrgan_video_ckpt_name(upscale_factor: Any) -> str:
    try:
        u = int(upscale_factor)
    except (TypeError, ValueError):
        u = 4
    return "RealESRGAN_x2plus.pth" if u <= 2 else "RealESRGAN_x4plus.pth"


def flush_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def frames_to_uint8_pil(frames) -> list:
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


def upscale_frames_streaming(
    upscale_model, frames: list, device: str, upscale_factor: int, batch_size: int = 50
) -> list:
    if tiled_scale is None:
        raise ImportError("tiled_scale not available from video_postprocessing")
    from diffusers.image_processor import VaeImageProcessor

    result = []
    n = len(frames)
    upscale_model.to(device)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = frames[start:end]
        if start % 40 == 0 or start == 0:
            print(f"   Upscaling frames {start+1}-{end}/{n}...", flush=True)
        arrs = []
        for f in batch:
            a = (
                np.asarray(
                    f if isinstance(f, Image.Image) else Image.fromarray(f),
                    dtype=np.float32,
                )
                / 255.0
            )
            arrs.append(a)
        batch_np = np.stack(arrs, axis=0)
        batch_t = (
            torch.from_numpy(np.transpose(batch_np, (0, 3, 1, 2)))
            .to(torch.float16)
            .to(device)
        )
        del batch_np, arrs
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
        upscaled_t = upscaled_t.clamp(0, 1)
        batch_out = VaeImageProcessor.pt_to_numpy(upscaled_t)
        del upscaled_t
        for i in range(len(batch_out)):
            result.append(Image.fromarray((batch_out[i] * 255).astype(np.uint8)))
        del batch_out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    upscale_model.to("cpu")
    return result


def rife_frames_streaming(
    rife_model, frames: list, device: str, original_fps: int, target_fps: int
) -> tuple:
    if calculate_rife_exp is None or make_inference is None:
        raise ImportError("calculate_rife_exp/make_inference not available")
    if target_fps <= original_fps:
        return frames, original_fps
    exp, actual_fps = calculate_rife_exp(original_fps, target_fps)
    if exp == 0:
        return frames, original_fps
    n = len(frames)
    print(
        f"   RIFE exp={exp} ({original_fps}fps -> {actual_fps}fps), "
        f"{n} frames -> {n * 2**exp} frames...",
        flush=True,
    )

    def pil_to_gpu(f) -> torch.Tensor:
        arr = np.asarray(f, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    def tensor_to_pil(t: torch.Tensor) -> Image.Image:
        arr = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        return Image.fromarray((arr * 255).astype(np.uint8))

    result = []
    for i in range(n):
        result.append(frames[i])
        if i < n - 1:
            with torch.no_grad():
                t0 = pil_to_gpu(frames[i])
                t1 = pil_to_gpu(frames[i + 1])
                midpoints = make_inference(
                    rife_model, t0, t1, upscale_amount=1, n=2**exp - 1
                )
                del t0, t1
                for mid_t in midpoints:
                    result.append(tensor_to_pil(mid_t))
                    del mid_t
                del midpoints
            if i % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        if i % 50 == 0:
            print(
                f"   RIFE pair {i+1}/{n}  ({len(result)} frames so far)...", flush=True
            )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, actual_fps


def resize_image_wan_space(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image.resize((SQUARE_DIM, SQUARE_DIM), Image.LANCZOS)
    aspect_ratio = width / height
    max_ar = MAX_DIM / MIN_DIM
    min_ar = MIN_DIM / MAX_DIM
    image_to_resize = image
    if aspect_ratio > max_ar:
        target_w, target_h = MAX_DIM, MIN_DIM
        crop_width = int(round(height * max_ar))
        left = (width - crop_width) // 2
        image_to_resize = image.crop((left, 0, left + crop_width, height))
    elif aspect_ratio < min_ar:
        target_w, target_h = MIN_DIM, MAX_DIM
        crop_height = int(round(width / min_ar))
        top = (height - crop_height) // 2
        image_to_resize = image.crop((0, top, width, top + crop_height))
    else:
        if width > height:
            target_w = MAX_DIM
            target_h = int(round(target_w / aspect_ratio))
        else:
            target_h = MAX_DIM
            target_w = int(round(target_h * aspect_ratio))
    final_w = round(target_w / MULTIPLE_OF) * MULTIPLE_OF
    final_h = round(target_h / MULTIPLE_OF) * MULTIPLE_OF
    final_w = max(MIN_DIM, min(MAX_DIM, final_w))
    final_h = max(MIN_DIM, min(MAX_DIM, final_h))
    return image_to_resize.resize((final_w, final_h), Image.LANCZOS)


def resize_and_crop_to_match(
    target_image: Image.Image, reference_image: Image.Image
) -> Image.Image:
    ref_width, ref_height = reference_image.size
    target_width, target_height = target_image.size
    scale = max(ref_width / target_width, ref_height / target_height)
    new_width, new_height = int(target_width * scale), int(target_height * scale)
    resized = target_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left, top = (new_width - ref_width) // 2, (new_height - ref_height) // 2
    return resized.crop((left, top, left + ref_width, top + ref_height))


def get_num_frames_from_duration(duration_seconds: float) -> int:
    """Match HF Space: duration × 16 fps, clamped to model frame band."""
    return 1 + int(
        np.clip(
            int(round(float(duration_seconds) * FIXED_FPS)),
            MIN_FRAMES_MODEL,
            MAX_FRAMES_MODEL,
        )
    )


def _bool_param(p: Dict[str, Any], key: str, default: bool = False) -> bool:
    v = p.get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def _patch_wan_i2v_prepare_latents_device_alignment(pipe: Any) -> None:
    """Align prepare_latents outputs when model CPU offload leaves condition on CPU but latents on CUDA.

    WanImageToVideoPipeline denoise does ``torch.cat([latents, condition], dim=1)``. Under
    ``enable_model_cpu_offload``, VAE encode can yield ``latent_condition`` on CPU while
    ``randn_tensor(..., device=_execution_device)`` places noise on GPU, which triggers
    ``wrapper_CUDA_cat`` device mismatch. Moving condition (and mask) to ``latents.device`` is safe
    when devices already match (no-op).
    """
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


def _apply_scheduler_for_run(
    pipe, original_scheduler, scheduler_name: str, flow_shift: float
) -> None:
    """Rebuild scheduler from frozen original config (Space-style), then restore after run."""
    scheduler_class = SCHEDULER_MAP.get(scheduler_name)
    if scheduler_class is None:
        print(f"⚠️  Unknown scheduler {scheduler_name!r}, keeping pipeline default")
        return
    config = copy.deepcopy(original_scheduler.config)
    try:
        if scheduler_class == FlowMatchEulerDiscreteScheduler:
            config["shift"] = flow_shift
        else:
            config["flow_shift"] = flow_shift
        pipe.scheduler = scheduler_class.from_config(config)
    except Exception as e:
        print(f"⚠️  Scheduler from_config failed ({scheduler_name}): {e}")


def load_pipeline():
    global PIPELINE_CACHE, PIPELINE_LOADED
    if PIPELINE_LOADED and MODEL_LOCAL_DIR in PIPELINE_CACHE:
        return PIPELINE_CACHE[MODEL_LOCAL_DIR]

    load_start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_sdnq_wan_model_downloaded()

    print(
        f"📦 [V2] Loading SDNQ pipeline from {MODEL_LOCAL_DIR} (repo ref: {MODEL_REPO})..."
    )
    pipe = diffusers.WanImageToVideoPipeline.from_pretrained(
        MODEL_LOCAL_DIR,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if hasattr(pipe, "text_encoder"):
        te = pipe.text_encoder
        if (
            hasattr(te, "shared")
            and hasattr(te, "encoder")
            and hasattr(te.encoder, "embed_tokens")
        ):
            if te.shared.weight.data_ptr() != te.encoder.embed_tokens.weight.data_ptr():
                te.encoder.embed_tokens.weight = te.shared.weight
        if hasattr(te, "tie_weights"):
            te.tie_weights()

    disable_offload = os.getenv("WAN_I2V_DISABLE_CPU_OFFLOAD", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if device == "cuda":
        if disable_offload:
            pipe = pipe.to(device)
            print("✓ Full GPU load (WAN_I2V_DISABLE_CPU_OFFLOAD)")
        else:
            try:
                pipe.enable_model_cpu_offload()
                print("✓ Model CPU offload enabled")
            except Exception:
                try:
                    pipe.enable_sequential_cpu_offload()
                    print("✓ Sequential CPU offload enabled")
                except Exception:
                    pipe = pipe.to(device)
    else:
        pipe = pipe.to(device)

    if device == "cuda":
        _patch_wan_i2v_prepare_latents_device_alignment(pipe)

    try:
        import flash_attn  # noqa: F401
    except ImportError:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    try:
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
    except Exception:
        pass
    try:
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing(slice_size="max")
    except Exception:
        pass

    original_scheduler = copy.deepcopy(pipe.scheduler)
    print(f"✓ [V2] Pipeline loaded ({time.time() - load_start_time:.1f}s)")

    PIPELINE_CACHE[MODEL_LOCAL_DIR] = {
        "pipe": pipe,
        "original_scheduler": original_scheduler,
    }
    PIPELINE_LOADED = True
    return PIPELINE_CACHE[MODEL_LOCAL_DIR]


def _build_pipe_call_kwargs(
    pipe,
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


def _export_video(
    frames: list, out_path: str, fps: int, video_quality: Optional[int]
) -> None:
    kwargs: Dict[str, Any] = {"fps": fps}
    if video_quality is not None:
        try:
            sig = inspect.signature(export_to_video)
            if "quality" in sig.parameters:
                kwargs["quality"] = int(video_quality)
            else:
                print("⚠️  export_to_video has no quality= param; skipping quality")
        except Exception as e:
            print(f"⚠️  export_to_video quality check failed: {e}")
    export_to_video(frames, out_path, **kwargs)


def run_inference_wan_22_i2v(
    job_id: str,
    prompt_or_prompts: Union[str, List[str]],
    params: Dict[str, Any],
    init_image: Optional[Image.Image] = None,
    last_image: Optional[Image.Image] = None,
) -> str:
    if init_image is None:
        raise ValueError("init_image is required for I2V inference")

    if isinstance(prompt_or_prompts, str):
        prompt = prompt_or_prompts
    else:
        plist = list(prompt_or_prompts)
        if len(plist) > 1:
            print(
                f"[{job_id}] [V2] Multiple prompts received; using first only (single-pass mode)."
            )
        prompt = plist[0] if plist else ""

    p = params or {}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    pdata = load_pipeline()
    pipe = pdata["pipe"]
    original_scheduler = pdata["original_scheduler"]

    gen_fps = int(p.get("fps", DEFAULT_FPS))
    if "num_frames" in p and p.get("num_frames") is not None:
        num_frames = int(p["num_frames"])
    else:
        dur = p.get("duration_seconds", p.get("duration"))
        if dur is not None:
            num_frames = get_num_frames_from_duration(float(dur))
        else:
            num_frames = 81

    guidance_scale = float(p.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
    guidance_scale_2 = float(p.get("guidance_scale_2", DEFAULT_GUIDANCE_SCALE_2))
    num_inference_steps = int(p.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS))
    seed = p.get("seed", None)
    if seed is None:
        seed = random.randint(0, 1000)
    else:
        seed = int(seed)

    negative_prompt = p.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    output_dir = p.get("output_dir", "/workspace")
    os.makedirs(output_dir, exist_ok=True)

    resize_image_flag = p.get("resize_image", True)
    resize_mode = str(p.get("resize_mode", "area480")).strip().lower()

    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]

    if resize_image_flag:
        if resize_mode == "wan_space":
            image = resize_image_wan_space(init_image)
        else:
            max_area = 480 * 832
            aspect_ratio = init_image.height / init_image.width
            height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
            width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
            image = init_image.resize((width, height))
    else:
        ow, oh = init_image.width, init_image.height
        width = (ow // mod_value) * mod_value
        height = (oh // mod_value) * mod_value
        image = (
            init_image.resize((width, height))
            if (width, height) != (ow, oh)
            else init_image
        )

    width, height = image.size

    processed_last: Optional[Image.Image] = None
    if last_image is not None:
        if resize_image_flag:
            if resize_mode == "wan_space":
                li = resize_image_wan_space(last_image)
            else:
                max_area = 480 * 832
                ar = last_image.height / last_image.width
                h2 = round(np.sqrt(max_area * ar)) // mod_value * mod_value
                w2 = round(np.sqrt(max_area / ar)) // mod_value * mod_value
                li = last_image.resize((w2, h2))
        else:
            li = last_image
        processed_last = resize_and_crop_to_match(li, image)

    scheduler_name = str(
        p.get("scheduler_name", p.get("scheduler", "UniPCMultistep"))
    ).strip()
    flow_shift = float(p.get("flow_shift", 3.0))
    video_quality = p.get("video_quality", 6)
    if video_quality is not None:
        video_quality = int(video_quality)

    fm_raw = p.get("frame_multiplier", None)
    enable_rife = _bool_param(p, "enable_rife", False)
    rife_target_fps = int(p.get("rife_target_fps", 30))
    if fm_raw is not None:
        fm = int(fm_raw)
        if fm > 16:
            enable_rife = True
            rife_target_fps = fm
        elif enable_rife and "rife_target_fps" not in p:
            pass

    enable_upscaling = _bool_param(p, "enable_upscaling", False) or _bool_param(
        p, "enable_upscale", False
    )
    upscale_factor = int(p.get("upscale_factor", 2))
    postprocess_interpolate_first = _bool_param(
        p, "postprocess_interpolate_first", False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).manual_seed(seed)

    print(
        f"[{job_id}] [V2] I2V single-pass: {width}x{height}, {num_frames}f @ {gen_fps}fps, "
        f"steps={num_inference_steps}, sched={scheduler_name}, flow_shift={flow_shift}"
    )

    generation_start = time.time()
    final_output: list
    try:
        _apply_scheduler_for_run(pipe, original_scheduler, scheduler_name, flow_shift)
        call_kw = _build_pipe_call_kwargs(
            pipe,
            image=image,
            last_image=processed_last,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        with torch.inference_mode():
            output = pipe(**call_kw).frames[0]
        final_output = frames_to_uint8_pil(output)
        del output
    finally:
        pipe.scheduler = original_scheduler

    flush_memory()
    generation_time = time.time() - generation_start

    postprocess_time = 0.0
    final_fps = gen_fps
    final_height = height
    final_width = width

    if POSTPROCESSING_AVAILABLE and (enable_rife or enable_upscaling):
        postprocess_start = time.time()
        rife_model = None
        upscale_model = None
        if enable_rife and check_rife_code_available():
            rife_path = None
            if download_rife_model:
                rife_path = download_rife_model(MODELS_DIR)
            if not rife_path:
                rife_path = os.path.join(MODELS_DIR, "rife")
            if rife_path and os.path.exists(rife_path) and os.listdir(rife_path):
                rife_model = load_rife_model(rife_path)
            else:
                print(
                    "⚠️  RIFE weights missing after download attempt; interpolation disabled"
                )
        if enable_upscaling:
            realesrgan_model_file = None
            if download_realesrgan_model:
                realesrgan_model_file = download_realesrgan_model(
                    MODELS_DIR, upscale_factor
                )
            if not realesrgan_model_file:
                realesrgan_model_file = os.path.join(
                    MODELS_DIR,
                    "realesrgan",
                    _realesrgan_video_ckpt_name(upscale_factor),
                )
            if os.path.exists(realesrgan_model_file):
                try:
                    upscale_model = load_sd_upscale(realesrgan_model_file, device)
                except Exception as e:
                    print(f"⚠️  Real-ESRGAN load failed: {str(e)[:100]}")

        if postprocess_interpolate_first:
            if enable_rife and rife_model:
                try:
                    final_output, final_fps = rife_frames_streaming(
                        rife_model, final_output, device, gen_fps, rife_target_fps
                    )
                    flush_memory()
                except Exception as e:
                    print(f"⚠️  RIFE failed: {str(e)[:100]}")
            if enable_upscaling and upscale_model:
                try:
                    final_output = upscale_frames_streaming(
                        upscale_model, final_output, device, upscale_factor
                    )
                    final_height = height * upscale_factor
                    final_width = width * upscale_factor
                    flush_memory()
                except Exception as e:
                    print(f"⚠️  Upscale failed: {str(e)[:100]}")
        else:
            if enable_upscaling and upscale_model:
                try:
                    final_output = upscale_frames_streaming(
                        upscale_model, final_output, device, upscale_factor
                    )
                    final_height = height * upscale_factor
                    final_width = width * upscale_factor
                    flush_memory()
                except Exception as e:
                    print(f"⚠️  Upscale failed: {str(e)[:100]}")
            if enable_rife and rife_model:
                try:
                    final_output, final_fps = rife_frames_streaming(
                        rife_model, final_output, device, gen_fps, rife_target_fps
                    )
                    flush_memory()
                except Exception as e:
                    print(f"⚠️  RIFE failed: {str(e)[:100]}")
        postprocess_time = time.time() - postprocess_start

    out_path = os.path.join(output_dir, f"wan_i2v_{job_id}.mp4")
    export_start = time.time()
    _export_video(final_output, out_path, fps=final_fps, video_quality=video_quality)
    export_time = time.time() - export_start

    total_time = generation_time + postprocess_time + export_time
    print(
        f"[{job_id}] [V2] ✓ {total_time:.1f}s (gen {generation_time:.1f}s, post {postprocess_time:.1f}s, "
        f"export {export_time:.1f}s) | {final_width}x{final_height} @ {final_fps}fps"
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return os.path.abspath(out_path)


if __name__ == "__main__":
    _prm_d1 = """
    a typical image to video prompt for wan 2.2 i2v model   
    """
    _prm_h1 = """
    a typical image to video prompt for wan 2.2 i2v model
    """

    _base_params: Dict[str, Any] = {
        "duration_seconds": 10.0,
        "guidance_scale": 1.0,
        "guidance_scale_2": 1.0,
        "flow_shift": 3.0,
        "video_quality": 9,
        "frame_multiplier": 32,
        "resize_mode": "area480",
        "seed": 42,
    }

    _script_dir = os.path.dirname(os.path.abspath(__file__))

    # Init + last-frame pairs in _script_dir: h1.jpeg + h1_last.jpeg, etc. Optional per-case last_fallback_url.
    _bench_cases: List[Dict[str, Any]] = [
        {
            "image_id": "d1",
            "image_file": "d1.jpeg",
            "last_image_file": "d1_last.png",
            "prompt": _prm_h1.strip(),
            "fallback_url": None,
        },
        {
            "image_id": "h1",
            "image_file": "h1.jpeg",
            "last_image_file": "h1_last.png",
            "prompt": _prm_h1.strip(),
            "fallback_url": None,
        },
        {
            "image_id": "h2",
            "image_file": "h2.jpeg",
            "last_image_file": "h2_last.png",
            "prompt": _prm_h1.strip(),
            "fallback_url": None,
        },
        {
            "image_id": "h3",
            "image_file": "h3.jpeg",
            "last_image_file": "h3_last.png",
            "prompt": _prm_h1.strip(),
            "fallback_url": None,
        },
        {
            "image_id": "h4",
            "image_file": "h4.jpeg",
            "last_image_file": "h4_last.png",
            "prompt": _prm_h1.strip(),
            "fallback_url": None,
        },
        {
            "image_id": "h5",
            "image_file": "h5.jpeg",
            "last_image_file": "h5_last.png",
            "prompt": _prm_h1.strip(),
            "fallback_url": None,
        },
    ]

    def _load_case_image(case: Dict[str, Any]) -> tuple[Optional[Image.Image], str]:
        path = os.path.join(_script_dir, case["image_file"])
        if os.path.isfile(path):
            return Image.open(path).convert("RGB"), path
        url = case.get("fallback_url")
        if url:
            with httpx.Client(timeout=60) as _c:
                raw = _c.get(url).content
            return Image.open(BytesIO(raw)).convert("RGB"), url
        return None, path

    def _load_case_last_image(
        case: Dict[str, Any],
    ) -> tuple[Optional[Image.Image], Optional[str]]:
        """Load last-frame PIL if last_image_file is set; optional last_fallback_url."""
        last_file = case.get("last_image_file")
        if not last_file:
            return None, None
        path = os.path.join(_script_dir, str(last_file))
        if os.path.isfile(path):
            return Image.open(path).convert("RGB"), path
        url = case.get("last_fallback_url")
        if url:
            with httpx.Client(timeout=60) as _c:
                raw = _c.get(url).content
            return Image.open(BytesIO(raw)).convert("RGB"), str(url)
        return None, path

    # Tuple needs a trailing comma for one element: (6,) — bare (6) is int 6 and breaks "for x in _step_counts".
    _step_counts = (6,)
    if isinstance(_step_counts, int):
        _step_counts = (_step_counts,)
    else:
        _step_counts = tuple(int(x) for x in _step_counts)

    _schedulers = sorted(SCHEDULER_MAP.keys())
    _bench_rows: List[Dict[str, Any]] = []

    _n_images = len(_bench_cases)
    _n_schedulers = len(_schedulers)
    _n_step_variants = len(_step_counts)
    _n_runs = _n_images * _n_schedulers * _n_step_variants
    print(
        f"[BENCH] {_n_images} images × {_n_schedulers} schedulers × {_n_step_variants} step counts "
        f"= {_n_runs} runs (wall clock = full gen + post + export)"
    )

    for _case in _bench_cases:
        _iid = str(_case["image_id"])
        _img, _img_src = _load_case_image(_case)
        if _img is None:
            print(
                f"[BENCH] SKIP image_id={_iid!r} (missing {_case['image_file']!r}, no fallback_url)"
            )
            continue
        _last_img, _last_src = _load_case_last_image(_case)
        if _case.get("last_image_file") and _last_img is None:
            print(
                f"[BENCH] SKIP image_id={_iid!r} (missing last {_case['last_image_file']!r}, no last_fallback_url)"
            )
            continue
        _prm = str(_case["prompt"])
        print(
            f"[BENCH] --- image_id={_iid} init={_img_src!r} "
            f"last={_last_src or 'none'} ---"
        )

        for _steps in _step_counts:
            for _sampler in _schedulers:
                _jid = f"bench_{_iid}_{_sampler}_{_steps}steps"
                _params = {
                    **_base_params,
                    "num_inference_steps": _steps,
                    "scheduler": _sampler,
                }
                _t0 = time.time()
                try:
                    _out = run_inference_wan_22_i2v(
                        _jid,
                        _prm,
                        _params,
                        init_image=_img,
                        last_image=_last_img,
                    )
                    _wall = time.time() - _t0
                    print(
                        f"[BENCH] OK  image={_iid} scheduler={_sampler!r} steps={_steps} "
                        f"wall_s={_wall:.1f} out={_out}"
                    )
                    _bench_rows.append(
                        {
                            "image_id": _iid,
                            "image_file": _case["image_file"],
                            "image_source": _img_src,
                            "last_image_file": _case.get("last_image_file"),
                            "last_image_source": _last_src,
                            "prompt": _prm,
                            "job_id": _jid,
                            "scheduler": _sampler,
                            "num_inference_steps": _steps,
                            "wall_seconds": round(_wall, 3),
                            "success": True,
                            "output_path": _out,
                            "error": None,
                        }
                    )
                except Exception as _e:
                    _wall = time.time() - _t0
                    print(
                        f"[BENCH] FAIL image={_iid} scheduler={_sampler!r} steps={_steps} "
                        f"wall_s={_wall:.1f} err={_e!r}"
                    )
                    _bench_rows.append(
                        {
                            "image_id": _iid,
                            "image_file": _case["image_file"],
                            "image_source": _img_src,
                            "last_image_file": _case.get("last_image_file"),
                            "last_image_source": _last_src,
                            "prompt": _prm,
                            "job_id": _jid,
                            "scheduler": _sampler,
                            "num_inference_steps": _steps,
                            "wall_seconds": round(_wall, 3),
                            "success": False,
                            "output_path": None,
                            "error": str(_e),
                        }
                    )

    _ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    _json_path = os.path.join(_script_dir, f"wan_i2v_scheduler_bench_{_ts}.json")
    _report: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": os.path.basename(__file__),
        "base_params": _base_params,
        "schedulers": _schedulers,
        "step_counts": list(_step_counts),
        "bench_cases": [
            {
                "image_id": c["image_id"],
                "image_file": c["image_file"],
                "last_image_file": c.get("last_image_file"),
                "has_fallback_url": bool(c.get("fallback_url")),
                "has_last_fallback_url": bool(c.get("last_fallback_url")),
                "prompt": str(c["prompt"]),
            }
            for c in _bench_cases
        ],
        "runs": _bench_rows,
        "run_count": len(_bench_rows),
    }
    with open(_json_path, "w", encoding="utf-8") as _jf:
        json.dump(_report, _jf, indent=2, ensure_ascii=False)
    print(f"\n[BENCH] Wrote JSON report: {_json_path}")

    print("\n=== BENCHMARK SUMMARY (wall_s per run) ===")
    if _bench_rows:
        _w_img = max(len("image"), max(len(r["image_id"]) for r in _bench_rows))
        _w_sampler = max(len("scheduler"), max(len(r["scheduler"]) for r in _bench_rows))
        for r in _bench_rows:
            _status = "ok" if r["success"] else "FAIL"
            print(
                f"  {_status:4}  image={r['image_id']:<{_w_img}}  steps={r['num_inference_steps']}  "
                f"wall_s={r['wall_seconds']:8.1f}  scheduler={r['scheduler']:<{_w_sampler}}"
            )
            if not r["success"] and r.get("error"):
                _err = r["error"]
                print(f"         {_err[:200]}{'…' if len(_err) > 200 else ''}")
    else:
        print("  (no completed runs — check missing images / fallbacks)")
