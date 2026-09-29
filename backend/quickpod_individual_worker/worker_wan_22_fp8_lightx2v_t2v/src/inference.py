#!/usr/bin/env python3
"""
WAN 2.2 T2V — SDNQ Space-behavior parity. Single-pass generation, Space-aligned
schedulers, dual CFG, duration→num_frames, export quality,
frame_multiplier via existing video_postprocessing RIFE.

Derived from the I2V SDNQ space-parity script with all image-handling removed.
Does not use torchao/AoTI.
"""

from __future__ import annotations

import copy
import gc
import inspect
import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import diffusers
import numpy as np
import torch
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
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

_DEFAULT_HF_REPO = "rockapaper/wan22-14b-t2v-lightning-sdnq-fp8-svd-r32"
_DEFAULT_LOCAL_LEAF = "fastdm_wan22_t2v_lightning_sdnq_fp8_svd_r32"

MODEL_REPO = os.getenv("HF_REPO_WAN_T2V", os.getenv("REPO_ID", _DEFAULT_HF_REPO))
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")


def _resolve_wan_t2v_local_dir() -> str:
    """WAN T2V weights directory. Never use bare MODELS_DIR — diffusers needs a repo-shaped tree."""
    md = os.path.normpath(os.path.expanduser(MODELS_DIR))
    default_dir = os.path.normpath(os.path.join(md, _DEFAULT_LOCAL_LEAF))
    raw = os.environ.get("WAN_T2V_LOCAL_DIR")
    if raw is None:
        return default_dir
    s = raw.strip()
    if not s:
        return default_dir
    cand = os.path.normpath(os.path.expanduser(s))
    if cand == md:
        print(
            f"⚠️  WAN_T2V_LOCAL_DIR={raw!r} points at MODELS_DIR; using {default_dir} for T2V.",
            flush=True,
        )
        return default_dir
    return cand


# Same as multi worker: never reuse generic MODEL_LOCAL_DIR (often I2V in shared images).
MODEL_LOCAL_DIR = _resolve_wan_t2v_local_dir()

SCHEDULER_MAP = {
    "UniPCMultistep": UniPCMultistepScheduler,
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


# ---------------------------------------------------------------------------
# Model download helpers
# ---------------------------------------------------------------------------

def _auto_download_disabled() -> bool:
    return os.getenv("WAN_T2V_DISABLE_AUTO_DOWNLOAD", "").lower() in (
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


def _model_index_pipeline_class(path: str) -> Optional[str]:
    idx = os.path.join(path, "model_index.json")
    if not os.path.isfile(idx):
        return None
    try:
        with open(idx, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = data.get("_class_name")
        return str(v) if v is not None else None
    except (OSError, json.JSONDecodeError):
        return None


def _raise_if_cache_is_i2v_not_t2v(path: str) -> None:
    """Catch env mix-ups where WAN_T2V_* still points at an I2V *folder*.

    SDNQ T2V repos may ship model_index _class_name=WanImageToVideoPipeline; only error when
    the directory basename looks like I2V.
    """
    cls = _model_index_pipeline_class(path)
    if not cls or ("ImageToVideo" not in cls and "I2V" not in cls):
        return
    base = os.path.basename(os.path.normpath(path)).lower()
    if not any(
        k in base for k in ("i2v", "image2video", "img2vid", "image-to-video")
    ):
        return
    raise RuntimeError(
        f"WAN T2V refused directory {path!r} ({cls}): path name looks like an I2V cache. "
        f"Point WAN_T2V_LOCAL_DIR at a T2V-only folder (default "
        f"{os.path.join(MODELS_DIR, _DEFAULT_LOCAL_LEAF)}) and download {MODEL_REPO!r} there."
    )


def ensure_sdnq_wan_model_downloaded() -> None:
    """Snapshot WAN T2V SDNQ repo into MODEL_LOCAL_DIR if missing or incomplete."""
    if _auto_download_disabled():
        if not _wan_diffusers_dir_ready(MODEL_LOCAL_DIR):
            raise FileNotFoundError(
                f"Model not present at {MODEL_LOCAL_DIR} and WAN_T2V_DISABLE_AUTO_DOWNLOAD is set"
            )
        _raise_if_cache_is_i2v_not_t2v(MODEL_LOCAL_DIR)
        return
    if _wan_diffusers_dir_ready(MODEL_LOCAL_DIR):
        _raise_if_cache_is_i2v_not_t2v(MODEL_LOCAL_DIR)
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
    _raise_if_cache_is_i2v_not_t2v(MODEL_LOCAL_DIR)
    print(f"✓ [V2] WAN T2V weights ready at {MODEL_LOCAL_DIR}", flush=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def load_pipeline():
    global PIPELINE_CACHE, PIPELINE_LOADED
    if PIPELINE_LOADED and MODEL_LOCAL_DIR in PIPELINE_CACHE:
        return PIPELINE_CACHE[MODEL_LOCAL_DIR]

    load_start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_sdnq_wan_model_downloaded()

    print(
        f"📦 [V2] Loading SDNQ T2V pipeline from {MODEL_LOCAL_DIR} (repo ref: {MODEL_REPO})..."
    )

    vae = diffusers.AutoencoderKLWan.from_pretrained(
        MODEL_LOCAL_DIR,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )

    pipe = diffusers.WanPipeline.from_pretrained(
        MODEL_LOCAL_DIR,
        vae=vae,
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

    if device == "cuda":
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
    print(f"✓ [V2] T2V Pipeline loaded ({time.time() - load_start_time:.1f}s)")

    PIPELINE_CACHE[MODEL_LOCAL_DIR] = {
        "pipe": pipe,
        "original_scheduler": original_scheduler,
    }
    PIPELINE_LOADED = True
    return PIPELINE_CACHE[MODEL_LOCAL_DIR]


# ---------------------------------------------------------------------------
# Pipeline call builder
# ---------------------------------------------------------------------------

def _build_pipe_call_kwargs(
    pipe,
    *,
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
    return kw


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main inference entry point
# ---------------------------------------------------------------------------

def run_inference_wan_22_t2v(
    job_id: str,
    prompt: str,
    params: Dict[str, Any],
) -> str:
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

    height = int(p.get("height", 480))
    width = int(p.get("width", 832))

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
        f"[{job_id}] [V2] T2V single-pass: {width}x{height}, {num_frames}f @ {gen_fps}fps, "
        f"steps={num_inference_steps}, sched={scheduler_name}, flow_shift={flow_shift}"
    )

    generation_start = time.time()
    final_output: list
    try:
        _apply_scheduler_for_run(pipe, original_scheduler, scheduler_name, flow_shift)
        call_kw = _build_pipe_call_kwargs(
            pipe,
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

    out_path = os.path.join(output_dir, f"wan_t2v_{job_id}.mp4")
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


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _base_params: Dict[str, Any] = {
        "duration_seconds": 5.0,
        "height": 480,
        "width": 832,
        "guidance_scale": 1.0,
        "guidance_scale_2": 1.0,
        "flow_shift": 3.0,
        "video_quality": 9,
        "frame_multiplier": 32,
        "seed": 42,
    }

    _schedulers = sorted(SCHEDULER_MAP.keys())
    _step_counts = (6,)
    if isinstance(_step_counts, int):
        _step_counts = (_step_counts,)
    else:
        _step_counts = tuple(int(x) for x in _step_counts)

    _bench_rows: List[Dict[str, Any]] = []
    _script_dir = os.path.dirname(os.path.abspath(__file__))

    _prompt = "A majestic dragon flying over a medieval castle at sunset, cinematic lighting, 4K"

    _n_schedulers = len(_schedulers)
    _n_step_variants = len(_step_counts)
    _n_runs = _n_schedulers * _n_step_variants
    print(
        f"[BENCH] {_n_schedulers} schedulers × {_n_step_variants} step counts "
        f"= {_n_runs} runs (wall clock = full gen + post + export)"
    )

    for _steps in _step_counts:
        for _sampler in _schedulers:
            _jid = f"bench_t2v_{_sampler}_{_steps}steps"
            _params = {
                **_base_params,
                "num_inference_steps": _steps,
                "scheduler": _sampler,
            }
            _t0 = time.time()
            try:
                _out = run_inference_wan_22_t2v(_jid, _prompt, _params)
                _wall = time.time() - _t0
                print(
                    f"[BENCH] OK  scheduler={_sampler!r} steps={_steps} "
                    f"wall_s={_wall:.1f} out={_out}"
                )
                _bench_rows.append(
                    {
                        "prompt": _prompt,
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
                    f"[BENCH] FAIL scheduler={_sampler!r} steps={_steps} "
                    f"wall_s={_wall:.1f} err={_e!r}"
                )
                _bench_rows.append(
                    {
                        "prompt": _prompt,
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
    _json_path = os.path.join(_script_dir, f"wan_t2v_scheduler_bench_{_ts}.json")
    _report: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": os.path.basename(__file__),
        "base_params": _base_params,
        "schedulers": _schedulers,
        "step_counts": list(_step_counts),
        "runs": _bench_rows,
        "run_count": len(_bench_rows),
    }
    with open(_json_path, "w", encoding="utf-8") as _jf:
        json.dump(_report, _jf, indent=2, ensure_ascii=False)
    print(f"\n[BENCH] Wrote JSON report: {_json_path}")

    print("\n=== BENCHMARK SUMMARY (wall_s per run) ===")
    if _bench_rows:
        _w_sampler = max(len("scheduler"), max(len(r["scheduler"]) for r in _bench_rows))
        for r in _bench_rows:
            _status = "ok" if r["success"] else "FAIL"
            print(
                f"  {_status:4}  steps={r['num_inference_steps']}  "
                f"wall_s={r['wall_seconds']:8.1f}  scheduler={r['scheduler']:<{_w_sampler}}"
            )
            if not r["success"] and r.get("error"):
                _err = r["error"]
                print(f"         {_err[:200]}{'…' if len(_err) > 200 else ''}")
    else:
        print("  (no completed runs)")
