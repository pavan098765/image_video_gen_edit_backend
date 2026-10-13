#!/usr/bin/env python3
"""
WAN 2.2 image-to-video for **SDNQ-quantized** diffusers checkpoints (FP8 family, same as ``quantize_sdnq_multi.py``).

Loads ``WanImageToVideoPipeline`` in bfloat16, applies ``fix_sdnq_text_encoder_tie``, uses **model / sequential /
balanced / full GPU** offload via ``WAN_I2V_OFFLOAD_MODE`` (default **model** for VRAM). CUDA runs use
``torch.autocast(bfloat16)`` during denoise so Accelerate offload stays dtype-consistent with SDNQ matmuls.

Space-style scheduling (``UniPCMultistep`` / ``FlowMatchEulerDiscrete``; see ``scheduler`` / ``scheduler_name`` params),
``flow_shift`` (UniPC) or ``shift`` (Flow Euler), dual guidance, optional ``last_image``, duration →
``num_frames`` with Wan **4k+1** frame snapping, and RIFE / Real-ESRGAN via ``video_postprocessing``.
Init and ``last_image`` always use ``resize_image_wan_space`` (no bypass).

Set ``WAN_I2V_INFERENCE_MODULE=inference.py`` (default) or use ``inference_sdnq_space_parity.py``, which
re-exports this module for compatibility with older images.
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
from io import BytesIO
import traceback
from typing import Any, Dict, List, Optional, Union

import httpx
import numpy as np
import torch
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
    WanImageToVideoPipeline,
)
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download
from PIL import Image

from sdnq import SDNQConfig  # noqa: F401 — register SDNQ quantized modules with diffusers

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

_DEFAULT_HF_REPO = (
    "rockapaper/wan2.2_14b_i2v_480p_lightning_diffusers_wamu_svi_fp8_r32_sdnq"
)
_DEFAULT_LOCAL_LEAF = "wan22_i2v_lightning_sdnq_fp8_svd_r32"

MODEL_REPO = os.getenv("WAN_I2V_MODEL_ID", "").strip() or os.getenv(
    "HF_REPO_WAN_I2V", os.getenv("REPO_ID", _DEFAULT_HF_REPO)
)
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
MODEL_LOCAL_DIR = os.getenv(
    "MODEL_LOCAL_DIR", os.path.join(MODELS_DIR, _DEFAULT_LOCAL_LEAF)
)

SCHEDULER_DEFAULT_NAME = "UniPCMultistep"
SCHEDULER_MAP = {
    "UniPCMultistep": UniPCMultistepScheduler,
    "FlowMatchEulerDiscrete": FlowMatchEulerDiscreteScheduler,
}
# Normalized keys (lowercase, no space/underscore) → SCHEDULER_MAP key
_SCHEDULER_NAME_ALIASES: Dict[str, str] = {
    "floweuler": "FlowMatchEulerDiscrete",
    "flowmatcheuler": "FlowMatchEulerDiscrete",
    "flowmatcheulerdiscrete": "FlowMatchEulerDiscrete",
}


def _resolve_scheduler_name(name: Optional[str]) -> str:
    """Map ``scheduler`` / ``scheduler_name`` param values to a ``SCHEDULER_MAP`` key."""
    raw = (name or "").strip() or SCHEDULER_DEFAULT_NAME
    if raw in SCHEDULER_MAP:
        return raw
    canon = raw.lower().replace(" ", "").replace("_", "")
    if canon in _SCHEDULER_NAME_ALIASES:
        return _SCHEDULER_NAME_ALIASES[canon]
    for k in SCHEDULER_MAP:
        if k.lower() == raw.lower():
            return k
    return raw

MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = int(os.getenv("WAN_I2V_MAX_FRAMES_CAP", "241"))
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


def _wan_i2v_offload_mode() -> str:
    raw = os.getenv("WAN_I2V_OFFLOAD_MODE", "").strip().lower()
    if raw in ("balanced", "balance", "pin_transformer", "transformer_gpu"):
        return "balanced"
    if raw in ("sequential", "seq"):
        return "sequential"
    if raw in ("none", "full", "gpu"):
        return "none"
    if raw in ("model", "offload", "cpu_offload", ""):
        return "model"
    print(f"⚠️  Unknown WAN_I2V_OFFLOAD_MODE={raw!r}; using model (default)", flush=True)
    return "model"


def _wan_i2v_attention_slicing_slice() -> Optional[Union[str, int]]:
    raw = os.getenv("WAN_I2V_ATTENTION_SLICING", "none").strip().lower()
    if raw in ("0", "false", "off", "none", "disable"):
        return None
    if raw == "max":
        return "max"
    if raw in ("auto", ""):
        return "auto"
    try:
        return max(1, int(raw))
    except ValueError:
        print(f"⚠️  Unknown WAN_I2V_ATTENTION_SLICING={raw!r}; using auto", flush=True)
        return "auto"


def _try_enable_model_cpu_offload(pipe: Any, cuda_dev: torch.device) -> None:
    try:
        pipe.enable_model_cpu_offload(device=cuda_dev)
    except TypeError:
        pipe.enable_model_cpu_offload()


def _try_enable_sequential_cpu_offload(pipe: Any, cuda_dev: torch.device) -> None:
    try:
        pipe.enable_sequential_cpu_offload(device=cuda_dev)
    except TypeError:
        pipe.enable_sequential_cpu_offload()


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


def ensure_wan_i2v_model_downloaded() -> None:
    if _auto_download_disabled():
        if not _wan_diffusers_dir_ready(MODEL_LOCAL_DIR):
            raise FileNotFoundError(
                f"Model not present at {MODEL_LOCAL_DIR} and WAN_I2V_DISABLE_AUTO_DOWNLOAD is set"
            )
        return
    if _wan_diffusers_dir_ready(MODEL_LOCAL_DIR):
        return
    os.makedirs(MODEL_LOCAL_DIR, exist_ok=True)
    print(f"⬇️  Downloading {MODEL_REPO} → {MODEL_LOCAL_DIR} …", flush=True)
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
    print(f"✓ WAN I2V (SDNQ) weights ready at {MODEL_LOCAL_DIR}", flush=True)


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


def max_frames_allowed() -> int:
    return int(MAX_FRAMES_MODEL)


def get_num_frames_from_duration(duration_seconds: float) -> int:
    raw = 1 + int(round(float(duration_seconds) * FIXED_FPS))
    return int(np.clip(raw, MIN_FRAMES_MODEL, max_frames_allowed()))


def wan_snap_num_frames_4k_plus_one(n: int) -> int:
    """Wan I2V expects num_frames − 1 divisible by 4."""
    cap = max_frames_allowed()
    n = int(np.clip(n, MIN_FRAMES_MODEL, cap))
    k = n - 1
    k = int(round(k / 4.0)) * 4
    out = k + 1
    return int(np.clip(out, MIN_FRAMES_MODEL, cap))


def _bool_param(p: Dict[str, Any], key: str, default: bool = False) -> bool:
    v = p.get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def _scheduler_config_to_plain(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (torch.dtype, torch.device)):
        return obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(_scheduler_config_to_plain(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _scheduler_config_to_plain(v) for k, v in obj.items()}
    if hasattr(obj, "items"):
        return {k: _scheduler_config_to_plain(v) for k, v in obj.items()}
    return obj


def _apply_scheduler_for_run(
    pipe, scheduler_restore: Dict[str, Any], scheduler_name: str, flow_shift: float
) -> None:
    sn = scheduler_name
    if sn not in SCHEDULER_MAP:
        print(
            f"⚠️  Unsupported scheduler {scheduler_name!r}; using {SCHEDULER_DEFAULT_NAME!r}",
            flush=True,
        )
        sn = SCHEDULER_DEFAULT_NAME
    scheduler_class = SCHEDULER_MAP[sn]
    config = copy.deepcopy(_scheduler_config_to_plain(dict(scheduler_restore["config"])))
    try:
        if scheduler_class is FlowMatchEulerDiscreteScheduler:
            config["shift"] = flow_shift
        else:
            config["flow_shift"] = flow_shift
        pipe.scheduler = scheduler_class.from_config(config)
    except Exception as e:
        print(f"⚠️  Scheduler from_config failed ({scheduler_name}): {e}", flush=True)


def fix_sdnq_text_encoder_tie(pipe: WanImageToVideoPipeline) -> None:
    if not hasattr(pipe, "text_encoder"):
        return
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


def configure_wan_pipe_memory(pipe: Any, device_str: str) -> None:
    """VRAM layout aligned with ``wan22_i2v_inference_sdnq`` + optional attention slicing."""
    cuda_dev = torch.device("cuda")
    if device_str != "cuda":
        pipe.to("cpu")
        print("✓ Pipeline on CPU (no CUDA)", flush=True)
        return

    _use_xformers_full = os.getenv("WAN_I2V_USE_XFORMERS_NOT_OFFLOAD", "").lower() in (
        "1",
        "true",
        "yes",
    )
    mode = _wan_i2v_offload_mode()

    if _use_xformers_full:
        pipe.to(device_str)
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print(
                "✓ xFormers + full GPU (WAN_I2V_USE_XFORMERS_NOT_OFFLOAD)",
                flush=True,
            )
        except Exception as e:
            print(
                f"⚠️  xFormers failed ({e!r}); falling back to model CPU offload",
                flush=True,
            )
            try:
                _try_enable_model_cpu_offload(pipe, cuda_dev)
                print("✓ Model CPU offload enabled", flush=True)
            except Exception:
                _try_enable_sequential_cpu_offload(pipe, cuda_dev)
                print("✓ Sequential CPU offload enabled", flush=True)

    elif mode == "balanced":
        excl = getattr(pipe, "_exclude_from_cpu_offload", None)
        if excl is None:
            pipe._exclude_from_cpu_offload = []
            excl = pipe._exclude_from_cpu_offload
        for name in ("transformer", "transformer_2"):
            if name not in excl:
                excl.append(name)
            comp = getattr(pipe, name, None)
            if comp is not None:
                comp.to(cuda_dev)
        try:
            _try_enable_sequential_cpu_offload(pipe, cuda_dev)
            print(
                "✓ Sequential offload + transformers on GPU (balanced)",
                flush=True,
            )
        except Exception as e:
            print(
                f"⚠️  balanced sequential offload failed ({e!r}); retrying model offload",
                flush=True,
            )
            for name in ("transformer", "transformer_2"):
                try:
                    pipe._exclude_from_cpu_offload.remove(name)
                except Exception:
                    pass
                comp = getattr(pipe, name, None)
                if comp is not None:
                    try:
                        comp.to("cpu")
                    except Exception:
                        pass
            try:
                _try_enable_model_cpu_offload(pipe, cuda_dev)
                print("✓ Model CPU offload enabled", flush=True)
            except Exception:
                _try_enable_sequential_cpu_offload(pipe, cuda_dev)
                print("✓ Sequential CPU offload enabled", flush=True)

    elif mode == "sequential":
        try:
            _try_enable_sequential_cpu_offload(pipe, cuda_dev)
            print("✓ Sequential CPU offload", flush=True)
        except Exception:
            _try_enable_model_cpu_offload(pipe, cuda_dev)
            print("✓ Model CPU offload (sequential failed)", flush=True)

    elif mode == "none":
        pipe.to(device_str)
        print("✓ Full pipeline on GPU (WAN_I2V_OFFLOAD_MODE=none)", flush=True)

    else:
        try:
            _try_enable_model_cpu_offload(pipe, cuda_dev)
            print("✓ Model CPU offload", flush=True)
        except Exception:
            pipe.to(device_str)
            print("✓ Full pipeline on GPU (offload failed)", flush=True)

    try:
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
    except Exception:
        pass
    try:
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
    except Exception:
        pass

    _aslice = _wan_i2v_attention_slicing_slice()
    if _aslice is not None:
        try:
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing(slice_size=_aslice)
                print(f"✓ Attention slicing ({_aslice!r})", flush=True)
        except Exception as e:
            print(f"⚠️  enable_attention_slicing failed: {e!r}", flush=True)

    if mode in ("none", "balanced") or _use_xformers_full:
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            if not _use_xformers_full:
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    print("✓ xFormers memory-efficient attention", flush=True)
                except Exception:
                    pass


def load_pipeline():
    global PIPELINE_CACHE, PIPELINE_LOADED
    if PIPELINE_LOADED and MODEL_LOCAL_DIR in PIPELINE_CACHE:
        return PIPELINE_CACHE[MODEL_LOCAL_DIR]

    load_start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_wan_i2v_model_downloaded()

    _root = str(os.path.abspath(MODEL_LOCAL_DIR))
    print(
        f"📦 Loading WAN I2V (SDNQ FP8) from {_root} (repo ref: {MODEL_REPO})…",
        flush=True,
    )

    _load_kw: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    try:
        pipe = WanImageToVideoPipeline.from_pretrained(_root, **_load_kw)
    except TypeError:
        _load_kw.pop("torch_dtype", None)
        _load_kw["dtype"] = torch.bfloat16
        pipe = WanImageToVideoPipeline.from_pretrained(_root, **_load_kw)

    fix_sdnq_text_encoder_tie(pipe)
    configure_wan_pipe_memory(pipe, device)

    original_scheduler = {
        "cls": type(pipe.scheduler),
        "config": copy.deepcopy(
            _scheduler_config_to_plain(dict(pipe.scheduler.config))
        ),
    }
    print(f"✓ Pipeline loaded ({time.time() - load_start_time:.1f}s)", flush=True)

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
    output_type: Optional[str] = None,
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
    if output_type is not None and "output_type" in params:
        kw["output_type"] = output_type
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


def _resolve_image_arg(raw: Any, timeout: int = 120) -> Optional[Image.Image]:
    if raw is None:
        return None
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        if s.startswith("http://") or s.startswith("https://"):
            import requests

            resp = requests.get(s, timeout=timeout)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        if os.path.exists(s):
            return Image.open(s).convert("RGB")
        raise ValueError(
            "init_image must be a presigned http(s) URL or an existing local file path"
        )
    raise TypeError(f"unsupported image arg type: {type(raw).__name__}")


def run_inference_wan_22_i2v(
    job_id: str,
    prompt_or_prompts: Union[str, List[str]],
    params: Dict[str, Any],
    init_image: Optional[Union[Image.Image, str]] = None,
    last_image: Optional[Union[Image.Image, str]] = None,
) -> str:
    p_in = dict(params or {})
    if init_image is None:
        init_image = p_in.pop("init_image", None) or p_in.pop("image", None)
    else:
        p_in.pop("init_image", None)
        p_in.pop("image", None)
    if last_image is None:
        last_image = p_in.pop("last_image", None) or p_in.pop("last_frame_image", None)
    else:
        p_in.pop("last_image", None)
        p_in.pop("last_frame_image", None)
    init_image = _resolve_image_arg(init_image)
    last_image = _resolve_image_arg(last_image)
    params = p_in
    if init_image is None:
        raise ValueError("init_image is required for I2V inference")

    if isinstance(prompt_or_prompts, str):
        prompt = prompt_or_prompts
    else:
        plist = list(prompt_or_prompts)
        if len(plist) > 1:
            print(
                f"[{job_id}] Multiple prompts received; using first only (single-pass mode).",
                flush=True,
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
    num_frames = wan_snap_num_frames_4k_plus_one(num_frames)

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

    image = resize_image_wan_space(init_image)
    width, height = image.size

    processed_last: Optional[Image.Image] = None
    if last_image is not None:
        li = resize_image_wan_space(last_image)
        processed_last = resize_and_crop_to_match(li, image)

    scheduler_name = _resolve_scheduler_name(
        str(p.get("scheduler_name", p.get("scheduler", SCHEDULER_DEFAULT_NAME))).strip()
    )
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
    gen_device = (
        "cpu"
        if device == "cuda"
        and _wan_i2v_offload_mode() in ("sequential", "model", "balanced")
        else device
    )
    generator = torch.Generator(device=gen_device).manual_seed(seed)

    print(
        f"[{job_id}] I2V: {width}x{height}, {num_frames}f @ {gen_fps}fps, "
        f"steps={num_inference_steps}, sched={scheduler_name}, flow_shift={flow_shift}",
        flush=True,
    )

    generation_start = time.time()
    final_output: list
    try:
        _apply_scheduler_for_run(pipe, original_scheduler, scheduler_name, flow_shift)
        out_type: Optional[str] = None
        if p.get("output_type") is not None:
            out_type = str(p.get("output_type")).strip() or None
        elif "output_type" in inspect.signature(pipe.__call__).parameters:
            out_type = "np"
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
            output_type=out_type,
        )
        with torch.inference_mode():
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = pipe(**call_kw).frames[0]
            else:
                output = pipe(**call_kw).frames[0]
        final_output = frames_to_uint8_pil(output)
        del output
    finally:
        r = original_scheduler
        pipe.scheduler = r["cls"].from_config(
            copy.deepcopy(_scheduler_config_to_plain(dict(r["config"])))
        )

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
                    "⚠️  RIFE weights missing after download attempt; interpolation disabled",
                    flush=True,
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
                    print(f"⚠️  Real-ESRGAN load failed: {str(e)[:100]}", flush=True)

        if postprocess_interpolate_first:
            if enable_rife and rife_model:
                try:
                    final_output, final_fps = rife_frames_streaming(
                        rife_model, final_output, device, gen_fps, rife_target_fps
                    )
                    flush_memory()
                except Exception as e:
                    print(f"⚠️  RIFE failed: {str(e)[:100]}", flush=True)
            if enable_upscaling and upscale_model:
                try:
                    final_output = upscale_frames_streaming(
                        upscale_model, final_output, device, upscale_factor
                    )
                    final_height = height * upscale_factor
                    final_width = width * upscale_factor
                    flush_memory()
                except Exception as e:
                    print(f"⚠️  Upscale failed: {str(e)[:100]}", flush=True)
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
                    print(f"⚠️  Upscale failed: {str(e)[:100]}", flush=True)
            if enable_rife and rife_model:
                try:
                    final_output, final_fps = rife_frames_streaming(
                        rife_model, final_output, device, gen_fps, rife_target_fps
                    )
                    flush_memory()
                except Exception as e:
                    print(f"⚠️  RIFE failed: {str(e)[:100]}", flush=True)
        postprocess_time = time.time() - postprocess_start

    out_path = os.path.join(output_dir, f"wan_i2v_{job_id}.mp4")
    export_start = time.time()
    _export_video(final_output, out_path, fps=final_fps, video_quality=video_quality)
    export_time = time.time() - export_start

    total_time = generation_time + postprocess_time + export_time
    print(
        f"[{job_id}] ✓ {total_time:.1f}s (gen {generation_time:.1f}s, post {postprocess_time:.1f}s, "
        f"export {export_time:.1f}s) | {final_width}x{final_height} @ {final_fps}fps",
        flush=True,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return os.path.abspath(out_path)


if __name__ == "__main__":
    BENCH_MAIN_MODE = "folder"
    BENCH_FOLDER_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__))
    )

    _prm_h1 = """   
    The woman, starting from her exact clothed pose in the input image, slowly and seductively removes all her clothing piece by piece over the first several seconds, sliding off her top and bra to fully expose her large bare breasts with natural jiggle and soft skin texture, then slips off her bottoms to reveal her completely nude body and detailed vagina. She maintains her exact facial identity and features with no change whatsoever, identical seductive expression, same head position and gaze locked to the input frame. After undressing she gently runs her hands over her now-naked body in a teasing manner. Smooth fluid realistic motion, natural breast physics and subtle body sway, photorealistic skin details and lighting, maintaining the exact same camera angle, framing, perspective and overall composition throughout the clip with only a very subtle slow push-in.
     """

    _base_params: Dict[str, Any] = {
        "duration_seconds": 10.0,
        "guidance_scale": 1.0,
        "guidance_scale_2": 1.0,
        "flow_shift": 3.0,
        "video_quality": 9,
        "frame_multiplier": 32,
        "seed": 42,
        "enable_rife": True,
        "rife_target_fps": 32,
        "enable_upscaling": True,
        "enable_upscale": True,
        "upscale_factor": 2,
    }

    _script_dir = os.path.dirname(os.path.abspath(__file__))

    _bench_cases: List[Dict[str, Any]] = [
        {
            "image_id": "d1",
            "image_file": "d1.jpeg",
            "last_image_file": "d1_last.png",
            "prompt": _prm_h1.strip(),
            "fallback_url": None,
        },
        {
            "image_id": "d2",
            "image_file": "d2.jpeg",
            "last_image_file": "d2_last.png",
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
        {
            "image_id": "h6",
            "image_file": "h6.jpeg",
            "last_image_file": "h6_last.png",
            "prompt": _prm_h1.strip(),
            "fallback_url": None,
        },
    ]

    _BENCH_IMAGE_SUFFIXES = (
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".PNG",
        ".JPG",
        ".JPEG",
        ".WEBP",
    )

    def _bench_optional_last_in_same_dir(init_path: str) -> Optional[str]:
        d, base = os.path.split(os.path.abspath(init_path))
        stem, _ = os.path.splitext(base)
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            cand = os.path.join(d, f"{stem}_last{ext}")
            if os.path.isfile(cand):
                return cand
        return None

    def _bench_cases_from_folder(folder: str, shared_prompt: str) -> List[Dict[str, Any]]:
        root = os.path.abspath(os.path.expanduser(folder))
        if not os.path.isdir(root):
            print(f"[BENCH] folder mode: not a directory {root!r}; no cases.", flush=True)
            return []
        names = sorted(os.listdir(root))
        out: List[Dict[str, Any]] = []
        for name in names:
            if not name.endswith(_BENCH_IMAGE_SUFFIXES):
                continue
            fp = os.path.join(root, name)
            if not os.path.isfile(fp):
                continue
            stem, ext = os.path.splitext(name)
            image_id = f"{stem}_{ext.lower().lstrip('.')}" if ext else stem
            last_p = _bench_optional_last_in_same_dir(fp)
            out.append(
                {
                    "image_id": image_id,
                    "image_file": fp,
                    "last_image_file": last_p,
                    "prompt": shared_prompt,
                    "fallback_url": None,
                }
            )
        return out

    def _load_case_image(
        case: Dict[str, Any], script_dir: str
    ) -> tuple[Optional[Image.Image], str]:
        raw = case["image_file"]
        if isinstance(raw, str) and os.path.isfile(raw):
            return Image.open(raw).convert("RGB"), raw
        path = os.path.join(script_dir, raw)
        if os.path.isfile(path):
            return Image.open(path).convert("RGB"), path
        url = case.get("fallback_url")
        if url:
            with httpx.Client(timeout=60) as _c:
                content = _c.get(url).content
            return Image.open(BytesIO(content)).convert("RGB"), url
        return None, path

    def _load_case_last_image(
        case: Dict[str, Any], script_dir: str
    ) -> tuple[Optional[Image.Image], Optional[str]]:
        last_file = case.get("last_image_file")
        if not last_file:
            return None, None
        lf = str(last_file)
        if os.path.isfile(lf):
            return Image.open(lf).convert("RGB"), lf
        path = os.path.join(script_dir, lf)
        if os.path.isfile(path):
            return Image.open(path).convert("RGB"), path
        url = case.get("last_fallback_url")
        if url:
            with httpx.Client(timeout=60) as _c:
                content = _c.get(url).content
            return Image.open(BytesIO(content)).convert("RGB"), str(url)
        return None, path

    _step_counts = (6,)
    if isinstance(_step_counts, int):
        _step_counts = (_step_counts,)
    else:
        _step_counts = tuple(int(x) for x in _step_counts)

    _schedulers = sorted(SCHEDULER_MAP.keys())
    _bench_rows: List[Dict[str, Any]] = []

    if BENCH_MAIN_MODE == "folder":
        _cases_to_run = _bench_cases_from_folder(
            BENCH_FOLDER_PATH, _prm_h1.strip()
        )
        print(
            f"[BENCH] mode=folder path={os.path.abspath(os.path.expanduser(BENCH_FOLDER_PATH))!r} "
            f"→ {len(_cases_to_run)} image(s)",
            flush=True,
        )
    elif BENCH_MAIN_MODE == "cases":
        _cases_to_run = list(_bench_cases)
        print("[BENCH] mode=cases (explicit _bench_cases list)", flush=True)
    else:
        raise ValueError(
            f"BENCH_MAIN_MODE must be 'cases' or 'folder', got {BENCH_MAIN_MODE!r}"
        )

    _n_images = len(_cases_to_run)
    _n_schedulers = len(_schedulers)
    _n_step_variants = len(_step_counts)
    _n_runs = _n_images * _n_schedulers * _n_step_variants
    print(
        f"[BENCH] {_n_images} images × {_n_schedulers} schedulers × {_n_step_variants} step counts "
        f"= {_n_runs} runs (wall clock = full gen + post + export)",
        flush=True,
    )

    for _case in _cases_to_run:
        _iid = str(_case["image_id"])
        _img, _img_src = _load_case_image(_case, _script_dir)
        if _img is None:
            print(
                f"[BENCH] SKIP image_id={_iid!r} (missing {_case['image_file']!r}, no fallback_url)",
                flush=True,
            )
            continue
        _last_img, _last_src = _load_case_last_image(_case, _script_dir)
        _prm = str(_case["prompt"])
        _last_disp = _last_src if _last_img is not None else "none (no last frame)"
        print(f"[BENCH] --- image_id={_iid} init={_img_src!r} last={_last_disp} ---")

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
                    print(traceback.format_exc())
                    print(f"Error: {_e!r}")
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
        "bench_main_mode": BENCH_MAIN_MODE,
        "bench_folder_path": (
            os.path.abspath(os.path.expanduser(BENCH_FOLDER_PATH))
            if BENCH_MAIN_MODE == "folder"
            else None
        ),
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
            for c in _cases_to_run
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
        _w_sampler = max(
            len("scheduler"), max(len(r["scheduler"]) for r in _bench_rows)
        )
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
