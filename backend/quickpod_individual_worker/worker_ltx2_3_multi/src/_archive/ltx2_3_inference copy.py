"""
LTX-2.3 **distilled-checkpoint-only** Diffusers worker (v3).

Same inference recipe as v2: stage 1 uses ``DISTILLED_SIGMA_VALUES`` (8 steps, ``guidance_scale=1.0``); optional
stage 2 uses the **LTX-2.3 spatial 2× upscaler** weights from ``LTX2_3_SPATIAL_UPSAMPLER_REPO``, then short refine
with ``STAGE_2_DISTILLED_SIGMA_VALUES`` (default 3 steps).

Diffusers exposes this step as ``LTX2LatentUpsamplePipeline`` + ``LTX2LatentUpsamplerModel`` (latent-space 2×); v3
names the **job flags** ``use_spatial_upsampler`` to match the dedicated spatial-upscaler Hub repo layout. Weights
still load from ``subfolder`` ``latent_upsampler`` inside that repo unless overridden by
``LTX2_3_SPATIAL_UPSAMPLER_SUBFOLDER``.

All **base** model weights still come from the distilled repo (default: rockapaper SDNQ). SDNQ requires ``sdnq``.

**Checkpoint layout:** distilled snapshot must expose ``LTX2Pipeline``, ``LTX2ImageToVideoPipeline``, and
``LTX2ConditionPipeline``. If I2V-only, use ``ltx2_3_inference.py`` instead.

Modes
-----
- **t2v**: ``LTX2Pipeline``
- **i2v**: ``LTX2ImageToVideoPipeline`` (``init_image`` — **https URL** or **local file path**)
- **flf2v**: ``LTX2ConditionPipeline`` + first/last frame URLs
- **iv2v**: ``LTX2ConditionPipeline`` + ``condition_video`` + ``condition_image`` URLs

Job params
----------
- ``use_spatial_upsampler`` (aliases: ``use_spatial_upsample``, ``enable_spatial_upsampler``, ``spatial_upsample``):
  two-stage (8 + spatial upsample + 3). Default **true**. (v3 does **not** read ``use_latent_upsampler`` — use v2 if
  you need that key name.)
- ``stage1_num_inference_steps`` / ``stage2_num_inference_steps``: default **8** / **3**.
- ``num_inference_steps``: when upsampler off, single-stage step count (default 8); when on, alias for stage1 if
  ``stage1_num_inference_steps`` unset.
- **iv2v**: ``condition_video``, ``condition_image`` (URLs); indices / strengths as in v2.

**Optional Civitai LoRA (transformer):** ``civitai_lora_model_url`` or ``civitai_model_url`` (model page or
``.../api/download/models/{versionId}``). Resolves a **safetensors** file via the
`Civitai Public API <https://developer.civitai.com/docs/api/public-rest>`_ , downloads into ``LTX2_3_LORA_CACHE`` (or
``lora_cache_dir``). ``civitai_version_id`` pins a version when using a model page (defaults to newest **LTX 2.3**-ish
version when present, else newest publish). Community T2V+I2V LoRAs typically affect **both** denoising passes; use
``lora_apply``: ``both`` (default), ``stage1``, or ``stage2`` with ``lora_strength`` (0–1+). Some downloads need
``CIVITAI_API_KEY``. Not compatible with a second LoRA until you ``unload_lora_weights`` (we replace when URL/version
changes). **FP8 / SDNQ** distilled bases (default Hub id includes ``fp8``/``sdnq``) inject LoRA adapters in float8 storage;
PyTorch CUDA does not implement the matmul those PEFT linears use for ``Float8_e4m3fn``, so after ``load_lora_weights`` we
**cast injected LoRA parameters** on the transformer to the pipeline compute dtype (typically bfloat16). If you still see
dtype errors, point ``LTX2_3_DISTILLED_REPO`` (or ``LTX2_3_DISTILLED_LOCAL``) at a **non-quantized** Diffusers distilled
snapshot instead.

**i2v / flf2v / iv2v — conditioning resolution:** With **spatial upsampler** (two-stage) on, large conditioning can OOM;
by default we **downscale** to fit ``max_condition_width`` × ``max_condition_height`` (defaults **768** × **512**),
preserving aspect on a **32px grid** (LANCZOS), unless ``skip_condition_resolution_cap`` is set. With upsampler **off**
(single-stage), job ``width`` / ``height`` are ignored; we **downscale** (never upscale) so the longer side fits
``single_stage_max_condition_long_side`` (default **1024**, env ``LTX2_3_SINGLE_STAGE_MAX_CONDITION_LONG_SIDE``), uniform
scale, **32px** grid — enough resolution for most LoRA runs on **~24GB** without the 2× stage. Two-stage with cap skipped uses
the legacy **768×512** default when width/height are omitted.

**Two-stage + stage-2 VRAM:** before the 2× refine denoise you can apply **stage-2-only** CPU offload (**sequential** or
**model**) on top of whatever main ``offload`` mode you use (``none``, ``model``, ``sequential``, or ``auto``). The
pipeline’s hooks are removed, stage-2 offload is applied for that forward, then hooks are cleared and **stage 1’s**
offload is restored (full ``.to(cuda)`` when main offload is ``none``). Configure with ``stage2_offload`` (``sequential``,
``model``, ``none``) or env ``LTX2_3_STAGE2_OFFLOAD`` (legacy ``stage2_sequential_offload`` false ⇒ ``none``). Default
stage-2 mode when unset: **sequential**.
Only **one** main pipeline (t2v vs i2v vs condition) can stay on the GPU at a time with offload none—loading another
type **evicts** the cached one first so ``--mode all`` does not OOM on a second ``from_pretrained``/``.to(cuda)``.

**Profiling / HF-gap parity:** set ``profile``/``LTX2_3_PROFILE`` for phase timing + VRAM logs; ``lora_fusion_cache``/
``LTX2_3_LORA_FUSION_CACHE`` saves LoRA tensor subset under the Civitai cache dir; ``encode_video_chunked`` or
``video_chunks_number`` forwards chunked mux when diffusers ``encode_video`` supports it. Native HF-style path:
``ltx2_3_inference_hf_parity`` (``ltx_pipelines`` + fallback to this module).

Entrypoints: ``run_inference_ltx2_3_t2v``, ``run_inference_ltx2_3_i2v_distilled``, ``run_inference_ltx2_3_flf2v``,
``run_inference_ltx2_3_iv2v_distilled``.
"""

from __future__ import annotations

import gc
import hashlib
import math
import os
import random
import re
import tempfile
import time
from contextlib import contextmanager
from io import BytesIO
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests
import torch
from PIL import Image

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_OUTPUT_DIR = os.environ.get("WORKSPACE_DIR", "/workspace")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Set by ``python ... --offload <mode>`` (non-auto). Empty means use ``LTX2_3_OFFLOAD_MODE`` env or ``auto``.
_OFFLOAD_MODE = ""
_STAGE2_OFFLOAD_CLI: Optional[str] = None

DISTILLED_REPO = os.environ.get(
    "LTX2_3_DISTILLED_REPO",
    "rockapaper/LTX-2.3-Distilled-Diffusers_tenc_fp8_sdnq_r64_s16",
).strip()
SPATIAL_UPSAMPLER_REPO = os.environ.get(
    "LTX2_3_SPATIAL_UPSAMPLER_REPO", "dg845/LTX-2.3-Spatial-Upsampler-Diffusers"
).strip()

ENV_DISTILLED_LOCAL = "LTX2_3_DISTILLED_LOCAL"
ENV_SPATIAL_LOCAL = "LTX2_3_SPATIAL_UPSAMPLER_LOCAL"
# HF repo ``dg845/LTX-2.3-Spatial-Upsampler-Diffusers`` stores weights under ``latent_upsampler/`` in Diffusers layout.
ENV_SPATIAL_UPSAMPLER_SUBFOLDER = "LTX2_3_SPATIAL_UPSAMPLER_SUBFOLDER"

DEFAULT_NEGATIVE = (
    "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, "
    "fused fingers, bad anatomy, weird hand, ugly, transition, static."
)

# Max canvas for conditioning (i2v / flf2v / iv2v): matches common Diffusers LTX-2.3 examples; reduces VRAM vs HD inputs.
DEFAULT_MAX_CONDITION_W = 768
DEFAULT_MAX_CONDITION_H = 512
# Single-stage (no spatial upsampler): fit conditioning inside this square so longer side <= cap (after 32px grid).
DEFAULT_SINGLE_STAGE_MAX_CONDITION_LONG_SIDE = 1024

_CACHE: Dict[str, Any] = {}


def _local_dir_if_any(path_or_id: str) -> Optional[str]:
    if not path_or_id or not isinstance(path_or_id, str):
        return None
    p = path_or_id.strip()
    if p.startswith(("http://", "https://")):
        return None
    if os.path.isdir(p):
        return os.path.abspath(p)
    return None


def _path_looks_sdnq_quantized(model_dir: str) -> bool:
    if not model_dir or not os.path.isdir(model_dir):
        return False
    for rel in ("quantization_config.json", "transformer/quantization_config.json"):
        if os.path.isfile(os.path.join(model_dir, rel)):
            return True
    return False


def _import_sdnq_register() -> None:
    from sdnq import SDNQConfig  # noqa: F401


def _source_needs_sdnq(source: str) -> bool:
    if not source or not isinstance(source, str):
        return False
    local = _local_dir_if_any(source)
    if local is not None:
        return _path_looks_sdnq_quantized(local)
    s = source.strip().lower()
    if "sdnq" in s:
        return True
    return str(os.environ.get("LTX2_3_ASSUME_SDNQ_WEIGHTS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _ensure_sdnq_registered_for_dir(model_dir: str) -> None:
    if not _path_looks_sdnq_quantized(model_dir):
        return
    try:
        _import_sdnq_register()
    except ImportError as e:
        raise ImportError(
            "SDNQ-quantized weights found but `sdnq` is not importable. Install from "
            "https://github.com/Disty0/sdnq or use full-precision local weights."
        ) from e


def _ensure_sdnq_for_pretrained_source(source: str) -> None:
    if not _source_needs_sdnq(source):
        return
    try:
        _import_sdnq_register()
    except ImportError as e:
        raise ImportError(
            "LTX2.3 v3 is loading SDNQ weights but `sdnq` is not importable."
        ) from e


def _retie_sdnq_t5_text_encoders(pipe: Any) -> None:
    for name in ("text_encoder", "text_encoder_2"):
        te = getattr(pipe, name, None)
        if te is None:
            continue
        if not (
            hasattr(te, "shared")
            and hasattr(te, "encoder")
            and hasattr(te.encoder, "embed_tokens")
        ):
            continue
        try:
            shared_ptr = te.shared.weight.data_ptr()
            embed_ptr = te.encoder.embed_tokens.weight.data_ptr()
        except Exception:
            continue
        if shared_ptr != embed_ptr:
            te.encoder.embed_tokens.weight = te.shared.weight
        if hasattr(te, "tie_weights"):
            try:
                te.tie_weights()
            except Exception:
                pass


def _from_pretrained_ltx(pipe_cls: Any, source: str, **kwargs: Any) -> Any:
    _ensure_sdnq_for_pretrained_source(source)
    local = _local_dir_if_any(source)
    if local is not None:
        _ensure_sdnq_registered_for_dir(local)
    pipe = pipe_cls.from_pretrained(source, **kwargs)
    if _source_needs_sdnq(source):
        _retie_sdnq_t5_text_encoders(pipe)
    return pipe


def _parse_bool(v: Any, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


# Diffusers worker vs Hugging Face Space (``ltx_pipelines`` + Gemma). See ``ltx2_3_inference_hf_parity`` for native path.
DIFFUSERS_VS_HF_SPACE: Tuple[str, ...] = (
    "enhance_prompt (Gemma via ``encode_prompts`` in ltx_pipelines) is not exposed on ``LTX2*`` Diffusers pipelines here.",
    "User ``audio_path`` driving joint denoise (HF ``LTX23DistilledA2VPipeline``) is not wired; Diffusers path synthesizes audio via the vocoder only.",
    "HF multi-LoRA CPU fuse + single ``copy_`` apply differs from this file’s single Civitai adapter via ``load_lora_weights`` (optional ``lora_fusion_cache`` saves LoRA tensor subset to disk).",
    "Chunked ``encode_video``: ``video_chunks_number`` is forwarded when the installed diffusers ``export_utils.encode_video`` accepts it.",
)


def _profile_on(p: Optional[Dict[str, Any]] = None) -> bool:
    if p is not None and p.get("profile") is not None:
        return _parse_bool(p.get("profile"), False)
    return _parse_bool(os.environ.get("LTX2_3_PROFILE"), False)


def _log_vram(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    free_b, total_b = torch.cuda.mem_get_info()
    print(
        f"[VRAM {tag}] allocated={alloc:.2f}GB peak={peak:.2f}GB "
        f"free={free_b / 1024**3:.2f}GB total={total_b / 1024**3:.2f}GB"
    )


@contextmanager
def _profile_phase(tag: str, p: Optional[Dict[str, Any]] = None) -> Iterator[None]:
    prof = _profile_on(p)
    if prof and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter() if prof else 0.0
    if prof:
        print(f"[PROFILE] begin {tag}")
        _log_vram(f"{tag}_before")
    try:
        yield
    finally:
        if prof:
            dt = time.perf_counter() - t0
            print(f"[PROFILE] end {tag} in {dt:.2f}s")
            _log_vram(f"{tag}_after")


def _resolve_pretrained(repo_or_local: str, env_local: str) -> str:
    local = os.environ.get(env_local, "").strip()
    if local and os.path.isdir(local):
        return local
    return repo_or_local


def _align_frames_8k_plus_1(n: int) -> int:
    n = max(9, int(n))
    return ((n - 1) // 8) * 8 + 1


def _align_spatial(w: int, h: int) -> Tuple[int, int]:
    w = max(32, int(w))
    h = max(32, int(h))
    w = (w // 32) * 32
    h = (h // 32) * 32
    return w, h


def _max_condition_canvas_pixels(p: Dict[str, Any]) -> Tuple[int, int]:
    mw = p.get("max_condition_width")
    mh = p.get("max_condition_height")
    if mw is None:
        mw = os.environ.get("LTX2_3_MAX_CONDITION_WIDTH", str(DEFAULT_MAX_CONDITION_W))
    if mh is None:
        mh = os.environ.get("LTX2_3_MAX_CONDITION_HEIGHT", str(DEFAULT_MAX_CONDITION_H))
    return _align_spatial(int(mw), int(mh))


def _condition_resolution_cap_enabled(p: Dict[str, Any]) -> bool:
    if p.get("skip_condition_resolution_cap") is not None:
        return not _parse_bool(p.get("skip_condition_resolution_cap"), False)
    return not _parse_bool(
        os.environ.get("LTX2_3_SKIP_CONDITION_RESOLUTION_CAP"), False
    )


def _single_stage_max_condition_long_side_pixels(p: Dict[str, Any]) -> int:
    """Aligned edge length for ``_scale_to_max_canvas(w,h, cap, cap)`` when spatial upsampler is off."""
    raw = p.get("single_stage_max_condition_long_side")
    if raw is None:
        raw = os.environ.get(
            "LTX2_3_SINGLE_STAGE_MAX_CONDITION_LONG_SIDE",
            str(DEFAULT_SINGLE_STAGE_MAX_CONDITION_LONG_SIDE),
        )
    cap = max(32, int(raw))
    cap = min(cap, 8192)
    return max(32, (cap // 32) * 32)


def _scale_to_max_canvas(w0: int, h0: int, max_w: int, max_h: int) -> Tuple[int, int]:
    """Uniform scale only (never upscale); fit inside max_w×max_h; 32px grid; aspect preserved; no padding."""
    w0, h0 = max(1, int(w0)), max(1, int(h0))
    cap_w = max(32, (int(max_w) // 32) * 32)
    cap_h = max(32, (int(max_h) // 32) * 32)
    scale = min(cap_w / w0, cap_h / h0, 1.0)
    if scale >= 1.0 - 1e-9:
        return _align_spatial(w0, h0)
    nw = max(32, int(math.floor(w0 * scale / 32.0)) * 32)
    nh = max(32, int(math.floor(nw * float(h0) / float(w0) / 32.0)) * 32)
    if nh > cap_h:
        nh = cap_h
        nw = max(32, int(math.floor(nh * float(w0) / float(h0) / 32.0)) * 32)
    if nw > cap_w:
        nw = cap_w
        nh = max(32, int(math.floor(nw * float(h0) / float(w0) / 32.0)) * 32)
    return nw, nh


def _resize_rgb_image(im: Image.Image, tw: int, th: int) -> Image.Image:
    if im.size == (tw, th):
        return im.convert("RGB")
    return im.convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)


def _resize_condition_video_frames(frames: Any, tw: int, th: int) -> Any:
    """Resize each frame to (tw, th); supports PIL list/tuple or single PIL."""
    if isinstance(frames, Image.Image):
        return _resize_rgb_image(frames, tw, th)
    if isinstance(frames, list):
        return [_resize_frame_to_wh(f, tw, th) for f in frames]
    if isinstance(frames, tuple):
        return tuple(_resize_frame_to_wh(f, tw, th) for f in frames)
    return frames


def _resize_frame_to_wh(frame: Any, tw: int, th: int) -> Any:
    if isinstance(frame, Image.Image):
        return _resize_rgb_image(frame, tw, th)
    try:
        import numpy as np

        if isinstance(frame, np.ndarray):
            im = Image.fromarray(frame)
            out = _resize_rgb_image(im, tw, th)
            return np.array(out)
    except ImportError:
        pass
    return frame


def _spatial_upsample_enabled(p: Dict[str, Any], *, default: bool) -> bool:
    for key in (
        "use_spatial_upsampler",
        "use_spatial_upsample",
        "enable_spatial_upsampler",
        "spatial_upsample",
    ):
        if key in p and p[key] is not None:
            return _parse_bool(p.get(key), default)
    return default


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _evict_cached_spatial_upsampler_if_full_gpu(pipe: Any) -> None:
    """When the full model lives on GPU (``offload=none``), drop the cached ``LTX2LatentUpsamplePipeline`` so its weights
    can be freed before stage-2 denoise at 2× resolution (often needed on 24–32GB). The upsampler is re-downloaded from
    cache on the next job that needs it.
    """
    if DEVICE != "cuda" or _effective_offload_mode() != "none":
        return
    vae = getattr(pipe, "vae", None)
    if vae is None:
        return
    _CACHE.pop(("spatial_upsample", id(vae)), None)
    _cleanup_cuda()


def _effective_offload_mode() -> str:
    """Return ``none`` | ``sequential`` | ``model`` | ``auto``. ``LTX2_3_DISABLE_OFFLOAD`` forces ``none``."""
    if _parse_bool(os.environ.get("LTX2_3_DISABLE_OFFLOAD"), False):
        return "none"
    raw = (
        (_OFFLOAD_MODE or os.environ.get("LTX2_3_OFFLOAD_MODE", "auto")).strip().lower()
    )
    if raw in ("", "default"):
        raw = "auto"
    if raw not in ("auto", "none", "sequential", "model"):
        raw = "auto"
    return raw


def _apply_sequential_offload(pipe: Any) -> None:
    if DEVICE != "cuda":
        return
    mode = _effective_offload_mode()
    if mode == "none":
        return
    if mode == "auto":
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
            return
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload(device=torch.device("cuda"))
        return
    if mode == "sequential":
        if not hasattr(pipe, "enable_sequential_cpu_offload"):
            raise RuntimeError(
                "offload=sequential requested but this pipeline has no enable_sequential_cpu_offload"
            )
        pipe.enable_sequential_cpu_offload()
        return
    if mode == "model":
        if not hasattr(pipe, "enable_model_cpu_offload"):
            raise RuntimeError(
                "offload=model requested but this pipeline has no enable_model_cpu_offload"
            )
        pipe.enable_model_cpu_offload(device=torch.device("cuda"))
        return


def _move_pipeline_to_cuda_when_no_offload(pipe: Any) -> None:
    """If CPU offload is off, ``from_pretrained`` leaves modules on CPU; ``prepare_latents`` then uses a CPU device
    while this worker passes a CUDA ``torch.Generator``, which raises in ``randn_tensor``.
    """
    if DEVICE != "cuda":
        return
    if _effective_offload_mode() != "none":
        return
    if not hasattr(pipe, "to"):
        return
    pipe.to(device=torch.device("cuda"))


def _normalize_stage2_offload_mode(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("sequential", "seq", "s"):
        return "sequential"
    if s in ("model", "m", "model_cpu"):
        return "model"
    if s in ("none", "off", "no", "false", "0"):
        return "none"
    return None


def _stage2_offload_mode(p: Optional[Dict[str, Any]] = None) -> str:
    """``sequential`` | ``model`` | ``none`` for stage-2-only offload (works with any main ``offload`` mode)."""
    if _STAGE2_OFFLOAD_CLI is not None:
        v = _normalize_stage2_offload_mode(_STAGE2_OFFLOAD_CLI)
        if v is not None:
            return v
    if p is not None:
        v = _normalize_stage2_offload_mode(p.get("stage2_offload"))
        if v is not None:
            return v
        if p.get("stage2_sequential_offload") is not None and not _parse_bool(
            p.get("stage2_sequential_offload"), True
        ):
            return "none"
    env_m = (os.environ.get("LTX2_3_STAGE2_OFFLOAD") or "").strip().lower()
    if env_m in ("sequential", "model", "none"):
        return env_m
    if not _parse_bool(os.environ.get("LTX2_3_STAGE2_SEQUENTIAL_OFFLOAD", "1"), True):
        return "none"
    return "sequential"


def _prepare_pipe_for_stage2_cpu_offload(
    pipe: Any, p: Optional[Dict[str, Any]] = None
) -> str:
    """Returns *none*, *sequential*, or *model* for paired restore.

    Drops stage-1 offload hooks (if any), applies stage-2 offload for the 2× refine forward. Use with
    ``_restore_pipe_full_gpu_after_stage2_cpu_offload`` so stage-1 offload (or full GPU) is restored.
    """
    if DEVICE != "cuda":
        return "none"
    kind = _stage2_offload_mode(p)
    if kind == "none":
        return "none"
    if hasattr(pipe, "remove_all_hooks"):
        pipe.remove_all_hooks()
    if kind == "sequential":
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload(device=torch.device("cuda"))
            return "sequential"
        return "none"
    if kind == "model":
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload(device=torch.device("cuda"))
            return "model"
        return "none"
    return "none"


def _restore_pipe_full_gpu_after_stage2_cpu_offload(pipe: Any, applied: str) -> None:
    if applied == "none":
        return
    if hasattr(pipe, "remove_all_hooks"):
        pipe.remove_all_hooks()
    if _effective_offload_mode() == "none":
        if hasattr(pipe, "to"):
            pipe.to(device=torch.device("cuda"))
    else:
        _apply_sequential_offload(pipe)
    _cleanup_cuda()


CIVITAI_API_BASE = "https://civitai.com/api/v1"
_CIVITAI_MODEL_PATH_RE = re.compile(
    r"civitai\.com/(?:models/|api/v1/models/)(\d+)", re.I
)
_CIVITAI_DOWNLOAD_VER_RE = re.compile(r"civitai\.com/api/download/models/(\d+)", re.I)


def _civitai_auth_headers(p: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    tok = ""
    if p is not None and p.get("civitai_api_token"):
        tok = str(p.get("civitai_api_token")).strip()
    if not tok:
        tok = (
            os.environ.get("CIVITAI_API_KEY")
            or os.environ.get("CIVITAI_API_TOKEN")
            or ""
        ).strip()
    if not tok:
        return {}
    return {"Authorization": f"Bearer {tok}"}


def _lora_cache_dir_from_params(p: Dict[str, Any]) -> str:
    v = (p.get("lora_cache_dir") or os.environ.get("LTX2_3_LORA_CACHE") or "").strip()
    if v:
        return v
    return os.path.join(DEFAULT_OUTPUT_DIR, "ltx2_3_lora_cache")


def _lora_fusion_cache_enabled(p: Dict[str, Any]) -> bool:
    if p.get("lora_fusion_cache") is not None:
        return _parse_bool(p.get("lora_fusion_cache"), False)
    return _parse_bool(os.environ.get("LTX2_3_LORA_FUSION_CACHE"), False)


def _lora_fusion_tensor_cache_path(p: Dict[str, Any], sig: str) -> str:
    h = hashlib.sha256(sig.encode("utf-8")).hexdigest()[:24]
    return os.path.join(_lora_cache_dir_from_params(p), f"lora_tensors_{h}.pt")


def _save_lora_fusion_tensor_cache(pipe: Any, p: Dict[str, Any], sig: str) -> None:
    if not _lora_fusion_cache_enabled(p):
        return
    tr = getattr(pipe, "transformer", None)
    if tr is None:
        return
    path = _lora_fusion_tensor_cache_path(p, sig)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = {
        k: v.detach().cpu().clone()
        for k, v in tr.state_dict().items()
        if "lora" in k.lower() and isinstance(v, torch.Tensor)
    }
    if not d:
        return
    torch.save(d, path)
    if _profile_on(p):
        print(f"[lora_fusion_cache] saved {len(d)} tensors -> {path}")


def _apply_lora_fusion_tensor_cache(pipe: Any, p: Dict[str, Any], sig: str) -> bool:
    """Overlay cached LoRA tensors onto the transformer after ``load_lora_weights`` (optional speed / dtype path)."""
    if not _lora_fusion_cache_enabled(p):
        return False
    path = _lora_fusion_tensor_cache_path(p, sig)
    if not os.path.isfile(path):
        return False
    tr = getattr(pipe, "transformer", None)
    if tr is None:
        return False
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    except Exception:
        return False
    if not isinstance(blob, dict):
        return False
    params = dict(tr.named_parameters())
    with torch.no_grad():
        for k, v in blob.items():
            if k not in params:
                continue
            t = params[k]
            try:
                t.data.copy_(v.to(device=t.device, dtype=t.dtype))
            except Exception:
                continue
    if _profile_on(p):
        print(f"[lora_fusion_cache] applied tensor overlay from {path}")
    return True


def _civitai_primary_safetensors_name(version_obj: Dict[str, Any]) -> str:
    files = version_obj.get("files") or []
    for f in files:
        if f.get("primary") and str(f.get("name", "")).lower().endswith(".safetensors"):
            return str(f["name"])
    for f in files:
        if str(f.get("name", "")).lower().endswith(".safetensors"):
            return str(f["name"])
    raise ValueError("Civitai version has no .safetensors file")


def _pick_civitai_version(
    model_json: Dict[str, Any], version_id: Optional[int]
) -> Dict[str, Any]:
    versions = list(model_json.get("modelVersions") or [])
    if not versions:
        raise ValueError("Civitai model has no published versions")
    if version_id is not None:
        for v in versions:
            if int(v.get("id", 0)) == int(version_id):
                return v
        raise ValueError(
            f"civitai_version_id {version_id} not in model {model_json.get('id')}"
        )
    prefer = [v for v in versions if "2.3" in str(v.get("baseModel", ""))]
    pool = prefer if prefer else versions
    return sorted(pool, key=lambda v: str(v.get("publishedAt") or ""), reverse=True)[0]


def _civitai_download_version(
    version_id: int, weight_basename: str, cache_dir: str, headers: Dict[str, str]
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", weight_basename)
    dest = os.path.join(cache_dir, f"civitai_v{version_id}_{safe}")
    if os.path.isfile(dest) and os.path.getsize(dest) > 4096:
        return dest
    url = f"https://civitai.com/api/download/models/{int(version_id)}"
    r = requests.get(url, headers=headers, stream=True, timeout=600)
    if r.status_code == 401:
        r.close()
        raise RuntimeError(
            "Civitai returned 401 — set CIVITAI_API_KEY (or pass civitai_api_token in job params) for this download."
        )
    r.raise_for_status()
    tmp = dest + ".partial"
    try:
        with open(tmp, "wb") as out:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out.write(chunk)
        os.replace(tmp, dest)
    finally:
        if os.path.isfile(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return dest


def _resolve_civitai_lora_file(p: Dict[str, Any]) -> Tuple[str, int]:
    """Download or reuse cached LoRA; return ``(absolute_path_to_safetensors, version_id)``."""
    raw = (p.get("civitai_lora_model_url") or p.get("civitai_model_url") or "").strip()
    if not raw:
        raise ValueError(
            "civitai_lora_model_url / civitai_model_url required for Civitai LoRA"
        )
    headers = _civitai_auth_headers(p)
    cache_dir = _lora_cache_dir_from_params(p)

    mv = _CIVITAI_DOWNLOAD_VER_RE.search(raw)
    if mv:
        version_id = int(mv.group(1))
        vr = requests.get(
            f"{CIVITAI_API_BASE}/model-versions/{version_id}",
            headers=headers,
            timeout=120,
        )
        vr.raise_for_status()
        vdata = vr.json()
        wn = _civitai_primary_safetensors_name(vdata)
    else:
        mm = _CIVITAI_MODEL_PATH_RE.search(raw)
        if not mm:
            raise ValueError(
                f"civitai URL not recognized (need .../models/{{id}} or .../download/models/{{versionId}}): {raw[:120]}"
            )
        model_id = int(mm.group(1))
        mr = requests.get(
            f"{CIVITAI_API_BASE}/models/{model_id}", headers=headers, timeout=120
        )
        mr.raise_for_status()
        mdata = mr.json()
        want_v = p.get("civitai_version_id")
        want_v_i = int(want_v) if want_v is not None else None
        ver = _pick_civitai_version(mdata, want_v_i)
        version_id = int(ver["id"])
        wn = _civitai_primary_safetensors_name(ver)

    local = _civitai_download_version(version_id, wn, cache_dir, headers)
    return os.path.abspath(local), version_id


def _is_float8_dtype(dt: torch.dtype) -> bool:
    s = str(dt).lower()
    if "float8" in s:
        return True
    for attr in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2"):
        t = getattr(torch, attr, None)
        if t is not None and dt == t:
            return True
    return False


def _cast_transformer_lora_params_to_compute_dtype(
    transformer: Any, compute_dtype: torch.dtype
) -> None:
    """PEFT LoRA linears can inherit float8 storage from FP8/SDNQ transformers; cast only ``*lora*`` params."""
    if transformer is None:
        return
    with torch.no_grad():
        for name, param in transformer.named_parameters():
            if "lora" not in name.lower():
                continue
            if not param.is_floating_point():
                continue
            if _is_float8_dtype(param.dtype):
                param.data = param.data.to(dtype=compute_dtype)


def _set_lora_adapter_scale(pipe: Any, adapter: Optional[str], scale: float) -> None:
    if not adapter or not hasattr(pipe, "set_adapters"):
        return
    w = float(scale)
    try:
        pipe.set_adapters(adapter, w)
    except TypeError:
        pipe.set_adapters([adapter], [w])


def _lora_stage_strengths(p: Dict[str, Any]) -> Tuple[float, float]:
    apply = str(p.get("lora_apply", "both")).strip().lower()
    base = float(p.get("lora_strength", 1.0))
    o1, o2 = p.get("lora_strength_stage1"), p.get("lora_strength_stage2")
    if o1 is not None:
        s1 = float(o1)
    elif apply == "stage2":
        s1 = 0.0
    else:
        s1 = base
    if o2 is not None:
        s2 = float(o2)
    elif apply == "stage1":
        s2 = 0.0
    else:
        s2 = base
    return s1, s2


def _lora_strength_single_pass(p: Dict[str, Any]) -> float:
    s1, s2 = _lora_stage_strengths(p)
    apply = str(p.get("lora_apply", "both")).strip().lower()
    if apply == "stage2":
        return s2
    if apply == "stage1":
        return s1
    return s1


def _lora_params_requested(p: Dict[str, Any]) -> bool:
    return bool(
        (p.get("civitai_lora_model_url") or p.get("civitai_model_url") or "").strip()
    )


def _sync_civitai_lora_on_pipe(pipe: Any, p: Dict[str, Any]) -> Optional[str]:
    """Load / unload Civitai LoRA on ``pipe``; return adapter name when active."""
    adapter = str(p.get("lora_adapter_name", "civitai_lora") or "civitai_lora").strip()
    if not _lora_params_requested(p):
        if getattr(pipe, "_ltx_civitai_sig", None) and hasattr(
            pipe, "unload_lora_weights"
        ):
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass
            pipe._ltx_civitai_sig = None
        return None

    local_path, version_id = _resolve_civitai_lora_file(p)
    sig = f"v{version_id}:{os.path.basename(local_path)}"
    if getattr(pipe, "_ltx_civitai_sig", None) == sig:
        return adapter

    if hasattr(pipe, "unload_lora_weights"):
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
    pipe._ltx_civitai_sig = None

    folder = os.path.dirname(local_path)
    weight_name = os.path.basename(local_path)
    pipe.load_lora_weights(folder, weight_name=weight_name, adapter_name=adapter)
    _cast_transformer_lora_params_to_compute_dtype(
        getattr(pipe, "transformer", None), DTYPE
    )
    _apply_lora_fusion_tensor_cache(pipe, p, sig)
    _save_lora_fusion_tensor_cache(pipe, p, sig)
    pipe._ltx_civitai_sig = sig
    return adapter


def _apply_lora_strength(pipe: Any, adapter: Optional[str], strength: float) -> None:
    _set_lora_adapter_scale(pipe, adapter, strength)


def _vae_tiling(pipe: Any) -> None:
    vae = getattr(pipe, "vae", None)
    if vae is not None and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()


def _open_image_uri(uri: str) -> Image.Image:
    u = uri.strip()
    if u.startswith("http://") or u.startswith("https://"):
        r = requests.get(u, timeout=180)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    if os.path.isfile(u):
        return Image.open(u).convert("RGB")
    raise ValueError(f"image path not found or unsupported URI: {uri!r}")


def _load_video_uri(uri: str) -> Any:
    """Return diffusers-compatible video (e.g. list of PIL), using temp file for remote bytes."""
    from diffusers.utils import load_video

    u = uri.strip()
    if u.startswith("http://") or u.startswith("https://"):
        r = requests.get(u, timeout=300)
        r.raise_for_status()
        suf = os.path.splitext(u.split("?")[0])[-1] or ".mp4"
        if suf.lower() not in {".mp4", ".webm", ".mov", ".mkv"}:
            suf = ".mp4"
        fd, path = tempfile.mkstemp(suffix=suf)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(r.content)
            return load_video(path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    if os.path.isfile(u):
        return load_video(u)
    raise ValueError(f"video path not found or unsupported URI: {uri!r}")


def get_gpu_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "vram_gb": 0.0, "name": "CPU"}
    p = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": p.name,
        "vram_gb": p.total_memory / 1024**3,
        "compute_capability": f"{p.major}.{p.minor}",
    }


def unload_pipelines() -> None:
    _CACHE.clear()
    _cleanup_cuda()


def _evict_all_cached_pipelines_if_full_gpu_for_new_load() -> None:
    """With ``offload=none``, one distilled LTX snapshot on CUDA can fill the whole card. Loading a *second* pipeline
    (e.g. i2v after t2v) would OOM on ``pipe.to(cuda)`` unless we drop the previous cached pipe first.
    """
    if DEVICE != "cuda" or _effective_offload_mode() != "none":
        return
    if not _CACHE:
        return
    _CACHE.clear()
    _cleanup_cuda()


def _distilled_source() -> str:
    return _resolve_pretrained(DISTILLED_REPO, ENV_DISTILLED_LOCAL)


def _get_t2v_pipe() -> Any:
    key = "t2v_distilled"
    if key in _CACHE:
        return _CACHE[key]
    _evict_all_cached_pipelines_if_full_gpu_for_new_load()
    from diffusers import LTX2Pipeline

    src = _distilled_source()
    pipe = _from_pretrained_ltx(LTX2Pipeline, src, torch_dtype=DTYPE)
    _apply_sequential_offload(pipe)
    _move_pipeline_to_cuda_when_no_offload(pipe)
    _vae_tiling(pipe)
    _CACHE[key] = pipe
    return pipe


def _get_condition_pipe() -> Any:
    key = "condition_distilled"
    if key in _CACHE:
        return _CACHE[key]
    _evict_all_cached_pipelines_if_full_gpu_for_new_load()
    from diffusers import LTX2ConditionPipeline

    src = _distilled_source()
    pipe = _from_pretrained_ltx(LTX2ConditionPipeline, src, torch_dtype=DTYPE)
    _apply_sequential_offload(pipe)
    _move_pipeline_to_cuda_when_no_offload(pipe)
    _vae_tiling(pipe)
    _CACHE[key] = pipe
    return pipe


def _get_i2v_pipe() -> Any:
    key = "i2v_distilled"
    if key in _CACHE:
        return _CACHE[key]
    _evict_all_cached_pipelines_if_full_gpu_for_new_load()
    from diffusers import LTX2ImageToVideoPipeline

    src = _distilled_source()
    pipe = _from_pretrained_ltx(LTX2ImageToVideoPipeline, src, torch_dtype=DTYPE)
    _apply_sequential_offload(pipe)
    _move_pipeline_to_cuda_when_no_offload(pipe)
    _vae_tiling(pipe)
    _CACHE[key] = pipe
    return pipe


def _get_spatial_upsample_pipe(vae: Any) -> Any:
    """2× step using LTX-2.3 spatial upscaler weights (Diffusers ``LTX2LatentUpsamplePipeline`` API)."""
    key = ("spatial_upsample", id(vae))
    if key in _CACHE:
        return _CACHE[key]
    from diffusers import LTX2LatentUpsamplePipeline
    from diffusers.pipelines.ltx2 import LTX2LatentUpsamplerModel

    root = _resolve_pretrained(SPATIAL_UPSAMPLER_REPO, ENV_SPATIAL_LOCAL)
    _ensure_sdnq_for_pretrained_source(root)
    local_spatial = _local_dir_if_any(root)
    if local_spatial is not None:
        _ensure_sdnq_registered_for_dir(local_spatial)
    subfolder = (
        os.environ.get(ENV_SPATIAL_UPSAMPLER_SUBFOLDER, "") or "latent_upsampler"
    ).strip()
    spatial_model = LTX2LatentUpsamplerModel.from_pretrained(
        root,
        subfolder=subfolder,
        torch_dtype=DTYPE,
    )
    upsample_pipe = LTX2LatentUpsamplePipeline(vae=vae, latent_upsampler=spatial_model)
    _apply_sequential_offload(upsample_pipe)
    _move_pipeline_to_cuda_when_no_offload(upsample_pipe)
    _CACHE[key] = upsample_pipe
    return upsample_pipe


def _encode_video_extra_kwargs(p: Optional[Dict[str, Any]], num_frames: int) -> Dict[str, Any]:
    """Optional ``video_chunks_number`` for diffusers ``encode_video`` when supported (HF-style chunking)."""
    if not p:
        return {}
    raw = p.get("video_chunks_number")
    if raw is None and _parse_bool(p.get("encode_video_chunked"), False):
        raw = max(1, (int(num_frames) + 63) // 64)
    if raw is None:
        return {}
    return {"video_chunks_number": int(raw)}


def _encode_out(
    video_np: Any,
    audio_tensor: Any,
    vocoder: Any,
    frame_rate: float,
    output_path: str,
    job_params: Optional[Dict[str, Any]] = None,
    *,
    num_frames: int = 0,
) -> None:
    import inspect

    from diffusers.pipelines.ltx2.export_utils import encode_video

    sr = int(vocoder.config.output_sampling_rate)
    kwargs: Dict[str, Any] = {
        "fps": frame_rate,
        "audio": audio_tensor.float().cpu(),
        "audio_sample_rate": sr,
        "output_path": output_path,
    }
    extra = _encode_video_extra_kwargs(job_params, num_frames or 1)
    try:
        sig = inspect.signature(encode_video)
        for k in list(extra.keys()):
            if k not in sig.parameters:
                extra.pop(k, None)
    except (TypeError, ValueError):
        extra = {}
    if extra and _profile_on(job_params):
        print(f"[encode_video] extra kwargs: {extra}")
    encode_video(video_np, **kwargs, **extra)


def _distilled_stage1_kwargs() -> Dict[str, Any]:
    return {
        "guidance_scale": 1.0,
        "stg_scale": 0.0,
        "modality_scale": 1.0,
        "guidance_rescale": 0.0,
        "audio_guidance_scale": 1.0,
        "audio_stg_scale": 0.0,
        "audio_modality_scale": 1.0,
        "audio_guidance_rescale": 0.0,
        "spatio_temporal_guidance_blocks": None,
        "use_cross_timestep": True,
    }


def _distilled_stage2_kwargs() -> Dict[str, Any]:
    return dict(_distilled_stage1_kwargs())


def _stage1_steps(p: Dict[str, Any], *, upsample: bool) -> int:
    if upsample:
        if p.get("stage1_num_inference_steps") is not None:
            return int(p["stage1_num_inference_steps"])
        if p.get("num_inference_steps") is not None:
            return int(p["num_inference_steps"])
        return 8
    if p.get("num_inference_steps") is not None:
        return int(p["num_inference_steps"])
    return 8


def _stage2_steps(p: Dict[str, Any]) -> int:
    return int(p.get("stage2_num_inference_steps", 3))


def run_inference_ltx2_3_t2v(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    return _run_t2v_distilled(job_id, prompt, dict(params or {}))


def run_inference_ltx2_3_i2v_distilled(
    job_id: str, prompt: str, params: Dict[str, Any]
) -> str:
    return _run_i2v_distilled(job_id, prompt, dict(params or {}))


def run_inference_ltx2_3_flf2v(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    return _run_flf2v_distilled(job_id, prompt, dict(params or {}))


def run_inference_ltx2_3_iv2v_distilled(
    job_id: str, prompt: str, params: Dict[str, Any]
) -> str:
    return _run_iv2v_distilled(job_id, prompt, dict(params or {}))


def _run_t2v_distilled(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    from diffusers.pipelines.ltx2.utils import (
        DISTILLED_SIGMA_VALUES,
        STAGE_2_DISTILLED_SIGMA_VALUES,
    )

    p = params
    width, height = _align_spatial(int(p.get("width", 768)), int(p.get("height", 512)))
    num_frames = _align_frames_8k_plus_1(int(p.get("num_frames", p.get("frames", 121))))
    frame_rate = float(p.get("frame_rate", p.get("fps", 24.0)))
    seed = int(p.get("seed", random.randint(0, 2**31 - 1)))
    neg = p.get("negative_prompt")
    if neg is None and _parse_bool(p.get("use_default_negative_prompt"), True):
        neg = DEFAULT_NEGATIVE

    upsample = _spatial_upsample_enabled(p, default=True)
    s1 = _stage1_steps(p, upsample=upsample)
    s2 = _stage2_steps(p)
    k1 = _distilled_stage1_kwargs()
    k2 = _distilled_stage2_kwargs()

    os.makedirs(p.get("output_dir", DEFAULT_OUTPUT_DIR), exist_ok=True)
    out_path = os.path.join(
        p.get("output_dir", DEFAULT_OUTPUT_DIR), f"t2v_{job_id}.mp4"
    )

    _cleanup_cuda()
    with _profile_phase("t2v:get_pipe_lora", p):
        pipe = _get_t2v_pipe()
        gen = torch.Generator(device=torch.device(DEVICE)).manual_seed(seed)
        lora_ad = _sync_civitai_lora_on_pipe(pipe, p)
    ls1, ls2 = _lora_stage_strengths(p)

    if not upsample:
        with _profile_phase("t2v:single_stage_denoise+encode", p):
            _apply_lora_strength(pipe, lora_ad, _lora_strength_single_pass(p))
            video, audio = pipe(
                prompt=prompt,
                negative_prompt=neg,
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=s1,
                sigmas=DISTILLED_SIGMA_VALUES,
                generator=gen,
                output_type="np",
                return_dict=False,
                **k1,
            )
            _encode_out(
                video[0],
                audio[0],
                pipe.vocoder,
                frame_rate,
                out_path,
                p,
                num_frames=num_frames,
            )
        _cleanup_cuda()
        return os.path.abspath(out_path)

    with _profile_phase("t2v:stage1_latent", p):
        _apply_lora_strength(pipe, lora_ad, ls1)
        video_latent, audio_latent = pipe(
            prompt=prompt,
            negative_prompt=neg,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=s1,
            sigmas=DISTILLED_SIGMA_VALUES,
            generator=gen,
            output_type="latent",
            return_dict=False,
            **k1,
        )

    with _profile_phase("t2v:spatial_latent_upsample", p):
        upsample_pipe = _get_spatial_upsample_pipe(pipe.vae)
        upscaled_video_latent = upsample_pipe(
            latents=video_latent,
            output_type="latent",
            return_dict=False,
        )[0]
        del video_latent
        del upsample_pipe
        _evict_cached_spatial_upsampler_if_full_gpu(pipe)

    w2, h2 = _align_spatial(width * 2, height * 2)
    _s2_cpu = _prepare_pipe_for_stage2_cpu_offload(pipe, p)
    try:
        with _profile_phase("t2v:stage2_denoise", p):
            _apply_lora_strength(pipe, lora_ad, ls2)
            video, audio = pipe(
                latents=upscaled_video_latent,
                audio_latents=audio_latent,
                prompt=prompt,
                negative_prompt=neg,
                width=w2,
                height=h2,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=s2,
                noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                generator=gen,
                output_type="np",
                return_dict=False,
                **k2,
            )
    finally:
        _restore_pipe_full_gpu_after_stage2_cpu_offload(pipe, _s2_cpu)

    with _profile_phase("t2v:encode_mp4", p):
        _encode_out(
            video[0],
            audio[0],
            pipe.vocoder,
            frame_rate,
            out_path,
            p,
            num_frames=num_frames,
        )
    _cleanup_cuda()
    return os.path.abspath(out_path)


def _run_i2v_distilled(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    from diffusers.pipelines.ltx2.utils import (
        DISTILLED_SIGMA_VALUES,
        STAGE_2_DISTILLED_SIGMA_VALUES,
    )

    p = params
    url = p.get("init_image") or p.get("image") or p.get("image_input")
    if not url:
        raise ValueError("init_image (URL or path) is required for i2v")
    image = _open_image_uri(str(url))

    upsample = _spatial_upsample_enabled(p, default=True)
    if upsample and _condition_resolution_cap_enabled(p):
        max_w, max_h = _max_condition_canvas_pixels(p)
        if p.get("width") is not None and p.get("height") is not None:
            w0, h0 = _align_spatial(int(p["width"]), int(p["height"]))
        else:
            w0, h0 = _align_spatial(image.size[0], image.size[1])
        tw, th = _scale_to_max_canvas(w0, h0, max_w, max_h)
        image = _resize_rgb_image(image, tw, th)
        width, height = tw, th
    elif upsample:
        width, height = _align_spatial(
            int(p.get("width", 768)), int(p.get("height", 512))
        )
        if (image.size[0], image.size[1]) != (width, height):
            image = _resize_rgb_image(image, width, height)
    else:
        cap = _single_stage_max_condition_long_side_pixels(p)
        width, height = _scale_to_max_canvas(
            image.size[0], image.size[1], cap, cap
        )
        if (image.size[0], image.size[1]) != (width, height):
            image = _resize_rgb_image(image, width, height)

    num_frames = _align_frames_8k_plus_1(int(p.get("num_frames", p.get("frames", 121))))
    frame_rate = float(p.get("frame_rate", p.get("fps", 24.0)))
    seed = int(p.get("seed", random.randint(0, 2**31 - 1)))
    neg = p.get("negative_prompt")
    s1 = _stage1_steps(p, upsample=upsample)
    s2 = _stage2_steps(p)
    k1 = _distilled_stage1_kwargs()
    k2 = _distilled_stage2_kwargs()

    os.makedirs(p.get("output_dir", DEFAULT_OUTPUT_DIR), exist_ok=True)
    out_path = os.path.join(
        p.get("output_dir", DEFAULT_OUTPUT_DIR), f"i2v_{job_id}.mp4"
    )

    _cleanup_cuda()
    with _profile_phase("i2v:get_pipe_lora", p):
        pipe = _get_i2v_pipe()
        gen = torch.Generator(device=torch.device(DEVICE)).manual_seed(seed)
        lora_ad = _sync_civitai_lora_on_pipe(pipe, p)
    ls1, ls2 = _lora_stage_strengths(p)

    if not upsample:
        with _profile_phase("i2v:single_stage_denoise+encode", p):
            _apply_lora_strength(pipe, lora_ad, _lora_strength_single_pass(p))
            video, audio = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=neg,
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=s1,
                sigmas=DISTILLED_SIGMA_VALUES,
                generator=gen,
                output_type="np",
                return_dict=False,
                **k1,
            )
            _encode_out(
                video[0],
                audio[0],
                pipe.vocoder,
                frame_rate,
                out_path,
                p,
                num_frames=num_frames,
            )
        _cleanup_cuda()
        return os.path.abspath(out_path)

    with _profile_phase("i2v:stage1_latent", p):
        _apply_lora_strength(pipe, lora_ad, ls1)
        video_latent, audio_latent = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=neg,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=s1,
            sigmas=DISTILLED_SIGMA_VALUES,
            generator=gen,
            output_type="latent",
            return_dict=False,
            **k1,
        )

    with _profile_phase("i2v:spatial_latent_upsample", p):
        upsample_pipe = _get_spatial_upsample_pipe(pipe.vae)
        upscaled_video_latent = upsample_pipe(
            latents=video_latent,
            output_type="latent",
            return_dict=False,
        )[0]
        del video_latent
        del upsample_pipe
        _evict_cached_spatial_upsampler_if_full_gpu(pipe)

    w2, h2 = _align_spatial(width * 2, height * 2)
    _s2_cpu = _prepare_pipe_for_stage2_cpu_offload(pipe, p)
    try:
        with _profile_phase("i2v:stage2_denoise", p):
            _apply_lora_strength(pipe, lora_ad, ls2)
            video, audio = pipe(
                image=image,
                latents=upscaled_video_latent,
                audio_latents=audio_latent,
                prompt=prompt,
                negative_prompt=neg,
                width=w2,
                height=h2,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=s2,
                noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                generator=gen,
                output_type="np",
                return_dict=False,
                **k2,
            )
    finally:
        _restore_pipe_full_gpu_after_stage2_cpu_offload(pipe, _s2_cpu)

    with _profile_phase("i2v:encode_mp4", p):
        _encode_out(
            video[0],
            audio[0],
            pipe.vocoder,
            frame_rate,
            out_path,
            p,
            num_frames=num_frames,
        )
    _cleanup_cuda()
    return os.path.abspath(out_path)


def _run_flf2v_distilled(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
    from diffusers.pipelines.ltx2.utils import (
        DISTILLED_SIGMA_VALUES,
        STAGE_2_DISTILLED_SIGMA_VALUES,
    )

    p = params
    first_u = p.get("first_frame_image") or p.get("first_image")
    last_u = p.get("last_frame_image") or p.get("last_image")
    if not first_u or not last_u:
        raise ValueError(
            "first_frame_image and last_frame_image are required for flf2v"
        )

    first = _open_image_uri(str(first_u))
    last = _open_image_uri(str(last_u))

    upsample = _spatial_upsample_enabled(p, default=True)
    if upsample and _condition_resolution_cap_enabled(p):
        max_w, max_h = _max_condition_canvas_pixels(p)
        if p.get("width") is not None and p.get("height") is not None:
            w0, h0 = _align_spatial(int(p["width"]), int(p["height"]))
        else:
            w0, h0 = _align_spatial(first.size[0], first.size[1])
        tw, th = _scale_to_max_canvas(w0, h0, max_w, max_h)
        first = _resize_rgb_image(first, tw, th)
        last = _resize_rgb_image(last, tw, th)
        width, height = tw, th
    elif upsample:
        width, height = _align_spatial(
            int(p.get("width", 768)), int(p.get("height", 512))
        )
        first = _resize_rgb_image(first, width, height)
        last = _resize_rgb_image(last, width, height)
    else:
        cap = _single_stage_max_condition_long_side_pixels(p)
        width, height = _scale_to_max_canvas(
            first.size[0], first.size[1], cap, cap
        )
        first = _resize_rgb_image(first, width, height)
        last = _resize_rgb_image(last, width, height)

    first_cond = LTX2VideoCondition(
        frames=first, index=0, strength=float(p.get("first_strength", 1.0))
    )
    last_cond = LTX2VideoCondition(
        frames=last, index=-1, strength=float(p.get("last_strength", 1.0))
    )
    conditions: List[Any] = [first_cond, last_cond]

    num_frames = _align_frames_8k_plus_1(int(p.get("num_frames", p.get("frames", 121))))
    frame_rate = float(p.get("frame_rate", p.get("fps", 24.0)))
    seed = int(p.get("seed", random.randint(0, 2**31 - 1)))
    neg = p.get("negative_prompt")
    if neg is None and _parse_bool(p.get("use_default_negative_prompt"), True):
        neg = DEFAULT_NEGATIVE

    s1 = _stage1_steps(p, upsample=upsample)
    s2 = _stage2_steps(p)
    k1 = _distilled_stage1_kwargs()
    k2 = _distilled_stage2_kwargs()

    os.makedirs(p.get("output_dir", DEFAULT_OUTPUT_DIR), exist_ok=True)
    out_path = os.path.join(
        p.get("output_dir", DEFAULT_OUTPUT_DIR), f"flf2v_{job_id}.mp4"
    )

    _cleanup_cuda()
    with _profile_phase("flf2v:get_pipe_lora", p):
        pipe = _get_condition_pipe()
        gen = torch.Generator(device=torch.device(DEVICE)).manual_seed(seed)
        lora_ad = _sync_civitai_lora_on_pipe(pipe, p)
    ls1, ls2 = _lora_stage_strengths(p)

    if not upsample:
        with _profile_phase("flf2v:single_stage_denoise+encode", p):
            _apply_lora_strength(pipe, lora_ad, _lora_strength_single_pass(p))
            video, audio = pipe(
                conditions=conditions,
                prompt=prompt,
                negative_prompt=neg,
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=s1,
                sigmas=DISTILLED_SIGMA_VALUES,
                generator=gen,
                output_type="np",
                return_dict=False,
                **k1,
            )
            _encode_out(
                video[0],
                audio[0],
                pipe.vocoder,
                frame_rate,
                out_path,
                p,
                num_frames=num_frames,
            )
        _cleanup_cuda()
        return os.path.abspath(out_path)

    with _profile_phase("flf2v:stage1_latent", p):
        _apply_lora_strength(pipe, lora_ad, ls1)
        video_latent, audio_latent = pipe(
            conditions=conditions,
            prompt=prompt,
            negative_prompt=neg,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=s1,
            sigmas=DISTILLED_SIGMA_VALUES,
            generator=gen,
            output_type="latent",
            return_dict=False,
            **k1,
        )

    with _profile_phase("flf2v:spatial_latent_upsample", p):
        upsample_pipe = _get_spatial_upsample_pipe(pipe.vae)
        upscaled_video_latent = upsample_pipe(
            latents=video_latent,
            output_type="latent",
            return_dict=False,
        )[0]
        del video_latent
        del upsample_pipe
        _evict_cached_spatial_upsampler_if_full_gpu(pipe)

    w2, h2 = _align_spatial(width * 2, height * 2)
    _s2_cpu = _prepare_pipe_for_stage2_cpu_offload(pipe, p)
    try:
        with _profile_phase("flf2v:stage2_denoise", p):
            _apply_lora_strength(pipe, lora_ad, ls2)
            video, audio = pipe(
                latents=upscaled_video_latent,
                audio_latents=audio_latent,
                prompt=prompt,
                negative_prompt=neg,
                width=w2,
                height=h2,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=s2,
                noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                generator=gen,
                output_type="np",
                return_dict=False,
                **k2,
            )
    finally:
        _restore_pipe_full_gpu_after_stage2_cpu_offload(pipe, _s2_cpu)

    with _profile_phase("flf2v:encode_mp4", p):
        _encode_out(
            video[0],
            audio[0],
            pipe.vocoder,
            frame_rate,
            out_path,
            p,
            num_frames=num_frames,
        )
    _cleanup_cuda()
    return os.path.abspath(out_path)


def _run_iv2v_distilled(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
    from diffusers.pipelines.ltx2.utils import (
        DEFAULT_NEGATIVE_PROMPT,
        DISTILLED_SIGMA_VALUES,
        STAGE_2_DISTILLED_SIGMA_VALUES,
    )

    p = params
    v_url = p.get("condition_video") or p.get("video_condition")
    i_url = p.get("condition_image") or p.get("image_condition")
    if not v_url or not i_url:
        raise ValueError(
            "condition_video and condition_image (URLs or paths) are required for iv2v"
        )

    vid_frames = _load_video_uri(str(v_url))
    cond_img = _open_image_uri(str(i_url))

    upsample = _spatial_upsample_enabled(p, default=True)
    if upsample and _condition_resolution_cap_enabled(p):
        max_w, max_h = _max_condition_canvas_pixels(p)
        if p.get("width") is not None and p.get("height") is not None:
            w0, h0 = _align_spatial(int(p["width"]), int(p["height"]))
        else:
            w0, h0 = _align_spatial(cond_img.size[0], cond_img.size[1])
        tw, th = _scale_to_max_canvas(w0, h0, max_w, max_h)
        cond_img = _resize_rgb_image(cond_img, tw, th)
        vid_frames = _resize_condition_video_frames(vid_frames, tw, th)
        width, height = tw, th
    elif upsample:
        width, height = _align_spatial(
            int(p.get("width", 768)), int(p.get("height", 512))
        )
        cond_img = _resize_rgb_image(cond_img, width, height)
        vid_frames = _resize_condition_video_frames(vid_frames, width, height)
    else:
        cap = _single_stage_max_condition_long_side_pixels(p)
        width, height = _scale_to_max_canvas(
            cond_img.size[0], cond_img.size[1], cap, cap
        )
        cond_img = _resize_rgb_image(cond_img, width, height)
        vid_frames = _resize_condition_video_frames(vid_frames, width, height)

    vi = int(p.get("condition_video_index", p.get("video_condition_index", 0)))
    ii = int(p.get("condition_image_index", p.get("image_condition_index", 8)))
    vst = float(p.get("condition_video_strength", 1.0))
    ist = float(p.get("condition_image_strength", 1.0))

    conditions = [
        LTX2VideoCondition(frames=vid_frames, index=vi, strength=vst),
        LTX2VideoCondition(frames=cond_img, index=ii, strength=ist),
    ]

    num_frames = _align_frames_8k_plus_1(int(p.get("num_frames", p.get("frames", 121))))
    frame_rate = float(p.get("frame_rate", p.get("fps", 24.0)))
    seed = int(p.get("seed", random.randint(0, 2**31 - 1)))
    neg = p.get("negative_prompt")
    if neg is None and _parse_bool(p.get("use_default_negative_prompt"), True):
        neg = DEFAULT_NEGATIVE_PROMPT

    s1 = _stage1_steps(p, upsample=upsample)
    s2 = _stage2_steps(p)
    k1 = _distilled_stage1_kwargs()
    k2 = _distilled_stage2_kwargs()

    os.makedirs(p.get("output_dir", DEFAULT_OUTPUT_DIR), exist_ok=True)
    out_path = os.path.join(
        p.get("output_dir", DEFAULT_OUTPUT_DIR), f"iv2v_{job_id}.mp4"
    )

    _cleanup_cuda()
    with _profile_phase("iv2v:get_pipe_lora", p):
        pipe = _get_condition_pipe()
        gen = torch.Generator(device=torch.device(DEVICE)).manual_seed(seed)
        lora_ad = _sync_civitai_lora_on_pipe(pipe, p)
    ls1, ls2 = _lora_stage_strengths(p)

    if not upsample:
        with _profile_phase("iv2v:single_stage_denoise+encode", p):
            _apply_lora_strength(pipe, lora_ad, _lora_strength_single_pass(p))
            video, audio = pipe(
                conditions=conditions,
                prompt=prompt,
                negative_prompt=neg,
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=s1,
                sigmas=DISTILLED_SIGMA_VALUES,
                generator=gen,
                output_type="np",
                return_dict=False,
                **k1,
            )
            _encode_out(
                video[0],
                audio[0],
                pipe.vocoder,
                frame_rate,
                out_path,
                p,
                num_frames=num_frames,
            )
        _cleanup_cuda()
        return os.path.abspath(out_path)

    with _profile_phase("iv2v:stage1_latent", p):
        _apply_lora_strength(pipe, lora_ad, ls1)
        video_latent, audio_latent = pipe(
            conditions=conditions,
            prompt=prompt,
            negative_prompt=neg,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=s1,
            sigmas=DISTILLED_SIGMA_VALUES,
            generator=gen,
            output_type="latent",
            return_dict=False,
            **k1,
        )

    with _profile_phase("iv2v:spatial_latent_upsample", p):
        upsample_pipe = _get_spatial_upsample_pipe(pipe.vae)
        upscaled_video_latent = upsample_pipe(
            latents=video_latent,
            output_type="latent",
            return_dict=False,
        )[0]
        del video_latent
        del upsample_pipe
        _evict_cached_spatial_upsampler_if_full_gpu(pipe)

    w2, h2 = _align_spatial(width * 2, height * 2)
    _s2_cpu = _prepare_pipe_for_stage2_cpu_offload(pipe, p)
    try:
        with _profile_phase("iv2v:stage2_denoise", p):
            _apply_lora_strength(pipe, lora_ad, ls2)
            video, audio = pipe(
                latents=upscaled_video_latent,
                audio_latents=audio_latent,
                prompt=prompt,
                negative_prompt=neg,
                width=w2,
                height=h2,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=s2,
                noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                generator=gen,
                output_type="np",
                return_dict=False,
                **k2,
            )
    finally:
        _restore_pipe_full_gpu_after_stage2_cpu_offload(pipe, _s2_cpu)

    with _profile_phase("iv2v:encode_mp4", p):
        _encode_out(
            video[0],
            audio[0],
            pipe.vocoder,
            frame_rate,
            out_path,
            p,
            num_frames=num_frames,
        )
    _cleanup_cuda()
    return os.path.abspath(out_path)


def preload_smallest() -> bool:
    try:
        _get_i2v_pipe()
        return True
    except Exception:
        return False


def print_ltx2_3_ab_offload_matrix_guidance() -> None:
    """Suggested A/B axes for ~32GB cards (RTX 5090 class): offload, stage-2 offload, resolution cap, upsampler."""
    lines = [
        "[LTX2_3 AB matrix] Run short smokes varying one axis at a time; log wall time + peak VRAM (set LTX2_3_PROFILE=1).",
        "  Row A: main offload — sequential vs model (avoid none if you OOM).",
        "  Row B: stage2_offload — sequential vs model vs none (two-stage jobs only).",
        "  Row C: use_spatial_upsampler — true vs false (single-stage uses single_stage_max_condition_long_side).",
        "  Row D: max_condition_width × max_condition_height — e.g. 768×512 vs native init (VRAM vs quality).",
        "CLI: use this module's --ab-smoke with --init-image for two timed i2v runs (sequential then model offload).",
    ]
    print("\n".join(lines))


def run_ltx2_3_ab_offload_i2v_smoke(
    init_image: str,
    prompt: str,
    output_dir: str,
    *,
    num_frames: int = 25,
    use_spatial_upsampler: bool = True,
) -> Dict[str, float]:
    """Two short i2v generations: offload sequential then offload model. Clears pipeline cache between runs."""
    global _OFFLOAD_MODE
    import uuid as _uuid

    os.makedirs(output_dir, exist_ok=True)
    nf = _align_frames_8k_plus_1(int(num_frames))
    timings: Dict[str, float] = {}
    for mode in ("sequential", "model"):
        unload_pipelines()
        _OFFLOAD_MODE = mode
        t0 = time.perf_counter()
        jid = f"ab_{mode}_{_uuid.uuid4().hex[:8]}"
        run_inference_ltx2_3_i2v_distilled(
            jid,
            prompt,
            {
                "output_dir": output_dir,
                "init_image": init_image,
                "num_frames": nf,
                "use_spatial_upsampler": use_spatial_upsampler,
                "profile": _parse_bool(os.environ.get("LTX2_3_PROFILE"), False),
            },
        )
        timings[mode] = time.perf_counter() - t0
        print(f"[ab-smoke] offload={mode}: {timings[mode]:.2f}s")
    _OFFLOAD_MODE = ""
    unload_pipelines()
    return timings


# ---------------------------------------------------------------------------
# Local smoke: python ltx2_3_inference_v3.py --mode t2v [--no-upsample]
# Wall clock from the start of ``if __name__`` is printed at the end; each mode timer includes its downloads and inference.
# With ``--offload none`` and two-stage, smoke caps ``--num-frames`` at 121 if you requested more (stage-2 VRAM).
# HTTP(S) assets for i2v / flf2v / iv2v are downloaded to ``<output-dir>/smoke_inputs/<run-id>/`` and jobs use those paths.
# For i2v-only with a **local** image, use ``--init-image /path/to.png`` (bypasses default astronaut download).
# --offload auto|none|sequential|model (CUDA only); ``auto`` = env LTX2_3_OFFLOAD_MODE or sequential-then-model.
# --stage2-offload sequential|model|none for two-stage 2× refine (optional; overrides LTX2_3_STAGE2_OFFLOAD; works with any --offload).
# ``LTX2_3_DISABLE_OFFLOAD=1`` still forces no offload.
# --width / --height apply to t2v only. i2v / flf2v / iv2v: with spatial upsample, default caps to 768×512 (aspect);
# with --no-upsample, downscale to fit longer side <= single-stage cap (default 1024px, env LTX2_3_SINGLE_STAGE_*).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import time
    import uuid
    from urllib.parse import unquote, urlparse

    _wall_t0 = time.perf_counter()

    _ap = argparse.ArgumentParser(
        description="Smoke-test LTX2.3 distilled-only (v3, spatial upsampler flags) worker."
    )
    _ap.add_argument(
        "--mode",
        choices=("t2v", "i2v", "flf2v", "iv2v", "all"),
        default="t2v",
    )
    _ap.add_argument("--output-dir", default="./ltx2_3_v3_test_outputs")
    _ap.add_argument(
        "--init-image",
        default="",
        help=(
            "i2v smoke: path to a local image file, or http(s) URL. "
            "If omitted, downloads the default HF astronaut.jpg into smoke_inputs."
        ),
    )
    _ap.add_argument("--num-frames", type=int, default=361)
    _ap.add_argument("--width", type=int, default=768)
    _ap.add_argument("--height", type=int, default=512)
    _ap.add_argument(
        "--no-upsample",
        action="store_true",
        help="Single-stage only (8 steps + DISTILLED_SIGMA_VALUES, no 2x spatial upsample refine).",
    )
    _ap.add_argument(
        "--single-stage-max-long",
        type=int,
        default=None,
        help=(
            "With --no-upsample / use_spatial_upsampler false: max longer side for conditioning (default 1024, or env "
            "LTX2_3_SINGLE_STAGE_MAX_CONDITION_LONG_SIDE). Job key: single_stage_max_condition_long_side."
        ),
    )
    _ap.add_argument(
        "--offload",
        choices=("auto", "none", "sequential", "model"),
        default="auto",
        help=(
            "CUDA memory offload: auto (sequential if available else model), none (full pipeline on GPU for stage 1), "
            "sequential, or model. With offload none, spatial upsampler cache is evicted after stage 1; "
            "for 2× refine use --stage2-offload / LTX2_3_STAGE2_OFFLOAD (applies on top of this mode; stage-1 offload "
            "is restored after stage 2). "
            "With auto, still honors LTX2_3_OFFLOAD_MODE env unless set to auto. "
            "Non-auto values override that env. CPU runs ignore offload."
        ),
    )
    _ap.add_argument(
        "--civitai-lora-url",
        default="",
        help="Optional Civitai model or download URL; merged into job params as civitai_lora_model_url.",
    )
    _ap.add_argument(
        "--civitai-version-id",
        type=int,
        default=None,
        help="Optional Civitai model version id when using a model page URL (civitai_version_id).",
    )
    _ap.add_argument(
        "--lora-strength",
        type=float,
        default=None,
        help="LoRA strength for both stages unless overridden (lora_strength). Default job value: 1.0.",
    )
    _ap.add_argument(
        "--lora-apply",
        choices=("both", "stage1", "stage2"),
        default=None,
        help="Which denoising stage(s) use the LoRA (lora_apply). Default: both.",
    )
    _ap.add_argument(
        "--stage2-offload",
        choices=("sequential", "model", "none"),
        default=None,
        help=(
            "When spatial upsample is on: stage-2-only CPU offload for the 2× refine (sequential / model / none), "
            "after temporarily removing stage-1 offload hooks. Works with any --offload (model, sequential, none, auto). "
            "Default when omitted: env LTX2_3_STAGE2_OFFLOAD or sequential."
        ),
    )
    _ap.add_argument(
        "--profile",
        action="store_true",
        help="Enable per-phase timing + VRAM logs (also set job param profile or env LTX2_3_PROFILE=1).",
    )
    _ap.add_argument(
        "--ab-matrix",
        action="store_true",
        help="Print suggested A/B axes for 32GB tuning (offload / stage2 / caps / upsampler) and exit.",
    )
    _ap.add_argument(
        "--ab-smoke",
        action="store_true",
        help=(
            "Run two short i2v jobs (sequential then model offload) for timing comparison; "
            "requires CUDA and --init-image local path. Uses --num-frames (capped workflow)."
        ),
    )
    _ap.add_argument(
        "--lora-fusion-cache",
        action="store_true",
        help="Enable LoRA tensor disk cache (lora_fusion_cache / LTX2_3_LORA_FUSION_CACHE).",
    )
    _args = _ap.parse_args()

    if _args.ab_matrix:
        print_ltx2_3_ab_offload_matrix_guidance()
        raise SystemExit(0)

    if _args.ab_smoke:
        if _args.profile:
            os.environ["LTX2_3_PROFILE"] = "1"
        if not torch.cuda.is_available():
            raise SystemExit("[ab-smoke] CUDA is required")
        pimg = (_args.init_image or "").strip()
        if not pimg or not os.path.isfile(pimg):
            raise SystemExit("[ab-smoke] pass --init-image /path/to.png (local file)")
        _wall_ab = time.perf_counter()
        run_ltx2_3_ab_offload_i2v_smoke(
            pimg,
            "A short cinematic clip for VRAM A/B smoke test.",
            _args.output_dir,
            num_frames=min(25, int(_args.num_frames)),
            use_spatial_upsampler=not _args.no_upsample,
        )
        print(
            f"[ab-smoke] total wall {time.perf_counter() - _wall_ab:.2f}s (see per-mode lines above)"
        )
        raise SystemExit(0)

    _OFFLOAD_MODE = "" if _args.offload == "auto" else _args.offload
    if _args.stage2_offload is not None:
        _STAGE2_OFFLOAD_CLI = _args.stage2_offload

    _smoke_num_frames = _args.num_frames
    if _args.offload == "none" and not _args.no_upsample and _smoke_num_frames > 121:
        print(
            f"[smoke] --offload none + 2-stage: capping --num-frames {_smoke_num_frames} -> 121 for stage-2 VRAM; "
            "set a lower value explicitly or use --no-upsample / offload auto."
        )
        _smoke_num_frames = 121

    os.makedirs(_args.output_dir, exist_ok=True)
    _jid = str(uuid.uuid4())[:8]
    _smoke_inputs_dir = os.path.join(_args.output_dir, "smoke_inputs", _jid)
    os.makedirs(_smoke_inputs_dir, exist_ok=True)

    prompt_t2v = "A cinematic drone shot of coastal cliffs at golden hour, gentle waves, slow camera push-in."

    # prompt_i2v = (
    #     "An astronaut hatches from a fragile egg on the surface of the Moon, the shell cracking and peeling apart in "
    #     "gentle low-gravity motion. Fine lunar dust lifts and drifts outward with each movement, floating in slow arcs "
    #     "before settling back onto the ground. The astronaut pushes free in a deliberate, weightless motion, small "
    #     "fragments of the egg tumbling and spinning through the air. In the background, the deep darkness of space subtly "
    #     "shifts as stars glide with the camera's movement, emphasizing vast depth and scale. The camera performs a "
    #     "smooth, cinematic slow push-in, with natural parallax between the foreground dust, the astronaut, and the "
    #     "distant starfield. Ultra-realistic detail, physically accurate low-gravity motion, cinematic lighting, and a "
    #     "breath-taking, movie-like shot."
    # )

    prompt_i2v = """
The fully nude woman, already completely naked in the exact starting pose from the input image where her hands are near her thighs, immediately raises both arms and firmly grabs her large bare breasts. She presses her breasts together hard with both hands, squeezes them intensely, lifts them upward, and kneads the soft flesh with strong rhythmic motions. She repeatedly squeezes, pushes her breasts together to create deep cleavage, bounces them, and plays with them energetically — fingers digging into the breast tissue, thumbs rubbing and pinching her erect nipples, making the breasts deform realistically under the pressure.

CRITICAL: She must actively PRESS and SQUEEZE her breasts firmly, not just touch or move hands around them. Strong squeezing action, visible breast flesh compression and jiggle with every press, deep cleavage formed repeatedly, energetic fondling and playing with her own boobs for the entire clip. Perfect detailed hands gripping and squeezing breast flesh.

Her face remains 100% identical with absolutely no change whatsoever throughout the entire clip — exact same facial identity, exact same facial features, eyes, expression, head angle, and proportions locked perfectly to the input image with zero alterations or drift. The camera stays completely static with no zoom in, no push-in, no dolly, no movement at all — exact same framing, exact same distance, exact same perspective and composition locked throughout the whole video. No slow motion, energetic fast-paced action only.

Rapid strong breast squeezing and pressing, vigorous kneading and bouncing, realistic soft flesh deformation and heavy jiggle physics, subtle aroused body sway, cinematic yet voyeuristic lighting highlighting natural skin tones and breast details. Photorealistic details, high consistency, fast motion, no camera movement, perfect anatomically correct hands gripping breasts firmly.
"""


#     prompt_i2v = """
# bl0wj0b, The fully nude woman, already completely naked in the exact starting pose and framing from the input image, leans forward and performs a fast sensual blowjob on a man's erect penis. She grasps the base firmly with her manicured nails, guides the shaft deep into her mouth, wraps her lips tightly around it, and sucks with quick rhythmic bobbing head movements, tongue rapidly working the underside, visible saliva sheen and dripping on the penis and lips. Her large bare breasts sway and jiggle energetically with the fast motion, detailed vagina and natural pubic hair fully visible and unchanged in position.

# Her face remains 100% identical with absolutely no change whatsoever throughout the entire clip — exact same facial identity, exact same facial features, eyes, expression, head angle, and proportions locked perfectly to the input image with zero alterations or drift. The camera stays completely static with no zoom in, no push-in, no dolly, no movement toward the face or body at all — exact same framing, exact same distance, exact same perspective and composition locked throughout the whole video. No slow motion, no lingering movements, fast paced action only.

# Quick energetic oral motion, rapid head bobbing and sucking, natural breast physics with energetic jiggle, realistic skin texture and moisture, cinematic yet voyeuristic lighting highlighting natural skin tones. Photorealistic details, high consistency, fast motion, no camera movement.
# """
#     prompt_i2v = """
# m15510n4ry, The fully nude woman, already completely naked in the exact starting pose and framing from the input image, lies on her back with legs spread as a man positions between her thighs and performs fast missionary sex. His erect penis thrusts quickly and deeply in and out of her detailed vagina with natural pubic hair, visible moisture sheen on both the penis and surrounding skin, rhythmic energetic pumping motion. Her large bare breasts bounce and jiggle vigorously with every thrust, body rocking naturally.

# Her face remains 100% identical with absolutely no change whatsoever throughout the entire clip — exact same facial identity, exact same facial features, eyes, expression, head angle, and proportions locked perfectly to the input image with zero alterations or drift. The camera stays completely static with no zoom in, no push-in, no dolly, no movement at all — exact same framing, exact same distance, exact same perspective and composition locked throughout the whole video. No slow motion, fast paced action only.

# Quick rhythmic thrusting, energetic breast physics and body movement, realistic skin texture and wetness, cinematic yet voyeuristic lighting highlighting natural skin tones. Photorealistic details, high consistency, fast motion, no camera movement.
# """

#     prompt_i2v = """
# c0wg1rl, The fully nude woman, already completely naked in the exact starting pose and framing from the input image, is in reverse cowgirl position with arms behind her back and legs spread, riding a man's erect penis. She bounces up and down quickly on the shaft, her detailed vagina taking it deep with fast rhythmic motions, visible moisture and skin stretching. Her large bare breasts bounce energetically with every rapid movement, ass jiggling naturally.

# Her face remains 100% identical with absolutely no change whatsoever throughout the entire clip — exact same facial identity, exact same facial features, eyes, expression, head angle, and proportions locked perfectly to the input image with zero alterations or drift. The camera stays completely static with no zoom in, no push-in, no dolly, no movement at all — exact same framing, exact same distance, exact same perspective and composition locked throughout the whole video. No slow motion, fast paced action only.

# Quick energetic riding and bouncing, vigorous breast and ass physics, realistic skin texture and wetness, cinematic voyeuristic lighting highlighting natural skin tones. Photorealistic details, high consistency, fast motion, no camera movement.
# """

#     prompt_i2v = """
# d0gg1e, The fully nude woman, already completely naked in the exact starting pose and framing from the input image, is on her hands and knees or face down ass up as a man fucks her from behind in fast doggy style. His erect penis thrusts rapidly and deeply in and out of her vagina and anus area with strong rhythmic pounding, detailed puffy vagina with natural pubic hair, visible moisture sheen and skin slapping. Her large bare breasts swing and jiggle vigorously with every hard thrust, body rocking forward.

# Her face remains 100% identical with absolutely no change whatsoever throughout the entire clip — exact same facial identity, exact same facial features, eyes, expression, head angle, and proportions locked perfectly to the input image with zero alterations or drift (she can look back toward the camera). The camera stays completely static with no zoom in, no push-in, no dolly, no movement at all — exact same framing, exact same distance, exact same perspective and composition locked throughout the whole video. No slow motion, fast paced action only.

# Rapid powerful thrusting, energetic breast swing and ass jiggle physics, realistic skin texture and wetness, cinematic voyeuristic lighting highlighting natural skin tones. Photorealistic details, high consistency, fast motion, no camera movement.
# """

    prompt_flf2v = (
        "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are "
        "delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright "
        "sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, "
        "low-angle perspective."
    )

    prompt_iv2v = """
      m15510n4ry style, from the starting clothed pose in the input image the woman seductively undresses piece by piece, sliding off all clothing to become fully nude with exposed breasts and detailed vagina with natural pubic hair. She then moves into missionary position lying on her back at the edge of the bed with legs spread as a man positions between them. Her face remains exactly identical — no change to facial features, identity or expression from the input frame. Smooth realistic undressing and position transition motion, natural skin tones with slight moisture sheen on bodies, even lighting highlighting textures. Exact same camera angle and framing locked, only subtle gentle push-in, photorealistic details and fluid animation.
      """

    _cosmos_vid = (
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/"
        "cosmos/cosmos-video2world-input-vid.mp4"
    )
    _cosmos_img = (
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/"
        "cosmos/cosmos-video2world-input.jpg"
    )

    _default_i2v_image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
    _flf_first = (
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/"
        "flf2v_input_first_frame.png"
    )
    _flf_last = (
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/"
        "flf2v_input_last_frame.png"
    )

    def _smoke_open_rgb(uri: str) -> Image.Image:
        u = uri.strip()
        if u.startswith("http://") or u.startswith("https://"):
            r = requests.get(u, timeout=180)
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGB")
        return Image.open(u).convert("RGB")

    def _smoke_filename_from_url(url: str) -> str:
        path = unquote(urlparse(url.strip()).path)
        base = os.path.basename(path)
        return base if base else "download.bin"

    def _smoke_download_if_url(uri: str, filename: Optional[str] = None) -> str:
        """If ``uri`` is http(s), download bytes to ``_smoke_inputs_dir`` and return absolute path; else return ``uri``."""
        u = uri.strip()
        if not (u.startswith("http://") or u.startswith("https://")):
            return os.path.abspath(u) if os.path.isfile(u) else u
        name = filename or _smoke_filename_from_url(u)
        out_abs = os.path.join(_smoke_inputs_dir, name)
        vid = name.lower().endswith((".mp4", ".webm", ".mov", ".mkv"))
        r = requests.get(u, timeout=300 if vid else 180)
        r.raise_for_status()
        with open(out_abs, "wb") as fo:
            fo.write(r.content)
        return os.path.abspath(out_abs)

    def _smoke_dims_from_image(uri: str) -> Tuple[int, int]:
        im = _smoke_open_rgb(uri)
        try:
            return _align_spatial(im.size[0], im.size[1])
        finally:
            im.close()

    _base_params: Dict[str, Any] = {
        "output_dir": _args.output_dir,
        "width": _args.width,
        "height": _args.height,
        "num_frames": _smoke_num_frames,
        "frame_rate": 24.0,
        "seed": random.randint(0, 1000),
        "use_spatial_upsampler": not _args.no_upsample,
    }
    if _args.single_stage_max_long is not None:
        _base_params["single_stage_max_condition_long_side"] = int(
            _args.single_stage_max_long
        )
    if _args.profile:
        _base_params["profile"] = True
    if _args.lora_fusion_cache:
        _base_params["lora_fusion_cache"] = True
    if (_args.civitai_lora_url or "").strip():
        _base_params["civitai_lora_model_url"] = _args.civitai_lora_url.strip()
        if _args.civitai_version_id is not None:
            _base_params["civitai_version_id"] = _args.civitai_version_id
        if _args.lora_strength is not None:
            _base_params["lora_strength"] = _args.lora_strength
        if _args.lora_apply is not None:
            _base_params["lora_apply"] = _args.lora_apply
        print(
            f"[smoke] Civitai LoRA URL set; cache dir: "
            f"{_lora_cache_dir_from_params(_base_params)}"
        )

    def _run_one(name: str) -> float:
        if name == "t2v":
            t0 = time.perf_counter()
            path = run_inference_ltx2_3_t2v(_jid, prompt_t2v, dict(_base_params))
            elapsed = time.perf_counter() - t0
            print(f"[t2v] ok (this mode) in {elapsed:.2f}s -> {path}")
            return elapsed
        if name == "i2v":
            t0 = time.perf_counter()
            p = dict(_base_params)
            _src = (_args.init_image or "").strip()
            if _src:
                if _src.startswith("http://") or _src.startswith("https://"):
                    _local_i2v = _smoke_download_if_url(
                        _src, _smoke_filename_from_url(_src)
                    )
                    _msg_extra = f"saved {_local_i2v}"
                else:
                    _local_i2v = os.path.abspath(_src)
                    if not os.path.isfile(_local_i2v):
                        raise SystemExit(
                            f"[i2v] --init-image is not a file: {_local_i2v!r}"
                        )
                    _msg_extra = f"local file {_local_i2v}"
            else:
                _local_i2v = _smoke_download_if_url(_default_i2v_image, "astronaut.jpg")
                _msg_extra = f"saved {_local_i2v}"
            im_sz = _smoke_open_rgb(_local_i2v)
            try:
                raw_w, raw_h = im_sz.size
            finally:
                im_sz.close()
            iw, ih = _align_spatial(raw_w, raw_h)
            p["width"], p["height"] = iw, ih
            p["init_image"] = _local_i2v
            if _args.no_upsample:
                cap = _single_stage_max_condition_long_side_pixels(p)
                cw, ch = _scale_to_max_canvas(raw_w, raw_h, cap, cap)
                print(
                    f"[i2v] init_image {raw_w}x{raw_h} -> conditioning {cw}x{ch} "
                    f"(single-stage long-side cap {cap}px; {_msg_extra})"
                )
            else:
                print(f"[i2v] using {iw}x{ih} from init_image ({_msg_extra})")
            path = run_inference_ltx2_3_i2v_distilled(_jid + "_i2v", prompt_i2v, p)
            elapsed = time.perf_counter() - t0
            print(f"[i2v] ok (this mode) in {elapsed:.2f}s -> {path}")
            return elapsed
        if name == "flf2v":
            t0 = time.perf_counter()
            _local_ff = _smoke_download_if_url(
                _flf_first, "flf2v_input_first_frame.png"
            )
            _local_lf = _smoke_download_if_url(_flf_last, "flf2v_input_last_frame.png")
            im_a = _smoke_open_rgb(_local_ff)
            im_b = _smoke_open_rgb(_local_lf)
            try:
                if im_a.size != im_b.size:
                    raise SystemExit(
                        f"flf2v smoke test: first/last frame size mismatch {im_a.size} vs {im_b.size}"
                    )
                fw, fh = _align_spatial(im_a.size[0], im_a.size[1])
            finally:
                im_a.close()
                im_b.close()
            p = dict(_base_params)
            p["width"], p["height"] = fw, fh
            p["first_frame_image"] = _local_ff
            p["last_frame_image"] = _local_lf
            print(
                f"[flf2v] using {fw}x{fh} from first_frame_image (saved {_smoke_inputs_dir})"
            )
            path = run_inference_ltx2_3_flf2v(_jid + "_flf", prompt_flf2v, p)
            elapsed = time.perf_counter() - t0
            print(f"[flf2v] ok (this mode) in {elapsed:.2f}s -> {path}")
            return elapsed
        if name == "iv2v":
            t0 = time.perf_counter()
            p = dict(_base_params)
            _local_cimg = _smoke_download_if_url(
                _cosmos_img, "cosmos_video2world-input.jpg"
            )
            _local_cvid = _smoke_download_if_url(
                _cosmos_vid, "cosmos-video2world-input-vid.mp4"
            )
            iw, ih = _smoke_dims_from_image(_local_cimg)
            p["width"], p["height"] = iw, ih
            p["condition_video"] = _local_cvid
            p["condition_image"] = _local_cimg
            p["condition_video_index"] = 0
            p["condition_image_index"] = 8
            print(
                f"[iv2v] using {iw}x{ih} from condition_image (saved {_smoke_inputs_dir})"
            )
            path = run_inference_ltx2_3_iv2v_distilled(_jid + "_iv2v", prompt_iv2v, p)
            elapsed = time.perf_counter() - t0
            print(f"[iv2v] ok (this mode) in {elapsed:.2f}s -> {path}")
            return elapsed
        return 0.0

    _modes = ("t2v", "i2v", "flf2v", "iv2v") if _args.mode == "all" else (_args.mode,)
    if any(x in _modes for x in ("i2v", "flf2v", "iv2v")):
        print(f"[smoke] URL inputs will be saved under: {_smoke_inputs_dir}")
    _gen_seconds: Dict[str, float] = {}
    for _m in _modes:
        print(f"--- running {_m} ---")
        _gen_seconds[_m] = _run_one(_m)
    print("--- smoke test generation time ---")
    for _m in _modes:
        print(
            f"  {_m}: {_gen_seconds[_m]:.2f}s (per mode: downloads + lazy pipeline load + inference)"
        )
    if len(_modes) > 1:
        print(f"  sum of per-mode: {sum(_gen_seconds[m] for m in _modes):.2f}s")
    print(f"  wall clock (python start -> now): {time.perf_counter() - _wall_t0:.2f}s")
    print("Done.")
