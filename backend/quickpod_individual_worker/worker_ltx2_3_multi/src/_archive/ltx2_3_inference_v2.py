"""
LTX-2.3 **distilled-checkpoint-only** Diffusers worker (v2).

All modes load weights from the **distilled** Diffusers repo (default: rockapaper SDNQ snapshot).
Inference follows the Diffusers "distilled checkpoint generation" recipe: stage 1 uses
``DISTILLED_SIGMA_VALUES`` with ``num_inference_steps=8`` and ``guidance_scale=1.0``; optional
stage 2 latent upsample then short refine with ``STAGE_2_DISTILLED_SIGMA_VALUES`` (default 3 steps).

Spatial / latent upsampler weights come from ``LTX2_3_SPATIAL_UPSAMPLER_REPO`` (not bundled in SDNQ
distilled repos). SDNQ quant checkpoints require the ``sdnq`` package.

**Checkpoint layout:** this file expects the distilled snapshot to expose ``LTX2Pipeline``, ``LTX2ImageToVideoPipeline``,
and ``LTX2ConditionPipeline`` (same class layout as Diffusers' distilled examples). If your distilled repo is
I2V-only, T2V / FLF2V / iv2v loads will fail — use ``ltx2_3_inference.py`` (full + LoRA two-stage) instead.

Modes
-----
- **t2v**: ``LTX2Pipeline``
- **i2v**: ``LTX2ImageToVideoPipeline`` (``init_image`` URL)
- **flf2v**: ``LTX2ConditionPipeline`` + first/last frame URLs
- **iv2v** (image + video conditions): ``LTX2ConditionPipeline`` + ``condition_video`` + ``condition_image`` URLs

Job params
----------
- **i2v / flf2v / iv2v** conditioning images (and iv2v ``condition_video`` frames) are scaled down uniformly if
  needed so spatial tokens stay in a safe range (see ``LTX2_3_MAX_CONDITION_*`` env vars). Aspect ratio is preserved;
  no letterboxing. Optional ``width`` / ``height`` act as an extra upper bound when both are set.
- ``use_latent_upsampler`` (aliases: ``use_latent_upsample``, ``enable_latent_upsampler``, ``latent_upsample``):
  two-stage (8 + upsample + 3). Default **true**.
- ``stage1_num_inference_steps`` / ``stage2_num_inference_steps``: default **8** / **3** (match distilled sigma lists).
- ``num_inference_steps``: when upsampler off, single-stage step count (default 8); when on, alias for stage1 if
  ``stage1_num_inference_steps`` unset.
- **iv2v**: ``condition_video``, ``condition_image`` (URLs); ``condition_video_index`` (default 0),
  ``condition_image_index`` (default 8); ``condition_video_strength`` / ``condition_image_strength`` (default 1.0).

Entrypoints: ``run_inference_ltx2_3_t2v``, ``run_inference_ltx2_3_i2v_distilled``, ``run_inference_ltx2_3_flf2v``,
``run_inference_ltx2_3_iv2v_distilled``.
"""

from __future__ import annotations

import gc
import os
import random
import tempfile
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from PIL import Image

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_OUTPUT_DIR = os.environ.get("WORKSPACE_DIR", "/workspace")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

DISTILLED_REPO = os.environ.get(
    "LTX2_3_DISTILLED_REPO", "rockapaper/LTX-2.3-Distilled-Diffusers_tenc_uint4_sdnq_r64_s16"
).strip()
SPATIAL_UPSAMPLER_REPO = os.environ.get(
    "LTX2_3_SPATIAL_UPSAMPLER_REPO", "dg845/LTX-2.3-Spatial-Upsampler-Diffusers"
).strip()

ENV_DISTILLED_LOCAL = "LTX2_3_DISTILLED_LOCAL"
ENV_SPATIAL_LOCAL = "LTX2_3_SPATIAL_UPSAMPLER_LOCAL"

DEFAULT_NEGATIVE = (
    "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, "
    "fused fingers, bad anatomy, weird hand, ugly, transition, static."
)

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
    return str(os.environ.get("LTX2_3_ASSUME_SDNQ_WEIGHTS", "")).strip().lower() in {"1", "true", "yes", "on"}


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
            "LTX2.3 v2 is loading SDNQ weights but `sdnq` is not importable."
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


def _condition_budget_limits() -> Tuple[int, int]:
    """Return (max_longer_side, max_pixels) for conditioning (i2v / flf2v / iv2v).

    LTX-2 memory scales with ~(H/32)*(W/32)*frames; public docs cite ~768² and 768×448 as typical shapes and
    recommend lowering resolution when OOM (see Lightricks LTX-Video-Trainer troubleshooting / community reports).
    Defaults approximate a 768×512-equivalent area cap with longest side 768.
    """
    max_long = int(os.environ.get("LTX2_3_MAX_CONDITION_LONG_SIDE", "768"))
    default_px = 768 * 512
    max_px = int(os.environ.get("LTX2_3_MAX_CONDITION_PIXELS", str(default_px)))
    return max(32, max_long), max(32 * 32, max_px)


def _target_wh_for_conditioning(nat_w: int, nat_h: int, p: Dict[str, Any]) -> Tuple[int, int]:
    """Uniform scale to satisfy budget; dimensions are multiples of 32; aspect ratio preserved."""
    max_long, max_px = _condition_budget_limits()
    cap_w: Optional[int] = None
    cap_h: Optional[int] = None
    if p.get("width") is not None and p.get("height") is not None:
        cap_w, cap_h = _align_spatial(int(p["width"]), int(p["height"]))

    wf, hf = float(nat_w), float(nat_h)
    if wf < 1.0 or hf < 1.0:
        raise ValueError("invalid conditioning spatial dimensions")

    s = 1.0
    s = min(s, float(max_long) / max(wf, hf))
    s = min(s, (float(max_px) / (wf * hf)) ** 0.5)
    if cap_w is not None and cap_h is not None:
        s = min(s, float(cap_w) / wf, float(cap_h) / hf)

    nw = max(32, int(wf * s) // 32 * 32)
    nh = max(32, int(round(nw * hf / wf / 32)) * 32)

    def _violates() -> bool:
        if nw * nh > max_px:
            return True
        if max(nw, nh) > max_long:
            return True
        if cap_w is not None and cap_h is not None and (nw > cap_w or nh > cap_h):
            return True
        return False

    while _violates() and min(nw, nh) > 32:
        if nw >= nh:
            nw -= 32
            nh = max(32, int(round(nw * hf / wf / 32)) * 32)
        else:
            nh -= 32
            nw = max(32, int(round(nh * wf / hf / 32)) * 32)

    return nw, nh


def _resize_pil_to_exact(im: Image.Image, w: int, h: int) -> Image.Image:
    if im.size == (w, h):
        return im
    return im.resize((w, h), Image.Resampling.LANCZOS)


def _resize_pil_video_frames(frames: List[Image.Image], w: int, h: int) -> List[Image.Image]:
    return [_resize_pil_to_exact(fr, w, h) for fr in frames]


def _latent_upsample_enabled(p: Dict[str, Any], *, default: bool) -> bool:
    for key in (
        "use_latent_upsampler",
        "use_latent_upsample",
        "enable_latent_upsampler",
        "latent_upsample",
    ):
        if key in p and p[key] is not None:
            return _parse_bool(p.get(key), default)
    return default


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _apply_sequential_offload(pipe: Any) -> None:
    if DEVICE != "cuda":
        return
    if _parse_bool(os.environ.get("LTX2_3_DISABLE_OFFLOAD"), False):
        return
    if hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload()
        return
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload(device=torch.device("cuda"))


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


def _distilled_source() -> str:
    return _resolve_pretrained(DISTILLED_REPO, ENV_DISTILLED_LOCAL)


def _get_t2v_pipe() -> Any:
    key = "t2v_distilled"
    if key in _CACHE:
        return _CACHE[key]
    from diffusers import LTX2Pipeline

    src = _distilled_source()
    pipe = _from_pretrained_ltx(LTX2Pipeline, src, torch_dtype=DTYPE)
    _apply_sequential_offload(pipe)
    _vae_tiling(pipe)
    _CACHE[key] = pipe
    return pipe


def _get_condition_pipe() -> Any:
    key = "condition_distilled"
    if key in _CACHE:
        return _CACHE[key]
    from diffusers import LTX2ConditionPipeline

    src = _distilled_source()
    pipe = _from_pretrained_ltx(LTX2ConditionPipeline, src, torch_dtype=DTYPE)
    _apply_sequential_offload(pipe)
    _vae_tiling(pipe)
    _CACHE[key] = pipe
    return pipe


def _get_i2v_pipe() -> Any:
    key = "i2v_distilled"
    if key in _CACHE:
        return _CACHE[key]
    from diffusers import LTX2ImageToVideoPipeline

    src = _distilled_source()
    pipe = _from_pretrained_ltx(LTX2ImageToVideoPipeline, src, torch_dtype=DTYPE)
    _apply_sequential_offload(pipe)
    _vae_tiling(pipe)
    _CACHE[key] = pipe
    return pipe


def _get_latent_upsample_pipe(vae: Any) -> Any:
    key = ("upsample", id(vae))
    if key in _CACHE:
        return _CACHE[key]
    from diffusers import LTX2LatentUpsamplePipeline
    from diffusers.pipelines.ltx2 import LTX2LatentUpsamplerModel

    root = _resolve_pretrained(SPATIAL_UPSAMPLER_REPO, ENV_SPATIAL_LOCAL)
    _ensure_sdnq_for_pretrained_source(root)
    local_spatial = _local_dir_if_any(root)
    if local_spatial is not None:
        _ensure_sdnq_registered_for_dir(local_spatial)
    latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
        root,
        subfolder="latent_upsampler",
        torch_dtype=DTYPE,
    )
    upsample_pipe = LTX2LatentUpsamplePipeline(vae=vae, latent_upsampler=latent_upsampler)
    _apply_sequential_offload(upsample_pipe)
    _CACHE[key] = upsample_pipe
    return upsample_pipe


def _encode_out(
    video_np: Any,
    audio_tensor: Any,
    vocoder: Any,
    frame_rate: float,
    output_path: str,
) -> None:
    from diffusers.pipelines.ltx2.export_utils import encode_video

    sr = int(vocoder.config.output_sampling_rate)
    encode_video(
        video_np,
        fps=frame_rate,
        audio=audio_tensor.float().cpu(),
        audio_sample_rate=sr,
        output_path=output_path,
    )


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


def run_inference_ltx2_3_i2v_distilled(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    return _run_i2v_distilled(job_id, prompt, dict(params or {}))


def run_inference_ltx2_3_flf2v(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    return _run_flf2v_distilled(job_id, prompt, dict(params or {}))


def run_inference_ltx2_3_iv2v_distilled(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    return _run_iv2v_distilled(job_id, prompt, dict(params or {}))


def _run_t2v_distilled(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES

    p = params
    width, height = _align_spatial(int(p.get("width", 768)), int(p.get("height", 512)))
    num_frames = _align_frames_8k_plus_1(int(p.get("num_frames", p.get("frames", 241))))
    frame_rate = float(p.get("frame_rate", p.get("fps", 24.0)))
    seed = int(p.get("seed", random.randint(0, 2**31 - 1)))
    neg = p.get("negative_prompt")
    if neg is None and _parse_bool(p.get("use_default_negative_prompt"), True):
        neg = DEFAULT_NEGATIVE

    upsample = _latent_upsample_enabled(p, default=True)
    s1 = _stage1_steps(p, upsample=upsample)
    s2 = _stage2_steps(p)
    k1 = _distilled_stage1_kwargs()
    k2 = _distilled_stage2_kwargs()

    os.makedirs(p.get("output_dir", DEFAULT_OUTPUT_DIR), exist_ok=True)
    out_path = os.path.join(p.get("output_dir", DEFAULT_OUTPUT_DIR), f"t2v_{job_id}.mp4")

    _cleanup_cuda()
    pipe = _get_t2v_pipe()
    gen = torch.Generator(device=torch.device(DEVICE)).manual_seed(seed)

    if not upsample:
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
        _encode_out(video[0], audio[0], pipe.vocoder, frame_rate, out_path)
        _cleanup_cuda()
        return os.path.abspath(out_path)

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

    upsample_pipe = _get_latent_upsample_pipe(pipe.vae)
    upscaled_video_latent = upsample_pipe(
        latents=video_latent,
        output_type="latent",
        return_dict=False,
    )[0]

    w2, h2 = _align_spatial(width * 2, height * 2)
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

    _encode_out(video[0], audio[0], pipe.vocoder, frame_rate, out_path)
    _cleanup_cuda()
    return os.path.abspath(out_path)


def _run_i2v_distilled(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES

    p = params
    url = p.get("init_image") or p.get("image") or p.get("image_input")
    if not url:
        raise ValueError("init_image (URL or path) is required for i2v")
    image = _open_image_uri(str(url))
    width, height = _target_wh_for_conditioning(image.size[0], image.size[1], p)
    image = _resize_pil_to_exact(image, width, height)
    num_frames = _align_frames_8k_plus_1(int(p.get("num_frames", p.get("frames", 241))))
    frame_rate = float(p.get("frame_rate", p.get("fps", 24.0)))
    seed = int(p.get("seed", random.randint(0, 2**31 - 1)))
    neg = p.get("negative_prompt")

    upsample = _latent_upsample_enabled(p, default=True)
    s1 = _stage1_steps(p, upsample=upsample)
    s2 = _stage2_steps(p)
    k1 = _distilled_stage1_kwargs()
    k2 = _distilled_stage2_kwargs()

    os.makedirs(p.get("output_dir", DEFAULT_OUTPUT_DIR), exist_ok=True)
    out_path = os.path.join(p.get("output_dir", DEFAULT_OUTPUT_DIR), f"i2v_{job_id}.mp4")

    _cleanup_cuda()
    pipe = _get_i2v_pipe()
    gen = torch.Generator(device=torch.device(DEVICE)).manual_seed(seed)

    if not upsample:
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
        _encode_out(video[0], audio[0], pipe.vocoder, frame_rate, out_path)
        _cleanup_cuda()
        return os.path.abspath(out_path)

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

    upsample_pipe = _get_latent_upsample_pipe(pipe.vae)
    upscaled_video_latent = upsample_pipe(
        latents=video_latent,
        output_type="latent",
        return_dict=False,
    )[0]

    w2, h2 = _align_spatial(width * 2, height * 2)
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

    _encode_out(video[0], audio[0], pipe.vocoder, frame_rate, out_path)
    _cleanup_cuda()
    return os.path.abspath(out_path)


def _run_flf2v_distilled(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES

    p = params
    first_u = p.get("first_frame_image") or p.get("first_image")
    last_u = p.get("last_frame_image") or p.get("last_image")
    if not first_u or not last_u:
        raise ValueError("first_frame_image and last_frame_image are required for flf2v")

    first = _open_image_uri(str(first_u))
    last = _open_image_uri(str(last_u))
    width, height = _target_wh_for_conditioning(first.size[0], first.size[1], p)
    first = _resize_pil_to_exact(first, width, height)
    last = _resize_pil_to_exact(last, width, height)
    first_cond = LTX2VideoCondition(frames=first, index=0, strength=float(p.get("first_strength", 1.0)))
    last_cond = LTX2VideoCondition(frames=last, index=-1, strength=float(p.get("last_strength", 1.0)))
    conditions: List[Any] = [first_cond, last_cond]

    num_frames = _align_frames_8k_plus_1(int(p.get("num_frames", p.get("frames", 241))))
    frame_rate = float(p.get("frame_rate", p.get("fps", 24.0)))
    seed = int(p.get("seed", random.randint(0, 2**31 - 1)))
    neg = p.get("negative_prompt")
    if neg is None and _parse_bool(p.get("use_default_negative_prompt"), True):
        neg = DEFAULT_NEGATIVE

    upsample = _latent_upsample_enabled(p, default=True)
    s1 = _stage1_steps(p, upsample=upsample)
    s2 = _stage2_steps(p)
    k1 = _distilled_stage1_kwargs()
    k2 = _distilled_stage2_kwargs()

    os.makedirs(p.get("output_dir", DEFAULT_OUTPUT_DIR), exist_ok=True)
    out_path = os.path.join(p.get("output_dir", DEFAULT_OUTPUT_DIR), f"flf2v_{job_id}.mp4")

    _cleanup_cuda()
    pipe = _get_condition_pipe()
    gen = torch.Generator(device=torch.device(DEVICE)).manual_seed(seed)

    if not upsample:
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
        _encode_out(video[0], audio[0], pipe.vocoder, frame_rate, out_path)
        _cleanup_cuda()
        return os.path.abspath(out_path)

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

    upsample_pipe = _get_latent_upsample_pipe(pipe.vae)
    upscaled_video_latent = upsample_pipe(
        latents=video_latent,
        output_type="latent",
        return_dict=False,
    )[0]

    w2, h2 = _align_spatial(width * 2, height * 2)
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

    _encode_out(video[0], audio[0], pipe.vocoder, frame_rate, out_path)
    _cleanup_cuda()
    return os.path.abspath(out_path)


def _run_iv2v_distilled(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
    from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES

    p = params
    v_url = p.get("condition_video") or p.get("video_condition")
    i_url = p.get("condition_image") or p.get("image_condition")
    if not v_url or not i_url:
        raise ValueError("condition_video and condition_image (URLs or paths) are required for iv2v")

    vid_frames = _load_video_uri(str(v_url))
    cond_img = _open_image_uri(str(i_url))
    width, height = _target_wh_for_conditioning(cond_img.size[0], cond_img.size[1], p)
    cond_img = _resize_pil_to_exact(cond_img, width, height)
    vid_frames = _resize_pil_video_frames(list(vid_frames), width, height)

    vi = int(p.get("condition_video_index", p.get("video_condition_index", 0)))
    ii = int(p.get("condition_image_index", p.get("image_condition_index", 8)))
    vst = float(p.get("condition_video_strength", 1.0))
    ist = float(p.get("condition_image_strength", 1.0))

    conditions = [
        LTX2VideoCondition(frames=vid_frames, index=vi, strength=vst),
        LTX2VideoCondition(frames=cond_img, index=ii, strength=ist),
    ]

    num_frames = _align_frames_8k_plus_1(int(p.get("num_frames", p.get("frames", 241))))
    frame_rate = float(p.get("frame_rate", p.get("fps", 24.0)))
    seed = int(p.get("seed", random.randint(0, 2**31 - 1)))
    neg = p.get("negative_prompt")
    if neg is None and _parse_bool(p.get("use_default_negative_prompt"), True):
        neg = DEFAULT_NEGATIVE_PROMPT

    upsample = _latent_upsample_enabled(p, default=True)
    s1 = _stage1_steps(p, upsample=upsample)
    s2 = _stage2_steps(p)
    k1 = _distilled_stage1_kwargs()
    k2 = _distilled_stage2_kwargs()

    os.makedirs(p.get("output_dir", DEFAULT_OUTPUT_DIR), exist_ok=True)
    out_path = os.path.join(p.get("output_dir", DEFAULT_OUTPUT_DIR), f"iv2v_{job_id}.mp4")

    _cleanup_cuda()
    pipe = _get_condition_pipe()
    gen = torch.Generator(device=torch.device(DEVICE)).manual_seed(seed)

    if not upsample:
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
        _encode_out(video[0], audio[0], pipe.vocoder, frame_rate, out_path)
        _cleanup_cuda()
        return os.path.abspath(out_path)

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

    upsample_pipe = _get_latent_upsample_pipe(pipe.vae)
    upscaled_video_latent = upsample_pipe(
        latents=video_latent,
        output_type="latent",
        return_dict=False,
    )[0]

    w2, h2 = _align_spatial(width * 2, height * 2)
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

    _encode_out(video[0], audio[0], pipe.vocoder, frame_rate, out_path)
    _cleanup_cuda()
    return os.path.abspath(out_path)


def preload_smallest() -> bool:
    try:
        _get_i2v_pipe()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Local smoke: python ltx2_3_inference_v2.py --mode t2v [--no-upsample]
# --width / --height apply to t2v; i2v / flf2v / iv2v derive size from assets with LTX2_3_MAX_CONDITION_* caps.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import time
    import uuid

    _ap = argparse.ArgumentParser(description="Smoke-test LTX2.3 distilled-only (v2) worker.")
    _ap.add_argument(
        "--mode",
        choices=("t2v", "i2v", "flf2v", "iv2v", "all"),
        default="t2v",
    )
    _ap.add_argument("--output-dir", default="./ltx2_3_v2_test_outputs")
    _ap.add_argument("--num-frames", type=int, default=240)
    _ap.add_argument("--width", type=int, default=768)
    _ap.add_argument("--height", type=int, default=512)
    _ap.add_argument(
        "--no-upsample",
        action="store_true",
        help="Single-stage only (8 steps + DISTILLED_SIGMA_VALUES, no 2x refine).",
    )
    _args = _ap.parse_args()

    os.makedirs(_args.output_dir, exist_ok=True)
    _jid = str(uuid.uuid4())[:8]

    prompt_t2v = (
        "A cinematic drone shot of coastal cliffs at golden hour, gentle waves, slow camera push-in."
    )

    prompt_i2v = (
        "An astronaut hatches from a fragile egg on the surface of the Moon, the shell cracking and peeling apart in "
        "gentle low-gravity motion. Fine lunar dust lifts and drifts outward with each movement, floating in slow arcs "
        "before settling back onto the ground. The astronaut pushes free in a deliberate, weightless motion, small "
        "fragments of the egg tumbling and spinning through the air. In the background, the deep darkness of space subtly "
        "shifts as stars glide with the camera's movement, emphasizing vast depth and scale. The camera performs a "
        "smooth, cinematic slow push-in, with natural parallax between the foreground dust, the astronaut, and the "
        "distant starfield. Ultra-realistic detail, physically accurate low-gravity motion, cinematic lighting, and a "
        "breath-taking, movie-like shot."
    )

    prompt_flf2v = (
        "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are "
        "delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright "
        "sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, "
        "low-angle perspective."
    )

    prompt_iv2v = (
        "The video depicts a long, straight highway stretching into the distance, flanked by metal guardrails. The road is "
        "divided into multiple lanes, with a few vehicles visible in the far distance. The surrounding landscape features "
        "dry, grassy fields on one side and rolling hills on the other. The sky is mostly clear with a few scattered "
        "clouds, suggesting a bright, sunny day. And then the camera switch to a winding mountain road covered in snow, "
        "with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The "
        "landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the "
        "solitude and beauty of a winter drive through a mountainous region."
    )

    _cosmos_vid = (
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/"
        "cosmos/cosmos-video2world-input-vid.mp4"
    )
    _cosmos_img = (
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/"
        "cosmos/cosmos-video2world-input.jpg"
    )

    _default_i2v_image = (
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
    )
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

    def _smoke_dims_from_image(uri: str) -> Tuple[int, int]:
        im = _smoke_open_rgb(uri)
        try:
            return _target_wh_for_conditioning(im.size[0], im.size[1], {})
        finally:
            im.close()

    _base_params: Dict[str, Any] = {
        "output_dir": _args.output_dir,
        "width": _args.width,
        "height": _args.height,
        "num_frames": _args.num_frames,
        "frame_rate": 24.0,
        "seed": 42,
        "use_latent_upsampler": not _args.no_upsample,
    }

    def _run_one(name: str) -> float:
        if name == "t2v":
            t0 = time.perf_counter()
            path = run_inference_ltx2_3_t2v(_jid, prompt_t2v, dict(_base_params))
            elapsed = time.perf_counter() - t0
            print(f"[t2v] ok in {elapsed:.2f}s -> {path}")
            return elapsed
        if name == "i2v":
            p = dict(_base_params)
            iw, ih = _smoke_dims_from_image(_default_i2v_image)
            p["width"], p["height"] = iw, ih
            p["init_image"] = _default_i2v_image
            print(f"[i2v] using {iw}x{ih} from init_image")
            t0 = time.perf_counter()
            path = run_inference_ltx2_3_i2v_distilled(_jid + "_i2v", prompt_i2v, p)
            elapsed = time.perf_counter() - t0
            print(f"[i2v] ok in {elapsed:.2f}s -> {path}")
            return elapsed
        if name == "flf2v":
            im_a = _smoke_open_rgb(_flf_first)
            im_b = _smoke_open_rgb(_flf_last)
            try:
                if im_a.size != im_b.size:
                    raise SystemExit(
                        f"flf2v smoke test: first/last frame size mismatch {im_a.size} vs {im_b.size}"
                    )
                fw, fh = _target_wh_for_conditioning(im_a.size[0], im_a.size[1], {})
            finally:
                im_a.close()
                im_b.close()
            p = dict(_base_params)
            p["width"], p["height"] = fw, fh
            p["first_frame_image"] = _flf_first
            p["last_frame_image"] = _flf_last
            print(f"[flf2v] using {fw}x{fh} from first_frame_image")
            t0 = time.perf_counter()
            path = run_inference_ltx2_3_flf2v(_jid + "_flf", prompt_flf2v, p)
            elapsed = time.perf_counter() - t0
            print(f"[flf2v] ok in {elapsed:.2f}s -> {path}")
            return elapsed
        if name == "iv2v":
            p = dict(_base_params)
            iw, ih = _smoke_dims_from_image(_cosmos_img)
            p["width"], p["height"] = iw, ih
            p["condition_video"] = _cosmos_vid
            p["condition_image"] = _cosmos_img
            p["condition_video_index"] = 0
            p["condition_image_index"] = 8
            print(f"[iv2v] using {iw}x{ih} from condition_image")
            t0 = time.perf_counter()
            path = run_inference_ltx2_3_iv2v_distilled(_jid + "_iv2v", prompt_iv2v, p)
            elapsed = time.perf_counter() - t0
            print(f"[iv2v] ok in {elapsed:.2f}s -> {path}")
            return elapsed
        return 0.0

    _modes = ("t2v", "i2v", "flf2v", "iv2v") if _args.mode == "all" else (_args.mode,)
    _gen_seconds: Dict[str, float] = {}
    for _m in _modes:
        print(f"--- running {_m} ---")
        _gen_seconds[_m] = _run_one(_m)
    print("--- smoke test generation time ---")
    for _m in _modes:
        print(f"  {_m}: {_gen_seconds[_m]:.2f}s")
    if len(_modes) > 1:
        print(f"  total: {sum(_gen_seconds[m] for m in _modes):.2f}s")
    print("Done.")
