"""
LTX-2.3 Diffusers worker: text-to-video, distilled image-to-video, first-last-frame (FLF2V), optional spatial latent
upsample + stage-2 distilled pass (see job params below).

Weight layout (HF download or local env):
- Full repo: LTX2Pipeline / LTX2ConditionPipeline (T2V + FLF2V). The public distilled Diffusers repo
  is published for LTX2ImageToVideoPipeline only; T2V and FLF2V still require the full checkpoint.
- Distilled repo: same I2V weights are always loaded with ``LTX2ImageToVideoPipeline`` (the HF snapshot may
  still say ``_class_name``: ``LTX2Pipeline`` in ``model_index.json``). Two-stage distilled sigmas + spatial
  latent upsampler apply at inference time.

Pre-quantized SDNQ weights (HF defaults above or local trees from ``work_space/quantize_sdnq_ltx2_3.py``) need
``quantization_config.json`` in the snapshot and the ``sdnq`` package importable (Docker/prep install from
https://github.com/Disty0/sdnq).

All pipelines use sequential CPU offload and VAE tiling unless disabled via env.

Job params (all modes):
- ``use_latent_upsampler`` (aliases: ``use_latent_upsample``, ``enable_latent_upsampler``, ``latent_upsample``):
  if true, run latent spatial upsampler then a short stage-2 denoise at 2x resolution. Defaults: **false** for
  ``ltx2_3_t2v`` / ``ltx2_3_flf2v``, **true** for ``ltx2_3_i2v_distilled`` (previous behavior).
- ``stage1_num_inference_steps`` / ``stage2_num_inference_steps``: two-stage schedules when upsampler is on;
  for single-stage I2V with upsampler off, ``num_inference_steps`` applies.
- T2V / FLF2V upsampler path (full checkpoint): loads stage-2 distilled LoRA from baked dir
  ``LTX2_3_STAGE2_LORA_LOCAL`` (or job ``stage2_lora_local``), else Hub
  ``stage2_lora_repo`` / ``stage2_lora_weight`` / ``stage2_lora_adapter_name`` / env
  ``LTX2_3_STAGE2_LORA_*``. **Not used** for ``ltx2_3_i2v_distilled`` (distilled I2V uses the
  distilled checkpoint only). File: ``ltx-2.3-22b-distilled-lora-384.safetensors`` from Lightricks/LTX-2.3.

Omni / R2 plugins: the runtime registry uses **three** model ids (``ltx2_3_t2v``, ``ltx2_3_i2v_distilled``,
``ltx2_3_flf2v``) with **three** different entrypoint functions in this file, but a **single** shared
inference zip (``ltx2_3_multi.zip``) — see ``worker_multi_common/build_inference_zips.py`` and
``omni_manifest.json`` ``inference_plugins``.
"""

from __future__ import annotations

import gc
import os
import random
import re
from contextlib import contextmanager
from io import BytesIO
from typing import Any, Dict, Iterator, Optional, Tuple

import requests
import torch
from PIL import Image

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_OUTPUT_DIR = os.environ.get("WORKSPACE_DIR", "/workspace")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Repos (override with local dirs for offline). Defaults: SDNQ uint4 (tenc) checkpoints on HF.
FULL_REPO = os.environ.get(
    "LTX2_3_FULL_REPO", "rockapaper/LTX-2.3-Diffusers_tenc_uint4_sdnq_r64_s16"
).strip()
DISTILLED_REPO = os.environ.get(
    "LTX2_3_DISTILLED_REPO", "rockapaper/LTX-2.3-Distilled-Diffusers_tenc_uint4_sdnq_r64_s16"
).strip()
# Spatial upsampler is still the public fp Diffusers release (not bundled in the SDNQ full/distilled repos).
SPATIAL_UPSAMPLER_REPO = os.environ.get(
    "LTX2_3_SPATIAL_UPSAMPLER_REPO", "dg845/LTX-2.3-Spatial-Upsampler-Diffusers"
).strip()

ENV_FULL_LOCAL = "LTX2_3_FULL_LOCAL"
ENV_DISTILLED_LOCAL = "LTX2_3_DISTILLED_LOCAL"
ENV_SPATIAL_LOCAL = "LTX2_3_SPATIAL_UPSAMPLER_LOCAL"
# Stage-2 distilled LoRA for full LTX2Pipeline / LTX2ConditionPipeline after spatial latent upsample
# (see https://huggingface.co/docs/diffusers/en/api/pipelines/ltx2 — two-stages generation).
ENV_STAGE2_LORA_REPO = "LTX2_3_STAGE2_LORA_REPO"
ENV_STAGE2_LORA_WEIGHT = "LTX2_3_STAGE2_LORA_WEIGHT"
ENV_STAGE2_LORA_LOCAL = "LTX2_3_STAGE2_LORA_LOCAL"
DEFAULT_STAGE2_LORA_REPO = "Lightricks/LTX-2.3"
DEFAULT_STAGE2_LORA_WEIGHT = "ltx-2.3-22b-distilled-lora-384.safetensors"
DEFAULT_STAGE2_LORA_ADAPTER = "stage_2_distilled"

DEFAULT_NEGATIVE_T2V = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)

_CACHE: Dict[str, Any] = {}


def _local_dir_if_any(path_or_id: str) -> Optional[str]:
    """HF repo ids are not directories; baked paths are."""
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
    """Import sdnq so Diffusers/Transformers quant hooks register (same pattern as WAN SDNQ scripts)."""
    from sdnq import SDNQConfig  # noqa: F401


def _source_needs_sdnq(source: str) -> bool:
    """True if we should register SDNQ before loading this checkpoint (local tree or hub id)."""
    if not source or not isinstance(source, str):
        return False
    local = _local_dir_if_any(source)
    if local is not None:
        return _path_looks_sdnq_quantized(local)
    s = source.strip().lower()
    if "sdnq" in s:
        return True
    return _parse_bool(os.environ.get("LTX2_3_ASSUME_SDNQ_WEIGHTS"), False)


def _ensure_sdnq_registered_for_dir(model_dir: str) -> None:
    """Diffusers needs SDNQ quantizer registered before loading pre-quantized weights."""
    if not _path_looks_sdnq_quantized(model_dir):
        return
    try:
        _import_sdnq_register()
    except ImportError as e:
        raise ImportError(
            "This LTX checkpoint looks SDNQ-quantized (found quantization_config.json under the model "
            "directory) but the `sdnq` package is not importable. Install sdnq (or add it to "
            "PYTHONPATH), or set LTX2_3_FULL_LOCAL / LTX2_3_DISTILLED_LOCAL to full-precision "
            "Diffusers weights."
        ) from e


def _ensure_sdnq_for_pretrained_source(source: str) -> None:
    if not _source_needs_sdnq(source):
        return
    try:
        _import_sdnq_register()
    except ImportError as e:
        raise ImportError(
            "LTX2.3 is configured to load SDNQ-quantized weights but `sdnq` is not importable. "
            "Install from https://github.com/Disty0/sdnq (see worker Dockerfile / prep_model.py)."
        ) from e


def _retie_sdnq_t5_text_encoders(pipe: Any) -> None:
    """SDNQ T5 loads can leave shared vs embed_tokens untied; match WAN SDNQ inference fix."""
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
    """from_pretrained with SDNQ registration for hub ids or quantized local trees."""
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


def _parse_stg_blocks(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, str):
        xs = [int(x) for x in re.findall(r"\d+", raw)]
        return xs if xs else None
    return raw


# Stage-2 refine on full checkpoint (T2V/FLF2V + latent upsampler + distilled LoRA): disable CFG/STG;
# matches Diffusers two-stage examples and standalone worker reference implementations.
_LTX2_FULL_CHECKPOINT_STAGE2_KWARGS: Dict[str, Any] = {
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


def _latent_upsample_enabled(p: Dict[str, Any], *, default: bool) -> bool:
    """Job param toggles spatial latent upsampler (two-stage) for t2v / flf2v / i2v."""
    for key in (
        "use_latent_upsampler",
        "use_latent_upsample",
        "enable_latent_upsampler",
        "latent_upsample",
    ):
        if key in p and p[key] is not None:
            return _parse_bool(p.get(key), default)
    return default


def _stage2_lora_adapter_name(p: Dict[str, Any]) -> str:
    v = p.get("stage2_lora_adapter_name")
    if v is not None and str(v).strip():
        return str(v).strip()
    return DEFAULT_STAGE2_LORA_ADAPTER


def _resolve_stage2_lora_dir_and_weight(p: Dict[str, Any]) -> Tuple[str, str]:
    """Prefer baked/local directory (contains ``.safetensors``) over Hub repo id for ``load_lora_weights``."""
    weight = (
        p.get("stage2_lora_weight")
        or os.environ.get(ENV_STAGE2_LORA_WEIGHT)
        or DEFAULT_STAGE2_LORA_WEIGHT
    ).strip()
    explicit = (p.get("stage2_lora_local") or os.environ.get(ENV_STAGE2_LORA_LOCAL, "")).strip()
    if explicit and os.path.isdir(explicit):
        candidate = os.path.join(explicit, weight)
        if os.path.isfile(candidate):
            return explicit, weight
    models_root = (os.environ.get("LTX2_3_MODELS_ROOT", "") or "").strip()
    if models_root:
        fallback_dir = os.path.join(models_root, "stage2_lora")
        if os.path.isdir(fallback_dir) and os.path.isfile(os.path.join(fallback_dir, weight)):
            return fallback_dir, weight
    repo = (
        p.get("stage2_lora_repo")
        or os.environ.get(ENV_STAGE2_LORA_REPO)
        or DEFAULT_STAGE2_LORA_REPO
    ).strip()
    return repo, weight


def _ensure_stage2_distilled_lora(pipe: Any, p: Dict[str, Any]) -> str:
    """Load Lightricks-style stage-2 distilled LoRA once per pipe instance (full T2V / FLF2V upsampler path)."""
    adapter = _stage2_lora_adapter_name(p)
    if getattr(pipe, "_ltx23_stage2_lora_loaded", False):
        return adapter
    source, weight = _resolve_stage2_lora_dir_and_weight(p)
    pipe.load_lora_weights(source, adapter_name=adapter, weight_name=weight)
    pipe._ltx23_stage2_lora_loaded = True
    return adapter


def _set_lora_adapter_scale(pipe: Any, adapter: str, scale: float) -> None:
    if not adapter or not hasattr(pipe, "set_adapters"):
        return
    w = float(scale)
    try:
        pipe.set_adapters(adapter, w)
    except TypeError:
        pipe.set_adapters([adapter], [w])


@contextmanager
def _full_checkpoint_stage2_scheduler_swap(pipe: Any) -> Iterator[Any]:
    """Swap to FlowMatch scheduler for stage-2 distilled sigmas; restore after (Diffusers LTX2 two-stage doc)."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

    saved_sched = pipe.scheduler
    new_sched = FlowMatchEulerDiscreteScheduler.from_config(
        saved_sched.config,
        use_dynamic_shifting=False,
        shift_terminal=None,
    )
    pipe.scheduler = new_sched
    try:
        yield STAGE_2_DISTILLED_SIGMA_VALUES
    finally:
        pipe.scheduler = saved_sched


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _apply_sequential_offload(pipe: Any) -> None:
    """Prefer sequential offload for lower peak VRAM; fall back to model CPU offload."""
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


def _load_image_from_url(url: str) -> Image.Image:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("image URL is required")
    u = url.strip()
    if u.startswith("http://") or u.startswith("https://"):
        r = requests.get(u, timeout=180)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    raise ValueError("expected http(s) URL for image input")


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


def _get_t2v_pipe():
    key = "t2v"
    if key in _CACHE:
        return _CACHE[key]
    from diffusers import LTX2Pipeline

    src = _resolve_pretrained(FULL_REPO, ENV_FULL_LOCAL)
    pipe = _from_pretrained_ltx(LTX2Pipeline, src, torch_dtype=DTYPE)
    _apply_sequential_offload(pipe)
    _vae_tiling(pipe)
    _CACHE[key] = pipe
    return pipe


def _get_condition_pipe():
    key = "condition"
    if key in _CACHE:
        return _CACHE[key]
    from diffusers import LTX2ConditionPipeline

    src = _resolve_pretrained(FULL_REPO, ENV_FULL_LOCAL)
    pipe = _from_pretrained_ltx(LTX2ConditionPipeline, src, torch_dtype=DTYPE)
    _apply_sequential_offload(pipe)
    _vae_tiling(pipe)
    _CACHE[key] = pipe
    return pipe


def _get_i2v_pipe():
    key = "i2v"
    if key in _CACHE:
        return _CACHE[key]
    from diffusers import LTX2ImageToVideoPipeline

    src = _resolve_pretrained(DISTILLED_REPO, ENV_DISTILLED_LOCAL)
    pipe = _from_pretrained_ltx(LTX2ImageToVideoPipeline, src, torch_dtype=DTYPE)
    _apply_sequential_offload(pipe)
    _vae_tiling(pipe)
    _CACHE[key] = pipe
    return pipe


def _get_latent_upsample_pipe(vae: Any):
    # Latent upsampler is paired with this pipeline's VAE; do not reuse across different checkpoints.
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


def run_inference_ltx2_3_t2v(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    return _run_t2v(job_id, prompt, dict(params or {}))


def run_inference_ltx2_3_i2v_distilled(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    return _run_i2v_distilled(job_id, prompt, dict(params or {}))


def run_inference_ltx2_3_flf2v(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    return _run_flf2v(job_id, prompt, dict(params or {}))


def _run_t2v(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    from diffusers.pipelines.ltx2.export_utils import encode_video

    p = params
    width, height = _align_spatial(int(p.get("width", 768)), int(p.get("height", 512)))
    num_frames = _align_frames_8k_plus_1(int(p.get("num_frames", p.get("frames", 121))))
    frame_rate = float(p.get("frame_rate", p.get("fps", 24.0)))
    seed = int(p.get("seed", random.randint(0, 2**31 - 1)))
    steps = int(p.get("num_inference_steps", 30))
    neg = p.get("negative_prompt")
    if neg is None:
        neg = DEFAULT_NEGATIVE_T2V if _parse_bool(p.get("use_default_negative_prompt"), True) else None

    guidance_scale = float(p.get("guidance_scale", 3.0))
    stg_scale = float(p.get("stg_scale", 1.0))
    modality_scale = float(p.get("modality_scale", 3.0))
    guidance_rescale = float(p.get("guidance_rescale", 0.7))
    audio_guidance_scale = float(p.get("audio_guidance_scale", 7.0))
    audio_stg_scale = float(p.get("audio_stg_scale", 1.0))
    audio_modality_scale = float(p.get("audio_modality_scale", 3.0))
    audio_guidance_rescale = float(p.get("audio_guidance_rescale", 0.7))
    stg_blocks = _parse_stg_blocks(p.get("spatio_temporal_guidance_blocks", [28]))

    os.makedirs(p.get("output_dir", DEFAULT_OUTPUT_DIR), exist_ok=True)
    out_path = os.path.join(p.get("output_dir", DEFAULT_OUTPUT_DIR), f"t2v_{job_id}.mp4")

    upsample = _latent_upsample_enabled(p, default=False)
    stage1_steps = int(p.get("stage1_num_inference_steps", p.get("num_inference_steps", steps)))
    stage2_steps = int(p.get("stage2_num_inference_steps", 3))

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
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            stg_scale=stg_scale,
            modality_scale=modality_scale,
            guidance_rescale=guidance_rescale,
            audio_guidance_scale=audio_guidance_scale,
            audio_stg_scale=audio_stg_scale,
            audio_modality_scale=audio_modality_scale,
            audio_guidance_rescale=audio_guidance_rescale,
            spatio_temporal_guidance_blocks=stg_blocks,
            generator=gen,
            output_type="np",
            return_dict=False,
        )
        encode_video(
            video[0],
            fps=frame_rate,
            audio=audio[0].float().cpu(),
            audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
            output_path=out_path,
        )
        _cleanup_cuda()
        return os.path.abspath(out_path)

    adapter = _ensure_stage2_distilled_lora(pipe, p)
    try:
        _set_lora_adapter_scale(pipe, adapter, 0.0)
        video_latent, audio_latent = pipe(
            prompt=prompt,
            negative_prompt=neg,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=stage1_steps,
            sigmas=None,
            guidance_scale=guidance_scale,
            stg_scale=stg_scale,
            modality_scale=modality_scale,
            guidance_rescale=guidance_rescale,
            audio_guidance_scale=audio_guidance_scale,
            audio_stg_scale=audio_stg_scale,
            audio_modality_scale=audio_modality_scale,
            audio_guidance_rescale=audio_guidance_rescale,
            spatio_temporal_guidance_blocks=stg_blocks,
            use_cross_timestep=True,
            generator=gen,
            output_type="latent",
            return_dict=False,
        )

        upsample_pipe = _get_latent_upsample_pipe(pipe.vae)
        upscaled_video_latent = upsample_pipe(
            latents=video_latent,
            output_type="latent",
            return_dict=False,
        )[0]

        w2, h2 = _align_spatial(width * 2, height * 2)
        with _full_checkpoint_stage2_scheduler_swap(pipe) as stage2_sigmas:
            _set_lora_adapter_scale(pipe, adapter, 1.0)
            video, audio = pipe(
                latents=upscaled_video_latent,
                audio_latents=audio_latent,
                prompt=prompt,
                negative_prompt=neg,
                width=w2,
                height=h2,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=stage2_steps,
                noise_scale=stage2_sigmas[0],
                sigmas=stage2_sigmas,
                generator=gen,
                output_type="np",
                return_dict=False,
                **_LTX2_FULL_CHECKPOINT_STAGE2_KWARGS,
            )
    finally:
        _set_lora_adapter_scale(pipe, adapter, 0.0)

    encode_video(
        video[0],
        fps=frame_rate,
        audio=audio[0].float().cpu(),
        audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
        output_path=out_path,
    )
    _cleanup_cuda()
    return os.path.abspath(out_path)


def _run_i2v_distilled(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES

    p = params
    url = p.get("init_image") or p.get("image") or p.get("image_input")
    if not url:
        raise ValueError("init_image (presigned URL) is required for ltx2_3_i2v_distilled")
    image = _load_image_from_url(str(url))

    width, height = _align_spatial(int(p.get("width", 768)), int(p.get("height", 512)))
    num_frames = _align_frames_8k_plus_1(int(p.get("num_frames", p.get("frames", 121))))
    frame_rate = float(p.get("frame_rate", p.get("fps", 24.0)))
    seed = int(p.get("seed", random.randint(0, 2**31 - 1)))

    stage1_steps = int(p.get("stage1_num_inference_steps", 8))
    stage2_steps = int(p.get("stage2_num_inference_steps", 3))
    one_shot_steps = int(p.get("num_inference_steps", stage1_steps))

    os.makedirs(p.get("output_dir", DEFAULT_OUTPUT_DIR), exist_ok=True)
    out_path = os.path.join(p.get("output_dir", DEFAULT_OUTPUT_DIR), f"i2v_{job_id}.mp4")

    upsample = _latent_upsample_enabled(p, default=True)

    _cleanup_cuda()
    pipe = _get_i2v_pipe()
    gen = torch.Generator(device=torch.device(DEVICE)).manual_seed(seed)
    neg = p.get("negative_prompt", None)

    if not upsample:
        video, audio = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=neg,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=one_shot_steps,
            sigmas=DISTILLED_SIGMA_VALUES,
            guidance_scale=1.0,
            stg_scale=0.0,
            modality_scale=1.0,
            guidance_rescale=0.0,
            audio_guidance_scale=1.0,
            audio_stg_scale=0.0,
            audio_modality_scale=1.0,
            audio_guidance_rescale=0.0,
            spatio_temporal_guidance_blocks=None,
            use_cross_timestep=True,
            generator=gen,
            output_type="np",
            return_dict=False,
        )
        _encode_out(
            video[0],
            audio[0],
            pipe.vocoder,
            frame_rate,
            out_path,
        )
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
        num_inference_steps=stage1_steps,
        sigmas=DISTILLED_SIGMA_VALUES,
        guidance_scale=1.0,
        stg_scale=0.0,
        modality_scale=1.0,
        guidance_rescale=0.0,
        audio_guidance_scale=1.0,
        audio_stg_scale=0.0,
        audio_modality_scale=1.0,
        audio_guidance_rescale=0.0,
        spatio_temporal_guidance_blocks=None,
        use_cross_timestep=True,
        generator=gen,
        output_type="latent",
        return_dict=False,
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
        num_inference_steps=stage2_steps,
        noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
        sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
        guidance_scale=1.0,
        stg_scale=0.0,
        modality_scale=1.0,
        guidance_rescale=0.0,
        audio_guidance_scale=1.0,
        audio_stg_scale=0.0,
        audio_modality_scale=1.0,
        audio_guidance_rescale=0.0,
        spatio_temporal_guidance_blocks=None,
        use_cross_timestep=True,
        generator=gen,
        output_type="np",
        return_dict=False,
    )

    _encode_out(
        video[0],
        audio[0],
        pipe.vocoder,
        frame_rate,
        out_path,
    )
    _cleanup_cuda()
    return os.path.abspath(out_path)


def _run_flf2v(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    from diffusers.pipelines.ltx2.export_utils import encode_video
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

    p = params
    first_u = p.get("first_frame_image") or p.get("first_image")
    last_u = p.get("last_frame_image") or p.get("last_image")
    if not first_u or not last_u:
        raise ValueError("first_frame_image and last_frame_image (presigned URLs) are required for flf2v")

    first = _load_image_from_url(str(first_u))
    last = _load_image_from_url(str(last_u))
    first_cond = LTX2VideoCondition(frames=first, index=0, strength=float(p.get("first_strength", 1.0)))
    last_cond = LTX2VideoCondition(frames=last, index=-1, strength=float(p.get("last_strength", 1.0)))
    conditions = [first_cond, last_cond]

    width, height = _align_spatial(int(p.get("width", 768)), int(p.get("height", 512)))
    num_frames = _align_frames_8k_plus_1(int(p.get("num_frames", p.get("frames", 121))))
    frame_rate = float(p.get("frame_rate", p.get("fps", 24.0)))
    seed = int(p.get("seed", random.randint(0, 2**31 - 1)))
    steps = int(p.get("num_inference_steps", 30))
    neg = p.get("negative_prompt")
    if neg is None:
        neg = DEFAULT_NEGATIVE_T2V if _parse_bool(p.get("use_default_negative_prompt"), True) else None

    guidance_scale = float(p.get("guidance_scale", 3.0))
    stg_scale = float(p.get("stg_scale", 1.0))
    modality_scale = float(p.get("modality_scale", 3.0))
    guidance_rescale = float(p.get("guidance_rescale", 0.7))
    audio_guidance_scale = float(p.get("audio_guidance_scale", 7.0))
    audio_stg_scale = float(p.get("audio_stg_scale", 1.0))
    audio_modality_scale = float(p.get("audio_modality_scale", 3.0))
    audio_guidance_rescale = float(p.get("audio_guidance_rescale", 0.7))
    stg_blocks = _parse_stg_blocks(p.get("spatio_temporal_guidance_blocks", [28]))

    upsample = _latent_upsample_enabled(p, default=False)
    stage1_steps = int(p.get("stage1_num_inference_steps", p.get("num_inference_steps", steps)))
    stage2_steps = int(p.get("stage2_num_inference_steps", 3))

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
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            stg_scale=stg_scale,
            modality_scale=modality_scale,
            guidance_rescale=guidance_rescale,
            audio_guidance_scale=audio_guidance_scale,
            audio_stg_scale=audio_stg_scale,
            audio_modality_scale=audio_modality_scale,
            audio_guidance_rescale=audio_guidance_rescale,
            spatio_temporal_guidance_blocks=stg_blocks,
            generator=gen,
            output_type="np",
            return_dict=False,
        )
        encode_video(
            video[0],
            fps=frame_rate,
            audio=audio[0].float().cpu(),
            audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
            output_path=out_path,
        )
        _cleanup_cuda()
        return os.path.abspath(out_path)

    adapter = _ensure_stage2_distilled_lora(pipe, p)
    try:
        _set_lora_adapter_scale(pipe, adapter, 0.0)
        video_latent, audio_latent = pipe(
            conditions=conditions,
            prompt=prompt,
            negative_prompt=neg,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=stage1_steps,
            sigmas=None,
            guidance_scale=guidance_scale,
            stg_scale=stg_scale,
            modality_scale=modality_scale,
            guidance_rescale=guidance_rescale,
            audio_guidance_scale=audio_guidance_scale,
            audio_stg_scale=audio_stg_scale,
            audio_modality_scale=audio_modality_scale,
            audio_guidance_rescale=audio_guidance_rescale,
            spatio_temporal_guidance_blocks=stg_blocks,
            use_cross_timestep=True,
            generator=gen,
            output_type="latent",
            return_dict=False,
        )

        upsample_pipe = _get_latent_upsample_pipe(pipe.vae)
        upscaled_video_latent = upsample_pipe(
            latents=video_latent,
            output_type="latent",
            return_dict=False,
        )[0]

        w2, h2 = _align_spatial(width * 2, height * 2)
        with _full_checkpoint_stage2_scheduler_swap(pipe) as stage2_sigmas:
            _set_lora_adapter_scale(pipe, adapter, 1.0)
            video, audio = pipe(
                latents=upscaled_video_latent,
                audio_latents=audio_latent,
                prompt=prompt,
                negative_prompt=neg,
                width=w2,
                height=h2,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=stage2_steps,
                noise_scale=stage2_sigmas[0],
                sigmas=stage2_sigmas,
                generator=gen,
                output_type="np",
                return_dict=False,
                **_LTX2_FULL_CHECKPOINT_STAGE2_KWARGS,
            )
    finally:
        _set_lora_adapter_scale(pipe, adapter, 0.0)

    encode_video(
        video[0],
        fps=frame_rate,
        audio=audio[0].float().cpu(),
        audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
        output_path=out_path,
    )
    _cleanup_cuda()
    return os.path.abspath(out_path)


def preload_smallest() -> bool:
    """Optional prewarm: load I2V pipe only (smaller than holding full + condition together)."""
    try:
        _get_i2v_pipe()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Local smoke test: run from this directory (or set PYTHONPATH to parent).
#   python ltx2_3_inference.py --mode t2v
#   python ltx2_3_inference.py --mode i2v
#   python ltx2_3_inference.py --mode flf2v
#   python ltx2_3_inference.py --mode all
# Requires: CUDA, diffusers with LTX2, HF weights (local env dirs or hub).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import time
    import uuid

    _p = argparse.ArgumentParser(description="Smoke-test LTX2.3 worker modes locally.")
    _p.add_argument(
        "--mode",
        choices=("t2v", "i2v", "flf2v", "all"),
        default="t2v",
        help="Which pipeline to run (all = t2v then i2v then flf2v, heavy).",
    )
    _p.add_argument("--output-dir", default="./ltx2_3_test_outputs", help="Output directory for mp4 files.")
    _p.add_argument("--num-frames", type=int, default=241, help="8k+1 aligned automatically if needed.")
    _p.add_argument(
        "--width",
        type=int,
        default=768,
        help="Output width for t2v only; i2v/flf2v use input image size (32px-aligned).",
    )
    _p.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output height for t2v only; i2v/flf2v use input image size (32px-aligned).",
    )
    _p.add_argument(
        "--latent-upsample",
        action="store_true",
        help="Set use_latent_upsampler for t2v/flf2v (downloads stage-2 LoRA + spatial upsampler).",
    )
    _p.add_argument(
        "--i2v-no-upsample",
        action="store_true",
        help="For i2v mode: single-stage at base resolution (use_latent_upsampler=false).",
    )
    _args = _p.parse_args()

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
    "breath-taking, movie-like shot.")

    prompt_flf2v = (
    "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are "
    "delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright "
    "sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, "
    "low-angle perspective."
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
            return _align_spatial(im.size[0], im.size[1])
        finally:
            im.close()

    _base_params: Dict[str, Any] = {
        "output_dir": _args.output_dir,
        "width": _args.width,
        "height": _args.height,
        "num_frames": _args.num_frames,
        "frame_rate": 24.0,
        "seed": 42,
    }
    if _args.latent_upsample:
        _base_params["use_latent_upsampler"] = True

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
            if _args.i2v_no_upsample:
                p["use_latent_upsampler"] = False
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
                fw, fh = _align_spatial(im_a.size[0], im_a.size[1])
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
        return 0.0

    _modes = ("t2v", "i2v", "flf2v") if _args.mode == "all" else (_args.mode,)
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
