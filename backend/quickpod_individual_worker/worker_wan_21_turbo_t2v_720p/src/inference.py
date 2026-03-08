#!/usr/bin/env python3
"""
Inference for WAN 2.1 TurboDiffusion T2V 720p worker.
Mirrors turbodiffusion/inference/wan2.1_t2v_infer.py exactly (lazy load, no CLI).
"""
from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch
from einops import rearrange, repeat
from tqdm import tqdm

try:
    import turbodiffusion as _td
    _td_root = os.path.dirname(os.path.abspath(_td.__file__))
    if _td_root not in sys.path:
        sys.path.insert(0, _td_root)
except Exception:
    pass

from imaginaire.utils.io import save_image_or_video
from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from turbodiffusion.inference.modify_model import create_model, tensor_kwargs

try:
    from video_postprocessing import (
        check_rife_code_available,
        load_rife_model,
        load_sd_upscale,
        apply_rife_interpolation,
        apply_realesrgan_upscaling,
    )
    _POSTPROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Video post-processing not available: {e}")
    _POSTPROCESSING_AVAILABLE = False

torch._dynamo.config.suppress_errors = True

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
VAE_PATH = os.path.join(CHECKPOINTS_DIR, "Wan2.1_VAE.pth")
TEXT_ENCODER_PATH = os.path.join(CHECKPOINTS_DIR, "models_t5_umt5-xxl-enc-bf16.pth")
DIT_PATH = os.path.join(CHECKPOINTS_DIR, "TurboWan2.1-T2V-14B-720P-quant.pth")

_net = None
_tokenizer = None


def _make_args(
    num_samples: int = 1,
    num_steps: int = 4,
    sigma_max: float = 80.0,
    resolution: str = "720p",
    aspect_ratio: str = "16:9",
    attention_type: str = "sagesla",
    sla_topk: float = 0.1,
    quant_linear: bool = True,
    default_norm: bool = False,
    model: str = "Wan2.1-14B",
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        num_samples=num_samples,
        num_steps=num_steps,
        sigma_max=sigma_max,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        attention_type=attention_type,
        sla_topk=sla_topk,
        quant_linear=quant_linear,
        default_norm=default_norm,
    )


def _load_models():
    global _net, _tokenizer
    if _tokenizer is not None:
        return
    args = _make_args()
    print("Loading DiT model...")
    _net = create_model(dit_path=DIT_PATH, args=args).cpu()
    torch.cuda.empty_cache()
    print("Loading VAE tokenizer...")
    _tokenizer = Wan2pt1VAEInterface(vae_pth=VAE_PATH)
    print("Models loaded.")


def run_inference_wan_turbo_t2v(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """Run TurboDiffusion WAN 2.1 T2V 720p. Flow mirrors wan2.1_t2v_infer.py."""
    num_frames = int(params.get("num_frames", 81))
    num_steps = int(params.get("num_steps", 4))
    sigma_max = float(params.get("sigma_max", 80.0))
    resolution = str(params.get("resolution", "720p"))
    aspect_ratio = str(params.get("aspect_ratio", "16:9"))
    seed = int(params.get("seed", 0))
    num_samples = 1
    output_dir = params.get("output_dir", "/workspace")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"wan_turbo_t2v_{job_id}.mp4")
    fps = int(params.get("fps", 16))
    enable_rife = bool(params.get("enable_rife", False))
    enable_upscaling = bool(params.get("enable_upscaling", False))
    rife_target_fps = int(params.get("rife_target_fps", 30))
    upscale_factor = int(params.get("upscale_factor", 2))
    postprocess_interpolate_first = bool(params.get("postprocess_interpolate_first", False))

    with torch.no_grad():
        text_emb = get_umt5_embedding(checkpoint_path=TEXT_ENCODER_PATH, prompts=prompt).to(**tensor_kwargs)
    clear_umt5_memory()

    _load_models()
    net = _net
    tokenizer = _tokenizer

    w, h = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
    condition = {"crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=num_samples)}

    state_shape = [
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(num_frames),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]
    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(seed)
    init_noise = torch.randn(
        num_samples,
        *state_shape,
        dtype=torch.float32,
        device=tensor_kwargs["device"],
        generator=generator,
    )

    mid_t = [1.5, 1.4, 1.0][: num_steps - 1]
    t_steps = torch.tensor(
        [math.atan(sigma_max), *mid_t, 0],
        dtype=torch.float64,
        device=init_noise.device,
    )
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    total_steps = t_steps.shape[0] - 1
    net.cuda()
    for _i, (t_cur, t_next) in enumerate(tqdm(list(zip(t_steps[:-1], t_steps[1:])), desc="Sampling", total=total_steps)):
        with torch.no_grad():
            v_pred = net(
                x_B_C_T_H_W=x.to(**tensor_kwargs),
                timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                **condition,
            ).to(torch.float64)
            x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                *x.shape,
                dtype=torch.float32,
                device=tensor_kwargs["device"],
                generator=generator,
            )
    samples = x.float()
    net.cpu()
    torch.cuda.empty_cache()

    with torch.no_grad():
        video = tokenizer.decode(samples)
    to_show = (1.0 + video.float().cpu().unsqueeze(0).clamp(-1, 1)) / 2.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    final_fps = fps

    if _POSTPROCESSING_AVAILABLE and (enable_rife or enable_upscaling):
        frames_np = (video[0].permute(1, 0, 2, 3).cpu().numpy() + 1.0) / 2.0
        frames_np = np.clip(frames_np, 0.0, 1.0).astype(np.float32)
        frames_np = np.transpose(frames_np, (0, 2, 3, 1))
        rife_model = None
        upscale_model = None
        if enable_rife and check_rife_code_available():
            rife_path = os.path.join(MODELS_DIR, "rife")
            if os.path.exists(rife_path) and os.listdir(rife_path):
                rife_model = load_rife_model(rife_path)
            else:
                print("⚠️  RIFE model not found, interpolation disabled")
        if enable_upscaling:
            realesrgan_path = os.path.join(MODELS_DIR, "realesrgan", f"RealESRGAN_x{upscale_factor}.pth")
            if os.path.exists(realesrgan_path):
                try:
                    upscale_model = load_sd_upscale(realesrgan_path, device)
                except Exception as e:
                    print(f"⚠️  Real-ESRGAN load failed: {str(e)[:100]}")
                    upscale_model = None
            else:
                print(f"⚠️  Real-ESRGAN {upscale_factor}x model not found")
        if postprocess_interpolate_first:
            if enable_rife and rife_model:
                try:
                    frames_np, final_fps = apply_rife_interpolation(
                        rife_model, frames_np, device, original_fps=fps, target_fps=rife_target_fps
                    )
                except Exception as e:
                    print(f"⚠️  RIFE failed: {e}")
            if enable_upscaling and upscale_model:
                try:
                    frames_np = apply_realesrgan_upscaling(upscale_model, frames_np, device)
                except Exception as e:
                    print(f"⚠️  Upscaling failed: {e}")
        else:
            if enable_upscaling and upscale_model:
                try:
                    frames_np = apply_realesrgan_upscaling(upscale_model, frames_np, device)
                except Exception as e:
                    print(f"⚠️  Upscaling failed: {e}")
            if enable_rife and rife_model:
                try:
                    frames_np, final_fps = apply_rife_interpolation(
                        rife_model, frames_np, device, original_fps=fps, target_fps=rife_target_fps
                    )
                except Exception as e:
                    print(f"⚠️  RIFE failed: {e}")
        out_tensor = torch.from_numpy(np.transpose(frames_np, (3, 0, 1, 2))).float()
        save_image_or_video(out_tensor, save_path, fps=final_fps)
    else:
        save_image_or_video(rearrange(to_show, "n b c t h w -> c t (n h) (b w)"), save_path, fps=fps)
    return os.path.abspath(save_path)
