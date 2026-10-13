"""
LongCat-Video — core inference only (no Gradio / spaces).

- If LONGCAT_REPO_ROOT (default ./LongCat-Video) is missing, the repo is cloned
  automatically from GitHub (same as the HF Space), unless LONGCAT_AUTO_CLONE=0.
- Model weights are fetched via huggingface_hub snapshot_download on first load.

Install Python deps from upstream README after the repo is present.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Callable, Literal, Optional, Union
from accelerate import cpu_offload_with_hook

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------------------------------------------------------
# Paths (must be set before ensure_repo / imports)
# ---------------------------------------------------------------------------
REPO_ROOT = os.environ.get("LONGCAT_REPO_ROOT", "LongCat-Video")
CHECKPOINT_DIR = os.environ.get(
    "LONGCAT_CHECKPOINT_DIR",
    os.path.join(REPO_ROOT, "weights", "LongCat-Video"),
)
HF_MODEL_ID = "meituan-longcat/LongCat-Video"
LONGCAT_GIT_URL = "https://github.com/meituan-longcat/LongCat-Video.git"
LONGCAT_AUTO_CLONE = os.environ.get("LONGCAT_AUTO_CLONE", "true").lower() in (
    "1",
    "true",
    "yes",
)


def ensure_repo(repo_path: str = REPO_ROOT) -> None:
    """Clone LongCat-Video if the directory does not exist (optional; HF app behavior)."""
    abs_repo = os.path.abspath(repo_path)
    if os.path.isdir(abs_repo):
        return
    parent = os.path.dirname(abs_repo)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    if not LONGCAT_AUTO_CLONE:
        raise RuntimeError(
            f"LongCat-Video repo not found at {abs_repo}. Clone manually:\n"
            f"  git clone {LONGCAT_GIT_URL} {abs_repo}\n"
            "Or set LONGCAT_AUTO_CLONE=1 (default)."
        )
    print(f"[LongCat] Cloning {LONGCAT_GIT_URL} -> {abs_repo} ...")
    try:
        subprocess.run(
            ["git", "clone", LONGCAT_GIT_URL, abs_repo],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "git not found on PATH; install Git or clone the repo manually."
        ) from e
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or "").strip()
        raise RuntimeError(f"git clone failed: {err or e}") from e
    print("[LongCat] Clone finished.")


ensure_repo()

_repo_abs = os.path.abspath(REPO_ROOT)
if _repo_abs not in sys.path:
    sys.path.insert(0, _repo_abs)

import numpy as np
import torch
from PIL import Image
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, UMT5EncoderModel

from longcat_video.context_parallel import context_parallel_util
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.modules.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from longcat_video.pipeline_longcat_video import LongCatVideoPipeline


def ensure_weights(local_dir: str = CHECKPOINT_DIR) -> None:
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        return
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=HF_MODEL_ID,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", "*.gitattributes", "assets/*"],
    )


def torch_gc() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _to_pil_image(
    image: Optional[Union[np.ndarray, Image.Image]],
) -> Optional[Image.Image]:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def load_pipeline(
    checkpoint_dir: str = CHECKPOINT_DIR,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
) -> LongCatVideoPipeline:
    ensure_weights(checkpoint_dir)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch_dtype or (torch.bfloat16 if dev == "cuda" else torch.float32)

    cp_split_hw = context_parallel_util.get_optimal_split(1)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir, subfolder="tokenizer"
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        checkpoint_dir, subfolder="text_encoder", torch_dtype=dtype
    )
    vae = AutoencoderKLWan.from_pretrained(
        checkpoint_dir, subfolder="vae", torch_dtype=dtype
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        checkpoint_dir, subfolder="scheduler"
    )
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        checkpoint_dir,
        enable_flashattn3=False,
        enable_flashattn2=False,
        enable_xformers=True,
        subfolder="dit",
        cp_split_hw=cp_split_hw,
        torch_dtype=dtype,
    )

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )

    cfg_step_lora = os.path.join(checkpoint_dir, "lora", "cfg_step_lora.safetensors")
    refinement_lora = os.path.join(checkpoint_dir, "lora", "refinement_lora.safetensors")
    pipe.dit.load_lora(cfg_step_lora, "cfg_step_lora")
    pipe.dit.load_lora(refinement_lora, "refinement_lora")

    # -----------------------------------------------------------------------
    # Patch multi_lora_forward to be device-agnostic.
    # The original casts dtype only; LoRA weights stay on CPU during offload
    # causing "tensors on different devices" crash.
    # -----------------------------------------------------------------------
    import types
    original_multi_lora_forward = pipe.dit.__class__.multi_lora_forward

    def _patched_multi_lora_forward(self, module, x):
        # Run base linear on whatever device x is on
        result = module.original_forward(x)
        for lora_name, lora in module.loras.items():
            if not lora.enabled:
                continue
            # Cast input to LoRA weight device AND dtype
            lx = lora.lora_down(
                x.to(device=lora.lora_down.weight.device,
                     dtype=lora.lora_down.weight.dtype)
            )
            lx = lora.lora_up(lx)
            # Cast result back to original device/dtype before adding
            result = result + lora.scale * lx.to(device=x.device, dtype=x.dtype)
        return result

    pipe.dit.__class__.multi_lora_forward = _patched_multi_lora_forward

    # -----------------------------------------------------------------------
    # Block-level CPU offload for DiT.
    # Only one transformer block lives on GPU at a time (~25GB / N_blocks).
    # text_encoder and vae are small enough to stay on GPU full time.
    # -----------------------------------------------------------------------
    text_encoder.to(dev)
    vae.to(dev)
    # Keep dit non-block parts (embedders, norms, final layers) on GPU
    # but offload the heavy transformer blocks individually
    pipe.dit.to(dev)

    # Find the transformer blocks ModuleList
    # Based on traceback: pipe.dit iterates `block` in forward at line 343
    blocks_attr = None
    for attr_name in ["blocks", "transformer_blocks", "layers", "double_blocks", "single_blocks"]:
        if hasattr(pipe.dit, attr_name):
            candidate = getattr(pipe.dit, attr_name)
            if isinstance(candidate, torch.nn.ModuleList) and len(candidate) > 0:
                blocks_attr = attr_name
                print(f"[LongCat] Found DiT blocks at pipe.dit.{attr_name} ({len(candidate)} blocks)")
                break

    if blocks_attr is None:
        # Fallback: find any ModuleList with the most elements
        best = (None, 0)
        for name, mod in pipe.dit.named_children():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) > best[1]:
                best = (name, len(mod))
        blocks_attr = best[0]
        print(f"[LongCat] Fallback: using pipe.dit.{blocks_attr} ({best[1]} blocks)")

    if blocks_attr is not None:
        blocks = getattr(pipe.dit, blocks_attr)
        # Offload all blocks to CPU, hook each individually
        hook = None
        for block in blocks:
            block.to("cpu")
            _, hook = cpu_offload_with_hook(block, dev, prev_module_hook=hook)
        print(f"[LongCat] Block-level offload applied to {len(blocks)} DiT blocks")
    else:
        print("[LongCat] WARNING: Could not find DiT blocks, falling back to full dit offload")
        pipe.dit.to("cpu")
        _, hook = cpu_offload_with_hook(pipe.dit, dev)

    return pipe
    
def generate_video(
    pipe: LongCatVideoPipeline,
    mode: Literal["t2v", "i2v"],
    prompt: str,
    *,
    negative_prompt: str = (
        "ugly, blurry, low quality, static, subtitles, watermark"
    ),
    image: Optional[Union[np.ndarray, Image.Image]] = None,
    height: int = 480,
    width: int = 832,
    resolution: Literal["480p", "720p"] = "480p",
    seed: int = 42,
    use_distill: bool = True,
    use_refine: bool = False,
    num_frames: int = 93,
    device: Optional[str] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> np.ndarray:
    """
    Run LongCat-Video (t2v or i2v), optionally with refinement stage.
    Returns video as numpy array (frames in model output format, same as upstream pipe).
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=dev).manual_seed(int(seed))

    def _prog(p: float, desc: str) -> None:
        if progress:
            progress(p, desc)

    is_distill = use_distill or use_refine

    if is_distill:
        pipe.dit.enable_loras(["cfg_step_lora"])
        num_inference_steps = 16
        guidance_scale = 1.0
        current_neg_prompt = ""
    else:
        num_inference_steps = 50
        guidance_scale = 4.0
        current_neg_prompt = negative_prompt

    _prog(0.0, "Stage 1: base generation")

    if mode == "t2v":
        output = pipe.generate_t2v(
            prompt=prompt,
            negative_prompt=current_neg_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            use_distill=is_distill,
            guidance_scale=guidance_scale,
            generator=generator,
        )[0]
    elif mode == "i2v":
        pil_image = _to_pil_image(image)
        if pil_image is None:
            raise ValueError("i2v requires a non-null image")
        output = pipe.generate_i2v(
            image=pil_image,
            prompt=prompt,
            negative_prompt=current_neg_prompt,
            resolution=resolution,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            use_distill=is_distill,
            guidance_scale=guidance_scale,
            generator=generator,
        )[0]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if is_distill:
        pipe.dit.disable_all_loras()

    torch_gc()

    if use_refine:
        _prog(0.5, "Stage 2: refinement")
        pipe.dit.enable_loras(["refinement_lora"])
        pipe.dit.enable_bsa()

        stage1_video_pil = [(frame * 255).astype(np.uint8) for frame in output]
        stage1_video_pil = [Image.fromarray(img) for img in stage1_video_pil]

        refine_image = _to_pil_image(image) if mode == "i2v" else None

        output = pipe.generate_refine(
            image=refine_image,
            prompt=prompt,
            stage1_video=stage1_video_pil,
            num_cond_frames=1 if mode == "i2v" else 0,
            num_inference_steps=50,
            generator=generator,
        )[0]

        pipe.dit.disable_all_loras()
        pipe.dit.disable_bsa()
        torch_gc()

    _prog(1.0, "Done")
    return output


def export_video(
    frames: np.ndarray,
    output_path: str,
    *,
    use_refine: bool,
) -> str:
    fps = 30 if use_refine else 15
    export_to_video(frames, output_path, fps=fps)
    return output_path


# ---------------------------------------------------------------------------
# Example CLI-style run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from diffusers.utils import load_image

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _dtype = torch.bfloat16 if _device == "cuda" else torch.float32

    t0 = time.perf_counter()
    print("Loading pipeline...")
    _pipe = load_pipeline(device=_device, torch_dtype=_dtype)
    print(f"Load time: {time.perf_counter() - t0:.2f}s")

    _prompt = "The cat in the image wags its tail and blinks."
    _neg = "ugly, blurry, low quality, static, subtitles, watermark"
    _init = load_image(
        "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/input_image.jpg"
    )

    _out = "longcat_i2v_demo.mp4"
    t1 = time.perf_counter()
    _frames = generate_video(
        _pipe,
        "i2v",
        _prompt,
        negative_prompt=_neg,
        num_frames=93,
        image=_init,
        resolution="480p",
        seed=42,
        use_distill=True,
        use_refine=False,
        progress=lambda p, d: print(f"  [{p:.0%}] {d}"),
    )
    export_video(_frames, _out, use_refine=False)
    print(f"Output: {_out} (inference {time.perf_counter() - t1:.2f}s)")
