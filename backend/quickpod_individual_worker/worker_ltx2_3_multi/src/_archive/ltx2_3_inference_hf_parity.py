"""
Hugging Face Space–style **native** LTX-2.3 distilled inference via ``ltx-core`` + ``ltx-pipelines``
when those packages are installed and checkpoints are available; otherwise (or on OOM) falls back
to the Diffusers worker in ``ltx2_3_inference copy.py`` (rockapaper SDNQ path).

Install (same idea as `LTX-2.3-Lora/app.py`): clone `Lightricks/LTX-2`, then
``pip install -e packages/ltx-core`` and ``pip install -e packages/ltx-pipelines``,
plus compatible ``xformers``. Set ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``.

Env (native weights, optional — otherwise Hub download under output dir):

- ``LTX2_3_NATIVE_DISTILLED_CKPT`` — path to ``ltx-2.3-22b-distilled.safetensors``
- ``LTX2_3_NATIVE_SPATIAL_UPSAMPLER`` — path to spatial upscaler ``.safetensors``
- ``LTX2_3_GEMMA_ROOT`` — local directory of Gemma snapshot (e.g. from ``snapshot_download``)

Force Diffusers compat (no native attempt):

- ``LTX2_3_HF_PARITY_FORCE_COMPAT=1`` or job param ``force_compat_diffusers``: true

**Clone without ``pip install -e``:** set ``LTX2_REPO_ROOT`` to the **LTX-2 git root** (directory that contains
``packages/ltx-core`` and ``packages/ltx-pipelines``). This module prepends ``packages/*/src`` to ``sys.path`` and
passes the same paths to subprocesses via ``PYTHONPATH`` (e.g. ``/work_space/LTX-2``).
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image as PILImage

_FORCE_COMPAT_ENV = "LTX2_3_HF_PARITY_FORCE_COMPAT"


def _discover_ltx2_repo_root() -> Optional[Path]:
    """Return LTX-2 repo root if ``packages/ltx-core/src`` and ``packages/ltx-pipelines/src`` exist."""
    for key in ("LTX2_REPO_ROOT", "LTX_2_REPO_ROOT", "LTX_REPO_ROOT"):
        v = (os.environ.get(key) or "").strip()
        if v:
            p = Path(v).expanduser().resolve()
            if (p / "packages" / "ltx-pipelines" / "src" / "ltx_pipelines").is_dir():
                return p
    for guess in (
        Path("/work_space/LTX-2"),
        Path.cwd() / "LTX-2",
        Path(__file__).resolve().parent / "LTX-2",
    ):
        try:
            p = guess.resolve()
        except OSError:
            continue
        if (p / "packages" / "ltx-pipelines" / "src" / "ltx_pipelines").is_dir():
            return p
    return None


def _ltx2_src_pair(repo: Path) -> Tuple[str, str]:
    """Order: pipelines first so it wins if both expose the same top-level (unlikely)."""
    pipe_src = str((repo / "packages" / "ltx-pipelines" / "src").resolve())
    core_src = str((repo / "packages" / "ltx-core" / "src").resolve())
    return pipe_src, core_src


def _ensure_ltx2_repo_on_sys_path() -> Optional[Path]:
    """Insert LTX-2 package ``src`` dirs on ``sys.path`` for in-process imports."""
    repo = _discover_ltx2_repo_root()
    if repo is None:
        return None
    pipe_src, core_src = _ltx2_src_pair(repo)
    for d in (core_src, pipe_src):
        if d not in sys.path:
            sys.path.insert(0, d)
    return repo


_REPO_FOR_SUBPROCESS: Optional[Path] = _ensure_ltx2_repo_on_sys_path()


def _subprocess_env_with_ltx_path(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = os.environ.copy()
    repo = _discover_ltx2_repo_root()
    if repo is not None:
        pipe_src, core_src = _ltx2_src_pair(repo)
        prev = (env.get("PYTHONPATH") or "").strip()
        prefix = f"{pipe_src}{os.pathsep}{core_src}"
        env["PYTHONPATH"] = f"{prefix}{os.pathsep}{prev}" if prev else prefix
    if extra:
        env.update(extra)
    return env


def _copy_inference_module() -> Any:
    """Load ``ltx2_3_inference copy.py`` (space in filename) as a module."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "ltx2_3_inference copy.py")
    spec = importlib.util.spec_from_file_location("ltx2_3_inference_worker_copy", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load worker copy from {path!r}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ``distilled.main()`` calls ``detect_checkpoint_path()`` before building the full argparse parser; that
# pre-parser requires ``--distilled-checkpoint-path`` and uses ``add_help=False``, so bare ``--help`` fails.
_CLI_HELP_DUMMY_CKPT = "/tmp/.ltx_hf_parity_distilled_help_placeholder.safetensors"


def ltx_distilled_cli_help() -> str:
    """Run ``ltx_pipelines.distilled --help`` (needs a dummy checkpoint flag first; file need not exist)."""
    try:
        r = subprocess.run(
            [
                sys.executable,
                "-W",
                "ignore::RuntimeWarning",
                "-m",
                "ltx_pipelines.distilled",
                "--distilled-checkpoint-path",
                _CLI_HELP_DUMMY_CKPT,
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=_subprocess_env_with_ltx_path(),
        )
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        if err:
            skip = _CLI_HELP_DUMMY_CKPT
            err_lines = [
                ln
                for ln in err.splitlines()
                if not (skip in ln and "Could not read checkpoint metadata" in ln)
            ]
            err = "\n".join(err_lines).strip()
        return (out + ("\n" + err if err else "")).strip() or "(empty stdout/stderr)"
    except Exception as e:
        return f"(ltx_pipelines.distilled --help failed: {type(e).__name__}: {e})"


def _force_compat(p: Optional[Dict[str, Any]] = None) -> bool:
    if p is not None and p.get("force_compat_diffusers") is not None:
        v = p.get("force_compat_diffusers")
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in {"1", "true", "yes", "on"}
    return os.environ.get(_FORCE_COMPAT_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _patch_xformers_attention() -> None:
    try:
        from ltx_core.model.transformer import attention as _attn_mod  # type: ignore
        from xformers.ops import memory_efficient_attention as _mea  # type: ignore

        _attn_mod.memory_efficient_attention = _mea
    except Exception:
        pass


def _native_weights(p: Dict[str, Any]) -> Tuple[str, str, str]:
    out_root = Path(
        str(p.get("output_dir") or os.environ.get("WORKSPACE_DIR") or "/workspace")
    )
    cache = out_root / "ltx_native_hf_weights"
    cache.mkdir(parents=True, exist_ok=True)

    ckpt = (os.environ.get("LTX2_3_NATIVE_DISTILLED_CKPT") or "").strip()
    if not ckpt:
        from huggingface_hub import hf_hub_download

        ckpt = hf_hub_download(
            repo_id="Lightricks/LTX-2.3",
            filename="ltx-2.3-22b-distilled.safetensors",
            local_dir=str(cache),
        )
    up = (os.environ.get("LTX2_3_NATIVE_SPATIAL_UPSAMPLER") or "").strip()
    if not up:
        from huggingface_hub import hf_hub_download

        up = hf_hub_download(
            repo_id="Lightricks/LTX-2.3",
            filename="ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
        )
    gemma = (os.environ.get("LTX2_3_GEMMA_ROOT") or "").strip()
    if not gemma:
        from huggingface_hub import snapshot_download

        gemma = snapshot_download(repo_id="Lightricks/gemma-3-12b-it-qat-q4_0-unquantized")
    return str(ckpt), str(up), str(gemma)


def _try_run_native_distilled(
    *,
    weight_ctx: Dict[str, Any],
    prompt: str,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    seed: int,
    images: List[Any],
    output_mp4: str,
    enhance_prompt: bool,
    audio_path: Optional[str],
) -> None:
    from ltx_core.model.video_vae import (  # type: ignore
        TilingConfig,
        get_video_chunks_number,
    )
    from ltx_core.quantization import QuantizationPolicy  # type: ignore
    from ltx_pipelines.distilled import DistilledPipeline  # type: ignore
    from ltx_pipelines.utils.media_io import encode_video  # type: ignore

    if audio_path:
        raise NotImplementedError(
            "Native hf_parity with audio_path needs the HF A2V distilled subclass; "
            "use force_compat_diffusers or the Diffusers worker."
        )

    ckpt, ups, gemma = _native_weights(weight_ctx)
    _patch_xformers_attention()

    pipeline = DistilledPipeline(
        distilled_checkpoint_path=ckpt,
        spatial_upsampler_path=ups,
        gemma_root=gemma,
        loras=[],
        quantization=QuantizationPolicy.fp8_cast(),
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    with torch.inference_mode():
        video, audio = pipeline(
            prompt=prompt,
            seed=int(seed),
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            frame_rate=float(frame_rate),
            images=images,
            tiling_config=tiling_config,
            enhance_prompt=bool(enhance_prompt),
        )
    encode_video(
        video=video,
        fps=frame_rate,
        audio=audio,
        output_path=output_mp4,
        video_chunks_number=video_chunks_number,
    )


def _align_frames_like_copy(n: int, copy_mod: Any) -> int:
    return int(copy_mod._align_frames_8k_plus_1(int(n)))


def run_inference_hf_parity_t2v(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    p = dict(params or {})
    copy_mod = _copy_inference_module()
    if _force_compat(p):
        return copy_mod.run_inference_ltx2_3_t2v(job_id, prompt, p)
    try:
        import ltx_pipelines  # noqa: F401
    except Exception:
        return copy_mod.run_inference_ltx2_3_t2v(job_id, prompt, p)

    w, h = copy_mod._align_spatial(int(p.get("width", 768)), int(p.get("height", 512)))
    nf = _align_frames_like_copy(
        int(p.get("num_frames", p.get("frames", 121))), copy_mod
    )
    fps = float(p.get("frame_rate", p.get("fps", 24.0)))
    seed = int(p.get("seed", 42))
    os.makedirs(p.get("output_dir", copy_mod.DEFAULT_OUTPUT_DIR), exist_ok=True)
    out_path = os.path.join(
        p.get("output_dir", copy_mod.DEFAULT_OUTPUT_DIR), f"hf_parity_t2v_{job_id}.mp4"
    )
    try:
        _try_run_native_distilled(
            weight_ctx=p,
            prompt=prompt,
            height=h,
            width=w,
            num_frames=nf,
            frame_rate=fps,
            seed=seed,
            images=[],
            output_mp4=out_path,
            enhance_prompt=bool(p.get("enhance_prompt", False)),
            audio_path=(str(p.get("audio_path")).strip() or None) if p.get("audio_path") else None,
        )
        return os.path.abspath(out_path)
    except torch.cuda.OutOfMemoryError:
        return copy_mod.run_inference_ltx2_3_t2v(job_id, prompt, p)
    except Exception:
        return copy_mod.run_inference_ltx2_3_t2v(job_id, prompt, p)


def run_inference_hf_parity_i2v(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    p = dict(params or {})
    copy_mod = _copy_inference_module()
    if _force_compat(p):
        return copy_mod.run_inference_ltx2_3_i2v_distilled(job_id, prompt, p)
    try:
        import ltx_pipelines  # noqa: F401
        from ltx_pipelines.utils.args import ImageConditioningInput  # type: ignore
    except Exception:
        return copy_mod.run_inference_ltx2_3_i2v_distilled(job_id, prompt, p)

    url = p.get("init_image") or p.get("image") or p.get("image_input")
    if not url:
        raise ValueError("init_image is required for hf_parity i2v")
    im = copy_mod._open_image_uri(str(url))
    out_root = Path(str(p.get("output_dir", copy_mod.DEFAULT_OUTPUT_DIR)))
    out_root.mkdir(parents=True, exist_ok=True)
    tmp_img = out_root / f"hf_parity_i2v_first_{job_id}.png"
    im.save(tmp_img)

    upsample = copy_mod._spatial_upsample_enabled(p, default=True)
    if upsample and copy_mod._condition_resolution_cap_enabled(p):
        max_w, max_h = copy_mod._max_condition_canvas_pixels(p)
        w0, h0 = copy_mod._align_spatial(im.size[0], im.size[1])
        tw, th = copy_mod._scale_to_max_canvas(w0, h0, max_w, max_h)
        im = im.resize((tw, th), PILImage.Resampling.LANCZOS)
        w, h = tw, th
    elif upsample:
        w, h = copy_mod._align_spatial(int(p.get("width", 768)), int(p.get("height", 512)))
        if im.size != (w, h):
            im = im.resize((w, h), PILImage.Resampling.LANCZOS)
    else:
        cap = copy_mod._single_stage_max_condition_long_side_pixels(p)
        w, h = copy_mod._scale_to_max_canvas(im.size[0], im.size[1], cap, cap)
        if im.size != (w, h):
            im = im.resize((w, h), PILImage.Resampling.LANCZOS)
    im.save(tmp_img)

    nf = _align_frames_like_copy(
        int(p.get("num_frames", p.get("frames", 121))), copy_mod
    )
    fps = float(p.get("frame_rate", p.get("fps", 24.0)))
    seed = int(p.get("seed", 42))
    out_path = os.path.join(
        str(p.get("output_dir", copy_mod.DEFAULT_OUTPUT_DIR)), f"hf_parity_i2v_{job_id}.mp4"
    )
    images = [
        ImageConditioningInput(path=str(tmp_img), frame_idx=0, strength=1.0),
    ]
    try:
        _try_run_native_distilled(
            weight_ctx=p,
            prompt=prompt,
            height=h,
            width=w,
            num_frames=nf,
            frame_rate=fps,
            seed=seed,
            images=images,
            output_mp4=out_path,
            enhance_prompt=bool(p.get("enhance_prompt", False)),
            audio_path=(str(p.get("audio_path")).strip() or None) if p.get("audio_path") else None,
        )
        return os.path.abspath(out_path)
    except torch.cuda.OutOfMemoryError:
        return copy_mod.run_inference_ltx2_3_i2v_distilled(job_id, prompt, p)
    except Exception:
        return copy_mod.run_inference_ltx2_3_i2v_distilled(job_id, prompt, p)


def run_inference_hf_parity_flf2v(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    p = dict(params or {})
    copy_mod = _copy_inference_module()
    if _force_compat(p):
        return copy_mod.run_inference_ltx2_3_flf2v(job_id, prompt, p)
    try:
        import ltx_pipelines  # noqa: F401
        from ltx_pipelines.utils.args import ImageConditioningInput  # type: ignore
    except Exception:
        return copy_mod.run_inference_ltx2_3_flf2v(job_id, prompt, p)

    first_u = p.get("first_frame_image") or p.get("first_image")
    last_u = p.get("last_frame_image") or p.get("last_image")
    if not first_u or not last_u:
        raise ValueError("first_frame_image and last_frame_image required for flf2v")
    first = copy_mod._open_image_uri(str(first_u))
    last = copy_mod._open_image_uri(str(last_u))
    out_root = Path(str(p.get("output_dir", copy_mod.DEFAULT_OUTPUT_DIR)))
    out_root.mkdir(parents=True, exist_ok=True)
    p_first = out_root / f"hf_parity_flf_first_{job_id}.png"
    p_last = out_root / f"hf_parity_flf_last_{job_id}.png"
    first.save(p_first)
    last.save(p_last)

    upsample = copy_mod._spatial_upsample_enabled(p, default=True)
    if upsample and copy_mod._condition_resolution_cap_enabled(p):
        max_w, max_h = copy_mod._max_condition_canvas_pixels(p)
        w0, h0 = copy_mod._align_spatial(first.size[0], first.size[1])
        tw, th = copy_mod._scale_to_max_canvas(w0, h0, max_w, max_h)
        first = first.resize((tw, th), PILImage.Resampling.LANCZOS)
        last = last.resize((tw, th), PILImage.Resampling.LANCZOS)
        w, h = tw, th
    elif upsample:
        w, h = copy_mod._align_spatial(int(p.get("width", 768)), int(p.get("height", 512)))
        if first.size != (w, h):
            first = first.resize((w, h), PILImage.Resampling.LANCZOS)
            last = last.resize((w, h), PILImage.Resampling.LANCZOS)
    else:
        cap = copy_mod._single_stage_max_condition_long_side_pixels(p)
        w, h = copy_mod._scale_to_max_canvas(first.size[0], first.size[1], cap, cap)
        first = first.resize((w, h), PILImage.Resampling.LANCZOS)
        last = last.resize((w, h), PILImage.Resampling.LANCZOS)
    first.save(p_first)
    last.save(p_last)

    nf = _align_frames_like_copy(
        int(p.get("num_frames", p.get("frames", 121))), copy_mod
    )
    fps = float(p.get("frame_rate", p.get("fps", 24.0)))
    seed = int(p.get("seed", 42))
    out_path = os.path.join(
        str(p.get("output_dir", copy_mod.DEFAULT_OUTPUT_DIR)), f"hf_parity_flf2v_{job_id}.mp4"
    )
    images = [
        ImageConditioningInput(path=str(p_first), frame_idx=0, strength=float(p.get("first_strength", 1.0))),
        ImageConditioningInput(
            path=str(p_last), frame_idx=nf - 1, strength=float(p.get("last_strength", 1.0))
        ),
    ]
    try:
        _try_run_native_distilled(
            weight_ctx=p,
            prompt=prompt,
            height=h,
            width=w,
            num_frames=nf,
            frame_rate=fps,
            seed=seed,
            images=images,
            output_mp4=out_path,
            enhance_prompt=bool(p.get("enhance_prompt", False)),
            audio_path=None,
        )
        return os.path.abspath(out_path)
    except torch.cuda.OutOfMemoryError:
        return copy_mod.run_inference_ltx2_3_flf2v(job_id, prompt, p)
    except Exception:
        return copy_mod.run_inference_ltx2_3_flf2v(job_id, prompt, p)


def run_inference_hf_parity_iv2v(job_id: str, prompt: str, params: Dict[str, Any]) -> str:
    """``ltx_pipelines`` distilled path used in HF Space targets first/last image conditioning; iv2v uses Diffusers."""
    copy_mod = _copy_inference_module()
    return copy_mod.run_inference_ltx2_3_iv2v_distilled(job_id, prompt, dict(params or {}))


def _default_i2v_prompt_from_copy_worker() -> str:
    """Match smoke ``prompt_i2v`` in ``ltx2_3_inference copy.py`` without importing its ``__main__`` block."""
    path = Path(__file__).resolve().parent / "ltx2_3_inference copy.py"
    if not path.is_file():
        return "A cinematic shot of the subject in motion."
    text = path.read_text(encoding="utf-8")
    m = re.search(r"(?m)^[ \t]*prompt_i2v\s*=\s*\"\"\"(.*?)\"\"\"", text, re.DOTALL)
    return (m.group(1).strip() if m else "A cinematic shot of the subject in motion.")


def _apply_copy_worker_offload_cli(
    copy_mod: Any, offload: str, stage2_offload: Optional[str]
) -> None:
    copy_mod._OFFLOAD_MODE = "" if offload == "auto" else str(offload)
    copy_mod._STAGE2_OFFLOAD_CLI = stage2_offload


def _hf_parity_cli_download_if_url(
    uri: str, dest_dir: str, filename: Optional[str] = None
) -> str:
    import requests
    from urllib.parse import unquote, urlparse

    u = uri.strip()
    if not (u.startswith("http://") or u.startswith("https://")):
        return os.path.abspath(u) if os.path.isfile(u) else u
    path = unquote(urlparse(u).path)
    base = os.path.basename(path)
    name = filename or (base if base else "download.bin")
    out_abs = os.path.join(dest_dir, name)
    vid = name.lower().endswith((".mp4", ".webm", ".mov", ".mkv"))
    r = requests.get(u, timeout=300 if vid else 180)
    r.raise_for_status()
    os.makedirs(dest_dir, exist_ok=True)
    with open(out_abs, "wb") as fo:
        fo.write(r.content)
    return os.path.abspath(out_abs)


if __name__ == "__main__":
    import argparse
    import random
    import time
    import uuid

    _ap = argparse.ArgumentParser(
        description=(
            "HF Space–style LTX-2.3 distilled inference (native ltx_pipelines when possible, "
            "else Diffusers worker). With no CLI args, prints upstream ``ltx_pipelines.distilled --help``."
        )
    )
    _ap.add_argument(
        "--print-native-distilled-help",
        action="store_true",
        help="Print only upstream ``python -m ltx_pipelines.distilled --help`` (dummy checkpoint workaround).",
    )
    _ap.add_argument(
        "--mode",
        choices=("t2v", "i2v", "flf2v", "iv2v", "all"),
        default=None,
        help="Smoke mode. Omit to print native distilled CLI help (legacy no-arg behavior).",
    )
    _ap.add_argument("--output-dir", default="./ltx2_3_v3_test_outputs")
    _ap.add_argument("--init-image", default="", help="i2v: local path or http(s) URL.")
    _ap.add_argument("--num-frames", type=int, default=361)
    _ap.add_argument("--width", type=int, default=768)
    _ap.add_argument("--height", type=int, default=512)
    _ap.add_argument("--no-upsample", action="store_true")
    _ap.add_argument("--single-stage-max-long", type=int, default=None)
    _ap.add_argument(
        "--offload",
        choices=("auto", "none", "sequential", "model"),
        default="auto",
    )
    _ap.add_argument("--civitai-lora-url", default="")
    _ap.add_argument("--civitai-version-id", type=int, default=None)
    _ap.add_argument("--lora-strength", type=float, default=None)
    _ap.add_argument(
        "--lora-apply",
        choices=("both", "stage1", "stage2"),
        default=None,
    )
    _ap.add_argument(
        "--stage2-offload",
        choices=("sequential", "model", "none"),
        default=None,
    )
    _ap.add_argument("--profile", action="store_true")
    _ap.add_argument("--lora-fusion-cache", action="store_true")
    _ap.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="Native path: pass enhance_prompt to DistilledPipeline (Diffusers path may ignore).",
    )
    _ap.add_argument(
        "--force-compat-diffusers",
        action="store_true",
        help="Skip native ltx_pipelines and use Diffusers worker only.",
    )
    _ap.add_argument(
        "--prompt",
        default="",
        help="Override text prompt. For i2v default, the worker copy's prompt_i2v is read from disk if omitted.",
    )
    _args = _ap.parse_args()

    if _args.print_native_distilled_help:
        print("=== ltx_pipelines.distilled --help (OOM / device spike) ===")
        print(ltx_distilled_cli_help())
        raise SystemExit(0)

    if _args.mode is None:
        print("=== ltx_pipelines.distilled --help (OOM / device spike) ===")
        print(ltx_distilled_cli_help())
        print(
            "\n--- hf_parity ---\n"
            "Re-run with --mode t2v|i2v|flf2v|iv2v|all and worker flags (e.g. --init-image, --output-dir). "
            "Use --print-native-distilled-help for the block above without this footer."
        )
        raise SystemExit(0)

    if _args.force_compat_diffusers:
        os.environ[_FORCE_COMPAT_ENV] = "1"

    _wall_t0 = time.perf_counter()
    copy_mod = _copy_inference_module()
    _apply_copy_worker_offload_cli(copy_mod, _args.offload, _args.stage2_offload)

    _smoke_num_frames = _args.num_frames
    if _args.offload == "none" and not _args.no_upsample and _smoke_num_frames > 121:
        print(
            f"[hf_parity] --offload none + 2-stage: capping --num-frames {_smoke_num_frames} -> 121 "
            "for stage-2 VRAM; use --no-upsample or offload auto."
        )
        _smoke_num_frames = 121

    os.makedirs(_args.output_dir, exist_ok=True)
    _jid = str(uuid.uuid4())[:8]
    _smoke_inputs_dir = os.path.join(_args.output_dir, "smoke_inputs", _jid)
    os.makedirs(_smoke_inputs_dir, exist_ok=True)

    prompt_t2v = (
        "A cinematic drone shot of coastal cliffs at golden hour, gentle waves, slow camera push-in."
    )
    prompt_i2v = (_args.prompt or "").strip() or _default_i2v_prompt_from_copy_worker()
    prompt_flf2v = (
        "CG animation style, a small blue bird takes off from the ground, flapping its wings. "
        "The bird's feathers are delicate, with a unique pattern on its chest. The background shows "
        "a blue sky with white clouds under bright sunshine. The camera follows the bird upward, "
        "capturing its flight and the vastness of the sky from a close-up, low-angle perspective."
    )
    prompt_iv2v = (
        "From the starting pose in the input image the subject moves naturally. "
        "Photorealistic details, consistent framing."
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
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/"
        "diffusers/astronaut.jpg"
    )
    _flf_first = (
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/"
        "flf2v_input_first_frame.png"
    )
    _flf_last = (
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/"
        "flf2v_input_last_frame.png"
    )

    _base_params: Dict[str, Any] = {
        "output_dir": _args.output_dir,
        "width": _args.width,
        "height": _args.height,
        "num_frames": _smoke_num_frames,
        "frame_rate": 24.0,
        "seed": random.randint(0, 1000),
        "use_spatial_upsampler": not _args.no_upsample,
        "enhance_prompt": bool(_args.enhance_prompt),
    }
    if _args.single_stage_max_long is not None:
        _base_params["single_stage_max_condition_long_side"] = int(_args.single_stage_max_long)
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
        fn = getattr(copy_mod, "_lora_cache_dir_from_params", None)
        if callable(fn):
            print(f"[hf_parity] Civitai LoRA URL set; cache dir: {fn(_base_params)}")

    def _run_one(name: str) -> float:
        if name == "t2v":
            use_prompt = (_args.prompt or "").strip() or prompt_t2v
            t0 = time.perf_counter()
            path = run_inference_hf_parity_t2v(_jid, use_prompt, dict(_base_params))
            elapsed = time.perf_counter() - t0
            print(f"[hf_parity t2v] ok in {elapsed:.2f}s -> {path}")
            return elapsed
        if name == "i2v":
            t0 = time.perf_counter()
            p = dict(_base_params)
            _src = (_args.init_image or "").strip()
            if _src:
                if _src.startswith("http://") or _src.startswith("https://"):
                    from urllib.parse import unquote, urlparse

                    path_u = unquote(urlparse(_src.strip()).path)
                    _local_i2v = _hf_parity_cli_download_if_url(
                        _src, _smoke_inputs_dir, os.path.basename(path_u) or "download.bin"
                    )
                    _msg_extra = f"saved {_local_i2v}"
                else:
                    _local_i2v = os.path.abspath(_src)
                    if not os.path.isfile(_local_i2v):
                        raise SystemExit(f"[hf_parity i2v] --init-image is not a file: {_local_i2v!r}")
                    _msg_extra = f"local file {_local_i2v}"
            else:
                _local_i2v = _hf_parity_cli_download_if_url(
                    _default_i2v_image, _smoke_inputs_dir, "astronaut.jpg"
                )
                _msg_extra = f"saved {_local_i2v}"
            im_sz = PILImage.open(_local_i2v).convert("RGB")
            try:
                raw_w, raw_h = im_sz.size
            finally:
                im_sz.close()
            iw, ih = copy_mod._align_spatial(raw_w, raw_h)
            p["width"], p["height"] = iw, ih
            p["init_image"] = _local_i2v
            if _args.no_upsample:
                cap = copy_mod._single_stage_max_condition_long_side_pixels(p)
                cw, ch = copy_mod._scale_to_max_canvas(raw_w, raw_h, cap, cap)
                print(
                    f"[hf_parity i2v] init_image {raw_w}x{raw_h} -> conditioning {cw}x{ch} "
                    f"(single-stage long-side cap {cap}px; {_msg_extra})"
                )
            else:
                print(f"[hf_parity i2v] using {iw}x{ih} from init_image ({_msg_extra})")
            path = run_inference_hf_parity_i2v(_jid + "_i2v", prompt_i2v, p)
            elapsed = time.perf_counter() - t0
            print(f"[hf_parity i2v] ok in {elapsed:.2f}s -> {path}")
            return elapsed
        if name == "flf2v":
            t0 = time.perf_counter()
            _local_ff = _hf_parity_cli_download_if_url(
                _flf_first, _smoke_inputs_dir, "flf2v_input_first_frame.png"
            )
            _local_lf = _hf_parity_cli_download_if_url(
                _flf_last, _smoke_inputs_dir, "flf2v_input_last_frame.png"
            )
            im_a = PILImage.open(_local_ff).convert("RGB")
            im_b = PILImage.open(_local_lf).convert("RGB")
            try:
                if im_a.size != im_b.size:
                    raise SystemExit(
                        f"flf2v: first/last frame size mismatch {im_a.size} vs {im_b.size}"
                    )
                fw, fh = copy_mod._align_spatial(im_a.size[0], im_a.size[1])
            finally:
                im_a.close()
                im_b.close()
            p = dict(_base_params)
            p["width"], p["height"] = fw, fh
            p["first_frame_image"] = _local_ff
            p["last_frame_image"] = _local_lf
            use_prompt = (_args.prompt or "").strip() or prompt_flf2v
            print(f"[hf_parity flf2v] using {fw}x{fh} ({_smoke_inputs_dir})")
            path = run_inference_hf_parity_flf2v(_jid + "_flf", use_prompt, p)
            elapsed = time.perf_counter() - t0
            print(f"[hf_parity flf2v] ok in {elapsed:.2f}s -> {path}")
            return elapsed
        if name == "iv2v":
            t0 = time.perf_counter()
            p = dict(_base_params)
            _local_cimg = _hf_parity_cli_download_if_url(
                _cosmos_img, _smoke_inputs_dir, "cosmos_video2world-input.jpg"
            )
            _local_cvid = _hf_parity_cli_download_if_url(
                _cosmos_vid, _smoke_inputs_dir, "cosmos-video2world-input-vid.mp4"
            )
            im_sz = PILImage.open(_local_cimg).convert("RGB")
            try:
                iw, ih = copy_mod._align_spatial(im_sz.size[0], im_sz.size[1])
            finally:
                im_sz.close()
            p["width"], p["height"] = iw, ih
            p["condition_video"] = _local_cvid
            p["condition_image"] = _local_cimg
            p["condition_video_index"] = 0
            p["condition_image_index"] = 8
            use_prompt = (_args.prompt or "").strip() or prompt_iv2v
            print(f"[hf_parity iv2v] using {iw}x{ih} ({_smoke_inputs_dir})")
            path = run_inference_hf_parity_iv2v(_jid + "_iv2v", use_prompt, p)
            elapsed = time.perf_counter() - t0
            print(f"[hf_parity iv2v] ok in {elapsed:.2f}s -> {path}")
            return elapsed
        return 0.0

    _modes = ("t2v", "i2v", "flf2v", "iv2v") if _args.mode == "all" else (_args.mode,)
    if any(x in _modes for x in ("i2v", "flf2v", "iv2v")):
        print(f"[hf_parity] URL inputs under: {_smoke_inputs_dir}")
    _gen_seconds: Dict[str, float] = {}
    for _m in _modes:
        print(f"--- hf_parity running {_m} ---")
        _gen_seconds[_m] = _run_one(_m)
    print("--- hf_parity generation time ---")
    for _m in _modes:
        print(f"  {_m}: {_gen_seconds[_m]:.2f}s")
    if len(_modes) > 1:
        print(f"  sum: {sum(_gen_seconds[m] for m in _modes):.2f}s")
    print(f"  wall: {time.perf_counter() - _wall_t0:.2f}s")
    print("Done.")
