"""
Bake all omni-multi shared weights into each flavor image (no runtime download for these).

Run from each *_multi/src/prep_model.py after the primary model snapshot.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import traceback
import urllib.request
from typing import Dict, Optional, Tuple

from huggingface_hub import hf_hub_download, snapshot_download

from qwen_2511_lora_registry import QWEN_2511_LORA_ADAPTERS

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

ROCKAPAPER_REALESRGAN = os.getenv("REALESRGAN_HF_REPO", "rockapaper/realesrgan")
QWEN_REPO = os.getenv("HF_REPO_QWEN_BAKED", "Qwen/Qwen-Image-Edit-2511")
AIO_REPO = os.getenv("HF_REPO_QWEN_AIO_TRANSFORMER", "Pr0f3ssi0n4ln00b/Phr00t-Qwen-Rapid-AIO")
AIO_VERSION_RAW = os.getenv("AIO_VERSION", "v19")
AIO_VERSION = (
    AIO_VERSION_RAW if str(AIO_VERSION_RAW).startswith("v") else f"v{AIO_VERSION_RAW}"
)
LAMA_REPO = os.getenv("LAMA_HF_REPO", "JosephCatrambone/big-lama-torchscript")
LAMA_FILENAME = os.getenv("LAMA_HF_FILENAME", "lama.pt")
AURASR_REPO = os.getenv("HF_REPO_AURASR", "fal/AuraSR-v2")
BIREFNET_REPO = os.getenv("HF_REPO_BIREFNET", "ZhengPeng7/BiRefNet")
RIFE_WEIGHTS_REPO = os.getenv("HF_REPO_RIFE", "AlexWortega/RIFE")
RIFE_GIT_URL = os.getenv("RIFE_GIT_URL", "https://github.com/hzwer/RIFE.git")

GLM_OCR_REPO = os.getenv("HF_REPO_GLM_OCR", "zai-org/GLM-OCR")

# Single-file stage-2 distilled LoRA for LTX-2.3 full-pipeline latent upsample (T2V/FLF2V); see Diffusers LTX2 docs.
LTX23_STAGE2_LORA_REPO = os.getenv("LTX2_3_STAGE2_LORA_REPO", "Lightricks/LTX-2.3")
LTX23_STAGE2_LORA_FILE = os.getenv(
    "LTX2_3_STAGE2_LORA_WEIGHT", "ltx-2.3-22b-distilled-lora-384.safetensors"
)

SWIN2SR_REPOS = {
    "swin2sr_x2": "caidas/swin2SR-classical-sr-x2-64",
    "swin2sr_x4": "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
    "swin2sr_x4_classic": "caidas/swin2SR-classical-sr-x4-64",
}

FACE_MODELS: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "realesr-general-x4v3.pth": (
        None,
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    ),
    "RealESRGAN_x4plus_anime_6B.pth": (
        None,
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    ),
    "GFPGANv1.2.pth": (
        None,
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth",
    ),
    "GFPGANv1.3.pth": (
        None,
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
    ),
    "GFPGANv1.4.pth": (
        None,
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    ),
    "RestoreFormer.pth": (
        None,
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth",
    ),
    "CodeFormer.pth": (
        None,
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/CodeFormer.pth",
    ),
}

# For each *_multi image, keep primary/shared inference code under
# <multi>/src/inference/<worker_dir>/src/*.py so Docker only needs COPY <multi>/src.
MULTI_PRIMARY_WORKER_DIR: Dict[str, str] = {
    "worker_flux_klein_9b_multi": "worker_flux_klein_9b",
    "worker_qwen_image_edit_2511_rapid_aio_multi": "worker_qwen_image_edit_2511_rapid_aio",
    "worker_z_image_turbo_multi": "worker_z_image_turbo",
    "worker_stable_diffusion_xl_multi": "worker_stable_diffusion_xl",
    "worker_stable_diffusion_xl_outpaint_multi": "worker_stable_diffusion_xl_outpaint",
    "worker_wan_22_fp8_lightx2v_t2v_multi": "worker_wan_22_fp8_lightx2v_t2v",
    "worker_wan_22_fp8_lightx2v_i2v_multi": "worker_wan_22_fp8_lightx2v_i2v",
    "worker_hunyan15_480p_t2V_distilled_multi": "worker_hunyan15_480p_t2V_distilled",
    "worker_hunyan15_480p_i2V_step_distilled_multi": "worker_hunyan15_480p_i2V_step_distilled",
    "worker_basic_utils_multi": "worker_basic_utils",
    "worker_video_upscalers_multi": "worker_video_upscalers",
    "worker_face_enhancer_multi": "worker_face_enhancer",
    "worker_image_upscaler_aurasrv2_multi": "worker_image_upscaler_aurasrv2",
    "worker_image_upscaler_swin2sr_multi": "worker_image_upscaler_swin2sr",
    "worker_glm_ocr_multi": "worker_glm_ocr",
    "worker_ltx2_3_multi": "worker_ltx2_3",
}

SHARED_WORKER_DIRS: Tuple[str, ...] = (
    "worker_basic_utils",
    "worker_image_upscaler_aurasrv2",
    "worker_image_upscaler_swin2sr",
    "worker_face_enhancer",
    "worker_video_upscalers",
    "worker_glm_ocr",
)

INFERENCE_EXCLUDE_FILENAMES: frozenset = frozenset(
    {
        "app.py",
        "prep_model.py",
        "generate_dockerfile.py",
        "merge.py",
        "huggingface_app.py",
        "requirements.txt",
    }
)


def _qpw_root(project_root: str) -> str:
    """Parent of the *_multi folder only — correct in a full checkout, wrong in slim Docker contexts."""
    return os.path.dirname(os.path.abspath(project_root))


def _resolve_qpw_root(project_root: str) -> str:
    """
    Directory that contains sibling worker_* trees (typically ``quickpod_individual_worker``).

    In a full repo, ``project_root`` is ``.../worker_foo_multi`` and its parent already holds
    ``worker_foo``, ``worker_basic_utils``, etc.

    In a slim Docker build, the parent may be ``/work_space`` with only the *_multi folder copied;
    then we walk up until we find a directory that contains the primary worker's ``src``, or honor
    ``QPW_ROOT`` / ``QUICKPOD_WORKERS_ROOT``.
    """
    project_root = os.path.abspath(project_root)
    for env_key in ("QPW_ROOT", "QUICKPOD_WORKERS_ROOT"):
        raw = (os.environ.get(env_key) or "").strip()
        if not raw:
            continue
        env_abs = os.path.abspath(raw)
        if os.path.isdir(env_abs):
            _log(f"Using {env_key}={env_abs} as quickpod workers root")
            return env_abs
        _log(f"WARN {env_key} is set but not a directory ({env_abs!r}); ignoring")

    multi_dir = os.path.basename(project_root)
    primary = MULTI_PRIMARY_WORKER_DIR.get(multi_dir)
    if primary:
        d = os.path.dirname(project_root)
        for _ in range(16):
            primary_src = os.path.join(d, primary, "src")
            if os.path.isdir(primary_src):
                return d
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent

    fallback = _qpw_root(project_root)
    _log(
        "WARN Could not locate sibling worker trees via primary src walk; "
        f"falling back to {fallback!r}. For Docker/slim checkouts set QPW_ROOT to your "
        "`quickpod_individual_worker` directory."
    )
    return fallback


def _log(msg: str) -> None:
    print(f"[omni-bake] {msg}")


def ensure_rockapaper_realesrgan(models_dir: str) -> None:
    d = os.path.join(models_dir, "realesrgan")
    os.makedirs(d, exist_ok=True)
    for fn in ("RealESRGAN_x2plus.pth", "RealESRGAN_x4plus.pth"):
        path = os.path.join(d, fn)
        if os.path.exists(path):
            continue
        try:
            hf_hub_download(
                repo_id=ROCKAPAPER_REALESRGAN,
                filename=fn,
                local_dir=d,
                resume_download=True,
            )
            _log(f"Real-ESRGAN {fn} ready")
        except Exception as e:
            _log(f"WARN Real-ESRGAN {fn}: {e}")
            traceback.print_exc()


def ensure_qwen_2511_stack(models_dir: str) -> None:
    qwen_dir = os.path.join(models_dir, "qwen_image_edit_2511")
    aio_dir = os.path.join(models_dir, "aio_transformers")
    lora_dir = os.path.join(models_dir, "lora_adapters")
    os.makedirs(qwen_dir, exist_ok=True)
    os.makedirs(aio_dir, exist_ok=True)
    os.makedirs(lora_dir, exist_ok=True)

    try:
        if not os.listdir(qwen_dir):
            snapshot_download(
                repo_id=QWEN_REPO,
                local_dir=qwen_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                ignore_patterns=["transformer/*"],
            )
        _log("Qwen-Image-Edit-2511 base snapshot ready (transformer/ from AIO repo)")
    except Exception as e:
        _log(f"WARN Qwen base: {e}")
        traceback.print_exc()

    aio_version_dir = os.path.join(aio_dir, AIO_VERSION)
    aio_target_dir = os.path.join(aio_version_dir, "transformer")
    os.makedirs(aio_target_dir, exist_ok=True)
    wrong_path = os.path.join(aio_version_dir, AIO_VERSION, "transformer")
    try:
        if os.path.exists(wrong_path) and os.listdir(wrong_path):
            for fn in os.listdir(wrong_path):
                src = os.path.join(wrong_path, fn)
                dst = os.path.join(aio_target_dir, fn)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
            try:
                os.rmdir(wrong_path)
                os.rmdir(os.path.join(aio_version_dir, AIO_VERSION))
            except OSError:
                pass
            _log("AIO: fixed nested v*/v*/transformer layout")
    except Exception as e:
        _log(f"WARN AIO layout fix: {e}")
        traceback.print_exc()

    try:
        if not os.listdir(aio_target_dir):
            snapshot_download(
                repo_id=AIO_REPO,
                local_dir=aio_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                allow_patterns=[f"{AIO_VERSION}/transformer/*"],
            )
        _log(f"AIO transformer ({AIO_VERSION}) ready")
    except Exception as e:
        _log(f"WARN AIO transformer: {e}")
        traceback.print_exc()

    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    for adapter_name, spec in QWEN_2511_LORA_ADAPTERS.items():
        repo = str(spec.get("repo") or "").strip()
        weights_raw = spec.get("weights")
        if not repo or not weights_raw:
            continue
        weights = (
            list(weights_raw)
            if isinstance(weights_raw, (list, tuple))
            else [str(weights_raw)]
        )
        adapter_sub = os.path.join(lora_dir, adapter_name.replace("/", "_"))
        os.makedirs(adapter_sub, exist_ok=True)
        for weight_file in weights:
            target = os.path.join(adapter_sub, str(weight_file))
            if os.path.exists(target):
                continue
            try:
                kw = {
                    "repo_id": repo,
                    "filename": str(weight_file),
                    "resume_download": True,
                }
                if hf_token:
                    kw["token"] = hf_token
                downloaded = hf_hub_download(**kw)
                shutil.copy2(downloaded, target)
                _log(f"LoRA baked: {adapter_name}/{weight_file}")
            except Exception as e:
                _log(f"WARN LoRA {adapter_name}/{weight_file}: {e}")
                traceback.print_exc()


def ensure_basic_utils(models_dir: str) -> None:
    cache = os.path.join(models_dir, "basic_utils")
    os.makedirs(cache, exist_ok=True)
    biref = os.path.join(cache, "birefnet")
    if not os.path.exists(biref) or not os.listdir(biref):
        try:
            snapshot_download(
                repo_id=BIREFNET_REPO,
                local_dir=biref,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            _log("BiRefNet (basic_utils) ready")
        except Exception as e:
            _log(f"WARN BiRefNet: {e}")
            traceback.print_exc()

    lama_dir = os.path.join(cache, "lama")
    os.makedirs(lama_dir, exist_ok=True)
    lama_pt = os.path.join(lama_dir, "lama.pt")
    if not os.path.exists(lama_pt):
        try:
            hf_hub_download(
                repo_id=LAMA_REPO,
                filename=LAMA_FILENAME,
                local_dir=lama_dir,
                resume_download=True,
            )
            src = os.path.join(lama_dir, LAMA_FILENAME)
            if LAMA_FILENAME != "lama.pt" and os.path.exists(src) and src != lama_pt:
                shutil.move(src, lama_pt)
            _log("LaMa TorchScript (lama.pt) ready")
        except Exception as e:
            _log(f"WARN LaMa: {e}")
            traceback.print_exc()


def ensure_aurasr(models_dir: str) -> None:
    out = os.path.join(models_dir, "aurasr_v2")
    os.makedirs(out, exist_ok=True)
    if os.listdir(out):
        _log("AuraSR-v2 already present")
        return
    try:
        snapshot_download(
            repo_id=AURASR_REPO,
            local_dir=out,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["model.ckpt"],
        )
        _log("AuraSR-v2 ready")
    except Exception as e:
        _log(f"WARN AuraSR: {e}")
        traceback.print_exc()


def ensure_swin2sr(models_dir: str) -> None:
    base = os.path.join(models_dir, "swin2sr")
    os.makedirs(base, exist_ok=True)
    tok = os.getenv("HF_TOKEN", "").strip()
    for name, repo_id in SWIN2SR_REPOS.items():
        local_dir = os.path.join(base, name)
        os.makedirs(local_dir, exist_ok=True)
        if os.listdir(local_dir):
            continue
        try:
            kwargs = {
                "repo_id": repo_id,
                "local_dir": local_dir,
                "local_dir_use_symlinks": False,
                "resume_download": True,
                "ignore_patterns": ["pytorch_model.bin"],
            }
            if tok:
                kwargs["token"] = tok
            snapshot_download(**kwargs)
            _log(f"Swin2SR {name} ready")
        except Exception as e:
            _log(f"WARN Swin2SR {name}: {e}")
            traceback.print_exc()


def ensure_face_enhancer(models_dir: str) -> None:
    d = os.path.join(models_dir, "realesrgan")
    os.makedirs(d, exist_ok=True)
    for filename, (_repo, url) in FACE_MODELS.items():
        path = os.path.join(d, filename)
        if os.path.exists(path):
            continue
        if not url:
            continue
        try:
            _log(f"Downloading face_enhancer weight {filename} …")
            urllib.request.urlretrieve(url, path)
            _log(f"{filename} ready")
        except Exception as e:
            _log(f"WARN {filename}: {e}")
            traceback.print_exc()




def ensure_ltx2_3_stage2_lora(ltx_models_root: str) -> None:
    """
    Bake the stage-2 distilled LoRA shard used after spatial latent upsample on **full**
    ``LTX2Pipeline`` / ``LTX2ConditionPipeline`` (text-to-video and FLF2V when
    ``use_latent_upsampler`` is true). Distilled I2V does **not** use this file.

    Writes: ``<ltx_models_root>/stage2_lora/<filename>`` (single ``hf_hub_download``, not a full repo snapshot).
    """
    stage2_dir = os.path.join(os.path.abspath(ltx_models_root), "stage2_lora")
    os.makedirs(stage2_dir, exist_ok=True)
    target = os.path.join(stage2_dir, LTX23_STAGE2_LORA_FILE)
    if os.path.isfile(target) and os.path.getsize(target) > 0:
        _log(f"LTX-2.3 stage-2 LoRA already present: {target}")
        return
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    try:
        kw = {
            "repo_id": LTX23_STAGE2_LORA_REPO,
            "filename": LTX23_STAGE2_LORA_FILE,
            "local_dir": stage2_dir,
            "resume_download": True,
        }
        if hf_token:
            kw["token"] = hf_token
        hf_hub_download(**kw)
        _log(f"LTX-2.3 stage-2 LoRA baked → {target}")
    except Exception as e:
        _log(f"WARN LTX-2.3 stage-2 LoRA: {e}")
        traceback.print_exc()


def ensure_glm_ocr(models_dir: str) -> None:
    out = os.path.join(models_dir, "glm_ocr")
    os.makedirs(out, exist_ok=True)
    if os.listdir(out):
        _log("GLM-OCR snapshot already present")
        return
    try:
        snapshot_download(
            repo_id=GLM_OCR_REPO,
            local_dir=out,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        _log("GLM-OCR (shared bake) ready")
    except Exception as e:
        _log(f"WARN GLM-OCR: {e}")
        traceback.print_exc()


def ensure_video_upscaler_stack(models_dir: str, qpw_root: str) -> None:
    rife_dir = os.path.join(models_dir, "rife")
    os.makedirs(rife_dir, exist_ok=True)
    if not os.listdir(rife_dir):
        try:
            snapshot_download(
                repo_id=RIFE_WEIGHTS_REPO,
                local_dir=rife_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            _log("RIFE weights ready")
        except Exception as e:
            _log(f"WARN RIFE weights: {e}")
            traceback.print_exc()

    rife_code = os.path.join(
        qpw_root, "worker_video_upscalers", "src", "rife"
    )
    if not os.path.isdir(rife_code) or not os.listdir(rife_code):
        try:
            parent = os.path.dirname(rife_code)
            os.makedirs(parent, exist_ok=True)
            if os.path.exists(rife_code):
                shutil.rmtree(rife_code, ignore_errors=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    RIFE_GIT_URL,
                    rife_code,
                ],
                check=True,
                cwd=parent,
            )
            _log("RIFE source cloned under worker_video_upscalers/src/rife")
        except Exception as e:
            _log(f"WARN RIFE git clone: {e}")
            traceback.print_exc()


def _count_py_under(dirpath: str) -> int:
    """Number of ``*.py`` files under dir (recursive), excluding ``__pycache__``."""
    if not os.path.isdir(dirpath):
        return 0
    n = 0
    for root, dirs, files in os.walk(dirpath):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            if fn.endswith(".py"):
                n += 1
    return n


def _copy_worker_src_tree(src_dir: str, dst_dir: str) -> None:
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"worker src not found: {src_dir}")
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir, ignore_errors=True)
    os.makedirs(dst_dir, exist_ok=True)

    copied_files = 0
    for root, dirs, files in os.walk(src_dir):
        # Skip bytecode + per-worker _archive/ (obsolete variants kept out of bakes and zips).
        dirs[:] = [d for d in dirs if d != "__pycache__" and d != "_archive"]
        rel_root = os.path.relpath(root, src_dir)
        rel_root = "" if rel_root == "." else rel_root
        out_root = os.path.join(dst_dir, rel_root) if rel_root else dst_dir
        os.makedirs(out_root, exist_ok=True)

        for fn in files:
            if fn in INFERENCE_EXCLUDE_FILENAMES:
                continue
            if fn.endswith((".pyc", ".pyo")):
                continue
            # Keep code-centric payload only (python modules + package metadata when present).
            if not fn.endswith(".py") and fn != "__init__.py":
                continue
            src_file = os.path.join(root, fn)
            dst_file = os.path.join(out_root, fn)
            shutil.copy2(src_file, dst_file)
            copied_files += 1

    if copied_files == 0:
        raise RuntimeError(f"no inference files copied from {src_dir}")


def sync_embedded_inference_sources(project_root: str, qpw_root: str) -> None:
    """
    Ensure primary + shared inference Python trees exist under:

        <project_root>/src/inference/<worker_dir>/src

    **Committed layout (slim Docker / normal prep):** each *_multi worker already ships
    ``src/inference/...`` in git; this step is a no-op unless something is missing.

    **Dev refresh:** optional monorepo sibling workers
    (``QPW_ROOT`` / ``quickpod_individual_worker/worker_*/src``) are used only to *fill
    gaps* or to overwrite when ``OMNI_REFRESH_EMBEDDED_INFERENCE=1``.
    """
    multi_dir = os.path.basename(os.path.abspath(project_root))
    primary = MULTI_PRIMARY_WORKER_DIR.get(multi_dir)
    if not primary:
        _log(f"WARN unknown multi worker dir for embedded sync: {multi_dir}")
        return

    # New organized layout: keep embedded inference code under src/inference.
    # Remove legacy src/embedded tree if it exists.
    legacy_embed_root = os.path.join(project_root, "src", "embedded")
    if os.path.isdir(legacy_embed_root):
        shutil.rmtree(legacy_embed_root, ignore_errors=True)

    embed_root = os.path.join(project_root, "src", "inference")
    os.makedirs(embed_root, exist_ok=True)

    worker_dirs = [primary]
    for w in SHARED_WORKER_DIRS:
        if w != primary:
            worker_dirs.append(w)

    force_refresh = os.environ.get("OMNI_REFRESH_EMBEDDED_INFERENCE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    if not force_refresh:
        missing = [
            wd
            for wd in worker_dirs
            if _count_py_under(os.path.join(embed_root, wd, "src")) == 0
        ]
        if not missing:
            _log(
                f"Embedded inference already present under {embed_root} "
                f"({len(worker_dirs)} workers); no sibling sync needed."
            )
            return

    copied = 0
    skipped_embedded = 0
    still_missing: list[str] = []

    for worker_dir in worker_dirs:
        dst_dir = os.path.join(embed_root, worker_dir, "src")
        has_embedded = _count_py_under(dst_dir) > 0
        if has_embedded and not force_refresh:
            skipped_embedded += 1
            continue

        src_dir = os.path.join(qpw_root, worker_dir, "src")
        if not os.path.isdir(src_dir):
            if not has_embedded:
                still_missing.append(worker_dir)
            continue
        try:
            _copy_worker_src_tree(src_dir, dst_dir)
            copied += 1
        except FileNotFoundError as e:
            _log(f"WARN embedded src sync failed for {worker_dir}: {e}")
        except Exception as e:
            _log(f"WARN embedded src sync failed for {worker_dir}: {e}")
            traceback.print_exc()

    if copied or skipped_embedded or still_missing:
        _log(
            "Embedded inference sync: "
            f"copied={copied}, already_in_tree={skipped_embedded}, "
            f"still_empty={len(still_missing)}"
        )
    if still_missing:
        _log(
            "WARN No ``.py`` under src/inference for: "
            + ", ".join(still_missing)
            + ". Commit those trees in the *_multi repo, set QPW_ROOT for a dev refresh, "
            "or set OMNI_REFRESH_EMBEDDED_INFERENCE=1 with siblings present. "
            "Runtime may still work if inference is loaded from R2 plugins."
        )


def run_omni_full_bake(project_root: str, *, include_qwen_stack: bool = False) -> None:
    """
    project_root: *_multi worker root (directory containing models/ and src/).

    include_qwen_stack: True only for worker_qwen_image_edit_2511_rapid_aio_multi prep.
    Other omni flavors rely on runtime download for Qwen.
    """
    project_root = os.path.abspath(project_root)
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    qpw = _resolve_qpw_root(project_root)

    _log(f"Starting full omni bake → {models_dir}")
    ensure_rockapaper_realesrgan(models_dir)
    if include_qwen_stack:
        ensure_qwen_2511_stack(models_dir)
    ensure_basic_utils(models_dir)
    ensure_aurasr(models_dir)
    ensure_swin2sr(models_dir)
    ensure_face_enhancer(models_dir)
    ensure_video_upscaler_stack(models_dir, qpw)
    ensure_glm_ocr(models_dir)
    sync_embedded_inference_sources(project_root, qpw)
    _log("Omni bake pass complete")
