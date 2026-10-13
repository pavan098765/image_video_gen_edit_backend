#!/usr/bin/env python3
"""
Bake-time prep for worker_ltx2_3_multi.

Installs SDNQ from GitHub, then bakes:

1. ``rockapaper/LTX-2.3-Distilled-Diffusers_tenc_uint4_sdnq_r64_s16`` (primary distilled checkpoint).
2. ``dg845/LTX-2.3-Spatial-Upsampler-Diffusers`` (stage-2 spatial latent upsampler for two-stage
   8-step stage-1 + 3-step stage-2 rendering).
3. CivitAI LoRAs declared in ``omni_manifest.json`` ``ltx2_3_civitai_loras`` are pre-warmed at
   bake time (entries with ``baked: true``); remaining entries are fetched at LTX worker
   startup (``app.py``). Requires ``CIVITAI_API_KEY`` env; if missing the pre-warm is skipped
   with a warning and LoRAs will be fetched at worker startup / first job instead
   (``ltx2_3_inference._resolve_civitai_lora_file``).

Finally runs the shared omni bake (``run_omni_full_bake``) so every ``OMNI_SHARED_BAKED_MODELS``
utility (basic_utils / aurasrv2 / swin2sr / video_upscaler / face_enhancer / glm_ocr) is baked
into this image as well — the pod becomes omni-capable for utilities without runtime download.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import traceback
from typing import Any, Dict

from huggingface_hub import snapshot_download

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEPS_DIR = os.path.join(PROJECT_ROOT, "deps")
SDNQ_GIT_URL = os.environ.get("SDNQ_GIT_URL", "https://github.com/Disty0/sdnq.git")
SDNQ_CLONE_DIR = os.environ.get("SDNQ_CLONE_DIR", os.path.join(DEPS_DIR, "sdnq"))

_MULTI_COMMON_SRC = os.path.join(os.path.dirname(PROJECT_ROOT), "worker_multi_common", "src")
if os.path.isdir(_MULTI_COMMON_SRC) and _MULTI_COMMON_SRC not in sys.path:
    sys.path.insert(0, _MULTI_COMMON_SRC)
# inference module (embedded multi layout); needed for civitai helper import
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
_LTX_INF_SRC = os.path.join(SCRIPT_DIR, "inference", "worker_ltx2_3", "src")
if os.path.isdir(_LTX_INF_SRC) and _LTX_INF_SRC not in sys.path:
    sys.path.insert(0, _LTX_INF_SRC)

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

MODELS_ROOT = os.path.join(PROJECT_ROOT, "models", "ltx2_3")
DISTILLED_DIR = os.path.join(MODELS_ROOT, "distilled")
SPATIAL_DIR = os.path.join(MODELS_ROOT, "spatial_upsampler")
LORA_CACHE_DIR = os.environ.get(
    "LTX2_3_LORA_CACHE", os.path.join(MODELS_ROOT, "lora_cache")
).strip()

DISTILLED_REPO = os.environ.get(
    "LTX2_3_DISTILLED_REPO", "rockapaper/LTX-2.3-Distilled-Diffusers_tenc_uint4_sdnq_r64_s16"
).strip()
SPATIAL_REPO = os.environ.get(
    "LTX2_3_SPATIAL_UPSAMPLER_REPO", "dg845/LTX-2.3-Spatial-Upsampler-Diffusers"
).strip()
CIVITAI_LORA_URL = os.environ.get(
    "CIVITAI_LORA_URL",
    "https://civitai.com/models/1811313/dr34ml4y-all-in-one-nsfw-wanltx2",
).strip()


def _ensure_sdnq() -> None:
    os.makedirs(DEPS_DIR, exist_ok=True)
    marker = os.path.join(SDNQ_CLONE_DIR, "pyproject.toml")
    if not os.path.isfile(marker):
        if os.path.isdir(SDNQ_CLONE_DIR):
            shutil.rmtree(SDNQ_CLONE_DIR, ignore_errors=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", SDNQ_GIT_URL, SDNQ_CLONE_DIR],
            check=True,
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", SDNQ_CLONE_DIR],
        check=True,
    )
    print(f"[ok] SDNQ ready at {SDNQ_CLONE_DIR}")


def _snapshot_if_empty(repo_id: str, local_dir: str) -> None:
    os.makedirs(local_dir, exist_ok=True)
    if os.listdir(local_dir):
        print(f"[skip] non-empty: {local_dir}")
        return
    print(f"[dl] snapshot_download {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"[ok] {repo_id} -> {local_dir}")


def _ensure_baked_civitai_loras() -> None:
    """
    Pre-warm CivitAI LoRA cache for every manifest entry with ``baked: true``.

    Soft-fails on missing CIVITAI_API_KEY so Docker builds without a secret still
    produce a usable image (runtime fetch still works on first job + startup warmup).
    """
    os.environ.setdefault("LTX2_3_LORA_CACHE", LORA_CACHE_DIR)
    os.makedirs(LORA_CACHE_DIR, exist_ok=True)

    try:
        from omni_config import get_ltx2_3_civitai_loras  # type: ignore
        loras = get_ltx2_3_civitai_loras()
    except Exception as e:
        print(f"[warn] cannot read manifest LoRA catalog ({e!r}); skipping bake-time pre-warm")
        loras = []

    baked = [x for x in loras if x.get("baked")]
    if not baked:
        legacy_url = os.environ.get("CIVITAI_LORA_URL", "").strip()
        if legacy_url:
            baked = [{
                "name": "legacy_civitai_lora",
                "url": legacy_url,
                "strength": 1.0,
                "baked": True,
                "version_id": None,
            }]
        else:
            print("[ok] no baked CivitAI LoRAs declared in manifest")
            return

    api_key = (os.environ.get("CIVITAI_API_KEY") or "").strip()
    if not api_key:
        print(
            f"[warn] CIVITAI_API_KEY not set; skipping bake-time pre-warm of "
            f"{len(baked)} LoRA(s); they will be fetched at first job"
        )
        return

    try:
        from ltx2_3_inference import _resolve_civitai_lora_file  # type: ignore
    except Exception as e:
        print(
            f"[warn] cannot import _resolve_civitai_lora_file ({e!r}); "
            "skipping CivitAI pre-warm"
        )
        return

    for entry in baked:
        url = entry["url"]
        nm = entry.get("name") or url
        params: Dict[str, Any] = {
            "civitai_lora_model_url": url,
            "lora_cache_dir": LORA_CACHE_DIR,
        }
        vid = entry.get("version_id")
        if vid is not None:
            params["civitai_version_id"] = int(vid)
        try:
            local_path, version_id = _resolve_civitai_lora_file(params)
            print(f"[ok] CivitAI LoRA baked: {nm} v{version_id} -> {local_path}")
        except Exception as e:
            print(f"[warn] CivitAI LoRA pre-warm failed ({nm}: {e!r}); will retry at runtime")


def split_large_safetensors(base_dir: str, max_size: int = 700 * 1024 * 1024) -> None:
    """Shard large ``.safetensors`` so ``generate_dockerfile.copy_model_layers`` can emit per-part COPY lines."""
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(".safetensors") and ".part-" not in fn:
                full_path = os.path.join(root, fn)
                if os.path.getsize(full_path) <= max_size:
                    continue
                subprocess.run(
                    ["split", "-b", "700M", full_path, f"{full_path}.part-"],
                    check=True,
                )
                os.remove(full_path)
                print(f"[ok] Split: {full_path}")


def _run_shared_omni_bake() -> None:
    try:
        from omni_prep_bake import run_omni_full_bake  # type: ignore
    except Exception as e:
        print(f"[warn] omni_prep_bake import failed ({e!r}); skipping shared bake")
        return
    try:
        run_omni_full_bake(PROJECT_ROOT, include_qwen_stack=False)
        print("[ok] shared omni bake complete")
    except Exception as e:
        print(f"[warn] run_omni_full_bake failed ({e!r}); continuing")


def main() -> None:
    try:
        _ensure_sdnq()
        os.makedirs(MODELS_ROOT, exist_ok=True)
        _snapshot_if_empty(DISTILLED_REPO, DISTILLED_DIR)
        _snapshot_if_empty(SPATIAL_REPO, SPATIAL_DIR)
        _ensure_baked_civitai_loras()
        _run_shared_omni_bake()
        split_large_safetensors(os.path.join(PROJECT_ROOT, "models"))
        subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "generate_dockerfile.py")],
            cwd=SCRIPT_DIR,
            check=False,
        )
        print("[ok] LTX-2.3 prep_model complete")
    except Exception as e:
        print(f"[fail] prep_model failed: {e}")
        traceback.print_exc()
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
