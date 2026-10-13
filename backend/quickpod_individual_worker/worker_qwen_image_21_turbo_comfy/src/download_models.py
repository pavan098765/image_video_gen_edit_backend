#!/usr/bin/env python3
"""Download LTX-2.5 INT8 weights into ComfyUI model folders."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("download_models")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = Path(os.environ.get("MODELS_MANIFEST_PATH", str(PROJECT_ROOT / "models_manifest.json")))
COMFYUI_DIR = Path(os.environ.get("COMFYUI_DIR", "/app/comfyui"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(COMFYUI_DIR / "models")))


def load_manifest(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or MANIFEST_PATH
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _hf_download(repo: str, filename: str, dest: Path, token: Optional[str], revision: Optional[str]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        logger.info("Already present: %s", dest)
        return
    from huggingface_hub import hf_hub_download

    kwargs: Dict[str, Any] = {
        "repo_id": repo,
        "filename": filename,
        "local_dir": str(dest.parent),
        "local_dir_use_symlinks": False,
    }
    if token:
        kwargs["token"] = token
    if revision:
        kwargs["revision"] = revision
    downloaded = hf_hub_download(**kwargs)
    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != dest.resolve():
        if dest.exists():
            dest.unlink()
        shutil.move(str(downloaded_path), str(dest))
    logger.info("Downloaded %s -> %s", filename, dest)


def ensure_weights(
    models_dir: Optional[Path] = None,
    token: Optional[str] = None,
    manifest_path: Optional[Path] = None,
) -> List[str]:
    """Download missing weights. Returns dest paths that are still missing."""
    if os.environ.get("SKIP_RUNTIME_MODEL_DOWNLOAD", "").strip() == "1":
        logger.warning("SKIP_RUNTIME_MODEL_DOWNLOAD=1 — skipping weight download")
        return []

    md = Path(models_dir or MODELS_DIR)
    md.mkdir(parents=True, exist_ok=True)
    token = token if token is not None else os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    manifest = load_manifest(manifest_path)
    revision = manifest.get("hf_revision")

    missing: List[str] = []
    for item in manifest.get("files") or []:
        dest = md / str(item["dest"]).replace("\\", "/")
        try:
            _hf_download(
                item["hf_repo"],
                item["hf_path"],
                dest,
                token,
                item.get("hf_revision") or revision,
            )
        except Exception:
            logger.exception("Failed to download %s", item.get("hf_path"))
        if not dest.is_file() or dest.stat().st_size <= 0:
            missing.append(str(dest))

    if missing:
        logger.error("Missing model files: %s", missing)
    else:
        logger.info("All LTX-2.5 INT8 weights are present")
    return missing


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    left = ensure_weights()
    if left:
        raise SystemExit(1)
