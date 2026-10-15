#!/usr/bin/env python3
"""Download MiniMax H3 weights and RIFE flownet.pkl into ComfyUI model folders."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import zipfile
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


def _hf_download(repo: str, filename: str, dest: Path, token: Optional[str]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        logger.info("Already present: %s", dest)
        return
    from huggingface_hub import hf_hub_download

    hf_name = str(filename).replace("\\", "/")
    if "/" in hf_name:
        local_root = dest
        for _ in Path(hf_name).parts[:-1]:
            local_root = local_root.parent
        local_dir = str(local_root)
    else:
        local_dir = str(dest.parent)

    kwargs: Dict[str, Any] = {
        "repo_id": repo,
        "filename": hf_name,
        "local_dir": local_dir,
        "local_dir_use_symlinks": False,
    }
    if token:
        kwargs["token"] = token
    downloaded = hf_hub_download(**kwargs)
    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != dest.resolve():
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        shutil.move(str(downloaded_path), str(dest))
    logger.info("Downloaded %s -> %s", filename, dest)


def _download_rife(entry: Dict[str, Any], token: Optional[str]) -> None:
    filename = entry.get("filename") or "flownet.pkl"
    dests = [COMFYUI_DIR / d if not str(d).startswith("/") else Path(d) for d in entry.get("dests") or []]
    dests = [
        (MODELS_DIR / d if not str(d).startswith("custom_nodes") and not Path(d).is_absolute() else COMFYUI_DIR / d)
        if not Path(str(d)).is_absolute()
        else Path(d)
        for d in (entry.get("dests") or [])
    ]
    # dests in manifest are relative to models/ except custom_nodes paths
    resolved: List[Path] = []
    for d in entry.get("dests") or []:
        d = str(d).replace("\\", "/")
        if d.startswith("custom_nodes/"):
            resolved.append(COMFYUI_DIR / d)
        else:
            resolved.append(MODELS_DIR / d)
    if resolved and all(p.is_file() and p.stat().st_size > 0 for p in resolved):
        logger.info("RIFE %s already present", filename)
        return

    import requests

    zip_url = entry.get("zip_url")
    if not zip_url:
        raise RuntimeError("RIFE zip_url missing from models_manifest.json")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "rife.zip"
        logger.info("Downloading RIFE zip %s", zip_url)
        with requests.get(zip_url, headers=headers, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        extract_dir = Path(tmp) / "extracted"
        extract_dir.mkdir()
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        found = None
        for root, _dirs, files in os.walk(extract_dir):
            if filename in files:
                found = Path(root) / filename
                break
        if found is None:
            raise FileNotFoundError(f"{filename} not found in RIFE zip")
        for dest in resolved:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(found, dest)
            logger.info("Installed RIFE model -> %s", dest)


def ensure_weights(
    models_dir: Optional[Path] = None,
    token: Optional[str] = None,
    manifest_path: Optional[Path] = None,
) -> List[str]:
    """Download missing weights. Returns list of dest paths that are still missing."""
    # NFS (planned): MODELS_DIR=/data/models + SKIP_RUNTIME_MODEL_DOWNLOAD=1
    # See backend/quickpod_individual_worker/NFS_VOLUME.md
    if os.environ.get("SKIP_RUNTIME_MODEL_DOWNLOAD", "").strip() == "1":
        logger.warning("SKIP_RUNTIME_MODEL_DOWNLOAD=1 — skipping weight download")
        return []

    md = Path(models_dir or MODELS_DIR)
    md.mkdir(parents=True, exist_ok=True)
    token = token if token is not None else os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    manifest = load_manifest(manifest_path)

    missing: List[str] = []
    for item in manifest.get("files") or []:
        dest = md / str(item["dest"]).replace("\\", "/")
        try:
            _hf_download(item["hf_repo"], item["hf_path"], dest, token)
        except Exception:
            logger.exception("Failed to download %s", item.get("hf_path"))
        if not dest.is_file() or dest.stat().st_size <= 0:
            missing.append(str(dest))

    rife = manifest.get("rife")
    if rife:
        try:
            _download_rife(rife, token)
        except Exception:
            logger.exception("Failed to download RIFE flownet.pkl")
            for d in rife.get("dests") or []:
                missing.append(str(d))

    if missing:
        logger.error("Missing model files: %s", missing)
    else:
        logger.info("All MiniMax H3 weights are present")
    return missing


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    left = ensure_weights()
    if left:
        raise SystemExit(1)
