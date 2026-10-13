"""
Optional JSON manifest (HTTP URL e.g. public R2 object, or local file) to tune omni workers
without rebuilding images.

Env:
  OMNI_RUNTIME_MANIFEST_URL / R2_MODEL_CONFIG_URL — HTTPS GET, JSON object (checked first)
  OMNI_RUNTIME_MANIFEST_PATH — local JSON path (fallback when URL is unset or fails).
  OMNI_RUNTIME_MANIFEST_SYNC_PATH — default true; when URL fetch succeeds, overwrite PATH file
    so the local fallback copy stays fresh across restarts.
    Docker images may provide this fallback file (or leave it absent/empty). R2 URL is
    the dynamic source of truth; local path is only an offline safety net.
  OMNI_RUNTIME_MANIFEST_TIMEOUT — seconds for HTTP (default 60)

Schema (top-level keys, all optional):
  download_specs — list of download spec dicts (same shape as multi_model_runtime._download_task
    expects: model, repo_id, local_dir, multi_snapshot, partial_profiles, etc.). Each entry
    with \"model\" replaces the built-in spec for that id, or appends if the model is new.
  supported_models_add — extra model ids to advertise (lowercase recommended)
  supported_models_remove — model ids to drop from the default OMNI_SUPPORTED_MODELS list
  model_eta_seconds — object mapping model id -> int seconds (merged over defaults)

Weights still download via huggingface_hub from the repo_id / snapshots in each spec; only
the configuration moves to R2.

``supplement_hub_files`` entries may set ``target_relpath`` (relative to ``local_dir`` /
``supplement_dest_dir``) to place a downloaded file at a fixed path; when that path is under
``transformer/``, existing ``*.safetensors`` and ``*.index.json`` in that folder are removed
first (Hunyuan T2V + LightX2V overlay).

Dynamic inference code (optional): set OMNI_RUNTIME_PLUGINS_ENABLED=1 and add an
``inference_plugins`` list; see ``omni_runtime_plugins`` module docstring. That path
downloads and registers Python entrypoints — treat URLs and manifest host as trusted.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_CACHE: Optional[Dict[str, Any]] = None


def clear_runtime_manifest_cache() -> None:
    global _CACHE
    _CACHE = None


def _load_from_path(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            logger.info("omni runtime manifest loaded from path=%s", path)
            return data
    except Exception as e:
        logger.warning("omni runtime manifest path read failed path=%s err=%s", path, e)
    return None


def _load_from_url(url: str) -> Optional[Dict[str, Any]]:
    if not url:
        return None
    timeout = float(os.getenv("OMNI_RUNTIME_MANIFEST_TIMEOUT", "60"))
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "quickpod-omni-runtime-manifest/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        data = json.loads(raw.decode("utf-8"))
        if isinstance(data, dict):
            logger.info(
                "omni runtime manifest loaded from url ok keys=%s",
                list(data.keys())[:24],
            )
            return data
    except urllib.error.HTTPError as e:
        logger.warning(
            "omni runtime manifest HTTP error url=%s code=%s", url, getattr(e, "code", e)
        )
    except Exception as e:
        logger.warning("omni runtime manifest fetch failed url=%s err=%s", url, e)
    return None


def _sync_manifest_to_path(path: str, data: Dict[str, Any]) -> None:
    """
    Best-effort local sync of remote manifest for offline fallback.
    Controlled by OMNI_RUNTIME_MANIFEST_SYNC_PATH (default: true).
    """
    if not path:
        return
    if os.getenv("OMNI_RUNTIME_MANIFEST_SYNC_PATH", "true").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        logger.info("omni runtime manifest synced to path=%s", path)
    except Exception as e:
        logger.warning("omni runtime manifest sync failed path=%s err=%s", path, e)


def load_runtime_manifest() -> Dict[str, Any]:
    """
    Load once per process; returns {} on missing config or failure (fail-open).

    Priority: URL (dynamic R2) > PATH (baked in Docker image).
    If the URL is set and reachable, it wins. If the URL fails or is unset,
    fall back to the local path. This lets R2 be the single source of truth
    while the baked file acts as an offline safety net.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    url = (
        os.getenv("OMNI_RUNTIME_MANIFEST_URL") or os.getenv("R2_MODEL_CONFIG_URL") or ""
    ).strip()
    path = (os.getenv("OMNI_RUNTIME_MANIFEST_PATH") or "").strip()
    if not url:
        logger.warning(
            "omni runtime manifest URL is not configured; using local manifest path only (can become stale)"
        )

    if url:
        result = _load_from_url(url)
        if result is not None:
            _sync_manifest_to_path(path, result)
            _CACHE = result
            return _CACHE
        logger.info("omni manifest URL failed/empty, trying local path fallback")

    if path:
        result = _load_from_path(path)
        if result is not None:
            _CACHE = result
            return _CACHE

    _CACHE = {}
    return _CACHE
