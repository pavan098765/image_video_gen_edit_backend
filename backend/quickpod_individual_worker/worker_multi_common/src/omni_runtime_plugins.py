"""
Opt-in dynamic inference plugins from the same JSON manifest as HF download overrides.

Requires OMNI_RUNTIME_PLUGINS_ENABLED=1 (or true/yes). Without it, manifest
``inference_plugins`` are skipped at startup but still installed when a spec's
``inference_archive`` is processed by the downloader loop.

Security: this executes Python from URLs you control (e.g. R2). Use HTTPS; treat
the manifest host as part of your trust boundary.

Manifest key ``inference_plugins``: list of objects:
  model            — registry key (lowercase recommended)
  archive_url      — https URL to .zip or .tar.gz
  kind             — optional: \"zip\" | \"gztar\" (inferred from URL if omitted)
  entry_relpath    — path inside the archive to the inference .py file
  callable_name    — function name called for each job (default: run_inference)
  prewarm_fn       — optional attribute name for prewarm hook
  unload_fn        — optional attribute name for unload hook
  no_prompt        — if true, fn signature is (job_id, params) like aurasr (default false)

Multiple rows may share the same ``archive_url`` (e.g. LTX-2.3: one ``ltx2_3_multi.zip`` with three
``model`` keys and three ``callable_name`` values). Each row still extracts to its own
``runtime_plugins/<model>/`` subdirectory.

Archives are extracted under OMNI_RUNTIME_PLUGINS_DIR (default /app/runtime_plugins/<model>).
Re-download is skipped when .quickpod_plugin.json matches archive_url + entry_relpath and the
entry .py already exists.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PLUGINS_DIR = os.getenv("OMNI_RUNTIME_PLUGINS_DIR", "/app/runtime_plugins")
_DEFAULT_TIMEOUT = float(os.getenv("OMNI_RUNTIME_MANIFEST_TIMEOUT", "120"))


def _plugins_globally_enabled() -> bool:
    """Manifest startup-scan is opt-in; per-spec inference_archive always runs."""
    return os.getenv("OMNI_RUNTIME_PLUGINS_ENABLED", "").lower() in ("1", "true", "yes")


def _infer_archive_kind(url: str, kind: Optional[str]) -> str:
    if kind and str(kind).strip():
        k = str(kind).strip().lower()
        return "zip" if k == "zip" else "gztar"
    return "zip" if url.lower().endswith(".zip") else "gztar"


def _safe_dirname(model: str) -> str:
    s = re.sub(r"[^a-z0-9_.-]+", "_", model.strip().lower())
    return s[:120] or "plugin"


def _http_download(url: str, timeout: float) -> str:
    fd, path = tempfile.mkstemp(suffix=".archive")
    os.close(fd)
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "quickpod-omni-runtime-plugin/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            with open(path, "wb") as out:
                shutil.copyfileobj(resp, out)
        return path
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        raise


def install_inference_plugin(
    plugin_spec: Dict[str, Any],
    model: str,
    plugins_root: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Optional[str]:
    """
    Download and extract an inference plugin archive.

    Returns the absolute path to the entry .py file, or None on failure.
    Idempotent: skips extraction when cache marker matches url+entry_relpath.
    """
    url = str(plugin_spec.get("archive_url") or "").strip()
    entry_rel = str(plugin_spec.get("entry_relpath") or "").strip().replace("\\", "/")
    kind = plugin_spec.get("kind")

    if not url or not entry_rel:
        logger.warning("install_inference_plugin: missing archive_url or entry_relpath model=%s", model)
        return None

    root = (plugins_root or _PLUGINS_DIR).rstrip("/")
    tm = timeout if timeout is not None else _DEFAULT_TIMEOUT
    sub = _safe_dirname(model)
    dest = os.path.join(root, sub)
    py_path = os.path.join(dest, entry_rel)
    marker = os.path.join(dest, ".quickpod_plugin.json")
    meta = {"archive_url": url, "entry_relpath": entry_rel}

    if os.path.isfile(marker) and os.path.isfile(py_path):
        try:
            with open(marker, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("archive_url") == url and cached.get("entry_relpath") == entry_rel:
                logger.info("install_inference_plugin: cache hit model=%s", model)
                return py_path
        except Exception:
            pass

    if os.path.isdir(dest):
        shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)

    archive = _http_download(url, tm)
    try:
        shutil.unpack_archive(archive, dest, _infer_archive_kind(url, kind))
    finally:
        try:
            os.remove(archive)
        except OSError:
            pass

    if not os.path.isfile(py_path):
        logger.error(
            "install_inference_plugin: entry missing after extract model=%s path=%s",
            model,
            py_path,
        )
        shutil.rmtree(dest, ignore_errors=True)
        return None

    try:
        with open(marker, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except OSError as e:
        logger.warning("install_inference_plugin: marker write failed: %s", e)

    logger.info("install_inference_plugin: installed model=%s py=%s", model, py_path)
    return py_path


def registry_entry_from_plugin_spec(
    plugin_spec: Dict[str, Any], py_path: str
) -> Dict[str, Any]:
    """Build a model_registry entry dict from a plugin spec and the installed .py path."""
    from image_registry import registry_entry_from_py  # local import avoids circular

    run_name = str(plugin_spec.get("callable_name") or "run_inference").strip() or "run_inference"
    pw = (plugin_spec.get("prewarm_fn") or "").strip() or None
    ul = (plugin_spec.get("unload_fn") or "").strip() or None
    no_prompt = bool(plugin_spec.get("no_prompt", False))
    return registry_entry_from_py(py_path, run_name, prewarm_name=pw, unload_name=ul, no_prompt=no_prompt)


# ---------------------------------------------------------------------------
# Legacy: manifest-scan at startup (only when OMNI_RUNTIME_PLUGINS_ENABLED=1).
# Downloader-loop driven hot registration handles the primary path.
# ---------------------------------------------------------------------------


def build_startup_plugin_entries() -> Dict[str, Any]:
    """
    If OMNI_RUNTIME_PLUGINS_ENABLED=1, install all inference_plugins from the manifest
    at startup (before the downloader loop runs) and return registry entries.
    Only useful for models that have NO HF weights (code-only plugins).
    """
    if not _plugins_globally_enabled():
        return {}
    try:
        from omni_runtime_manifest import load_runtime_manifest

        manifest = load_runtime_manifest()
    except Exception:
        return {}
    raw: List[Any] = manifest.get("inference_plugins") or []
    if not isinstance(raw, list) or not raw:
        return {}
    out: Dict[str, Any] = {}
    for spec in raw:
        if not isinstance(spec, dict):
            continue
        model = str(spec.get("model") or "").strip().lower()
        if not model:
            continue
        try:
            py_path = install_inference_plugin(spec, model)
            if py_path:
                out[model] = registry_entry_from_plugin_spec(spec, py_path)
        except Exception as e:
            logger.error("build_startup_plugin_entries model=%s err=%s", model, e)
    return out


def merge_manifest_inference_plugins(base: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay startup plugin entries onto a static registry (used at import time)."""
    extra = build_startup_plugin_entries()
    if not extra:
        return base
    merged = dict(base)
    merged.update(extra)
    return merged
