"""External GPU providers used after Quickpod search is exhausted."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from gpu_providers.base import CreatedPod, WorkerSpec
from gpu_providers import runpod, salad, vast

logger = logging.getLogger("gpu_providers")

DEFAULT_ORDER = ("runpod", "vast", "salad")


def provider_order() -> List[str]:
    raw = (os.getenv("GPU_PROVIDER_ORDER") or os.getenv("EXTERNAL_GPU_PROVIDERS") or "").strip()
    if not raw:
        names = list(DEFAULT_ORDER)
    else:
        names = [p.strip().lower() for p in raw.split(",") if p.strip()]
    disabled = {
        p.strip().lower()
        for p in (os.getenv("GPU_PROVIDERS_DISABLED") or "").split(",")
        if p.strip()
    }
    out = []
    for name in names:
        if name in ("off", "none", "quickpod"):
            continue
        if name in disabled:
            continue
        if name not in ("runpod", "vast", "salad"):
            continue
        if name not in out:
            out.append(name)
    return out


def is_provider_configured(name: str) -> bool:
    if name == "runpod":
        return runpod.is_configured()
    if name == "vast":
        return vast.is_configured()
    if name == "salad":
        return salad.is_configured()
    return False


def any_external_configured() -> bool:
    return any(is_provider_configured(n) for n in provider_order())


def create_on_providers(spec: WorkerSpec) -> Optional[CreatedPod]:
    for name in provider_order():
        if not is_provider_configured(name):
            logger.info("Skipping GPU provider %s (API key/config missing)", name)
            continue
        logger.info("Trying GPU provider %s for model=%s gpus=%s", name, spec.model, spec.allowed_gpus)
        try:
            if name == "runpod":
                created = runpod.create(spec)
            elif name == "vast":
                created = vast.create(spec)
            else:
                created = salad.create(spec)
        except Exception as e:
            logger.warning("GPU provider %s create error: %s", name, e)
            continue
        if created and created.pod_id:
            logger.info(
                "Created %s pod %s url=%s gpu=%s",
                created.provider,
                created.pod_id,
                created.public_url or "(pending)",
                created.gpu_type,
            )
            return created
        logger.info("GPU provider %s had no capacity for model=%s", name, spec.model)
    return None


def destroy_external(provider: str, pod_id: str, meta: Optional[Dict[str, Any]] = None) -> bool:
    name = (provider or "").strip().lower()
    meta = meta or {}
    try:
        if name == "runpod":
            return runpod.destroy(pod_id, meta)
        if name == "vast":
            return vast.destroy(pod_id, meta)
        if name == "salad":
            return salad.destroy(pod_id, meta)
    except Exception as e:
        logger.warning("Destroy %s %s failed: %s", name, pod_id, e)
        return False
    logger.warning("Unknown GPU provider %s for destroy of %s", provider, pod_id)
    return False


def refresh_public_url(provider: str, meta: Optional[Dict[str, Any]] = None) -> str:
    name = (provider or "").strip().lower()
    meta = meta or {}
    if name == "vast":
        try:
            return vast.refresh_public_url(meta, int(meta.get("http_port") or 8000))
        except Exception:
            return ""
    return str(meta.get("public_url") or "")
