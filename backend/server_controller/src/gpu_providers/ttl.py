"""Idle TTL: Quickpod always 15 min; other providers close after the job unless personal leftover exists."""

from __future__ import annotations

import os
from typing import Any, Optional


def provider_of_worker(worker: Any) -> str:
    info = getattr(worker, "info", None) or {}
    if isinstance(info, dict):
        raw = str(info.get("gpu_provider") or "").strip().lower()
        if raw:
            return raw
    return "quickpod"


def worker_model_keys(worker: Any) -> list:
    keys = []
    model = getattr(worker, "model", None)
    if model:
        keys.append(str(model).strip())
    info = getattr(worker, "info", None) or {}
    if isinstance(info, dict):
        for m in info.get("supported_models") or []:
            if m:
                keys.append(str(m).strip())
    seen = set()
    out = []
    for k in keys:
        lk = k.lower()
        if lk and lk not in seen:
            seen.add(lk)
            out.append(k)
    return out


def has_personal_leftover(worker: Any) -> bool:
    try:
        from db import SessionLocal
        import personal_queue
    except Exception:
        return False
    models = worker_model_keys(worker)
    db = SessionLocal()
    try:
        return personal_queue.has_leftover_work(db, models)
    except Exception:
        return False
    finally:
        db.close()


def idle_timeout_seconds_for_worker(worker: Any, default_idle: int) -> int:
    """
    Quickpod: 15 minute leftover window (billing minimum).
    RunPod / Vast / Salad: destroy as soon as idle unless personal dashboard
    has matching work, in which case they also get 15 minutes.
    """
    provider = provider_of_worker(worker)
    qp_ttl = int(os.getenv("QUICKPOD_IDLE_TIMEOUT_SECONDS", "900"))
    ext_ttl = int(os.getenv("EXTERNAL_POD_IDLE_TIMEOUT_SECONDS", "900"))
    if provider in ("", "quickpod"):
        return qp_ttl if qp_ttl > 0 else default_idle
    if has_personal_leftover(worker):
        return ext_ttl
    return 0


def should_keep_external_after_job(worker: Any) -> bool:
    if provider_of_worker(worker) in ("", "quickpod"):
        return True
    return has_personal_leftover(worker)
