"""Studio API — new /studio/* routes. Do not use POST /submit."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from credit_service import lock_credits_for_job, refund_credits_for_job
from db import Job, SessionLocal, Worker, utc_naive_now
from studio_models import (
    FOOTAGE_REQUIRED,
    PIPELINES,
    SceneAction,
    StudioJobCreate,
    StudioUploadRequest,
)
from studio_storyboard import all_approved, empty_storyboard, upsert_scene
from vultr_pool import pool as vultr_pool

studio_router = APIRouter(tags=["studio"])

REVIEW_TTL_HOURS = int(os.getenv("STUDIO_REVIEW_TTL_HOURS", "48"))
_pending_pull: list[str] = []


def _db() -> Session:
    return SessionLocal()


def _user_from_auth(auth: dict) -> str:
    return auth.get("user_id") or auth.get("firebase_uid")


async def _client_auth(request: Request, authorization: Optional[str] = Header(None)) -> dict:
    from routes import require_firebase_auth

    return await require_firebase_auth(request, authorization)


def _worker_auth(authorization: Optional[str] = Header(None)) -> None:
    secret = os.getenv("WORKER_SECRET") or os.getenv("WORKER_HMAC_SECRET") or os.getenv("SHARED_SECRET")
    if not secret:
        return
    token = (authorization or "").replace("Bearer ", "")
    if token != secret:
        raise HTTPException(401, "worker auth failed")


def _public_status(job: Job) -> str:
    raw = (job.status or "").lower()
    mapping = {
        "pending": "queued",
        "queued": "queued",
        "dispatching": "provisioning",
        "running": "running",
        "reviewing": "review",
        "completed": "done",
        "failed": "failed",
        "timeout": "failed",
        "cancelled": "failed",
    }
    params = job.params or {}
    stage = (params.get("studio_stage") or "").lower()
    if stage in {"stitching", "uploading"}:
        return stage
    return mapping.get(raw, raw or "queued")


def _sanitize_job(job: Job) -> dict[str, Any]:
    params = dict(job.params or {})
    prefs = dict(params.get("prefs") or {})
    return {
        "job_id": job.job_id,
        "pipeline": params.get("pipeline"),
        "review_mode": params.get("review_mode", True),
        "prefs": prefs,
        "asset_urls": [
            {"r2_key": k, "kind": "upload"} for k in (params.get("asset_keys") or [])
        ],
        "r2": {"prefix": f"studio/{job.user_id}/{job.job_id}"},
        "resume": bool(params.get("resume")),
        "continue_after_review": bool(params.get("continue_after_review")),
        "regenerate_scene_id": params.get("regenerate_scene_id"),
        "compose_strategy": params.get("compose_strategy"),
        "scene_runtimes": params.get("scene_runtimes"),
    }


def route_auto_pipeline(prefs: dict, asset_keys: list[str]) -> str:
    talking = bool(prefs.get("talking") or prefs.get("talking_clip"))
    long_video = bool(prefs.get("long_video"))
    native_short = bool(prefs.get("native_short"))
    if native_short or prefs.get("mode") == "shorts":
        return "shorts"
    if asset_keys and talking:
        return "talking-head"
    if asset_keys and long_video:
        return "clip-factory"
    if prefs.get("real_footage_only") or prefs.get("mode") == "from_evidence":
        return "documentary-montage"
    if prefs.get("mode") == "from_footage" and asset_keys:
        return "hybrid"
    return "animated-explainer"


@studio_router.post("/studio/jobs")
async def create_studio_job(body: StudioJobCreate, auth: dict = Depends(_client_auth)):
    user_id = _user_from_auth(auth)
    pipeline = body.pipeline
    if pipeline == "auto":
        pipeline = route_auto_pipeline(body.prefs.model_dump(), body.asset_keys)
    if pipeline == "cinematic":
        raise HTTPException(400, "cinematic is not in the Studio engine (no film T2V / avatar path)")
    if pipeline in FOOTAGE_REQUIRED and not body.asset_keys:
        raise HTTPException(400, f"{pipeline} requires footage uploads")
    if pipeline not in FOOTAGE_REQUIRED and not (body.prefs.topic or "").strip():
        if pipeline not in {"clip-factory", "talking-head"}:
            raise HTTPException(400, "topic required")

    job_id = str(uuid.uuid4())
    params = {
        "job_kind": "studio",
        "pipeline": pipeline,
        "review_mode": body.review_mode,
        "prefs": body.prefs.model_dump(),
        "asset_keys": body.asset_keys,
        "studio_storyboard": empty_storyboard(job_id),
        "studio_stage": "queued",
        "compose_strategy": "single_runtime",
    }
    db = _db()
    try:
        job = Job(
            job_id=job_id,
            model="studio",
            prompt=(body.prefs.topic or pipeline)[:500],
            params=params,
            status="queued",
            user_id=user_id,
            fcm_token=body.fcm_token,
        )
        db.add(job)
        ok, msg, locked = lock_credits_for_job(db, user_id, job_id, "studio", params=params)
        if not ok:
            db.rollback()
            raise HTTPException(402, msg)
        job.credits_locked = locked
        db.commit()
    finally:
        db.close()

    placement = vultr_pool.place_job(job_id, pipeline)
    _pending_pull.append(job_id)
    if placement.get("action") == "provisioning":
        db = _db()
        try:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job:
                job.status = "dispatching"
                job.params = {**(job.params or {}), "studio_stage": "provisioning"}
                db.commit()
        finally:
            db.close()
    return {"job_id": job_id, "status": "queued", "placement": placement}


@studio_router.get("/studio/jobs/{job_id}")
async def get_studio_job(job_id: str, auth: dict = Depends(_client_auth)):
    user_id = _user_from_auth(auth)
    db = _db()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job or job.model != "studio":
            raise HTTPException(404, "not found")
        if job.user_id != user_id:
            raise HTTPException(403, "forbidden")
        expire_review_if_needed(db, job)
        db.commit()
        return {
            "job_id": job.job_id,
            "status": _public_status(job),
            "stage": (job.params or {}).get("studio_stage"),
            "pipeline": (job.params or {}).get("pipeline"),
            "compose_strategy": (job.params or {}).get("compose_strategy"),
            "scene_runtimes": (job.params or {}).get("scene_runtimes"),
            "error": job.error,
        }
    finally:
        db.close()


@studio_router.get("/studio/jobs/{job_id}/storyboard")
async def get_storyboard(job_id: str, auth: dict = Depends(_client_auth)):
    user_id = _user_from_auth(auth)
    db = _db()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job or job.user_id != user_id:
            raise HTTPException(404, "not found")
        expire_review_if_needed(db, job)
        db.commit()
        board = (job.params or {}).get("studio_storyboard") or empty_storyboard(job_id)
        from routes import create_presigned_get_url

        for scene in board.get("scenes") or []:
            key = scene.get("poster_key") or scene.get("poster")
            pkey = scene.get("preview_key") or scene.get("preview")
            if key and _looks_like_r2_key(key):
                scene["poster_url"] = create_presigned_get_url(key)
            if pkey and _looks_like_r2_key(pkey):
                scene["preview_url"] = create_presigned_get_url(pkey)
        return board
    finally:
        db.close()


@studio_router.post("/studio/jobs/{job_id}/scenes/{scene_id}")
async def scene_action(job_id: str, scene_id: str, body: SceneAction, auth: dict = Depends(_client_auth)):
    user_id = _user_from_auth(auth)
    db = _db()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job or job.user_id != user_id:
            raise HTTPException(404, "not found")
        if job.status != "reviewing":
            raise HTTPException(409, "job is not in review")
        params = dict(job.params or {})
        board = params.get("studio_storyboard") or empty_storyboard(job_id)
        if body.action == "approve":
            upsert_scene(board, scene_id, approved=True)
        elif body.action == "patch_copy":
            upsert_scene(board, scene_id, script_line=body.script_line, approved=False)
        elif body.action == "replace":
            upsert_scene(
                board,
                scene_id,
                replace_r2_key=body.replace_r2_key,
                explore_job_id=body.explore_job_id,
                approved=True,
            )
        elif body.action == "regenerate":
            upsert_scene(board, scene_id, approved=False, regenerate=True)
            params["regenerate_scene_id"] = scene_id
            params["resume"] = True
            # Parent stays REVIEWING while one scene is regenerated.
            _pending_pull.append(job_id)
            job.status = "reviewing"
        params["studio_storyboard"] = board
        job.params = params
        db.commit()
        return {"ok": True, "storyboard": board}
    finally:
        db.close()


@studio_router.post("/studio/jobs/{job_id}/continue")
async def continue_job(job_id: str, request: Request, auth: dict = Depends(_client_auth)):
    user_id = _user_from_auth(auth)
    db = _db()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job or job.user_id != user_id:
            raise HTTPException(404, "not found")
        board = (job.params or {}).get("studio_storyboard") or {}
        if not all_approved(board):
            raise HTTPException(409, "scenes pending approval")
        params = dict(job.params or {})
        params["continue_after_review"] = True
        params["studio_stage"] = "stitching"
        try:
            raw = await request.json()
            if isinstance(raw, dict) and raw.get("cuts"):
                params["cuts"] = raw["cuts"]
        except Exception:
            pass
        job.params = params
        job.status = "running"
        db.commit()
        _pending_pull.append(job_id)
        return {"ok": True, "status": "stitching"}
    finally:
        db.close()


@studio_router.get("/studio/jobs/{job_id}/result")
async def studio_result(job_id: str, auth: dict = Depends(_client_auth)):
    user_id = _user_from_auth(auth)
    db = _db()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job or job.user_id != user_id:
            raise HTTPException(404, "not found")
        if job.status != "completed":
            raise HTTPException(409, "not ready")
        from routes import create_presigned_get_url

        key = job.result_r2_key or f"studio/{user_id}/{job_id}/final.mp4"
        return {"url": create_presigned_get_url(key), "key": key}
    finally:
        db.close()


@studio_router.post("/studio/uploads")
async def studio_uploads(body: StudioUploadRequest, auth: dict = Depends(_client_auth)):
    user_id = _user_from_auth(auth)
    job_id = body.job_id or "inbox"
    key = f"studio/{user_id}/{job_id}/{body.filename}"
    from routes import create_presigned_put_url

    url = create_presigned_put_url(key)
    if not url:
        raise HTTPException(500, "presign failed")
    return {"upload_url": url, "key": key}


@studio_router.post("/studio/internal/review_ready")
async def review_ready(payload: dict[str, Any], authorization: Optional[str] = Header(None)):
    _worker_auth(authorization)
    job_id = payload.get("job_id")
    db = _db()
    try:
        job = db.query(Job).filter(Job.job_id == job_id, Job.model == "studio").first()
        if not job:
            raise HTTPException(404, "not found")
        params = dict(job.params or {})
        if payload.get("storyboard"):
            params["studio_storyboard"] = payload["storyboard"]
        if payload.get("scene_runtimes"):
            params["scene_runtimes"] = payload["scene_runtimes"]
        if payload.get("compose_strategy"):
            params["compose_strategy"] = payload["compose_strategy"]
        params["studio_stage"] = "review"
        params["review_entered_at"] = utc_naive_now().isoformat()
        job.params = params
        job.status = "reviewing"
        db.commit()
        pipeline = params.get("pipeline") or "animated-explainer"
        vultr_pool.release_job(job_id, pipeline)
        await _notify(job, "review", "Your Studio scenes are ready to review")
        return {"ok": True, "status": "reviewing"}
    finally:
        db.close()


@studio_router.post("/studio/internal/complete")
async def studio_complete(payload: dict[str, Any], authorization: Optional[str] = Header(None)):
    _worker_auth(authorization)
    job_id = payload.get("job_id")
    db = _db()
    try:
        job = db.query(Job).filter(Job.job_id == job_id, Job.model == "studio").first()
        if not job:
            raise HTTPException(404, "not found")
        job.status = "completed"
        job.result_r2_key = payload.get("result_r2_key")
        job.completed_at = utc_naive_now()
        params = dict(job.params or {})
        params["studio_stage"] = "done"
        if payload.get("scene_runtimes"):
            params["scene_runtimes"] = payload["scene_runtimes"]
        if payload.get("compose_strategy"):
            params["compose_strategy"] = payload["compose_strategy"]
        job.params = params
        db.commit()
        pipeline = params.get("pipeline") or "animated-explainer"
        vultr_pool.release_job(job_id, pipeline)
        await _notify(job, "done", "Your Studio video is ready")
        return {"ok": True}
    finally:
        db.close()


@studio_router.post("/studio/internal/failed")
async def studio_failed(payload: dict[str, Any], authorization: Optional[str] = Header(None)):
    _worker_auth(authorization)
    job_id = payload.get("job_id")
    db = _db()
    try:
        job = db.query(Job).filter(Job.job_id == job_id, Job.model == "studio").first()
        if not job:
            raise HTTPException(404, "not found")
        if job.status == "reviewing":
            return {"ok": True, "skipped_refund": True}
        job.status = "failed"
        job.error = payload.get("error")
        db.commit()
        refund_credits_for_job(None, job_id, "studio_failed")
        pipeline = (job.params or {}).get("pipeline") or "animated-explainer"
        vultr_pool.release_job(job_id, pipeline)
        return {"ok": True}
    finally:
        db.close()


@studio_router.get("/studio/internal/next-job")
async def next_job(authorization: Optional[str] = Header(None)):
    _worker_auth(authorization)
    while _pending_pull:
        job_id = _pending_pull.pop(0)
        db = _db()
        try:
            job = db.query(Job).filter(Job.job_id == job_id, Job.model == "studio").first()
            if not job:
                continue
            if job.status in {"completed", "failed", "cancelled"}:
                continue
            # Regen while reviewing: keep REVIEWING so credits are not treated as a failure.
            if job.status != "reviewing":
                job.status = "running"
                params = dict(job.params or {})
                params["studio_stage"] = "running"
                job.params = params
                db.commit()
            return _sanitize_job(job)
        finally:
            db.close()
    return {}


@studio_router.post("/studio/internal/heartbeat")
async def studio_heartbeat(payload: dict[str, Any], authorization: Optional[str] = Header(None)):
    """Studio engine worker heartbeat. Does not dispatch GPU jobs."""
    _worker_auth(authorization)
    worker_id = payload.get("worker_id") or "studio-engine"
    db = _db()
    try:
        worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
        if not worker:
            worker = Worker(
                worker_id=worker_id,
                container_id=payload.get("vultr_id") or worker_id,
                model="studio_engine",
                status="ready",
                endpoint=payload.get("endpoint") or "",
                info={
                    "chrome_slots": payload.get("chrome_slots", 1),
                    "ram_mb": payload.get("ram_mb", 16384),
                    "supported_models": ["studio_engine"],
                },
            )
            db.add(worker)
        worker.model = "studio_engine"
        worker.status = "ready"
        worker.last_heartbeat = utc_naive_now()
        worker.last_active = utc_naive_now()
        if payload.get("vultr_id"):
            worker.container_id = payload.get("vultr_id")
        info = dict(worker.info or {})
        info["supported_models"] = ["studio_engine"]
        if payload.get("vultr_id"):
            info["vultr_id"] = payload.get("vultr_id")
        worker.info = info
        db.commit()
        vultr_pool.reap_idle()
        return {"ok": True, "model": "studio_engine"}
    finally:
        db.close()


def _looks_like_r2_key(value: str) -> bool:
    if not value or value.startswith("/") or "\\" in value:
        return False
    if value.startswith("project") or value.startswith("http"):
        return False
    return True


@studio_router.post("/studio/internal/presign")
async def studio_presign(payload: dict[str, Any], authorization: Optional[str] = Header(None)):
    _worker_auth(authorization)
    key = payload.get("key")
    if not key or not _looks_like_r2_key(str(key)):
        raise HTTPException(400, "invalid key")
    method = (payload.get("method") or "put").lower()
    from routes import create_presigned_get_url, create_presigned_put_url

    if method == "get":
        url = create_presigned_get_url(str(key))
    else:
        url = create_presigned_put_url(str(key))
    if not url:
        raise HTTPException(500, "presign failed")
    return {"url": url, "key": key}


@studio_router.post("/studio/internal/material-job")
async def studio_material_job(payload: dict[str, Any], authorization: Optional[str] = Header(None)):
    """Proxy a still/video plate for the engine. Fail closed if no image provider is configured."""
    _worker_auth(authorization)
    prompt = (payload.get("prompt") or "").strip()
    job_id = payload.get("job_id") or "material"
    if not prompt:
        raise HTTPException(400, "prompt required")
    fal_key = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
    if not fal_key:
        raise HTTPException(503, "material_provider_unavailable")
    import requests as _req

    resp = _req.post(
        "https://fal.run/fal-ai/flux/schnell",
        headers={"Authorization": f"Key {fal_key}", "Content-Type": "application/json"},
        json={"prompt": prompt, "image_size": "landscape_16_9"},
        timeout=120,
    )
    if resp.status_code >= 400:
        raise HTTPException(502, f"flux failed: {resp.text[:300]}")
    images = (resp.json().get("images") or [])
    if not images or not images[0].get("url"):
        raise HTTPException(502, "flux returned no image")
    image_url = images[0]["url"]
    blob = _req.get(image_url, timeout=60)
    if blob.status_code >= 400:
        raise HTTPException(502, "flux download failed")
    from routes import create_presigned_get_url, create_presigned_put_url

    key = f"studio/materials/{job_id}/{uuid.uuid4().hex}.png"
    put = create_presigned_put_url(key)
    if not put:
        raise HTTPException(500, "presign failed")
    up = _req.put(put, data=blob.content, headers={"Content-Type": "image/png"}, timeout=60)
    if up.status_code >= 400:
        raise HTTPException(502, "r2 put failed")
    get_url = create_presigned_get_url(key)
    return {"url": get_url, "key": key}


def expire_review_if_needed(db: Session, job: Job) -> None:
    if job.status != "reviewing":
        return
    entered = (job.params or {}).get("review_entered_at")
    if not entered:
        return
    try:
        ts = datetime.fromisoformat(entered)
    except ValueError:
        return
    if datetime.now(timezone.utc).replace(tzinfo=None) - ts.replace(tzinfo=None) > timedelta(hours=REVIEW_TTL_HOURS):
        job.status = "cancelled"
        job.error = "review_ttl_expired"
        # Do not refund while reviewing; TTL cancel also does not refund (user abandoned after lock).


async def _notify(job: Job, status: str, body: str) -> None:
    if not job.fcm_token:
        return
    try:
        from routes import send_fcm_notification

        await send_fcm_notification(
            job.fcm_token,
            "Studio",
            body,
            {"job_id": job.job_id, "status": status, "outputType": "studio_video"},
        )
    except Exception:
        return
