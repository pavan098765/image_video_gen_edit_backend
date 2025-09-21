# server_controller/main.py
# run: uvicorn main:app --host 0.0.0.0 --port 8000

import os
import uuid
import asyncio
import logging
import json
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import FastAPI, BackgroundTasks, Request, HTTPException, Depends, Header
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from db import init_db, SessionLocal, Worker, Job
from pod_manager import PodManager
from job_manager import JobManager, JobState
from security import (
    verify_signature,
    verify_api_key,
    hash_fingerprint,
    register_or_update_device,
    validate_device_signature,
    time_datetime_now,
)

# Optional R2 client
import boto3
from botocore.client import Config

# ------------------------
# ENV & INIT
# ------------------------
load_dotenv()
init_db()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("controller")

email_quickpod = os.getenv("QUICKPOD_EMAIL", "None")
password_quickpod = os.getenv("QUICKPOD_PASSWORD", "None")
warm_pod_uuid = os.getenv("QUICKPOD_POD_UUID")
IMAGE_PREFIX = os.getenv("IMAGE_PREFIX", "quickpod/")
CONTROLLER_PUBLIC_URL = os.getenv("CONTROLLER_PUBLIC_URL", "")
CONTROLLER_SHARED_SECRET = os.getenv("CONTROLLER_SHARED_SECRET", "")
WORKER_IDLE_TIMEOUT = int(os.getenv("WORKER_IDLE_TIMEOUT", "10"))
WORKER_REGISTRATION_TIMEOUT = int(os.getenv("WORKER_REGISTRATION_TIMEOUT", "180"))

# R2
R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_BUCKET = os.getenv("R2_BUCKET")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
s3_client = None
if R2_ENDPOINT and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY:
    s3_client = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )

# Managers
pod_manager = PodManager(email_quickpod, password_quickpod, warm_pod_uuid)
job_manager: Optional[JobManager] = None

app = FastAPI(title="Server Controller", root_path="/imgvidgen")


# ------------------------
# Pydantic Models
# ------------------------
class SubmitRequest(BaseModel):
    model: str
    prompt: Optional[str] = ""
    params: Optional[Dict[str, Any]] = {}


class RegisterPayload(BaseModel):
    worker_id: str
    models: list
    endpoint: str = ""  # keep optional-equivalent behavior by allowing empty string
    container_id: Optional[str] = None
    pod_uuid: Optional[str] = None          # Preferred canonical pod identifier (QuickPod)
    public_ip: Optional[str] = None         # Optional public IP of the pod (if available)
    info: Optional[Dict[str, Any]] = None


class JobCompletePayload(BaseModel):
    job_id: str
    result_r2_key: str
    failed: Optional[bool] = False


# ------------------------
# Auth Dependencies
# ------------------------
async def require_any_auth(
    authorization: Optional[str] = Header(None),
    x_device_uuid: Optional[str] = Header(None),
    x_fingerprint: Optional[str] = Header(None),
    x_timestamp: Optional[str] = Header(None),
    x_signature: Optional[str] = Header(None),
):
    if authorization and verify_api_key(authorization):
        return {"type": "api_key", "key": authorization}

    if not x_fingerprint or not x_timestamp or not x_signature:
        raise HTTPException(status_code=401, detail="Missing auth headers")

    fingerprint_hash = (
        x_fingerprint.lower()
        if len(x_fingerprint) == 64
        and all(c in "0123456789abcdef" for c in x_fingerprint.lower())
        else hash_fingerprint(x_fingerprint)
    )

    if not validate_device_signature(fingerprint_hash, x_timestamp, x_signature):
        raise HTTPException(
            status_code=403, detail="Invalid device signature or timestamp"
        )

    rec = register_or_update_device(
        fingerprint_hash,
        x_device_uuid,
        meta={"last_seen_at": time_datetime_now().isoformat()},
    )
    return {
        "type": "device",
        "fingerprint_hash": fingerprint_hash,
        "device_uuid": rec.get("device_uuid"),
    }


# ------------------------
# DB Helpers
# ------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def find_ready_worker(db: Session, model: str) -> Optional[Worker]:
    return (
        db.query(Worker)
        .filter(Worker.model == model, Worker.status == "ready")
        .order_by(Worker.last_active.desc())
        .first()
    )


def create_presigned_put_url(key: str, expires_in: int = 3600) -> Optional[str]:
    """Generate a presigned PUT URL for R2 so worker can upload results directly."""
    if not s3_client:
        return None
    try:
        return s3_client.generate_presigned_url(
            "put_object",
            Params={"Bucket": R2_BUCKET, "Key": key},
            ExpiresIn=expires_in,
        )
    except Exception as e:
        logger.error(f"Failed to generate presigned URL for key {key}: {e}")
        return None


def upsert_worker(
    db: Session,
    worker_id: str,
    model: str,
    endpoint: str,
    container_id: Optional[str],
    info: Optional[Dict[str, Any]],
):
    """
    NOTE: 'container_id' DB column stores the QuickPod pod_uuid (canonical pod id).
    We intentionally keep the DB column name as container_id for backward compatibility.
    """
    w = db.query(Worker).filter(Worker.worker_id == worker_id).first()
    now = datetime.utcnow()
    is_warm = (
        container_id.startswith(warm_pod_uuid)
        if container_id and warm_pod_uuid
        else False
    )
    if not w:
        w = Worker(
            worker_id=worker_id,
            model=model,
            endpoint=endpoint,
            status="ready",
            container_id=container_id,
            last_heartbeat=now,
            last_active=now,
            info=info,
            is_warm_pod=is_warm,
        )
        db.add(w)
    else:
        w.endpoint = endpoint
        w.model = model
        w.container_id = container_id or w.container_id
        w.status = "ready"
        w.last_heartbeat = now
        w.last_active = now
        w.info = info
        if container_id:
            w.is_warm_pod = is_warm
    db.commit()
    return w


def assign_job_to_worker(job_id: str, worker_id: str, presigned_url: str):
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
        if not job or not worker:
            logger.error(
                "Job or Worker not found (job=%s, worker=%s)", job_id, worker_id
            )
            return

        run_payload = {
            "job_id": job.job_id,
            "prompt": job.prompt,
            "model": job.model,
            "mode": job.params.get("mode"),
            "params": job.params,
            "upload_url": presigned_url,  # <-- worker gets this
            "result_key": job.result_r2_key,  # optional, for reference
        }
        try:
            url = f"{worker.endpoint.rstrip('/')}/run"
            resp = requests.post(url, json=run_payload, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            logger.error("Failed to push job %s to worker %s: %s", job_id, worker_id, e)
            job_manager.update_job_state(
                job.job_id, JobState.FAILED, f"Dispatch error: {e}"
            )
            return

        job.worker_id = worker.worker_id
        job_manager.update_job_state(job.job_id, JobState.RUNNING)
        if worker.is_warm_pod:
            worker.status = "processing"
        else:
            worker.status = "busy"
        worker.last_active = datetime.utcnow()
        db.commit()
        logger.info(
            "Dispatched job %s to worker %s (%s)", job_id, worker_id, worker.endpoint
        )
    except Exception:
        db.rollback()
        logger.exception("Error assigning job %s to worker %s", job_id, worker_id)
    finally:
        db.close()


# ------------------------
# Background Tasks
# ------------------------
async def idle_worker_reaper_task():
    logger.info("Starting idle worker reaper")
    while True:
        db = SessionLocal()
        try:
            threshold = datetime.utcnow() - timedelta(seconds=WORKER_IDLE_TIMEOUT)
            idle_workers = (
                db.query(Worker)
                .filter(Worker.last_active < threshold, Worker.status == "ready")
                .all()
            )
            for w in idle_workers:
                if w.is_warm_pod:
                    continue
                try:
                    if w.container_id:
                        await pod_manager.stop_individual_pod(w.container_id)
                    w.status = "offline"
                    db.commit()
                except Exception as e:
                    logger.error(f"Failed to stop worker {w.worker_id}: {e}")
                running_jobs = (
                    db.query(Job)
                    .filter(
                        Job.worker_id == w.worker_id,
                        Job.status == JobState.RUNNING.value,
                    )
                    .all()
                )
                for job in running_jobs:
                    job_manager.update_job_state(
                        job.job_id, JobState.FAILED, "Worker went offline"
                    )
        finally:
            db.close()
        await asyncio.sleep(max(1, WORKER_IDLE_TIMEOUT // 2))


async def pod_cleanup_task():
    while True:
        db = SessionLocal()
        try:
            pods_to_cleanup = (
                db.query(Worker)
                .filter(Worker.status == "terminated", Worker.is_warm_pod == False)
                .all()
            )
            for pod in pods_to_cleanup:
                try:
                    if pod.container_id:
                        await pod_manager.stop_individual_pod(pod.container_id)
                    db.delete(pod)
                    db.commit()
                except Exception as e:
                    logger.error(f"Failed pod cleanup {pod.worker_id}: {e}")
                    pod.status = "error"
                    pod.info = {
                        "error": str(e),
                        "cleanup_attempted_at": datetime.utcnow().isoformat(),
                    }
                    db.commit()
        finally:
            db.close()
        await asyncio.sleep(15)


async def job_recovery_task():
    while True:
        try:
            await job_manager.check_for_stuck_jobs()
        except Exception:
            logger.exception("Error in job recovery task")
        await asyncio.sleep(30)


# ------------------------
# Startup
# ------------------------
@app.on_event("startup")
async def startup_event():
    global job_manager
    try:
        job_manager = JobManager(SessionLocal, pod_manager)
        await pod_manager.initialize()
        asyncio.create_task(idle_worker_reaper_task())
        asyncio.create_task(pod_cleanup_task())
        asyncio.create_task(job_recovery_task())
        logger.info("Controller startup complete")
    except Exception as e:
        logger.error(f"Error during startup: {e}")


# ------------------------
# API Endpoints
# ------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/job_complete")
async def job_complete(payload: JobCompletePayload):
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == payload.job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if payload.failed:
            job_manager.update_job_state(
                job.job_id, JobState.FAILED, "Worker reported failure"
            )
        else:
            job_manager.update_job_state(job.job_id, JobState.COMPLETED)
            job.result_r2_key = payload.result_r2_key

        worker = db.query(Worker).filter(Worker.worker_id == job.worker_id).first()
        if worker:
            if worker.is_warm_pod:
                worker.status = "ready"
            else:
                worker.status = "terminated"
            worker.last_active = datetime.utcnow()
        db.commit()
    finally:
        db.close()
    return {"status": "ok"}


@app.post("/heartbeat")
async def heartbeat(request: Request):
    body = await request.body()
    sig = request.headers.get("X-Signature", "")
    if not verify_signature(sig, body):
        raise HTTPException(status_code=401, detail="Invalid signature")

    data = json.loads(body)
    worker_id = data.get("worker_id")
    if not worker_id:
        raise HTTPException(status_code=400, detail="Missing worker_id")

    with SessionLocal() as db:
        w = db.query(Worker).filter(Worker.worker_id == worker_id).first()
        if not w:
            raise HTTPException(status_code=404, detail="Worker not found")
        w.last_heartbeat = datetime.utcnow()
        w.last_active = datetime.utcnow()
        db.commit()

    return {"status": "ok"}


@app.post("/register_worker")
async def register_worker(request: Request):
    body = await request.body()
    sig = request.headers.get("X-Signature", "")
    if not verify_signature(sig, body):
        raise HTTPException(status_code=401, detail="Invalid signature")

    data = json.loads(body)
    payload = RegisterPayload(**data)

    if not payload.models:
        raise HTTPException(
            status_code=400, detail="Worker must declare at least one model"
        )

    # prefer pod_uuid (QuickPod canonical id) over legacy container_id
    container_id_received = payload.pod_uuid or payload.container_id

    # determine endpoint: prefer explicit endpoint, else use public_ip if given
    endpoint = payload.endpoint or (f"http://{payload.public_ip}" if payload.public_ip else "")

    with SessionLocal() as db:
        for model in payload.models:
            upsert_worker(
                db,
                payload.worker_id,
                model,
                endpoint,
                container_id_received,
                payload.info,
            )

    logger.info(
        "Registered worker %s for models %s (pod_uuid=%s, container_id=%s, endpoint=%s)",
        payload.worker_id,
        payload.models,
        payload.pod_uuid,
        payload.container_id,
        endpoint,
    )
    return {"status": "ok"}


@app.post("/submit")
async def submit(
    req: SubmitRequest,
    background_tasks: BackgroundTasks,
    auth=Depends(require_any_auth),
):
    job_id = str(uuid.uuid4())
    db = SessionLocal()
    result_key = f"{IMAGE_PREFIX}{job_id}.png"
    try:
        job = Job(
            job_id=job_id,
            model=req.model,
            prompt=req.prompt,
            params=req.params or {},
            status=JobState.QUEUED.value,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            timeout_at=datetime.utcnow()
            + timedelta(seconds=req.params.get("timeout", 3600)),
            result_r2_key=result_key,
        )
        db.add(job)
        db.commit()
    finally:
        db.close()

    # Generate presigned PUT URL for this job
    presigned_url = create_presigned_put_url(result_key)
    if not presigned_url:
        raise HTTPException(status_code=500, detail="Failed to generate upload URL")

    try:
        worker_id = await pod_manager.dispatch_job(
            job_id, req.model, req.params.get("mode")
        )
        background_tasks.add_task(
            assign_job_to_worker, job_id, worker_id, presigned_url
        )
        return {
            "job_id": job_id,
            "status": "dispatched",
            "routed_to_worker": worker_id,
            "upload_url": presigned_url,
            "result_key": result_key,
        }
    except Exception as e:
        logger.error("Failed to dispatch job %s: %s", job_id, e)
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job:
                job.status = JobState.FAILED.value
                job.error = str(e)
                db.commit()
        finally:
            db.close()
        raise HTTPException(status_code=500, detail=f"Dispatch failed: {e}")
