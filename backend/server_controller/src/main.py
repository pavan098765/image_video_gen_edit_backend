# controller/main.py
import os
import time
import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import FastAPI, BackgroundTasks, Request, HTTPException, Depends, Header
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from sqlalchemy.orm import Session

from db import init_db, SessionLocal, Worker, Job, Device
from salad_api import start_salad_worker, delete_salad_worker
from security import (
    verify_signature,
    verify_api_key, hash_fingerprint, register_or_update_device,
    validate_device_signature, time_datetime_now
)

# Optional boto3 for presign
import boto3
from botocore.client import Config
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type


load_dotenv()
init_db()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("controller")

# Config
WORKER_IDLE_TIMEOUT = int(os.getenv("WORKER_IDLE_TIMEOUT", "10"))
WORKER_REGISTRATION_TIMEOUT = int(os.getenv("WORKER_REGISTRATION_TIMEOUT", "180"))
IMAGE_PREFIX = os.getenv("SALAD_WORKER_IMAGE_PREFIX", "")
GPU_CLASS_IDS = [g for g in os.getenv("SALAD_GPU_CLASS_IDS", "").split(",") if g]
CONTROLLER_PUBLIC_URL = os.getenv("CONTROLLER_PUBLIC_URL", "")
CONTROLLER_SHARED_SECRET = os.getenv("CONTROLLER_SHARED_SECRET", "")

# R2 (optional - used for presign endpoint)
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
        config=Config(signature_version="s3v4")
    )

app = FastAPI(title="Server Controller")

# Pydantic models
class SubmitRequest(BaseModel):
    model: str
    prompt: Optional[str] = ""
    params: Optional[Dict[str, Any]] = {}

class RegisterPayload(BaseModel):
    worker_id: str
    models: list
    endpoint: str
    container_id: Optional[str] = None
    info: Optional[Dict[str, Any]] = None

class JobCompletePayload(BaseModel):
    job_id: str
    result_r2_key: str
    failed: Optional[bool] = False 

class RegisterDeviceRequest(BaseModel):
    fingerprint: str
    device_uuid: Optional[str] = None
    meta: Optional[dict] = None

# Auth dependency (unchanged)
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
        raise HTTPException(status_code=401, detail="Missing auth (api key or device headers)")

    raw = x_fingerprint
    if len(raw) == 64 and all(c in "0123456789abcdef" for c in raw.lower()):
        fingerprint_hash = raw.lower()
    else:
        fingerprint_hash = hash_fingerprint(raw)

    if not validate_device_signature(fingerprint_hash, x_timestamp, x_signature):
        raise HTTPException(status_code=403, detail="Invalid device signature or timestamp")

    rec = register_or_update_device(fingerprint_hash, x_device_uuid, meta={"last_seen_at": time_datetime_now().isoformat()})
    return {"type": "device", "fingerprint_hash": fingerprint_hash, "device_uuid": rec.get("device_uuid")}

async def require_device_auth(
    x_device_uuid: Optional[str] = Header(None),
    x_fingerprint: Optional[str] = Header(None),
    x_timestamp: Optional[str] = Header(None),
    x_signature: Optional[str] = Header(None),
):
    if not x_fingerprint or not x_timestamp or not x_signature:
        raise HTTPException(status_code=401, detail="Missing device headers")

    raw = x_fingerprint
    fingerprint_hash = (
        raw.lower()
        if len(raw) == 64 and all(c in "0123456789abcdef" for c in raw.lower())
        else hash_fingerprint(raw)
    )

    if not validate_device_signature(fingerprint_hash, x_timestamp, x_signature):
        raise HTTPException(status_code=403, detail="Invalid device signature or timestamp")

    rec = register_or_update_device(fingerprint_hash, x_device_uuid, meta={"last_seen_at": time_datetime_now().isoformat()})

    db = SessionLocal()
    try:
        dev = (
            db.query(Device)
            .filter(Device.fingerprint_hash == fingerprint_hash)
            .order_by(Device.last_seen.desc())
            .first()
        )
        if not dev:
            raise HTTPException(status_code=403, detail="Device not registered")
        if dev.banned:
            raise HTTPException(status_code=403, detail="Device banned")
    finally:
        db.close()

    return {"fingerprint_hash": fingerprint_hash, "device_uuid": rec.get("device_uuid")}


# DB helpers
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def find_ready_worker(db: Session, model: str) -> Optional[Worker]:
    return db.query(Worker).filter(Worker.model == model, Worker.status == "ready").order_by(Worker.last_active.desc()).first()

def upsert_worker(db: Session, worker_id: str, model: str, endpoint: str, container_id: Optional[str], info: Optional[Dict[str,Any]]):
    w = db.query(Worker).filter(Worker.worker_id == worker_id).first()
    now = datetime.utcnow()
    if not w:
        w = Worker(worker_id=worker_id, model=model, endpoint=endpoint, status="ready", container_id=container_id, last_heartbeat=now, last_active=now, info=info)
        db.add(w)
    else:
        w.endpoint = endpoint
        w.model = model
        w.container_id = container_id or w.container_id
        w.status = "ready"
        w.last_heartbeat = now
        w.last_active = now
        w.info = info
    db.commit()
    return w

def assign_job_to_worker(db: Session, job: Job, worker: Worker):
    job.worker_id = worker.worker_id
    job.status = "running"
    job.updated_at = datetime.utcnow()
    db.commit()
    worker.status = "busy"
    worker.last_active = datetime.utcnow()
    db.commit()

# idle reaper
async def idle_worker_reaper_task():
    logger.info("Starting idle worker reaper with timeout=%s", WORKER_IDLE_TIMEOUT)
    while True:
        try:
            db = SessionLocal()
            threshold = datetime.utcnow() - timedelta(seconds=WORKER_IDLE_TIMEOUT)
            idle_workers = db.query(Worker).filter(
                Worker.last_active < threshold,
                Worker.status == "ready"
            ).all()

            for w in idle_workers:
                logger.info(
                    "Reaper: shutting idle worker %s model=%s endpoint=%s container=%s",
                    w.worker_id, w.model, w.endpoint, w.container_id
                )
                try:
                    if w.container_id:
                        delete_salad_worker(w.container_id)
                except Exception as e:
                    logger.exception("Failed to delete container for worker %s: %s", w.worker_id, e)

                # mark worker offline
                w.status = "offline"
                db.commit()

                # Mark all running jobs assigned to this worker as failed
                running_jobs = db.query(Job).filter(Job.worker_id == w.worker_id, Job.status == "running").all()
                for job in running_jobs:
                    job.status = "failed"
                    job.error = "Worker went offline"
                    job.updated_at = datetime.utcnow()
                db.commit()

            db.close()
        except Exception as e:
            logger.exception("Error in reaper loop: %s", e)

        await asyncio.sleep(max(1, WORKER_IDLE_TIMEOUT // 2))
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(idle_worker_reaper_task())
    logger.info("Controller startup complete")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/register_worker")
async def register_worker(request: Request):
    body = await request.body()
    sig = request.headers.get("X-Signature", "")
    if not verify_signature(sig, body):
        raise HTTPException(status_code=401, detail="Invalid signature")
    data = await request.json()
    payload = RegisterPayload(**data)

    if not payload.models or len(payload.models) == 0:
        raise HTTPException(status_code=400, detail="Worker must declare at least one model in 'models'")

    db = SessionLocal()
    try:
        # Use the first model for routing (workers may support multiple but we keep a simple mapping)
        model = payload.models[0]
        w = upsert_worker(db, payload.worker_id, model, payload.endpoint, payload.container_id, payload.info)
        logger.info("Registered worker %s for model %s at %s", payload.worker_id, model, payload.endpoint)
    finally:
        db.close()

    return {"status": "ok"}

@app.post("/heartbeat")
async def heartbeat(request: Request):
    """
    Workers should call this periodically (e.g., every 20s) with HMAC signature.
    Body: { "worker_id": "...", "timestamp": 123456789 }
    """
    body = await request.body()
    sig = request.headers.get("X-Signature", "")
    if not verify_signature(sig, body):
        raise HTTPException(status_code=401, detail="Invalid signature")
    data = await request.json()
    worker_id = data.get("worker_id")
    if not worker_id:
        raise HTTPException(status_code=400, detail="worker_id required")
    db = SessionLocal()
    try:
        w = db.query(Worker).filter(Worker.worker_id == worker_id).first()
        if not w:
            raise HTTPException(status_code=404, detail="worker not registered")
        w.last_heartbeat = datetime.utcnow()
        w.last_active = datetime.utcnow()
        w.status = "ready"
        db.commit()
    finally:
        db.close()
    return {"status": "ok"}

@app.post("/job_complete")
async def job_complete(request: Request):
    body = await request.body()
    sig = request.headers.get("X-Signature", "")
    if not verify_signature(sig, body):
        raise HTTPException(status_code=401, detail="Invalid signature")
    data = await request.json()
    payload = JobCompletePayload(**data)

    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == payload.job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if payload.failed:
            job.status = "failed"
            job.error = "Worker reported failure"
        else:
            job.status = "completed"
            job.result_r2_key = payload.result_r2_key

        job.updated_at = datetime.utcnow()

        # mark worker ready again
        if job.worker_id:
            w = db.query(Worker).filter(Worker.worker_id == job.worker_id).first()
            if w:
                w.status = "ready"
                w.last_active = datetime.utcnow()
        db.commit()
    finally:
        db.close()
    logger.info("Job %s completed, result stored %s", payload.job_id, payload.result_r2_key)
    return {"status": "ok"}

@app.post("/presign_upload")
def presign_upload(filename: str, expiration: int = 3600, auth=Depends(require_any_auth)):
    """
    Optionally provide presigned PUT URL for workers to upload outputs.
    Requires R2 credentials in config.env.
    Returns { url, r2_key }.
    """
    if not s3_client:
        raise HTTPException(status_code=503, detail="R2 not configured on controller")
    r2_key = f"results/{uuid.uuid4()}/{filename}"
    url = s3_client.generate_presigned_url(
        ClientMethod="put_object",
        Params={"Bucket": R2_BUCKET, "Key": r2_key},
        ExpiresIn=expiration
    )
    return {"url": url, "r2_key": r2_key}

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3),
       retry=retry_if_exception_type(requests.RequestException))
def post_job_to_worker(endpoint: str, payload: dict):
    run_url = f"{endpoint.rstrip('/')}/run"
    r = requests.post(run_url, json=payload, timeout=60)
    r.raise_for_status()
    return r

def send_job_to_worker(endpoint: str, job_id: str, model: str, prompt: str, params: dict):
    payload = {"job_id": job_id, "prompt": prompt, "params": params}
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            logger.error("send_job_to_worker: job %s not found", job_id)
            return
        w = db.query(Worker).filter(Worker.endpoint == endpoint).first()
        if not w:
            logger.error("send_job_to_worker: worker for endpoint %s not found", endpoint)
            job.status = "failed"
            job.error = "worker not found"
            job.updated_at = datetime.utcnow()
            db.commit()
            return

        try:
            post_job_to_worker(endpoint, payload)  # with retries
        except Exception as e:
            logger.exception("Failed sending job %s to worker after retries", job_id)
            job.status = "failed"
            job.error = str(e)
            job.updated_at = datetime.utcnow()
            db.commit()
            return

        assign_job_to_worker(db, job, w)
        logger.info("Job %s accepted by worker %s", job_id, w.worker_id)
    finally:
        db.close()
        
@app.post("/submit")
async def submit(req: SubmitRequest, background_tasks: BackgroundTasks, auth=Depends(require_any_auth)):
    """
    Submit a job:
    - If an existing worker for model is ready, route to it.
    - Else: create a Salad worker for the model and wait for registration.
    """
    logger.info("Authenticated request: %s", auth)

    job_id = str(uuid.uuid4())
    db = SessionLocal()
    job = Job(job_id=job_id, model=req.model, prompt=req.prompt, params=req.params or {}, status="queued", created_at=datetime.utcnow(), updated_at=datetime.utcnow())
    db.add(job)
    db.commit()
    logger.info("Received job %s model=%s", job_id, req.model)

    # Try find existing worker
    w = find_ready_worker(db, req.model)
    if w:
        logger.info("Found ready worker %s for model %s - routing job %s", w.worker_id, req.model, job_id)
        # route in background
        background_tasks.add_task(send_job_to_worker, w.endpoint, job_id, req.model, req.prompt, req.params or {})
        db.close()
        return {"job_id": job_id, "status": "queued", "routed_to_worker": w.worker_id}

    # No existing worker -> spawn
    logger.info("No existing worker for %s - spinning new Salad worker", req.model)
    image = f"{IMAGE_PREFIX}:{req.model}"
    worker_name = f"worker-{req.model}-{job_id}"
    env = {
        "MODEL_KEY": req.model,
        "JOB_ID": job_id,
        "CONTROLLER_URL": CONTROLLER_PUBLIC_URL or f"http://{os.getenv('CONTROLLER_HOST','localhost')}:{os.getenv('CONTROLLER_PORT','8080')}",
        "CONTROLLER_SHARED_SECRET": CONTROLLER_SHARED_SECRET
    }
    try:
        resp = start_salad_worker(worker_name, image, GPU_CLASS_IDS, env)
    except Exception as e:
        logger.exception("Failed to create salad worker: %s", e)
        job.status = "failed"
        job.error = str(e)
        db.commit()
        db.close()
        raise HTTPException(status_code=500, detail="failed to start worker")

    # attempt to grab container id (best-effort)
    container_id = None
    try:
        if isinstance(resp, dict):
            container_id = resp.get("container", {}).get("id") or resp.get("id") or resp.get("container_id")
    except Exception:
        container_id = None

    # wait for worker to register
    logger.info("Waiting up to %s sec for worker registration...", WORKER_REGISTRATION_TIMEOUT)
    registered_worker = None
    start = time.time()
    deadline = start + WORKER_REGISTRATION_TIMEOUT
    while time.time() < deadline:
        candidate = find_ready_worker(db, req.model)
        if candidate:
            registered_worker = candidate
            break
        await asyncio.sleep(1)
    if not registered_worker:
        # cleanup
        if container_id:
            try:
                delete_salad_worker(container_id)
            except Exception:
                logger.exception("Failed to delete container after registration timeout")
        job.status = "failed"
        job.error = "worker registration timeout"
        db.commit()
        db.close()
        raise HTTPException(status_code=500, detail="worker registration timeout")

    # update worker row with container id and mark busy
    db = SessionLocal()
    try:
        w_row = db.query(Worker).filter(Worker.worker_id == registered_worker.worker_id).first()
        if w_row:
            w_row.container_id = container_id or w_row.container_id
            w_row.last_active = datetime.utcnow()
            w_row.status = "busy"
            db.commit()
    finally:
        db.close()

    # dispatch job
    background_tasks.add_task(send_job_to_worker, registered_worker.endpoint, job_id, req.model, req.prompt, req.params or {})
    return {"job_id": job_id, "status": "dispatched", "routed_to_worker": registered_worker.worker_id}

@app.get("/job_status/{job_id}")
def job_status(job_id: str, auth=Depends(require_any_auth)):
    db = SessionLocal()
    job = db.query(Job).filter(Job.job_id == job_id).first()
    db.close()
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "job_id": job.job_id, "model": job.model, "status": job.status,
        "result_r2_key": job.result_r2_key, "error": job.error, "worker_id": job.worker_id
    }

@app.get("/workers")
def list_workers(auth=Depends(require_any_auth)):
    db = SessionLocal()
    rows = db.query(Worker).all()
    out = []
    for w in rows:
        out.append({"worker_id": w.worker_id, "model": w.model, "endpoint": w.endpoint, "status": w.status, "last_active": w.last_active.isoformat() if w.last_active else None})
    db.close()
    return out
