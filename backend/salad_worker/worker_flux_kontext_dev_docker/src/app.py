# \image-video-generator\backend\src\salad_worker\worker_flux-kontext-dev_docker/src/app.py
import os
import time
import uuid
import json
import logging
import asyncio
import hmac
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime

import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Optional boto3 (if worker will upload directly)
try:
    import boto3
    from botocore.client import Config
except Exception:
    boto3 = None

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")

# Configuration from env - REQUIRED
WORKER_ID = os.getenv("WORKER_ID", str(uuid.uuid4()))
MODEL_KEY = os.getenv("MODEL_KEY", os.getenv("MODEL", "qwen-image"))
CONTROLLER_URL = os.getenv("CONTROLLER_URL")  # public URL of controller, required
CONTROLLER_SHARED_SECRET = os.getenv("CONTROLLER_SHARED_SECRET")
CONTROLLER_API_KEY = os.getenv("CONTROLLER_API_KEY", "")  # optional: used for controller /presign_upload auth
WORKER_PUBLIC_URL = os.getenv("WORKER_PUBLIC_URL")  # public endpoint (must be reachable)
PORT = int(os.getenv("PORT", "8080"))

# Heartbeat & concurrency
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "20"))  # seconds
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))  # 1 = serial by default

# Workspace paths
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# R2 options (optional)
R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_BUCKET = os.getenv("R2_BUCKET")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")

# Create s3 client if credentials present
s3_client = None
if R2_ENDPOINT and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and boto3:
    s3_client = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )
    logger.info("Initialized R2 boto3 client")

# Basic validation
if not CONTROLLER_URL:
    logger.error("CONTROLLER_URL env must be set - controller public URL")
    raise RuntimeError("CONTROLLER_URL missing")

if not CONTROLLER_SHARED_SECRET:
    logger.error("CONTROLLER_SHARED_SECRET env must be set")
    raise RuntimeError("CONTROLLER_SHARED_SECRET missing")

# FastAPI app
app = FastAPI(title=f"Salad Worker ({MODEL_KEY})")

# Job queue & status
job_queue: asyncio.Queue = asyncio.Queue()
job_status: Dict[str, Dict[str, Any]] = {}
semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

# Pydantic models
class RunRequest(BaseModel):
    job_id: str
    prompt: str
    params: Dict[str, Any] = {}

# -----------------------
# HMAC utilities
# -----------------------
def sign_payload_bytes(payload_bytes: bytes) -> str:
    """
    Exactly the same algorithm as controller.security.sign_payload_bytes:
    returns "sha256=<hex>"
    """
    mac = hmac.new(CONTROLLER_SHARED_SECRET.encode("utf-8"), payload_bytes, hashlib.sha256)
    return "sha256=" + mac.hexdigest()

def post_signed_json(url: str, payload: dict, timeout: int = 10) -> requests.Response:
    b = json.dumps(payload).encode("utf-8")
    sig = sign_payload_bytes(b)
    headers = {"X-Signature": sig, "Content-Type": "application/json"}
    # Optionally add Authorization if you want controller endpoints that require it
    if CONTROLLER_API_KEY:
        headers["Authorization"] = f"Bearer {CONTROLLER_API_KEY}"
    resp = requests.post(url, data=b, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp

# -----------------------
# Controller registration
# -----------------------
@retry(wait=wait_exponential(multiplier=1, min=2, max=20), stop=stop_after_attempt(6), retry=retry_if_exception_type(Exception))
def register_with_controller():
    """
    Register this worker with controller. Retries on failure.
    """
    register_payload = {
        "worker_id": WORKER_ID,
        "models": [MODEL_KEY],
        "endpoint": WORKER_PUBLIC_URL or f"http://{os.getenv('HOSTNAME')}:{PORT}",
        "container_id": os.getenv("CONTAINER_ID", None),
        "info": {"started_at": datetime.utcnow().isoformat()}
    }
    url = f"{CONTROLLER_URL.rstrip('/')}/register_worker"
    logger.info("Registering with controller: %s -> %s", url, register_payload)
    resp = post_signed_json(url, register_payload, timeout=10)
    logger.info("Register response: %s", resp.text)
    return resp.json()

@retry(wait=wait_exponential(multiplier=1, min=2, max=20), stop=stop_after_attempt(6), retry=retry_if_exception_type(Exception))
def send_heartbeat():
    """
    Send heartbeat to controller. Called periodically.
    """
    payload = {"worker_id": WORKER_ID, "timestamp": int(time.time())}
    url = f"{CONTROLLER_URL.rstrip('/')}/heartbeat"
    b = json.dumps(payload).encode("utf-8")
    sig = sign_payload_bytes(b)
    headers = {"X-Signature": sig, "Content-Type": "application/json"}
    logger.debug("Heartbeat -> %s", url)
    r = requests.post(url, data=b, headers=headers, timeout=5)
    r.raise_for_status()
    return r.json()

# -----------------------
# Presign/upload helpers
# -----------------------
@retry(wait=wait_exponential(multiplier=1, min=2, max=20), stop=stop_after_attempt(5), retry=retry_if_exception_type(Exception))
def request_presign(filename: str, expiration: int = 3600) -> Dict[str, str]:
    """
    Ask controller to presign a PUT URL (controller must be configured with R2 keys).
    POST /presign_upload?filename=...&expiration=...
    Returns { "url": "...", "r2_key": "..." }
    """
    url = f"{CONTROLLER_URL.rstrip('/')}/presign_upload?filename={filename}&expiration={expiration}"
    headers = {}
    if CONTROLLER_API_KEY:
        headers["Authorization"] = f"Bearer {CONTROLLER_API_KEY}"
    logger.info("Requesting presign for %s", filename)
    r = requests.post(url, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()

@retry(wait=wait_exponential(multiplier=1, min=2, max=20), stop=stop_after_attempt(5), retry=retry_if_exception_type(Exception))
def upload_file_to_presigned_url(presigned_url: str, local_path: str):
    """
    Upload file bytes to presigned PUT URL.
    """
    logger.info("Uploading %s to presigned URL", local_path)
    with open(local_path, "rb") as f:
        # We must not set any auth headers; presigned URL handles auth.
        r = requests.put(presigned_url, data=f, timeout=300)
    if not (200 <= r.status_code < 300):
        raise RuntimeError(f"Failed presigned upload: {r.status_code} {r.text}")
    logger.info("Upload OK (presigned)")

@retry(wait=wait_exponential(multiplier=1, min=2, max=20), stop=stop_after_attempt(5), retry=retry_if_exception_type(Exception))
def upload_file_direct_s3(local_path: str, r2_key: str):
    """
    Upload using boto3 (R2) if worker has credentials baked in.
    """
    if not s3_client:
        raise RuntimeError("S3 client not configured in worker")
    logger.info("Uploading %s to R2 (direct) -> %s", local_path, r2_key)
    s3_client.upload_file(local_path, R2_BUCKET, r2_key)
    logger.info("Upload OK (direct)")

# -----------------------
# Notify controller job complete
# -----------------------
@retry(wait=wait_exponential(multiplier=1, min=2, max=20), stop=stop_after_attempt(6), retry=retry_if_exception_type(Exception))
def notify_controller_job_complete(job_id: str, r2_key: str):
    payload = {"job_id": job_id, "result_r2_key": r2_key}
    url = f"{CONTROLLER_URL.rstrip('/')}/job_complete"
    logger.info("Notifying controller job_complete for %s -> %s", job_id, r2_key)
    resp = post_signed_json(url, payload, timeout=10)
    logger.info("Controller job_complete response: %s", resp.text)
    return resp.json()

# -----------------------
# Inference placeholder
# -----------------------
async def run_inference_and_save(prompt: str, params: Dict[str, Any], job_id: str) -> str:
    """
    Replace this with your real inference call.
    Must return the local path to the saved output file.
    """
    logger.info("RUN inference stub: job=%s prompt=%.60s", job_id, prompt[:60])
    # Example: simulate some runtime between 4-10 sec
    simulated = int(params.get("simulated_time", 6))
    await asyncio.sleep(simulated)
    out_path = os.path.join(WORKSPACE_DIR, f"output_{job_id}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + os.urandom(4096))
    logger.info("Saved dummy output to %s", out_path)
    return out_path

# -----------------------
# Job processing loop
# -----------------------
async def process_job_item(item: Dict[str, Any]):
    job_id = item["job_id"]
    prompt = item["prompt"]
    params = item.get("params", {})
    logger.info("Processing job %s", job_id)
    job_status[job_id] = {"status": "running", "started_at": datetime.utcnow().isoformat()}
    try:
        # acquire slot
        async with semaphore:
            # perform inference (replace with real call)
            out_local = await run_inference_and_save(prompt, params, job_id)
            filename = os.path.basename(out_local)

            # upload: prefer direct S3 if available, otherwise presign via controller
            if s3_client:
                r2_key = f"results/{job_id}/{filename}"
                upload_file_direct_s3(out_local, r2_key)
            else:
                presign = request_presign(filename)  # may raise
                upload_file_to_presigned_url(presign["url"], out_local)
                r2_key = presign["r2_key"]

            # notify controller job complete
            notify_controller_job_complete(job_id, r2_key)

            # update status
            job_status[job_id] = {"status": "completed", "result_r2_key": r2_key, "finished_at": datetime.utcnow().isoformat()}
            logger.info("Job %s completed; r2_key=%s", job_id, r2_key)
    except Exception as e:
        logger.exception("Job %s failed: %s", job_id, e)
        job_status[job_id] = {"status": "failed", "error": str(e), "failed_at": datetime.utcnow().isoformat()}

async def worker_loop():
    logger.info("Starting worker loop")
    while True:
        item = await job_queue.get()
        try:
            await process_job_item(item)
        finally:
            job_queue.task_done()

# -----------------------
# FastAPI endpoints
# -----------------------
@app.post("/run")
async def run_endpoint(req: RunRequest):
    """
    Controller or other callers post jobs here: {"job_id": "...", "prompt": "...", "params": {...}}
    Returns accepted immediately. Job processed in background.
    """
    job_id = req.job_id
    if job_id in job_status and job_status[job_id].get("status") in ("queued", "running"):
        raise HTTPException(status_code=409, detail="job already queued or running")
    await job_queue.put({"job_id": job_id, "prompt": req.prompt, "params": req.params or {}})
    job_status[job_id] = {"status": "queued", "queued_at": datetime.utcnow().isoformat()}
    logger.info("Enqueued job %s", job_id)
    return {"status": "accepted", "job_id": job_id}

@app.get("/health")
def health():
    return {"status": "ok", "worker_id": WORKER_ID, "model": MODEL_KEY, "queue_size": job_queue.qsize()}

@app.get("/status")
def status():
    return {"queue_size": job_queue.qsize(), "jobs": job_status}

# -----------------------
# Startup / background tasks
# -----------------------
@app.on_event("startup")
async def on_startup():
    # 1) load model (synchronous heavy work OK here)
    # TODO: replace this with actual model-loading (ComfyUI/diffusers/gguf loader)
    logger.info("Loading model (stub) for %s ...", MODEL_KEY)
    # If you want to actually load model into memory on startup, do it here synchronously,
    # so when billing starts the worker is ready to accept jobs.
    # Example: load into global MODEL var.
    await asyncio.sleep(0.1)

    # 2) register with controller (sync, but with retry)
    try:
        register_with_controller()
    except Exception as e:
        logger.exception("Failed to register with controller on startup: %s", e)
        # If registration fails permanently, we should exit to let the controller try again or fail
        raise

    # 3) start heartbeat loop
    async def hb_loop():
        while True:
            try:
                send_heartbeat()
            except Exception as e:
                logger.exception("Heartbeat failed: %s", e)
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    asyncio.create_task(hb_loop())

    # 4) start job processing loop
    asyncio.create_task(worker_loop())

# -----------------------
# Graceful shutdown hint (optional)
# -----------------------
@app.on_event("shutdown")
def on_shutdown():
    logger.info("Worker shutting down")
