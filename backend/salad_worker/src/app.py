# salad_worker/app.py
import os
import time
import uuid
import json
import logging
import asyncio
import hmac
import hashlib
from typing import Dict, Any
from datetime import datetime

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")

# Configuration from env
WORKER_ID = os.getenv("WORKER_ID", str(uuid.uuid4()))
MODEL_KEY = os.getenv("MODEL_KEY", os.getenv("MODEL", "qwen-image"))
CONTROLLER_URL = os.getenv("CONTROLLER_URL")  # Required
CONTROLLER_SHARED_SECRET = os.getenv("CONTROLLER_SHARED_SECRET")
CONTROLLER_API_KEY = os.getenv("CONTROLLER_API_KEY", "")
WORKER_PUBLIC_URL = os.getenv("WORKER_PUBLIC_URL")
PORT = int(os.getenv("PORT", "8080"))

HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "20"))
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))

WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# Validation
if not CONTROLLER_URL:
    raise RuntimeError("CONTROLLER_URL missing")
if not CONTROLLER_SHARED_SECRET:
    raise RuntimeError("CONTROLLER_SHARED_SECRET missing")

# FastAPI app
app = FastAPI(title=f"Salad Worker ({MODEL_KEY})")

# Job queue & status
job_queue: asyncio.Queue = asyncio.Queue()
job_status: Dict[str, Dict[str, Any]] = {}
running_jobs_tasks: Dict[str, asyncio.Task] = {}
semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
shutdown_tasks: list = []

# Pydantic models
class RunRequest(BaseModel):
    job_id: str
    prompt: str
    params: Dict[str, Any] = {}

# -----------------------
# HMAC utilities
# -----------------------
def sign_payload_bytes(payload_bytes: bytes) -> str:
    mac = hmac.new(CONTROLLER_SHARED_SECRET.encode("utf-8"), payload_bytes, hashlib.sha256)
    return "sha256=" + mac.hexdigest()

def post_signed_json(url: str, payload: dict, timeout: int = 10) -> requests.Response:
    b = json.dumps(payload).encode("utf-8")
    sig = sign_payload_bytes(b)
    headers = {"X-Signature": sig, "Content-Type": "application/json"}
    if CONTROLLER_API_KEY:
        headers["Authorization"] = f"Bearer {CONTROLLER_API_KEY}"
    resp = requests.post(url, data=b, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp

# -----------------------
# Controller registration
# -----------------------
@retry(wait=wait_exponential(multiplier=1, min=2, max=20),
       stop=stop_after_attempt(6),
       retry=retry_if_exception_type(Exception))
def register_with_controller():
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

@retry(wait=wait_exponential(multiplier=1, min=2, max=20),
       stop=stop_after_attempt(6),
       retry=retry_if_exception_type(Exception))
def send_heartbeat():
    payload = {
        "worker_id": WORKER_ID,
        "timestamp": int(time.time()),
        "running_jobs": [
            {"job_id": jid, "status": info["status"]}
            for jid, info in job_status.items() if info["status"] == "running"
        ]
    }
    url = f"{CONTROLLER_URL.rstrip('/')}/heartbeat"
    b = json.dumps(payload).encode("utf-8")
    sig = sign_payload_bytes(b)
    headers = {"X-Signature": sig, "Content-Type": "application/json"}
    r = requests.post(url, data=b, headers=headers, timeout=5)
    r.raise_for_status()
    return r.json()

# -----------------------
# Presign/upload helpers
# -----------------------
@retry(wait=wait_exponential(multiplier=1, min=2, max=20),
       stop=stop_after_attempt(5),
       retry=retry_if_exception_type(Exception))
def request_presign(filename: str, expiration: int = 3600) -> Dict[str, str]:
    url = f"{CONTROLLER_URL.rstrip('/')}/presign_upload?filename={filename}&expiration={expiration}"
    headers = {}
    if CONTROLLER_API_KEY:
        headers["Authorization"] = f"Bearer {CONTROLLER_API_KEY}"
    logger.info("Requesting presign for %s", filename)
    r = requests.post(url, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()

@retry(wait=wait_exponential(multiplier=1, min=2, max=20),
       stop=stop_after_attempt(5),
       retry=retry_if_exception_type(Exception))
def upload_file_to_presigned_url(presigned_url: str, local_path: str):
    logger.info("Uploading %s to presigned URL", local_path)
    with open(local_path, "rb") as f:
        r = requests.put(presigned_url, data=f, timeout=300)
    if not (200 <= r.status_code < 300):
        raise RuntimeError(f"Failed presigned upload: {r.status_code} {r.text}")
    logger.info("Upload OK (presigned)")

@retry(wait=wait_exponential(multiplier=1, min=2, max=20),
       stop=stop_after_attempt(6),
       retry=retry_if_exception_type(Exception))
def notify_controller_job_complete(job_id: str, r2_key: str = "", failed: bool = False):
    payload = {"job_id": job_id, "result_r2_key": r2_key if not failed else None, "failed": failed}
    url = f"{CONTROLLER_URL.rstrip('/')}/job_complete"
    logger.info("Notifying controller job_complete: %s (failed=%s)", job_id, failed)
    resp = post_signed_json(url, payload, timeout=10)
    logger.info("Controller response for job %s: %s", job_id, resp.text)
    return resp.json()

# -----------------------
# Inference placeholder
# -----------------------
async def run_inference_and_save(prompt: str, params: Dict[str, Any], job_id: str) -> str:
    logger.info("RUN inference stub: job=%s prompt=%.60s", job_id, prompt[:60])
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

    task = asyncio.current_task()
    running_jobs_tasks[job_id] = task

    try:
        async with semaphore:
            out_local = await run_inference_and_save(prompt, params, job_id)
            filename = os.path.basename(out_local)

            # Upload via presign only
            presign = request_presign(filename)
            upload_file_to_presigned_url(presign["url"], out_local)
            r2_key = presign["r2_key"]

            notify_controller_job_complete(job_id, r2_key)

            job_status[job_id] = {"status": "completed", "result_r2_key": r2_key,
                                  "finished_at": datetime.utcnow().isoformat()}
            logger.info("Job %s completed; r2_key=%s", job_id, r2_key)

    except asyncio.CancelledError:
        logger.warning("Job %s cancelled during shutdown", job_id)
        job_status[job_id] = {"status": "cancelled", "cancelled_at": datetime.utcnow().isoformat()}
        try:
            notify_controller_job_complete(job_id, r2_key="", failed=True)
        except Exception as notify_err:
            logger.exception("Failed to notify controller of cancelled job: %s", notify_err)
        raise
    except Exception as e:
        logger.exception("Job %s failed: %s", job_id, e)
        job_status[job_id] = {"status": "failed", "error": str(e),
                              "failed_at": datetime.utcnow().isoformat()}
        try:
            notify_controller_job_complete(job_id, r2_key="", failed=True)
        except Exception as notify_err:
            logger.exception("Failed to notify controller of job failure: %s", notify_err)
    finally:
        running_jobs_tasks.pop(job_id, None)

async def worker_loop():
    logger.info("Starting worker loop")
    try:
        while True:
            item = await job_queue.get()
            if item:
                try:
                    await process_job_item(item)
                finally:
                    job_queue.task_done()
    except asyncio.CancelledError:
        logger.info("worker_loop cancelled, cleaning up...")

# -----------------------
# FastAPI endpoints
# -----------------------
@app.post("/run")
async def run_endpoint(req: RunRequest):
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
    global shutdown_tasks
    shutdown_tasks = []

    logger.info("Loading model (stub) for %s ...", MODEL_KEY)
    await asyncio.sleep(0.1)

    try:
        register_with_controller()
    except Exception as e:
        logger.exception("Failed to register with controller on startup: %s", e)
        raise

    # Heartbeat loop
    async def hb_loop():
        while True:
            try:
                send_heartbeat()
            except Exception as e:
                logger.exception("Heartbeat failed: %s", e)
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    hb_task = asyncio.create_task(hb_loop())
    worker_task = asyncio.create_task(worker_loop())
    shutdown_tasks.extend([hb_task, worker_task])

# -----------------------
# Graceful shutdown
# -----------------------
@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Worker shutting down - cancelling background tasks and in-progress jobs...")

    # Cancel in-progress job tasks
    for job_id, task in list(running_jobs_tasks.items()):
        logger.info("Cancelling in-progress job %s", job_id)
        task.cancel()
    running_jobs_tasks.clear()

    # Cancel background tasks
    for task in shutdown_tasks:
        task.cancel()

    # Wait briefly for tasks to exit
    await asyncio.sleep(0.1)
    logger.info("All background tasks and in-progress jobs cancelled")
