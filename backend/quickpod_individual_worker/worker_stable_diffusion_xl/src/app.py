import os
import time
import json
import logging
import asyncio
import hmac
import hashlib
import gc
from typing import Dict, Any, Optional
from datetime import datetime

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt

# -----------------------
# GPU cleanup
# -----------------------
def cleanup_gpu():
    """Free GPU memory completely (stateless safe mode)."""
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()

# -----------------------
# Import inference function
# -----------------------
from sdxl_I2I_inference import run_inference_sdxl_i2i
from sdxl_inpaint_inference import run_inference_sdxl_inpaint
from sdxl_T2I_inference import run_inference_sdxl_t2i

# -----------------------
# Model Registry
# -----------------------
MODEL_REGISTRY = {
    "sdxl_t2i": run_inference_sdxl_t2i,
    "sdxl_inpaint":run_inference_sdxl_inpaint,
    "sdxl_i2i":run_inference_sdxl_i2i
}

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")

# -----------------------
# Configuration from env
# -----------------------
CONTAINER_ID = os.getenv("CONTAINER_LABEL")
POD_UUID = CONTAINER_ID
PUBLIC_IPADDR = os.getenv("PUBLIC_IPADDR")
WORKER_ID = POD_UUID

MODEL_KEY = os.getenv("MODEL_KEY", "diffusers-outpaint-worker")
CONTROLLER_URL = os.getenv("CONTROLLER_URL")
CONTROLLER_SHARED_SECRET = os.getenv("CONTROLLER_SHARED_SECRET")
CONTROLLER_API_KEY = os.getenv("CONTROLLER_API_KEY", "")
WORKER_PUBLIC_URL = PUBLIC_IPADDR
INTERNAL_PORT = 8000
EXTERNAL_PORT = os.getenv(f"QUICKPOD_PORT_{INTERNAL_PORT}")
PORT = int(EXTERNAL_PORT) if EXTERNAL_PORT else INTERNAL_PORT

HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "20"))
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))

WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title=f"QuickPod Worker ({MODEL_KEY})")

# Job queue & status
job_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
job_status: Dict[str, Dict[str, Any]] = {}
semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
last_model_key: Optional[str] = None

# -----------------------
# Request Models
# -----------------------
class RunRequest(BaseModel):
    job_id: str
    prompt: str
    model: str
    mode: str | None = None   # not used here
    params: Dict[str, Any] = {}
    upload_url: str
    result_key: str

# -----------------------
# HMAC utils
# -----------------------
def sign_payload_bytes(payload_bytes: bytes) -> str:
    mac = hmac.new(
        CONTROLLER_SHARED_SECRET.encode("utf-8"), payload_bytes, hashlib.sha256
    )
    return "sha256=" + mac.hexdigest()

def signed_headers(payload: dict) -> Dict[str, str]:
    b = json.dumps(payload).encode("utf-8")
    sig = sign_payload_bytes(b)
    headers = {"X-Signature": sig, "Content-Type": "application/json"}
    if CONTROLLER_API_KEY:
        headers["Authorization"] = f"Bearer {CONTROLLER_API_KEY}"
    return headers

def post_signed_json(url: str, payload: dict, timeout: int = 10) -> requests.Response:
    headers = signed_headers(payload)
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp

# -----------------------
# Controller integration
# -----------------------
@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(6))
def register_with_controller():
    register_payload = {
        "worker_id": WORKER_ID,
        "models": list(MODEL_REGISTRY.keys()),
        "endpoint": f"http://{PUBLIC_IPADDR}:{PORT}",
        "container_id": POD_UUID,
        "pod_uuid": POD_UUID,
        "public_ip": PUBLIC_IPADDR,
        "info": {"started_at": datetime.utcnow().isoformat()},
    }
    url = f"{CONTROLLER_URL.rstrip('/')}/register_worker"
    logger.info("Registering worker with controller: %s", register_payload)
    resp = post_signed_json(url, register_payload, timeout=10)
    return resp.json()

@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(6))
def send_heartbeat():
    payload = {"worker_id": WORKER_ID, "pod_uuid": POD_UUID, "timestamp": int(time.time())}
    url = f"{CONTROLLER_URL.rstrip('/')}/heartbeat"
    resp = post_signed_json(url, payload, timeout=5)
    return resp.json()

# -----------------------
# Inference dispatcher
# -----------------------
def resolve_inference_callable(model: str, mode: Optional[str]):
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model}")
    return MODEL_REGISTRY[model]

async def run_inference_and_save(
    job_id: str, prompt: str, model: str, mode: Optional[str], params: Dict[str, Any]
) -> str:
    global last_model_key
    loop = asyncio.get_event_loop()
    params = params.copy()
    params.setdefault("output_dir", WORKSPACE_DIR)

    inference_func = resolve_inference_callable(model, mode)
    model_key = model

    if last_model_key and last_model_key != model_key:
        logger.info(
            f"Switching model {last_model_key} -> {model_key}, cleaning up GPU first"
        )
        cleanup_gpu()

    try:
        out_path = await loop.run_in_executor(
            None, inference_func, job_id, prompt, params
        )
        return out_path
    finally:
        last_model_key = model_key

# -----------------------
# Job Processing
# -----------------------
async def process_job_item(item: Dict[str, Any]):
    job_id = item["job_id"]
    model = item["model"]
    mode = item.get("mode")
    prompt = item["prompt"]
    params = item.get("params", {})
    upload_url = item["upload_url"]
    result_key = item["result_key"]

    logger.info(f"Processing job {job_id} with model={model}")
    job_status[job_id] = {
        "status": "running",
        "started_at": datetime.utcnow().isoformat(),
    }

    try:
        async with semaphore:
            out_local = await run_inference_and_save(
                job_id, prompt, model, mode, params
            )
            with open(out_local, "rb") as f:
                resp = requests.put(upload_url, data=f)
                resp.raise_for_status()

            payload = {"job_id": job_id, "result_r2_key": result_key}
            url = f"{CONTROLLER_URL.rstrip('/')}/job_complete"
            post_signed_json(url, payload)

            job_status[job_id] = {
                "status": "completed",
                "result_r2_key": result_key,
                "finished_at": datetime.utcnow().isoformat(),
            }
            logger.info("Job %s completed", job_id)
    except Exception as e:
        logger.exception("Job %s failed: %s", job_id, e)
        job_status[job_id] = {"status": "failed", "error": str(e)}
        payload = {
            "job_id": job_id,
            "result_r2_key": None,
            "failed": True,
            "error": str(e),
        }
        try:
            post_signed_json(f"{CONTROLLER_URL.rstrip('/')}/job_complete", payload)
        except Exception as e2:
            logger.error("Failed to notify controller of failure: %s", e2)

async def worker_loop():
    logger.info("Starting worker loop")
    while True:
        item = await job_queue.get()
        try:
            await process_job_item(item)
        finally:
            job_queue.task_done()

# -----------------------
# FastAPI Endpoints
# -----------------------
@app.post("/run")
async def run_endpoint(req: RunRequest):
    if req.job_id in job_status and job_status[req.job_id].get("status") in (
        "queued",
        "running",
    ):
        raise HTTPException(status_code=409, detail="job already queued or running")
    await job_queue.put(req.dict())
    job_status[req.job_id] = {
        "status": "queued",
        "queued_at": datetime.utcnow().isoformat(),
    }
    logger.info("Enqueued job %s", req.job_id)
    return {"status": "accepted", "job_id": req.job_id}

@app.get("/health")
def health():
    return {"status": "ok", "worker_id": WORKER_ID, "queue_size": job_queue.qsize()}

@app.get("/status")
def status():
    return {"queue_size": job_queue.qsize(), "jobs": job_status}

# -----------------------
# Startup / Shutdown
# -----------------------
@app.on_event("startup")
async def on_startup():
    logger.info("Worker starting in smart cleanup mode")
    try:
        register_with_controller()
    except Exception as e:
        logger.exception("Failed to register worker with controller: %s", e)
        raise

    async def hb_loop():
        while True:
            try:
                send_heartbeat()
            except Exception as e:
                logger.exception("Heartbeat failed: %s", e)
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    asyncio.create_task(hb_loop())
    asyncio.create_task(worker_loop())

@app.on_event("shutdown")
def on_shutdown():
    logger.info("Worker shutting down; final GPU cleanup")
    cleanup_gpu()
