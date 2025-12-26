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
import uuid
from contextlib import asynccontextmanager
import concurrent.futures

import httpx
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt

import torch

# -----------------------
# Dedicated Thread Pool for Inference
# -----------------------
# CRITICAL: Use a dedicated executor for inference to avoid blocking FastAPI
# This ensures /health and /run endpoints always respond, even during heavy GPU work
#
# IMPORTANT: Only 1 worker to prevent OOM on limited VRAM cards!
# Multiple simultaneous inferences will exhaust GPU memory.
INFERENCE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="inference"
)

# IO-bound work (uploads, controller callbacks, HTTP)
# MUST be separate so inference never blocks networking
IO_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="io"
)

# -----------------------
# GPU cleanup
# -----------------------
def cleanup_gpu(verbose: bool = False):
    """
    Fully frees GPU memory and CUDA context.
    Use this between model loads to ensure a clean slate.
    """
    try:
        if verbose:
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"[GPU] Before cleanup: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved")

        # 1️⃣ Delete all PyTorch GPU tensors
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    del obj
            except Exception:
                pass

        # 2️⃣ Run Python garbage collector
        gc.collect()

        # 3️⃣ Clear PyTorch cache and IPC handles
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # 4️⃣ Synchronize and reset memory stats
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        # 5️⃣ Final garbage collection (torch.cuda.reset() doesn't exist!)
        gc.collect()
        
        # 6️⃣ One more empty_cache after GC
        torch.cuda.empty_cache()

        if verbose:
            print(f"[GPU] After cleanup: {torch.cuda.memory_allocated()/1024**2:.2f} MB allocated")
            print("[GPU] GPU memory cleanup completed.")
    except Exception as e:
        print(f"[GPU] Cleanup warning: {e}")
        pass

# -----------------------
# Import inference functions
# -----------------------
from fastwan_1b_inference import (
    run_inference_wan_1b_t2i,
    run_inference_wan_1b_t2v,
)

# -----------------------
# Model Registry (FastWAN only)
# -----------------------
MODEL_REGISTRY = {
    "fastwan": {
        "t2i": run_inference_wan_1b_t2i,
        "t2v": run_inference_wan_1b_t2v,
    }
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

MODEL_KEY = os.getenv("MODEL_KEY", "fastwan-worker")
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
# Job queue
# -----------------------
job_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
job_status: Dict[str, Dict[str, Any]] = {}
semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
last_model_key: Optional[str] = None

# -----------------------
# Request Model
# -----------------------
class RunRequest(BaseModel):
    job_id: str
    prompt: str
    model: str
    mode: str | None = None
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

def post_signed_json(url: str, payload: dict, timeout: int = 10, max_retries: int = 3) -> requests.Response:
    """
    Post signed JSON with retry logic for resilience.
    """
    headers = signed_headers(payload)
    last_error = None
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < max_retries - 1:
                backoff = 2 ** attempt
                logger.warning(f"POST to {url} timeout (attempt {attempt + 1}/{max_retries}), retrying in {backoff}s...")
                time.sleep(backoff)
                continue
        except requests.exceptions.ConnectionError as e:
            last_error = e
            if attempt < max_retries - 1:
                backoff = 2 ** attempt
                logger.warning(f"POST to {url} connection error (attempt {attempt + 1}/{max_retries}), retrying in {backoff}s...")
                time.sleep(backoff)
                continue
        except requests.exceptions.HTTPError as e:
            if e.response and 400 <= e.response.status_code < 500:
                raise
            last_error = e
            if attempt < max_retries - 1:
                backoff = 2 ** attempt
                logger.warning(f"POST to {url} server error (attempt {attempt + 1}/{max_retries}), retrying in {backoff}s...")
                time.sleep(backoff)
                continue
        except Exception as e:
            last_error = e
            raise
    
    logger.error(f"POST to {url} failed after {max_retries} attempts: {last_error}")
    raise last_error

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
    payload = {
        "worker_id": WORKER_ID,
        "pod_uuid": POD_UUID,
        "timestamp": int(time.time()),
    }
    url = f"{CONTROLLER_URL.rstrip('/')}/heartbeat"
    resp = post_signed_json(url, payload, timeout=5)
    return resp.json()

# -----------------------
# Inference dispatcher
# -----------------------
def resolve_inference_callable(model: str, mode: Optional[str]):
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model}")
    inference_func = MODEL_REGISTRY[model]
    if isinstance(inference_func, dict):
        if not mode or mode not in inference_func:
            raise ValueError(
                f"Mode required for {model}. Available: {list(inference_func.keys())}"
            )
        inference_func = inference_func[mode]
    return inference_func

async def run_inference_and_save(
    job_id: str, prompt: str, model: str, mode: Optional[str], params: Dict[str, Any]
) -> str:
    """
    Run inference in a dedicated thread pool to avoid blocking FastAPI endpoints.
    
    CRITICAL: Uses INFERENCE_EXECUTOR (dedicated pool) instead of the default executor.
    This ensures /health and /run endpoints remain responsive during heavy GPU work.
    """
    global last_model_key
    loop = asyncio.get_event_loop()
    params = params.copy()
    params.setdefault("output_dir", WORKSPACE_DIR)

    inference_func = resolve_inference_callable(model, mode)
    model_key = f"{model}:{mode or ''}"

    if last_model_key and last_model_key != model_key:
        logger.info(
            f"Switching model {last_model_key} -> {model_key}, cleaning up GPU first"
        )
        cleanup_gpu()

    try:
        # CRITICAL: Use dedicated INFERENCE_EXECUTOR to avoid blocking FastAPI event loop
        out_path = await loop.run_in_executor(
            INFERENCE_EXECUTOR, inference_func, job_id, prompt, params
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

    logger.info(f"Processing job {job_id} with model={model}, mode={mode}")
    job_status[job_id] = {
        "status": "running",
        "started_at": datetime.utcnow().isoformat(),
    }

    try:
        async with semaphore:
            out_local = await run_inference_and_save(
                job_id, prompt, model, mode, params
            )
            if upload_url.startswith("http://dummy-upload"):
                logger.info("Skipping upload for dummy URL")
            else:
                with open(out_local, "rb") as f:
                    resp = requests.put(upload_url, data=f)
                    resp.raise_for_status()

            payload = {"job_id": job_id, "result_r2_key": result_key}
            url = f"{CONTROLLER_URL.rstrip('/')}/job_complete"
            post_signed_json(url, payload, timeout=30)

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
            "result_r2_key": "None",
            "failed": True,
            # "error": str(e), <- currently not accepted by controller, but will add later
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
# Auto-test
# -----------------------
async def run_autotest():
    await asyncio.sleep(5)  # short delay to let server settle
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            # --- T2I test ---
            t2i_job_id = f"autotest-t2i-{uuid.uuid4()}"
            t2i_params = {
                "steps": 10,
                "guidance_scale": 0.0,
                "seed": 42,
                "randomize_seed": True,
                "height": 896,
                "width": 896,
                "negative_prompt": "blurry, distorted, lowres",
                "output_dir": "./outputs",
            }
            logger.info(f"🔥 Running T2I auto-test job {t2i_job_id}")
            resp = await client.post(
                "http://127.0.0.1:8000/run",
                json={
                    "job_id": t2i_job_id,
                    "prompt": "A cinematic portrait of a woman, ultra realistic, 8k lighting",
                    "model": "fastwan",
                    "mode": "t2i",
                    "params": t2i_params,
                    "upload_url": "http://dummy-upload/skip",
                    "result_key": f"{t2i_job_id}.png",        
                },
            )
            if resp.status_code == 200:
                logger.info(f"🔥 T2I auto-test success: {resp.json()}")
            else:
                logger.error(f"🔥 T2I auto-test failed: {resp.status_code} {resp.text}")

            # --- T2V test ---
            t2v_job_id = f"autotest-t2v-{uuid.uuid4()}"
            t2v_params = {
                "steps": 8,
                "guidance_scale": 0.0,
                "seed": 42,
                "randomize_seed": True,
                "height": 896,
                "width": 896,
                "duration_seconds": 5,
                "negative_prompt": "blurry, distorted, lowres",
                "output_dir": "./outputs",
            }
            logger.info(f"🔥 Running T2V auto-test job {t2v_job_id}")
            resp = await client.post(
                "http://127.0.0.1:8000/run",
                json={
                    "job_id": t2v_job_id,
                    "prompt": "A cinematic shot of a beach at sunset filled with people, smooth motion",
                    "model": "fastwan",
                    "mode": "t2v",
                    "params": t2v_params,
                    "upload_url": "http://dummy-upload/skip",   
                    "result_key": f"{t2v_job_id}.png",         
                },
            )
            if resp.status_code == 200:
                logger.info(f"🔥 T2V auto-test success: {resp.json()}")
            else:
                logger.error(f"🔥 T2V auto-test failed: {resp.status_code} {resp.text}")

    except Exception as e:
        logger.error(f"🔥 Auto-test exception: {e}")

# -----------------------
# Lifespan
# -----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # uvicorn will bind sockets first, then this runs
    logger.info("✅ Application fully started, registering worker now")

    try:
        register_with_controller()
    except Exception as e:
        logger.exception("Failed to register worker with controller: %s", e)
        raise

    asyncio.create_task(worker_loop())
    asyncio.create_task(_hb_loop())

    if os.getenv("ENABLE_AUTOTEST", "1") == "1":
        asyncio.create_task(run_autotest())

    yield  # server runs until shutdown

    logger.info("Worker shutting down; final GPU cleanup")
    cleanup_gpu()

# -----------------------
# Heartbeat loop
# -----------------------
async def _hb_loop():
    """Heartbeat loop - uses IO_EXECUTOR to avoid blocking."""
    loop = asyncio.get_event_loop()
    while True:
        try:
            await loop.run_in_executor(IO_EXECUTOR, send_heartbeat)
        except Exception as e:
            logger.debug(f"Heartbeat failed: {e}")
        await asyncio.sleep(HEARTBEAT_INTERVAL)

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title=f"QuickPod Worker ({MODEL_KEY})", lifespan=lifespan)

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
