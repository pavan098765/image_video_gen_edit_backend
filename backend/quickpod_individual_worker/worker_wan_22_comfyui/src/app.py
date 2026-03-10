#!/usr/bin/env python3
"""
app.py
======

FastAPI worker for Wan 2.2 I2V generation using ComfyUI as the inference
backend.  Follows the same architectural pattern as the existing diffusers-
based workers (job queue, heartbeat, controller registration, etc.) but
delegates actual inference to a headless ComfyUI server running inside the
same container.

Architecture:
    [Controller] -> POST /run -> [FastAPI :8000] -> POST /prompt -> [ComfyUI :8188]
                                                 <- poll /history
                                                 -> PUT upload -> [R2]
                                                 -> POST /job_complete -> [Controller]
"""

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
import httpx
import base64
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt
from contextlib import asynccontextmanager
import concurrent.futures

# -----------------------
# Dedicated Thread Pool for Inference
# -----------------------
# Only 1 worker to prevent multiple prompts from overwhelming the GPU.
# ComfyUI handles its own GPU management, but we still serialize jobs.
INFERENCE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="inference"
)

# IO-bound work (uploads, controller callbacks, HTTP)
IO_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="io"
)

# -----------------------
# Import inference function
# -----------------------
from comfyui_inference import run_inference_comfyui

# -----------------------
# Model Registry
# -----------------------
MODEL_REGISTRY = {
    "wan_22_comfyui": run_inference_comfyui,
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

MODEL_KEY = os.getenv("MODEL_KEY", "wan-22-comfyui-worker")
CONTROLLER_URL = os.getenv("CONTROLLER_URL", "").strip()
CONTROLLER_SHARED_SECRET = os.getenv("CONTROLLER_SHARED_SECRET", "")
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
# Job queue and state management
# -----------------------
job_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
job_status: Dict[str, Dict[str, Any]] = {}
job_status_lock = asyncio.Lock()
semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
last_model_key: Optional[str] = None
worker_ready = asyncio.Event()
startup_completed = False

# -----------------------
# ComfyUI readiness state
# -----------------------
_comfyui_ready = False
_comfyui_error: Optional[str] = None

# -----------------------
# Request Models
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
# Media URL Resolution
# -----------------------
# Fields that may contain URLs (presigned R2 GET URLs) instead of base64 data.
# At dispatch time the server converts R2 keys to presigned GET URLs.
# This helper downloads the binary and converts to base64 so inference code
# (which expects base64 strings) works without modification.
MEDIA_FIELDS = frozenset([
    "init_image", "mask_image", "image", "source_image", "face_image",
    "reference_image", "style_image", "control_image", "input_image",
    "ref_image", "ref_video", "input_video",
])


def _is_url(value: str) -> bool:
    return isinstance(value, str) and (value.startswith("http://") or value.startswith("https://"))


def _download_to_base64(url: str, timeout: int = 120) -> str:
    """Download binary from URL and return base64-encoded string."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return base64.b64encode(resp.content).decode("utf-8")


def resolve_media_urls(params: Dict[str, Any], job_id: str = "") -> Dict[str, Any]:
    """
    Resolve any URL-valued media fields in params to base64 strings.
    
    If a known media field contains a URL (e.g. presigned R2 download URL),
    download the binary and replace with base64 so downstream inference
    functions work unchanged.
    
    Handles both single string values and lists of strings.
    """
    resolved = dict(params)
    for field in MEDIA_FIELDS:
        value = resolved.get(field)
        if value is None:
            continue
        if isinstance(value, str) and _is_url(value):
            try:
                logger.info(f"[{job_id}] Downloading {field} from URL ({len(value)} chars)...")
                resolved[field] = _download_to_base64(value)
                logger.info(f"[{job_id}] Downloaded {field}: {len(resolved[field])} base64 chars")
            except Exception as e:
                logger.error(f"[{job_id}] Failed to download {field} from URL: {e}")
                raise
        elif isinstance(value, list):
            new_list = []
            for i, item in enumerate(value):
                if isinstance(item, str) and _is_url(item):
                    try:
                        logger.info(f"[{job_id}] Downloading {field}[{i}] from URL...")
                        new_list.append(_download_to_base64(item))
                    except Exception as e:
                        logger.error(f"[{job_id}] Failed to download {field}[{i}]: {e}")
                        raise
                else:
                    new_list.append(item)
            resolved[field] = new_list
    return resolved
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
    """Post signed JSON with retry logic."""
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
                logger.error(f"POST to {url} failed with {e.response.status_code}: {e}")
                raise
            last_error = e
            if attempt < max_retries - 1:
                backoff = 2 ** attempt
                logger.warning(f"POST to {url} server error (attempt {attempt + 1}/{max_retries}), retrying in {backoff}s...")
                time.sleep(backoff)
                continue
        except Exception as e:
            last_error = e
            logger.error(f"POST to {url} unexpected error: {e}")
            raise

    logger.error(f"POST to {url} failed after {max_retries} attempts: {last_error}")
    raise last_error

# -----------------------
# Controller integration
# -----------------------
@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(6))
def register_with_controller():
    if not CONTROLLER_URL:
        logger.info("No CONTROLLER_URL configured; skipping registration.")
        return {"skipped": True}

    register_payload = {
        "worker_id": WORKER_ID,
        "models": list(MODEL_REGISTRY.keys()),
        "endpoint": f"http://{PUBLIC_IPADDR}:{PORT}",
        "container_id": POD_UUID,
        "pod_uuid": POD_UUID,
        "public_ip": PUBLIC_IPADDR,
        "info": {
            "started_at": datetime.utcnow().isoformat(),
            "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
            "queue_capacity": job_queue.maxsize,
        },
    }
    url = f"{CONTROLLER_URL.rstrip('/')}/register_worker"

    logger.info(f"Registering worker with controller at {CONTROLLER_URL}")
    logger.info(f"Worker endpoint: http://{PUBLIC_IPADDR}:{PORT}")
    logger.info(f"Worker ID: {WORKER_ID}")

    try:
        resp = post_signed_json(url, register_payload, timeout=15, max_retries=6)
        result = resp.json()
        logger.info(f"Worker registered successfully: {result}")
        return result
    except Exception as e:
        logger.error(f"Worker registration failed: {e}")
        raise

@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(3))
def send_heartbeat():
    if not CONTROLLER_URL:
        return {"skipped": True}

    try:
        queue_size = job_queue.qsize()
        active_jobs = sum(1 for j in job_status.values() if j.get("status") == "running")
    except Exception:
        queue_size = 0
        active_jobs = 0

    if not _comfyui_ready:
        status = "starting"
    elif startup_completed:
        status = "busy" if (active_jobs > 0 or queue_size > 0) else "ready"
    else:
        status = "starting"

    payload = {
        "worker_id": WORKER_ID,
        "pod_uuid": POD_UUID,
        "timestamp": int(time.time()),
        "status": status,
        "model_loaded": _comfyui_ready,
        "metrics": {
            "queue_size": queue_size,
            "active_jobs": active_jobs,
            "max_concurrent": MAX_CONCURRENT_JOBS,
        }
    }

    url = f"{CONTROLLER_URL.rstrip('/')}/heartbeat"

    try:
        resp = post_signed_json(url, payload, timeout=8, max_retries=3)
        return resp.json()
    except Exception as e:
        logger.debug(f"Heartbeat failed: {e}")
        return {"error": str(e)}

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
    """Run inference in a dedicated thread pool to avoid blocking FastAPI."""
    loop = asyncio.get_event_loop()
    params = params.copy()
    params.setdefault("output_dir", WORKSPACE_DIR)
    if mode:
        params.setdefault("mode", mode)

    inference_func = resolve_inference_callable(model, mode)

    try:
        out_path = await loop.run_in_executor(
            INFERENCE_EXECUTOR, inference_func, job_id, prompt, params
        )
        return out_path
    except Exception:
        raise

# -----------------------
# Job Processing
# -----------------------
async def process_job_item(item: Dict[str, Any]):
    """Process a single job with comprehensive error handling."""
    job_id = item["job_id"]
    model = item["model"]
    mode = item.get("mode")
    prompt = item["prompt"]
    params = item.get("params", {})
    upload_url = item["upload_url"]
    result_key = item["result_key"]

    start_time = datetime.utcnow()
    logger.info(f"[{job_id}] Dequeued, processing with model={model}")

    # Set running immediately so pollers see progress (avoid appearing stuck as "queued")
    async with job_status_lock:
        job_status[job_id] = {
            "status": "running",
            "started_at": start_time.isoformat(),
            "model": model,
        }

    # Resolve any URL-valued media fields to base64 before inference
    try:
        params = resolve_media_urls(params, job_id)
    except Exception as e:
        logger.error(f"[{job_id}] Failed to resolve media URLs: {e}")
        # Will be caught by outer try/except and reported as failure

    try:
        async with semaphore:
            # Run inference
            logger.info(f"[{job_id}] Starting inference...")
            out_local = await run_inference_and_save(job_id, prompt, model, mode, params)
            logger.info(f"[{job_id}] Inference completed, output: {out_local}")

            # Upload result
            if upload_url.startswith("http://dummy-upload"):
                logger.info(f"[{job_id}] Skipping upload (dummy). File at {out_local}")
            else:
                logger.info(f"[{job_id}] Uploading result to R2...")
                upload_success = await _upload_with_retry(out_local, upload_url, job_id)
                if not upload_success:
                    raise Exception("Failed to upload result after retries")
                logger.info(f"[{job_id}] Upload completed successfully")

            # Notify controller
            is_autotest = job_id.startswith("autotest-")
            if CONTROLLER_URL and not is_autotest:
                logger.info(f"[{job_id}] Notifying controller of completion...")
                payload = {"job_id": job_id, "result_r2_key": result_key}
                url = f"{CONTROLLER_URL.rstrip('/')}/job_complete"
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        IO_EXECUTOR,
                        post_signed_json,
                        url,
                        payload,
                        10,
                        5,
                    )
                    logger.info(f"[{job_id}] Controller notified successfully")
                except Exception as notify_error:
                    logger.error(
                        f"[{job_id}] CRITICAL: Failed to notify controller: {notify_error}. "
                        f"Result at: {result_key}"
                    )
            elif is_autotest:
                logger.info(f"[{job_id}] Autotest job - skipping controller notification")

            # Mark completed
            completion_time = datetime.utcnow()
            processing_time = (completion_time - start_time).total_seconds()

            async with job_status_lock:
                job_status[job_id] = {
                    "status": "completed",
                    "result_r2_key": result_key,
                    "finished_at": completion_time.isoformat(),
                    "processing_time": processing_time,
                }

            logger.info(f"[{job_id}] Job completed successfully in {processing_time:.1f}s")
            asyncio.create_task(_cleanup_job_status(job_id, delay=30))

    except Exception as e:
        error_str = str(e)
        error_type = type(e).__name__
        completion_time = datetime.utcnow()
        processing_time = (completion_time - start_time).total_seconds()

        logger.exception(f"[{job_id}] Job failed after {processing_time:.1f}s: {error_type}: {error_str}")

        is_oom = "out of memory" in error_str.lower() or "oom" in error_str.lower()

        async with job_status_lock:
            job_status[job_id] = {
                "status": "failed",
                "error": error_str,
                "error_type": error_type,
                "is_oom": is_oom,
                "finished_at": completion_time.isoformat(),
                "processing_time": processing_time,
            }

        # Notify controller of failure
        is_autotest = job_id.startswith("autotest-")
        if CONTROLLER_URL and not is_autotest:
            payload = {
                "job_id": job_id,
                "result_r2_key": None,
                "failed": True,
                "error": error_str,
            }
            try:
                url = f"{CONTROLLER_URL.rstrip('/')}/job_complete"
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    IO_EXECUTOR,
                    post_signed_json,
                    url,
                    payload,
                    10,
                    5,
                )
                logger.info(f"[{job_id}] Failure notification sent to controller")
            except Exception as notify_error:
                logger.error(
                    f"[{job_id}] CRITICAL: Failed to notify controller of failure: {notify_error}"
                )

        asyncio.create_task(_cleanup_job_status(job_id, delay=60))


async def _upload_with_retry(file_path: str, upload_url: str, job_id: str, max_retries: int = 3) -> bool:
    loop = asyncio.get_running_loop()

    def _upload():
        with open(file_path, "rb") as f:
            resp = requests.put(upload_url, data=f, timeout=120)
            resp.raise_for_status()

    for attempt in range(max_retries):
        try:
            await loop.run_in_executor(IO_EXECUTOR, _upload)
            logger.info(f"[{job_id}] Upload succeeded ({attempt + 1}/{max_retries})")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                backoff = 2 ** attempt
                logger.warning(
                    f"[{job_id}] Upload failed ({attempt + 1}/{max_retries}): {e}, retrying in {backoff}s"
                )
                await asyncio.sleep(backoff)
            else:
                logger.error(f"[{job_id}] Upload failed after {max_retries} attempts: {e}")
                return False
    return False


async def _cleanup_job_status(job_id: str, delay: int = 30):
    await asyncio.sleep(delay)
    async with job_status_lock:
        if job_id in job_status:
            status = job_status[job_id].get("status")
            if status in ("completed", "failed"):
                del job_status[job_id]


async def worker_loop():
    """Main worker loop - processes jobs from the queue."""
    logger.info("Starting worker loop")
    consecutive_errors = 0
    max_consecutive_errors = 5

    while True:
        try:
            item = await job_queue.get()
            job_id = item.get("job_id", "unknown")

            try:
                await process_job_item(item)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                logger.exception(
                    f"[{job_id}] Critical error (consecutive: {consecutive_errors}/{max_consecutive_errors}): {e}"
                )
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(
                        f"Worker has {consecutive_errors} consecutive errors! Resetting..."
                    )
                    consecutive_errors = 0
            finally:
                job_queue.task_done()

        except Exception as outer_error:
            logger.exception(f"Critical error in worker loop: {outer_error}")
            await asyncio.sleep(1)


# -----------------------
# ComfyUI startup
# -----------------------
def _start_comfyui_sync():
    """Start ComfyUI server and wait for it to be ready (blocking)."""
    global _comfyui_ready, _comfyui_error

    try:
        import comfyui_bridge

        logger.info("Starting ComfyUI server...")
        comfyui_bridge.start_server()
        comfyui_bridge.wait_until_ready()

        # Pre-convert workflow to verify it loads correctly
        from comfyui_inference import _get_api_workflow
        _get_api_workflow()

        _comfyui_ready = True
        _comfyui_error = None
        logger.info("ComfyUI server is ready and workflow loaded.")

    except Exception as e:
        logger.exception(f"Failed to start ComfyUI: {e}")
        _comfyui_error = str(e)
        _comfyui_ready = False


async def _wait_for_comfyui_and_register():
    """Wait for ComfyUI to be ready, then register with controller."""
    global startup_completed

    max_wait = 600  # 10 minutes max for ComfyUI startup
    waited = 0

    while not _comfyui_ready and waited < max_wait:
        if _comfyui_error:
            logger.error(f"ComfyUI startup failed: {_comfyui_error}")
            break
        await asyncio.sleep(2)
        waited += 2
        if waited % 30 == 0:
            logger.info(f"Still waiting for ComfyUI... ({waited}s elapsed)")

    if not _comfyui_ready:
        logger.error(
            f"ComfyUI did not become ready within {max_wait}s. "
            f"Error: {_comfyui_error or 'timeout'}"
        )

    startup_completed = _comfyui_ready
    worker_ready.set()

    if startup_completed:
        logger.info("ComfyUI ready, worker is now READY")
    else:
        logger.warning("Worker starting but ComfyUI not fully ready")

    # Register with controller
    try:
        register_with_controller()
        logger.info("Worker registered with controller")
    except Exception as e:
        logger.exception("Failed to register worker: %s", e)


# -----------------------
# Lifespan
# -----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global startup_completed

    logger.info("Application starting up...")

    # Start background tasks
    asyncio.create_task(worker_loop())
    asyncio.create_task(_hb_loop())
    asyncio.create_task(_health_monitor_loop())

    startup_completed = False

    # Start ComfyUI in the inference executor thread
    loop = asyncio.get_event_loop()
    logger.info("Starting ComfyUI in INFERENCE_EXECUTOR thread...")
    loop.run_in_executor(INFERENCE_EXECUTOR, _start_comfyui_sync)

    # Wait for ComfyUI and register in background
    asyncio.create_task(_wait_for_comfyui_and_register())

    logger.info("Worker starting - ComfyUI loading in background, health endpoint responding")

    yield

    # Graceful shutdown
    logger.info("Worker shutting down...")
    startup_completed = False

    if not job_queue.empty():
        logger.info(f"Waiting for {job_queue.qsize()} remaining jobs...")
        try:
            await asyncio.wait_for(job_queue.join(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(f"Shutdown timeout, {job_queue.qsize()} jobs may be incomplete")

    # Stop ComfyUI
    try:
        import comfyui_bridge
        comfyui_bridge.stop_server()
    except Exception as e:
        logger.warning(f"Error stopping ComfyUI: {e}")

    INFERENCE_EXECUTOR.shutdown(wait=True, cancel_futures=False)
    logger.info("Worker shutdown complete")


# -----------------------
# Health monitoring
# -----------------------
async def _health_monitor_loop():
    check_interval = 60
    logger.info("Starting health monitor loop")

    while True:
        try:
            await asyncio.sleep(check_interval)

            # Check ComfyUI health
            if _comfyui_ready:
                import comfyui_bridge
                if not comfyui_bridge.health_check():
                    logger.warning("ComfyUI health check failed! Server may be unresponsive.")

            # Check for stuck jobs
            async with job_status_lock:
                for jid, status in job_status.items():
                    if status.get("status") == "running":
                        started_at_str = status.get("started_at")
                        if started_at_str:
                            started_at = datetime.fromisoformat(started_at_str)
                            runtime = (datetime.utcnow() - started_at).total_seconds()
                            # Warn if job running > 15 minutes (video gen is slower)
                            if runtime > 900:
                                logger.warning(f"Job {jid} running for {runtime:.0f}s (>15min)")

            queue_size = job_queue.qsize()
            if queue_size > job_queue.maxsize * 0.8:
                logger.warning(f"Queue nearly full: {queue_size}/{job_queue.maxsize}")

        except Exception as e:
            logger.error(f"Health monitor error: {e}")


# -----------------------
# Heartbeat loop
# -----------------------
async def _hb_loop():
    logger.info("Starting heartbeat loop")
    while True:
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(IO_EXECUTOR, send_heartbeat)
        except Exception as e:
            logger.exception("Heartbeat failed: %s", e)
        await asyncio.sleep(HEARTBEAT_INTERVAL)


async def _record_job_status_later(job_id: str, model: str):
    try:
        async with job_status_lock:
            job_status[job_id] = {
                "status": "queued",
                "queued_at": datetime.utcnow().isoformat(),
                "model": model,
            }
    except Exception as e:
        logger.warning(f"[{job_id}] Failed to record job status: {e}")


# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title=f"QuickPod Worker ({MODEL_KEY})", lifespan=lifespan)

@app.post("/run")
async def run_endpoint(req: RunRequest):
    """Accept a new job for processing."""
    logger.info(f"[{req.job_id}] Received job request for model={req.model}")

    # Duplicate check
    try:
        await asyncio.wait_for(job_status_lock.acquire(), timeout=2.0)
        try:
            if req.job_id in job_status:
                current_status = job_status[req.job_id].get("status")
                if current_status in ("queued", "running"):
                    logger.warning(f"[{req.job_id}] Already {current_status}")
                    raise HTTPException(status_code=409, detail=f"Job already {current_status}")
                if current_status in ("completed", "failed"):
                    del job_status[req.job_id]
        finally:
            job_status_lock.release()
    except asyncio.TimeoutError:
        logger.warning(f"[{req.job_id}] Lock timeout, proceeding anyway")

    if job_queue.full():
        raise HTTPException(status_code=503, detail="Worker queue full")

    if not startup_completed:
        if not _comfyui_ready:
            raise HTTPException(status_code=503, detail="ComfyUI still loading")
        else:
            logger.warning(f"[{req.job_id}] Not fully ready, but ComfyUI loaded - accepting")

    if req.model not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")

    try:
        await job_queue.put(req.dict())
        asyncio.create_task(_record_job_status_later(req.job_id, req.model))

        logger.info(f"[{req.job_id}] Enqueued (queue size: {job_queue.qsize()})")
        return {
            "status": "accepted",
            "job_id": req.job_id,
            "queue_position": job_queue.qsize(),
            "worker_id": WORKER_ID,
        }
    except Exception as e:
        logger.exception(f"[{req.job_id}] Failed to enqueue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Non-blocking health check that ALWAYS responds quickly."""
    try:
        try:
            await asyncio.wait_for(job_status_lock.acquire(), timeout=1.0)
            try:
                active_jobs = sum(1 for j in job_status.values() if j.get("status") == "running")
                queued_jobs = sum(1 for j in job_status.values() if j.get("status") == "queued")
            finally:
                job_status_lock.release()
        except asyncio.TimeoutError:
            active_jobs = MAX_CONCURRENT_JOBS
            queued_jobs = job_queue.qsize()

        # Check ComfyUI health (non-blocking)
        comfyui_healthy = False
        if _comfyui_ready:
            try:
                import comfyui_bridge
                comfyui_healthy = comfyui_bridge.health_check()
            except Exception:
                comfyui_healthy = False

        if not _comfyui_ready:
            overall_status = "starting"
        elif not startup_completed:
            overall_status = "starting"
        elif not comfyui_healthy:
            overall_status = "unhealthy"
        elif job_queue.full():
            overall_status = "overloaded"
        elif active_jobs >= MAX_CONCURRENT_JOBS:
            overall_status = "busy"
        else:
            overall_status = "ok"

        return {
            "status": overall_status,
            "worker_id": WORKER_ID,
            "queue_size": job_queue.qsize(),
            "queue_capacity": job_queue.maxsize,
            "active_jobs": active_jobs,
            "queued_jobs": queued_jobs,
            "max_concurrent": MAX_CONCURRENT_JOBS,
            "model_loaded": _comfyui_ready,
            "model_error": _comfyui_error,
            "comfyui_healthy": comfyui_healthy,
            "startup_completed": startup_completed,
            "ready": startup_completed and not job_queue.full() and comfyui_healthy,
            "processing": active_jobs > 0,
            "healthy": True,
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "busy",
            "worker_id": WORKER_ID,
            "queue_size": 0,
            "queue_capacity": job_queue.maxsize,
            "active_jobs": 1,
            "healthy": True,
            "processing": True,
            "error": str(e),
        }


@app.get("/logs/comfyui")
async def get_comfyui_logs(tail: int = 200, format: str = "text"):
    """
    Return the last N lines of the ComfyUI server log (stdout/stderr).
    Use this to see what ComfyUI is actually processing instead of only 'queued'/'running'.

    Query params:
      tail: number of lines from the end (default 200, max 5000)
      format: 'text' (plain, default) or 'json' (array of lines)
    """
    try:
        import comfyui_bridge
        log_path = comfyui_bridge.get_comfyui_log_path()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ComfyUI bridge not available: {e}")

    if not os.path.isfile(log_path):
        return {"lines": [], "message": "Log file not created yet (ComfyUI may not have started)."} if format == "json" else "Log file not created yet (ComfyUI may not have started).\n"

    tail = max(1, min(5000, tail))
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read log: {e}")

    last = lines[-tail:] if len(lines) > tail else lines
    last = [ln.rstrip("\n\r") for ln in last]

    if format == "json":
        return {"lines": last, "total_lines": len(lines), "path": log_path}
    return "\n".join(last) + ("\n" if last else "")


@app.get("/status")
async def status():
    """Detailed status including all jobs."""
    async with job_status_lock:
        jobs_copy = job_status.copy()

    comfyui_healthy = False
    if _comfyui_ready:
        try:
            import comfyui_bridge
            comfyui_healthy = comfyui_bridge.health_check()
        except Exception:
            pass

    return {
        "queue_size": job_queue.qsize(),
        "queue_capacity": job_queue.maxsize,
        "jobs": jobs_copy,
        "worker_id": WORKER_ID,
        "models": list(MODEL_REGISTRY.keys()),
        "startup_completed": startup_completed,
        "comfyui_ready": _comfyui_ready,
        "comfyui_healthy": comfyui_healthy,
        "comfyui_error": _comfyui_error,
        "comfyui_log_url": "/logs/comfyui",
    }
