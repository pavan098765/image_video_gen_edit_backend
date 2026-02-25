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

import torch

# -----------------------
# Dedicated Thread Pool for Inference
# -----------------------
INFERENCE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="inference"
)

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
    """
    try:
        if verbose:
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"[GPU] Before cleanup: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved")

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    del obj
            except Exception:
                pass

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()

        if verbose:
            print(f"[GPU] After cleanup: {torch.cuda.memory_allocated()/1024**2:.2f} MB allocated")
            print("[GPU] GPU memory cleanup completed.")
    except Exception as e:
        print(f"[GPU] Cleanup warning: {e}")
        pass

# -----------------------
# Import inference function
# -----------------------
from qwen_image_edit_inference import run_inference_qwen_image_edit

# -----------------------
# Model Registry
# -----------------------
MODEL_REGISTRY = {
    "qwen_image_edit": run_inference_qwen_image_edit,
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

MODEL_KEY = os.getenv("MODEL_KEY", "qwen-edit-worker")
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
job_queue: asyncio.Queue = asyncio.Queue(maxsize=30)
job_status: Dict[str, Dict[str, Any]] = {}
job_status_lock = asyncio.Lock()
semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
last_model_key: Optional[str] = None
worker_ready = asyncio.Event()
startup_completed = False


def _peek_next_job_id() -> Optional[str]:
    """Peek at the next job in the queue without removing it."""
    try:
        if not job_queue.empty():
            return job_queue._queue[0].get("job_id")
    except (IndexError, AttributeError):
        pass
    return None

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
    
    try:
        resp = post_signed_json(url, register_payload, timeout=15, max_retries=6)
        result = resp.json()
        logger.info(f"✅ Worker registered successfully: {result}")
        return result
    except Exception as e:
        logger.error(f"❌ Worker registration failed: {e}")
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
    
    # Determine actual worker status based on job state
    if not startup_completed:
        status = "starting"
    else:
        status = "busy" if (active_jobs > 0 or queue_size > 0) else "ready"
    
    payload = {
        "worker_id": WORKER_ID,
        "pod_uuid": POD_UUID,
        "timestamp": int(time.time()),
        "status": status,
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
    global last_model_key
    loop = asyncio.get_event_loop()
    params = params.copy()
    params.setdefault("output_dir", WORKSPACE_DIR)

    inference_func = resolve_inference_callable(model, mode)
    model_key = model

    if last_model_key and last_model_key != model_key:
        logger.info(f"Switching model {last_model_key} -> {model_key}, cleaning up GPU first")
        cleanup_gpu()

    try:
        out_path = await loop.run_in_executor(INFERENCE_EXECUTOR, inference_func, job_id, prompt, params)
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
    
    start_time = datetime.utcnow()
    logger.info(f"[{job_id}] Processing job with model={model}")
    # Resolve any URL-valued media fields to base64 before inference
    try:
        params = resolve_media_urls(params, job_id)
    except Exception as e:
        logger.error(f"[{job_id}] Failed to resolve media URLs: {e}")
        # Will be caught by outer try/except and reported as failure
    
    async with job_status_lock:
        job_status[job_id] = {
            "status": "running", 
            "started_at": start_time.isoformat(),
            "model": model,
        }

    try:
        async with semaphore:
            out_local = await run_inference_and_save(job_id, prompt, model, mode, params)
            logger.info(f"[{job_id}] Inference completed, output: {out_local}")

            if upload_url.startswith("http://dummy-upload"):
                logger.info(f"[{job_id}] Skipping upload (dummy)")
            else:
                upload_success = await _upload_with_retry(out_local, upload_url, job_id)
                if not upload_success:
                    raise Exception("Failed to upload result after retries")

            is_autotest = job_id.startswith("autotest-")
            if CONTROLLER_URL and not is_autotest:
                payload = {"job_id": job_id, "result_r2_key": result_key, "next_job_id": _peek_next_job_id()}
                url = f"{CONTROLLER_URL.rstrip('/')}/job_complete"
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(IO_EXECUTOR, post_signed_json, url, payload, 10, 5)
                except Exception as notify_error:
                    logger.error(f"[{job_id}] Failed to notify controller: {notify_error}")

            completion_time = datetime.utcnow()
            processing_time = (completion_time - start_time).total_seconds()
            
            async with job_status_lock:
                job_status[job_id] = {
                    "status": "completed",
                    "result_r2_key": result_key,
                    "finished_at": completion_time.isoformat(),
                    "processing_time": processing_time,
                }
            
            logger.info(f"[{job_id}] Job completed in {processing_time:.1f}s")
            asyncio.create_task(_cleanup_job_status(job_id, delay=30))
            _check_gpu_memory(job_id)
            
    except Exception as e:
        error_str = str(e)
        completion_time = datetime.utcnow()
        processing_time = (completion_time - start_time).total_seconds()
        
        logger.exception(f"[{job_id}] Job failed: {error_str}")
        
        is_oom = "out of memory" in error_str.lower() or "oom" in error_str.lower()
        if is_oom:
            cleanup_gpu(verbose=True)
        
        async with job_status_lock:
            job_status[job_id] = {"status": "failed", "error": error_str, "is_oom": is_oom}
        
        is_autotest = job_id.startswith("autotest-")
        if CONTROLLER_URL and not is_autotest:
            payload = {"job_id": job_id, "result_r2_key": None, "failed": True, "error": error_str, "next_job_id": _peek_next_job_id()}
            try:
                url = f"{CONTROLLER_URL.rstrip('/')}/job_complete"
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(IO_EXECUTOR, post_signed_json, url, payload, 10, 5)
            except Exception:
                pass
        
        asyncio.create_task(_cleanup_job_status(job_id, delay=60))

async def _upload_with_retry(file_path: str, upload_url: str, job_id: str, max_retries: int = 3) -> bool:
    loop = asyncio.get_running_loop()

    def _upload():
        with open(file_path, "rb") as f:
            resp = requests.put(upload_url, data=f, timeout=60)
            resp.raise_for_status()

    for attempt in range(max_retries):
        try:
            await loop.run_in_executor(IO_EXECUTOR, _upload)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                logger.error(f"[{job_id}] Upload failed after {max_retries} attempts: {e}")
                return False
    return False

def _check_gpu_memory(job_id: str):
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            if allocated > 20:
                logger.warning(f"[{job_id}] High GPU memory: {allocated:.2f}GB")
    except Exception:
        pass

async def _cleanup_job_status(job_id: str, delay: int = 30):
    await asyncio.sleep(delay)
    async with job_status_lock:
        if job_id in job_status and job_status[job_id].get("status") in ("completed", "failed"):
            del job_status[job_id]

async def worker_loop():
    logger.info("Starting worker loop")
    consecutive_errors = 0
    
    while True:
        try:
            item = await job_queue.get()
            try:
                await process_job_item(item)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                logger.exception(f"Critical error: {e}")
                if consecutive_errors >= 5:
                    cleanup_gpu(verbose=True)
                    consecutive_errors = 0
            finally:
                job_queue.task_done()
        except Exception as outer_error:
            logger.exception(f"Worker loop error: {outer_error}")
            await asyncio.sleep(1)

# -----------------------
# Auto-test
# -----------------------
async def run_autotest():
    await asyncio.sleep(5)
    logger.info("🚀 Starting Qwen Image Edit auto-test")

    img_url = "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/Snapinsta.app_262322643_4937545659591345_7487236998017046274_n_1080.jpg"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "image/*,*/*;q=0.8"}

    try:
        async with httpx.AsyncClient(headers=headers, timeout=60.0) as client:
            resp = await client.get(img_url)
            resp.raise_for_status()
            img_bytes = resp.content
    except Exception as e:
        logger.error(f"❌ Failed to fetch test image: {e}")
        return

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    test_job_id = f"autotest-qwen-edit-{uuid.uuid4()}"
    params = {
        "cfg_scale": 1.0,
        "output_dir": "./outputs",
        "init_image": img_b64,
        "num_inference_steps": 4,
        "scale_image": False,
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            "http://127.0.0.1:8000/run",
            json={
                "job_id": test_job_id,
                "prompt": "Change body orientation to face the camera directly",
                "model": "qwen_image_edit",
                "params": params,
                "upload_url": "http://dummy-upload/skip",
                "result_key": f"{test_job_id}.png",
            },
        )
        logger.info(f"🔥 Auto-test response: {resp.status_code} {resp.text}")

# -----------------------
# Health monitoring
# -----------------------
async def _health_monitor_loop():
    while True:
        try:
            await asyncio.sleep(60)
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                if allocated > 22:
                    logger.warning(f"⚠️ HIGH GPU MEMORY: {allocated:.2f}GB")
        except Exception as e:
            logger.error(f"Health monitor error: {e}")

async def _hb_loop():
    while True:
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(IO_EXECUTOR, send_heartbeat)
        except Exception as e:
            logger.debug(f"Heartbeat failed: {e}")
        await asyncio.sleep(HEARTBEAT_INTERVAL)

async def _record_job_status_later(job_id: str, model: str):
    try:
        async with job_status_lock:
            job_status[job_id] = {"status": "queued", "queued_at": datetime.utcnow().isoformat(), "model": model}
    except Exception:
        pass

# -----------------------
# Lifespan
# -----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global startup_completed
    
    logger.info("🚀 Application starting up...")
    
    asyncio.create_task(worker_loop())
    asyncio.create_task(_hb_loop())
    asyncio.create_task(_health_monitor_loop())
    
    startup_completed = True
    worker_ready.set()
    
    try:
        register_with_controller()
    except Exception as e:
        logger.exception("Failed to register: %s", e)

    if os.getenv("ENABLE_AUTOTEST", "1") == "1":
        asyncio.create_task(run_autotest())
    
    yield
    
    logger.info("🛑 Shutting down...")
    startup_completed = False
    
    if not job_queue.empty():
        try:
            await asyncio.wait_for(job_queue.join(), timeout=30.0)
        except asyncio.TimeoutError:
            pass
    
    INFERENCE_EXECUTOR.shutdown(wait=True, cancel_futures=False)
    IO_EXECUTOR.shutdown(wait=False)
    cleanup_gpu()
    logger.info("✅ Shutdown complete")

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title=f"QuickPod Worker ({MODEL_KEY})", lifespan=lifespan)

@app.post("/run")
async def run_endpoint(req: RunRequest):
    try:
        await asyncio.wait_for(job_status_lock.acquire(), timeout=2.0)
        try:
            if req.job_id in job_status and job_status[req.job_id].get("status") in ("queued", "running"):
                raise HTTPException(status_code=409, detail="Job already queued or running")
            if req.job_id in job_status:
                del job_status[req.job_id]
        finally:
            job_status_lock.release()
    except asyncio.TimeoutError:
        pass
    
    if job_queue.full():
        raise HTTPException(status_code=503, detail="Queue full")
    
    if not startup_completed:
        raise HTTPException(status_code=503, detail="Worker not ready")
    
    if req.model not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")
    
    await job_queue.put(req.dict())
    asyncio.create_task(_record_job_status_later(req.job_id, req.model))
    logger.info(f"Enqueued job {req.job_id}")
    return {"status": "accepted", "job_id": req.job_id, "queue_position": job_queue.qsize(), "worker_id": WORKER_ID}

@app.get("/health")
async def health():
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
        
        if not startup_completed:
            overall_status = "starting"
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
            "startup_completed": startup_completed,
            "ready": startup_completed and not job_queue.full(),
            "processing": active_jobs > 0,
            "healthy": True,
        }
    except Exception as e:
        return {"status": "busy", "worker_id": WORKER_ID, "healthy": True, "error": str(e)}

@app.get("/status")
async def status():
    async with job_status_lock:
        jobs_copy = job_status.copy()
    return {"queue_size": job_queue.qsize(), "jobs": jobs_copy, "worker_id": WORKER_ID, "startup_completed": startup_completed}
