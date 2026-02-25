import os
import time
import json
import logging
import asyncio
import hmac
import hashlib
import gc
from typing import Dict, Any, Optional, List
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
INFERENCE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
IO_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="io")

# -----------------------
# GPU cleanup
# -----------------------
def cleanup_gpu(verbose: bool = False):
    try:
        if verbose:
            print(f"[GPU] Before: {torch.cuda.memory_allocated()/1024**2:.2f} MB allocated")
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    del obj
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()
        # ipc_collect() may fail if PyTorch version doesn't support certain allocator options
        try:
            torch.cuda.ipc_collect()
        except (RuntimeError, AttributeError) as e:
            if "expandable_segments" in str(e) or "CachingAllocator" in str(e):
                # PyTorch version doesn't support this option - skip it
                pass
            else:
                raise
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()
        if verbose:
            print(f"[GPU] After: {torch.cuda.memory_allocated()/1024**2:.2f} MB allocated")
    except Exception as e:
        print(f"[GPU] Cleanup warning: {e}")

# -----------------------
# Import inference function
# -----------------------
from inference_face_enhancer import run_inference_face_enhancer

# -----------------------
# Model Registry
# -----------------------
MODEL_REGISTRY = {
    "face_enhancer": run_inference_face_enhancer,
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

MODEL_KEY = os.getenv("MODEL_KEY", "face-enhancer-worker")
CONTROLLER_URL = os.getenv("CONTROLLER_URL", "").strip()
CONTROLLER_SHARED_SECRET = os.getenv("CONTROLLER_SHARED_SECRET", "")
CONTROLLER_API_KEY = os.getenv("CONTROLLER_API_KEY", "")
INTERNAL_PORT = 8000
EXTERNAL_PORT = os.getenv(f"QUICKPOD_PORT_{INTERNAL_PORT}")
PORT = int(EXTERNAL_PORT) if EXTERNAL_PORT else INTERNAL_PORT

HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "20"))
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))

WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# -----------------------
# Job queue and state
# -----------------------
job_queue: asyncio.Queue = asyncio.Queue(maxsize=30)
job_status: Dict[str, Dict[str, Any]] = {}
job_status_lock = asyncio.Lock()
semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
last_model_key: Optional[str] = None
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
# Request Model
# -----------------------
class RunRequest(BaseModel):
    job_id: str
    prompt: Optional[str] = None
    model: str
    mode: str | None = None
    params: Dict[str, Any] = {}
    upload_url: str
    result_key: str


class BulkRunRequest(BaseModel):
    jobs: List[RunRequest]

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
    return "sha256=" + hmac.new(CONTROLLER_SHARED_SECRET.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()

def signed_headers(payload: dict) -> Dict[str, str]:
    b = json.dumps(payload).encode("utf-8")
    headers = {"X-Signature": sign_payload_bytes(b), "Content-Type": "application/json"}
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
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except requests.exceptions.HTTPError as e:
            if e.response and 400 <= e.response.status_code < 500:
                raise
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except Exception:
            raise
    raise last_error

# -----------------------
# Controller integration
# -----------------------
@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(6))
def register_with_controller():
    if not CONTROLLER_URL:
        return {"skipped": True}
    payload = {
        "worker_id": WORKER_ID, "models": list(MODEL_REGISTRY.keys()),
        "endpoint": f"http://{PUBLIC_IPADDR}:{PORT}", "container_id": POD_UUID,
        "pod_uuid": POD_UUID, "public_ip": PUBLIC_IPADDR,
        "info": {"started_at": datetime.utcnow().isoformat()},
    }
    return post_signed_json(f"{CONTROLLER_URL.rstrip('/')}/register_worker", payload, timeout=15, max_retries=6).json()

@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(3))
def send_heartbeat():
    if not CONTROLLER_URL:
        return {"skipped": True}
    # Determine actual worker status based on job state
    if not startup_completed:
        status = "starting"
    else:
        try:
            has_running = any(j.get("status") == "running" for j in job_status.values())
            has_queued = not job_queue.empty()
            status = "busy" if (has_running or has_queued) else "ready"
        except Exception:
            status = "ready"
    payload = {"worker_id": WORKER_ID, "pod_uuid": POD_UUID, "timestamp": int(time.time()), "status": status}
    try:
        return post_signed_json(f"{CONTROLLER_URL.rstrip('/')}/heartbeat", payload, timeout=8, max_retries=3).json()
    except Exception:
        return {}

# -----------------------
# Inference dispatcher
# -----------------------
def resolve_inference_callable(model: str, mode: Optional[str]):
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model}")
    return MODEL_REGISTRY[model]

async def run_inference_and_save(job_id: str, prompt: Optional[str], model: str, mode: Optional[str], params: Dict[str, Any]) -> str:
    global last_model_key
    loop = asyncio.get_event_loop()
    params = params.copy()
    params.setdefault("output_dir", WORKSPACE_DIR)
    inference_func = resolve_inference_callable(model, mode)

    if last_model_key and last_model_key != model:
        cleanup_gpu()

    try:
        return await loop.run_in_executor(INFERENCE_EXECUTOR, inference_func, job_id, params)
    finally:
        last_model_key = model

# -----------------------
# Job Processing
# -----------------------
async def process_job_item(item: Dict[str, Any]):
    job_id, model, prompt = item["job_id"], item["model"], item.get("prompt")
    params, upload_url, result_key = item.get("params", {}), item["upload_url"], item["result_key"]

    # Resolve any URL-valued media fields to base64 before inference
    try:
        params = resolve_media_urls(params, job_id)
    except Exception as e:
        logger.error(f"[{job_id}] Failed to resolve media URLs: {e}")

    async with job_status_lock:
        job_status[job_id] = {"status": "running", "started_at": datetime.utcnow().isoformat()}

    try:
        async with semaphore:
            out_local = await run_inference_and_save(job_id, prompt, model, None, params)
            if not upload_url.startswith("http://dummy-upload"):
                if not await _upload_with_retry(out_local, upload_url, job_id):
                    raise Exception("Upload failed")

            is_autotest = job_id.startswith("autotest-")
            if CONTROLLER_URL and not is_autotest:
                try:
                    await asyncio.get_running_loop().run_in_executor(IO_EXECUTOR, post_signed_json,
                        f"{CONTROLLER_URL.rstrip('/')}/job_complete", {"job_id": job_id, "result_r2_key": result_key, "next_job_id": _peek_next_job_id()}, 10, 5)
                except Exception:
                    pass

            async with job_status_lock:
                job_status[job_id] = {"status": "completed", "result_r2_key": result_key}
            logger.info(f"[{job_id}] Completed")
            asyncio.create_task(_cleanup_job_status(job_id, 30))
    except Exception as e:
        logger.exception(f"[{job_id}] Failed: {e}")
        if "out of memory" in str(e).lower():
            cleanup_gpu(verbose=True)
        async with job_status_lock:
            job_status[job_id] = {"status": "failed", "error": str(e)}
        is_autotest = job_id.startswith("autotest-")
        if CONTROLLER_URL and not is_autotest:
            try:
                await asyncio.get_running_loop().run_in_executor(IO_EXECUTOR, post_signed_json,
                    f"{CONTROLLER_URL.rstrip('/')}/job_complete", {"job_id": job_id, "result_r2_key": None, "failed": True, "next_job_id": _peek_next_job_id()}, 10, 5)
            except Exception:
                pass
        asyncio.create_task(_cleanup_job_status(job_id, 60))

async def _upload_with_retry(file_path: str, upload_url: str, job_id: str, max_retries: int = 3) -> bool:
    def _upload():
        with open(file_path, "rb") as f:
            requests.put(upload_url, data=f, timeout=60).raise_for_status()
    for attempt in range(max_retries):
        try:
            await asyncio.get_running_loop().run_in_executor(IO_EXECUTOR, _upload)
            return True
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
    return False

async def _cleanup_job_status(job_id: str, delay: int = 30):
    await asyncio.sleep(delay)
    async with job_status_lock:
        if job_id in job_status and job_status[job_id].get("status") in ("completed", "failed"):
            del job_status[job_id]

async def worker_loop():
    consecutive_errors = 0
    while True:
        try:
            item = await job_queue.get()
            try:
                await process_job_item(item)
                consecutive_errors = 0
            except Exception:
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    cleanup_gpu(verbose=True)
                    consecutive_errors = 0
            finally:
                job_queue.task_done()
        except Exception:
            await asyncio.sleep(1)

# -----------------------
# Auto-test
# -----------------------
async def run_autotest():
    await asyncio.sleep(5)
    logger.info("🚀 Starting Face Enhancer auto-test")

    # Check GPU VRAM - disable autotest on low-memory cards to avoid OOM
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram_gb < 20:
            logger.info(f"⚠️ Skipping autotest on {vram_gb:.0f}GB GPU (< 20GB) to avoid OOM conflicts with real jobs")
            return
    except Exception as e:
        logger.warning(f"Could not check GPU VRAM for autotest decision: {e}")
        # Proceed cautiously

    # Fetch test image
    img_url = "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/Snapinsta.app_262322643_4937545659591345_7487236998017046274_n_1080.jpg"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
        "Referer": "https://ibb.co",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    }
    
    img_b64 = None
    try:
        async with httpx.AsyncClient(headers=headers, timeout=60.0) as client:
            resp = await client.get(img_url)
            resp.raise_for_status()
            img_b64 = base64.b64encode(resp.content).decode("utf-8")
    except Exception as e:
        logger.error(f"❌ Failed to fetch test image: {e}")
        return

    # Import GFPGAN_MODELS from inference file
    from inference_face_enhancer import GFPGAN_MODELS

    # Test all available versions
    versions_to_test = list(GFPGAN_MODELS.keys())
    results = {}
    
    autotest_output_dir = os.path.join(WORKSPACE_DIR, "autotest_outputs")
    os.makedirs(autotest_output_dir, exist_ok=True)
    
    logger.info(f"Testing {len(versions_to_test)} versions: {versions_to_test}")
    logger.info(f"Results will be saved to: {autotest_output_dir}")

    async with httpx.AsyncClient(timeout=300.0) as client:
        for i, version in enumerate(versions_to_test):
            test_job_id = f"autotest-face-enhancer-{version.replace(' ', '-').replace('.', '-')}-{uuid.uuid4()}"
            
            params = {
                "init_image": img_b64,
                "version": version,
                "scale": 2.0,
                "output_dir": autotest_output_dir,  # Save to dedicated autotest output dir
            }
            
            try:
                start_time = time.time()
                resp = await client.post(
                    "http://127.0.0.1:8000/run",
                    json={
                        "job_id": test_job_id,
                        "prompt": "",
                        "model": "face_enhancer",
                        "params": params,
                        "upload_url": "http://dummy-upload/skip",  # Don't upload autotest results
                        "result_key": f"{test_job_id}.png",
                    },
                )
                resp.raise_for_status()
                response_data = resp.json()
                
                # Wait for job to complete
                status_url = f"http://127.0.0.1:8000/status"
                job_completed = False
                wait_start = time.time()
                while not job_completed and (time.time() - wait_start < 300):  # Max 5 min wait per job
                    await asyncio.sleep(2)
                    status_resp = await client.get(status_url)
                    status_resp_data = status_resp.json()
                    job_info = status_resp_data.get("jobs", {}).get(test_job_id)
                    if job_info and job_info.get("status") in ["completed", "failed"]:
                        job_completed = True
                        results[version] = {
                            "status": job_info["status"],
                            "processing_time": job_info.get("processing_time"),
                            "error": job_info.get("error"),
                            "output_path": os.path.join(autotest_output_dir, f"face_enhancer_{test_job_id}.png")
                        }
                        break
                    elif job_info and job_info.get("status") == "running":
                        if (time.time() - wait_start) % 60 < 2:  # Log every minute
                            logger.info(f"[{test_job_id}] Still running... ({int(time.time() - wait_start)}s)")
                
                if not job_completed:
                    results[version] = {"status": "pending", "error": "Job timed out"}
                    logger.warning(f"[{test_job_id}] Job timed out after 300s.")

                logger.info(f"🔥 Auto-test response for {version}: {resp.status_code} {resp.text}")
                
            except Exception as e:
                logger.error(f"❌ Auto-test for {version} failed: {e}")
                results[version] = {"status": "failed", "error": str(e)}
            
            await asyncio.sleep(2)  # Small delay between tests
    
    logger.info("\n--- Auto-test Summary ---")
    for version, res in results.items():
        if res["status"] == "completed":
            logger.info(f"✅ {version}: Completed in {res.get('processing_time', 0):.1f}s. Output: {res.get('output_path', 'N/A')}")
        elif res["status"] == "failed":
            logger.error(f"❌ {version}: Failed. Error: {res.get('error', 'N/A')}")
        else:
            logger.warning(f"⏳ {version}: {res['status']}. Error: {res.get('error', 'N/A')}")
    logger.info("--- End of Auto-test Summary ---")

# -----------------------
# Loops
# -----------------------
async def _health_monitor_loop():
    while True:
        await asyncio.sleep(60)
        try:
            if torch.cuda.is_available() and torch.cuda.memory_allocated() / 1024**3 > 22:
                logger.warning("⚠️ HIGH GPU MEMORY")
        except Exception:
            pass

async def _hb_loop():
    while True:
        try:
            await asyncio.get_running_loop().run_in_executor(IO_EXECUTOR, send_heartbeat)
        except Exception:
            pass
        await asyncio.sleep(HEARTBEAT_INTERVAL)

async def _record_job_status_later(job_id: str, model: str):
    async with job_status_lock:
        job_status[job_id] = {"status": "queued", "queued_at": datetime.utcnow().isoformat(), "model": model}

# -----------------------
# Lifespan
# -----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global startup_completed
    asyncio.create_task(worker_loop())
    asyncio.create_task(_hb_loop())
    asyncio.create_task(_health_monitor_loop())
    startup_completed = True
    try:
        register_with_controller()
    except Exception as e:
        logger.exception("Registration failed: %s", e)
    if os.getenv("ENABLE_AUTOTEST", "1") == "1":
        asyncio.create_task(run_autotest())
    yield
    startup_completed = False
    if not job_queue.empty():
        try:
            await asyncio.wait_for(job_queue.join(), timeout=30.0)
        except asyncio.TimeoutError:
            pass
    INFERENCE_EXECUTOR.shutdown(wait=True, cancel_futures=False)
    IO_EXECUTOR.shutdown(wait=False)
    cleanup_gpu()

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
    return {"status": "accepted", "job_id": req.job_id, "queue_position": job_queue.qsize(), "worker_id": WORKER_ID}


@app.post("/run_bulk")
async def run_bulk_endpoint(req: BulkRunRequest):
    """
    Accept multiple jobs for processing in a single request.
    Used for bulk edit and multi-variation requests.
    """
    if not req.jobs:
        raise HTTPException(status_code=400, detail="No jobs provided")
    if not startup_completed:
        raise HTTPException(status_code=503, detail="Worker not ready")
    
    available_slots = job_queue.maxsize - job_queue.qsize()
    if available_slots < len(req.jobs):
        raise HTTPException(
            status_code=503,
            detail=f"Not enough queue capacity. Available: {available_slots}, Requested: {len(req.jobs)}"
        )
    
    accepted_ids = []
    for job in req.jobs:
        if job.model not in MODEL_REGISTRY:
            logger.warning(f"[BULK] Unknown model {job.model} for job {job.job_id}, skipping")
            continue
        
        try:
            await asyncio.wait_for(job_status_lock.acquire(), timeout=2.0)
            try:
                if job.job_id in job_status and job_status[job.job_id].get("status") in ("queued", "running"):
                    logger.warning(f"[BULK] Job {job.job_id} already {job_status[job.job_id]['status']}, skipping")
                    continue
                if job.job_id in job_status:
                    del job_status[job.job_id]
            finally:
                job_status_lock.release()
        except asyncio.TimeoutError:
            pass
        
        await job_queue.put(job.dict())
        asyncio.create_task(_record_job_status_later(job.job_id, job.model))
        accepted_ids.append(job.job_id)
    
    logger.info(f"[BULK] Enqueued {len(accepted_ids)} of {len(req.jobs)} jobs")
    return {
        "status": "accepted",
        "job_ids": accepted_ids,
        "total_accepted": len(accepted_ids),
        "total_requested": len(req.jobs),
        "worker_id": WORKER_ID,
        "queue_size": job_queue.qsize(),
    }


@app.get("/health")
async def health():
    try:
        try:
            await asyncio.wait_for(job_status_lock.acquire(), timeout=1.0)
            try:
                active = sum(1 for j in job_status.values() if j.get("status") == "running")
            finally:
                job_status_lock.release()
        except asyncio.TimeoutError:
            active = MAX_CONCURRENT_JOBS
        status = "starting" if not startup_completed else "overloaded" if job_queue.full() else "busy" if active >= MAX_CONCURRENT_JOBS else "ok"
        return {"status": status, "worker_id": WORKER_ID, "queue_size": job_queue.qsize(), "active_jobs": active,
            "startup_completed": startup_completed, "ready": startup_completed and not job_queue.full(), "healthy": True}
    except Exception as e:
        return {"status": "busy", "worker_id": WORKER_ID, "healthy": True, "error": str(e)}

@app.get("/status")
async def status():
    async with job_status_lock:
        return {"queue_size": job_queue.qsize(), "jobs": job_status.copy(), "worker_id": WORKER_ID}
