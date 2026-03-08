# WAN 2.1 TurboDiffusion T2V 720p worker - app.py (T2V only, no init_image)
import os
import time
import json
import logging
import asyncio
import hmac
import hashlib
import gc
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
import httpx
import concurrent.futures

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt
from contextlib import asynccontextmanager

import torch

POD_START_TIME = time.time()
FIRST_INFERENCE_START_TIME = None
_first_inference_lock = asyncio.Lock()
INFERENCE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
IO_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="io")


def cleanup_gpu(verbose: bool = False):
    try:
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    del obj
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception as e:
        print(f"[GPU] Cleanup warning: {e}")


try:
    from merge import main as merge_main
    print("Running merge.py...")
    merge_main()
    print("Merge completed")
except SystemExit as e:
    print(f"CRITICAL: Merge failed: {e.code}")
    raise
except Exception as e:
    print(f"Merge error: {e}")

from inference import run_inference_wan_turbo_t2v

MODEL_REGISTRY = {"wan_21_turbo_t2v_720p": {"t2v": run_inference_wan_turbo_t2v}}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")

CONTAINER_ID = os.getenv("CONTAINER_LABEL")
POD_UUID = CONTAINER_ID
PUBLIC_IPADDR = os.getenv("PUBLIC_IPADDR")
WORKER_ID = POD_UUID
MODEL_KEY = os.getenv("MODEL_KEY", "wan-21-turbo-t2v-720p-worker")
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

job_queue: asyncio.Queue = asyncio.Queue(maxsize=30)
job_status: Dict[str, Dict[str, Any]] = {}
job_status_lock = asyncio.Lock()
semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
last_model_key: Optional[str] = None
startup_completed = False


def _peek_next_job_id() -> Optional[str]:
    try:
        if not job_queue.empty():
            return job_queue._queue[0].get("job_id")
    except (IndexError, AttributeError):
        pass
    return None


class RunRequest(BaseModel):
    job_id: str
    prompt: str
    model: str
    mode: str | None = None
    params: Dict[str, Any] = {}
    upload_url: str
    result_key: str


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


@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(6))
def register_with_controller():
    if not CONTROLLER_URL:
        return {"skipped": True}
    payload = {
        "worker_id": WORKER_ID,
        "models": list(MODEL_REGISTRY.keys()),
        "endpoint": f"http://{PUBLIC_IPADDR}:{PORT}",
        "container_id": POD_UUID,
        "pod_uuid": POD_UUID,
        "public_ip": PUBLIC_IPADDR,
        "info": {"started_at": datetime.utcnow().isoformat()},
    }
    return post_signed_json(f"{CONTROLLER_URL.rstrip('/')}/register_worker", payload, timeout=15, max_retries=6).json()


@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(3))
def send_heartbeat():
    if not CONTROLLER_URL:
        return {"skipped": True}
    status = "starting" if not startup_completed else "busy" if any(j.get("status") == "running" for j in job_status.values()) or not job_queue.empty() else "ready"
    payload = {"worker_id": WORKER_ID, "pod_uuid": POD_UUID, "timestamp": int(time.time()), "status": status}
    try:
        return post_signed_json(f"{CONTROLLER_URL.rstrip('/')}/heartbeat", payload, timeout=8, max_retries=3).json()
    except Exception:
        return {}


def resolve_inference_callable(model: str, mode: Optional[str]):
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model}")
    func = MODEL_REGISTRY[model]
    if isinstance(func, dict):
        if not mode or mode not in func:
            raise ValueError(f"Mode required for {model}. Available: {list(func.keys())}")
        func = func[mode]
    return func


async def run_inference_and_save(
    job_id: str, prompt: str, model: str, mode: Optional[str], params: Dict[str, Any]
) -> Tuple[str, Dict[str, float]]:
    global last_model_key, FIRST_INFERENCE_START_TIME
    loop = asyncio.get_event_loop()
    params = params.copy()
    params.setdefault("output_dir", WORKSPACE_DIR)
    async with _first_inference_lock:
        if FIRST_INFERENCE_START_TIME is None:
            FIRST_INFERENCE_START_TIME = time.time()
            startup_time = FIRST_INFERENCE_START_TIME - POD_START_TIME
        else:
            startup_time = None
    inference_func = resolve_inference_callable(model, mode)
    model_key = f"{model}:{mode or ''}"
    if last_model_key and last_model_key != model_key:
        cleanup_gpu()
    inference_start_time = time.time()
    try:
        out_path = await loop.run_in_executor(INFERENCE_EXECUTOR, inference_func, job_id, prompt, params)
        metrics = {"processing_time_seconds": round(time.time() - inference_start_time, 2)}
        if startup_time is not None:
            metrics["startup_time_seconds"] = round(startup_time, 2)
        return out_path, metrics
    finally:
        last_model_key = model_key


async def process_job_item(item: Dict[str, Any]):
    job_id, model, mode, prompt = item["job_id"], item["model"], item.get("mode"), item["prompt"]
    params, upload_url, result_key = item.get("params", {}), item["upload_url"], item["result_key"]
    async with job_status_lock:
        job_status[job_id] = {"status": "running", "started_at": datetime.utcnow().isoformat()}
    try:
        async with semaphore:
            is_autotest = job_id.startswith("autotest-")
            if CONTROLLER_URL and not is_autotest:
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        IO_EXECUTOR, post_signed_json,
                        f"{CONTROLLER_URL.rstrip('/')}/job_started", {"job_id": job_id, "worker_id": WORKER_ID}, 8, 3)
                except Exception as e:
                    logger.warning(f"[{job_id}] job_started: {e}")
            out_local, metrics = await run_inference_and_save(job_id, prompt, model, mode, params)
            if not upload_url.startswith("http://dummy-upload"):
                if not await _upload_with_retry(out_local, upload_url, job_id):
                    raise Exception("Upload failed")
            if CONTROLLER_URL and not is_autotest:
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        IO_EXECUTOR, post_signed_json, f"{CONTROLLER_URL.rstrip('/')}/job_complete",
                        {"job_id": job_id, "result_r2_key": result_key, "next_job_id": _peek_next_job_id(), "metrics": metrics}, 10, 5)
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
        if CONTROLLER_URL and not job_id.startswith("autotest-"):
            try:
                await asyncio.get_running_loop().run_in_executor(
                    IO_EXECUTOR, post_signed_json, f"{CONTROLLER_URL.rstrip('/')}/job_complete",
                    {"job_id": job_id, "result_r2_key": None, "failed": True, "next_job_id": _peek_next_job_id(), "metrics": None}, 10, 5)
            except Exception:
                pass
        asyncio.create_task(_cleanup_job_status(job_id, 60))


async def _upload_with_retry(file_path: str, upload_url: str, job_id: str, max_retries: int = 3) -> bool:
    def _upload():
        with open(file_path, "rb") as f:
            requests.put(upload_url, data=f, timeout=120).raise_for_status()
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
    while True:
        try:
            item = await job_queue.get()
            try:
                await process_job_item(item)
            except Exception:
                pass
            finally:
                job_queue.task_done()
        except Exception:
            await asyncio.sleep(1)


async def run_autotest():
    await asyncio.sleep(5)
    logger.info("Starting WAN 2.1 Turbo T2V 720p auto-test")
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            t2v_job_id = f"autotest-t2v-{uuid.uuid4()}"
            resp = await client.post("http://127.0.0.1:8000/run", json={
                "job_id": t2v_job_id,
                "prompt": "A cinematic shot of a beach at sunset",
                "model": "wan_21_turbo_t2v_720p",
                "mode": "t2v",
                "params": {
                    "num_frames": 81,
                    "num_steps": 4,
                    "fps": 16,
                    "resolution": "720p",
                    "aspect_ratio": "16:9",
                    "enable_rife": True,
                    "enable_upscaling": True,
                    "upscale_factor": 2,
                    "rife_target_fps": 30,
                },
                "upload_url": "http://dummy-upload/skip",
                "result_key": f"{t2v_job_id}.mp4",
            })
            logger.info(f"T2V test: {resp.status_code}")
        except Exception as e:
            logger.error(f"T2V test failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global startup_completed
    asyncio.create_task(worker_loop())
    asyncio.create_task(_hb_loop())
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
        return {"status": status, "worker_id": WORKER_ID, "queue_size": job_queue.qsize(), "active_jobs": active, "startup_completed": startup_completed, "ready": startup_completed and not job_queue.full(), "healthy": True}
    except Exception as e:
        return {"status": "busy", "worker_id": WORKER_ID, "healthy": True, "error": str(e)}


@app.get("/status")
async def status():
    async with job_status_lock:
        return {"queue_size": job_queue.qsize(), "jobs": job_status.copy(), "worker_id": WORKER_ID}
