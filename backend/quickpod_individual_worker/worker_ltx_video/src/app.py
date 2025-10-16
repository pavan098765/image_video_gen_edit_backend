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

# -----------------------
# Auto-merge split .safetensors
# -----------------------
try:
    from merge import main as merge_main
    print("🔍 Scanning for .safetensors.part-* files...")
    merge_main()
except Exception as e:
    print(f"⚠️ merge.py auto-merge skipped or failed: {e}")

# -----------------------
# GPU cleanup
# -----------------------
def cleanup_gpu():
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()

# -----------------------
# Import the refactored LTX Video inference
# -----------------------
from ltx_video_I2V_inference import run_inference_ltx_t2v_i2v_v2v as run_inference_ltx_video

# -----------------------
# Model registry
# -----------------------
MODEL_REGISTRY = {"ltx_video": run_inference_ltx_video}

# -----------------------
# Logging / Config
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")

CONTAINER_ID = os.getenv("CONTAINER_LABEL")
POD_UUID = CONTAINER_ID
PUBLIC_IPADDR = os.getenv("PUBLIC_IPADDR")
WORKER_ID = POD_UUID
MODEL_KEY = os.getenv("MODEL_KEY", "LTX-Video")
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

job_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
job_status: Dict[str, Dict[str, Any]] = {}
semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

# -----------------------
# Request schema
# -----------------------
class RunRequest(BaseModel):
    job_id: str
    prompt: str
    model: str
    mode: Optional[str] = None
    params: Dict[str, Any] = {}
    upload_url: str
    result_key: str

# -----------------------
# Controller helpers
# -----------------------
def sign_payload_bytes(payload_bytes: bytes) -> str:
    mac = hmac.new(CONTROLLER_SHARED_SECRET.encode("utf-8"), payload_bytes, hashlib.sha256)
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

@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(6))
def register_with_controller():
    if not CONTROLLER_URL:
        logger.info("No controller configured; skipping registration.")
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
    url = f"{CONTROLLER_URL.rstrip('/')}/register_worker"
    logger.info("Registering worker: %s", payload)
    resp = post_signed_json(url, payload)
    return resp.json()

@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(6))
def send_heartbeat():
    if not CONTROLLER_URL:
        return {"skipped": True}
    payload = {"worker_id": WORKER_ID, "pod_uuid": POD_UUID, "timestamp": int(time.time())}
    url = f"{CONTROLLER_URL.rstrip('/')}/heartbeat"
    return post_signed_json(url, payload).json()

# -----------------------
# Job handling
# -----------------------
async def run_inference_and_save(job_id, prompt, model, mode, params):
    loop = asyncio.get_event_loop()
    func = MODEL_REGISTRY[model]
    out_path = await loop.run_in_executor(None, func, job_id, prompt, params)
    return out_path

async def process_job(item):
    job_id = item["job_id"]
    model = item["model"]
    prompt = item["prompt"]
    params = item.get("params", {})
    upload_url = item["upload_url"]
    result_key = item["result_key"]

    logger.info(f"Processing job {job_id}")
    job_status[job_id] = {"status": "running", "started_at": datetime.utcnow().isoformat()}

    try:
        async with semaphore:
            out_path = await run_inference_and_save(job_id, prompt, model, None, params)
            if not upload_url.startswith("http://dummy-upload"):
                with open(out_path, "rb") as f:
                    requests.put(upload_url, data=f).raise_for_status()
            if CONTROLLER_URL:
                post_signed_json(f"{CONTROLLER_URL.rstrip('/')}/job_complete",
                                 {"job_id": job_id, "result_r2_key": result_key})
            job_status[job_id] = {"status": "completed", "result_r2_key": result_key}
            logger.info(f"✅ Job {job_id} completed.")
    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        job_status[job_id] = {"status": "failed", "error": str(e)}

async def worker_loop():
    while True:
        job = await job_queue.get()
        try:
            await process_job(job)
        finally:
            job_queue.task_done()

async def _hb_loop():
    while True:
        try:
            send_heartbeat()
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
        await asyncio.sleep(HEARTBEAT_INTERVAL)

# -----------------------
# Auto-test (T2V + I2V)
# -----------------------
async def run_autotest():
    await asyncio.sleep(5)
    logger.info("🚀 Starting LTX Video auto-test (T2V + I2V)")
    test_img_url = "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/input_image.jpg"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.get(test_img_url)
            resp.raise_for_status()
            img_b64 = base64.b64encode(resp.content).decode("utf-8")
        except Exception as e:
            logger.error(f"❌ Failed to fetch test image: {e}")
            return

    job_id_t2v = f"autotest-ltx-t2v-{uuid.uuid4()}"
    params_t2v = {
        "mode": "text-to-video",
        "height": 512,
        "width": 704,
        "duration": 1.5,
        "randomize_seed": True,
        "guidance_scale": 3.0,
    }
    job_id_i2v = f"autotest-ltx-i2v-{uuid.uuid4()}"
    params_i2v = params_t2v.copy()
    params_i2v["mode"] = "image-to-video"
    params_i2v["init_image"] = img_b64

    async with httpx.AsyncClient(timeout=600.0) as client:
        for jid, model_params in [(job_id_t2v, params_t2v), (job_id_i2v, params_i2v)]:
            resp = await client.post(
                "http://127.0.0.1:8000/run",
                json={
                    "job_id": jid,
                    "prompt": "A cinematic drone shot of mountains at sunrise",
                    "model": "ltx_video",
                    "params": model_params,
                    "upload_url": "http://dummy-upload/skip",
                    "result_key": f"{jid}.mp4",
                },
            )
            logger.info(f"🔥 Auto-test {model_params['mode']} response: {resp.status_code} {resp.text}")

# -----------------------
# Lifespan
# -----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("✅ Starting worker registration...")
    try:
        register_with_controller()
    except Exception as e:
        logger.error(f"Registration failed: {e}")
    asyncio.create_task(worker_loop())
    asyncio.create_task(_hb_loop())
    if os.getenv("ENABLE_AUTOTEST", "1") == "1":
        asyncio.create_task(run_autotest())
    yield
    logger.info("Worker shutting down.")
    cleanup_gpu()

# -----------------------
# FastAPI App
# -----------------------
app = FastAPI(title="QuickPod Worker (LTX Video)", lifespan=lifespan)

@app.post("/run")
async def run_endpoint(req: RunRequest):
    if req.job_id in job_status and job_status[req.job_id].get("status") in ("queued", "running"):
        raise HTTPException(status_code=409, detail="Job already queued or running")
    await job_queue.put(req.dict())
    job_status[req.job_id] = {"status": "queued", "queued_at": datetime.utcnow().isoformat()}
    logger.info("Enqueued job %s", req.job_id)
    return {"status": "accepted", "job_id": req.job_id}

@app.get("/health")
def health():
    return {"status": "ok", "worker_id": WORKER_ID, "queue_size": job_queue.qsize()}

@app.get("/status")
def status():
    return {"queue_size": job_queue.qsize(), "jobs": job_status}
