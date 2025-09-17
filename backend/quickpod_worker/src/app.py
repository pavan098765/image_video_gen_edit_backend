import os
import time
import uuid
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
# Import inference functions from workers
# -----------------------
from quickpod_worker.worker_ltx_video.src.ltx_video_I2V_inference import run_inference_ltx_video_i2v
from quickpod_worker.worker_ltx_video.src.ltx_video_V2V_inference import run_inference_ltx_video_v2v

from backend.src.quickpod_worker.worker_fastwan.src.fastwan_inference import run_inference_wan_t2i, run_inference_wan_t2v

from quickpod_worker.worker_flux_kontext_dev.src.flux_kontext_dev_inference import run_inference_flux_kontext_dev

from quickpod_worker.worker_qwen_image_edit.src.qwen_image_edit_inference import run_inference_qwen_image_edit

from backend.src.quickpod_worker.worker_sdxl_photomaker.src.sdxl_photomaker_T2I_inference import run_inference_sdxl_photomaker_t2i

from quickpod_worker.worker_stable_diffusion_15.src.sd15_I2I_inference import run_inference_sd15_i2i
from quickpod_worker.worker_stable_diffusion_15.src.sd15_inpaint_inference import run_inference_sd15_inpaint
from quickpod_worker.worker_stable_diffusion_15.src.sd15_T2I_inference import run_inference_sd15_t2i

from quickpod_worker.worker_stable_diffusion_35.src.sd35_I2I_inference import run_inference_sd35_i2i
from quickpod_worker.worker_stable_diffusion_35.src.sd35_inpaint_inference import run_inference_sd35_inpaint
from quickpod_worker.worker_stable_diffusion_35.src.sd35_T2I_inference import run_inference_sd35_t2i

from quickpod_worker.worker_stable_diffusion_xl.src.sdxl_I2I_inference import run_inference_sdxl_i2i
from quickpod_worker.worker_stable_diffusion_xl.src.sdxl_inpaint_inference import run_inference_sdxl_inpaint
from quickpod_worker.worker_stable_diffusion_xl.src.sdxl_T2I_inference import run_inference_sdxl_t2i

from quickpod_worker.worker_chroma_flux.src.chroma_flux_T2I_inference import run_inference_chroma_flux_t2i

# -----------------------
# Model Registry
# -----------------------
MODEL_REGISTRY = {
    "ltx_video": {
        "i2v": run_inference_ltx_video_i2v,
        "v2v": run_inference_ltx_video_v2v,
    },
    "fastwan": {
        # "14b_i2v": run_inference_fastwan_i2v, 
        "1b_t2v": run_inference_wan_t2v,
        "5b_i2v": run_inference_wan_t2v,
        "5b_t2v": run_inference_wan_t2v,
        "5b_t2i": run_inference_wan_t2i,
    },
    "flux_kontext_dev": run_inference_flux_kontext_dev, 
    "qwen_image_edit": run_inference_qwen_image_edit,
    "sdxl_photomaker": run_inference_sdxl_photomaker_t2i,
    "sd15": {
        "i2i": run_inference_sd15_i2i,
        "t2i": run_inference_sd15_t2i,
        "inpaint": run_inference_sd15_inpaint,
    },
    "sd35": {
        "i2i": run_inference_sd35_i2i,
        "t2i": run_inference_sd35_t2i,
        "inpaint": run_inference_sd35_inpaint,
    },
    "sdxl": {
        "i2i": run_inference_sdxl_i2i,
        "t2i": run_inference_sdxl_t2i,
        "inpaint": run_inference_sdxl_inpaint,
    },
    "chroma_flux": run_inference_chroma_flux_t2i,
}

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")

# -----------------------
# Configuration from env
# -----------------------
WORKER_ID = os.getenv("WORKER_ID", str(uuid.uuid4()))
MODEL_KEY = os.getenv("MODEL_KEY", "multi-model-worker")
CONTROLLER_URL = os.getenv("CONTROLLER_URL")
CONTROLLER_SHARED_SECRET = os.getenv("CONTROLLER_SHARED_SECRET")
CONTROLLER_API_KEY = os.getenv("CONTROLLER_API_KEY", "")
WORKER_PUBLIC_URL = os.getenv("WORKER_PUBLIC_URL")
PORT = int(os.getenv("PORT", "8080"))

HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "20"))
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))

WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title=f"QuickPod Worker ({MODEL_KEY})")

# Job queue & status
job_queue: asyncio.Queue = asyncio.Queue()
job_status: Dict[str, Dict[str, Any]] = {}
semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

# Track last model for smart cleanup
last_model_key: Optional[str] = None

# -----------------------
# Request Models
# -----------------------
class RunRequest(BaseModel):
    job_id: str
    prompt: str
    model: str
    mode: str | None = None
    params: Dict[str, Any] = {}

# -----------------------
# HMAC utils
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
# Controller integration
# -----------------------
@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(6))
def register_with_controller():
    register_payload = {
        "worker_id": WORKER_ID,
        "models": list(MODEL_REGISTRY.keys()),
        "endpoint": WORKER_PUBLIC_URL or f"http://{os.getenv('HOSTNAME')}:{PORT}",
        "container_id": os.getenv("CONTAINER_ID", None),
        "info": {"started_at": datetime.utcnow().isoformat()},
    }
    url = f"{CONTROLLER_URL.rstrip('/')}/register_worker"
    logger.info("Registering worker with controller: %s", register_payload)
    resp = post_signed_json(url, register_payload, timeout=10)
    return resp.json()

@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(6))
def send_heartbeat():
    payload = {"worker_id": WORKER_ID, "timestamp": int(time.time())}
    url = f"{CONTROLLER_URL.rstrip('/')}/heartbeat"
    b = json.dumps(payload).encode("utf-8")
    sig = sign_payload_bytes(b)
    headers = {"X-Signature": sig, "Content-Type": "application/json"}
    r = requests.post(url, data=b, headers=headers, timeout=5)
    r.raise_for_status()
    return r.json()

# -----------------------
# Inference dispatcher (Smart Cleanup Mode)
# -----------------------
def resolve_inference_callable(model: str, mode: Optional[str]):
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model}")
    inference_func = MODEL_REGISTRY[model]
    if isinstance(inference_func, dict):
        if not mode or mode not in inference_func:
            raise ValueError(f"Mode required for {model}. Available: {list(inference_func.keys())}")
        inference_func = inference_func[mode]
    return inference_func

async def run_inference_and_save(job_id: str, prompt: str, model: str, mode: Optional[str], params: Dict[str, Any]) -> str:
    global last_model_key
    loop = asyncio.get_event_loop()
    params = params.copy()
    params.setdefault("output_dir", WORKSPACE_DIR)

    inference_func = resolve_inference_callable(model, mode)
    model_key = f"{model}:{mode or ''}"

    # cleanup only if model switched
    if last_model_key and last_model_key != model_key:
        logger.info(f"Switching model {last_model_key} -> {model_key}, cleaning up GPU first")
        cleanup_gpu()

    try:
        out_path = await loop.run_in_executor(None, inference_func, job_id, prompt, params)
        return out_path
    finally:
        last_model_key = model_key  # track last used model

# -----------------------
# Job Processing
# -----------------------
async def process_job_item(item: Dict[str, Any]):
    job_id = item["job_id"]
    model = item["model"]
    mode = item.get("mode")
    prompt = item["prompt"]
    params = item.get("params", {})

    logger.info(f"Processing job {job_id} with model={model}, mode={mode}")
    job_status[job_id] = {"status": "running", "started_at": datetime.utcnow().isoformat()}

    try:
        async with semaphore:
            out_local = await run_inference_and_save(job_id, prompt, model, mode, params)
            filename = os.path.basename(out_local)

            r2_key = f"results/{job_id}/{filename}"
            payload = {"job_id": job_id, "result_r2_key": r2_key}
            url = f"{CONTROLLER_URL.rstrip('/')}/job_complete"
            post_signed_json(url, payload)

            job_status[job_id] = {
                "status": "completed",
                "result_r2_key": r2_key,
                "finished_at": datetime.utcnow().isoformat(),
            }
            logger.info("Job %s completed", job_id)
    except Exception as e:
        logger.exception("Job %s failed: %s", job_id, e)
        job_status[job_id] = {"status": "failed", "error": str(e)}

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
    if req.job_id in job_status and job_status[req.job_id].get("status") in ("queued", "running"):
        raise HTTPException(status_code=409, detail="job already queued or running")
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

# -----------------------
# Startup / Shutdown
# -----------------------
@app.on_event("startup")
async def on_startup():
    logger.info("Worker starting in smart cleanup mode (reuse same model, cleanup on switch)")
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
