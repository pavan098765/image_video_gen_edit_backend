import os
import re
import time
import json
import logging
import asyncio
import hmac
import hashlib
import gc
from typing import Dict, Any, Optional, List, Tuple
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
# CRITICAL: Use a dedicated executor for inference to avoid blocking FastAPI
# This ensures /health and /run endpoints always respond, even during heavy GPU work
# 
# Why this matters:
# - Python's GIL can block the event loop during CPU-heavy operations
# - Default executor has limited workers that get exhausted
# - A dedicated pool ensures inference has its own threads
#
# IMPORTANT: Only 1 worker to prevent OOM on 24GB cards (4090)!
# Multiple simultaneous inferences will exhaust GPU memory.
# Jobs are queued internally (job_queue) anyway, so 1 thread is sufficient.
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
# Pod startup time tracking
# -----------------------
POD_START_TIME = time.time()
FIRST_INFERENCE_START_TIME = None
_first_inference_lock = asyncio.Lock()

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
# Import inference function
# -----------------------
from qwen_image_edit_2509_inference import run_inference_qwen_image_edit_2509

# -----------------------
# Model Registry
# -----------------------
MODEL_REGISTRY = {
    "qwen_image_edit_2509": run_inference_qwen_image_edit_2509,
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

MODEL_KEY = os.getenv("MODEL_KEY", "qwen-worker")
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
job_status_lock = asyncio.Lock()  # Protect job_status updates
semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
last_model_key: Optional[str] = None
worker_ready = asyncio.Event()  # Signal when worker is fully ready
startup_completed = False  # Track startup completion


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
    mode: str | None = None   # not used, but kept for consistency
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
    
    Critical for job_complete notifications - must succeed or job stays stuck.
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
                backoff = 2 ** attempt  # 1s, 2s, 4s
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
            # Don't retry on 4xx errors (client errors)
            if e.response and 400 <= e.response.status_code < 500:
                logger.error(f"POST to {url} failed with {e.response.status_code}: {e}")
                raise
            # Retry on 5xx errors (server errors)
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
    
    # All retries failed
    logger.error(f"POST to {url} failed after {max_retries} attempts: {last_error}")
    raise last_error

# -----------------------
# Controller integration
# -----------------------
@retry(wait=wait_exponential(min=2, max=20), stop=stop_after_attempt(6))
def register_with_controller():
    """
    Register worker with controller with retry logic.
    
    Critical for worker to be discovered and receive jobs.
    """
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
    logger.info(f"Pod UUID: {POD_UUID}")
    
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
    """
    Send heartbeat to controller with retry logic.
    
    Includes worker status for server visibility.
    """
    if not CONTROLLER_URL:
        return {"skipped": True}
    
    # Get current worker status
    try:
        queue_size = job_queue.qsize()
        active_jobs = sum(1 for j in job_status.values() if j.get("status") == "running")
    except Exception:
        queue_size = 0
        active_jobs = 0
    
    # Determine status based on model loading state and job activity
    if not _model_loading_complete:
        status = "starting"  # Model still loading
    elif startup_completed:
        status = "busy" if (active_jobs > 0 or queue_size > 0) else "ready"
    else:
        status = "starting"
    
    payload = {
        "worker_id": WORKER_ID,
        "pod_uuid": POD_UUID,
        "timestamp": int(time.time()),
        "status": status,
        "model_loaded": _model_loading_complete,
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
        # Don't crash on heartbeat failure - just log
        logger.debug(f"Heartbeat failed: {e}")
        return {"error": str(e)}

# -----------------------
# Together AI Prompt Enhancement
# -----------------------
TOGETHER_API_KEY = os.environ.get(
    "TOGETHER_API_KEY",
    "no_api_key",
)
TOGETHER_MODEL = "ServiceNow-AI/Apriel-1.6-15b-Thinker"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

_ENHANCE_SYSTEM_PROMPT_IMAGE_EDIT = (
    "You are an expert cinematic prompt engineer specialized in enhancing prompts for AI image editing models.\n\n"
    "Your task is NOT to change the user's intent, but to expand and refine it into a highly detailed, visually descriptive, and editing-optimized prompt.\n\n"
    "Follow these principles:\n\n"
    "1. Preserve Core Intent\n- Never alter the main subject or meaning.\n- Only enhance clarity, detail, and visual richness.\n\n"
    "2. Add Missing Visual Information\nEnhance prompts with:\n- Subject description (appearance, clothing, age, materials, textures)\n"
    "- Environment and background\n- Lighting (cinematic lighting, volumetric light, rim light, neon glow, etc.)\n"
    "- Camera details (lens, depth of field, perspective, framing)\n- Mood and atmosphere\n- Color palette\n- Composition\n"
    "- Time of day / weather\n- Style references if appropriate\n\n"
    "3. Editing Requests\nClearly describe:\n- what remains unchanged\n- what is modified\n- integration realism\n"
    "- lighting consistency\n- perspective matching\n\n"
    "4. Cinematic & Photorealistic Quality Boost\nWhen suitable, include:\n- ultra detailed\n- cinematic\n- photorealistic\n"
    "- high dynamic range\n- global illumination\n- 8k / high resolution\n- professional photography\n- shallow depth of field\n"
    "- film grain\n- realistic shadows\n\n"
    "5. Avoid Negative Behavior\nDo NOT:\n- add unrelated objects\n- change gender, ethnicity, or identity unless asked\n"
    "- introduce unwanted elements\n- overcomplicate simple prompts\n\n"
    "6. Output Format Rules\nAlways output:\n\nENHANCED PROMPT:\n<final enhanced prompt>\n\n"
    "Optional sections when useful:\n\nNEGATIVE PROMPT:\n<artifacts to avoid>\n\n"
    "EDIT INSTRUCTIONS:\n<only if editing related>\n\n"
    "Do NOT explain reasoning.\nDo NOT include commentary.\nOnly output the final structured result.\n\n"
    "7. Style Intelligence\nAutomatically infer best style based on user intent.\n\n"
    "8. Model Awareness\nOptimize prompts to work well with diffusion models such as:\nStable Diffusion, SDXL, Flux.\n\n"
    "9. Precision Over Length\nBe descriptive but meaningful. Avoid useless adjectives.\n\n"
    "Your goal: produce prompts that generate visually stunning, coherent, and realistic edited images with minimal artifacts."
)


def _enhance_prompt(prompt: str, job_id: str) -> str:
    """
    Enhance a prompt using Together AI LLM for image editing.

    Returns enhanced prompt, or original prompt on failure.
    """
    if TOGETHER_API_KEY == "no_api_key":
        logger.warning("PROMPT ENHANCE REQUESTED, SKIPPING.. NO API KEY SET IN ENV VAR")
        return prompt

    logger.info("[%s] ENHANCE: calling Together AI (%s) to enhance prompt (type=image_edit)", job_id, TOGETHER_MODEL)
    logger.info("[%s] ENHANCE: original prompt = '%s'", job_id, prompt[:200] + ("..." if len(prompt) > 200 else ""))

    try:
        resp = requests.post(
            TOGETHER_API_URL,
            headers={
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": TOGETHER_MODEL,
                "messages": [
                    {"role": "system", "content": _ENHANCE_SYSTEM_PROMPT_IMAGE_EDIT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        logger.info("[%s] ENHANCE: raw response (%d chars): %s", job_id, len(content), content[:300])

        match = re.search(r"ENHANCED PROMPT:\s*\n?(.*?)(?:\n\s*\n(?:NEGATIVE PROMPT:|VIDEO PARAMETERS:|EDIT INSTRUCTIONS:)|$)", content, re.DOTALL)
        if match:
            enhanced = match.group(1).strip()
        else:
            cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            enhanced = cleaned if cleaned else content

        if enhanced:
            logger.info("[%s] ENHANCE: enhanced prompt = '%s'", job_id, enhanced[:200] + ("..." if len(enhanced) > 200 else ""))
            return enhanced

    except Exception as e:
        logger.warning("[%s] ENHANCE: prompt enhancement failed: %s", job_id, e, exc_info=True)

    logger.info("[%s] ENHANCE: falling back to original prompt", job_id)
    return prompt


# -----------------------
# Inference dispatcher
# -----------------------
def resolve_inference_callable(model: str, mode: Optional[str]):
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model}")
    return MODEL_REGISTRY[model]

async def run_inference_and_save(
    job_id: str, prompt: str, model: str, mode: Optional[str], params: Dict[str, Any]
) -> Tuple[str, Dict[str, float]]:
    """Run inference in a dedicated thread pool. Returns (output_path, metrics_dict)."""
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

    if params.get("enhance_prompt", False):
        prompt = await loop.run_in_executor(IO_EXECUTOR, _enhance_prompt, prompt, job_id)

    inference_func = resolve_inference_callable(model, mode)
    model_key = model
    if last_model_key and last_model_key != model_key:
        logger.info(f"Switching model {last_model_key} -> {model_key}, cleaning up GPU first")
        cleanup_gpu()

    inference_start_time = time.time()
    try:
        out_path = await loop.run_in_executor(INFERENCE_EXECUTOR, inference_func, job_id, prompt, params)
        processing_time = time.time() - inference_start_time
        metrics = {"processing_time_seconds": round(processing_time, 2)}
        if startup_time is not None:
            metrics["startup_time_seconds"] = round(startup_time, 2)
        return out_path, metrics
    finally:
        last_model_key = model_key

# -----------------------
# Job Processing
# -----------------------
async def process_job_item(item: Dict[str, Any]):
    """
    Process a single job with comprehensive error handling and recovery.
    
    Ensures job always completes (success or failure) and notifies controller.
    """
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
    
    # Update status with lock
    async with job_status_lock:
        job_status[job_id] = {
            "status": "running", 
            "started_at": start_time.isoformat(),
            "model": model,
        }

    try:
        async with semaphore:
            # Run inference
            logger.info(f"[{job_id}] Starting inference...")
            out_local, metrics = await run_inference_and_save(job_id, prompt, model, mode, params)
            logger.info(f"[{job_id}] Inference completed, output: {out_local}")

            # Upload result with retry
            if upload_url.startswith("http://dummy-upload"):
                logger.info(f"[{job_id}] Skipping upload (dummy). File at {out_local}")
            else:
                logger.info(f"[{job_id}] Uploading result to R2...")
                upload_success = await _upload_with_retry(out_local, upload_url, job_id)
                if not upload_success:
                    raise Exception("Failed to upload result after retries")
                logger.info(f"[{job_id}] Upload completed successfully")

            # Notify controller with retry (skip for autotest jobs - they're not on the server)
            is_autotest = job_id.startswith("autotest-")
            if CONTROLLER_URL and not is_autotest:
                logger.info(f"[{job_id}] Notifying controller of completion...")
                payload = {"job_id": job_id, "result_r2_key": result_key, "next_job_id": _peek_next_job_id(), "metrics": metrics}
                url = f"{CONTROLLER_URL.rstrip('/')}/job_complete"
                try:
                    # post_signed_json(url, payload, timeout=10, max_retries=5)
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
                    # CRITICAL: If notification fails, log extensively but don't fail job
                    # Server will eventually recover via timeout/recovery system
                    logger.error(
                        f"[{job_id}] CRITICAL: Failed to notify controller after retries: {notify_error}. "
                        f"Job completed but server may not know. Result at: {result_key}"
                    )
                    # Don't raise - job actually succeeded, just notification failed
            elif is_autotest:
                logger.info(f"[{job_id}] Autotest job - skipping controller notification")

            # Mark as completed
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
            
            # Clean up completed job after delay
            asyncio.create_task(_cleanup_job_status(job_id, delay=30))
            
            # Check GPU memory after job
            _check_gpu_memory(job_id)
            
    except Exception as e:
        # Comprehensive error handling
        error_str = str(e)
        error_type = type(e).__name__
        completion_time = datetime.utcnow()
        processing_time = (completion_time - start_time).total_seconds()
        
        logger.exception(f"[{job_id}] Job failed after {processing_time:.1f}s: {error_type}: {error_str}")
        
        # Check if it's a GPU OOM error
        is_oom = "out of memory" in error_str.lower() or "oom" in error_str.lower()
        if is_oom:
            logger.error(f"[{job_id}] GPU OOM detected! Cleaning up GPU memory...")
            cleanup_gpu(verbose=True)
        
        # Update status
        async with job_status_lock:
            job_status[job_id] = {
                "status": "failed", 
                "error": error_str,
                "error_type": error_type,
                "is_oom": is_oom,
                "finished_at": completion_time.isoformat(),
                "processing_time": processing_time,
            }
        
        # Notify controller with retry (skip for autotest jobs - they're not on the server)
        is_autotest = job_id.startswith("autotest-")
        if CONTROLLER_URL and not is_autotest:
            fail_metrics = {}
            async with _first_inference_lock:
                if FIRST_INFERENCE_START_TIME is None:
                    FIRST_INFERENCE_START_TIME = time.time()
                    fail_metrics["startup_time_seconds"] = round(FIRST_INFERENCE_START_TIME - POD_START_TIME, 2)
            payload = {
                "job_id": job_id, 
                "result_r2_key": None, 
                "failed": True, 
                "error": error_str,
                "next_job_id": _peek_next_job_id(),
                "metrics": fail_metrics if fail_metrics else None,
            }
            try:
                url = f"{CONTROLLER_URL.rstrip('/')}/job_complete"
                # post_signed_json(url, payload, timeout=10, max_retries=5)
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
                    f"[{job_id}] CRITICAL: Failed to notify controller of failure: {notify_error}. "
                    f"Job failed but server may not know."
                )
        elif is_autotest:
            logger.info(f"[{job_id}] Autotest job failed - skipping controller notification")
        
        # Clean up failed job after delay
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


def _check_gpu_memory(job_id: str):
    """
    Check GPU memory usage and log warning if high.
    Helps detect memory leaks early.
    """
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3
            
            if allocated > 20:  # Warning if > 20GB allocated
                logger.warning(
                    f"[{job_id}] High GPU memory usage: {allocated:.2f}GB allocated, "
                    f"{reserved:.2f}GB reserved"
                )
            else:
                logger.debug(
                    f"[{job_id}] GPU memory: {allocated:.2f}GB allocated, "
                    f"{reserved:.2f}GB reserved"
                )
    except Exception as e:
        logger.debug(f"[{job_id}] GPU memory check failed: {e}")


async def _cleanup_job_status(job_id: str, delay: int = 30):
    """
    Clean up job status after a delay to prevent 409 conflicts.
    Keeps status for a short time to allow status queries from server.
    """
    await asyncio.sleep(delay)
    async with job_status_lock:
        if job_id in job_status:
            status = job_status[job_id].get("status")
            if status in ("completed", "failed"):
                logger.debug(f"[{job_id}] Cleaning up job_status after {delay}s")
                del job_status[job_id]


async def worker_loop():
    """
    Main worker loop with comprehensive error handling.
    
    Ensures worker never crashes even if job processing fails catastrophically.
    """
    logger.info("Starting worker loop")
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    while True:
        try:
            item = await job_queue.get()
            job_id = item.get("job_id", "unknown")
            
            try:
                logger.debug(f"[{job_id}] Dequeued job from queue (queue size: {job_queue.qsize()})")
                await process_job_item(item)
                consecutive_errors = 0  # Reset on success
                
            except Exception as e:
                consecutive_errors += 1
                logger.exception(
                    f"[{job_id}] Critical error processing job "
                    f"(consecutive errors: {consecutive_errors}/{max_consecutive_errors}): {e}"
                )
                
                # If too many consecutive errors, something is seriously wrong
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(
                        f"Worker has {consecutive_errors} consecutive errors! "
                        f"Cleaning up GPU and resetting error counter..."
                    )
                    try:
                        cleanup_gpu(verbose=True)
                        consecutive_errors = 0
                        logger.info("GPU cleanup completed, continuing operations")
                    except Exception as cleanup_error:
                        logger.error(f"GPU cleanup failed: {cleanup_error}")
                
                # Always mark job done even if processing failed
                # This prevents queue from getting stuck
                
            finally:
                job_queue.task_done()
                
        except Exception as outer_error:
            # Catch-all for any errors in the loop itself
            logger.exception(f"Critical error in worker loop: {outer_error}")
            # Don't exit - keep loop running
            await asyncio.sleep(1)  # Brief pause before continuing

# -----------------------
# Auto-test
# -----------------------
async def run_autotest():
    await asyncio.sleep(5)
    logger.info("🚀 Starting Qwen Image Edit 2509 auto-test")
    
    # Check GPU VRAM - disable autotest on low-memory cards to avoid OOM
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram_gb < 30:
            logger.info(f"⚠️ Skipping autotest on {vram_gb:.0f}GB GPU (< 30GB) to avoid OOM conflicts with real jobs")
            return
    except Exception as e:
        logger.warning(f"Could not check GPU VRAM for autotest decision: {e}")
        # Proceed cautiously

    img_url = "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/Snapinsta.app_262322643_4937545659591345_7487236998017046274_n_1080.jpg"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
        "Referer": "https://ibb.co",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    }
    try:
        async with httpx.AsyncClient(headers=headers,timeout=60.0) as client:
            resp = await client.get(img_url)
            resp.raise_for_status()
            img_bytes = resp.content
    except Exception as e:
        logger.error(f"❌ Failed to fetch test image: {e}")
        return

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    test_job_id = f"autotest-qwen-2509-{uuid.uuid4()}"
    params = {
        "cfg_scale": 1.0,
        "output_dir": "./outputs",
        "init_image": img_b64,
        "negative_prompt": " ",
        "num_inference_steps": 4,
        "scale_image": False,  # DISABLED upscaling for autotest to save GPU memory
        "scale_factor": 2,
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            "http://127.0.0.1:8000/run",
            json={
                "job_id": test_job_id,
                "prompt": "Change body orientation to face the camera directly",
                "model": "qwen_image_edit_2509",
                "params": params,
                "upload_url": "http://dummy-upload/skip",
                "result_key": f"{test_job_id}.png",
            },
        )
        logger.info(f"🔥 Auto-test response: {resp.status_code} {resp.text}")

# -----------------------
# Model Preloading
# -----------------------
_model_loading_complete = False
_model_loading_error: Optional[str] = None


def _preload_model_sync():
    """
    Preload the model synchronously in a thread.
    
    This is called from a background thread during startup to avoid:
    1. Blocking the FastAPI event loop
    2. Health check timeouts during model loading
    
    The model loading takes 60-120 seconds on first load.
    After loading, the worker reports "ready" to the controller.
    """
    global _model_loading_complete, _model_loading_error
    
    try:
        logger.info("🔄 Starting model preload in background thread...")
        start_time = time.time()
        
        # Import and call get_pipeline to trigger model loading
        from qwen_image_edit_2509_inference import get_pipeline
        
        # This will load the model (takes 60-120 seconds)
        pipeline = get_pipeline()
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Model preloaded successfully in {elapsed:.1f}s")
        
        _model_loading_complete = True
        _model_loading_error = None
        
    except Exception as e:
        logger.exception(f"❌ Model preload failed: {e}")
        _model_loading_error = str(e)
        _model_loading_complete = False


async def _wait_for_model_and_register():
    """
    Wait for model to load, then register with controller.
    
    This runs as a background task so it doesn't block startup.
    The worker reports "starting" status until model is loaded.
    """
    global startup_completed
    
    # Wait for model loading to complete (check every 2 seconds)
    max_wait = 300  # 5 minutes max wait
    waited = 0
    
    while not _model_loading_complete and waited < max_wait:
        if _model_loading_error:
            logger.error(f"Model loading failed: {_model_loading_error}")
            break
        
        await asyncio.sleep(2)
        waited += 2
        
        if waited % 30 == 0:
            logger.info(f"⏳ Still waiting for model to load... ({waited}s elapsed)")
    
    if not _model_loading_complete:
        logger.error(
            f"Model loading did not complete within {max_wait}s. "
            f"Error: {_model_loading_error or 'timeout'}"
        )
        # Still try to register - controller will see "starting" status
    
    # Mark as ready AFTER model is loaded
    startup_completed = _model_loading_complete
    worker_ready.set()
    
    if startup_completed:
        logger.info("✅ Model loaded, worker is now READY")
    else:
        logger.warning("⚠️ Worker starting but model not fully loaded")
    
    # Register with controller
    try:
        register_with_controller()
        logger.info("✅ Worker registered with controller")
    except Exception as e:
        logger.exception("Failed to register worker with controller: %s", e)
        # Don't crash - worker might still work
    
    # Optional: Run autotest after model is loaded
    if startup_completed and os.getenv("ENABLE_AUTOTEST", "1") == "1":
        asyncio.create_task(run_autotest())


# -----------------------
# Lifespan
# -----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global startup_completed
    
    logger.info("🚀 Application starting up...")
    
    # Start background tasks FIRST (worker loop, heartbeat, health monitor)
    asyncio.create_task(worker_loop())
    asyncio.create_task(_hb_loop())
    asyncio.create_task(_health_monitor_loop())
    
    # CRITICAL: Keep startup_completed = False until model is loaded
    # This tells the server we're still "starting" and not ready for jobs
    startup_completed = False
    
    # Start model loading using the DEDICATED INFERENCE_EXECUTOR
    # This uses the same pool that will handle inference, ensuring:
    # 1. FastAPI event loop is never blocked
    # 2. /health endpoint always responds
    # 3. Model is preloaded in the inference thread for consistency
    loop = asyncio.get_event_loop()
    
    logger.info("🔄 Starting model preload in INFERENCE_EXECUTOR thread...")
    loop.run_in_executor(INFERENCE_EXECUTOR, _preload_model_sync)
    
    # Start background task to wait for model and then register
    asyncio.create_task(_wait_for_model_and_register())
    
    logger.info("📡 Worker starting - model loading in background, health endpoint responding")
    
    yield
    
    # Graceful shutdown
    logger.info("🛑 Worker shutting down gracefully...")
    startup_completed = False
    
    # Wait for queue to drain (with timeout)
    if not job_queue.empty():
        logger.info(f"Waiting for {job_queue.qsize()} remaining jobs to complete...")
        try:
            await asyncio.wait_for(job_queue.join(), timeout=30.0)
            logger.info("All jobs completed successfully")
        except asyncio.TimeoutError:
            logger.warning(f"Shutdown timeout, {job_queue.qsize()} jobs may be incomplete")
    
    # Shutdown inference executor gracefully
    logger.info("Shutting down inference executor...")
    INFERENCE_EXECUTOR.shutdown(wait=True, cancel_futures=False)
    
    # Final GPU cleanup
    logger.info("Final GPU cleanup")
    cleanup_gpu()
    logger.info("✅ Worker shutdown complete")

# -----------------------
# Health monitoring
# -----------------------
async def _health_monitor_loop():
    """
    Periodic health monitoring to detect issues early.
    
    Checks:
    - GPU memory
    - Queue depth
    - Stuck jobs
    - Endpoint reachability
    """
    check_interval = 60  # Check every 60 seconds
    logger.info("Starting health monitor loop")
    
    while True:
        try:
            await asyncio.sleep(check_interval)
            
            # Check GPU memory
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                
                if allocated > 22:  # Critical if > 22GB
                    logger.warning(
                        f"⚠️  HIGH GPU MEMORY: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved. "
                        f"May cause OOM on next job!"
                    )
            
            # Check for stuck jobs
            async with job_status_lock:
                stuck_count = 0
                for jid, status in job_status.items():
                    if status.get("status") == "running":
                        started_at_str = status.get("started_at")
                        if started_at_str:
                            started_at = datetime.fromisoformat(started_at_str)
                            runtime = (datetime.utcnow() - started_at).total_seconds()
                            
                            # Warn if job running > 5 minutes
                            if runtime > 300:
                                logger.warning(
                                    f"⚠️  Job {jid} has been running for {runtime:.0f}s (>5min). "
                                    f"May be stuck!"
                                )
                                stuck_count += 1
                
                if stuck_count > 0:
                    logger.warning(f"⚠️  {stuck_count} potentially stuck job(s) detected")
            
            # Check queue depth
            queue_size = job_queue.qsize()
            if queue_size > job_queue.maxsize * 0.8:  # Warn if > 80% full
                logger.warning(
                    f"⚠️  Queue nearly full: {queue_size}/{job_queue.maxsize} jobs. "
                    f"May start rejecting jobs soon!"
                )
            
            # Periodic status summary
            async with job_status_lock:
                running = sum(1 for j in job_status.values() if j.get("status") == "running")
                queued = sum(1 for j in job_status.values() if j.get("status") == "queued")
            
            logger.debug(
                f"📊 Health check: Queue={queue_size}, Running={running}, Queued={queued}, "
                f"Startup={startup_completed}"
            )
            
        except Exception as e:
            logger.error(f"Health monitor error: {e}")


# -----------------------
# Heartbeat loop
# -----------------------
async def _hb_loop():
    """
    Periodic heartbeat to controller with status updates.
    """
    logger.info("Starting heartbeat loop")
    
    while True:
        try:
            # send_heartbeat()
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
    """
    Accept a new job for processing.
    
    Enhanced logic:
    - Check if job is already queued/running (avoid duplicates)
    - Check if queue is full (prevent overload)
    - Idempotent: accepting same job_id twice returns success if already accepted
    - Validates worker is healthy before accepting
    """
    logger.info(f"[{req.job_id}] Received job request for model={req.model}")
    
    # Use timeout for lock to prevent blocking during heavy inference
    try:
        await asyncio.wait_for(job_status_lock.acquire(), timeout=2.0)
        try:
            # Check if job exists and what state it's in
            if req.job_id in job_status:
                current_status = job_status[req.job_id].get("status")
                
                # If job is already queued or running, return 409
                if current_status in ("queued", "running"):
                    logger.warning(f"[{req.job_id}] Already {current_status}, rejecting duplicate")
                    raise HTTPException(status_code=409, detail=f"Job already {current_status}")
                
                # If job completed/failed, it was cleaned up but server is retrying
                # This is OK - accept as new job (server might have missed completion)
                if current_status in ("completed", "failed"):
                    logger.info(f"[{req.job_id}] Was {current_status}, accepting as new job")
                    del job_status[req.job_id]
        finally:
            job_status_lock.release()
    except asyncio.TimeoutError:
        # Lock timeout - worker is busy but we can still try to accept the job
        logger.warning(f"[{req.job_id}] Lock timeout during duplicate check, proceeding anyway")
    
    # Check if queue is full
    if job_queue.full():
        logger.error(f"[{req.job_id}] Queue full ({job_queue.maxsize}), rejecting")
        raise HTTPException(status_code=503, detail="Worker queue full, try another worker")
    
    # Check if worker is ready (model loaded)
    if not startup_completed:
        if not _model_loading_complete:
            # Model is still loading - reject the job
            logger.warning(f"[{req.job_id}] Model still loading, rejecting job")
            raise HTTPException(
                status_code=503, 
                detail="Worker model still loading, try another worker"
            )
        else:
            logger.warning(f"[{req.job_id}] Worker not fully ready yet, but model loaded - accepting")
    
    # Validate model is supported
    if req.model not in MODEL_REGISTRY:
        logger.error(f"[{req.job_id}] Unknown model: {req.model}")
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")
    
    # Accept the job
    try:
        await job_queue.put(req.dict())
        # Record status asynchronously — DO NOT BLOCK /run
        asyncio.create_task(_record_job_status_later(req.job_id, req.model))
        
        # Update status with timeout to avoid blocking
        # try:
        #     await asyncio.wait_for(job_status_lock.acquire(), timeout=2.0)
        #     try:
        #         job_status[req.job_id] = {
        #             "status": "queued", 
        #             "queued_at": datetime.utcnow().isoformat(),
        #             "model": req.model,
        #         }
        #     finally:
        #         job_status_lock.release()
        # except asyncio.TimeoutError:
        #     # Lock timeout - job is queued but status not tracked (non-critical)
        #     logger.warning(f"[{req.job_id}] Lock timeout updating status, job still queued")
        
        logger.info(f"[{req.job_id}] Enqueued successfully (queue size: {job_queue.qsize()})")
        
        return {
            "status": "accepted",
            "job_id": req.job_id,
            "queue_position": job_queue.qsize(),
            "worker_id": WORKER_ID,
        }
        
    except Exception as e:
        logger.exception(f"[{req.job_id}] Failed to enqueue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run_bulk")
async def run_bulk_endpoint(req: BulkRunRequest):
    """
    Accept multiple jobs for processing in a single request.
    Used for bulk edit and multi-variation requests.
    """
    if not req.jobs:
        raise HTTPException(status_code=400, detail="No jobs provided")
    
    if not startup_completed:
        if not _model_loading_complete:
            raise HTTPException(status_code=503, detail="Worker model still loading, try another worker")
    
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
                if job.job_id in job_status:
                    current_status = job_status[job.job_id].get("status")
                    if current_status in ("queued", "running"):
                        logger.warning(f"[BULK] Job {job.job_id} already {current_status}, skipping")
                        continue
                    if current_status in ("completed", "failed"):
                        del job_status[job.job_id]
            finally:
                job_status_lock.release()
        except asyncio.TimeoutError:
            pass
        
        await job_queue.put(job.dict())
        asyncio.create_task(_record_job_status_later(job.job_id, job.model))
        accepted_ids.append(job.job_id)
    
    logger.info(f"[BULK] Enqueued {len(accepted_ids)} of {len(req.jobs)} jobs (queue size: {job_queue.qsize()})")
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
    """
    CRITICAL: Non-blocking health check that ALWAYS responds quickly.
    
    Enhanced with detailed worker status.
    
    IMPORTANT: This endpoint must NEVER block, even during heavy processing.
    Server uses this to determine if worker is alive - timeout = pod destruction!
    
    Returns:
    - status: "ok" if ready, "starting" if not ready yet, "overloaded" if queue full, "busy" if processing
    - worker_id: unique worker identifier
    - queue_size: current queue size
    - queue_capacity: max queue size
    - active_jobs: number of jobs being processed
    - startup_completed: whether worker finished initialization
    - ready: whether worker can accept new jobs
    - processing: whether worker is currently processing jobs (important for server!)
    """
    try:
        # Use asyncio.wait_for to prevent lock blocking
        # If lock takes > 1s, something is seriously wrong but we still respond
        try:
            # FIXED: asyncio.wait_for returns a coroutine, not an async context manager
            # Must await the lock acquisition, then manually release
            await asyncio.wait_for(job_status_lock.acquire(), timeout=1.0)
            try:
                active_jobs = sum(1 for j in job_status.values() if j.get("status") == "running")
                queued_jobs = sum(1 for j in job_status.values() if j.get("status") == "queued")
            finally:
                job_status_lock.release()
        except asyncio.TimeoutError:
            # Lock timeout - worker is extremely busy, but still alive!
            logger.warning("Health check: job_status_lock timeout, worker is very busy")
            active_jobs = MAX_CONCURRENT_JOBS  # Assume max
            queued_jobs = job_queue.qsize()
        
        # Determine overall status
        # CRITICAL: Use model loading state, not just startup_completed
        if not _model_loading_complete:
            overall_status = "starting"  # Model still loading
        elif not startup_completed:
            overall_status = "starting"  # Model loaded but not fully ready
        elif job_queue.full():
            overall_status = "overloaded"
        elif active_jobs >= MAX_CONCURRENT_JOBS:
            overall_status = "busy"  # IMPORTANT: busy != dead!
        else:
            overall_status = "ok"
        
        # Worker is processing if it has active jobs
        is_processing = active_jobs > 0
        
        return {
            "status": overall_status,
            "worker_id": WORKER_ID,
            "queue_size": job_queue.qsize(),
            "queue_capacity": job_queue.maxsize,
            "active_jobs": active_jobs,
            "queued_jobs": queued_jobs,
            "max_concurrent": MAX_CONCURRENT_JOBS,
            "model_loaded": _model_loading_complete,  # NEW: explicitly show model state
            "model_error": _model_loading_error,  # NEW: show error if any
            "startup_completed": startup_completed,
            "ready": startup_completed and not job_queue.full(),
            "processing": is_processing,  # NEW: critical for server!
            "healthy": True,  # We're responding, so we're healthy!
        }
    except Exception as e:
        # Even if everything fails, respond with minimal info
        # This prevents pod destruction due to health check timeout
        logger.error(f"Health check error: {e}")
        return {
            "status": "busy",  # Assume busy if we can't check
            "worker_id": WORKER_ID,
            "queue_size": 0,
            "queue_capacity": job_queue.maxsize,
            "active_jobs": 1,  # Assume processing
            "healthy": True,  # We're responding!
            "processing": True,
            "error": str(e),
        }


@app.get("/status")
async def status():
    """Detailed status including all jobs"""
    async with job_status_lock:
        jobs_copy = job_status.copy()
    
    return {
        "queue_size": job_queue.qsize(),
        "queue_capacity": job_queue.maxsize,
        "jobs": jobs_copy,
        "worker_id": WORKER_ID,
        "startup_completed": startup_completed,
    }