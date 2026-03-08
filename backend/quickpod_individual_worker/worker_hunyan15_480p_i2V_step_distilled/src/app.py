import os
import re
import sys
import time
import json
import logging
import asyncio
import hmac
import hashlib
import gc
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import uuid
import httpx
import base64
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential, stop_after_attempt
from contextlib import asynccontextmanager
import concurrent.futures

import torch

# -----------------------
# Pod startup time tracking
# -----------------------
POD_START_TIME = time.time()
FIRST_INFERENCE_START_TIME = None
_first_inference_lock = asyncio.Lock()

# -----------------------
# Dedicated Thread Pool for Inference
# -----------------------
# CRITICAL: Use dedicated executor for inference to avoid blocking FastAPI
# Only 1 worker to prevent OOM on 24GB cards (4090/5090)
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
    except Exception as e:
        print(f"[GPU] Cleanup warning: {e}")

# -----------------------
# Import inference functions
# -----------------------
from hunyan15_video_inference import (
    run_inference_hunyan15_i2v,
    unload_pipeline,
    get_gpu_info,
)

# -----------------------
# Model Registry
# -----------------------
MODEL_REGISTRY = {
    "hunyan15_480p_i2v_step_distilled": run_inference_hunyan15_i2v,
}

# -----------------------
# Logging
# -----------------------
# Configure logging to write to both console and file
LOG_FILE = os.getenv("LOG_FILE", "/app/worker/uvicorn.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Create file handler
file_handler = logging.FileHandler(LOG_FILE, mode='a')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
file_handler.setFormatter(file_formatter)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
console_handler.setFormatter(console_formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# Remove any existing handlers to avoid duplicates
root_logger.handlers = []
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Get worker logger
logger = logging.getLogger("worker")

# Create a custom print function that logs to file and prints to console
_original_print = print
def log_print(*args, **kwargs):
    """Print that also writes to log file."""
    message = ' '.join(str(arg) for arg in args)
    # Write to log file via logger
    logger.info(f"[PRINT] {message}")
    # Also print to stdout (which uvicorn will capture)
    _original_print(*args, **kwargs, file=sys.stdout, flush=True)

# Store original print for internal use if needed
import builtins
builtins._original_print = _original_print

# Replace print in this module's namespace
print = log_print

logger.info(f"Logging configured: console + file ({LOG_FILE})")
print(f"Logging initialized: writing to {LOG_FILE}")

# -----------------------
# Configuration from env
# -----------------------
CONTAINER_ID = os.getenv("CONTAINER_LABEL")
POD_UUID = CONTAINER_ID
PUBLIC_IPADDR = os.getenv("PUBLIC_IPADDR")
WORKER_ID = POD_UUID

MODEL_KEY = os.getenv("MODEL_KEY", "hunyan15_480p_i2v_step_distilled")
CONTROLLER_URL = os.getenv("CONTROLLER_URL", "").strip()
CONTROLLER_SHARED_SECRET = os.getenv("CONTROLLER_SHARED_SECRET", "")
CONTROLLER_API_KEY = os.getenv("CONTROLLER_API_KEY", "")
WORKER_PUBLIC_URL = PUBLIC_IPADDR
INTERNAL_PORT = 8000
EXTERNAL_PORT = os.getenv(f"QUICKPOD_PORT_{INTERNAL_PORT}")
PORT = int(EXTERNAL_PORT) if EXTERNAL_PORT else INTERNAL_PORT

HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "20"))
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))
DEFAULT_PIPELINE = os.getenv("HUNYAN15_DEFAULT_PIPELINE", "i2v")

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
startup_completed = True  # Qwen-style: ready immediately, pipeline loads on first job


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
    """Request model for video generation jobs."""
    job_id: str = Field(..., description="Unique job identifier")
    prompt: str = Field(..., description="Text prompt for video generation")
    model: str = Field(..., description="Model key (hunyan15_480p_i2v_step_distilled)")
    mode: Optional[str] = Field(None, description="Generation mode: image-to-video step distilled 480p")
    params: Dict[str, Any] = Field(default_factory=dict, description="Generation parameters")
    upload_url: str = Field(..., description="URL to upload result")
    result_key: str = Field(..., description="R2 key for result")


class BulkRunRequest(BaseModel):
    jobs: List[RunRequest]


# -----------------------
# Media URL Resolution
# -----------------------
# Fields that may contain URLs (presigned R2 GET URLs) instead of base64 data.
# init_image / image: download to local path and pass path (no base64 encode/decode).
# Other media fields: download and convert to base64 for backward compatibility.
MEDIA_FIELDS = frozenset([
    "init_image", "image", "mask_image", "source_image", "face_image",
    "reference_image", "style_image", "control_image", "input_image",
    "ref_image", "ref_video", "input_video",
])
# I2V: resolve by downloading to path so inference gets path (no base64)
INIT_IMAGE_FIELDS = frozenset(["init_image", "image"])


def _is_url(value: str) -> bool:
    return isinstance(value, str) and (value.startswith("http://") or value.startswith("https://"))


def _download_to_base64(url: str, timeout: int = 120) -> str:
    """Download binary from URL and return base64-encoded string."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return base64.b64encode(resp.content).decode("utf-8")


def _download_url_to_path(url: str, dest_path: str, timeout: int = 120) -> None:
    """Download binary from URL and write to file. No base64."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)


def _sanitize_job_id_for_path(job_id: str) -> str:
    """Make job_id safe for use in filenames."""
    return re.sub(r"[^\w\-.]", "_", job_id)


def resolve_media_urls(params: Dict[str, Any], job_id: str = "") -> Dict[str, Any]:
    """
    Resolve URL-valued media fields: init_image/image -> download to path (no base64).
    Other media fields -> download and replace with base64 for downstream code.
    """
    resolved = dict(params)
    for field in MEDIA_FIELDS:
        value = resolved.get(field)
        if value is None:
            continue
        if isinstance(value, str) and _is_url(value):
            try:
                if field in INIT_IMAGE_FIELDS:
                    # Download to path; inference will use path (no base64)
                    ext = ".jpg"
                    safe_id = _sanitize_job_id_for_path(job_id or "job")
                    path = os.path.join(WORKSPACE_DIR, f"init_{safe_id}{ext}")
                    logger.info(f"[{job_id}] Downloading {field} from R2 to path ({len(value)} chars)...")
                    _download_url_to_path(value, path)
                    resolved[field] = path
                    logger.info(f"[{job_id}] Downloaded {field} -> {path}")
                else:
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
                        if field in INIT_IMAGE_FIELDS:
                            ext = ".jpg"
                            safe_id = _sanitize_job_id_for_path(f"{job_id or 'job'}_{i}")
                            path = os.path.join(WORKSPACE_DIR, f"init_{safe_id}{ext}")
                            logger.info(f"[{job_id}] Downloading {field}[{i}] from R2 to path...")
                            _download_url_to_path(item, path)
                            new_list.append(path)
                        else:
                            logger.info(f"[{job_id}] Downloading {field}[{i}] from URL...")
                            new_list.append(_download_to_base64(item))
                    except Exception as e:
                        logger.error(f"[{job_id}] Failed to download {field}[{i}]: {e}")
                        raise
                else:
                    new_list.append(item)
            resolved[field] = new_list
    return resolved

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    worker_id: str
    queue_size: int
    queue_capacity: int
    active_jobs: int
    queued_jobs: int
    max_concurrent: int
    model_loaded: bool
    model_error: Optional[str]
    startup_completed: bool
    ready: bool
    processing: bool
    healthy: bool
    gpu_info: Optional[Dict[str, Any]] = None


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
        except requests.exceptions.ConnectionError as e:
            last_error = e
            if attempt < max_retries - 1:
                backoff = 2 ** attempt
                logger.warning(f"POST to {url} connection error (attempt {attempt + 1}/{max_retries}), retrying in {backoff}s...")
                time.sleep(backoff)
        except requests.exceptions.HTTPError as e:
            if e.response and 400 <= e.response.status_code < 500:
                logger.error(f"POST to {url} failed with {e.response.status_code}: {e}")
                raise
            last_error = e
            if attempt < max_retries - 1:
                backoff = 2 ** attempt
                logger.warning(f"POST to {url} server error (attempt {attempt + 1}/{max_retries}), retrying in {backoff}s...")
                time.sleep(backoff)
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
    """Register worker with controller."""
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
            "gpu_info": get_gpu_info(),
        },
    }
    url = f"{CONTROLLER_URL.rstrip('/')}/register_worker"
    
    logger.info(f"Registering worker with controller at {CONTROLLER_URL}")
    logger.info(f"Worker endpoint: http://{PUBLIC_IPADDR}:{PORT}")
    logger.info(f"Worker ID: {WORKER_ID}")
    
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
    """Send heartbeat to controller."""
    if not CONTROLLER_URL:
        return {"skipped": True}
    
    try:
        queue_size = job_queue.qsize()
        active_jobs = sum(1 for j in job_status.values() if j.get("status") == "running")
    except Exception:
        queue_size = 0
        active_jobs = 0
    
    # Determine status based on job activity (Qwen-style: always ready)
    status = "busy" if (active_jobs > 0 or queue_size > 0) else "ready"
    
    payload = {
        "worker_id": WORKER_ID,
        "pod_uuid": POD_UUID,
        "timestamp": int(time.time()),
        "status": status,
        "model_loaded": True,
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
# Together AI Prompt Enhancement
# -----------------------
TOGETHER_API_KEY = os.environ.get(
    "TOGETHER_API_KEY",
    "no_api_key",
)
TOGETHER_MODEL = "ServiceNow-AI/Apriel-1.6-15b-Thinker"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

_ENHANCE_SYSTEM_PROMPT_VIDEO_I2V = (
    "You are an expert cinematic prompt engineer specialized in enhancing prompts for AI image-to-video generation models.\n\n"
    "Your task is NOT to change the user's intent, but to expand and refine it into a highly detailed, visually descriptive, and generation-optimized prompt.\n\n"
    "Follow these principles:\n\n"
    "1. Preserve Core Intent\n- Never alter the main subject or meaning.\n- Only enhance clarity, detail, and visual richness.\n"
    "- The input image is the primary visual reference; the prompt should describe desired motion and changes.\n\n"
    "2. Add Missing Visual Information\nEnhance prompts with:\n- Subject description (appearance, clothing, age, materials, textures)\n"
    "- Environment and background\n- Lighting (cinematic lighting, volumetric light, rim light, neon glow, etc.)\n"
    "- Camera details (lens, depth of field, perspective, framing)\n- Mood and atmosphere\n- Color palette\n- Composition\n"
    "- Motion (for video prompts)\n- Time of day / weather\n- Style references if appropriate\n\n"
    "3. Cinematic & Photorealistic Quality Boost\nWhen suitable, include:\n- ultra detailed\n- cinematic\n- photorealistic\n"
    "- high dynamic range\n- global illumination\n- 8k / high resolution\n- shallow depth of field\n"
    "- film grain\n- realistic shadows\n\n"
    "4. Video Prompt Enhancement\nAdd:\n- camera movement (slow pan, dolly, handheld, drone, tracking shot)\n"
    "- subject motion\n- environmental motion\n- temporal continuity\n- physics realism\n- cinematic pacing\n\n"
    "5. Editing Requests\nClearly describe:\n- what remains unchanged from the input image\n- what is modified or animated\n"
    "- integration realism\n- lighting consistency\n- perspective matching\n\n"
    "6. Avoid Negative Behavior\nDo NOT:\n- add unrelated objects\n- change gender, ethnicity, or identity unless asked\n"
    "- introduce unwanted elements\n- overcomplicate simple prompts\n\n"
    "7. Output Format Rules\nAlways output:\n\nENHANCED PROMPT:\n<final enhanced prompt>\n\n"
    "Optional sections when useful:\n\nNEGATIVE PROMPT:\n<artifacts to avoid>\n\n"
    "VIDEO PARAMETERS:\n<only if video related>\n\n"
    "Do NOT explain reasoning.\nDo NOT include commentary.\nOnly output the final structured result.\n\n"
    "8. Style Intelligence\nAutomatically infer best style based on user intent.\n\n"
    "9. Model Awareness\nOptimize prompts to work well with video generation models such as:\nWAN, Hunyuan, LTX, Sora-style pipelines.\n\n"
    "10. Precision Over Length\nBe descriptive but meaningful. Avoid useless adjectives.\n\n"
    "Your goal: produce prompts that generate visually stunning, coherent, and realistic videos from images with minimal artifacts."
)


def _enhance_prompt(prompt: str, job_id: str) -> str:
    """
    Enhance a prompt using Together AI LLM for video I2V generation.

    Returns enhanced prompt, or original prompt on failure.
    """
    if TOGETHER_API_KEY == "no_api_key":
        logger.warning("PROMPT ENHANCE REQUESTED, SKIPPING.. NO API KEY SET IN ENV VAR")
        return prompt

    logger.info("[%s] ENHANCE: calling Together AI (%s) to enhance prompt (type=video_i2v)", job_id, TOGETHER_MODEL)
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
                    {"role": "system", "content": _ENHANCE_SYSTEM_PROMPT_VIDEO_I2V},
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
    """Run inference in dedicated thread pool. Returns (output_path, metrics_dict)."""
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

    # Prompt enhancement for video I2V
    if params.get("enhance_prompt", False):
        prompt = await loop.run_in_executor(IO_EXECUTOR, _enhance_prompt, prompt, job_id)
    
    if mode:
        params.setdefault("mode", mode)

    inference_func = resolve_inference_callable(model, mode)
    model_key = model

    if last_model_key and last_model_key != model_key:
        logger.info(f"Switching model {last_model_key} -> {model_key}, cleaning up GPU first")
        cleanup_gpu()

    inference_start_time = time.time()
    try:
        image_input = params.pop("init_image", None) or params.pop("image", None)
        if model == "hunyan15_480p_i2v_step_distilled":
            if not image_input:
                raise ValueError(
                    "image_input is required for i2v models. "
                    "Provide 'init_image' or 'image' in params (presigned R2 download URL)."
                )
            out_path = await loop.run_in_executor(
                INFERENCE_EXECUTOR, inference_func, job_id, prompt, image_input, params
            )
        else:
            out_path = await loop.run_in_executor(
                INFERENCE_EXECUTOR, inference_func, job_id, prompt, params
            )
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
    """Process a single job with comprehensive error handling."""
    global FIRST_INFERENCE_START_TIME
    job_id = item["job_id"]
    model = item["model"]
    mode = item.get("mode")
    prompt = item["prompt"]
    params = item.get("params", {})
    upload_url = item["upload_url"]
    result_key = item["result_key"]
    
    start_time = datetime.utcnow()
    logger.info(f"[{job_id}] Processing job with model={model}, mode={mode or 'default'}")
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
            "mode": mode,
        }

    try:
        async with semaphore:
            logger.info(f"[{job_id}] Starting inference...")
            logger.info(f"[{job_id}] Inference parameters: model={model}, mode={mode}, output_dir={params.get('output_dir', 'default')}")
            try:
                out_local, metrics = await run_inference_and_save(job_id, prompt, model, mode, params)
                logger.info(f"[{job_id}] ✅ Inference completed successfully, output: {out_local}")
                
                # Verify output file exists
                if out_local and os.path.exists(out_local):
                    file_size = os.path.getsize(out_local) / 1024 / 1024  # MB
                    logger.info(f"[{job_id}] ✅ Output file verified: {out_local} ({file_size:.2f} MB)")
                else:
                    logger.warning(f"[{job_id}] ⚠️ Output file not found: {out_local}")
            except Exception as inference_error:
                error_type = type(inference_error).__name__
                error_msg = str(inference_error)
                logger.error(f"[{job_id}] ❌ Inference error: {error_type}: {error_msg}")
                logger.exception(f"[{job_id}] Full inference traceback (logged to uvicorn.log):")
                raise

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
                payload = {"job_id": job_id, "result_r2_key": result_key, "next_job_id": _peek_next_job_id(), "metrics": metrics}
                url = f"{CONTROLLER_URL.rstrip('/')}/job_complete"
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(IO_EXECUTOR, post_signed_json, url, payload, 10, 5)
                    logger.info(f"[{job_id}] Controller notified successfully")
                except Exception as notify_error:
                    logger.error(f"[{job_id}] CRITICAL: Failed to notify controller: {notify_error}")
            elif is_autotest:
                logger.info(f"[{job_id}] Autotest job - skipping controller notification")

            # Mark completed
            completion_time = datetime.utcnow()
            processing_time = metrics.get("processing_time_seconds", (completion_time - start_time).total_seconds())
            
            async with job_status_lock:
                job_status[job_id] = {
                    "status": "completed",
                    "result_r2_key": result_key,
                    "finished_at": completion_time.isoformat(),
                    "processing_time": processing_time,
                }
            
            logger.info(f"[{job_id}] Job completed successfully in {processing_time:.1f}s")
            asyncio.create_task(_cleanup_job_status(job_id, delay=30))
            _check_gpu_memory(job_id)
            
    except Exception as e:
        error_str = str(e)
        error_type = type(e).__name__
        completion_time = datetime.utcnow()
        processing_time = (completion_time - start_time).total_seconds()
        
        logger.exception(f"[{job_id}] Job failed after {processing_time:.1f}s: {error_type}: {error_str}")
        
        is_oom = "out of memory" in error_str.lower() or "oom" in error_str.lower()
        if is_oom:
            logger.error(f"[{job_id}] GPU OOM detected! Cleaning up...")
            cleanup_gpu(verbose=True)
        
        async with job_status_lock:
            job_status[job_id] = {
                "status": "failed", 
                "error": error_str,
                "error_type": error_type,
                "is_oom": is_oom,
                "finished_at": completion_time.isoformat(),
                "processing_time": processing_time,
            }
        
        is_autotest = job_id.startswith("autotest-")
        if CONTROLLER_URL and not is_autotest:
            fail_metrics = {}
            async with _first_inference_lock:
                if FIRST_INFERENCE_START_TIME is None:
                    FIRST_INFERENCE_START_TIME = time.time()
                    fail_metrics["startup_time_seconds"] = round(FIRST_INFERENCE_START_TIME - POD_START_TIME, 2)
            payload = {"job_id": job_id, "result_r2_key": None, "failed": True, "error": error_str, "next_job_id": _peek_next_job_id(), "metrics": fail_metrics if fail_metrics else None}
            try:
                url = f"{CONTROLLER_URL.rstrip('/')}/job_complete"
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(IO_EXECUTOR, post_signed_json, url, payload, 10, 5)
                logger.info(f"[{job_id}] Failure notification sent to controller")
            except Exception as notify_error:
                logger.error(f"[{job_id}] CRITICAL: Failed to notify controller of failure: {notify_error}")
        
        asyncio.create_task(_cleanup_job_status(job_id, delay=60))


async def _upload_with_retry(file_path: str, upload_url: str, job_id: str, max_retries: int = 3) -> bool:
    """Upload file with retry logic."""
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
                logger.warning(f"[{job_id}] Upload failed ({attempt + 1}/{max_retries}): {e}, retrying in {backoff}s")
                await asyncio.sleep(backoff)
            else:
                logger.error(f"[{job_id}] Upload failed after {max_retries} attempts: {e}")
                return False
    return False


def _check_gpu_memory(job_id: str):
    """Check GPU memory usage and warn if high."""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            
            if allocated > 20:
                logger.warning(f"[{job_id}] High GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            else:
                logger.debug(f"[{job_id}] GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    except Exception as e:
        logger.debug(f"[{job_id}] GPU memory check failed: {e}")


async def _cleanup_job_status(job_id: str, delay: int = 30):
    """Clean up job status after delay."""
    await asyncio.sleep(delay)
    async with job_status_lock:
        if job_id in job_status:
            status = job_status[job_id].get("status")
            if status in ("completed", "failed"):
                logger.debug(f"[{job_id}] Cleaning up job_status after {delay}s")
                del job_status[job_id]


async def worker_loop():
    """Main worker loop."""
    logger.info("Starting worker loop")
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    while True:
        try:
            item = await job_queue.get()
            job_id = item.get("job_id", "unknown")
            
            try:
                logger.debug(f"[{job_id}] Dequeued job (queue size: {job_queue.qsize()})")
                await process_job_item(item)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                logger.exception(f"[{job_id}] Critical error (errors: {consecutive_errors}/{max_consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(f"Worker has {consecutive_errors} consecutive errors! Cleaning up GPU...")
                    try:
                        cleanup_gpu(verbose=True)
                        consecutive_errors = 0
                        logger.info("GPU cleanup completed, continuing operations")
                    except Exception as cleanup_error:
                        logger.error(f"GPU cleanup failed: {cleanup_error}")
            finally:
                job_queue.task_done()
                
        except Exception as outer_error:
            logger.exception(f"Critical error in worker loop: {outer_error}")
            await asyncio.sleep(1)


# -----------------------
# Auto-test Helper Functions
# -----------------------
async def download_image_as_base64(url: str, client: httpx.AsyncClient) -> str:
    """Download image from URL and convert to base64."""
    try:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        raise


# -----------------------
# Auto-test
# -----------------------
async def run_autotest():
    """
    Run comprehensive auto-test to verify all worker modes are functioning.
    
    Tests:
    1. Image-to-Video (I2V) - Animate an image (downloads and converts to base64)
    """
    logger.info("🚀 Autotest function called - starting in 5 seconds...")
    print("AUTOTEST: Function called - starting in 5 seconds...")
    await asyncio.sleep(5)
    logger.info("🚀 Starting Hunyan15 480p I2V Step Distilled Video comprehensive auto-test")
    print("=" * 80)
    print("AUTOTEST: Starting comprehensive auto-test")
    print("=" * 80)
    
    # Check GPU VRAM
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram_gb < 20:
            logger.info(f"⚠️ Skipping autotest on {vram_gb:.0f}GB GPU (<20GB)")
            return
        logger.info(f"✅ GPU VRAM: {vram_gb:.1f}GB - sufficient for autotest")
    except Exception as e:
        logger.warning(f"Could not check GPU VRAM: {e}")

    # Test assets (public URLs for testing)
    TEST_IMAGE_URL = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=768&h=512&fit=crop"
    TEST_VIDEO_URL = None  # V2V test will be skipped if no test video available
    
    autotest_results = {
       
        "i2v": None,
      
    }

    async with httpx.AsyncClient(timeout=600.0) as client:
        
        # Pre-download test images and convert to base64
        logger.info("📥 Downloading test images...")
        test_image_b64 = None
       
        
        try:
            test_image_b64 = await download_image_as_base64(TEST_IMAGE_URL, client)
            logger.info(f"✅ Test image downloaded ({len(test_image_b64)//1024}KB base64)")
        except Exception as e:
            logger.error(f"❌ Failed to download test image: {e}")
        
        # ═══════════════════════════════════════════════════════════════
        # TEST 1: Image-to-Video (I2V)
        # ═══════════════════════════════════════════════════════════════
        logger.info("━" * 50)
        logger.info("🖼️ TEST 1/1: Image-to-Video (I2V)")
        logger.info("━" * 50)
        
        if test_image_b64:
            i2v_job_id = f"autotest-hunyan15_480p_i2V_step_distilled-i2v-{uuid.uuid4().hex[:8]}"
            
            # Create autotest output directory with absolute path
            autotest_output_dir = os.path.join(WORKSPACE_DIR, "autotest_outputs")
            os.makedirs(autotest_output_dir, exist_ok=True)
            logger.info(f"📁 Autotest output directory: {autotest_output_dir}")
            
            # init_image as URL (R2/presigned style) - no base64
            i2v_params = {
                "mode": "image-to-video 480p i2v step distilled",
                "height": 480,
                "width": 768,
                "duration": 5.0,
                "num_inference_steps": 12,
                "cfg_scale": 7.5,
                "init_image": TEST_IMAGE_URL,
                "output_dir": autotest_output_dir,
            }
            
            try:
                resp = await client.post(
                    "http://127.0.0.1:8000/run",
                    json={
                        "job_id": i2v_job_id,
                        "prompt": "The mountains come alive with flowing clouds and gentle wind, cinematic movement",
                        "model": "hunyan15_480p_i2v_step_distilled",
                        "mode": "image-to-video",
                        "params": i2v_params,
                        "upload_url": "http://dummy-upload/skip",
                        "result_key": f"{i2v_job_id}.mp4",
                    },
                )
                autotest_results["i2v"] = resp.status_code == 200
                logger.info(f"{'✅' if resp.status_code == 200 else '❌'} I2V test: {resp.status_code}")
                
                if resp.status_code == 200:
                    # Wait for job to complete and check if video file exists
                    expected_video_path = os.path.join(autotest_output_dir, f"i2v_{i2v_job_id}.mp4")
                    logger.info(f"⏳ Waiting for video generation... (checking: {expected_video_path})")
                    
                    # Wait up to 5 minutes for video to be generated (video generation can take 2-4 minutes)
                    max_wait = 300
                    waited = 0
                    video_found = False
                    job_completed = False
                    
                    while waited < max_wait:
                        await asyncio.sleep(5)
                        waited += 5
                        
                        # Check if video file exists
                        if os.path.exists(expected_video_path):
                            file_size = os.path.getsize(expected_video_path)
                            logger.info(f"✅ Autotest video saved: {expected_video_path} ({file_size / 1024 / 1024:.2f} MB)")
                            video_found = True
                            break
                        
                        # Check job status
                        async with job_status_lock:
                            job_info = job_status.get(i2v_job_id, {})
                            job_status_str = job_info.get("status", "unknown")
                            
                            if job_status_str == "completed":
                                job_completed = True
                                if not video_found:
                                    # Job completed but video not found - check for errors
                                    if "error" in job_info:
                                        error_msg = job_info.get("error", "Unknown error")
                                        error_type = job_info.get("error_type", "Unknown")
                                        logger.error(f"❌ Job completed with error: {error_type}: {error_msg}")
                                        autotest_results["i2v"] = False
                                    else:
                                        logger.warning(f"⚠️ Job marked as completed but video file not found")
                                        autotest_results["i2v"] = False
                                else:
                                    # Job completed and video found
                                    autotest_results["i2v"] = True
                                break
                            elif job_status_str == "failed":
                                error_msg = job_info.get("error", "Unknown error")
                                error_type = job_info.get("error_type", "Unknown")
                                processing_time = job_info.get("processing_time", 0)
                                logger.error(f"❌ Job failed after {processing_time:.1f}s: {error_type}: {error_msg}")
                                logger.error(f"❌ Full error details logged to uvicorn.log")
                                autotest_results["i2v"] = False
                                break
                        
                        if waited % 30 == 0:
                            logger.info(f"⏳ Still waiting for video... ({waited}s elapsed, job status: {job_status_str if 'job_status_str' in locals() else 'checking...'})")
                    
                    # Final status check
                    if not video_found and not job_completed:
                        # Check job status one more time
                        async with job_status_lock:
                            final_job_info = job_status.get(i2v_job_id, {})
                            final_status = final_job_info.get("status", "unknown")
                            
                            if final_status == "failed":
                                error_msg = final_job_info.get("error", "Unknown error")
                                error_type = final_job_info.get("error_type", "Unknown")
                                logger.error(f"❌ Job failed: {error_type}: {error_msg}")
                                logger.error(f"❌ Check uvicorn.log for full error traceback")
                                autotest_results["i2v"] = False
                            elif final_status == "completed":
                                logger.warning(f"⚠️ Job completed but video file not found: {expected_video_path}")
                                autotest_results["i2v"] = False
                            else:
                                logger.warning(f"⚠️ Video file not found after {max_wait}s: {expected_video_path}")
                                logger.warning(f"⚠️ Job status: {final_status} (may still be processing)")
                                autotest_results["i2v"] = False
                        
                        # List files in output directory for debugging
                        if os.path.exists(autotest_output_dir):
                            files = os.listdir(autotest_output_dir)
                            logger.info(f"📂 Files in output directory: {files}")
                        else:
                            logger.warning(f"⚠️ Output directory does not exist: {autotest_output_dir}")
                        
                        # Check all files in workspace for debugging
                        if os.path.exists(WORKSPACE_DIR):
                            all_files = []
                            for root, dirs, files in os.walk(WORKSPACE_DIR):
                                for f in files:
                                    if f.endswith('.mp4'):
                                        all_files.append(os.path.join(root, f))
                            if all_files:
                                logger.info(f"📂 Found MP4 files in workspace: {all_files}")
                else:
                    logger.error(f"❌ I2V test failed with status {resp.status_code}")
                    
            except Exception as e:
                autotest_results["i2v"] = False
                logger.error(f"❌ I2V test failed: {e}")
                import traceback
                logger.exception("Full traceback:")
        else:
            logger.info("⏩ I2V test skipped (failed to download test image)")
            autotest_results["i2v"] = "skipped"
        
        
    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    logger.info("━" * 50)
    logger.info("📊 AUTO-TEST SUMMARY")
    logger.info("━" * 50)
    
    passed = sum(1 for v in autotest_results.values() if v is True)
    skipped = sum(1 for v in autotest_results.values() if v == "skipped")
    failed = sum(1 for v in autotest_results.values() if v is False)
    total = len(autotest_results)
    
    for test_name, result in autotest_results.items():
        if result is True:
            status = "✅ PASS"
        elif result == "skipped":
            status = "⏩ SKIP"
        else:
            status = "❌ FAIL"
        logger.info(f"  {test_name.upper():12s} : {status}")
    
    logger.info("━" * 50)
    logger.info(f"📈 Results: {passed}/{total} passed, {skipped} skipped, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed! Worker is fully operational.")
    else:
        logger.warning(f"⚠️ {failed} test(s) failed. Check logs for details.")


# -----------------------
# Health monitoring
# -----------------------
async def _health_monitor_loop():
    """Periodic health monitoring."""
    check_interval = 60
    logger.info("Starting health monitor loop")
    
    while True:
        try:
            await asyncio.sleep(check_interval)
            
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                
                if allocated > 22:
                    logger.warning(f"⚠️ HIGH GPU MEMORY: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            # Check for stuck jobs
            async with job_status_lock:
                for jid, status in job_status.items():
                    if status.get("status") == "running":
                        started_at_str = status.get("started_at")
                        if started_at_str:
                            started_at = datetime.fromisoformat(started_at_str)
                            runtime = (datetime.utcnow() - started_at).total_seconds()
                            
                            if runtime > 600:  # 10 minutes for video
                                logger.warning(f"⚠️ Job {jid} running for {runtime:.0f}s (>10min)")
            
            queue_size = job_queue.qsize()
            if queue_size > job_queue.maxsize * 0.8:
                logger.warning(f"⚠️ Queue nearly full: {queue_size}/{job_queue.maxsize}")
                
        except Exception as e:
            logger.error(f"Health monitor error: {e}")


async def _hb_loop():
    """Heartbeat loop."""
    logger.info("Starting heartbeat loop")
    
    while True:
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(IO_EXECUTOR, send_heartbeat)
        except Exception as e:
            logger.exception("Heartbeat failed: %s", e)
        
        await asyncio.sleep(HEARTBEAT_INTERVAL)


async def _record_job_status_later(job_id: str, model: str):
    """Record job status asynchronously."""
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
# Lifespan
# -----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Hunyan15 480p I2V Step Distilled Worker starting up...")
    print("=" * 80)
    print("WORKER STARTUP: Hunyan15 480p I2V Step Distilled Worker")
    print("=" * 80)
    
    # Start background tasks
    asyncio.create_task(worker_loop())
    asyncio.create_task(_hb_loop())
    asyncio.create_task(_health_monitor_loop())
    
    # Qwen-style: register immediately, pipeline loads on first job
    worker_ready.set()
    try:
        register_with_controller()
        logger.info("✅ Worker registered with controller")
    except Exception as e:
        logger.exception("Failed to register worker: %s", e)
    
    # Run autotest if enabled
    if os.getenv("ENABLE_AUTOTEST", "1") == "1":
        asyncio.create_task(run_autotest())
    
    logger.info("📡 Worker ready - accepting jobs (pipeline loads on first inference)")
    print("WORKER: Ready, accepting jobs...")
    
    yield
    
    # Shutdown
    logger.info("🛑 Worker shutting down gracefully...")
    startup_completed = False
    
    if not job_queue.empty():
        logger.info(f"Waiting for {job_queue.qsize()} remaining jobs...")
        try:
            await asyncio.wait_for(job_queue.join(), timeout=30.0)
            logger.info("All jobs completed")
        except asyncio.TimeoutError:
            logger.warning(f"Shutdown timeout, {job_queue.qsize()} jobs may be incomplete")
    
    # Unload models
    logger.info("Unloading models...")
    unload_pipeline()
    
    # Shutdown executors
    logger.info("Shutting down inference executor...")
    INFERENCE_EXECUTOR.shutdown(wait=True, cancel_futures=False)
    IO_EXECUTOR.shutdown(wait=False)
    
    cleanup_gpu()
    logger.info("✅ Worker shutdown complete")


# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(
    title=f"QuickPod Worker ({MODEL_KEY})",
    description="Hunyan15 480p Image-to-Video Step Distilled Worker",
    version="2.0.0",
    lifespan=lifespan
)


@app.post("/run")
async def run_endpoint(req: RunRequest):
    """
    Submit a video generation job.
    
    Modes:
    - text-to-video: Generate video from text prompt
    - image-to-video: Animate an image with text guidance
    - video-to-video: Transform a video with text guidance
    - keyframe-interpolation: Interpolate between keyframe images
    """
    logger.info(f"[{req.job_id}] Received job request for model={req.model}, mode={req.mode}")
    
    try:
        await asyncio.wait_for(job_status_lock.acquire(), timeout=2.0)
        try:
            if req.job_id in job_status:
                current_status = job_status[req.job_id].get("status")
                
                if current_status in ("queued", "running"):
                    logger.warning(f"[{req.job_id}] Already {current_status}")
                    raise HTTPException(status_code=409, detail=f"Job already {current_status}")
                
                if current_status in ("completed", "failed"):
                    logger.info(f"[{req.job_id}] Was {current_status}, accepting as new job")
                    del job_status[req.job_id]
        finally:
            job_status_lock.release()
    except asyncio.TimeoutError:
        logger.warning(f"[{req.job_id}] Lock timeout during duplicate check")
    
    if job_queue.full():
        logger.error(f"[{req.job_id}] Queue full ({job_queue.maxsize})")
        raise HTTPException(status_code=503, detail="Worker queue full")
    
    if req.model not in MODEL_REGISTRY:
        logger.error(f"[{req.job_id}] Unknown model: {req.model}")
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")
    
    try:
        await job_queue.put(req.dict())
        asyncio.create_task(_record_job_status_later(req.job_id, req.model))
        
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
    """Accept multiple jobs in one request. Used for bulk edit and multi-variation."""
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
                    logger.warning(f"[BULK] Job {job.job_id} already queued/running, skipping")
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


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint - ALWAYS responds quickly.
    
    Returns worker status including:
    - Model loading state
    - Queue status
    - GPU info
    - Processing status
    """
    try:
        try:
            await asyncio.wait_for(job_status_lock.acquire(), timeout=1.0)
            try:
                active_jobs = sum(1 for j in job_status.values() if j.get("status") == "running")
                queued_jobs = sum(1 for j in job_status.values() if j.get("status") == "queued")
            finally:
                job_status_lock.release()
        except asyncio.TimeoutError:
            logger.warning("Health check: job_status_lock timeout")
            active_jobs = MAX_CONCURRENT_JOBS
            queued_jobs = job_queue.qsize()
        
        # Determine status (Qwen-style: always ready)
        if job_queue.full():
            overall_status = "overloaded"
        elif active_jobs >= MAX_CONCURRENT_JOBS:
            overall_status = "busy"
        else:
            overall_status = "ok"
        
        is_processing = active_jobs > 0
        
        # Get GPU info
        gpu_info = None
        try:
            gpu_info = get_gpu_info()
        except Exception:
            pass
        
        return HealthResponse(
            status=overall_status,
            worker_id=WORKER_ID or "unknown",
            queue_size=job_queue.qsize(),
            queue_capacity=job_queue.maxsize,
            active_jobs=active_jobs,
            queued_jobs=queued_jobs,
            max_concurrent=MAX_CONCURRENT_JOBS,
            model_loaded=True,
            model_error=None,
            startup_completed=True,
            ready=not job_queue.full(),
            processing=is_processing,
            healthy=True,
            gpu_info=gpu_info,
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="busy",
            worker_id=WORKER_ID or "unknown",
            queue_size=0,
            queue_capacity=job_queue.maxsize,
            active_jobs=1,
            queued_jobs=0,
            max_concurrent=MAX_CONCURRENT_JOBS,
            model_loaded=True,
            model_error=str(e),
            startup_completed=True,
            ready=False,
            processing=True,
            healthy=True,
            gpu_info=None,
        )


@app.get("/status")
async def status():
    """Detailed status including all jobs."""
    async with job_status_lock:
        jobs_copy = job_status.copy()
    
    return {
        "queue_size": job_queue.qsize(),
        "queue_capacity": job_queue.maxsize,
        "jobs": jobs_copy,
        "worker_id": WORKER_ID,
        "startup_completed": True,
        "model_loaded": True,
        "model_error": None,
        "gpu_info": get_gpu_info() if torch.cuda.is_available() else None,
    }


@app.get("/models")
async def list_models():
    """List available models and their capabilities."""
    return {
        "models": list(MODEL_REGISTRY.keys()),
        "modes": [
            "text-to-video",
            "image-to-video",
            "video-to-video",
            "keyframe-interpolation",
        ],
        "default_pipeline": DEFAULT_PIPELINE,
        "supported_pipelines": ["two_stage", "distilled", "ic_lora", "keyframe"],
    }


@app.post("/unload")
async def unload_models():
    """Unload all models and free GPU memory."""
    try:
        unload_pipeline()
        cleanup_gpu(verbose=True)
        return {"status": "success", "message": "Models unloaded, GPU memory freed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
