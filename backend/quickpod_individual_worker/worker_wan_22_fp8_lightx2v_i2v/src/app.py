from io import BytesIO
import os
import re
import time
import json
import logging
import asyncio
import hmac
import hashlib
import gc
import math
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime
import uuid
import httpx
from PIL import Image
import concurrent.futures

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt
from contextlib import asynccontextmanager

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
        torch.cuda.ipc_collect()
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
# Import inference functions
# -----------------------
# Auto-merge split safetensors if merge.py exists
# This MUST run before loading any models to ensure all safetensors are properly merged
try:
    from merge import main as merge_main
    print("🔄 Running merge.py to combine split safetensors...")
    merge_main()
    print("✅ Merge completed")
except SystemExit as e:
    # merge.py calls sys.exit(1) on error - this is critical
    print(f"❌ CRITICAL: Merge failed with exit code {e.code}")
    print("   Model files may be incomplete. Check logs above for details.")
    raise  # Re-raise to prevent worker from starting with incomplete models
except Exception as e:
    print(f"⚠️  Merge script error (non-critical): {e}")
    import traceback
    traceback.print_exc()
    # Continue - might be no parts to merge

from inference import run_inference_wan_22_i2v

# -----------------------
# Model Registry
# -----------------------
MODEL_REGISTRY = {
    "wan_22_fp8_lightx2v_i2v": {
        "i2v": run_inference_wan_22_i2v,
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

MODEL_KEY = os.getenv("MODEL_KEY", "wan-22-fp8-lightx2v-i2v-worker")
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
    prompt: str
    model: str
    mode: str | None = None
    params: Dict[str, Any] = {}
    upload_url: str
    result_key: str


class BulkRunRequest(BaseModel):
    jobs: List[RunRequest]


# -----------------------
# Init image resolution (I2V only)
# -----------------------
# Server sends init_image as presigned R2 GET URL only (no base64). Download -> bytes -> PIL.
# No base64 encode/decode in the pipeline.
def _is_init_image_url(value: Any) -> bool:
    return isinstance(value, str) and (value.startswith("http://") or value.startswith("https://"))


def _resolve_init_image_to_pil(raw: Any, job_id: str = "") -> Optional[Image.Image]:
    """Resolve init_image from R2 presigned URL to PIL. URL only; no base64."""
    if not raw or not isinstance(raw, str):
        return None
    if not _is_init_image_url(raw):
        logger.error(f"[{job_id}] init_image must be a presigned R2 URL (http/https), got non-URL string")
        raise ValueError("init_image must be a presigned R2 download URL")
    try:
        resp = requests.get(raw, timeout=120)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        logger.error(f"[{job_id}] Failed to download init_image from R2: {e}")
        raise

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
    # CRITICAL FIX: Report "busy" when jobs are running or queued, not just "ready"
    if not startup_completed:
        status = "starting"
    else:
        # Check if any jobs are running or queued
        # Note: This is called from async context via run_in_executor, so we can safely
        # read job_status dict (dict reads are thread-safe for concurrent access)
        try:
            # Quick check: any running jobs?
            has_running = any(j.get("status") == "running" for j in job_status.values())
            # Check queue size (qsize() is thread-safe)
            queue_size = job_queue.qsize()
            
            if has_running or queue_size > 0:
                status = "busy"
            else:
                status = "ready"
        except Exception:
            # Fallback: if check fails, assume ready (conservative - better than blocking)
            status = "ready"
    
    payload = {"worker_id": WORKER_ID, "pod_uuid": POD_UUID, "timestamp": int(time.time()), "status": status}
    try:
        return post_signed_json(f"{CONTROLLER_URL.rstrip('/')}/heartbeat", payload, timeout=8, max_retries=3).json()
    except Exception:
        return {}

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


def _enhance_prompt(prompt: str, job_id: str, init_image_url: Optional[str] = None) -> str:
    """
    Enhance a prompt using Together AI LLM for video I2V generation.
    If init_image_url is provided (presigned R2 URL), sends image + prompt so the
    LLM can produce an image-aware enhanced prompt instead of generic text.

    Returns enhanced prompt, or original prompt on failure.
    """
    if TOGETHER_API_KEY == "no_api_key":
        logger.warning("PROMPT ENHANCE REQUESTED, SKIPPING.. NO API KEY SET IN ENV VAR")
        return prompt

    logger.info("[%s] ENHANCE: calling Together AI (%s) to enhance prompt (type=video_i2v)", job_id, TOGETHER_MODEL)
    logger.info("[%s] ENHANCE: original prompt = '%s'", job_id, prompt[:200] + ("..." if len(prompt) > 200 else ""))

    # Build user message content: text only, or text + image when init_image URL is available
    if init_image_url and _is_init_image_url(init_image_url):
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": init_image_url}},
        ]
        logger.info("[%s] ENHANCE: sending prompt + init_image URL to LLM (image-aware)", job_id)
    else:
        user_content = prompt

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
                    {"role": "user", "content": user_content},
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
# Together AI Prompt Extension (for long video multi-clip)
# -----------------------
_EXTEND_SYSTEM_PROMPT = (
    "You are an expert at creating video segment prompts for AI image-to-video generation.\n\n"
    "The user will provide a single prompt describing a video they want to generate from an image. "
    "You must output exactly N segment prompts (one per video clip) that follow the user's intent.\n\n"
    "Rules:\n"
    "1. Output exactly N prompts, one for each segment. Each segment is ~5 seconds of video.\n"
    "2. Each prompt must be SPECIFIC to the user's scene/action—NOT generic. Do NOT use generic phrases like "
    "'Continue the scene with fluid natural motion' or 'seamless continuation' unless the user's prompt is vague.\n"
    "3. First segment: Establish motion/action from the image. Describe what happens first.\n"
    "4. Middle segments: Natural continuation that follows the narrative. Each should advance the story/action.\n"
    "5. Last segment: Natural conclusion or continuation that fits the overall intent.\n"
    "6. Preserve the user's style, subject, setting, and tone across all segments.\n"
    "7. Output format: A JSON array of exactly N strings. Example: [\"prompt1\", \"prompt2\", \"prompt3\"]\n"
    "8. Do NOT include any text before or after the JSON array. Only output the JSON array.\n"
    "9. Each prompt should be 1-3 sentences, descriptive and specific to that segment."
)


def _extend_prompt(prompt: str, num_segments: int, job_id: str) -> list:
    """
    Expand one user prompt into N segment prompts using Together AI LLM.
    Used for long video (multi-clip) generation where each clip needs its own prompt.

    Returns list of N prompts, or [prompt] * num_segments on failure.
    """
    if num_segments <= 1:
        return [prompt]

    if TOGETHER_API_KEY == "no_api_key":
        logger.warning("[%s] PROMPT EXTEND REQUESTED, SKIPPING.. NO API KEY SET", job_id)
        return [prompt] * num_segments

    logger.info("[%s] EXTEND: calling Together AI to expand prompt into %d segments", job_id, num_segments)

    user_content = (
        f"The user wants to generate a {num_segments * 5}-second video from an image. "
        f"Create exactly {num_segments} segment prompts for the following:\n\n{prompt}"
    )

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
                    {"role": "system", "content": _EXTEND_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.6,
                "max_tokens": 2048,
            },
            timeout=45,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        logger.info("[%s] EXTEND: raw response (%d chars)", job_id, len(content))

        # Try JSON array first
        content_clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        # Find JSON array in response
        match = re.search(r"\[[\s\S]*?\]", content_clean)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list) and len(parsed) >= num_segments:
                prompts = [str(p).strip() for p in parsed[:num_segments] if p]
                if len(prompts) == num_segments:
                    logger.info("[%s] EXTEND: got %d segment prompts", job_id, len(prompts))
                    return prompts
            elif isinstance(parsed, list) and parsed:
                prompts = [str(p).strip() for p in parsed if p]
                while len(prompts) < num_segments:
                    prompts.append(prompts[-1])
                return prompts[:num_segments]

        # Fallback: numbered list (1. prompt, 2. prompt, ...)
        lines = [l.strip() for l in content_clean.split("\n") if l.strip()]
        prompts = []
        for line in lines:
            m = re.match(r"^\d+[\.\)]\s*(.+)", line)
            if m:
                prompts.append(m.group(1).strip())
        if len(prompts) >= num_segments:
            return prompts[:num_segments]

    except Exception as e:
        logger.warning("[%s] EXTEND: prompt extension failed: %s", job_id, e, exc_info=True)

    logger.info("[%s] EXTEND: falling back to same prompt for all %d segments", job_id, num_segments)
    return [prompt] * num_segments


# -----------------------
# Inference dispatcher
# -----------------------
def resolve_inference_callable(model: str, mode: Optional[str]):
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model}")
    func = MODEL_REGISTRY[model]
    if isinstance(func, dict):
        if not mode or mode not in func:
            raise ValueError(f"Mode required for {model}. Available: {list(func.keys())}")
        func = func[mode]
    return func

async def run_inference_and_save(job_id: str, prompt: str, model: str, mode: Optional[str], params: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
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
    if params.get("enhance_prompt", False):
        init_image_url = params.get("init_image")  # pass URL as-is for LLM; do not pop, inference needs it later
        prompt = await loop.run_in_executor(IO_EXECUTOR, _enhance_prompt, prompt, job_id, init_image_url)

    # Prompt extension for long video (multi-clip): expand into N segment prompts
    prompts: Union[str, List[str]] = prompt
    duration = params.get("duration")
    if duration is not None and duration > 5 and params.get("extend_prompt", True):
        num_clips = max(1, math.ceil(float(duration) / 5))
        prompts = await loop.run_in_executor(IO_EXECUTOR, _extend_prompt, prompt, num_clips, job_id)

    inference_start_time = time.time()
    try:
        if model == "wan_22_fp8_lightx2v_i2v" and mode == "i2v":
            # Resolve init_image (R2 URL) to PIL once; no base64 in pipeline
            raw_init = params.pop("init_image", None)
            init_image = await loop.run_in_executor(IO_EXECUTOR, _resolve_init_image_to_pil, raw_init, job_id) if raw_init else None
            out_path = await loop.run_in_executor(INFERENCE_EXECUTOR, inference_func, job_id, prompts, params, init_image)
        else:
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
    global FIRST_INFERENCE_START_TIME
    job_id, model, mode, prompt = item["job_id"], item["model"], item.get("mode"), item["prompt"]
    params, upload_url, result_key = item.get("params", {}), item["upload_url"], item["result_key"]
    # I2V: init_image (R2 URL) is resolved to PIL inside run_inference_and_save; no resolve_media_urls

    async with job_status_lock:
        job_status[job_id] = {"status": "running", "started_at": datetime.utcnow().isoformat()}

    try:
        async with semaphore:
            # Notify server that inference is ACTUALLY starting now (not just queued)
            # This resets started_at to the real processing start time
            is_autotest = job_id.startswith("autotest-")
            if CONTROLLER_URL and not is_autotest:
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        IO_EXECUTOR, post_signed_json,
                        f"{CONTROLLER_URL.rstrip('/')}/job_started",
                        {"job_id": job_id, "worker_id": WORKER_ID},
                        8, 3
                    )
                except Exception as e:
                    logger.warning(f"[{job_id}] Failed to notify job_started: {e}")

            out_local, metrics = await run_inference_and_save(job_id, prompt, model, mode, params)
            if not upload_url.startswith("http://dummy-upload"):
                if not await _upload_with_retry(out_local, upload_url, job_id):
                    raise Exception("Upload failed")

            is_autotest = job_id.startswith("autotest-")
            if CONTROLLER_URL and not is_autotest:
                try:
                    await asyncio.get_running_loop().run_in_executor(IO_EXECUTOR, post_signed_json,
                        f"{CONTROLLER_URL.rstrip('/')}/job_complete", {"job_id": job_id, "result_r2_key": result_key, "next_job_id": _peek_next_job_id(), "metrics": metrics}, 10, 5)
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
            fail_metrics = {}
            async with _first_inference_lock:
                if FIRST_INFERENCE_START_TIME is None:
                    FIRST_INFERENCE_START_TIME = time.time()
                    fail_metrics["startup_time_seconds"] = round(FIRST_INFERENCE_START_TIME - POD_START_TIME, 2)
            try:
                await asyncio.get_running_loop().run_in_executor(IO_EXECUTOR, post_signed_json,
                    f"{CONTROLLER_URL.rstrip('/')}/job_complete", {"job_id": job_id, "result_r2_key": None, "failed": True, "next_job_id": _peek_next_job_id(), "metrics": fail_metrics if fail_metrics else None}, 10, 5)
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
    logger.info("Starting WAN 2.2 I2V auto-test")

    headers = {"User-Agent": "Mozilla/5.0", "Accept": "image/*,*/*;q=0.8"}
    img_url = "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/Snapinsta.app_262322643_4937545659591345_7487236998017046274_n_1080.jpg"

    # Optional: test long video (2 clips) when AUTOTEST_DURATION=10
    duration = int(os.getenv("AUTOTEST_DURATION", "5"))
    base_params = {
        "num_frames": 81,
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "fps": 16,
        "init_image": img_url,
    }
    if duration > 5:
        base_params["duration"] = duration
        base_params["extend_prompt"] = True
        logger.info(f"I2V autotest: duration={duration}s ({max(1, (duration + 4) // 5)} clips)")

    async with httpx.AsyncClient(headers=headers, timeout=300.0) as client:
        try:
            i2v_job_id = f"autotest-i2v-{uuid.uuid4()}"
            resp = await client.post("http://127.0.0.1:8000/run", json={
                "job_id": i2v_job_id,
                "prompt": "Animate this image with cinematic motion",
                "model": "wan_22_fp8_lightx2v_i2v",
                "mode": "i2v",
                "params": base_params,
                "upload_url": "http://dummy-upload/skip",
                "result_key": f"{i2v_job_id}.mp4",
            })
            logger.info(f"I2V test: {resp.status_code}")
        except Exception as e:
            logger.error(f"I2V test failed: {e}")

# -----------------------
# Loops
# -----------------------
async def _health_monitor_loop():
    while True:
        await asyncio.sleep(60)
        try:
            if torch.cuda.is_available() and torch.cuda.memory_allocated() / 1024**3 > 22:
                logger.warning("HIGH GPU MEMORY")
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
