import os
import re
import traceback
import uuid
import asyncio
import logging
import json
import anyio
from fastapi.responses import StreamingResponse
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import time

from fastapi import (
    FastAPI,
    BackgroundTasks,
    Query,
    Request,
    HTTPException,
    Depends,
    Header,
    Response,
)
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.orm import Session
import firebase_admin
from firebase_admin import credentials, messaging


from db import init_db, SessionLocal, Worker, Job
from pod_manager import PodManager
from job_manager import JobManager, JobState
from security import (
    verify_signature,
    verify_api_key,
    hash_fingerprint,
    register_or_update_device,
    validate_device_signature,
    time_datetime_now,
)

# Optional R2 client
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from pathlib import Path
import mimetypes

# ------------------------
# ENV & INIT
# ------------------------
load_dotenv(Path(__file__).parent / "config.env")
init_db()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("controller")


# --- Firebase Admin explicit init with projectId + good diagnostics ---
def _init_firebase_admin():
    try:
        sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        logger.info("Initializing Firebase Admin SDK...")
        logger.info(f"sa_path : {sa_path}")
        if not sa_path:
            raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS not set")

        exists = os.path.exists(sa_path)
        logger.info("Firebase SA path: %s (exists=%s)", sa_path, exists)
        if not exists:
            raise FileNotFoundError(f"Service account file not found at {sa_path}")

        with open(sa_path, "r", encoding="utf-8") as f:
            sa_json = json.load(f)

        project_id = (
            sa_json.get("project_id")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCLOUD_PROJECT")
        )
        if not project_id:
            # let user override via env if their SA lacks project_id
            raise RuntimeError(
                "No project_id found in service account JSON and GOOGLE_CLOUD_PROJECT not set"
            )

        cred = credentials.Certificate(sa_json)
        firebase_admin.initialize_app(cred, {"projectId": project_id})
        logger.info("✅ Firebase Admin initialized with projectId=%s", project_id)
    except Exception as e:
        logger.exception("🔥 Firebase Admin init failed: %s", e)
        raise


# call once at module load (where your current init is)
_init_firebase_admin()

email_quickpod = os.getenv("QUICKPOD_EMAIL", "None")
password_quickpod = os.getenv("QUICKPOD_PASSWORD", "None")
warm_pod_uuid = os.getenv("QUICKPOD_POD_UUID")
IMAGE_PREFIX = os.getenv("IMAGE_PREFIX", "quickpod/")
CONTROLLER_PUBLIC_URL = os.getenv("CONTROLLER_PUBLIC_URL", "")
CONTROLLER_SHARED_SECRET = os.getenv("CONTROLLER_SHARED_SECRET", "")
HEARTBEAT_INTERVAL = 20  # how often the worker actually pings
HEARTBEAT_GRACE_MULTIPLIER = 2  # allow 2 missed pings before reaping
WORKER_IDLE_TIMEOUT = HEARTBEAT_INTERVAL * HEARTBEAT_GRACE_MULTIPLIER
WORKER_REGISTRATION_TIMEOUT = int(os.getenv("WORKER_REGISTRATION_TIMEOUT", "60"))

# R2
R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_BUCKET = os.getenv("R2_BUCKET")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
s3_client = None
if R2_ENDPOINT and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY:
    s3_client = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )

# Managers
# pod_manager = PodManager(email_quickpod, password_quickpod, warm_pod_uuid)
# Managers
POD_STRATEGY = os.getenv("POD_STRATEGY", "machine_based").lower()
# interpret a few aliases
if POD_STRATEGY in ("individual", "machine", "machine_based"):
    pod_strategy_env = (
        "machine_based"  # use our machine-based logic for individual-pod only flow
    )
elif POD_STRATEGY in ("warm", "warm_and_individual", "default"):
    pod_strategy_env = "warm_and_individual"
else:
    pod_strategy_env = POD_STRATEGY  # allow custom values if already exact

# only pass warm_pod_uuid if warm strategy selected
if pod_strategy_env == "warm_and_individual":
    pod_manager = PodManager(email_quickpod, password_quickpod, warm_pod_uuid)
else:
    # disable warm pod usage
    pod_manager = PodManager(email_quickpod, password_quickpod, None)

# also tell pod_manager about selected strategy via env (it already reads POD_STRATEGY itself)
os.environ["POD_STRATEGY"] = pod_strategy_env
ENABLE_AUTOTEST = os.getenv("ENABLE_AUTOTEST", "true").lower() == "true"
logger.info(
    f"Pod strategy: {pod_strategy_env}, AUTOTEST mode: {'ON' if ENABLE_AUTOTEST else 'OFF'}"
)
job_manager: Optional[JobManager] = None

app = FastAPI(title="Server Controller")

# === Enhanced Security Configuration ===
# --- Logging for Security Events ---
security_logger = logging.getLogger("security")
security_handler = logging.FileHandler("security.log")
security_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
security_handler.setFormatter(security_formatter)
security_logger.addHandler(security_handler)
security_logger.setLevel(logging.INFO)

# --- Persistence file for bans
BANS_FILE = Path(Path(__file__).parent, "banned_ips.json")

# --- Core Protection Configuration (tuneable) ---
BLOCKED_METHODS = {"PROPFIND", "CONNECT", "TRACE", "TRACK", "PRI"}
BLOCK_EMPTY_UA = True
BAN_DURATION = 300  # seconds (temporary ban default)
RATE_LIMIT = 40  # requests per minute per IP
MAX_BODY_SIZE = 10_000_000  # 10 MB
SUSPICIOUS_SCORE_THRESHOLD = 5  # heuristic ban trigger
PERMANENT_BAN_ON_THRESHOLD = (
    True  # if suspicion score passes threshold, ban permanently
)

# Trust / whitelist
TRUSTED_IPS = set()  # add any static IPs here if you want them always safe
TRUSTED_WORKER_IP_TTL_SECONDS = (
    60 * 60 * 24
)  # 24 hours trust for newly-registered worker IPs

# Known malicious user-agents and scanners (keep lowercase)
BLOCKED_USER_AGENTS = [
    "curl",
    "wget",
    "python-requests",
    "sqlmap",
    "nmap",
    "nikto",
    "masscan",
    "scrapy",
    "libwww-perl",
    "zgrab",
    "httpclient",
    "scanbot",
    "nessus",
    "dirbuster",
    "acunetix",
    "wpscan",
    "whatweb",
    "httprobe",
    "bot",
    "go-http-client",
]

# Known exploit or reconnaissance path patterns
SUSPICIOUS_PATHS = [
    r"\.\./",
    r"etc/passwd",
    r"wp-admin",
    r"phpmyadmin",
    r"\.env",
    r"\.git",
    r"/owa/",
    r"/solr/",
    r"/admin",
    r"/console",
    r"/status\.php",
    r"/login",
    r"/jasperserver",
    r"/owncloud",
    r"/zabbix",
    r"/WebInterface",
    r"/sitecore",
    r"/cgi-bin",
    r"/index\.jsp",
    r"/OA_HTML",
    r"/Telerik",
    r"cf_scripts",
]

# Suspicious header keys and payload keywords
SUSPICIOUS_HEADERS = ["X-Original-URL", "X-Rewrite-URL", "Forwarded"]
INJECTION_KEYWORDS = [
    b"<script",
    b"<?php",
    b"union select",
    b"drop table",
    b"--",
    b"alert(",
]

# In-memory IP tracking
# For persistent bans use structure: banned_ips[ip] = {"expiry": <float|None>, "permanent": bool, "reason": str, "created_at": float}
banned_ips: Dict[str, Dict[str, Any]] = {}
rate_limiter: Dict[str, list] = {}
ip_suspicion_score: Dict[str, int] = {}

# Trusted worker IPs: trusted_worker_ips[ip] = expiry_ts
trusted_worker_ips: Dict[str, float] = {}

# async lock for persistence-safe operations
ban_lock = asyncio.Lock()


# ---------- Persistence helpers ----------
async def persist_bans():
    """Write banned_ips to disk (atomic rename)."""
    async with ban_lock:
        try:
            data = banned_ips.copy()
            tmp = BANS_FILE.with_suffix(".tmp")

            def _write():
                print(f"🟢 Writing bans file to {tmp}")
                with open(tmp, "w") as f:
                    json.dump(data, f)
                tmp.replace(BANS_FILE)

            await asyncio.to_thread(_write)
        except Exception:
            security_logger.exception("Failed to persist banned_ips to disk")


async def load_bans():
    """Load banned_ips from disk at startup."""
    async with ban_lock:
        if not BANS_FILE.exists():
            return
        try:

            def _read():
                with open(BANS_FILE, "r") as f:
                    return json.load(f)

            loaded = await asyncio.to_thread(_read)
            # loaded is ip -> dict
            for ip, info in loaded.items():
                banned_ips[ip] = {
                    "expiry": info.get("expiry"),
                    "permanent": bool(info.get("permanent", False)),
                    "reason": info.get("reason", ""),
                    "created_at": info.get("created_at", time.time()),
                }
            security_logger.info(f"Loaded {len(banned_ips)} persisted banned IPs")
        except Exception:
            security_logger.exception("Failed to load banned_ips from disk")


# ---------- Helpers ----------
def _now_ts() -> float:
    return time.time()


def is_banned(ip: str) -> bool:
    # skip trusted ips
    if ip in TRUSTED_IPS:
        return False
    info = banned_ips.get(ip)
    if not info:
        return False
    if info.get("permanent"):
        return True
    expiry = info.get("expiry")
    if expiry and expiry > _now_ts():
        return True
    # expired
    try:
        del banned_ips[ip]
    except KeyError:
        pass
    # persist asynchronously
    try:
        asyncio.create_task(persist_bans())
    except Exception:
        pass
    return False


async def ban_ip(
    ip: str, reason: str, duration: Optional[int] = None, permanent: bool = False
):
    """Ban IP. If permanent=True, set permanent, otherwise set expiry."""
    async with ban_lock:
        if permanent:
            banned_ips[ip] = {
                "expiry": None,
                "permanent": True,
                "reason": reason,
                "created_at": _now_ts(),
            }
            security_logger.warning(f"IP banned (permanent) ({ip}) — reason: {reason}")
        else:
            exp = _now_ts() + (duration or BAN_DURATION)
            banned_ips[ip] = {
                "expiry": exp,
                "permanent": False,
                "reason": reason,
                "created_at": _now_ts(),
            }
            security_logger.warning(f"IP banned ({ip}) until {exp} — reason: {reason}")
        # persist immediately
        try:
            await persist_bans()
        except Exception:
            security_logger.exception("Failed to persist bans after ban_ip")


async def unban_ip(ip: str):
    async with ban_lock:
        if ip in banned_ips:
            banned_ips.pop(ip, None)
            await persist_bans()
            security_logger.info(f"IP unbanned: {ip}")
            return True
        return False


def check_rate_limit(ip: str) -> bool:
    now = _now_ts()
    timestamps = rate_limiter.get(ip, [])
    timestamps = [t for t in timestamps if now - t < 60]  # last minute
    if len(timestamps) >= RATE_LIMIT:
        rate_limiter[ip] = timestamps
        return False
    timestamps.append(now)
    rate_limiter[ip] = timestamps
    return True


async def increase_suspicion(ip: str, amount: int, reason: str):
    score = ip_suspicion_score.get(ip, 0) + amount
    ip_suspicion_score[ip] = score
    security_logger.info(
        f"Suspicion {score}/{SUSPICIOUS_SCORE_THRESHOLD} for {ip} — {reason}"
    )
    if score >= SUSPICIOUS_SCORE_THRESHOLD:
        await ban_ip(
            ip,
            f"suspicious behavior (score={score})",
            permanent=PERMANENT_BAN_ON_THRESHOLD,
        )


def add_trusted_worker_ip(ip: str):
    """Mark an IP trusted for some TTL (used when a worker registers)."""
    trusted_worker_ips[ip] = _now_ts() + TRUSTED_WORKER_IP_TTL_SECONDS
    security_logger.info(
        f"Added trusted worker IP {ip} for {TRUSTED_WORKER_IP_TTL_SECONDS}s"
    )


def is_trusted_worker_ip(ip: str) -> bool:
    exp = trusted_worker_ips.get(ip)
    if not exp:
        return False
    if exp > _now_ts():
        return True
    # expired
    try:
        del trusted_worker_ips[ip]
    except KeyError:
        pass
    return False


# Paths behavior:
# - WORKER_PATHS: endpoints used by workers (register_worker, heartbeat) -> exempt from UA/permanent banning
# - TRUSTED_API_PATHS: endpoints that may require Authorization Bearer to skip deep body inspection (/submit, /job_complete)
# - PUBLIC_PATHS: endpoints open to public (like /download) -> no checks apply
WORKER_PATHS = ["/register_worker", "/heartbeat"]
TRUSTED_API_PATHS = ["/submit", "/job_complete", "/job_status"]
PUBLIC_PATHS = ["/download", "/job_status"]


# ---------- Middleware ----------
@app.middleware("http")
async def aggressive_security_middleware(request: Request, call_next):
    # Determine client IP, prefer X-Forwarded-For if present (Cloudflare/proxy scenarios)
    ip_header = request.headers.get("x-forwarded-for", "") or ""
    if ip_header:
        # pick left-most IP (original client)
        ip = ip_header.split(",")[0].strip()
    else:
        ip = request.client.host or "unknown"
    path = request.url.path.lower()
    ua = request.headers.get("user-agent", "").lower()
    auth_header = request.headers.get("authorization", "")

    # Allow trusted static IPs immediately
    if ip in TRUSTED_IPS:
        return await call_next(request)

    # Banned check
    if is_banned(ip):
        security_logger.warning(f"Blocked request from banned IP: {ip} path={path}")
        return Response("Access Denied", status_code=403)

    # Rate limiting applies to everyone (including workers)
    if not path.startswith("/job_status") and not check_rate_limit(ip):
        # temporary ban for rate limit abuse
        await ban_ip(ip, "rate limit exceeded", duration=BAN_DURATION, permanent=False)
        return Response("Too Many Requests", status_code=429)

    # Block obviously dangerous methods -> permanent ban
    if request.method.upper() in BLOCKED_METHODS:
        await ban_ip(ip, f"blocked method {request.method}", permanent=True)
        return Response("Method Not Allowed", status_code=405)

    # If this is a worker endpoint, skip UA/permanent checks — let handlers verify signatures
    if any(path.startswith(p) for p in WORKER_PATHS):
        # Allow worker endpoints to process (they will verify signatures inside handlers)
        try:
            return await call_next(request)
        except Exception as e:
            security_logger.exception(
                f"Worker path exception from {ip} path={path}: {e}"
            )
            return Response("Internal Server Error", status_code=500)

    # Trusted API paths (like /submit, /job_complete) require Authorization: Bearer to skip deep body scan
    if any(path.startswith(p) for p in TRUSTED_API_PATHS):
        if not (
            auth_header.startswith("Bearer ") or request.headers.get("x-fingerprint")
        ):
            await increase_suspicion(ip, 3, "trusted endpoint without auth")
            return Response("Forbidden", status_code=403)
        try:
            return await call_next(request)
        except Exception as e:
            security_logger.exception(
                f"Trusted API path exception from {ip} path={path}: {e}"
            )
            return Response("Internal Server Error", status_code=500)

    # ✅ Public safe paths (no auth needed, like /download)
    if any(path.startswith(p) for p in PUBLIC_PATHS):
        try:
            return await call_next(request)
        except Exception as e:
            security_logger.exception(
                f"Public path exception from {ip} path={path}: {e}"
            )
            return Response("Internal Server Error", status_code=500)

    # From here on, we apply aggressive checks to public/untrusted endpoints
    # Suspicious path detection -> permanent ban
    for pattern in SUSPICIOUS_PATHS:
        if re.search(pattern, path, re.IGNORECASE):
            await increase_suspicion(ip, 2, f"scanner path: {path}")
            await ban_ip(ip, f"scanner path matched: {pattern}", permanent=True)
            return Response("Forbidden", status_code=403)

    # User-Agent validation -> permanent ban (aggressive)
    if BLOCK_EMPTY_UA and not ua.strip():
        await ban_ip(ip, "empty user-agent", permanent=True)
        return Response("Forbidden", status_code=403)

    if any(bad in ua for bad in BLOCKED_USER_AGENTS):
        await ban_ip(ip, f"bad user-agent: {ua}", permanent=True)
        return Response("Forbidden", status_code=403)

    # Suspicious headers -> permanent ban
    for header in SUSPICIOUS_HEADERS:
        if header in request.headers:
            await ban_ip(ip, f"suspicious header {header}", permanent=True)
            return Response("Forbidden", status_code=403)

    # Payload inspection for public endpoints (not trusted, not workers)
    if request.method in {"POST", "PUT", "PATCH"}:
        try:
            body = await request.body()
            if len(body) > MAX_BODY_SIZE:
                await ban_ip(ip, "oversized body", permanent=False)
                return Response("Payload Too Large", status_code=413)
            for keyword in INJECTION_KEYWORDS:
                if keyword in body.lower():
                    await ban_ip(ip, f"injection pattern {keyword}", permanent=True)
                    return Response("Forbidden", status_code=403)
        except Exception:
            pass

    # heuristic anomalies -> increase suspicion (may lead to ban)
    if len(path) > 200 or path.count("/") > 20:
        await increase_suspicion(ip, 1, "path anomaly")
    if re.search(r"[<>{}\"';]", path):
        await increase_suspicion(ip, 1, "path character anomaly")

    # Normal request flow
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        security_logger.exception(f"Unhandled exception from {ip} path={path}: {e}")
        return Response("Internal Server Error", status_code=500)


@app.middleware("http")
async def reject_unknown_paths(request: Request, call_next):
    """
    Rejects any HTTP request whose path is not part of the allowed API routes.
    This prevents scanners or bots from probing unlisted paths like /robots.txt or /.
    """
    path = request.url.path
    # Define which paths are actually allowed for public/worker/API use
    allowed_prefixes = [
        "/health",
        "/submit",
        "/job_complete",
        "/job_status",
        "/register_worker",
        "/heartbeat",
        "/download",
    ]

    # Allow FastAPI's built-in /docs or /openapi.json only if DEBUG is set
    if os.getenv("DEBUG", "false").lower() == "true":
        allowed_prefixes.extend(["/docs", "/openapi.json", "/redoc"])

    # If path not matching any allowed prefix → reject early
    if not any(path.startswith(p) for p in allowed_prefixes):
        return Response("Forbidden", status_code=403)

    # Normal request flow
    return await call_next(request)


# ------------------------
# Pydantic Models
# ------------------------
class SubmitRequest(BaseModel):
    model: str
    prompt: Optional[str] = ""
    params: Optional[Dict[str, Any]] = {}
    fcm_token: Optional[str] = None


class RegisterPayload(BaseModel):
    worker_id: str
    models: list
    endpoint: str = ""  # keep optional-equivalent behavior by allowing empty string
    container_id: Optional[str] = None
    pod_uuid: Optional[str] = None  # Preferred canonical pod identifier (QuickPod)
    public_ip: Optional[str] = None  # Optional public IP of the pod (if available)
    info: Optional[Dict[str, Any]] = None


class JobCompletePayload(BaseModel):
    job_id: str
    result_r2_key: Optional[str] = None
    failed: Optional[bool] = False
    error: Optional[str] = None  # also capture error message if sent


# ------------------------
# Auth Dependencies
# ------------------------
async def require_any_auth(
    authorization: Optional[str] = Header(None),
    x_device_uuid: Optional[str] = Header(None),
    x_fingerprint: Optional[str] = Header(None),
    x_timestamp: Optional[str] = Header(None),
    x_signature: Optional[str] = Header(None),
):
    if authorization and verify_api_key(authorization):
        return {"type": "api_key", "key": authorization}

    if not x_fingerprint or not x_timestamp or not x_signature:
        raise HTTPException(status_code=401, detail="Missing auth headers")

    fingerprint_hash = (
        x_fingerprint.lower()
        if len(x_fingerprint) == 64
        and all(c in "0123456789abcdef" for c in x_fingerprint.lower())
        else hash_fingerprint(x_fingerprint)
    )

    if not validate_device_signature(fingerprint_hash, x_timestamp, x_signature):
        raise HTTPException(
            status_code=403, detail="Invalid device signature or timestamp"
        )

    rec = register_or_update_device(
        fingerprint_hash,
        x_device_uuid,
        meta={"last_seen_at": time_datetime_now().isoformat()},
    )
    return {
        "type": "device",
        "fingerprint_hash": fingerprint_hash,
        "device_uuid": rec.get("device_uuid"),
    }


# ------------------------
# DB Helpers
# ------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def find_ready_worker(db: Session, model: str) -> Optional[Worker]:
    return (
        db.query(Worker)
        .filter(Worker.model == model, Worker.status == "ready")
        .order_by(Worker.last_active.desc())
        .first()
    )


def create_presigned_put_url(key: str, expires_in: int = 3600) -> Optional[str]:
    """Generate a presigned PUT URL for R2 so worker can upload results directly."""
    if not s3_client:
        return None
    try:
        return s3_client.generate_presigned_url(
            "put_object",
            Params={"Bucket": R2_BUCKET, "Key": key},
            ExpiresIn=expires_in,
        )
    except Exception as e:
        logger.error(f"Failed to generate presigned URL for key {key}: {e}")
        return None


def upsert_worker(
    db: Session,
    worker_id: str,
    model: str,
    endpoint: str,
    container_id: Optional[str],
    info: Optional[Dict[str, Any]],
):
    """
    NOTE: 'container_id' DB column stores the QuickPod pod_uuid (canonical pod id).
    We intentionally keep the DB column name as container_id for backward compatibility.
    """
    w = db.query(Worker).filter(Worker.worker_id == worker_id).first()
    now = datetime.utcnow()
    # Use equality for warm pod detection
    is_warm = (
        (container_id == warm_pod_uuid) if container_id and warm_pod_uuid else False
    )
    if not w:
        w = Worker(
            worker_id=worker_id,
            model=model,
            endpoint=endpoint,
            status="ready",
            container_id=container_id,
            last_heartbeat=now,
            last_active=now,
            info=info,
            is_warm_pod=is_warm,
        )
        db.add(w)
    else:
        w.endpoint = endpoint
        w.model = model
        w.container_id = container_id or w.container_id
        w.status = "ready"
        w.last_heartbeat = now
        w.last_active = now
        w.info = info
        if container_id:
            w.is_warm_pod = is_warm
    db.commit()
    return w

def assign_job_to_worker(job_id: str, worker_id: str, presigned_url: str) -> bool:
    """
    Atomically reserve job for worker (DISPATCHING) using SELECT FOR UPDATE.
    Returns True on successful dispatch, False otherwise.
    """
    db = SessionLocal()
    try:
        # Lock row
        job = (
            db.query(Job)
            .filter(Job.job_id == job_id)
            .with_for_update(nowait=True)
            .first()
        )
        if not job:
            logger.error("assign_job_to_worker: job not found %s", job_id)
            return False

        # Only allow dispatch from QUEUED
        if job.status != JobState.QUEUED.value:
            logger.info(
                "assign_job_to_worker: job %s not QUEUED (state=%s) -> skip",
                job_id, job.status,
            )
            return False

        # reserve
        job.worker_id = worker_id
        job.status = JobState.DISPATCHING.value
        job.updated_at = datetime.utcnow()
        db.commit()

        worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
        if not worker:
            job.status = JobState.QUEUED.value
            job.worker_id = None
            db.commit()
            return False

        run_payload = {
            "job_id": job.job_id,
            "prompt": job.prompt,
            "model": job.model,
            "mode": job.params.get("mode"),
            "params": job.params,
            "upload_url": presigned_url,
            "result_key": job.result_r2_key,
        }

        url = f"{worker.endpoint.rstrip('/')}/run"
        last_exc = None

        for attempt in range(5):
            try:
                resp = requests.post(url, json=run_payload, timeout=10)
                resp.raise_for_status()
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                if attempt < 4:
                    time.sleep(3 * (attempt + 1))
                continue

        if last_exc:
            # revert job to QUEUED
            j = (
                db.query(Job)
                .filter(Job.job_id == job_id)
                .with_for_update()
                .first()
            )
            if j:
                j.status = JobState.QUEUED.value
                j.worker_id = None
                j.updated_at = datetime.utcnow()
                db.commit()

            # mark machine bad
            machine_id = getattr(worker, "machine_id", None)
            if machine_id:
                pod_manager.mark_machine_bad(str(machine_id))

            # destroy pod
            container_id = getattr(worker, "container_id", None)
            if container_id:
                anyio.from_thread.run(pod_manager.destroy_individual_pod, container_id)
                asyncio.create_task(pod_manager.release_slot_on_pod(container_id, job_id))

            return False

        # Success -> mark RUNNING
        job = db.query(Job).filter(Job.job_id == job_id).first()
        job.status = JobState.RUNNING.value
        job.updated_at = datetime.utcnow()

        job_worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
        if job_worker:
            job_worker.status = "processing" if job_worker.is_warm_pod else "busy"
            job_worker.last_active = datetime.utcnow()

        db.commit()
        logger.info("Dispatched job %s to worker %s", job_id, worker_id)
        return True

    except Exception:
        logger.exception("assign_job_to_worker unexpected error")
        return False
    finally:
        db.close()

# ------------------------
# Background Tasks
# ------------------------


async def idle_worker_reaper_task():
    """
    Robust reaper:
    - Destroys stale pods (no heartbeat for HEARTBEAT_STALE_SECS)
    - Destroys pods with no active jobs for IDLE_TIMEOUT_SECS
    - Ignores busy/processing status if jobs are actually gone or stale
    """
    # Prefer env-driven values or keep these defaults
    HEARTBEAT_STALE_SECS = int(os.getenv("HEARTBEAT_STALE_SECS", "90"))
    IDLE_TIMEOUT_SECS = int(os.getenv("IDLE_TIMEOUT_SECS", "10"))
    SWEEP_INTERVAL_SECS = int(os.getenv("REAPER_SWEEP_INTERVAL_SECS", "30"))
    logger.info(f"IDLE_TIMEOUT_SECS {IDLE_TIMEOUT_SECS}")
    # Include any worker statuses your system uses that should be considered by the reaper
    STATUSES_TO_CONSIDER = ["ready", "busy", "processing"]

    logger.info(
        f"Starting idle worker reaper (stale>{HEARTBEAT_STALE_SECS}s, idle>{IDLE_TIMEOUT_SECS}s, sweep={SWEEP_INTERVAL_SECS}s)"
    )

    while True:
        await asyncio.sleep(SWEEP_INTERVAL_SECS)

        db = SessionLocal()
        now = datetime.utcnow()
        stale_cutoff = now - timedelta(seconds=HEARTBEAT_STALE_SECS)
        stale_count = 0
        idle_count = 0

        try:
            workers = (
                db.query(Worker).filter(Worker.status.in_(STATUSES_TO_CONSIDER)).all()
            )

            for w in workers:
                try:
                    # STALE: no heartbeat recently => destroy immediately
                    if not w.last_heartbeat or w.last_heartbeat < stale_cutoff:
                        logger.warning(
                            "⚠️ Worker %s stale (last_heartbeat=%s) → destroying pod %s",
                            w.worker_id,
                            w.last_heartbeat,
                            getattr(w, "container_id", None),
                        )
                        if getattr(w, "container_id", None):
                            try:
                                await pod_manager.destroy_individual_pod(w.container_id)
                                stale_count += 1
                            except Exception as e:
                                logger.error(
                                    "Destroy failed for stale pod %s: %s",
                                    w.container_id,
                                    e,
                                )
                        # mark offline and commit safely
                        w.status = "offline"
                        try:
                            db.commit()
                        except Exception:
                            db.rollback()
                        continue

                    # ---------------------------------------------
                    # READY → IDLE / DESTROY LOGIC (CORRECTED)
                    # ---------------------------------------------
                    if w.status == "ready":

                        # 1. If any pending jobs exist for this model → keep alive
                        pending_jobs = (
                            db.query(Job)
                            .filter(
                                Job.model == w.model,
                                Job.status.in_([JobState.QUEUED.value, JobState.DISPATCHING.value])
                            )
                            .count()
                        )

                        if pending_jobs > 0:
                            logger.debug(
                                "READY worker %s kept alive: pending jobs exist for model=%s",
                                w.worker_id, w.model
                            )
                            continue

                        # 2. Compute real idle time based on last completed job
                        last_done = (
                            w.last_job_finished_at or
                            w.created_at or
                            w.last_active
                        )
                        idle_for = (now - last_done).total_seconds()

                        # 3. If idle_for exceeds timeout → destroy pod
                        if idle_for > IDLE_TIMEOUT_SECS:
                            logger.info(
                                "💤 READY worker %s idle for %.1f seconds (limit=%s) → destroying pod %s",
                                w.worker_id, idle_for, IDLE_TIMEOUT_SECS, w.container_id
                            )
                            try:
                                if getattr(w, "container_id", None):
                                    await pod_manager.destroy_individual_pod(w.container_id)
                                    w.status = "offline"
                                    db.commit()
                                    idle_count += 1
                            except Exception as e:
                                logger.error(
                                    "Destroy failed for READY-idle pod %s: %s",
                                    w.container_id, e
                                )
                                db.rollback()
                            continue

                        # 4. Otherwise, still within idle window → keep alive
                        logger.debug(
                            "READY worker %s idle_for=%.1f < timeout=%s → keeping",
                            w.worker_id, idle_for, IDLE_TIMEOUT_SECS
                        )
                        continue

                    # Worker must explicitly be idle to be reaped
                    if w.status != "idle":
                        logger.debug(
                            "Worker %s not idle (status=%s) → skip idle check",
                            w.worker_id,
                            w.status,
                        )
                        continue

                    # Grace window before destroying
                    if (now - w.last_active).total_seconds() < 5:
                        logger.debug(
                            "Worker %s recently active → skip destroy", w.worker_id
                        )
                        continue

                    # IDLE: find any active jobs attached to this worker
                    active_jobs = (
                        db.query(Job)
                        .filter(
                            Job.worker_id == w.worker_id,
                            Job.status.in_(
                                [
                                    JobState.RUNNING.value,
                                    JobState.QUEUED.value,
                                    JobState.DISPATCHING.value,
                                ]
                            ),
                        )
                        .all()
                    )

                    # Pending ANY jobs for this model (unassigned jobs)
                    pending_jobs = (
                        db.query(Job)
                        .filter(
                            Job.model == w.model,
                            Job.status.in_(
                                [JobState.QUEUED.value, JobState.DISPATCHING.value]
                            ),
                        )
                        .count()
                    )

                    # If pending jobs exist → keep pod alive
                    if pending_jobs > 0:
                        logger.debug(
                            "Worker %s skipped — pending jobs exist for model=%s",
                            w.worker_id,
                            w.model,
                        )
                        continue

                    # If there are active jobs, treat them as "recent" only if updated_at is recent.
                    recent_jobs = []
                    for j in active_jobs:
                        j_time = j.updated_at or j.created_at or datetime(1970, 1, 1)
                        if (now - j_time) < timedelta(seconds=IDLE_TIMEOUT_SECS):
                            recent_jobs.append(j)

                    # If there are zero active jobs OR all active jobs are stale -> reap/stop
                    if (not active_jobs) or (active_jobs and not recent_jobs):
                        # safe to reap/stop this worker/pod
                        if getattr(w, "is_warm_pod", False):
                            logger.info(
                                "💤 Idle warm pod %s (worker=%s) → stopping",
                                w.container_id,
                                w.worker_id,
                            )
                            try:
                                await pod_manager.stop_warm_pod(w.container_id)
                            except Exception as e:
                                logger.warning(
                                    "Failed to stop warm pod %s: %s", w.container_id, e
                                )
                        else:
                            # machine-based or individual pod: destroy
                            if getattr(w, "container_id", None):
                                logger.info(
                                    "💤 Idle pod %s (worker=%s) → destroying",
                                    w.container_id,
                                    w.worker_id,
                                )
                                try:
                                    await pod_manager.destroy_individual_pod(
                                        w.container_id
                                    )
                                    idle_count += 1
                                except Exception as e:
                                    logger.warning(
                                        "Destroy failed for idle pod %s: %s",
                                        w.container_id,
                                        e,
                                    )

                        # update state in DB
                        w.status = "offline"
                        try:
                            db.commit()
                        except Exception:
                            db.rollback()
                    else:
                        logger.debug(
                            "Skipping worker %s: %d active jobs (%d recent)",
                            w.worker_id,
                            len(active_jobs),
                            len(recent_jobs),
                        )

                except Exception as e:
                    logger.error(
                        "Reaper error on worker %s: %s",
                        getattr(w, "worker_id", None),
                        e,
                    )
                    db.rollback()

            logger.info(
                "Reaper sweep complete → %d stale, %d idle reaped",
                stale_count,
                idle_count,
            )

        except Exception as e:
            logger.error("Reaper loop error: %s", e)
        finally:
            db.close()


async def pod_cleanup_task():
    logger.info("Starting pod cleanup task (enhanced)")
    while True:
        db = SessionLocal()
        try:
            pods_to_cleanup = (
                db.query(Worker)
                .filter(
                    Worker.status.in_(["terminated", "offline", "error"]),
                    Worker.is_warm_pod == False,
                )
                .all()
            )
            if pods_to_cleanup:
                logger.info("Cleaning up %d pods", len(pods_to_cleanup))
            for pod in pods_to_cleanup:
                try:
                    if pod.container_id:
                        await pod_manager.stop_individual_pod(pod.container_id)
                    db.delete(pod)
                    db.commit()
                except Exception as e:
                    logger.error(f"Failed pod cleanup {pod.worker_id}: {e}")
                    pod.status = "error"
                    pod.info = {
                        "error": str(e),
                        "cleanup_attempted_at": datetime.utcnow().isoformat(),
                    }
                    db.commit()
        finally:
            db.close()
        await asyncio.sleep(15)


async def job_recovery_task():
    while True:
        try:
            await job_manager.check_for_stuck_jobs()
        except Exception:
            logger.exception("Error in job recovery task")
        await asyncio.sleep(30)


# ------------------------
# Startup
# ------------------------
@app.on_event("startup")
async def startup_event():
    global job_manager
    try:
        job_manager = JobManager(SessionLocal, pod_manager)
        JOB_AUTO_RECOVER = os.getenv("JOB_AUTO_RECOVER", "false").lower() == "true"
        ENABLE_AUTOSCALER = os.getenv("ENABLE_AUTOSCALER", "true").lower() == "true"

        await pod_manager.initialize()
        asyncio.create_task(idle_worker_reaper_task())
        asyncio.create_task(pod_cleanup_task())

        if JOB_AUTO_RECOVER:
            logger.info("Job Recovery mode ON")
            asyncio.create_task(job_recovery_task())
        else:
            logger.info("Job Recovery mode OFF")

        # Start autoscaler if enabled
        if ENABLE_AUTOSCALER:
            logger.info("Autoscaler enabled — starting background loop")
            asyncio.create_task(pod_manager.autoscaler_loop())
        else:
            logger.info("Autoscaler disabled")

        if not BANS_FILE.exists():

            async def create_empty_ban_file():
                async with ban_lock:
                    with open(BANS_FILE, "w") as f:
                        json.dump({}, f)

            asyncio.create_task(create_empty_ban_file())
            logger.info("Created IP bans file")

        await load_bans()
        logger.info(f"Loaded {len(banned_ips)} IP bans on startup")

        logger.info("Controller startup complete")
    except Exception as e:
        logger.error(f"Error during startup: {e}")


# ------------------------
# API Endpoints
# ------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/download/{job_id}")
async def download(job_id: str, request: Request):
    """
    Streams image/video from Cloudflare R2 with full Range (HTTP 206) support.
    Memory-safe and optimized for concurrent users & large files.
    """
    db = SessionLocal()
    try:
        # 1️⃣ Find job entry
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job or not job.output_key:
            raise HTTPException(
                status_code=404, detail="Job not found or output missing"
            )

        key = job.output_key

        # 2️⃣ Retrieve metadata (content length, type)
        try:
            head = s3_client.head_object(Bucket=R2_BUCKET, Key=key)
            total_size = head["ContentLength"]
            content_type = (
                head.get("ContentType")
                or mimetypes.guess_type(key)[0]
                or "application/octet-stream"
            )
        except ClientError as e:
            logger.error(f"R2 head_object failed for {key}: {e}")
            raise HTTPException(status_code=404, detail="File not found in R2")

        # 3️⃣ Check if client sent a Range header (e.g., for video seeking)
        range_header = request.headers.get("range")
        start = 0
        end = total_size - 1
        status_code = 200
        headers = {}

        if range_header:
            # Parse header like: bytes=1000-2000
            try:
                match = range_header.strip().lower().replace("bytes=", "").split("-")
                if match[0]:
                    start = int(match[0])
                if len(match) > 1 and match[1]:
                    end = int(match[1])
                if end >= total_size:
                    end = total_size - 1
                if start > end:
                    raise ValueError("Invalid range")
                status_code = 206  # Partial content
                headers["Content-Range"] = f"bytes {start}-{end}/{total_size}"
                headers["Accept-Ranges"] = "bytes"
                headers["Content-Length"] = str(end - start + 1)
            except Exception as e:
                logger.warning(f"Invalid Range header {range_header}: {e}")
                start, end = 0, total_size - 1

        else:
            headers["Content-Length"] = str(total_size)
            headers["Accept-Ranges"] = "bytes"

        # 4️⃣ Fetch from R2 with Range (for partial downloads)
        range_param = f"bytes={start}-{end}"
        try:
            s3_obj = s3_client.get_object(Bucket=R2_BUCKET, Key=key, Range=range_param)
            stream = s3_obj["Body"]
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "InvalidRange":
                raise HTTPException(
                    status_code=416, detail="Requested Range Not Satisfiable"
                )
            logger.error(f"Error fetching range {range_param} for {key}: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch file from R2")

        # 5️⃣ Async generator to yield chunks
        async def file_iterator():
            try:
                while True:
                    chunk = stream.read(1024 * 1024)  # 1MB
                    if not chunk:
                        break
                    yield chunk
            finally:
                stream.close()

        # 6️⃣ Set final headers
        headers.update(
            {
                "Content-Type": content_type,
                "Cache-Control": "public, max-age=86400, immutable",
                "Content-Disposition": f'inline; filename="{os.path.basename(key)}"',
            }
        )

        # 7️⃣ Return StreamingResponse with 200 or 206
        return StreamingResponse(
            file_iterator(), status_code=status_code, headers=headers
        )

    finally:
        db.close()


@app.post("/job_complete")
async def job_complete(payload: JobCompletePayload):
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == payload.job_id).first()
        if not job:
            if payload.job_id.startswith("autotest-"):
                logger.info(f"Auto-test job {payload.job_id} completed.")
                return {"status": "ok"}
            raise HTTPException(status_code=404, detail="Job not found")

        # --- IDEMPOTENCY GUARD ---
        if job.status == JobState.COMPLETED.value:
            logger.info("job_complete called but job %s already COMPLETED — ignoring duplicate", job.job_id)
            return {"status": "ok", "job_id": job.job_id}

        worker = db.query(Worker).filter(Worker.worker_id == job.worker_id).first()

        # ------------------ HANDLE EXPLICIT FAILURE ------------------
        if payload.failed:
            reason = payload.error or "Worker reported failure"
            logger.warning(f"❌ Worker failed job {job.job_id}: {reason}")

            job_manager.update_job_state(
                job.job_id, JobState.FAILED, reason, force=True
            )
            job.error = reason
            job.output_url = None
            db.commit()

            # Terminate pod + mark machine bad
            if worker:
                try:
                    await pod_manager.remove_pending_job(job.job_id, job.model)
                except Exception:
                    pass

                if getattr(worker, "machine_id", None):
                    pod_manager.mark_machine_bad(worker.machine_id)

                if getattr(worker, "container_id", None):
                    try:
                        await pod_manager.destroy_individual_pod(worker.container_id)
                        # release pod queue slot as result arrived
                        asyncio.create_task(pod_manager.release_slot_on_pod(container_id, job.job_id))
                        logger.error(f"🔥 Destroyed failed pod {worker.container_id}")
                    except Exception as e:
                        logger.error(f"⚠️ Failed destroy {worker.container_id}: {e}")

            # Retry job on new machine
            try:
                logger.info(f"♻️ Requeueing job {job.job_id} on NEW machine...")
                await pod_manager.add_pending_job(job.job_id, job.model)
                new_worker = await pod_manager.dispatch_job(
                    job.job_id, job.model, job.params.get("mode")
                )
                new_presigned = create_presigned_put_url(job.result_r2_key)
                asyncio.create_task(
                    asyncio.to_thread(
                        assign_job_to_worker, job.job_id, new_worker, new_presigned
                    )
                )
                return {
                    "status": "requeued",
                    "job_id": job.job_id,
                    "worker": new_worker,
                }
            except Exception as e:
                logger.error(f"❌ Failed requeue {job.job_id}: {e}")
                return {"status": "failed_final", "job_id": job.job_id}

        # ------------------ CHECK OOM IN ERROR LOG ------------------
        oom_detected = False
        try:
            if job.error_log:
                for entry in job.error_log:
                    err = (entry.get("error") or "").lower()
                    if any(
                        tok in err
                        for tok in ["cuda out of memory", "oom", "cublas", "no memory"]
                    ):
                        oom_detected = True
                        logger.warning(f"⚠️ OOM detected in {job.job_id}")
                        break
        except Exception as e:
            logger.warning(f"OOM scan error for job {job.job_id}: {e}")

        # ------------------ VERIFY R2 UPLOAD BEFORE MARKING COMPLETED ------------------
        key = payload.result_r2_key
        file_found = False
        if key:
            for attempt in range(3):
                try:
                    s3_client.head_object(Bucket=R2_BUCKET, Key=key)
                    file_found = True
                    logger.info(f"✅ R2 verified for {key}")
                    break
                except Exception:
                    await asyncio.sleep(5)

        failed_validation = oom_detected or (not file_found)

        if failed_validation:
            # ------------------ FAILED VALIDATION PATH ------------------
            reason = "OOM" if oom_detected else "R2 missing"
            logger.error(f"🚨 Post-run validation failed {job.job_id}: {reason}")

            job_manager.update_job_state(
                job.job_id, JobState.FAILED, reason, force=True
            )
            job.error = reason
            job.output_url = None
            db.commit()

            if worker:
                try:
                    await pod_manager.remove_pending_job(job.job_id, job.model)
                except Exception:
                    pass

                if getattr(worker, "machine_id", None):
                    pod_manager.mark_machine_bad(worker.machine_id)

                if getattr(worker, "container_id", None):
                    try:
                        await pod_manager.destroy_individual_pod(worker.container_id)
                        # release pod queue slot as result arrived
                        asyncio.create_task(pod_manager.release_slot_on_pod(container_id, job.job_id))
                        logger.error(
                            f"🔥 Destroyed pod {worker.container_id} after invalid output"
                        )
                    except Exception as e:
                        logger.error(f"⚠️ Pod destroy failed: {e}")

            # requeue
            try:
                await pod_manager.add_pending_job(job.job_id, job.model)
                new_worker = await pod_manager.dispatch_job(
                    job.job_id, job.model, job.params.get("mode")
                )
                new_presigned = create_presigned_put_url(key)
                asyncio.create_task(
                    asyncio.to_thread(
                        assign_job_to_worker, job.job_id, new_worker, new_presigned
                    )
                )
                return {
                    "status": "requeued",
                    "job_id": job.job_id,
                    "worker": new_worker,
                }
            except Exception as e:
                logger.error(f"❌ requeue failed {job.job_id}: {e}")
                return {"status": "failed_no_output", "job_id": job.job_id}

        # ------------------ SUCCESS PATH ------------------
        # ✅ Validation passed — mark COMPLETED
        job_manager.update_job_state(job.job_id, JobState.COMPLETED, force=True)
        job.result_r2_key = payload.result_r2_key
        job.output_key = payload.result_r2_key
        job.output_url = f"{CONTROLLER_PUBLIC_URL}/download/{job.job_id}"
        db.commit()

        if worker:
            try:
                await pod_manager.remove_pending_job(job.job_id, job.model)
            except Exception as e:
                logger.error(f"❌ Failed to remove pending job {job.job_id}: {e}")

            # --- Count active jobs still running on the same worker ---
            try:
                active_jobs = (
                    db.query(Job)
                    .filter(
                        Job.worker_id == worker.worker_id,
                        Job.status.in_(
                            [
                                JobState.RUNNING.value,
                                JobState.DISPATCHING.value,
                                JobState.QUEUED.value,
                            ]
                        ),
                    )
                    .count()
                )
            except Exception as e:
                logger.error(
                    f"❌ Failed to count active jobs for worker {worker.worker_id}: {e}"
                )
                active_jobs = 0

            container_id = getattr(worker, "container_id", None)
            try:
                if container_id:
                    # ✅ Only destroy pod if truly no other active jobs exist
                    if active_jobs > 0:
                        logger.info(
                            "Keeping pod %s alive for worker %s — %d other active jobs remain",
                            container_id,
                            worker.worker_id,
                            active_jobs,
                        )
                        worker.status = "busy"
                    else:
                        # No active jobs — decide if we destroy or keep alive
                        if (
                            not worker.is_warm_pod
                            and pod_manager.pod_strategy != "machine_based"
                        ):
                            logger.info(
                                "Destroying ephemeral pod %s for worker %s after successful completion of job %s (no active jobs left)",
                                container_id,
                                worker.worker_id,
                                job.job_id,
                            )
                            try:
                                await pod_manager.destroy_individual_pod(container_id)
                                # release pod queue slot as result arrived
                                asyncio.create_task(pod_manager.release_slot_on_pod(container_id, job.job_id))
                            except Exception as e:
                                logger.exception(
                                    "Failed to destroy individual pod %s: %s",
                                    container_id,
                                    e,
                                )
                            # Update worker state
                            try:
                                db.refresh(worker)
                            except Exception:
                                pass
                            if getattr(worker, "container_id", None):
                                worker.container_id = None
                            worker.status = "terminated"
                        else:
                            # Machine-based or warm pod → keep alive
                            worker.status = "ready"
                            worker.last_job_finished_at = datetime.utcnow()
                            logger.info(
                                "Keeping machine-based/warm pod %s (worker=%s) alive for reuse",
                                container_id,
                                worker.worker_id,
                            )
                else:
                    # No container id present; mark readiness based on activity
                    worker.status = "ready" if active_jobs == 0 else "busy"
            except Exception as e:
                logger.warning(
                    "Error handling pod lifecycle for worker %s: %s",
                    getattr(worker, "worker_id", None),
                    e,
                )

            # Update last_active and commit DB state
            try:
                worker.last_active = datetime.utcnow()
                db.commit()
                logger.info(
                    "Worker %s updated in DB (status=%s, container_id=%s)",
                    worker.worker_id,
                    worker.status,
                    worker.container_id,
                )
            except Exception as e:
                db.rollback()
                logger.error(
                    "Failed to commit worker update for %s: %s",
                    getattr(worker, "worker_id", None),
                    e,
                )

        # ------------------ SEND FCM NOTIFICATION ------------------
        if job.params.get("fcm_token"):
            data = {
                "job_id": job.job_id,
                "status": "completed",
                "download_url": job.output_url,
            }
            asyncio.create_task(
                asyncio.to_thread(
                    send_fcm_notification,
                    job.params["fcm_token"],
                    "Processing Complete.",
                    "Tap to view your result.",
                    data,
                )
            )

        return {"status": "ok", "job_id": job.job_id, "download_url": job.output_url}

    finally:
        db.close()


@app.post("/heartbeat")
async def heartbeat(request: Request):
    body = await request.body()
    sig = request.headers.get("X-Signature", "")
    if not verify_signature(sig, body):
        raise HTTPException(status_code=401, detail="Invalid signature")

    data = json.loads(body)
    worker_id = data.get("worker_id")
    if not worker_id:
        raise HTTPException(status_code=400, detail="Missing worker_id")

    # Only update last_heartbeat here. Do NOT touch last_active.
    # last_active should be updated on real job-related events (dispatch/job_complete).
    with SessionLocal() as db:
        w = db.query(Worker).filter(Worker.worker_id == worker_id).first()
        if not w:
            raise HTTPException(status_code=404, detail="Worker not found")
        w.last_heartbeat = datetime.utcnow()
        # DO NOT update last_active here
        try:
            db.commit()
        except Exception:
            db.rollback()
            logger.exception("Failed committing heartbeat update for %s", worker_id)
            raise HTTPException(status_code=500, detail="DB commit failed")

    return {"status": "ok"}


@app.post("/register_worker")
async def register_worker(request: Request):
    """
    Secure worker registration handler:
    - Verifies signature
    - Validates pod_uuid belongs to orchestrator's active pods
    - Destroys any unexpected pods automatically to prevent cost
    - Tracks recently destroyed pods to ignore re-registrations
    """

    body = await request.body()
    sig = request.headers.get("X-Signature", "")
    if not verify_signature(sig, body):
        raise HTTPException(status_code=401, detail="Invalid signature")

    data = json.loads(body)
    payload = RegisterPayload(**data)

    if not payload.models:
        raise HTTPException(
            status_code=400, detail="Worker must declare at least one model"
        )

    container_id_received = payload.pod_uuid or payload.container_id
    endpoint = payload.endpoint or (
        f"http://{payload.public_ip}" if payload.public_ip else ""
    )
    pod_uuid = container_id_received or "unknown"

    # ---- VALIDATION LOGIC ----
    expected_pods = set(pod_manager.active_pods.keys())
    recently_destroyed = getattr(pod_manager, "recently_destroyed", {})
    expiry_window = 5  # seconds

    # 1️⃣ Reject pods destroyed recently (always enforced)
    if (
        pod_uuid in recently_destroyed
        and time.time() - recently_destroyed[pod_uuid] < expiry_window
    ):
        logger.warning(
            f"⚠️ Ignoring re-registration from recently destroyed pod {pod_uuid} "
            f"(IP={endpoint}) — within expiry window."
        )
        return Response(status_code=403, content="Pod was recently destroyed")

    # 2️⃣ Handle unknown pods based on ENABLE_AUTOTEST
    if pod_uuid not in expected_pods:
        if ENABLE_AUTOTEST:
            logger.warning(
                f"⚠️ [AUTOTEST MODE] Allowing unknown pod {pod_uuid} to register normally "
                f"(IP={endpoint})."
            )
        else:
            logger.warning(
                f"🚫 Unexpected worker registration attempt from pod {pod_uuid} "
                f"(IP={endpoint}). Initiating forced destroy to prevent cost."
            )
            # fire-and-forget async destroy
            asyncio.create_task(pod_manager.destroy_unexpected_pod(pod_uuid))
            return Response(status_code=403, content="Unrequested pod destroyed")

    # ---- NORMAL REGISTRATION FLOW ----
    with SessionLocal() as db:
        for model in payload.models:
            upsert_worker(
                db,
                payload.worker_id,
                model,
                endpoint,
                container_id_received,
                payload.info,
            )

    logger.info(
        "✅ Registered worker %s for models %s (pod_uuid=%s, endpoint=%s)",
        payload.worker_id,
        payload.models,
        pod_uuid,
        endpoint,
    )

    # Update cache
    try:
        if container_id_received:
            pod_info = pod_manager.individual_pods.get(container_id_received)
            if pod_info and pod_info.machine_id:
                await pod_manager.mark_image_cached_on_machine(
                    pod_info.machine_id, payload.models[0]
                )
    except Exception as e:
        logger.warning(f"Cache sync error for {container_id_received}: {e}")

    return {"status": "ok"}


@app.get("/job_status")
async def job_status(
    job_id: str = Query(..., description="Job ID to check status"),
    auth=Depends(require_any_auth),
):
    """
    Client polls this endpoint with job_id to get current job status.
    Returns the job's current state and metadata.
    """
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        response = {
            "job_id": job.job_id,
            "status": job.status,
            "model": job.model,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
        }

        # If failed, include error info for debugging
        if job.status == JobState.FAILED.value:
            response["error"] = job.error
            response["error_log"] = job.error_log or []

        return response

    finally:
        db.close()


def send_fcm_notification(token: str, title: str, body: str, data: dict):
    """Send a push notification to a device via FCM."""
    try:
        if "job_id" not in data:
            raise ValueError("data must include job_id")
        data.setdefault("status", "completed")
        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
            ),
            data={str(k): str(v) for k, v in data.items()},
            token=token,
        )
        response = messaging.send(message)
        logger.info(f"✅ Sent FCM notification: {response}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to send FCM notification: {e}")
        return False


@app.post("/submit")
async def submit(
    req: SubmitRequest,
    background_tasks: BackgroundTasks,
    auth=Depends(require_any_auth),
):
    job_id = str(uuid.uuid4())
    db = SessionLocal()
    result_key = f"{IMAGE_PREFIX}{job_id}.png"
    with open("req_param.json", "w", encoding="utf-8") as f:
        json.dump(req.dict(), f, indent=4)

    try:
        # Create new job record
        job = Job(
            job_id=job_id,
            model=req.model,
            prompt=req.prompt,
            params=req.params or {},
            status=JobState.QUEUED.value,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            timeout_at=datetime.utcnow()
            + timedelta(seconds=req.params.get("timeout", 3600)),
            result_r2_key=result_key,
            output_key=result_key,
            output_url=f"{CONTROLLER_PUBLIC_URL}/download/{job_id}",  # ✅ controller-managed redirect URL
        )
        db.add(job)
        db.commit()

        # ✅ Store FCM token temporarily in params for notification
        if req.fcm_token:
            logger.info(f"Storing FCM token {req.fcm_token} for job {job_id}")
            job.params["fcm_token"] = req.fcm_token
            db.commit()
    finally:
        db.close()

    # ✅ Generate presigned PUT URL for worker upload (unchanged)
    presigned_url = create_presigned_put_url(result_key)
    if not presigned_url:
        raise HTTPException(status_code=500, detail="Failed to generate upload URL")

    try:
        # Track job as pending and dispatch to a worker
        await pod_manager.add_pending_job(job_id, req.model)
        worker_id = await pod_manager.dispatch_job(
            job_id, req.model, req.params.get("mode")
        )
        background_tasks.add_task(
            assign_job_to_worker, job_id, worker_id, presigned_url
        )

        # ✅ Simplified: always return controller /download endpoint (no presigned GET)
        download_url = f"{CONTROLLER_PUBLIC_URL}/download/{job_id}"

        return {
            "job_id": job_id,
            "status": "dispatched",
            "routed_to_worker": worker_id,
            "upload_url": presigned_url,  # ✅ worker uploads here
            "result_key": result_key,
            "download_url": download_url,  # ✅ app fetches from here (controller redirect)
        }

    except Exception as e:
        logger.error("Failed to dispatch job %s: %s", job_id, e)
        logger.error(traceback.format_exc())
        print()
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job:
                job.status = JobState.FAILED.value
                job.error = str(e)
                db.commit()
        finally:
            db.close()
        raise HTTPException(status_code=500, detail=f"Dispatch failed: {e}")
