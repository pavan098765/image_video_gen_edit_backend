# server_controller/main.py

import os
import re
import uuid
import asyncio
import logging
import json
import aiohttp
from fastapi.responses import StreamingResponse
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import time

from fastapi import FastAPI, BackgroundTasks, Query, Request, HTTPException, Depends, Header, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.orm import Session

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

email_quickpod = os.getenv("QUICKPOD_EMAIL", "None")
password_quickpod = os.getenv("QUICKPOD_PASSWORD", "None")
warm_pod_uuid = os.getenv("QUICKPOD_POD_UUID")
IMAGE_PREFIX = os.getenv("IMAGE_PREFIX", "quickpod/")
CONTROLLER_PUBLIC_URL = os.getenv("CONTROLLER_PUBLIC_URL", "")
CONTROLLER_SHARED_SECRET = os.getenv("CONTROLLER_SHARED_SECRET", "")
HEARTBEAT_INTERVAL = 10           # how often the worker actually pings
HEARTBEAT_GRACE_MULTIPLIER = 2    # allow 2 missed pings before reaping
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
    pod_strategy_env = "machine_based"  # use our machine-based logic for individual-pod only flow
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
logger.info(f"Pod strategy: {pod_strategy_env}, AUTOTEST mode: {'ON' if ENABLE_AUTOTEST else 'OFF'}")
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
BAN_DURATION = 300           # seconds (temporary ban default)
RATE_LIMIT = 20              # requests per minute per IP
MAX_BODY_SIZE = 10_000_000   # 10 MB
SUSPICIOUS_SCORE_THRESHOLD = 5  # heuristic ban trigger
PERMANENT_BAN_ON_THRESHOLD = True  # if suspicion score passes threshold, ban permanently

# Trust / whitelist
TRUSTED_IPS = set()  # add any static IPs here if you want them always safe
TRUSTED_WORKER_IP_TTL_SECONDS = 60 * 60 * 24  # 24 hours trust for newly-registered worker IPs

# Known malicious user-agents and scanners (keep lowercase)
BLOCKED_USER_AGENTS = [
    "curl", "wget", "python-requests", "sqlmap", "nmap", "nikto",
    "masscan", "scrapy", "libwww-perl", "zgrab", "httpclient", "scanbot",
    "nessus", "dirbuster", "acunetix", "wpscan", "whatweb", "httprobe",
    "bot", "go-http-client"
]

# Known exploit or reconnaissance path patterns
SUSPICIOUS_PATHS = [
    r"\.\./", r"etc/passwd", r"wp-admin", r"phpmyadmin", r"\.env", r"\.git",
    r"/owa/", r"/solr/", r"/admin", r"/console", r"/status\.php", r"/login",
    r"/jasperserver", r"/owncloud", r"/zabbix", r"/WebInterface", r"/sitecore",
    r"/cgi-bin", r"/index\.jsp", r"/OA_HTML", r"/Telerik", r"cf_scripts"
]

# Suspicious header keys and payload keywords
SUSPICIOUS_HEADERS = ["X-Original-URL", "X-Rewrite-URL", "Forwarded"]
INJECTION_KEYWORDS = [b"<script", b"<?php", b"union select", b"drop table", b"--", b"alert("]

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

async def ban_ip(ip: str, reason: str, duration: Optional[int] = None, permanent: bool = False):
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
    security_logger.info(f"Suspicion {score}/{SUSPICIOUS_SCORE_THRESHOLD} for {ip} — {reason}")
    if score >= SUSPICIOUS_SCORE_THRESHOLD:
        await ban_ip(ip, f"suspicious behavior (score={score})", permanent=PERMANENT_BAN_ON_THRESHOLD)

# def increase_suspicion(ip: str, amount: int, reason: str):
#     score = ip_suspicion_score.get(ip, 0) + amount
#     ip_suspicion_score[ip] = score
#     security_logger.info(f"Suspicion {score}/{SUSPICIOUS_SCORE_THRESHOLD} for {ip} — {reason}")
#     if score >= SUSPICIOUS_SCORE_THRESHOLD:
#         # default: make permanent ban if configured
#         asyncio.create_task(ban_ip(ip, f"suspicious behavior (score={score})", permanent=PERMANENT_BAN_ON_THRESHOLD))

def add_trusted_worker_ip(ip: str):
    """Mark an IP trusted for some TTL (used when a worker registers)."""
    trusted_worker_ips[ip] = _now_ts() + TRUSTED_WORKER_IP_TTL_SECONDS
    security_logger.info(f"Added trusted worker IP {ip} for {TRUSTED_WORKER_IP_TTL_SECONDS}s")

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
PUBLIC_PATHS = ["/download"] 

# ---------- Middleware ----------
@app.middleware("http")
async def aggressive_security_middleware(request: Request, call_next):
    # Determine client IP, prefer X-Forwarded-For if present (Cloudflare/proxy scenarios)
    ip_header = request.headers.get("x-forwarded-for", "") or ""
    if ip_header:
        # pick left-most IP (original client)
        ip = ip_header.split(",")[0].strip()
    else:
        ip = (request.client.host or "unknown")
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
    if not check_rate_limit(ip):
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
            security_logger.exception(f"Worker path exception from {ip} path={path}: {e}")
            return Response("Internal Server Error", status_code=500)

    # Trusted API paths (like /submit, /job_complete) require Authorization: Bearer to skip deep body scan
    if any(path.startswith(p) for p in TRUSTED_API_PATHS):
        if not (auth_header.startswith("Bearer ") or request.headers.get("x-fingerprint")):
            await increase_suspicion(ip, 3, "trusted endpoint without auth")
            return Response("Forbidden", status_code=403)
        try:
            return await call_next(request)
        except Exception as e:
            security_logger.exception(f"Trusted API path exception from {ip} path={path}: {e}")
            return Response("Internal Server Error", status_code=500)

    # ✅ Public safe paths (no auth needed, like /download)
    if any(path.startswith(p) for p in PUBLIC_PATHS):
        try:
            return await call_next(request)
        except Exception as e:
            security_logger.exception(f"Public path exception from {ip} path={path}: {e}")
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
    pod_uuid: Optional[str] = None          # Preferred canonical pod identifier (QuickPod)
    public_ip: Optional[str] = None         # Optional public IP of the pod (if available)
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
    is_warm = (container_id == warm_pod_uuid) if container_id and warm_pod_uuid else False
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


def assign_job_to_worker(job_id: str, worker_id: str, presigned_url: str):
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
        if not job or not worker:
            logger.error(
                "Job or Worker not found (job=%s, worker=%s)", job_id, worker_id
            )
            return

        run_payload = {
            "job_id": job.job_id,
            "prompt": job.prompt,
            "model": job.model,
            "mode": job.params.get("mode"),
            "params": job.params,
            "upload_url": presigned_url,  # <-- worker gets this
            "result_key": job.result_r2_key,  # optional, for reference
        }
       
        url = f"{worker.endpoint.rstrip('/')}/run"
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                resp = requests.post(url, json=run_payload, timeout=10)
                resp.raise_for_status()
                break  # success
            except Exception as e:
                if attempt < max_attempts - 1:
                    wait = 3 * (attempt + 1)
                    logger.warning(
                        f"Attempt {attempt+1}/{max_attempts} failed to push job {job_id} to {url}: {e}. Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                    continue
                else:
                    logger.error(
                        f"❌ Failed to push job {job_id} to worker {worker_id} after {max_attempts} attempts: {e}"
                    )
                    job_manager.update_job_state(
                        job.job_id, JobState.FAILED, f"Dispatch error: {e}"
                    )
                    if worker:
                        worker.status = "terminated"
                        db.commit()
                    return
                
        job.worker_id = worker.worker_id
        job_manager.update_job_state(job.job_id, JobState.RUNNING)
        if worker.is_warm_pod:
            worker.status = "processing"
        else:
            worker.status = "busy"
        worker.last_active = datetime.utcnow()
        db.commit()
        logger.info(
            "Dispatched job %s to worker %s (%s)", job_id, worker_id, worker.endpoint
        )
    except Exception:
        db.rollback()
        logger.exception("Error assigning job %s to worker %s", job_id, worker_id)
    finally:
        db.close()


# ------------------------
# Background Tasks
# ------------------------
# async def idle_worker_reaper_task():
#     logger.info("Starting idle worker reaper")
#     while True:
#         db = SessionLocal()
#         try:
#             threshold = datetime.utcnow() - timedelta(seconds=WORKER_IDLE_TIMEOUT)
#             idle_workers = (
#                 db.query(Worker)
#                 .filter(
#                     Worker.last_active < threshold,
#                     Worker.status == "ready"   # only reap ready, not dispatching
#                 )
#                 .all()
#             )
#             print("IDLE WORKERS", idle_workers)
#             for w in idle_workers:
#                 try:
#                     if w.is_warm_pod:
                    
#                         # check if any jobs tied to this warm pod are still dispatching
#                         active_jobs = (
#                             db.query(Job)
#                             .filter(
#                                 Job.worker_id == w.worker_id,
#                                 Job.status.in_([JobState.QUEUED.value, JobState.RUNNING.value, JobState.DISPATCHING.value]),
#                             )
#                             .count()
#                         )
#                         print("active_jobs ", active_jobs)

#                         if active_jobs == 0:
#                             await pod_manager.stop_warm_pod(w.container_id)
#                             w.status = "offline"
#                     else:
#                         if w.container_id:
#                             await asyncio.sleep(300) 
#                             logger.info(f"Reaping idle individual pod {w.container_id}")
#                             await pod_manager.destroy_individual_pod(w.container_id)
#                         w.status = "offline"
#                     db.commit()
#                 except Exception as e:
#                     logger.error(f"Failed to stop worker {w.worker_id}: {e}")

#                 # fail any jobs tied to this worker
#                 running_jobs = (
#                     db.query(Job)
#                     .filter(
#                         Job.worker_id == w.worker_id,
#                         Job.status == JobState.RUNNING.value,
#                     )
#                     .all()
#                 )
#                 for job in running_jobs:
#                     job_manager.update_job_state(
#                         job.job_id, JobState.FAILED, "Worker went offline"
#                     )
#         finally:
#             db.close()
#         await asyncio.sleep(max(1, WORKER_IDLE_TIMEOUT // 2))

# async def idle_worker_reaper_task():
#     logger.info("Starting idle worker reaper")
#     while True:
#         db = SessionLocal()
#         try:
#             threshold = datetime.utcnow() - timedelta(seconds=WORKER_IDLE_TIMEOUT)
#             # Include ready and busy pods — they can both be reaped if idle *and* job-free
#             idle_workers = (
#                 db.query(Worker)
#                 .filter(
#                     Worker.last_active < threshold,
#                     Worker.status.in_(["ready", "busy"]),
#                 )
#                 .all()
#             )
#             logger.info("IDLE WORKERS %s", [w.worker_id for w in idle_workers])

#             for w in idle_workers:
#                 try:
#                     # 1️⃣ Check if this worker still has active jobs
#                     active_jobs = (
#                         db.query(Job)
#                         .filter(
#                             Job.worker_id == w.worker_id,
#                             Job.status.in_([
#                                 JobState.QUEUED.value,
#                                 JobState.RUNNING.value,
#                                 JobState.DISPATCHING.value,
#                             ]),
#                         )
#                         .count()
#                     )

#                     if active_jobs > 0:
#                         logger.info(
#                             f"Skipping reaper for worker {w.worker_id} "
#                             f"(has {active_jobs} active jobs)"
#                         )
#                         continue  # skip reaping

#                     # 2️⃣ No active jobs — safe to reap
#                     if w.is_warm_pod:
#                         logger.info(
#                             f"Reaping idle warm pod {w.container_id} (no active jobs)"
#                         )
#                         await pod_manager.stop_warm_pod(w.container_id)
#                         w.status = "offline"
#                     else:
#                         if w.container_id:
#                             logger.info(
#                                 f"Reaping idle individual pod {w.container_id} (no active jobs)"
#                             )
#                             # optional grace period to allow for late registration/polling
#                             await asyncio.sleep(5)
#                             await pod_manager.destroy_individual_pod(w.container_id)
#                         w.status = "offline"

#                     db.commit()

#                 except Exception as e:
#                     logger.error(f"Failed to stop worker {w.worker_id}: {e}")
#                     db.rollback()

#                 # 3️⃣ Fail any jobs that were left in RUNNING state (just in case)
#                 running_jobs = (
#                     db.query(Job)
#                     .filter(
#                         Job.worker_id == w.worker_id,
#                         Job.status == JobState.RUNNING.value,
#                     )
#                     .all()
#                 )
#                 for job in running_jobs:
#                     job_manager.update_job_state(
#                         job.job_id, JobState.FAILED, "Worker went offline"
#                     )

#         finally:
#             db.close()

#         # Check more frequently for responsiveness
#         await asyncio.sleep(max(5, WORKER_IDLE_TIMEOUT // 2))

async def idle_worker_reaper_task():
    logger.info("Starting idle worker reaper (enhanced)")
    while True:
        db = SessionLocal()
        try:
            now = datetime.utcnow()
            heartbeat_threshold = now - timedelta(seconds=WORKER_IDLE_TIMEOUT)

            # 🔹 Find workers that haven't heartbeated recently (stale)
            stale_workers = (
                db.query(Worker)
                .filter(Worker.last_heartbeat < heartbeat_threshold)
                .all()
            )

            # 🔹 Find workers idle and not busy
            idle_workers = (
                db.query(Worker)
                .filter(
                    Worker.last_active < heartbeat_threshold,
                    Worker.status.in_(["ready", "busy"]),
                )
                .all()
            )

            logger.info(
                "Reaper sweep: %d stale, %d idle",
                len(stale_workers),
                len(idle_workers),
            )

            # Merge both sets (avoid duplicates)
            all_targets = {w.worker_id: w for w in stale_workers + idle_workers}.values()

            for w in all_targets:
                try:
                    # 1️⃣ Check if this worker still has active jobs
                    active_jobs = (
                        db.query(Job)
                        .filter(
                            Job.worker_id == w.worker_id,
                            Job.status.in_([
                                JobState.QUEUED.value,
                                JobState.RUNNING.value,
                                JobState.DISPATCHING.value,
                            ]),
                        )
                        .all()
                    )

                    if active_jobs:
                        # Check if they are stale jobs (no update for a long time)
                        stale_jobs = [
                            j for j in active_jobs
                            if (now - j.updated_at) > timedelta(seconds=WORKER_IDLE_TIMEOUT)
                        ]
                        if stale_jobs:
                            logger.warning(
                                f"Worker {w.worker_id} has stale jobs ({len(stale_jobs)}). "
                                f"Marking offline and failing them."
                            )
                            for job in stale_jobs:
                                job_manager.update_job_state(
                                    job.job_id,
                                    JobState.FAILED,
                                    "Worker heartbeat lost; job failed",
                                )
                            if w.container_id:
                                await pod_manager.destroy_individual_pod(w.container_id)
                            w.status = "offline"
                            db.commit()
                        else:
                            logger.info(
                                f"Skipping worker {w.worker_id}: still active jobs updating."
                            )
                            continue
                    else:
                        # No active jobs, safe to reap pod
                        if w.is_warm_pod:
                            logger.info(
                                f"Reaping idle warm pod {w.container_id} (no active jobs)"
                            )
                            await pod_manager.stop_warm_pod(w.container_id)
                        else:
                            if w.container_id:
                                logger.info(
                                    f"Reaping idle individual pod {w.container_id} (no active jobs)"
                                )
                                await asyncio.sleep(2)
                                await pod_manager.destroy_individual_pod(w.container_id)
                        w.status = "offline"
                        db.commit()

                except Exception as e:
                    logger.error(f"Failed to process worker {w.worker_id}: {e}")
                    db.rollback()

        finally:
            db.close()

        # 🕒 Run every half of idle timeout (e.g. every 15s if timeout=30s)
        await asyncio.sleep(max(5, WORKER_IDLE_TIMEOUT // 2))


async def pod_cleanup_task():
    logger.info("Starting pod cleanup task (enhanced)")
    while True:
        db = SessionLocal()
        try:
            pods_to_cleanup = (
                db.query(Worker)
                .filter(Worker.status.in_(["terminated", "offline", "error"]), Worker.is_warm_pod == False)
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
            raise HTTPException(status_code=404, detail="Job not found or output missing")

        key = job.output_key

        # 2️⃣ Retrieve metadata (content length, type)
        try:
            head = s3_client.head_object(Bucket=R2_BUCKET, Key=key)
            total_size = head["ContentLength"]
            content_type = head.get("ContentType") or mimetypes.guess_type(key)[0] or "application/octet-stream"
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
                raise HTTPException(status_code=416, detail="Requested Range Not Satisfiable")
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
        headers.update({
            "Content-Type": content_type,
            "Cache-Control": "public, max-age=86400, immutable",
            "Content-Disposition": f'inline; filename="{os.path.basename(key)}"',
        })

        # 7️⃣ Return StreamingResponse with 200 or 206
        return StreamingResponse(file_iterator(), status_code=status_code, headers=headers)

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

        if payload.failed:
            job_manager.update_job_state(job.job_id, JobState.FAILED, payload.error or "Worker reported failure")
        else:
            job_manager.update_job_state(job.job_id, JobState.COMPLETED)
            job.result_r2_key = payload.result_r2_key
            job.output_key = payload.result_r2_key
            job.output_url = f"{CONTROLLER_PUBLIC_URL}/download/{job.job_id}"

        worker = db.query(Worker).filter(Worker.worker_id == job.worker_id).first()

        if worker:
            await pod_manager.remove_pending_job(job.job_id, job.model)

            # 🔍 Check if there are still active jobs tied to this worker
            active_jobs = (
                db.query(Job)
                .filter(
                    Job.worker_id == worker.worker_id,
                    Job.status.in_([
                        JobState.RUNNING.value,
                        JobState.DISPATCHING.value,
                        JobState.QUEUED.value
                    ])
                )
                .count()
            )

            if active_jobs > 0:
                logger.info(
                    f"Worker {worker.worker_id} still has {active_jobs} active jobs — keeping pod alive."
                )
                worker.status = "ready" if worker.is_warm_pod else "busy"
            else:
                worker.status = "ready" if worker.is_warm_pod else "terminated"
                logger.info(f"Worker {worker.worker_id} has no active jobs — marking {'ready' if worker.is_warm_pod else 'terminated'}.")

            worker.last_active = datetime.utcnow()
            db.commit()

        # ✅ Notify user via FCM if token exists
        try:
            fcm_token = job.params.get("fcm_token")
            if fcm_token:
                asyncio.create_task(send_fcm_notification(
                    token=fcm_token,
                    title=f"Processing Complete.",
                    body=f"Tap to view your result.",
                    data={"job_id": job.job_id}
                ))
        except Exception as e:
            logger.error(f"FCM push failed for job {job.job_id}: {e}")

        return {
            "status": "ok",
            "job_id": job.job_id,
            "result_key": job.result_r2_key,
            "download_url": job.output_url,  # ✅ always return controller redirect URL
        }

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

    with SessionLocal() as db:
        w = db.query(Worker).filter(Worker.worker_id == worker_id).first()
        if not w:
            raise HTTPException(status_code=404, detail="Worker not found")
        w.last_heartbeat = datetime.utcnow()
        w.last_active = datetime.utcnow()
        db.commit()

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
        raise HTTPException(status_code=400, detail="Worker must declare at least one model")

    container_id_received = payload.pod_uuid or payload.container_id
    endpoint = payload.endpoint or (f"http://{payload.public_ip}" if payload.public_ip else "")
    pod_uuid = container_id_received or "unknown"

    # ---- VALIDATION LOGIC ----
    expected_pods = set(pod_manager.active_pods.keys())
    recently_destroyed = getattr(pod_manager, "recently_destroyed", {})
    expiry_window = 5  # seconds

    # 1️⃣ Reject pods destroyed recently (always enforced)
    if pod_uuid in recently_destroyed and time.time() - recently_destroyed[pod_uuid] < expiry_window:
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
                await pod_manager.mark_image_cached_on_machine(pod_info.machine_id, payload.models[0])
    except Exception as e:
        logger.warning(f"Cache sync error for {container_id_received}: {e}")

    return {"status": "ok"}


@app.get("/job_status")
async def job_status(job_id: str = Query(..., description="Job ID to check status"),
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

async def send_fcm_notification(token: str, title: str, body: str, data: dict):
    """Send FCM push notification to a specific device token."""
    fcm_key = os.getenv("FCM_SERVER_KEY")
    if not fcm_key:
        logger.warning("⚠️ FCM_SERVER_KEY missing, skipping push notification")
        return

    headers = {
        "Authorization": f"key={fcm_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "to": token,
        "notification": {
            "title": title,
            "body": body,
        },
        "data": data,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://fcm.googleapis.com/fcm/send",
                headers=headers,
                json=payload,
            ) as resp:
                text = await resp.text()
                if resp.status == 200:
                    logger.info(f"✅ Sent FCM notification: {text}")
                else:
                    logger.error(f"❌ FCM failed {resp.status}: {text}")
    except Exception as e:
        logger.error(f"Failed to send FCM notification: {e}")
        

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
        json.dump(req.params, f, indent=4)

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
            timeout_at=datetime.utcnow() + timedelta(seconds=req.params.get("timeout", 3600)),
            result_r2_key=result_key,
            output_key=result_key,
            output_url=f"{CONTROLLER_PUBLIC_URL}/download/{job_id}",  # ✅ controller-managed redirect URL
        )
        db.add(job)
        db.commit()

        # ✅ Store FCM token temporarily in params for notification
        if req.fcm_token:
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
        worker_id = await pod_manager.dispatch_job(job_id, req.model, req.params.get("mode"))
        background_tasks.add_task(assign_job_to_worker, job_id, worker_id, presigned_url)

        # ✅ Simplified: always return controller /download endpoint (no presigned GET)
        download_url = f"{CONTROLLER_PUBLIC_URL}/download/{job_id}"

        return {
            "job_id": job_id,
            "status": "dispatched",
            "routed_to_worker": worker_id,
            "upload_url": presigned_url,   # ✅ worker uploads here
            "result_key": result_key,
            "download_url": download_url,  # ✅ app fetches from here (controller redirect)
        }

    except Exception as e:
        logger.error("Failed to dispatch job %s: %s", job_id, e)
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
