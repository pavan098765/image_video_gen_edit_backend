# routes.py
"""
API Routes

Provides all API endpoints:
- Public endpoints (health, download)
- Client endpoints (submit, job_status)
- Worker endpoints (register_worker, heartbeat, job_complete)
- Admin endpoints (optional)
"""

import os
import uuid
import time
import asyncio
import mimetypes
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import requests

from fastapi import APIRouter, Request, Response, HTTPException, Depends, Header, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from botocore.exceptions import ClientError

from db import (
    get_session,
    SessionLocal,
    Worker,
    Job,
    create_audit_log,
    utc_naive_now,
)
from models import (
    SubmitRequest,
    SubmitResponse,
    RegisterWorkerRequest,
    HeartbeatRequest,
    JobCompleteRequest,
    JobStatusResponse,
    HealthResponse,
    ErrorResponse,
    JobStatus,
    WorkerStatus,
)
from security import (
    verify_signature,
    verify_api_key,
    hash_fingerprint,
    validate_device_signature,
    register_or_update_device,
    ip_ban_manager,
)
from exceptions import (
    JobNotFoundError,
    InvalidModelError,
    AuthenticationError,
    InvalidSignatureError,
)
from job_manager import JobManager, JobState
from logging_config import get_audit_logger

logger = logging.getLogger("routes")
audit_logger = get_audit_logger()

# Create routers
public_router = APIRouter(tags=["Public"])
client_router = APIRouter(tags=["Client"])
worker_router = APIRouter(tags=["Worker"])
admin_router = APIRouter(prefix="/admin", tags=["Admin"])


# =============================================================================
# Dependency Injection
# =============================================================================

# These will be set during app initialization
_pod_manager = None
_job_manager = None
_s3_client = None
_r2_bucket = None
_r2_endpoint = None
_model_requirements = {}
_image_prefix = ""
_controller_url = ""


def set_dependencies(
    pod_manager,
    job_manager: JobManager,
    s3_client,
    r2_bucket: str,
    r2_endpoint: str,
    model_requirements: Dict,
    image_prefix: str,
    controller_url: str,
):
    """Set route dependencies during app initialization"""
    global _pod_manager, _job_manager, _s3_client, _r2_bucket, _r2_endpoint
    global _model_requirements, _image_prefix, _controller_url
    
    _pod_manager = pod_manager
    _job_manager = job_manager
    _s3_client = s3_client
    _r2_bucket = r2_bucket
    _r2_endpoint = r2_endpoint
    _model_requirements = model_requirements
    _image_prefix = image_prefix
    _controller_url = controller_url


def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# =============================================================================
# Authentication Dependencies
# =============================================================================

async def require_any_auth(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_device_uuid: Optional[str] = Header(None),
    x_fingerprint: Optional[str] = Header(None),
    x_timestamp: Optional[str] = Header(None),
    x_signature: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """
    Require either API key or device signature authentication.
    """
    ip = get_client_ip(request)
    
    # Try API key first
    if authorization and verify_api_key(authorization):
        audit_logger.auth_success(ip, "api_key")
        return {"type": "api_key", "key": authorization}
    
    # Try device signature
    if not x_fingerprint or not x_timestamp or not x_signature:
        ip_ban_manager.record_failed_auth(ip)
        await ip_ban_manager.check_brute_force(ip)
        raise HTTPException(status_code=401, detail="Missing auth headers")
    
    # Normalize fingerprint
    fingerprint_hash = (
        x_fingerprint.lower()
        if len(x_fingerprint) == 64 and all(c in "0123456789abcdef" for c in x_fingerprint.lower())
        else hash_fingerprint(x_fingerprint)
    )
    
    # Validate signature
    if not validate_device_signature(fingerprint_hash, x_timestamp, x_signature):
        ip_ban_manager.record_failed_auth(ip)
        await ip_ban_manager.check_brute_force(ip)
        audit_logger.auth_failure(ip, "device", "Invalid signature")
        raise HTTPException(status_code=403, detail="Invalid device signature")
    
    # Register/update device
    device_info = register_or_update_device(
        fingerprint_hash,
        x_device_uuid,
        meta={"last_auth": utc_naive_now().isoformat()}
    )
    
    audit_logger.auth_success(ip, "device", actor=fingerprint_hash[:16])
    
    return {
        "type": "device",
        "fingerprint_hash": fingerprint_hash,
        "device_uuid": device_info.get("device_uuid"),
    }


async def require_worker_signature(
    request: Request,
    x_signature: Optional[str] = Header(None),
) -> bool:
    """
    Require valid HMAC signature from worker.
    """
    if not x_signature:
        raise HTTPException(status_code=401, detail="Missing signature")
    
    body = await request.body()
    
    if not verify_signature(x_signature, body):
        ip = get_client_ip(request)
        audit_logger.auth_failure(ip, "worker", "Invalid signature")
        raise HTTPException(status_code=403, detail="Invalid signature")
    
    return True


# =============================================================================
# Helper Functions
# =============================================================================

def create_presigned_put_url(key: str, expires_in: int = 3600) -> Optional[str]:
    """Generate presigned PUT URL for R2"""
    if not _s3_client or not _r2_bucket:
        return None
    
    try:
        return _s3_client.generate_presigned_url(
            "put_object",
            Params={"Bucket": _r2_bucket, "Key": key},
            ExpiresIn=expires_in,
        )
    except Exception as e:
        logger.error(f"Failed to generate presigned URL: {e}")
        return None


def validate_model(model: str) -> str:
    """Validate model name against requirements"""
    model_lower = model.lower()
    
    if not _model_requirements:
        return model_lower
    
    if model_lower not in _model_requirements and model_lower != "default":
        available = list(_model_requirements.keys())
        raise InvalidModelError(model, available)
    
    return model_lower


async def send_fcm_notification(token: str, title: str, body: str, data: Dict = None):
    """Send FCM push notification"""
    try:
        from firebase_admin import messaging
        
        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            data=data or {},
            token=token,
        )
        
        response = messaging.send(message)
        logger.info(f"FCM notification sent: {response}")
        
    except Exception as e:
        logger.error(f"FCM notification failed: {e}")


def _calculate_timing_metrics(job: Job, completion_time: datetime) -> Dict[str, Optional[float]]:
    """Calculate all timing metrics for a completed job"""
    metrics = {
        "total_time": None,
        "processing_time": None,
        "queue_time": None,
        "dispatch_time": None,
    }
    
    # Total time: created_at to completed_at
    if job.created_at:
        total_delta = (completion_time - job.created_at).total_seconds()
        metrics["total_time"] = round(total_delta, 2)
    
    # Processing time: started_at to completed_at
    if job.started_at:
        processing_delta = (completion_time - job.started_at).total_seconds()
        metrics["processing_time"] = round(processing_delta, 2)
    
    # Queue time: queued_at to started_at (or dispatched_at if started_at not set)
    if job.queued_at:
        end_time = job.started_at or job.dispatched_at or completion_time
        if end_time:
            queue_delta = (end_time - job.queued_at).total_seconds()
            metrics["queue_time"] = round(queue_delta, 2)
    
    # Dispatch time: dispatched_at to started_at
    if job.dispatched_at and job.started_at:
        dispatch_delta = (job.started_at - job.dispatched_at).total_seconds()
        metrics["dispatch_time"] = round(dispatch_delta, 2)
    
    return metrics


def _log_timing_metrics(job_id: str, model: str, timing_metrics: Dict[str, Optional[float]]):
    """Log timing metrics to console in a formatted way"""
    logger.info("=" * 80)
    logger.info(f"⏱️  JOB TIMING REPORT: {job_id}")
    logger.info(f"   Model: {model}")
    
    if timing_metrics["total_time"] is not None:
        logger.info(f"   📊 Total Time: {timing_metrics['total_time']:.2f}s (created → completed)")
    
    if timing_metrics["processing_time"] is not None:
        logger.info(f"   ⚡ Processing Time: {timing_metrics['processing_time']:.2f}s (started → completed)")
    
    if timing_metrics["queue_time"] is not None:
        logger.info(f"   ⏳ Queue Time: {timing_metrics['queue_time']:.2f}s (queued → started)")
    
    if timing_metrics["dispatch_time"] is not None:
        logger.info(f"   🚀 Dispatch Time: {timing_metrics['dispatch_time']:.2f}s (dispatched → started)")
    
    logger.info("=" * 80)


# =============================================================================
# Public Endpoints
# =============================================================================

@public_router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns server status and optionally checks component connectivity.
    """
    components = {}
    
    # Check database
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        components["database"] = "ok"
    except Exception:
        components["database"] = "error"
    
    # Check R2
    if _s3_client:
        try:
            _s3_client.head_bucket(Bucket=_r2_bucket)
            components["r2"] = "ok"
        except Exception:
            components["r2"] = "error"
    
    return HealthResponse(
        status="ok" if all(v == "ok" for v in components.values()) else "degraded",
        timestamp=utc_naive_now(),
        version=os.getenv("APP_VERSION", "1.0.0"),
        components=components if components else None,
    )


# @public_router.get("/download/{job_id}")
# async def download_result(job_id: str, request: Request):
#     """
#     Download job result from R2 storage.
    
#     Supports HTTP Range requests for video streaming.
    
#     IMPORTANT: Uses retry logic to handle transient R2 connection issues.
#     """
#     if not _s3_client or not _r2_bucket:
#         raise HTTPException(status_code=503, detail="Storage not configured")
    
#     # Get job
#     db = SessionLocal()
#     try:
#         job = db.query(Job).filter(Job.job_id == job_id).first()
#         if not job:
#             raise HTTPException(status_code=404, detail="Job not found")
        
#         if job.status != JobState.COMPLETED.value:
#             raise HTTPException(status_code=404, detail="Job not completed")
        
#         r2_key = job.result_r2_key
#         if not r2_key:
#             raise HTTPException(status_code=404, detail="No result available")
#     finally:
#         db.close()
    
#     def _stream_from_r2(bucket: str, key: str, range_str: Optional[str] = None):
#         """
#         Stream file from R2 storage.
        
#         Note: This is a synchronous generator used by StreamingResponse.
#         R2 errors during streaming will be caught by the client as incomplete response.
#         """
#         try:
#             if range_str:
#                 resp = _s3_client.get_object(Bucket=bucket, Key=key, Range=range_str)
#             else:
#                 resp = _s3_client.get_object(Bucket=bucket, Key=key)
            
#             # Stream chunks
#             for chunk in resp["Body"].iter_chunks():
#                 yield chunk
                
#         except Exception as e:
#             logger.error(f"R2 stream error for {key}: {e}")
#             # For streaming responses, we can't easily retry mid-stream
#             # The client will see an incomplete response and can retry
#             raise
    
#     try:
#         # Get object metadata
#         head = _s3_client.head_object(Bucket=_r2_bucket, Key=r2_key)
#         content_length = head["ContentLength"]
#         content_type = head.get("ContentType", mimetypes.guess_type(r2_key)[0] or "application/octet-stream")
        
#         # Handle Range request
#         range_header = request.headers.get("range")
#         if range_header:
#             # Parse range
#             range_match = range_header.replace("bytes=", "").split("-")
#             start = int(range_match[0]) if range_match[0] else 0
#             end = int(range_match[1]) if len(range_match) > 1 and range_match[1] else content_length - 1
            
#             range_str = f"bytes={start}-{end}"
            
#             return StreamingResponse(
#                 _stream_from_r2(_r2_bucket, r2_key, range_str),
#                 status_code=206,
#                 media_type=content_type,
#                 headers={
#                     "Content-Range": f"bytes {start}-{end}/{content_length}",
#                     "Accept-Ranges": "bytes",
#                     "Content-Length": str(end - start + 1),
#                 }
#             )
        
#         # Full download - DON'T set Content-Length to avoid mismatch errors
#         # Use chunked transfer encoding instead - this prevents 
#         # "Too little data for declared Content-Length" errors if R2 stream fails.
#         # Clients handle chunked encoding well, and retrying the request works.
#         return StreamingResponse(
#             _stream_from_r2(_r2_bucket, r2_key),
#             media_type=content_type,
#             headers={
#                 "Accept-Ranges": "bytes",
#                 # X-Content-Length is informational only, not enforced
#                 "X-Content-Length": str(content_length),
#             }
#         )
        
#     except ClientError as e:
#         if e.response["Error"]["Code"] == "404":
#             raise HTTPException(status_code=404, detail="File not found")
#         logger.error(f"R2 download error for job {job_id}: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

@public_router.get("/download/{job_id}")
async def download_result(job_id: str, request: Request):
    if not _s3_client or not _r2_bucket:
        raise HTTPException(status_code=503, detail="Storage not configured")

    logger.info(f"[download] job_id={job_id}")

    # ----------------------------
    # Fetch job
    # ----------------------------
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job or job.status != JobState.COMPLETED.value:
            raise HTTPException(status_code=404, detail="Result not available")

        r2_key = job.result_r2_key
        if not r2_key:
            raise HTTPException(status_code=404, detail="No result key")
    finally:
        db.close()

    # ----------------------------
    # HEAD object (log EVERYTHING)
    # ----------------------------
    try:
        head = _s3_client.head_object(Bucket=_r2_bucket, Key=r2_key)
    except Exception as e:
        logger.exception("[download] head_object failed")
        raise HTTPException(status_code=404, detail="File not found")

    total_size = int(head["ContentLength"])
    content_type = head.get(
        "ContentType",
        mimetypes.guess_type(r2_key)[0] or "application/octet-stream",
    )

    logger.info(f"[download] size={total_size} type={content_type}")

    # ----------------------------
    # SAFE stream generator (NO raises)
    # ----------------------------
    def stream_from_r2(range_header: str | None):
        try:
            kwargs = dict(Bucket=_r2_bucket, Key=r2_key)
            if range_header:
                kwargs["Range"] = range_header

            resp = _s3_client.get_object(**kwargs)

            for chunk in resp["Body"].iter_chunks(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                yield chunk

        except GeneratorExit:
            logger.info("[download] client disconnected")
            return

        except Exception as e:
            logger.exception("[download] stream error")
            return  # <- NEVER raise

    # ----------------------------
    # RANGE handling
    # ----------------------------
    range_header = request.headers.get("range")

    if range_header:
        try:
            start_str, end_str = range_header.replace("bytes=", "").split("-")
            start = int(start_str)
            end = int(end_str) if end_str else total_size - 1
            end = min(end, total_size - 1)
        except Exception:
            raise HTTPException(status_code=416, detail="Invalid Range")

        content_length = end - start + 1
        range_str = f"bytes={start}-{end}"

        logger.info(f"[download] range {range_str}")

        return StreamingResponse(
            stream_from_r2(range_str),
            status_code=206,
            media_type=content_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{total_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
            },
        )

    # ----------------------------
    # FULL download
    # ----------------------------
    logger.info("[download] full download")

    return StreamingResponse(
        stream_from_r2(None),
        media_type=content_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(total_size),
            "Content-Disposition": f'inline; filename="{os.path.basename(r2_key)}"',
        },
    )


# =============================================================================
# Client Endpoints
# =============================================================================

@client_router.post("/submit", response_model=SubmitResponse)
async def submit_job(
    request: Request,
    payload: SubmitRequest,
    background_tasks: BackgroundTasks,
    auth: Dict = Depends(require_any_auth),
):
    """
    Submit a new job for processing.
    
    Returns job ID and optionally dispatches immediately if a worker is ready.
    """
    ip = get_client_ip(request)
    
    # Validate model
    try:
        model = validate_model(payload.model)
    except InvalidModelError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Generate job ID and result key
    job_id = str(uuid.uuid4())
    result_key = f"{_image_prefix}/{job_id}.png"
    
    # Generate presigned URL
    upload_url = create_presigned_put_url(result_key)
    
    # Create job in database with QUEUED status directly
    # CRITICAL: Create as QUEUED to avoid blocking update_job_state call
    # This allows us to return response immediately without waiting for dispatch
    db = SessionLocal()
    try:
        job = Job(
            job_id=job_id,
            model=model,
            prompt=payload.prompt,
            params=payload.params or {},
            status=JobState.QUEUED.value,  # Start as QUEUED - no need for separate update
            result_r2_key=result_key,
            output_key=result_key,
            fcm_token=payload.fcm_token,
            created_at=utc_naive_now(),
            updated_at=utc_naive_now(),
            timeout_at=utc_naive_now() + timedelta(minutes=30),
        )
        db.add(job)
        
        # Audit log is optional - don't block if it fails
        try:
            create_audit_log(
                db,
                event_type="job_created",
                message=f"Job {job_id} created for model {model}",
                target=job_id,
                target_type="job",
                actor_ip=ip,
                details={"model": model},
            )
        except Exception as audit_error:
            logger.warning(f"Failed to create audit log for job {job_id}: {audit_error}")
        
        db.commit()
    except Exception as db_error:
        logger.error(f"Failed to create job {job_id}: {db_error}")
        try:
            db.rollback()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="Failed to create job")
    finally:
        db.close()
    
    # Log job creation (non-blocking, can fail silently)
    try:
        audit_logger.job_created(job_id, model, ip, auth.get("fingerprint_hash", auth.get("type")))
    except Exception:
        pass  # Non-critical
    
    # Dispatch in background task - return immediately without waiting
    if _pod_manager:
        background_tasks.add_task(_dispatch_job_background, job_id, model, payload.params.get("mode"))
    elif _job_manager:
        # Fallback: if no pod_manager, at least add to pending queue
        # (but this should rarely happen in normal operation)
        logger.warning(f"No pod_manager available, job {job_id} will be handled by recovery system")
    
    # Build download URL
    download_url = f"{_controller_url}/download/{job_id}" if _controller_url else f"/download/{job_id}"
    
    # Return IMMEDIATELY - no blocking operations after this point
    return SubmitResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        model=model,
        routed_to_worker=None,  # Will be set when actually dispatched
        upload_url=upload_url,
        result_key=result_key,
        download_url=download_url,
        created_at=utc_naive_now(),
    )


async def _dispatch_job_background(job_id: str, model: str, mode: Optional[str], retry_count: int = 0):
    """
    Background task to dispatch job - includes actual HTTP dispatch to worker.
    
    CRITICAL FIX: Track machines attempted per job to limit to 3 DIFFERENT machines.
    This prevents infinite loops when all available machines are problematic.
    """
    # Maximum of 3 DIFFERENT machines per job
    MAX_MACHINE_ATTEMPTS = 3
    
    # Get current machines attempted from job's error_log
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found in database")
            return
        
        # Track machines attempted
        machines_attempted = set()
        if job.error_log:
            for entry in job.error_log:
                if isinstance(entry, dict) and entry.get("machine_id"):
                    machines_attempted.add(str(entry["machine_id"]))
        
        # Check if we've exhausted all machine attempts
        if len(machines_attempted) >= MAX_MACHINE_ATTEMPTS and retry_count > 0:
            logger.error(
                f"Job {job_id} has failed on {len(machines_attempted)} different machines: {machines_attempted}. "
                f"Marking as permanently FAILED."
            )
            if _job_manager:
                _job_manager.mark_failed(
                    job_id, 
                    f"Failed on {MAX_MACHINE_ATTEMPTS} different machines: {list(machines_attempted)}",
                    None
                )
            else:
                job.status = JobState.FAILED.value
                job.error = f"Failed on {MAX_MACHINE_ATTEMPTS} different machines"
                job.updated_at = utc_naive_now()
                db.commit()
            return
            
    finally:
        db.close()
    
    try:
        # Get worker_id via pod manager
        worker_id = await _job_manager.assign_job(job_id, model, mode)
        logger.info(f"Job {job_id} assigned to worker {worker_id}, now dispatching...")
        
        # Actually dispatch the job to the worker
        success = await _dispatch_to_worker(job_id, worker_id)
        
        if success:
            logger.info(f"Job {job_id} successfully dispatched to worker {worker_id}")
        else:
            logger.error(f"Job {job_id} failed to dispatch to worker {worker_id}")
            
            # CRITICAL: Trigger retry to create new pod on different machine
            # This handles cases where:
            # 1. Health check failed (machine blacklisted, pod destroyed)
            # 2. Machine 619 (5090) blacklisted → should try int4 variant on 4090
            max_retries = 5  # More retries to allow trying different machines
            if retry_count < max_retries:
                retry_delay = 10  # Wait 10s for blacklist to settle
                logger.warning(
                    f"Job {job_id} dispatch failed (attempt {retry_count + 1}/{max_retries}). "
                    f"Machines attempted so far: {machines_attempted}. "
                    f"Retrying in {retry_delay}s (will try next variant/machine)..."
                )
                await asyncio.sleep(retry_delay)
                
                # Trigger retry (this will call assign_job again, which respects blacklist)
                asyncio.create_task(_dispatch_job_background(job_id, model, mode, retry_count + 1))
            else:
                logger.error(
                    f"Job {job_id} failed to dispatch after {max_retries} attempts. "
                    f"Job will remain in QUEUED for recovery system to handle."
                )
            
    except Exception as e:
        logger.error(f"Background dispatch failed for {job_id}: {e}")
        # Ensure job is reverted to QUEUED so recovery system can pick it up
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job and job.status not in [
                JobState.COMPLETED.value,
                JobState.FAILED.value,
                JobState.RUNNING.value,  # Don't revert if already running
            ]:
                job.status = JobState.QUEUED.value
                job.worker_id = None
                job.updated_at = utc_naive_now()
                if job.error_log is None:
                    job.error_log = []
                job.error_log.append({
                    "timestamp": utc_naive_now().isoformat(),
                    "state": "dispatch_background_error",
                    "error": str(e),
                })
                db.commit()
                logger.info(
                    f"Job {job_id} reverted to QUEUED after dispatch error. "
                    f"Will be picked up by recovery system."
                )
        finally:
            db.close()


async def _check_worker_health(endpoint: str, timeout: int = 5) -> bool:
    """
    Check if worker's /health endpoint is responding.
    Returns True if worker is healthy, False otherwise.
    
    DEPRECATED: Use _check_worker_health_enhanced for detailed status.
    """
    try:
        health_url = f"{endpoint.rstrip('/')}/health"
        resp = requests.get(health_url, timeout=timeout)
        return resp.status_code == 200
    except Exception as e:
        logger.debug(f"Worker health check failed for {endpoint}: {e}")
        return False


async def _check_worker_health_enhanced(endpoint: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Enhanced health check that returns detailed worker status.
    
    CRITICAL: Understands difference between "busy processing" vs "broken"!
    
    Returns:
    {
        "healthy": bool,       # True if worker responded
        "ready": bool,         # True if worker can accept jobs NOW
        "status": str,         # "ok", "starting", "busy", "overloaded", "error", "timeout"
        "queue_size": int,     # Jobs waiting in worker queue
        "active_jobs": int,    # Jobs currently being processed
        "processing": bool,    # True if worker is processing jobs (CRITICAL!)
        "details": dict,       # Full health response
    }
    """
    result = {
        "healthy": False,
        "ready": False,
        "status": "error",
        "queue_size": 0,
        "active_jobs": 0,
        "processing": False,
        "details": {}
    }
    
    try:
        health_url = f"{endpoint.rstrip('/')}/health"
        resp = requests.get(health_url, timeout=timeout)
        
        if resp.status_code == 200:
            result["healthy"] = True
            data = resp.json()
            result["details"] = data
            
            # Parse worker status
            status = data.get("status", "unknown")
            result["status"] = status
            result["queue_size"] = data.get("queue_size", 0)
            result["active_jobs"] = data.get("active_jobs", 0)
            result["processing"] = data.get("processing", False)  # NEW: critical field!
            
            # Worker is ready if explicitly stated, or if status indicates readiness
            result["ready"] = data.get("ready", status in ("ok", "busy"))
            
            logger.debug(
                f"Worker health: status={status}, ready={result['ready']}, "
                f"processing={result['processing']}, queue={result['queue_size']}, "
                f"active={result['active_jobs']}"
            )
        else:
            logger.debug(f"Worker health check returned {resp.status_code}")
            
    except requests.exceptions.Timeout:
        # CRITICAL: Timeout doesn't mean worker is broken!
        # Worker might be busy processing and can't respond quickly
        logger.debug(f"Worker health check timeout for {endpoint} (worker may be busy processing)")
        result["status"] = "timeout"
        result["processing"] = True  # Assume processing if timeout (conservative)
        
    except Exception as e:
        logger.debug(f"Worker health check failed for {endpoint}: {e}")
        result["status"] = "error"
    
    return result


async def _recover_jobs_on_worker(
    worker_id: str,
    exclude_job_id: Optional[str] = None,
    db: Optional[Any] = None,
) -> List[str]:
    """
    Recover jobs that were RUNNING on a worker (when the pod is being destroyed).
    
    CRITICAL: When a pod dies unexpectedly, any RUNNING jobs on it must be requeued
    so they can be processed by a different worker. Otherwise, users see jobs stuck
    in "processing" state forever.
    
    Args:
        worker_id: Worker ID whose jobs should be recovered
        exclude_job_id: Job ID to exclude from recovery (e.g., the job that triggered destroy)
        db: Optional existing DB session
        
    Returns:
        List of job IDs that were recovered
    """
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    recovered_jobs = []
    
    try:
        # Find all RUNNING and DISPATCHING jobs on this worker
        jobs_to_recover = db.query(Job).filter(
            Job.worker_id == worker_id,
            Job.status.in_([JobState.RUNNING.value, JobState.DISPATCHING.value]),
        ).all()
        
        for job in jobs_to_recover:
            # Skip the excluded job (if specified)
            if exclude_job_id and job.job_id == exclude_job_id:
                continue
            
            # Requeue the job
            old_status = job.status
            job.status = JobState.QUEUED.value
            job.worker_id = None
            job.updated_at = utc_naive_now()
            
            # Clear any retry count to give job fresh attempts
            if job.retry_count is None:
                job.retry_count = 0
            
            # Log the recovery in error_log
            if job.error_log is None:
                job.error_log = []
            job.error_log.append({
                "timestamp": utc_naive_now().isoformat(),
                "state": "pod_destruction_recovery",
                "error": f"Job recovered (was {old_status}) after worker {worker_id} pod was destroyed",
                "previous_status": old_status,
            })
            
            recovered_jobs.append(job.job_id)
            logger.warning(
                f"Recovered job {job.job_id} (was {old_status}) from destroyed worker {worker_id}"
            )
        
        if recovered_jobs:
            db.commit()
            logger.info(
                f"Recovered {len(recovered_jobs)} job(s) from worker {worker_id}: {recovered_jobs}"
            )
        
    except Exception as e:
        logger.error(f"Error recovering jobs from worker {worker_id}: {e}")
        db.rollback()
    finally:
        if close_db:
            db.close()
    
    return recovered_jobs


async def _dispatch_to_worker(job_id: str, worker_id: str) -> bool:
    """
    Actually dispatch a job to a worker via HTTP POST to /run endpoint.
    
    This is the critical step that sends the job to the worker!
    """
    db = SessionLocal()
    try:
        # Get job details
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            logger.error(f"_dispatch_to_worker: job {job_id} not found")
            return False
        
        # Get worker details
        worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
        if not worker:
            logger.error(f"_dispatch_to_worker: worker {worker_id} not found")
            # Revert job status
            job.status = JobState.QUEUED.value
            job.worker_id = None
            db.commit()
            return False
        
        # CRITICAL FIX: Check worker DB status BEFORE attempting dispatch
        # Workers in "starting" state are still loading models and will fail
        if worker.status == "starting":
            logger.warning(
                f"Worker {worker_id} is in 'starting' state (models not loaded). "
                f"Cannot dispatch job {job_id}. Reverting to QUEUED."
            )
            job.status = JobState.QUEUED.value
            job.worker_id = None
            job.updated_at = utc_naive_now()
            if job.error_log is None:
                job.error_log = []
            job.error_log.append({
                "timestamp": utc_naive_now().isoformat(),
                "state": "dispatch_worker_not_ready",
                "error": f"Worker {worker_id} still in 'starting' state",
            })
            db.commit()
            return False
        
        # Verify pod still exists in pod_manager tracking
        # (If pod was destroyed, it won't be in tracking anymore)
        pod_uuid = worker.container_id
        if _pod_manager and pod_uuid:
            async with _pod_manager._lock:
                pod_exists = pod_uuid in _pod_manager.individual_pods
            
            if not pod_exists:
                logger.warning(
                    f"Pod {pod_uuid} (worker {worker_id}) no longer exists in tracking. "
                    f"Job {job_id} cannot be dispatched. This likely means the pod was destroyed."
                )
                # Revert job to QUEUED and clear worker assignment
                job.status = JobState.QUEUED.value
                job.worker_id = None
                job.updated_at = utc_naive_now()
                if job.error_log is None:
                    job.error_log = []
                job.error_log.append({
                    "timestamp": utc_naive_now().isoformat(),
                    "state": "dispatch_failed_pod_destroyed",
                    "error": f"Pod {pod_uuid} was destroyed before dispatch",
                })
                db.commit()
                return False
        
        # Mark job as DISPATCHING (use job_manager to ensure dispatched_at is set)
        if _job_manager:
            _job_manager.update_job_state(job_id, JobState.DISPATCHING)
        else:
            job.status = JobState.DISPATCHING.value
            job.dispatched_at = utc_naive_now()
            job.updated_at = utc_naive_now()
            db.commit()
        
        # Generate presigned URL for upload
        presigned_url = create_presigned_put_url(job.result_r2_key) if job.result_r2_key else None
        
        # Build the run payload
        run_payload = {
            "job_id": job.job_id,
            "prompt": job.prompt,
            "model": job.model,
            "mode": (job.params or {}).get("mode"),
            "params": job.params or {},
            "upload_url": presigned_url,
            "result_key": job.result_r2_key,
        }
        
        # Get worker endpoint
        endpoint = worker.endpoint
        if not endpoint:
            logger.error(f"Worker {worker_id} has no endpoint configured")
            job.status = JobState.QUEUED.value
            job.worker_id = None
            db.commit()
            return False
        
        url = f"{endpoint.rstrip('/')}/run"
        logger.info(f"Dispatching job {job_id} to {url}")
        
        # CRITICAL FIX: Check if worker is healthy before dispatching
        # BUT: Don't destroy pod if it's just busy processing jobs!
        health_check = await _check_worker_health_enhanced(endpoint, timeout=5)
        
        # Check if health check failed
        if not health_check.get("healthy", False):
            # Health check completely failed (no response at all)
            worker_status = health_check.get("status", "unknown")
            is_processing = health_check.get("processing", False)
            
            # CRITICAL: If worker has running jobs in DB, check if they're actually stuck
            # A job marked "RUNNING" but running for > 10 minutes is likely stuck
            has_healthy_running_jobs = False
            try:
                now = utc_naive_now()
                running_jobs = db.query(Job).filter(
                    Job.worker_id == worker_id,
                    Job.status == JobState.RUNNING.value
                ).all()
                
                # Check if any running jobs are healthy (not stuck)
                max_job_runtime_seconds = 600  # 10 minutes max for image processing
                
                for rjob in running_jobs:
                    if rjob.started_at:
                        runtime = (now - rjob.started_at).total_seconds()
                        if runtime < max_job_runtime_seconds:
                            # Job is running and within reasonable time - worker is busy but healthy
                            has_healthy_running_jobs = True
                            logger.info(
                                f"Job {rjob.job_id} has been running for {runtime:.0f}s (< {max_job_runtime_seconds}s), "
                                f"worker {worker_id} is BUSY but healthy"
                            )
                        else:
                            # Job stuck for too long
                            logger.error(
                                f"Job {rjob.job_id} has been running for {runtime:.0f}s (> {max_job_runtime_seconds}s), "
                                f"likely STUCK! Worker {worker_id} may be frozen/crashed."
                            )
                    else:
                        # Job has no started_at - likely stuck in dispatch
                        logger.warning(
                            f"Job {rjob.job_id} marked RUNNING but has no started_at timestamp, "
                            f"likely stuck in dispatch"
                        )
                
                if has_healthy_running_jobs:
                    logger.warning(
                        f"Worker {worker_id} health check failed BUT has healthy RUNNING job(s). "
                        f"Worker is likely just BUSY processing. Proceeding with dispatch (job will queue)."
                    )
                    # Proceed with dispatch - worker is busy but functioning
                elif len(running_jobs) > 0:
                    logger.error(
                        f"Worker {worker_id} has {len(running_jobs)} RUNNING job(s) but ALL are stuck "
                        f"(running > {max_job_runtime_seconds}s). Worker is FROZEN. Will destroy."
                    )
                    # Don't proceed - worker is stuck, destroy it below
                    has_healthy_running_jobs = False
                    
            except Exception as e:
                logger.error(f"Error checking running jobs for worker {worker_id}: {e}")
            
            # Only destroy if truly unhealthy (not just busy with healthy jobs)
            if not has_healthy_running_jobs:
                logger.error(
                    f"Worker {worker_id} health check failed (status: {worker_status}) "
                    f"and has no running jobs. Pod may be broken. Waiting 5s and retrying..."
                )
                await asyncio.sleep(5)
                health_check = await _check_worker_health_enhanced(endpoint, timeout=5)
                
                if not health_check.get("healthy", False):
                    logger.error(
                        f"Worker {worker_id} still unhealthy after retry. "
                        f"Marking pod as bad and reverting job to QUEUED."
                    )
                    
                    # Revert job to QUEUED
                    if _job_manager:
                        _job_manager.update_job_state(job_id, JobState.QUEUED, force=True)
                    else:
                        job.status = JobState.QUEUED.value
                        job.worker_id = None
                        job.updated_at = utc_naive_now()
                        if job.error_log is None:
                            job.error_log = []
                        job.error_log.append({
                            "timestamp": utc_naive_now().isoformat(),
                            "state": "health_check_failed",
                            "error": "Worker failed health check twice before dispatch",
                        })
                        db.commit()
                    
                    # Get machine_id and check for OTHER running jobs
                    machine_id = None
                    if _pod_manager and pod_uuid:
                        async with _pod_manager._lock:
                            pod_info = _pod_manager.individual_pods.get(pod_uuid)
                            if pod_info:
                                machine_id = pod_info.machine_id
                    
                    # CRITICAL: Check if OTHER jobs are running on this pod before destroying
                    # We don't want to destroy a pod that has successfully running jobs
                    other_running_jobs = []
                    try:
                        other_running = db.query(Job).filter(
                            Job.worker_id == worker_id,
                            Job.status == JobState.RUNNING.value,
                            Job.job_id != job_id,  # Exclude current job
                        ).all()
                        other_running_jobs = [j.job_id for j in other_running]
                    except Exception as e:
                        logger.error(f"Error checking for other running jobs: {e}")
                    
                    if other_running_jobs:
                        # DON'T DESTROY THE POD - other jobs are running!
                        logger.warning(
                            f"Worker {worker_id} has {len(other_running_jobs)} OTHER running job(s): {other_running_jobs}. "
                            f"NOT destroying pod {pod_uuid}. Job {job_id} will retry on a different worker."
                        )
                        # Don't blacklist machine either - it's working, just busy
                        return False
                    
                    # No other jobs running - safe to destroy
                    # INCREASED blacklist time to 600s (10 minutes) to prevent repeated failures
                    # Machine 619 kept being reused because 60s was too short
                    if machine_id and _pod_manager:
                        await _pod_manager.mark_machine_bad(str(machine_id), duration=600)
                        logger.info(f"Blacklisted machine {machine_id} for 600s (10 min) due to unhealthy worker")
                    
                    # Destroy the pod
                    if _pod_manager and pod_uuid:
                        try:
                            # Before destroying, recover any RUNNING jobs (shouldn't be any, but just in case)
                            await _recover_jobs_on_worker(worker_id, exclude_job_id=job_id, db=db)
                            
                            await _pod_manager.destroy_individual_pod(
                                pod_uuid,
                                reason="Health check failed twice before dispatch"
                            )
                            logger.info(f"Destroyed unhealthy pod {pod_uuid}")
                        except Exception as e:
                            logger.error(f"Failed to destroy unhealthy pod {pod_uuid}: {e}")
                    
                    return False
        
        # If worker is not ready but is healthy, check specific status
        if not health_check.get("ready", False) and health_check.get("healthy", False):
            worker_status = health_check.get("status", "unknown")
            is_processing = health_check.get("processing", False)
            
            if worker_status == "starting":
                logger.info(
                    f"Worker {worker_id} is still starting up. "
                    f"Waiting 10s for worker to become ready..."
                )
                await asyncio.sleep(10)
                health_check = await _check_worker_health_enhanced(endpoint, timeout=5)
                
                if not health_check.get("ready", False):
                    logger.warning(
                        f"Worker {worker_id} still not ready after 10s (status: {health_check.get('status')}). "
                        f"Proceeding with dispatch anyway - job may queue on worker."
                    )
            elif worker_status == "overloaded":
                logger.warning(
                    f"Worker {worker_id} is overloaded (queue full). "
                    f"Reverting job to QUEUED to try another worker."
                )
                # Don't blacklist - worker is functioning, just full
                if _job_manager:
                    _job_manager.update_job_state(job_id, JobState.QUEUED, force=True)
                return False
            elif worker_status == "busy" and is_processing:
                logger.info(
                    f"Worker {worker_id} is busy processing jobs. "
                    f"Proceeding with dispatch - job will queue behind active job."
                )
                # Proceed - worker is functioning, just busy
            else:
                logger.warning(
                    f"Worker {worker_id} not ready (status: {worker_status}). "
                    f"Proceeding with dispatch anyway."
                )
        
        # Make the HTTP request with smart retry logic
        # CRITICAL: Worker returns 409 if same job_id is already queued
        # So we must NOT retry if the worker received the request (even if we timed out)
        last_error = None
        max_attempts = 3
        worker_received_job = False  # Track if worker confirmed receipt
        
        for attempt in range(max_attempts):
            try:
                # Longer timeouts for image processing models: 20s, 30s, 40s
                # This gives worker time to respond even under load
                timeout = 20 + (attempt * 10)
                
                logger.debug(
                    f"Dispatch attempt {attempt + 1}/{max_attempts} for job {job_id} "
                    f"(timeout={timeout}s)"
                )
                
                resp = requests.post(url, json=run_payload, timeout=timeout)
                resp.raise_for_status()
                
                # Success!
                worker_received_job = True
                last_error = None
                logger.info(f"Worker {worker_id} accepted job {job_id} successfully")
                break
                
            except requests.exceptions.Timeout as e:
                last_error = e
                error_str = str(e).lower()
                
                # READ timeout means request was sent but response didn't come back
                # Worker likely received it and queued it!
                if "read timeout" in error_str or "read timed out" in error_str:
                    if attempt == 0:
                        # First attempt read timeout - retry once in case network is slow
                        logger.warning(
                            f"Dispatch attempt {attempt + 1}/{max_attempts} READ TIMED OUT for job {job_id} "
                            f"(timeout={timeout}s). Retrying once in case of slow network..."
                        )
                        await asyncio.sleep(5)
                        continue
                    else:
                        # Second/third attempt read timeout - worker definitely has it
                        logger.warning(
                            f"Dispatch attempt {attempt + 1}/{max_attempts} READ TIMED OUT for job {job_id}. "
                            f"Worker likely received job on first attempt. Treating as success."
                        )
                        last_error = None  # Clear error - worker has the job
                        break  # Exit retry loop
                else:
                    # CONNECT timeout - safe to retry
                    logger.warning(
                        f"Dispatch attempt {attempt + 1}/{max_attempts} CONNECT TIMED OUT for job {job_id}: {e}"
                    )
                    if attempt < max_attempts - 1:
                        backoff = 5 * (2 ** attempt)
                        logger.info(f"Waiting {backoff}s before retry...")
                        await asyncio.sleep(backoff)
                    continue
                    
            except requests.exceptions.ConnectionError as e:
                last_error = e
                error_str = str(e).lower()
                
                # Connection refused/reset - safe to retry
                logger.warning(
                    f"Dispatch attempt {attempt + 1}/{max_attempts} connection failed for job {job_id}: {e}"
                )
                if attempt < max_attempts - 1:
                    # Longer backoff for connection errors: 5s, 10s
                    backoff = 5 * (2 ** attempt)
                    logger.info(f"Waiting {backoff}s before retry (connection error)...")
                    await asyncio.sleep(backoff)
                continue
                
            except requests.exceptions.HTTPError as e:
                last_error = e
                
                if not e.response:
                    logger.warning(
                        f"Dispatch attempt {attempt + 1}/{max_attempts} HTTP error with no response for job {job_id}: {e}"
                    )
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(3 * (attempt + 1))
                    continue
                
                status_code = e.response.status_code
                
                # 409 Conflict means worker already has this job - SUCCESS!
                if status_code == 409:
                    logger.info(
                        f"Worker returned 409 for job {job_id} - job already queued on worker. "
                        f"Treating as successful dispatch."
                    )
                    worker_received_job = True
                    last_error = None  # Clear error - this is actually success
                    break
                
                # 503 Service Unavailable means worker queue is full
                elif status_code == 503:
                    logger.warning(
                        f"Worker returned 503 for job {job_id} - queue full. "
                        f"Will try different worker/pod."
                    )
                    # Don't retry on same worker - it's overloaded
                    last_error = e
                    break  # Exit retry loop to try different worker
                
                # Other HTTP errors
                else:
                    logger.warning(
                        f"Dispatch attempt {attempt + 1}/{max_attempts} HTTP {status_code} error for job {job_id}: {e}"
                    )
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(3 * (attempt + 1))
                    continue
                    
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Dispatch attempt {attempt + 1}/{max_attempts} failed for job {job_id}: {e}"
                )
                if attempt < max_attempts - 1:
                    backoff = 3 * (attempt + 1)
                    await asyncio.sleep(backoff)
                continue
        
        # Check if worker confirmed receipt (either 200 or 409)
        if worker_received_job:
            # Worker has the job - success!
            logger.info(f"Worker {worker_id} confirmed receipt of job {job_id}")
            # Continue to mark job as RUNNING below
            last_error = None
        
        if last_error:
            logger.error(f"All dispatch attempts failed for job {job_id}: {last_error}")
            
            # Classify error type to determine if machine should be blacklisted
            error_str = str(last_error).lower()
            is_http_error = isinstance(last_error, requests.exceptions.HTTPError)
            is_connection_error = isinstance(last_error, requests.exceptions.ConnectionError) or \
                                  "connection refused" in error_str or \
                                  "failed to establish" in error_str
            is_read_timeout = isinstance(last_error, requests.exceptions.Timeout) and \
                             ("read timeout" in error_str or "read timed out" in error_str)
            is_connect_timeout = isinstance(last_error, requests.exceptions.Timeout) and \
                                "connect timeout" in error_str
            
            # Check for 503 (queue full) - don't blacklist, just try different worker
            if is_http_error and hasattr(last_error, 'response') and last_error.response:
                if last_error.response.status_code == 503:
                    logger.info(
                        f"Worker {worker_id} queue full (503). Reverting job to QUEUED to try different worker."
                    )
                    if _job_manager:
                        _job_manager.update_job_state(job_id, JobState.QUEUED, force=True)
                    else:
                        job.status = JobState.QUEUED.value
                        job.worker_id = None
                        job.updated_at = utc_naive_now()
                        db.commit()
                    return False  # Don't destroy pod - it's working fine
            
            # READ timeout is special: worker might have received the job
            if is_read_timeout:
                logger.warning(
                    f"Dispatch READ TIMEOUT for job {job_id}. Worker may have received job and queued it. "
                    f"Marking job as DISPATCHING and letting worker call job_complete when done. "
                    f"NOT destroying pod."
                )
                # Keep job in DISPATCHING state - worker will call job_complete
                # Don't destroy pod - worker might be processing
                return True  # Treat as successful (worker will complete it)
            
            should_blacklist_machine = is_connection_error or is_connect_timeout
            blacklist_duration = 600  # 10 minutes for all failures (prevent rapid retries)
            
            if is_connect_timeout:
                # Connect timeout - machine might be completely unreachable
                should_blacklist_machine = True
                blacklist_duration = 600  # 10 minutes
                logger.warning(
                    f"Dispatch CONNECT TIMEOUT for job {job_id} - network or firewall issue"
                )
            
            # Get job details first
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if not job:
                logger.error(f"Job {job_id} not found after dispatch failure")
                return False
            
            pod_uuid = worker.container_id if worker else None
            
            # Get machine_id from pod_info (more reliable than worker.machine_id)
            machine_id = None
            if _pod_manager and pod_uuid:
                async with _pod_manager._lock:
                    pod_info = _pod_manager.individual_pods.get(pod_uuid)
                    if pod_info:
                        machine_id = pod_info.machine_id
            
            # Fallback to worker.machine_id if pod_info doesn't have it
            if not machine_id and worker:
                machine_id = worker.machine_id
            
            error_type = "connection" if is_connection_error else \
                        ("connect_timeout" if is_connect_timeout else "other")
            
            logger.info(
                f"Dispatch failure for pod {pod_uuid}, machine_id={machine_id}, "
                f"error_type={error_type}, should_blacklist={should_blacklist_machine}"
            )
            
            # After first job dispatch failure, immediately destroy pod and requeue all jobs
            # Check for ALL jobs associated with this pod (assigned to this worker in database)
            all_jobs_on_pod = []
            
            if pod_uuid and worker_id:
                # 1. Find jobs assigned to this worker in database
                all_jobs_on_pod = db.query(Job).filter(
                    Job.worker_id == worker_id,
                    Job.status.notin_([
                        JobState.COMPLETED.value,
                        JobState.FAILED.value,
                        JobState.TIMEOUT.value,
                        JobState.CANCELLED.value,
                        JobState.RECOVERED.value,
                    ])
                ).all()
                
                # Note: PodInfo doesn't store job IDs in a queue, so we rely only on database assignments
                
                logger.info(
                    f"Found {len(all_jobs_on_pod)} job(s) assigned to worker {worker_id} on pod {pod_uuid}"
                )
            
            # Analyze job states on this pod to decide if we should destroy immediately
            if len(all_jobs_on_pod) > 0:
                # Count jobs in different states
                running_jobs = [j for j in all_jobs_on_pod if j.status == JobState.RUNNING.value]
                queued_dispatching_jobs = [j for j in all_jobs_on_pod 
                                          if j.status in [JobState.QUEUED.value, JobState.DISPATCHING.value]]
                
                logger.warning(
                    f"Dispatch failed for job {job_id} on pod {pod_uuid}. "
                    f"Found {len(all_jobs_on_pod)} job(s) on this pod: "
                    f"{len(running_jobs)} RUNNING, {len(queued_dispatching_jobs)} QUEUED/DISPATCHING"
                )
                
                # CRITICAL DECISION: Should we destroy the pod immediately?
                # 
                # If there are RUNNING jobs, we need to be careful:
                # - Connection refused/broken pod → Pod is dead, destroy immediately (jobs are lost anyway)
                # - 409 Conflict → Pod working fine, just busy → DON'T destroy, wait for jobs to complete
                # - Timeouts → Pod might be slow → DON'T destroy, wait for jobs to complete
                
                should_destroy_pod_immediately = False
                
                if len(running_jobs) > 0:
                    # There are RUNNING jobs - be careful!
                    # CRITICAL FIX: Check if RUNNING jobs are actually stuck (exceeded timeout)
                    # This catches jobs that failed silently on the worker
                    
                    job_timeout_minutes = int(os.getenv("JOB_TIMEOUT_MINUTES", "30"))
                    running_job_timeout = timedelta(minutes=job_timeout_minutes)
                    now = utc_naive_now()
                    
                    stuck_running_jobs = []
                    active_running_jobs = []
                    
                    for rj in running_jobs:
                        # Check if job has been running longer than timeout
                        job_start = rj.dispatched_at or rj.updated_at or rj.created_at
                        if job_start:
                            running_duration = now - job_start
                            if running_duration > running_job_timeout:
                                stuck_running_jobs.append(rj)
                                logger.warning(
                                    f"RUNNING job {rj.job_id} has been running for {running_duration.total_seconds() / 60:.1f} min "
                                    f"(timeout: {job_timeout_minutes} min) - likely stuck/silently failed"
                                )
                            else:
                                active_running_jobs.append(rj)
                        else:
                            # No timestamp, assume it's active
                            active_running_jobs.append(rj)
                    
                    if is_connection_error:
                        # Connection refused = pod is broken/dead
                        # RUNNING jobs will never complete anyway, so destroy
                        logger.error(
                            f"Pod {pod_uuid} has {len(running_jobs)} RUNNING job(s), but connection refused. "
                            f"Pod is broken - destroying immediately. RUNNING jobs will be marked as FAILED."
                        )
                        should_destroy_pod_immediately = True
                    elif len(active_running_jobs) == 0 and len(stuck_running_jobs) > 0:
                        # All RUNNING jobs are stuck/timed out - destroy pod
                        logger.error(
                            f"Pod {pod_uuid} has {len(stuck_running_jobs)} STUCK RUNNING job(s) "
                            f"(all exceeded {job_timeout_minutes}min timeout). "
                            f"Treating as dead pod - destroying immediately."
                        )
                        should_destroy_pod_immediately = True
                    elif len(active_running_jobs) > 0:
                        # There are legitimately active RUNNING jobs - keep pod alive
                        # 409 or timeout = pod is working, just busy/slow
                        # Let RUNNING jobs complete, only requeue the failed job
                        logger.info(
                            f"Pod {pod_uuid} has {len(active_running_jobs)} active RUNNING job(s) "
                            f"(within timeout). Dispatch failed with {error_type}, but pod may still be working. "
                            f"NOT destroying pod - will let RUNNING jobs complete. "
                            f"Only requeueing the failed job {job_id}."
                        )
                        # Just requeue this single failed job
                        if _job_manager:
                            _job_manager.update_job_state(job_id, JobState.QUEUED, force=True)
                        else:
                            job.status = JobState.QUEUED.value
                            job.worker_id = None
                            job.updated_at = utc_naive_now()
                            if job.error_log is None:
                                job.error_log = []
                            job.error_log.append({
                                "timestamp": utc_naive_now().isoformat(),
                                "state": "dispatch_failed_pod_busy",
                                "error": f"Dispatch failed but pod has RUNNING jobs, only requeueing this job",
                            })
                            db.commit()
                        return False  # Return false but don't destroy pod
                else:
                    # No RUNNING jobs - safe to destroy and requeue all
                    logger.info(
                        f"Pod {pod_uuid} has no RUNNING jobs, safe to destroy and requeue all {len(all_jobs_on_pod)} job(s)"
                    )
                    should_destroy_pod_immediately = True
                
                if not should_destroy_pod_immediately:
                    return False  # Exit early - don't destroy pod
                
                # Collect all affected job IDs
                affected_job_ids = [j.job_id for j in all_jobs_on_pod]
                affected_model = job.model  # We already fetched job above
                
                # Mark machine as bad (only for connection/timeout errors)
                if _pod_manager and machine_id and should_blacklist_machine:
                    try:
                        await _pod_manager.mark_machine_bad(str(machine_id), duration=blacklist_duration)
                        logger.info(
                            f"Marked machine {machine_id} as bad for {blacklist_duration}s "
                            f"due to dispatch failure (error_type: {'connection' if is_connection_error else 'timeout'})"
                        )
                    except Exception as e:
                        logger.error(f"Failed to mark machine {machine_id} as bad: {e}")
                elif machine_id and not should_blacklist_machine:
                    logger.info(
                        f"Not blacklisting machine {machine_id} - error type doesn't warrant blacklist"
                    )
                
                # Destroy the pod
                if _pod_manager and pod_uuid:
                    try:
                        await _pod_manager.destroy_individual_pod(
                            pod_uuid,
                            reason=f"Dispatch failure - {len(affected_job_ids)} jobs affected"
                        )
                        logger.info(f"Destroyed pod {pod_uuid} due to dispatch failure")
                    except Exception as e:
                        logger.error(f"Failed to destroy pod {pod_uuid}: {e}")
                
                # Revert all affected jobs and clear worker assignment
                # RUNNING jobs should be marked as FAILED (worker might still be processing)
                # QUEUED/DISPATCHING jobs should be reverted to QUEUED for requeue
                jobs_to_requeue = []
                for affected_job_id in affected_job_ids:
                    affected_job = db.query(Job).filter(Job.job_id == affected_job_id).first()
                    if not affected_job:
                        continue
                    
                    current_state = JobState(affected_job.status)
                    error_msg = f"Pod {pod_uuid} destroyed due to dispatch failure"
                    
                    if current_state == JobState.RUNNING:
                        # RUNNING jobs: RETRY instead of permanently failing
                        # CRITICAL FIX: User's jobs should not be lost due to infrastructure issues
                        # Track retry count to prevent infinite retries
                        
                        running_retry_count = affected_job.retry_count or 0
                        max_running_retries = 2  # Allow 2 retries for RUNNING jobs
                        
                        if running_retry_count < max_running_retries:
                            # Requeue for retry on different machine
                            logger.warning(
                                f"Job {affected_job_id} was RUNNING when pod was destroyed. "
                                f"Requeueing for retry (attempt {running_retry_count + 1}/{max_running_retries}) "
                                f"on different machine."
                            )
                            affected_job.status = JobState.QUEUED.value
                            affected_job.worker_id = None
                            affected_job.retry_count = running_retry_count + 1
                            affected_job.updated_at = utc_naive_now()
                            if affected_job.error_log is None:
                                affected_job.error_log = []
                            affected_job.error_log.append({
                                "timestamp": utc_naive_now().isoformat(),
                                "state": "running_pod_destroyed_retry",
                                "error": error_msg,
                                "retry_count": affected_job.retry_count,
                                "machine_id": str(machine_id) if machine_id else None,
                            })
                            db.commit()
                            jobs_to_requeue.append(affected_job_id)
                        else:
                            # Exceeded retry limit - mark as failed
                            logger.error(
                                f"Job {affected_job_id} was RUNNING when pod was destroyed. "
                                f"Retry limit ({max_running_retries}) exceeded. Marking as FAILED."
                            )
                            if _job_manager:
                                try:
                                    _job_manager.mark_failed(
                                        affected_job_id,
                                        error=f"{error_msg} (retry limit exceeded)",
                                        worker_id=affected_job.worker_id
                                    )
                                except Exception as e:
                                    logger.error(f"Failed to mark job {affected_job_id} as failed: {e}")
                            else:
                                affected_job.status = JobState.FAILED.value
                                affected_job.error = f"{error_msg} (retry limit exceeded)"
                                affected_job.updated_at = utc_naive_now()
                                if affected_job.error_log is None:
                                    affected_job.error_log = []
                                affected_job.error_log.append({
                                    "timestamp": utc_naive_now().isoformat(),
                                    "state": "running_pod_destroyed_failed",
                                    "error": error_msg,
                                    "machine_id": str(machine_id) if machine_id else None,
                                })
                                db.commit()
                            
                            # Send FCM notification for RUNNING job that failed permanently
                            if affected_job.fcm_token and not affected_job.notification_sent:
                                try:
                                    asyncio.create_task(
                                        send_fcm_notification(
                                            affected_job.fcm_token,
                                            "Job Failed",
                                            f"Your {affected_job.model} job failed due to pod failure",
                                            {"job_id": affected_job_id, "status": "failed", "error": error_msg},
                                        )
                                    )
                                    affected_job.notification_sent = True
                                except Exception as e:
                                    logger.error(f"Failed to send FCM notification for job {affected_job_id}: {e}")
                            
                            # Clear worker assignment
                            affected_job.worker_id = None
                            db.commit()
                    else:
                        # QUEUED or DISPATCHING jobs: revert to QUEUED for requeue
                        if _job_manager:
                            try:
                                _job_manager.update_job_state(affected_job_id, JobState.QUEUED, force=True)
                                affected_job.worker_id = None
                                db.commit()
                                jobs_to_requeue.append(affected_job_id)
                            except Exception as e:
                                logger.error(f"Failed to requeue job {affected_job_id}: {e}")
                        else:
                            affected_job.status = JobState.QUEUED.value
                            affected_job.worker_id = None
                            affected_job.updated_at = utc_naive_now()
                            if affected_job.error_log is None:
                                affected_job.error_log = []
                            affected_job.error_log.append({
                                "timestamp": utc_naive_now().isoformat(),
                                "state": "dispatch_failed_pod_destroyed",
                                "error": error_msg,
                                "machine_id": str(machine_id) if machine_id else None,  # Track for retry limiting
                            })
                            db.commit()
                            jobs_to_requeue.append(affected_job_id)
                
                # Only requeue jobs that are in QUEUED state (not RUNNING)
                affected_job_ids = jobs_to_requeue
                
                # Create new pod and requeue jobs in background (only if there are jobs to requeue)
                if _pod_manager and affected_model and len(affected_job_ids) > 0:
                    logger.info(
                        f"Creating new pod for model {affected_model} and requeueing {len(affected_job_ids)} job(s)"
                    )
                    # Trigger requeue in background (fire and forget)
                    asyncio.create_task(
                        _requeue_jobs_after_pod_failure(affected_job_ids, affected_model)
                    )
                elif len(affected_job_ids) == 0:
                    logger.info(
                        f"No jobs to requeue for model {affected_model} after pod destruction "
                        f"(all jobs were RUNNING and marked as FAILED)"
                    )
                
                return False
            
            # No other jobs on pod - just revert this single job
            # (This shouldn't happen often, but handle it gracefully)
            logger.warning(
                f"Dispatch failed for job {job_id} but no other jobs found on pod {pod_uuid}. "
                f"Reverting job to QUEUED."
            )
            job.status = JobState.QUEUED.value
            job.worker_id = None
            job.updated_at = utc_naive_now()
            if job.error_log is None:
                job.error_log = []
            job.error_log.append({
                "timestamp": utc_naive_now().isoformat(),
                "state": "dispatch_failed",
                "error": str(last_error),
                "machine_id": str(machine_id) if machine_id else None,  # Track machine for retry limiting
                "pod_uuid": pod_uuid,
            })
            db.commit()
            
            # Mark machine bad even for single job failure
            if _pod_manager and machine_id:
                try:
                    await _pod_manager.mark_machine_bad(str(machine_id))
                except Exception:
                    pass
            
            # Release slot on pod
            if _pod_manager and pod_uuid:
                try:
                    await _pod_manager.release_slot_on_pod(pod_uuid, job_id)
                except Exception:
                    pass
            
            return False
        
        # Success! Mark job as RUNNING
        if _job_manager:
            _job_manager.mark_running(job_id, worker_id)
        else:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job:
                job.status = JobState.RUNNING.value
                job.started_at = utc_naive_now()
        
        # CRITICAL: Remove job from pending_jobs to prevent autoscaler from creating unnecessary pods
        if _pod_manager and job:
            try:
                await _pod_manager.remove_pending_job(job_id, job.model)
            except Exception as e:
                logger.debug(f"Failed to remove job {job_id} from pending queue: {e}")
                job.updated_at = utc_naive_now()
            
            # Update worker status
            worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
            if worker:
                worker.status = "processing" if worker.is_warm_pod else "busy"
                worker.last_active = utc_naive_now()
            
            db.commit()
        
        logger.info(f"Job {job_id} dispatched successfully to worker {worker_id}")
        return True
        
    except Exception as e:
        logger.exception(f"_dispatch_to_worker error for job {job_id}: {e}")
        return False
    finally:
        db.close()


@client_router.get("/job_status", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    auth: Dict = Depends(require_any_auth),
):
    """
    Get current job status.
    """
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        result_url = None
        if job.status == JobState.COMPLETED.value and job.result_r2_key:
            result_url = f"{_controller_url}/download/{job_id}" if _controller_url else f"/download/{job_id}"
        
        return JobStatusResponse(
            job_id=job.job_id,
            status=JobStatus(job.status),
            model=job.model,
            created_at=job.created_at,
            updated_at=job.updated_at,
            completed_at=job.completed_at,
            worker_id=job.worker_id,
            retry_count=job.retry_count or 0,
            result_url=result_url,
            error=job.error,
            error_log=job.error_log,
        )
    finally:
        db.close()


# =============================================================================
# Worker Endpoints
# =============================================================================

@worker_router.post("/register_worker")
async def register_worker(
    request: Request,
    payload: RegisterWorkerRequest,
    _: bool = Depends(require_worker_signature),
):
    """
    Worker registration endpoint.
    
    Called by worker pods when they start up and are ready.
    """
    ip = get_client_ip(request)
    
    # Get pod UUID (prefer pod_uuid over container_id)
    pod_uuid = payload.pod_uuid or payload.container_id
    
    if not pod_uuid:
        raise HTTPException(status_code=400, detail="pod_uuid or container_id required")
    
    # Validate pod belongs to orchestrator
    if _pod_manager:
        active_pods = _pod_manager.active_pods
        if pod_uuid not in active_pods:
            # Check if recently destroyed
            if _pod_manager.lifecycle.is_recently_destroyed(pod_uuid):
                logger.warning(f"Rejected registration from recently destroyed pod {pod_uuid}")
                raise HTTPException(status_code=409, detail="Pod recently destroyed")
            
            # RESILIENT FIX: Unknown pod attempting to register
            # This can happen due to:
            # 1. Race condition during pod creation
            # 2. Server restart while pods were running
            # 3. Zombie pod from previous server instance
            #
            # For server resilience (especially on weak VMs), we:
            # - Accept registration to avoid blocking worker
            # - Add pod to active_pods to track it
            # - Schedule zombie detection as non-blocking background task
            
            logger.warning(
                f"Unknown pod {pod_uuid} attempting registration from {ip}. "
                f"Accepting registration and adding to active pods for tracking."
            )
            
            # Add to active pods so we can track it
            # This prevents 500 errors from blocking operations on weak servers
            _pod_manager.active_pods[pod_uuid] = {
                "machine_id": "unknown",
                "model": payload.models[0] if payload.models else "unknown",
                "created_at": utc_naive_now().isoformat(),
                "endpoint": payload.endpoint,
                "recovered_from_unknown": True,
            }
            logger.info(f"Added unknown pod {pod_uuid} to active_pods for tracking")
    
    # Add IP to trusted workers
    ip_ban_manager.add_trusted_worker_ip(ip)
    
    # Get first model (workers typically register with one model)
    model = payload.models[0] if payload.models else "unknown"
    
    # Upsert worker with comprehensive error handling
    # CRITICAL: Use try/except to prevent 500 errors on weak servers
    db = SessionLocal()
    try:
        worker = db.query(Worker).filter(Worker.worker_id == payload.worker_id).first()
        now = utc_naive_now()
        
        is_warm = (
            pod_uuid == os.getenv("QUICKPOD_POD_UUID") 
            if pod_uuid and os.getenv("QUICKPOD_POD_UUID") 
            else False
        )
        
        # Check if worker's /run endpoint is actually ready (optional - proceed even if fails)
        # Use short timeouts to avoid blocking on weak servers
        worker_ready = True
        initial_status = "ready"  # Default to ready
        
        if payload.endpoint:
            try:
                # Wrap health check with timeout protection
                health_check = await asyncio.wait_for(
                    _check_worker_health(payload.endpoint, timeout=2),
                    timeout=3.0
                )
                if not health_check:
                    logger.warning(
                        f"Worker {payload.worker_id} registration: /health endpoint not responding, "
                        f"but proceeding with registration"
                    )
                    worker_ready = False  # Mark as starting instead of ready
            except asyncio.TimeoutError:
                logger.warning(f"Worker {payload.worker_id} health check timed out, marking as starting")
                worker_ready = False
            except Exception as e:
                logger.warning(f"Worker health check failed during registration: {e}")
                worker_ready = False
        
        # Use enhanced health check to get detailed worker status (with timeout protection)
        if payload.endpoint and not worker_ready:
            try:
                health_info = await asyncio.wait_for(
                    _check_worker_health_enhanced(payload.endpoint, timeout=2),
                    timeout=3.0
                )
                if health_info["healthy"]:
                    # Use status from worker's health endpoint
                    worker_status = health_info.get("status", "unknown")
                    if worker_status == "starting":
                        initial_status = "starting"
                    elif worker_status in ("ok", "busy"):
                        initial_status = "ready"
                    else:
                        initial_status = "starting"  # Conservative: wait for explicit "ok"
                    
                    logger.info(
                        f"Worker {payload.worker_id} health check: status={worker_status}, "
                        f"ready={health_info.get('ready')}, setting initial_status={initial_status}"
                    )
                else:
                    logger.warning(
                        f"Worker {payload.worker_id} health check failed during registration, "
                        f"but proceeding with status='starting'"
                    )
                    initial_status = "starting"
            except asyncio.TimeoutError:
                logger.warning(f"Worker {payload.worker_id} enhanced health check timed out")
                initial_status = "starting"
            except Exception as e:
                logger.warning(f"Worker {payload.worker_id} enhanced health check error: {e}")
                initial_status = "starting"
        
        if not worker:
            worker = Worker(
                worker_id=payload.worker_id,
                model=model,
                endpoint=payload.endpoint,
                status=initial_status,
                container_id=pod_uuid,
                last_heartbeat=now,
                last_active=now,
                created_at=now,
                info=payload.info,
                is_warm_pod=is_warm,
            )
            db.add(worker)
        else:
            worker.endpoint = payload.endpoint
            worker.model = model
            worker.container_id=pod_uuid
            worker.status = initial_status
            worker.last_heartbeat = now
            worker.last_active = now
            worker.info = payload.info
            if pod_uuid:
                worker.is_warm_pod = is_warm
        
        # If initially starting, schedule a task to POLL for readiness
        # (models can take 60-120s to load, so single check after 8s is insufficient)
        if initial_status == "starting" and payload.endpoint:
            logger.info(
                f"Worker {payload.worker_id} marked as 'starting', will poll for readiness every 10s"
            )
            # Store endpoint for background task
            endpoint_to_check = payload.endpoint
            worker_id_to_check = payload.worker_id
            
            async def _poll_for_readiness():
                """
                Poll worker health every 10s until it becomes ready.
                
                CRITICAL: Models can take 60-120 seconds to load.
                We must keep polling until worker reports "ok" status.
                """
                max_wait_time = 180  # Max 3 minutes to become ready
                poll_interval = 10  # Check every 10 seconds
                start_time = asyncio.get_event_loop().time()
                
                while asyncio.get_event_loop().time() - start_time < max_wait_time:
                    await asyncio.sleep(poll_interval)
                    
                    # Check current worker status in DB
                    db = SessionLocal()
                    try:
                        w = db.query(Worker).filter(Worker.worker_id == worker_id_to_check).first()
                        
                        if not w:
                            logger.warning(f"Worker {worker_id_to_check} no longer exists, stopping readiness poll")
                            return
                        
                        if w.status != "starting":
                            # Worker already transitioned to another state
                            logger.debug(f"Worker {worker_id_to_check} status is '{w.status}', stopping readiness poll")
                            return
                        
                        # Check worker health
                        health_info = await _check_worker_health_enhanced(endpoint_to_check, timeout=5)
                        
                        if health_info.get("ready"):
                            w.status = "ready"
                            db.commit()
                            elapsed = asyncio.get_event_loop().time() - start_time
                            logger.info(
                                f"Worker {worker_id_to_check} is now READY after {elapsed:.0f}s "
                                f"(health status={health_info.get('status')})"
                            )
                            return
                        
                        # Still not ready - log status and continue polling
                        elapsed = asyncio.get_event_loop().time() - start_time
                        logger.debug(
                            f"Worker {worker_id_to_check} still starting after {elapsed:.0f}s "
                            f"(health status={health_info.get('status')})"
                        )
                        
                    finally:
                        db.close()
                
                # Timed out - worker never became ready
                logger.warning(
                    f"Worker {worker_id_to_check} never became ready after {max_wait_time}s. "
                    f"Worker may have failed to load models. Keeping in 'starting' state."
                )
            
            asyncio.create_task(_poll_for_readiness())
        
        # Update pod manager tracking
        if _pod_manager and pod_uuid in _pod_manager.individual_pods:
            from pod_manager import PodStatus
            _pod_manager.individual_pods[pod_uuid].status = PodStatus.READY
            _pod_manager.individual_pods[pod_uuid].last_active = now
        
        # Audit log is optional - don't fail registration if it fails
        try:
            create_audit_log(
                db,
                event_type="worker_registered",
                message=f"Worker {payload.worker_id} registered",
                target=payload.worker_id,
                target_type="worker",
                actor_ip=ip,
                details={"model": model, "pod_uuid": pod_uuid},
            )
        except Exception as audit_error:
            logger.warning(f"Failed to create audit log for worker registration: {audit_error}")
        
        db.commit()
        
    except Exception as db_error:
        # CRITICAL: Catch database errors to prevent 500s
        # Still return success if we got this far - worker is valid
        logger.error(f"Database error during worker registration: {db_error}")
        try:
            db.rollback()
        except Exception:
            pass
        # Return success anyway - worker can still operate
        # This prevents weak servers from rejecting valid workers
        return {"status": "ok", "worker_id": payload.worker_id, "warning": "partial_registration"}
    finally:
        db.close()
    
    # Audit logger is optional - don't fail if it errors
    try:
        audit_logger.worker_registered(payload.worker_id, pod_uuid, ip, model)
    except Exception as e:
        logger.warning(f"Audit logger error (non-critical): {e}")
    
    logger.info(f"Worker {payload.worker_id} registered for model {model} (pod: {pod_uuid})")
    
    return {"status": "ok", "worker_id": payload.worker_id}


# In-memory heartbeat cache for fast responses (synced to DB periodically)
# CRITICAL: This avoids DB queries on every heartbeat, making the endpoint fast
_heartbeat_cache: Dict[str, Dict[str, Any]] = {}
_heartbeat_cache_lock = asyncio.Lock()


@worker_router.post("/heartbeat")
async def worker_heartbeat(
    request: Request,
    payload: HeartbeatRequest,
    _: bool = Depends(require_worker_signature),
):
    """
    Worker heartbeat endpoint.
    
    Updates in-memory cache IMMEDIATELY (lock-free write for speed).
    CRITICAL: Must be ultra-fast - workers depend on this.
    
    Lock-free dict write is safe in asyncio (single-threaded execution).
    Lock only needed for sync task batch reads.
    """
    now = utc_naive_now()
    
    # Lock-free cache update - dict assignment is atomic in Python
    # This is safe in asyncio since Python code runs single-threaded
    # Lock is only needed when sync task reads multiple entries
    try:
        _heartbeat_cache[payload.worker_id] = {
            "last_heartbeat": now,
            "status": payload.status.value if payload.status else None,
            "metrics": payload.metrics,
            "needs_sync": True,  # Mark for DB sync
        }
    except Exception:
        # If cache update fails for any reason, just continue
        # Sync task will eventually catch up via DB reads
        pass
    
    # Return immediately - zero blocking operations
    return {"status": "ok"}


async def sync_heartbeat_cache_to_db():
    """
    Background task: Sync heartbeat cache to database every 30 seconds.
    This keeps the DB updated without blocking heartbeat responses.
    """
    while True:
        try:
            await asyncio.sleep(30)  # Sync every 30 seconds
            
            # Get workers that need sync
            async with _heartbeat_cache_lock:
                to_sync = {
                    worker_id: data.copy() 
                    for worker_id, data in _heartbeat_cache.items() 
                    if data.get("needs_sync")
                }
                # Mark as synced
                for worker_id in to_sync:
                    _heartbeat_cache[worker_id]["needs_sync"] = False
            
            if not to_sync:
                continue
            
            # Batch update to DB
            db = SessionLocal()
            try:
                for worker_id, data in to_sync.items():
                    worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
                    if worker:
                        worker.last_heartbeat = data["last_heartbeat"]
                        worker.last_active = data["last_heartbeat"]
                        if data["status"]:
                            worker.status = data["status"]
                        if data["metrics"]:
                            worker.resource_metrics = data["metrics"]
                
                db.commit()
                logger.debug(f"Synced {len(to_sync)} worker heartbeats to DB")
            except Exception as e:
                logger.warning(f"Failed to sync heartbeats to DB: {e}")
                try:
                    db.rollback()
                except Exception:
                    pass
            finally:
                db.close()
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(f"Heartbeat sync task error: {e}")


@worker_router.post("/job_complete")
async def job_complete(
    request: Request,
    payload: JobCompleteRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(require_worker_signature),
):
    """
    Job completion notification from worker.
    
    Handles both success and failure cases.
    """
    job_id = payload.job_id
    
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Check if job is already in a terminal state
        # This can happen if the pod was destroyed while the job was running
        current_state = JobState(job.status)
        if current_state in [JobState.COMPLETED, JobState.FAILED, JobState.TIMEOUT, JobState.CANCELLED]:
            logger.warning(
                f"Worker attempted to complete job {job_id} but it's already in {current_state.value} state. "
                f"This likely means the pod was destroyed while the job was running. "
                f"Ignoring completion request (idempotent)."
            )
            # Return success to avoid worker retries
            return {"status": "ok", "message": f"Job already in {current_state.value} state"}
        
        # If job is in QUEUED state, it means it was requeued after pod destruction
        # The worker is trying to complete an old job that's been requeued
        if current_state == JobState.QUEUED:
            logger.warning(
                f"Worker attempted to complete job {job_id} but it's in QUEUED state. "
                f"This likely means the job was requeued after pod destruction. "
                f"Ignoring completion request."
            )
            return {"status": "ok", "message": "Job was requeued, ignoring completion"}
        
        worker_id = job.worker_id
        pod_uuid = None
        
        if worker_id:
            worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
            if worker:
                pod_uuid = worker.container_id
        
        if payload.failed:
            error_str = (payload.error or "Unknown error").lower()
            
            # Check if this is a retryable infrastructure failure
            is_cuda_incompatibility = "no kernel image is available" in error_str and "cuda" in error_str
            is_oom_error = "oom" in error_str or "out of memory" in error_str
            is_other_cuda_error = "cuda" in error_str and "error" in error_str
            
            # Infrastructure failures that should trigger retry
            is_infrastructure_failure = is_cuda_incompatibility or is_oom_error
            
            # Get retry count from job
            retry_count = job.retry_count if hasattr(job, 'retry_count') and job.retry_count else 0
            max_retries = 3  # Allow up to 3 retries for infrastructure failures
            
            # Get variant and machine info
            variant = None
            machine_id = None
            pod_uuid = None
            if worker_id:
                worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
                if worker and worker.container_id and _pod_manager:
                    pod_uuid = worker.container_id
                    async with _pod_manager._lock:
                        pod_info = _pod_manager.individual_pods.get(worker.container_id)
                        if pod_info:
                            variant = pod_info.variant
                            machine_id = pod_info.machine_id
            
            # CRITICAL: CUDA kernel incompatibility (int4 variant on wrong GPU)
            # Move heavy operations to background to avoid blocking response
            if is_cuda_incompatibility:
                logger.error(
                    f"CUDA kernel incompatibility detected for job {job_id}! "
                    f"This likely means int4 variant is incompatible with the GPU. "
                    f"Error: {payload.error}"
                )
                
                if variant == "int4" and machine_id:
                    logger.warning(
                        f"Int4 variant failed on machine {machine_id} due to CUDA incompatibility. "
                        f"Blacklisting machine {machine_id} for int4 variant (10 min)."
                    )
                    # Move to background task to avoid blocking response
                    if _pod_manager:
                        background_tasks.add_task(
                            _pod_manager.mark_machine_bad,
                            str(machine_id),
                            600  # duration
                        )
                    
                    # Destroy the pod in background - it won't work for any int4 jobs
                    if pod_uuid and _pod_manager:
                        background_tasks.add_task(
                            _handle_incompatible_pod_destruction,
                            _pod_manager,
                            pod_uuid,
                            "CUDA kernel incompatibility (int4 variant)"
                        )
            
            # Check for OOM errors - remove pod from tracking immediately, then destroy
            elif is_oom_error:
                logger.warning(f"OOM detected for job {job_id} on pod {pod_uuid} (machine {machine_id})")
                # CRITICAL: Remove pod from tracking IMMEDIATELY to prevent reuse
                # Also mark worker as offline in DB immediately
                if pod_uuid and _pod_manager:
                    # Remove from tracking immediately (synchronously)
                    async with _pod_manager._lock:
                        if pod_uuid in _pod_manager.individual_pods:
                            logger.warning(f"Removing pod {pod_uuid} from tracking immediately due to OOM (will be destroyed in background)")
                            _pod_manager.individual_pods.pop(pod_uuid, None)
                            _pod_manager.individual_pod_managers.pop(pod_uuid, None)
                    
                    # Mark worker as offline in DB immediately (synchronously)
                    if worker_id:
                        try:
                            worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
                            if worker:
                                worker.status = "offline"
                                db.commit()
                                logger.info(f"Marked worker {worker_id} as offline due to OOM")
                        except Exception as e:
                            logger.warning(f"Failed to mark worker offline: {e}")
                    
                    # Schedule destruction in background (cleanup)
                    background_tasks.add_task(
                        _pod_manager.destroy_individual_pod,
                        pod_uuid,
                        f"OOM detected on job {job_id}"
                    )
                # Also blacklist the machine (less critical - pod removal is the main fix)
                if machine_id and _pod_manager:
                    background_tasks.add_task(
                        _pod_manager.mark_machine_bad,
                        str(machine_id),
                        300  # 5 min for OOM
                    )
            
            # Check for other critical errors
            elif is_other_cuda_error:
                logger.warning(f"CUDA error detected for job {job_id}: {payload.error}")
            
            # CRITICAL: Retry infrastructure failures instead of marking as permanently failed
            if is_infrastructure_failure and retry_count < max_retries:
                logger.warning(
                    f"Job {job_id} failed due to infrastructure issue (retry {retry_count + 1}/{max_retries}). "
                    f"Requeuing for retry on different machine/variant. Error: {payload.error}"
                )
                
                # Increment retry count and requeue
                job.retry_count = retry_count + 1
                job.status = JobState.QUEUED.value
                job.worker_id = None
                job.error = f"Retry {job.retry_count}: {payload.error}"
                job.updated_at = utc_naive_now()
                
                # Add to error log
                error_log = job.error_log or []
                error_log.append({
                    "timestamp": utc_naive_now().isoformat(),
                    "worker_id": worker_id,
                    "error": payload.error,
                    "retry_count": job.retry_count,
                    "machine_id": machine_id,
                    "variant": variant,
                })
                job.error_log = error_log
                db.commit()
                
                logger.info(
                    f"Job {job_id} requeued for retry (attempt {job.retry_count}/{max_retries}). "
                    f"Will try different machine/variant."
                )
                
                # Trigger dispatch for the requeued job
                if _pod_manager:
                    job_mode = job.mode if hasattr(job, 'mode') else None
                    asyncio.create_task(_dispatch_job_background(job_id, job.model, job_mode))
                
                return {"status": "requeued", "retry_count": job.retry_count}
            
            else:
                # Either not a retryable error or retries exhausted
                if is_infrastructure_failure:
                    logger.error(
                        f"Job {job_id} failed after {retry_count} retries. "
                        f"Marking as permanently failed. Error: {payload.error}"
                    )
                
                # Mark job as permanently failed
                if _job_manager:
                    _job_manager.mark_failed(job_id, payload.error or "Unknown error", worker_id)
                else:
                    job.status = JobState.FAILED.value
                    job.error = payload.error
                    job.updated_at = utc_naive_now()
                    db.commit()
                
                # CRITICAL: Remove failed job from pending_jobs (in background)
                if _pod_manager and job.model:
                    background_tasks.add_task(
                        _remove_pending_job_safe,
                        _pod_manager,
                        job_id,
                        job.model,
                        "failed"
                    )
                
                audit_logger.job_failed(job_id, worker_id, payload.error)
            
        else:
            # Validate R2 file exists
            if _s3_client and _r2_bucket and payload.result_r2_key:
                try:
                    _s3_client.head_object(Bucket=_r2_bucket, Key=payload.result_r2_key)
                except ClientError:
                    logger.error(f"Result file not found: {payload.result_r2_key}")
                    raise HTTPException(status_code=400, detail="Result file not found in storage")
            
            # Mark job completed
            if _job_manager:
                _job_manager.mark_completed(
                    job_id,
                    result_r2_key=payload.result_r2_key,
                    output_url=f"{_controller_url}/download/{job_id}" if _controller_url else None,
                )
            else:
                completion_time = utc_naive_now()
                job.status = JobState.COMPLETED.value
                job.result_r2_key = payload.result_r2_key
                job.completed_at = completion_time
                job.updated_at = completion_time
                
                # Calculate timing metrics
                timing_metrics = _calculate_timing_metrics(job, completion_time)
                job.total_time_seconds = timing_metrics["total_time"]
                job.processing_time_seconds = timing_metrics["processing_time"]
                job.queue_time_seconds = timing_metrics["queue_time"]
                job.dispatch_time_seconds = timing_metrics["dispatch_time"]
                
                db.commit()
                
                # Log timing metrics
                _log_timing_metrics(job_id, job.model, timing_metrics)
            
            # Update worker's last_job_finished_at timestamp
            if worker_id:
                worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
                if worker:
                    worker.last_job_finished_at = utc_naive_now()
                    worker.status = "ready"  # Reset to ready after job completion
                    db.commit()
            
            # Update pod manager
            if _pod_manager and pod_uuid:
                background_tasks.add_task(
                    _pod_manager.mark_job_completed_on_pod,
                    pod_uuid,
                    job_id,
                )
            
            # CRITICAL: Remove job from pending_jobs (in background, may already be removed when it started running)
            if _pod_manager and job.model:
                background_tasks.add_task(
                    _remove_pending_job_safe,
                    _pod_manager,
                    job_id,
                    job.model,
                    "completed"
                )
            
            audit_logger.job_completed(job_id, worker_id)
        
        # Send FCM notification
        fcm_token = job.fcm_token
        if fcm_token and not job.notification_sent:
            background_tasks.add_task(
                _send_job_notification,
                fcm_token,
                job_id,
                job.model,
                not payload.failed,
            )
            job.notification_sent = True
            db.commit()
    
    except HTTPException:
        raise
    except Exception as e:
        # CRITICAL: Don't fail job_complete on database errors
        # Return success anyway - we don't want worker to keep retrying
        logger.error(f"Database error during job_complete for {job_id}: {e}")
        try:
            db.rollback()
        except Exception:
            pass
        # Return success to prevent worker from retrying forever
        return {"status": "ok", "job_id": job_id, "warning": "db_error"}
    finally:
        db.close()
    
    return {"status": "ok", "job_id": job_id}


async def _handle_incompatible_pod_destruction(
    pod_manager,
    pod_uuid: str,
    reason: str
):
    """Background task to destroy incompatible pod"""
    try:
        await pod_manager.destroy_individual_pod(pod_uuid, reason=reason)
        logger.info(f"Destroyed incompatible pod {pod_uuid}: {reason}")
    except Exception as e:
        logger.error(f"Failed to destroy incompatible pod {pod_uuid}: {e}")


async def _remove_pending_job_safe(
    pod_manager,
    job_id: str,
    model: str,
    status: str
):
    """Background task to safely remove job from pending queue"""
    try:
        await pod_manager.remove_pending_job(job_id, model)
    except Exception as e:
        logger.debug(f"Failed to remove {status} job {job_id} from pending queue: {e}")


async def _send_job_notification(token: str, job_id: str, model: str, success: bool):
    """Background task to send FCM notification"""
    if success:
        await send_fcm_notification(
            token,
            "Job Complete",
            f"Your {model} job has finished!",
            {"job_id": job_id, "status": "completed"},
        )
    else:
        await send_fcm_notification(
            token,
            "Job Failed",
            f"Your {model} job has failed",
            {"job_id": job_id, "status": "failed"},
        )


async def _requeue_jobs_after_pod_failure(job_ids: list, model: str):
    """
    Background task to create a new pod and requeue jobs after pod failure.
    
    This is called when a pod fails to accept job dispatches and multiple jobs
    are affected. Creates a new pod and dispatches all affected jobs to it.
    """
    if not _job_manager or not _pod_manager:
        logger.error("Cannot requeue jobs: job_manager or pod_manager not available")
        return
    
    logger.info(f"Requeueing {len(job_ids)} job(s) for model {model} after pod failure")
    
    try:
        # Wait for the old pod to be fully destroyed and removed from tracking
        # Check that pod is no longer in pod_manager's tracking
        max_wait = 10  # Wait up to 10 seconds
        waited = 0
        while waited < max_wait:
            async with _pod_manager._lock:
                # Check if there are any pods for this model that are being destroyed
                # (We want to make sure the old pod is gone before creating a new one)
                active_pods_for_model = [
                    p for p in _pod_manager.individual_pods.values()
                    if p.model == model
                ]
            
            if len(active_pods_for_model) == 0:
                # No active pods for this model, safe to create new one
                break
            
            await asyncio.sleep(1)
            waited += 1
        
        if waited >= max_wait:
            logger.warning(
                f"Waited {max_wait}s for old pods to be destroyed, proceeding anyway"
            )
        
        # Create a new pod for this model
        # Get mode from first job if available
        mode = None
        db = SessionLocal()
        try:
            first_job = db.query(Job).filter(Job.job_id == job_ids[0]).first()
            if first_job and first_job.params:
                mode = first_job.params.get("mode")
        finally:
            db.close()
        
        # Create new pod
        try:
            new_pod_uuid = await _pod_manager.create_machine_based_pod(model, mode)
        except Exception as e:
            logger.error(
                f"Failed to create new pod for model {model}: {e}. "
                f"Jobs will remain in QUEUED state and may be picked up by recovery system or autoscaler."
            )
            # Mark jobs with error so they can be recovered
            db = SessionLocal()
            try:
                for job_id in job_ids:
                    job = db.query(Job).filter(Job.job_id == job_id).first()
                    if job and job.status == JobState.QUEUED.value:
                        if job.error_log is None:
                            job.error_log = []
                        job.error_log.append({
                            "timestamp": utc_naive_now().isoformat(),
                            "state": "requeue_failed",
                            "error": f"Failed to create new pod: {str(e)}",
                        })
                        job.updated_at = utc_naive_now()
                db.commit()
            finally:
                db.close()
            return
        
        if not new_pod_uuid:
            logger.error(f"Failed to create new pod for model {model}, jobs will remain in QUEUED state")
            return
        
        logger.info(f"Created new pod {new_pod_uuid} for model {model}, now dispatching {len(job_ids)} job(s)")
        
        # Wait for worker registration (similar to _wait_for_worker_registration)
        new_worker_id = None
        start_time = time.time()
        timeout = 120  # 2 minutes timeout
        
        while time.time() - start_time < timeout:
            db = SessionLocal()
            try:
                worker = db.query(Worker).filter(
                    Worker.container_id == new_pod_uuid,
                    Worker.model == model,
                    Worker.status == "ready",
                ).first()
                
                if worker:
                    new_worker_id = worker.worker_id
                    logger.info(f"New worker {new_worker_id} registered for pod {new_pod_uuid}")
                    break
            finally:
                db.close()
            
            await asyncio.sleep(1)
        
        if not new_worker_id:
            logger.error(f"Timeout waiting for worker registration on new pod {new_pod_uuid}")
            return
        
        # Dispatch all jobs to the new pod
        # Only requeue jobs that are in QUEUED or DISPATCHING states
        success_count = 0
        skipped_count = 0
        for job_id in job_ids:
            try:
                # Check job state before requeueing
                db = SessionLocal()
                try:
                    job = db.query(Job).filter(Job.job_id == job_id).first()
                    if not job:
                        logger.warning(f"Job {job_id} not found during requeue")
                        skipped_count += 1
                        continue
                    
                    current_state = JobState(job.status)
                    
                    # Skip jobs that are in terminal states or RUNNING
                    if current_state in [JobState.COMPLETED, JobState.FAILED, JobState.TIMEOUT, 
                                         JobState.CANCELLED, JobState.RUNNING]:
                        logger.info(
                            f"Skipping requeue for job {job_id}: already in {current_state.value} state"
                        )
                        skipped_count += 1
                        continue
                    
                    # Only requeue QUEUED or DISPATCHING jobs
                    if current_state not in [JobState.QUEUED, JobState.DISPATCHING]:
                        logger.warning(
                            f"Job {job_id} is in {current_state.value} state, expected QUEUED or DISPATCHING. "
                            f"Skipping requeue."
                        )
                        skipped_count += 1
                        continue
                    
                    # Update worker assignment
                    job.worker_id = new_worker_id
                    db.commit()
                finally:
                    db.close()
                
                # Dispatch the job
                success = await _dispatch_to_worker(job_id, new_worker_id)
                if success:
                    success_count += 1
                    logger.info(f"Successfully requeued job {job_id} to new pod {new_pod_uuid}")
                else:
                    logger.warning(f"Failed to dispatch requeued job {job_id} to new pod {new_pod_uuid}")
                
                # Small delay between dispatches
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error requeueing job {job_id}: {e}")
                skipped_count += 1
        
        logger.info(
            f"Requeue complete: {success_count}/{len(job_ids)} jobs successfully dispatched "
            f"to new pod {new_pod_uuid} ({skipped_count} skipped due to invalid state)"
        )
        
    except Exception as e:
        logger.exception(f"Error in _requeue_jobs_after_pod_failure: {e}")


# =============================================================================
# Admin Endpoints (Optional)
# =============================================================================

@admin_router.get("/workers")
async def list_workers(
    status: Optional[str] = None,
    model: Optional[str] = None,
    auth: Dict = Depends(require_any_auth),
):
    """List all workers with optional filters"""
    db = SessionLocal()
    try:
        query = db.query(Worker)
        
        if status:
            query = query.filter(Worker.status == status)
        if model:
            query = query.filter(Worker.model == model)
        
        workers = query.all()
        
        return {
            "workers": [w.to_dict() for w in workers],
            "total": len(workers),
            "active": sum(1 for w in workers if w.status in ("ready", "busy", "processing")),
            "idle": sum(1 for w in workers if w.status == "idle"),
        }
    finally:
        db.close()


@admin_router.get("/jobs")
async def list_jobs(
    status: Optional[str] = None,
    model: Optional[str] = None,
    limit: int = 50,
    auth: Dict = Depends(require_any_auth),
):
    """List jobs with optional filters"""
    db = SessionLocal()
    try:
        query = db.query(Job)
        
        if status:
            query = query.filter(Job.status == status)
        if model:
            query = query.filter(Job.model == model)
        
        jobs = query.order_by(Job.created_at.desc()).limit(limit).all()
        
        return {
            "jobs": [j.to_dict() for j in jobs],
            "total": len(jobs),
        }
    finally:
        db.close()


@admin_router.post("/ban_ip")
async def ban_ip_endpoint(
    ip: str,
    reason: str = "Manual ban",
    duration: Optional[int] = None,
    permanent: bool = False,
    auth: Dict = Depends(require_any_auth),
):
    """Ban an IP address"""
    await ip_ban_manager.ban_ip(ip, reason, duration, permanent)
    return {"status": "ok", "ip": ip, "permanent": permanent}


@admin_router.post("/unban_ip")
async def unban_ip_endpoint(
    ip: str,
    auth: Dict = Depends(require_any_auth),
):
    """Unban an IP address"""
    result = await ip_ban_manager.unban_ip(ip)
    return {"status": "ok" if result else "not_found", "ip": ip}

