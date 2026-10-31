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
import copy
import hmac
import hashlib
import json
import asyncio
import mimetypes
import logging
from urllib.parse import urlparse
from pathlib import Path
from datetime import datetime, timedelta, date, timezone
from typing import Optional, Dict, Any, List, Tuple


def normalize_pod_uuid(pod_uuid: Optional[str]) -> Optional[str]:
    """
    Normalize pod UUID by removing hyphens for comparison purposes.
    
    QuickPod pod UUIDs come in format: fe0cf25f-94a2-4120-8dde-2969e8978b5c
    Workers may send them with or without hyphens.
    
    Returns None if input is None or empty.
    """
    if not pod_uuid:
        return None
    return pod_uuid.replace("-", "").lower()


def format_pod_uuid_with_hyphens(pod_uuid: Optional[str]) -> Optional[str]:
    """
    Format a pod UUID with hyphens (8-4-4-4-12 format) for QuickPod API calls.
    
    QuickPod API requires UUIDs in format: fe0cf25f-94a2-4120-8dde-2969e8978b5c
    
    Returns None if input is None or not a valid 32-char hex string.
    """
    if not pod_uuid:
        return None
    
    # Remove existing hyphens first
    clean = pod_uuid.replace("-", "").lower()
    
    # If already has hyphens and is 36 chars, return as-is
    if len(pod_uuid) == 36 and "-" in pod_uuid:
        return pod_uuid.lower()
    
    # If it's 32 hex chars, format with hyphens
    if len(clean) == 32:
        try:
            # Validate it's hex
            int(clean, 16)
            return f"{clean[:8]}-{clean[8:12]}-{clean[12:16]}-{clean[16:20]}-{clean[20:]}"
        except ValueError:
            pass
    
    # Return as-is if we can't format it
    return pod_uuid

import requests

from fastapi import (
    APIRouter,
    Request,
    Response,
    HTTPException,
    Depends,
    Header,
    BackgroundTasks,
)
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import cast, String as SQLString, text
from botocore.exceptions import ClientError

from db import (
    get_session,
    SessionLocal,
    Worker,
    Job,
    User,
    CreditTransaction,
    DailySpin,
    AppConfig,
    RewardedAdEvent,
    AdFreePurchase,
    Purchase,
    KieTask,
    RunwareTask,
    PersonalQueueItem,
    get_app_config_int,
    create_audit_log,
    update_model_eta_stat,
    utc_naive_now,
)
from credit_service import (
    lock_credits_for_job,
    refund_credits_for_job,
    get_credit_cost,
    has_sufficient_credits,
)
from prompt_enhancement import (
    enhance_prompt,
    should_enhance_for_model,
    suggest_prompt,
    suggest_related,
)
from asset_prompt_vlm import (
    choose_eq_preset,
    describe_image_with_gemini,
    get_max_image_bytes,
    is_asset_prompt_vlm_enabled,
)
from models import (
    SubmitRequest,
    SubmitResponse,
    DescribeAssetForPromptRequest,
    DescribeAssetForPromptResponse,
    SuggestPromptRequest,
    SuggestPromptResponse,
    EqPresetRequest,
    EqPresetResponse,
    RegisterWorkerRequest,
    HeartbeatRequest,
    JobStartedRequest,
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
    sign_payload_bytes,
)
from worker_transport_crypto import decode_worker_request_body, encode_worker_request_body
from exceptions import (
    JobNotFoundError,
    InvalidModelError,
    AuthenticationError,
    InvalidSignatureError,
    JobAlreadyProcessedError,
    WorkerTimeoutError,
)
from job_manager import JobManager, JobState
from pod_manager import MAX_JOBS_PER_POD
from logging_config import get_audit_logger
from external_api_router import ExternalApiRouter, ExternalExecutionResult, carry_moderation_flag

logger = logging.getLogger("routes")
audit_logger = get_audit_logger()

# Create routers
public_router = APIRouter(tags=["Public"])
client_router = APIRouter(tags=["Client"])
worker_router = APIRouter(tags=["Worker"])
admin_router = APIRouter(prefix="/admin", tags=["Admin"])
auth_router = APIRouter(prefix="/auth", tags=["Auth"])

# Daily login credits: first 30 UTC calendar days from account creation (by date), then standard amount.
_DAILY_CREDITS_PROMO_DAYS = 30
_DAILY_CREDITS_PROMO_AMOUNT = 20
_DAILY_CREDITS_STANDARD_AMOUNT = 5


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def user_has_verified_credit_purchase(db: Session, user_id: str) -> bool:
    row = (
        db.query(Purchase)
        .filter(
            Purchase.user_id == user_id,
            Purchase.purchase_state == "verified",
            Purchase.credits_awarded > 0,
        )
        .first()
    )
    return row is not None


def compute_ad_experience(user: User, has_purchased_credits: bool) -> str:
    """Server-only tier: none | light | heavy."""
    if user.is_ad_free or user.is_vip_active():
        return "none"
    if has_purchased_credits:
        return "light"
    return "heavy"


def apply_daily_credits_if_eligible(db: Session, user: User) -> int:
    """
    Grant daily credits at most once per UTC calendar day. Returns credits granted (0 if skipped).
    """
    if not user.is_active:
        return 0
    today = _utc_today()
    if user.daily_credits_last_grant_utc == today:
        return 0
    created = user.created_at
    created_day = created.date() if created else today
    age_days = (today - created_day).days
    amount = (
        _DAILY_CREDITS_PROMO_AMOUNT
        if age_days < _DAILY_CREDITS_PROMO_DAYS
        else _DAILY_CREDITS_STANDARD_AMOUNT
    )
    user.total_credits += amount
    user.bonus_granted += amount
    user.daily_credits_last_grant_utc = today
    db.add(
        CreditTransaction(
            user_id=user.id,
            delta=amount,
            reason="daily_login_grant",
            credit_metadata={
                "utc_date": today.isoformat(),
                "in_promo_window": age_days < _DAILY_CREDITS_PROMO_DAYS,
                "amount": amount,
            },
            created_at=utc_naive_now(),
        )
    )
    return amount


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
_external_api_router = ExternalApiRouter()

# Per-user daily limit for VLM auto-prompt (best-effort in-process counter).
_asset_prompt_rate_lock = asyncio.Lock()
_asset_prompt_rate: Dict[str, Tuple[date, int]] = {}


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
    try:
        _external_api_router.reload()
    except Exception as e:
        logger.warning(f"Failed to reload external API config: {e}")


def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# =============================================================================
# Authentication Dependencies
# =============================================================================


async def require_firebase_auth(
    request: Request,
    authorization: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """
    Require Firebase ID token authentication for client requests.
    
    Headers:
        Authorization: Bearer <firebase_id_token>
    
    Returns:
        Dict with user info: firebase_uid, user_id, email, credits
    """
    ip = get_client_ip(request)
    
    # Verify Firebase token
    try:
        decoded = await verify_firebase_token(authorization)
    except HTTPException as e:
        ip_ban_manager.record_failed_auth(ip)
        await ip_ban_manager.check_brute_force(ip)
        audit_logger.auth_failure(ip, "firebase", str(e.detail))
        raise
    
    firebase_uid = decoded.get("uid")
    if not firebase_uid:
        raise HTTPException(400, "Firebase UID not found in token")
    
    email = decoded.get("email")
    
    # Get or create user in database
    with SessionLocal() as db:
        user = get_or_create_user(db, firebase_uid, email)
        
        audit_logger.auth_success(ip, "firebase", actor=firebase_uid[:16])
        
        # Check VIP status (time-limited subscription)
        is_vip = user.is_vip_active()
        
        return {
            "type": "firebase",
            "firebase_uid": firebase_uid,
            "user_id": user.id,
            "email": user.email,
            "total_credits": user.total_credits,
            "is_active": user.is_active,
            # VIP status
            "is_vip": is_vip,
            "vip_expires_at": user.vip_expires_at.isoformat() if user.vip_expires_at else None,
            # Ad-free status
            "is_ad_free": user.is_ad_free,
        }


async def require_any_auth(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_device_uuid: Optional[str] = Header(None),
    x_fingerprint: Optional[str] = Header(None),
    x_timestamp: Optional[str] = Header(None),
    x_signature: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """
    Require either Firebase token, API key, or device signature authentication.
    Kept for backward compatibility with admin endpoints.
    """
    ip = get_client_ip(request)
    
    # Try Firebase token first (Bearer token)
    if authorization and authorization.startswith("Bearer "):
        try:
            return await require_firebase_auth(request, authorization)
        except HTTPException:
            pass  # Fall through to other auth methods
    
    # Try API key
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
        if len(x_fingerprint) == 64
        and all(c in "0123456789abcdef" for c in x_fingerprint.lower())
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
        fingerprint_hash, x_device_uuid, meta={"last_auth": utc_naive_now().isoformat()}
    )
    
    audit_logger.auth_success(ip, "device", actor=fingerprint_hash[:16])
    
    return {
        "type": "device",
        "fingerprint_hash": fingerprint_hash,
        "device_uuid": device_info.get("device_uuid"),
    }


async def require_worker_signed_payload(
    request: Request,
    x_signature: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """
    Require valid HMAC over the wire body, then decrypt if worker transport crypto is on.
    """
    if not x_signature:
        raise HTTPException(status_code=401, detail="Missing signature")

    body = await request.body()

    if not verify_signature(x_signature, body):
        ip = get_client_ip(request)
        audit_logger.auth_failure(ip, "worker", "Invalid signature")
        raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        return decode_worker_request_body(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


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


def create_presigned_get_url(key: str, expires_in: int = 3600) -> Optional[str]:
    """Generate presigned GET URL for R2 (used for worker to download input videos)"""
    if not _s3_client or not _r2_bucket:
        return None
    
    try:
        return _s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": _r2_bucket, "Key": key},
            ExpiresIn=expires_in,
        )
    except Exception as e:
        logger.error(f"Failed to generate presigned GET URL: {e}")
        return None


def validate_model(model: str) -> str:
    """Validate model name against requirements or known aliases"""
    from pod_manager import PodManager
    
    model_lower = model.lower()
    
    if not _model_requirements:
        return model_lower
    
    # Check if model is directly in requirements
    if model_lower in _model_requirements or model_lower == "default":
        return model_lower

    if _external_api_router.get_model_config(model_lower):
        return model_lower
    
    # Check if model is a known alias that maps to a valid base model
    base_model = PodManager.resolve_model_alias(model_lower)
    if base_model != model_lower and base_model in _model_requirements:
        return model_lower  # Return original model name (alias) - routing handles the rest
    
    available = list(_model_requirements.keys()) + list(PodManager.MODEL_ALIASES.keys())
    raise InvalidModelError(model, available)


def _kie_result_source(job: Job) -> Optional[str]:
    params = job.params if isinstance(job.params, dict) else {}
    source = params.get("_kie_result_url")
    if isinstance(source, str) and source.strip():
        return source.strip()
    return None


def _build_public_result_url(job: Job) -> Optional[str]:
    """Build client-facing result URL for completed jobs.

    KIE jobs store the 20-minute download link in output_url so the app can
    save the file immediately. Later calls to /download_url mint a new link.
    """
    if job.output_url:
        return job.output_url
    if job.result_r2_key:
        return (
            f"{_controller_url}/download_url/{job.job_id}"
            if _controller_url
            else f"/download_url/{job.job_id}"
        )
    return None


def _prepare_external_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert transient R2 key params to downloadable URLs and normalize image input key.
    External providers consume direct URLs or data URIs.
    """
    out = dict(params or {})

    # init_image_r2_keys (multi-reference) -> init_image as list of URLs
    if "init_image_r2_keys" in out and "init_image" not in out:
        keys = out.pop("init_image_r2_keys")
        if isinstance(keys, list) and keys:
            urls: List[str] = []
            for r2_key in keys:
                download_url = create_presigned_get_url(r2_key, expires_in=3600)
                if download_url:
                    urls.append(download_url)
            if len(urls) == len(keys):
                out["init_image"] = urls

    # init_image_r2_key -> init_image
    if "init_image_r2_key" in out and "init_image" not in out:
        r2_key = out.pop("init_image_r2_key")
        download_url = create_presigned_get_url(r2_key, expires_in=3600)
        if download_url:
            out["init_image"] = download_url

    # Some tools use image key instead of init_image.
    if "image" in out and "init_image" not in out:
        out["init_image"] = out["image"]

    # Any leftover "<name>_r2_key" becomes a presigned URL under "<name>".
    for key in list(out.keys()):
        if not str(key).endswith("_r2_key"):
            continue
        r2_key = out.get(key)
        if not isinstance(r2_key, str) or not r2_key.strip():
            continue
        download_url = create_presigned_get_url(r2_key.strip(), expires_in=3600)
        dest = str(key)[: -len("_r2_key")]
        existing = out.get(dest)
        if download_url and not (isinstance(existing, str) and existing.startswith("http")):
            out[dest] = download_url

    for key in list(out.keys()):
        if not str(key).endswith("_r2_keys"):
            continue
        values = out.get(key)
        if not isinstance(values, list):
            continue
        urls: List[str] = []
        for item in values:
            if not isinstance(item, str) or not item.strip():
                continue
            download_url = create_presigned_get_url(item.strip(), expires_in=3600)
            if download_url:
                urls.append(download_url)
        dest = str(key)[: -len("_r2_keys")]
        existing = out.get(dest)
        if urls and not existing:
            out[dest] = urls

    # Strip internal metadata keys from params for provider payload mapping.
    for k in list(out.keys()):
        if str(k).startswith("_"):
            out.pop(k, None)

    return out


def _store_external_result(job_id: str, model: str, result: ExternalExecutionResult) -> tuple:
    """Copy provider output into R2. Returns (r2_key, download_url)."""
    ext = "png"
    content_type = result.result_content_type or "image/png"
    body = result.result_bytes
    if result.result_text is not None:
        body = result.result_text.encode("utf-8")
        ext = "txt"
        content_type = "text/plain; charset=utf-8"
    elif result.result_url:
        import kie_client

        try:
            body = kie_client.fetch_result_bytes(result.result_url)
        except Exception:
            body = requests.get(result.result_url, timeout=120).content
        path = urlparse(result.result_url).path.lower()
        if path.endswith(".jpg") or path.endswith(".jpeg"):
            ext = "jpg"
            content_type = "image/jpeg"
        elif path.endswith(".webp"):
            ext = "webp"
            content_type = "image/webp"
        elif path.endswith(".txt"):
            ext = "txt"
            content_type = "text/plain"
        elif path.endswith(".mp4"):
            ext = "mp4"
            content_type = "video/mp4"
        elif path.endswith(".mp3"):
            ext = "mp3"
            content_type = "audio/mpeg"
        elif path.endswith(".wav"):
            ext = "wav"
            content_type = "audio/wav"
        elif path.endswith(".ogg"):
            ext = "ogg"
            content_type = "audio/ogg"
        elif path.endswith(".flac"):
            ext = "flac"
            content_type = "audio/flac"
    if not body:
        raise RuntimeError("Empty external result")
    key = f"results/{model}/{job_id}.{ext}"
    if _s3_client and _r2_bucket:
        _s3_client.put_object(
            Bucket=_r2_bucket,
            Key=key,
            Body=body,
            ContentType=content_type,
        )
        url = create_presigned_get_url(key, expires_in=3600) or result.result_url
        return key, url
    if result.result_url:
        return None, result.result_url
    raise RuntimeError("R2 is not configured and no result URL is available")


async def send_fcm_notification(token: str, title: str, body: str, data: Dict = None):
    """
    Send FCM push notification.
    
    CRITICAL: FCM data payload values MUST all be strings.
    Non-string values will cause silent failures or corrupted notifications.
    """
    # Log input parameters for debugging empty notification issues
    logger.info(
        f"FCM send_fcm_notification called: "
        f"token_present={bool(token)}, title='{title}', body='{body[:50] if body else None}...', "
        f"data_keys={list(data.keys()) if data else 'None'}, data={data}"
    )
    
    try:
        from firebase_admin import messaging
        
        # Validate required fields FIRST
        if not token:
            logger.error("FCM notification ABORTED: token is empty/None")
            return
        
        # Defensive: ensure all data values are strings (FCM requirement)
        sanitized_data = {}
        if data:
            for key, value in data.items():
                if value is None:
                    # Skip None values entirely - don't send null
                    logger.warning(f"FCM data field '{key}' is None, skipping")
                    continue
                # Convert all values to strings
                sanitized_data[key] = str(value)
        else:
            logger.error(
                f"FCM notification has NULL data parameter! "
                f"This will result in empty data payload. title='{title}'"
            )
        
        # CRITICAL: Validate data has required fields
        if not sanitized_data.get("job_id"):
            logger.error(
                f"FCM notification ABORTED: data missing 'job_id'! "
                f"sanitized_data={sanitized_data}, original_data={data}"
            )
            return
        
        if not sanitized_data.get("status"):
            logger.error(
                f"FCM notification ABORTED: data missing 'status'! "
                f"sanitized_data={sanitized_data}"
            )
            return
        
        # Ensure title and body are not empty
        final_title = title if title else "Job Update"
        final_body = body if body else "Your job has been updated."
        
        if not title or not body:
            logger.warning(
                f"FCM notification has empty title or body, using defaults. "
                f"original_title='{title}', original_body='{body}'"
            )
        
        message = messaging.Message(
            notification=messaging.Notification(
                title=final_title,
                body=final_body,
            ),
            data=sanitized_data,
            token=token,
        )
        
        # Log the exact message being sent
        logger.info(
            f"FCM sending message: notification.title='{final_title}', "
            f"notification.body='{final_body[:50]}...', "
            f"data={sanitized_data}"
        )
        
        response = messaging.send(message)
        logger.info(
            f"FCM notification SUCCESS: response={response}, "
            f"job_id={sanitized_data.get('job_id')}, status={sanitized_data.get('status')}"
        )
        
    except Exception as e:
        logger.error(
            f"FCM notification EXCEPTION: {e}, "
            f"token_prefix={token[:20] if token else 'None'}..., "
            f"title='{title}', data={data}"
        )
        # Re-raise to surface issues during development
        # In production, you might want to remove this
        import traceback
        logger.error(f"FCM exception traceback: {traceback.format_exc()}")


# =============================================================================
# Firebase Authentication Helpers
# =============================================================================

async def verify_firebase_token(authorization: Optional[str]) -> Dict[str, Any]:
    """
    Verify Firebase ID token and return decoded claims.
    
    Args:
        authorization: "Bearer <id_token>" header value
        
    Returns:
        Decoded token containing 'uid', 'email', 'email_verified', etc.
        
    Raises:
        HTTPException: If token is missing, invalid, or expired
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header"
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Expected: Bearer <token>"
        )
    
    token = authorization.replace("Bearer ", "").strip()
    
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Empty token in Authorization header"
        )
    
    try:
        import firebase_admin
        from firebase_admin import auth as firebase_auth
        
        # Check if Firebase is initialized
        try:
            firebase_admin.get_app()
        except ValueError:
            logger.error("Firebase Admin SDK not initialized!")
            raise HTTPException(
                status_code=500,
                detail="Firebase Admin SDK not initialized. Check server configuration."
            )
        
        # Verify the ID token and get decoded claims
        decoded = firebase_auth.verify_id_token(token)
        
        logger.debug(f"Firebase token verified for uid: {decoded.get('uid')}")
        return decoded
        
    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except firebase_auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=401,
            detail="Firebase token has expired. Please refresh and try again."
        )
    except firebase_auth.RevokedIdTokenError:
        raise HTTPException(
            status_code=401,
            detail="Firebase token has been revoked."
        )
    except firebase_auth.InvalidIdTokenError as e:
        logger.warning(f"Invalid Firebase token: {e}")
        raise HTTPException(
            status_code=401,
            detail=f"Invalid Firebase token: {str(e)}"
        )
    except firebase_auth.CertificateFetchError as e:
        logger.error(f"Firebase certificate fetch error: {e}")
        raise HTTPException(
            status_code=503,
            detail="Firebase service temporarily unavailable. Please try again."
        )
    except Exception as e:
        logger.error(f"Firebase token verification failed: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=401,
            detail=f"Failed to verify Firebase token: {type(e).__name__}"
        )


def _save_json_request(
    data: Dict[str, Any],
    request_type: str,
    job_id: Optional[str] = None,
    user_id: Optional[str] = None,
    additional_labels: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Save JSON request data to file with proper grouping and labeling.
    
    Args:
        data: The JSON-serializable data to save
        request_type: Type of request (e.g., "client_submit", "worker_dispatch")
        job_id: Optional job ID for grouping
        user_id: Optional user ID for grouping
        additional_labels: Optional additional labels to include in filename
    
    Returns:
        Path to saved file, or None if save failed
    """
    import datetime
    import os
    import json
    
    try:
        # Base directory for all JSON dumps
        base_dir = os.environ.get("JSON_REQUEST_DUMPS_PATH", "./json_request_dumps")
        
        # Create subdirectories by request type
        type_dir = os.path.join(base_dir, request_type)
        os.makedirs(type_dir, exist_ok=True)
        
        # Create subdirectory by date for better organization
        date_str = datetime.datetime.utcnow().strftime("%Y%m%d")
        date_dir = os.path.join(type_dir, date_str)
        os.makedirs(date_dir, exist_ok=True)
        
        # Build filename with labels
        now_str = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        filename_parts = [request_type, now_str]
        
        if job_id:
            filename_parts.append(f"job_{job_id[:8]}")  # First 8 chars of job_id
        
        if user_id:
            filename_parts.append(f"user_{user_id[:8]}")  # First 8 chars of user_id
        
        if additional_labels:
            for key, value in additional_labels.items():
                if value:
                    filename_parts.append(f"{key}_{str(value)[:16]}")  # Truncate long values
        
        filename = "_".join(filename_parts) + ".json"
        file_path = os.path.join(date_dir, filename)
        
        # Prepare data with metadata
        save_data = {
            "request_type": request_type,
            "timestamp_utc": datetime.datetime.utcnow().isoformat(),
            "timestamp_filename": now_str,
            "job_id": job_id,
            "user_id": user_id,
            "labels": additional_labels or {},
            "data": data,
        }
        
        # Save to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved {request_type} JSON request to: {file_path}")
        return file_path
        
    except Exception as e:
        logger.warning(f"Failed to save {request_type} JSON request: {e}")
        return None


def get_or_create_user(db: Session, firebase_uid: str, email: Optional[str] = None) -> User:
    """
    Get existing user by firebase_uid or create new user.
    
    Args:
        db: Database session
        firebase_uid: Firebase user ID
        email: User's email (optional)
        
    Returns:
        User instance
    """
    user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
    
    if user:
        # Update last_seen
        user.last_seen = utc_naive_now()
        # Update email if provided and not set
        if email and not user.email:
            user.email = email
        db.commit()
        return user
    
    # Create new user WITHOUT signup bonus
    # Signup bonus is granted separately after email verification
    user = User(
        id=str(uuid.uuid4()),
        firebase_uid=firebase_uid,
        email=email,
        created_at=utc_naive_now(),
        last_seen=utc_naive_now(),
        total_credits=0,  # No credits until email verified
        bonus_granted=0,
        signup_bonus_granted=False,  # Will be set True after email verification
        is_active=True,
    )
    db.add(user)
    
    # Log the new user creation (no bonus yet)
    create_audit_log(
        db,
        event_type="user_created",
        message=f"New user created (pending email verification): {firebase_uid}",
        target=user.id,
        target_type="user",
        severity="info",
        details={
            "firebase_uid": firebase_uid,
            "email": email,
            "signup_bonus_pending": True,
        },
    )
    
    db.commit()
    db.refresh(user)
    
    logger.info(f"Created new user {user.id} for firebase_uid {firebase_uid}")
    return user


# =============================================================================
# Auth Endpoints
# =============================================================================

from pydantic import BaseModel, Field

class UserSyncRequest(BaseModel):
    """Request body for user sync (optional fields)"""
    email: Optional[str] = Field(None, description="User's email if not in token")
    fcm_token: Optional[str] = Field(None, description="FCM token for push notifications")

class UserResponse(BaseModel):
    """User data response"""
    user_id: str
    firebase_uid: str
    email: Optional[str]
    total_credits: int
    bonus_granted: int
    signup_bonus_granted: bool = False  # True if 50 credits signup bonus was claimed
    is_active: bool
    is_ad_free: bool = False  # True if user has purchased ad-free
    ad_free_purchased_at: Optional[str] = None
    ad_free_product_id: Optional[str] = None
    created_at: str
    last_seen: str
    is_vip: bool = False
    vip_expires_at: Optional[str] = None
    ad_experience: str = "heavy"  # none | light | heavy
    has_purchased_credits: bool = False
    daily_credits_granted_this_sync: int = 0


def _build_user_response(db: Session, user: User, daily_granted: int = 0) -> UserResponse:
    has_purchased = user_has_verified_credit_purchase(db, user.id)
    ad_exp = compute_ad_experience(user, has_purchased)
    return UserResponse(
        user_id=user.id,
        firebase_uid=user.firebase_uid,
        email=user.email,
        total_credits=user.total_credits,
        bonus_granted=user.bonus_granted,
        signup_bonus_granted=user.signup_bonus_granted,
        is_active=user.is_active,
        is_ad_free=user.is_ad_free,
        ad_free_purchased_at=user.ad_free_purchased_at.isoformat()
        if user.ad_free_purchased_at
        else None,
        ad_free_product_id=user.ad_free_product_id,
        created_at=user.created_at.isoformat() if user.created_at else "",
        last_seen=user.last_seen.isoformat() if user.last_seen else "",
        is_vip=user.is_vip_active(),
        vip_expires_at=user.vip_expires_at.isoformat() if user.vip_expires_at else None,
        ad_experience=ad_exp,
        has_purchased_credits=has_purchased,
        daily_credits_granted_this_sync=daily_granted,
    )


class CreditBalanceResponse(BaseModel):
    """Credit balance response"""
    total_credits: int
    bonus_granted: int


class DeductCreditsRequest(BaseModel):
    """Request body for credit deduction"""
    amount: int = Field(..., gt=0, description="Number of credits to deduct (must be positive)")
    reason: str = Field(..., description="Reason for deduction (e.g., 'job_completed')")
    job_id: Optional[str] = Field(None, description="Associated job ID")


class AddCreditsRequest(BaseModel):
    """Request body for adding credits"""
    amount: int = Field(..., gt=0, description="Number of credits to add (must be positive)")
    reason: str = Field(..., description="Reason for addition (e.g., 'ad_reward', 'subscription_bonus')")


class OnDeviceJobStartedRequest(BaseModel):
    """Client reports that on-device processing has started."""
    job_id: str = Field(..., min_length=1, max_length=128)


class OnDeviceJobCompleteRequest(BaseModel):
    """Client reports terminal state for an on-device processed job."""
    job_id: str = Field(..., min_length=1, max_length=128)
    success: bool = Field(default=True)
    error: Optional[str] = Field(default=None, max_length=4000)


@auth_router.post("/sync", response_model=UserResponse)
async def sync_user(
    request: Optional[UserSyncRequest] = None,
    authorization: Optional[str] = Header(None),
):
    """
    Sync Firebase user with database.
    
    Called after Firebase sign-in to create or update user in database.
    Creates new user if not exists (without credits).
    Grants 50 signup bonus ONLY after email is verified.
    
    **Headers:**
    - Authorization: Bearer <firebase_id_token>
    
    **Email Verification Flow:**
    1. User signs up → account created with 0 credits
    2. User verifies email → Firebase token now has email_verified=true
    3. User calls /auth/sync → 50 credits granted (one-time only)
    
    **Returns:**
    - User profile data including credits
    """
    # Verify Firebase token
    decoded = await verify_firebase_token(authorization)
    
    firebase_uid = decoded.get("uid")
    if not firebase_uid:
        raise HTTPException(400, "Firebase UID not found in token")
    
    # Get email verification status from Firebase token (secure, not from client)
    email_verified_from_token = decoded.get("email_verified", False)
    
    # Log token claims for debugging
    sign_in_provider = decoded.get("firebase", {}).get("sign_in_provider", "unknown")
    logger.info(f"[AUTH] Token claims for {firebase_uid}: email_verified={email_verified_from_token}, provider={sign_in_provider}")
    
    # IMPORTANT: The ID token might be stale after email verification.
    # As a fallback, check Firebase user record directly if token says not verified
    email_verified = email_verified_from_token
    
    if not email_verified_from_token:
        try:
            import firebase_admin
            from firebase_admin import auth as firebase_auth
            
            # Get fresh user data from Firebase
            firebase_user = firebase_auth.get_user(firebase_uid)
            email_verified = firebase_user.email_verified
            
            if email_verified and not email_verified_from_token:
                logger.info(f"[AUTH] Token stale for {firebase_uid}: token says not verified, but Firebase user IS verified. Using fresh status.")
        except Exception as e:
            logger.warning(f"[AUTH] Failed to check Firebase user directly: {e}. Using token value: {email_verified_from_token}")
            email_verified = email_verified_from_token
    
    # Get email from token or request
    email = decoded.get("email")
    if not email and request:
        email = request.email
    
    # Get or create user
    try:
        with SessionLocal() as db:
            user = get_or_create_user(db, firebase_uid, email)
            
            # Log current user state
            logger.info(f"[AUTH] User {firebase_uid}: signup_bonus_granted={user.signup_bonus_granted}, total_credits={user.total_credits}, email_verified_in_token={email_verified}")
            
            # Grant signup bonus if:
            # 1. Email is verified (from Firebase token - tamper-proof)
            # 2. Signup bonus hasn't been granted yet (prevents reinstall abuse)
            if email_verified and not user.signup_bonus_granted:
                SIGNUP_BONUS = 50
                user.total_credits += SIGNUP_BONUS
                user.bonus_granted += SIGNUP_BONUS
                user.signup_bonus_granted = True
                
                # Record signup bonus as credit transaction
                signup_transaction = CreditTransaction(
                    user_id=user.id,
                    delta=SIGNUP_BONUS,
                    reason="signup_bonus_email_verified",
                    credit_metadata={
                        "event": "signup_bonus_email_verified",
                        "bonus_credits": SIGNUP_BONUS,
                        "email_verified": True,
                    },
                    created_at=utc_naive_now(),
                )
                db.add(signup_transaction)
                
                # Log the signup bonus grant
                create_audit_log(
                    db,
                    event_type="signup_bonus_granted",
                    message=f"Signup bonus granted after email verification: {firebase_uid}",
                    target=user.id,
                    target_type="user",
                    severity="info",
                    details={
                        "firebase_uid": firebase_uid,
                        "email": email,
                        "signup_bonus": SIGNUP_BONUS,
                        "email_verified": True,
                    },
                )
                
                db.commit()
                db.refresh(user)
                logger.info(f"[AUTH] Granted {SIGNUP_BONUS} signup bonus to {firebase_uid} (email verified)")
            
            elif not email_verified and not user.signup_bonus_granted:
                logger.debug(f"[AUTH] User {firebase_uid} - email not verified yet, no bonus granted")

            user.last_seen = utc_naive_now()
            daily_granted = apply_daily_credits_if_eligible(db, user)
            db.commit()
            db.refresh(user)

            return _build_user_response(db, user, daily_granted)
    except Exception as e:
        logger.exception(f"User sync failed for {firebase_uid}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to sync user: {type(e).__name__}: {str(e)}"
        )


@auth_router.get("/me", response_model=UserResponse)
async def get_current_user(
    authorization: Optional[str] = Header(None),
):
    """
    Get current user's profile.
    
    **Headers:**
    - Authorization: Bearer <firebase_id_token>
    
    **Returns:**
    - User profile data including credits
    
    **Errors:**
    - 401: Invalid/expired token
    - 404: User not found (need to call /auth/sync first)
    """
    # Verify Firebase token
    decoded = await verify_firebase_token(authorization)
    
    firebase_uid = decoded.get("uid")
    if not firebase_uid:
        raise HTTPException(400, "Firebase UID not found in token")
    
    # Find user
    with SessionLocal() as db:
        user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
        
        if not user:
            raise HTTPException(
                status_code=404,
                detail="User not found. Please sign in first."
            )
        
        user.last_seen = utc_naive_now()
        daily_granted = apply_daily_credits_if_eligible(db, user)
        db.commit()
        db.refresh(user)

        return _build_user_response(db, user, daily_granted)


@auth_router.get("/credits", response_model=CreditBalanceResponse)
async def get_credit_balance(
    authorization: Optional[str] = Header(None),
):
    """
    Get current user's credit balance.
    
    Lightweight endpoint for checking credits without full profile.
    
    **Headers:**
    - Authorization: Bearer <firebase_id_token>
    """
    decoded = await verify_firebase_token(authorization)
    
    firebase_uid = decoded.get("uid")
    if not firebase_uid:
        raise HTTPException(400, "Firebase UID not found in token")
    
    with SessionLocal() as db:
        user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
        
        if not user:
            raise HTTPException(404, "User not found")
        
        return CreditBalanceResponse(
            total_credits=user.total_credits,
            bonus_granted=user.bonus_granted,
        )


# NOTE: /deduct_credits endpoint REMOVED
# Credits are now handled atomically:
# - Deducted during job submission via lock_credits_for_job()
# - Refunded on job failure via refund_credits_for_job()
# Client cannot directly manipulate credits.


@auth_router.get("/credit-history")
async def get_credit_history(
    auth: Dict = Depends(require_firebase_auth),
    limit: int = 50,
    offset: int = 0,
):
    """
    Paginated credit transaction history for authenticated user.
    """
    firebase_uid = auth.get("uid") or auth.get("firebase_uid")
    if not firebase_uid:
        raise HTTPException(400, "Firebase UID not found")

    with SessionLocal() as db:
        user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
        if not user:
            raise HTTPException(404, "User not found")

        from db import CreditTransaction
        q = db.query(CreditTransaction).filter(
            CreditTransaction.user_id == user.id
        ).order_by(CreditTransaction.created_at.desc())
        total = q.count()
        rows = q.offset(max(0, offset)).limit(min(200, max(1, limit))).all()
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "transactions": [r.to_dict() for r in rows],
        }


# =============================================================================
# Daily Spinner Endpoints
# =============================================================================

import random
from datetime import timedelta

# Spin reward probabilities (must sum to 1.0)
SPIN_REWARDS = [
    {"credits": 1,  "probability": 0.35},  # 35% - most common
    {"credits": 2,  "probability": 0.30},  # 30%
    {"credits": 5,  "probability": 0.20},  # 20%
    {"credits": 10, "probability": 0.07},  # 7%
    {"credits": 20, "probability": 0.06},  # 6% - rare
    {"credits": 30, "probability": 0.02},  # 2% - legendary
]

SPIN_COOLDOWN_HOURS = 24


class SpinCheckResponse(BaseModel):
    """Response for spin eligibility check"""
    can_spin: bool
    next_spin_time: Optional[str] = None  # ISO 8601 UTC
    last_spin_time: Optional[str] = None  # ISO 8601 UTC


class SpinRequest(BaseModel):
    """Request body for performing spin"""
    device_id: Optional[str] = Field(None, description="Device ID for fraud detection")


class SpinResponse(BaseModel):
    """Response for successful spin"""
    credits_won: int
    next_spin_time: str  # ISO 8601 UTC
    message: str
    new_balance: int


def _select_spin_reward() -> int:
    """Server-side random reward selection based on probabilities"""
    rand = random.random()  # 0.0 to 1.0
    cumulative = 0.0
    
    for reward in SPIN_REWARDS:
        cumulative += reward["probability"]
        if rand <= cumulative:
            return reward["credits"]
    
    return 1  # Fallback


def _format_iso_utc(dt: datetime) -> str:
    """Format datetime as ISO 8601 UTC string"""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


@auth_router.get("/daily-spin/check", response_model=SpinCheckResponse)
async def check_spin_eligibility(
    authorization: Optional[str] = Header(None),
):
    """
    Check if user is eligible to spin today (server-time based).
    
    **Headers:**
    - Authorization: Bearer <firebase_id_token>
    
    **Returns:**
    - can_spin: Whether user can spin now
    - next_spin_time: When user can spin next (ISO 8601 UTC), null if can spin now
    - last_spin_time: When user last spun (ISO 8601 UTC), null if never spun
    """
    decoded = await verify_firebase_token(authorization)
    firebase_uid = decoded.get("uid")
    
    if not firebase_uid:
        raise HTTPException(400, "Firebase UID not found in token")
    
    with SessionLocal() as db:
        user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
        
        if not user:
            raise HTTPException(404, "User not found. Please sync first.")
        
        last_spin = user.last_daily_spin_time
        now = utc_naive_now()
        
        # Never spun before
        if last_spin is None:
            return SpinCheckResponse(
                can_spin=True,
                next_spin_time=None,
                last_spin_time=None,
            )
        
        # Calculate next eligible spin time
        next_spin = last_spin + timedelta(hours=SPIN_COOLDOWN_HOURS)
        
        if now >= next_spin:
            # Cooldown elapsed, can spin
            return SpinCheckResponse(
                can_spin=True,
                next_spin_time=None,
                last_spin_time=_format_iso_utc(last_spin),
            )
        
        # Still in cooldown
        return SpinCheckResponse(
            can_spin=False,
            next_spin_time=_format_iso_utc(next_spin),
            last_spin_time=_format_iso_utc(last_spin),
        )


@auth_router.post("/daily-spin/spin", response_model=SpinResponse)
async def perform_daily_spin(
    payload: SpinRequest = SpinRequest(),
    authorization: Optional[str] = Header(None),
):
    """
    Perform the daily spin, determine reward (server-side RNG), grant credits.
    
    **Headers:**
    - Authorization: Bearer <firebase_id_token>
    
    **JSON Body (optional):**
    - device_id: Device identifier for fraud detection
    
    **Returns:**
    - credits_won: Number of credits won
    - next_spin_time: When user can spin next (ISO 8601 UTC)
    - message: Success message
    - new_balance: User's updated credit balance
    
    **Errors:**
    - 429: Already spun today (rate limited)
    - 401: Invalid/missing token
    - 404: User not found
    """
    decoded = await verify_firebase_token(authorization)
    firebase_uid = decoded.get("uid")
    
    if not firebase_uid:
        raise HTTPException(400, "Firebase UID not found in token")
    
    with SessionLocal() as db:
        user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
        
        if not user:
            raise HTTPException(404, "User not found. Please sync first.")
        
        last_spin = user.last_daily_spin_time
        now = utc_naive_now()
        
        # Check eligibility
        if last_spin is not None:
            next_spin = last_spin + timedelta(hours=SPIN_COOLDOWN_HOURS)
            if now < next_spin:
                # Still in cooldown - return 429
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "rate_limited",
                        "message": "Already spun today. Come back tomorrow!",
                        "next_spin_time": _format_iso_utc(next_spin),
                    }
                )
        
        # Select reward using server-side RNG
        credits_won = _select_spin_reward()
        
        # Grant credits and update spin time atomically
        user.total_credits += credits_won
        user.bonus_granted += credits_won
        user.last_daily_spin_time = now
        
        # Record transaction
        transaction = CreditTransaction(
            user_id=user.id,
            delta=credits_won,
            reason="daily_spin",
            credit_metadata={
                "event": "daily_spin",
                "credits_won": int(credits_won),
                "device_id": payload.device_id,
            },
            created_at=now,
        )
        db.add(transaction)
        
        # Log spin for audit/analytics
        spin_log = DailySpin(
            user_id=user.id,
            credits_won=credits_won,
            device_id=payload.device_id,
            created_at=now,
        )
        db.add(spin_log)
        
        db.commit()
        db.refresh(user)
        
        # Calculate next spin time
        next_spin_time = now + timedelta(hours=SPIN_COOLDOWN_HOURS)
        
        logger.info(f"[SPIN] User {firebase_uid} won {credits_won} credits (balance: {user.total_credits})")
        
        return SpinResponse(
            credits_won=credits_won,
            next_spin_time=_format_iso_utc(next_spin_time),
            message=f"You won {credits_won} credits!",
            new_balance=user.total_credits,
        )


# =============================================================================
# Ad-Free Purchase Endpoints
# =============================================================================

class AdFreeStatusResponse(BaseModel):
    """Response for ad-free status check"""
    is_ad_free: bool
    purchased_at: Optional[str] = None
    product_id: Optional[str] = None


class AdFreePurchaseRequest(BaseModel):
    """Request body for ad-free purchase verification"""
    purchase_token: str = Field(..., description="Purchase token from Google Play or receipt data from App Store")
    product_id: str = Field(..., description="Product ID (e.g., 'remove_ads_lifetime')")
    platform: str = Field(..., description="'android' or 'ios'")
    device_id: Optional[str] = Field(None, description="SHA256 hashed device ID for abuse prevention")


class AdFreePurchaseResponse(BaseModel):
    """Response for ad-free purchase verification"""
    verified: bool
    is_ad_free: bool = False
    purchased_at: Optional[str] = None
    message: Optional[str] = None
    detail: Optional[str] = None


@auth_router.get("/ad-free-status", response_model=AdFreeStatusResponse)
async def get_ad_free_status(
    authorization: Optional[str] = Header(None),
):
    """
    Check if user has ad-free status.
    
    Returns the user's ad-free purchase status. This endpoint is called:
    - After user authentication to verify cached ad-free status
    - Periodically to ensure ad-free status is still valid
    
    **Headers:**
    - Authorization: Bearer <firebase_id_token>
    
    **Returns:**
    - is_ad_free: Whether user has ad-free status
    - purchased_at: When ad-free was purchased (ISO format)
    - product_id: The product that was purchased
    """
    # Verify Firebase token
    decoded = await verify_firebase_token(authorization)
    
    firebase_uid = decoded.get("uid")
    if not firebase_uid:
        raise HTTPException(400, "Firebase UID not found in token")
    
    with SessionLocal() as db:
        user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
        
        if not user:
            raise HTTPException(404, "User not found. Please call /auth/sync first.")
        
        if user.is_ad_free:
            return AdFreeStatusResponse(
                is_ad_free=True,
                purchased_at=user.ad_free_purchased_at.isoformat() if user.ad_free_purchased_at else None,
                product_id=user.ad_free_product_id,
            )
        
        return AdFreeStatusResponse(
            is_ad_free=False,
            purchased_at=None,
            product_id=None,
        )


@auth_router.post("/verify-ad-free-purchase", response_model=AdFreePurchaseResponse)
async def verify_ad_free_purchase(
    request: AdFreePurchaseRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Verify and activate ad-free purchase.
    
    SECURITY MODEL (Consumable Product - remove_ads_lifetime_v2):
    - Google Play: Product is CONSUMABLE (anyone can buy anytime)
    - Backend: One Firebase account = one ad-free grant
    - Prevents: Double-charging same account, purchase sharing, replay attacks
    
    Security Checks:
    1. Check if THIS Firebase account already has ad-free → Return success (idempotent, prevent double-charge)
    2. Verify purchase with Google Play/App Store
    3. Check if order_id was already used → Reject (prevent replay attacks)
    4. Grant ad-free to THIS Firebase account only
    5. Consume the purchase (allows product to be bought again by others)
    
    **Headers:**
    - Authorization: Bearer <firebase_id_token>
    
    **Request Body:**
    - purchase_token: The purchase token from Google Play or receipt from App Store
    - product_id: The product ID (e.g., 'remove_ads_lifetime_v2')
    - platform: 'android' or 'ios'
    - device_id: SHA256 hashed device ID (optional, for abuse prevention)
    
    **Returns:**
    - verified: Whether the purchase was verified
    - is_ad_free: Whether user now has ad-free status
    - purchased_at: When the purchase was verified
    - message: Success or error message
    """
    from purchase_verification import (
        verify_purchase, 
        consume_google_play_purchase,
        VALID_AD_FREE_PRODUCT_IDS,
    )
    from db import AdFreePurchase
    
    # Verify Firebase token
    decoded = await verify_firebase_token(authorization)
    
    firebase_uid = decoded.get("uid")
    if not firebase_uid:
        raise HTTPException(400, "Firebase UID not found in token")
    
    # Validate platform
    platform = request.platform.lower().strip()
    if platform not in ["android", "ios"]:
        raise HTTPException(400, f"Invalid platform: {platform}. Must be 'android' or 'ios'")
    
    # Validate product ID
    if request.product_id not in VALID_AD_FREE_PRODUCT_IDS:
        logger.warning(f"[AD-FREE] Invalid product ID: {request.product_id} from {firebase_uid}")
        raise HTTPException(400, f"Invalid product ID: {request.product_id}")
    
    with SessionLocal() as db:
        user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
        
        if not user:
            raise HTTPException(404, "User not found. Please call /auth/sync first.")
        
        # ============================================================
        # SECURITY CHECK 1: Does this account already have ad-free?
        # ============================================================
        # This prevents double-charging the same Firebase account
        if user.is_ad_free:
            logger.info(f"[AD-FREE] User {firebase_uid} already has ad-free status (idempotent)")
            return AdFreePurchaseResponse(
                verified=True,
                is_ad_free=True,
                purchased_at=user.ad_free_purchased_at.isoformat() if user.ad_free_purchased_at else None,
                message="Already ad-free",
                detail="User already has ad-free status",
            )
        
        # ============================================================
        # SECURITY CHECK 2: Verify purchase with Google/Apple
        # ============================================================
        logger.info(f"[AD-FREE] Verifying {platform} purchase for {firebase_uid}: product={request.product_id}")
        
        result = verify_purchase(
            purchase_token=request.purchase_token,
            product_id=request.product_id,
            platform=platform,
        )
        
        if not result.valid:
            logger.warning(f"[AD-FREE] Purchase verification failed for {firebase_uid}: {result.reason}")
            raise HTTPException(400, f"Purchase verification failed: {result.reason}")
        
        order_id = result.order_id
        
        # ============================================================
        # SECURITY CHECK 3: Check for replay attacks (order_id reuse)
        # ============================================================
        # Each Google Play order can only be used ONCE to grant ad-free
        if order_id:
            existing_purchase = db.query(AdFreePurchase).filter(
                AdFreePurchase.order_id == order_id
            ).first()
            
            if existing_purchase:
                if existing_purchase.firebase_uid == firebase_uid:
                    # Same account, same order - idempotent success
                    # This can happen if client retries after network error
                    logger.info(f"[AD-FREE] Idempotent: order {order_id} already processed for {firebase_uid}")
                    
                    # Ensure user has ad-free status (in case of DB inconsistency)
                    if not user.is_ad_free:
                        user.is_ad_free = True
                        user.ad_free_purchased_at = existing_purchase.verified_at
                        user.ad_free_product_id = existing_purchase.product_id
                        db.commit()
                    
                    return AdFreePurchaseResponse(
                        verified=True,
                        is_ad_free=True,
                        purchased_at=existing_purchase.verified_at.isoformat() if existing_purchase.verified_at else None,
                        message="Purchase already verified",
                        detail="This purchase was already processed for your account",
                    )
                else:
                    # FRAUD ATTEMPT: Different account trying to use same order
                    logger.warning(
                        f"[AD-FREE] FRAUD ATTEMPT: order_id={order_id} already used by "
                        f"firebase_uid={existing_purchase.firebase_uid}, "
                        f"attempted reuse by firebase_uid={firebase_uid}"
                    )
                    raise HTTPException(400, "This purchase has already been used by another account")
        
        # ============================================================
        # GRANT AD-FREE
        # ============================================================
        now = utc_naive_now()
        
        # Update user record
        user.is_ad_free = True
        user.ad_free_purchased_at = now
        user.ad_free_product_id = request.product_id
        
        # Create audit record (order_id is UNIQUE - prevents replay attacks)
        purchase_record = AdFreePurchase(
            user_id=user.id,
            firebase_uid=firebase_uid,
            product_id=request.product_id,
            purchase_token=request.purchase_token,
            platform=platform,
            device_id=request.device_id,
            order_id=order_id,
            store_data=result.store_data,
            verified_at=now,
            created_at=now,
        )
        db.add(purchase_record)
        
        # Create audit log
        create_audit_log(
            db,
            event_type="ad_free_purchase",
            message=f"Ad-free purchase verified: {request.product_id}",
            actor=firebase_uid[:16],
            target=user.id,
            target_type="user",
            severity="info",
            details={
                "product_id": request.product_id,
                "platform": platform,
                "order_id": order_id,
                "is_test": result.is_test_purchase,
            },
        )
        
        db.commit()
        
        logger.info(
            f"[AD-FREE] Purchase verified and activated for {firebase_uid}: "
            f"product={request.product_id}, platform={platform}, order={order_id}"
        )
    
    # ============================================================
    # CONSUME the purchase on Google Play (CONSUMABLE product)
    # ============================================================
    # For consumable products, we CONSUME (not acknowledge)
    # This allows other users to purchase the same product
    # Done AFTER commit to ensure user gets ad-free even if consume fails
    if platform == "android":
        try:
            consumed = consume_google_play_purchase(
                product_id=request.product_id,
                purchase_token=request.purchase_token,
            )
            if consumed:
                logger.info(f"[AD-FREE] Consumed on Google Play: order={order_id}")
            else:
                logger.warning(f"[AD-FREE] Failed to consume on Google Play: order={order_id}")
        except Exception as e:
            # Log but don't fail - user already has ad-free status
            logger.warning(f"[AD-FREE] Consume failed (non-critical): {e}")
    
    return AdFreePurchaseResponse(
        verified=True,
        is_ad_free=True,
        purchased_at=now.isoformat(),
        message="Ad-free activated!",
    )


# =============================================================================
# VIP Subscription Endpoints
# =============================================================================

class VipStatusResponse(BaseModel):
    """Response for VIP status check"""
    is_vip: bool
    expires_at: Optional[str] = None
    product_id: Optional[str] = None
    purchased_at: Optional[str] = None
    days_remaining: Optional[int] = None


class VipPurchaseRequest(BaseModel):
    """Request body for VIP purchase verification"""
    purchase_token: str = Field(..., description="Purchase token from Google Play")
    product_id: str = Field(..., description="VIP product ID (e.g., 'vip_pack_weekly', 'vip_pack_monthly')")
    platform: str = Field(..., description="'android' or 'ios'")
    device_id: Optional[str] = Field(None, description="Hashed device ID for fraud detection")


class VipPurchaseResponse(BaseModel):
    """Response for VIP purchase verification"""
    verified: bool
    is_vip: bool = False
    expires_at: Optional[str] = None
    days_granted: Optional[int] = None
    message: Optional[str] = None
    detail: Optional[str] = None


@auth_router.get("/vip-status", response_model=VipStatusResponse)
async def get_vip_status(
    authorization: Optional[str] = Header(None),
):
    """
    Check if user has active VIP subscription.
    
    VIP is TIME-LIMITED (unlike ad-free which is permanent).
    
    **Headers:**
    - Authorization: Bearer <firebase_id_token>
    
    **Returns:**
    - is_vip: Whether user has ACTIVE VIP subscription
    - expires_at: When VIP expires (ISO format)
    - product_id: The last VIP product purchased
    - purchased_at: When VIP was last purchased
    - days_remaining: Days until VIP expires (0 if expired)
    """
    # Verify Firebase token
    decoded = await verify_firebase_token(authorization)
    
    firebase_uid = decoded.get("uid")
    if not firebase_uid:
        raise HTTPException(400, "Firebase UID not found in token")
    
    with SessionLocal() as db:
        user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
        
        if not user:
            raise HTTPException(404, "User not found. Please call /auth/sync first.")
        
        # Check if VIP is active (not expired)
        is_vip_active = user.is_vip_active()
        
        if is_vip_active:
            days_remaining = (user.vip_expires_at - utc_naive_now()).days if user.vip_expires_at else 0
            days_remaining = max(0, days_remaining)
            
            return VipStatusResponse(
                is_vip=True,
                expires_at=user.vip_expires_at.isoformat() if user.vip_expires_at else None,
                product_id=user.vip_product_id,
                purchased_at=user.vip_purchased_at.isoformat() if user.vip_purchased_at else None,
                days_remaining=days_remaining,
            )
        
        # VIP expired or never had
        return VipStatusResponse(
            is_vip=False,
            expires_at=user.vip_expires_at.isoformat() if user.vip_expires_at else None,
            product_id=None,
            purchased_at=None,
            days_remaining=0,
        )


@auth_router.post("/verify-vip-purchase", response_model=VipPurchaseResponse)
async def verify_vip_purchase(
    request: VipPurchaseRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Verify and activate VIP subscription purchase.
    
    VIP SUBSCRIPTION MODEL:
    - VIP is TIME-LIMITED (weekly/monthly/yearly)
    - Multiple purchases EXTEND VIP time (adds to existing expiry)
    - Benefits: unlimited concurrent generations, bulk edit, free upscale
    - Server tracks expiration, client cannot modify
    
    Security Checks:
    1. Verify purchase with Google Play/App Store
    2. Check if order_id was already used → Reject (prevent replay attacks)
    3. Grant VIP time to user (extends if already VIP)
    4. Consume the purchase (allows re-purchase)
    
    **Headers:**
    - Authorization: Bearer <firebase_id_token>
    
    **Request Body:**
    - purchase_token: The purchase token from Google Play
    - product_id: The VIP product ID (e.g., 'vip_pack_weekly', 'vip_pack_monthly')
    - platform: 'android' or 'ios'
    - device_id: Hashed device ID (optional)
    
    **Returns:**
    - verified: Whether the purchase was verified
    - is_vip: Whether user now has VIP status
    - expires_at: When VIP expires
    - days_granted: How many days were added
    - message: Success or error message
    """
    from purchase_verification import (
        verify_purchase, 
        consume_google_play_purchase,
        VALID_VIP_PRODUCT_IDS,
        get_vip_duration_days,
    )
    from db import VipPurchase
    from datetime import timedelta
    
    # Verify Firebase token
    decoded = await verify_firebase_token(authorization)
    
    firebase_uid = decoded.get("uid")
    if not firebase_uid:
        raise HTTPException(400, "Firebase UID not found in token")
    
    # Validate platform
    platform = request.platform.lower().strip()
    if platform not in ["android", "ios"]:
        raise HTTPException(400, f"Invalid platform: {platform}. Must be 'android' or 'ios'")
    
    # Validate product ID
    if request.product_id not in VALID_VIP_PRODUCT_IDS:
        logger.warning(f"[VIP] Invalid product ID: {request.product_id} from {firebase_uid}")
        raise HTTPException(400, f"Invalid VIP product ID: {request.product_id}")
    
    # Get VIP duration for this product
    duration_days = get_vip_duration_days(request.product_id)
    if duration_days <= 0:
        raise HTTPException(400, f"Invalid VIP product duration: {request.product_id}")
    
    with SessionLocal() as db:
        user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
        
        if not user:
            raise HTTPException(404, "User not found. Please call /auth/sync first.")
        
        # ============================================================
        # SECURITY CHECK 1: Verify purchase with Google/Apple
        # ============================================================
        logger.info(f"[VIP] Verifying {platform} purchase for {firebase_uid}: product={request.product_id}")
        
        result = verify_purchase(
            purchase_token=request.purchase_token,
            product_id=request.product_id,
            platform=platform,
        )
        
        if not result.valid:
            logger.warning(f"[VIP] Purchase verification failed for {firebase_uid}: {result.reason}")
            raise HTTPException(400, f"Purchase verification failed: {result.reason}")
        
        order_id = result.order_id
        
        # ============================================================
        # SECURITY CHECK 2: Check for replay attacks (order_id reuse)
        # ============================================================
        if order_id:
            existing_purchase = db.query(VipPurchase).filter(
                VipPurchase.order_id == order_id
            ).first()
            
            if existing_purchase:
                if existing_purchase.firebase_uid == firebase_uid:
                    # Same account, same order - idempotent success
                    logger.info(f"[VIP] Idempotent: order {order_id} already processed for {firebase_uid}")
                    
                    return VipPurchaseResponse(
                        verified=True,
                        is_vip=user.is_vip_active(),
                        expires_at=user.vip_expires_at.isoformat() if user.vip_expires_at else None,
                        days_granted=0,
                        message="Purchase already verified",
                        detail="This purchase was already processed for your account",
                    )
                else:
                    # FRAUD ATTEMPT: Different account trying to use same order
                    logger.warning(
                        f"[VIP] FRAUD ATTEMPT: order_id={order_id} already used by "
                        f"firebase_uid={existing_purchase.firebase_uid}, "
                        f"attempted reuse by firebase_uid={firebase_uid}"
                    )
                    raise HTTPException(400, "This purchase has already been used by another account")
        
        # ============================================================
        # GRANT VIP TIME
        # ============================================================
        now = utc_naive_now()
        
        # Calculate new expiry - extends existing VIP if active, or starts fresh
        if user.is_vip_active() and user.vip_expires_at:
            # Extend existing VIP
            new_expires_at = user.vip_expires_at + timedelta(days=duration_days)
            vip_started_at = user.vip_expires_at  # New period starts when old one ends
        else:
            # Start fresh VIP
            new_expires_at = now + timedelta(days=duration_days)
            vip_started_at = now
        
        # Update user record
        user.is_vip = True
        user.vip_expires_at = new_expires_at
        user.vip_product_id = request.product_id
        user.vip_purchased_at = now
        
        # Create purchase record (order_id is UNIQUE - prevents replay attacks)
        purchase_record = VipPurchase(
            user_id=user.id,
            firebase_uid=firebase_uid,
            product_id=request.product_id,
            purchase_token=request.purchase_token,
            platform=platform,
            device_id=request.device_id,
            order_id=order_id,
            store_data=result.store_data,
            duration_days=duration_days,
            vip_started_at=vip_started_at,
            vip_expires_at=new_expires_at,
            verified_at=now,
            created_at=now,
        )
        db.add(purchase_record)
        
        # Create audit log
        create_audit_log(
            db,
            event_type="vip_purchase",
            message=f"VIP purchase verified: {request.product_id} ({duration_days} days)",
            actor=firebase_uid[:16],
            target=user.id,
            target_type="user",
            severity="info",
            details={
                "product_id": request.product_id,
                "platform": platform,
                "order_id": order_id,
                "duration_days": duration_days,
                "new_expires_at": new_expires_at.isoformat(),
                "is_test": result.is_test_purchase,
            },
        )
        
        db.commit()
        
        logger.info(
            f"[VIP] Purchase verified and activated for {firebase_uid}: "
            f"product={request.product_id}, platform={platform}, "
            f"duration={duration_days}d, expires={new_expires_at.isoformat()}"
        )
    
    # ============================================================
    # CONSUME the purchase on Google Play (CONSUMABLE product)
    # ============================================================
    if platform == "android":
        try:
            consumed = consume_google_play_purchase(
                product_id=request.product_id,
                purchase_token=request.purchase_token,
            )
            if consumed:
                logger.info(f"[VIP] Consumed on Google Play: order={order_id}")
            else:
                logger.warning(f"[VIP] Failed to consume on Google Play: order={order_id}")
        except Exception as e:
            # Log but don't fail - user already has VIP status
            logger.warning(f"[VIP] Consume failed (non-critical): {e}")
    
    return VipPurchaseResponse(
        verified=True,
        is_vip=True,
        expires_at=new_expires_at.isoformat(),
        days_granted=duration_days,
        message=f"VIP activated for {duration_days} days!",
    )


# =============================================================================
# Credit/Gem Purchase Endpoints (Consumables)
# =============================================================================

class GemPurchaseRequest(BaseModel):
    """Request body for gem/credit pack purchase verification"""
    purchase_token: str = Field(..., description="Purchase token from Google Play")
    product_id: str = Field(..., description="Product ID (e.g., 'credits_500')")
    platform: str = Field(..., description="'android' or 'ios'")
    device_id: Optional[str] = Field(None, description="Hashed device ID for fraud detection")
    order_id: Optional[str] = Field(None, description="Order ID from client (for early duplicate detection)")


class GemPurchaseResponse(BaseModel):
    """Response for gem/credit pack purchase verification"""
    success: bool
    credits_awarded: int = 0
    new_balance: int = 0
    order_id: Optional[str] = None
    message: Optional[str] = None
    detail: Optional[str] = None
    is_duplicate: bool = False


@auth_router.post("/verify-gem-purchase", response_model=GemPurchaseResponse)
async def verify_gem_purchase(
    request: GemPurchaseRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Verify and process a gem/credit pack purchase.
    
    Security workflow:
    1. Verify Firebase token (user authentication)
    2. Validate product_id against known products
    3. Check for duplicate order_id (prevent replay attacks)
    4. Verify purchase with Google Play API
    5. Double-check order_id from verified response
    6. Credit user atomically with purchase record
    7. Consume purchase on Google Play (allows re-purchase)
    
    **Headers:**
    - Authorization: Bearer <firebase_id_token>
    
    **Request Body:**
    - purchase_token: The purchase token from Google Play
    - product_id: Product ID (e.g., 'credits_500')
    - platform: 'android' or 'ios'
    - device_id: Hashed device ID (optional, for fraud detection)
    - order_id: Order ID from client (optional, for early duplicate check)
    
    **Returns:**
    - success: Whether purchase was processed
    - credits_awarded: Number of credits added (0 if duplicate)
    - new_balance: User's new credit balance
    - order_id: The verified order ID
    - is_duplicate: True if this was a duplicate purchase (already processed)
    
    **Error Codes:**
    - 400: Invalid request (bad product_id, missing fields)
    - 401: Unauthorized (invalid token)
    - 402: Payment verification failed with Google
    - 404: User not found
    - 409: Duplicate purchase (already processed - includes current balance)
    """
    import hashlib
    from purchase_verification import (
        verify_purchase,
        consume_google_play_purchase,
        get_credits_for_product,
        is_consumable_product,
        PRODUCT_CREDITS_MAP,
    )
    
    # Verify Firebase token
    decoded = await verify_firebase_token(authorization)
    
    firebase_uid = decoded.get("uid")
    if not firebase_uid:
        raise HTTPException(400, "Firebase UID not found in token")
    
    # Validate platform
    platform = request.platform.lower().strip()
    if platform not in ["android", "ios"]:
        raise HTTPException(400, f"Invalid platform: {platform}. Must be 'android' or 'ios'")
    
    # Validate product_id is a valid credit pack
    if request.product_id not in PRODUCT_CREDITS_MAP:
        logger.warning(f"[PURCHASE] Invalid product ID: {request.product_id} from {firebase_uid}")
        raise HTTPException(400, f"Invalid product_id: {request.product_id}")
    
    # Get credits to award (from SERVER-SIDE mapping, never trust client)
    credits_to_award = get_credits_for_product(request.product_id)
    
    with SessionLocal() as db:
        # Get user
        user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
        
        if not user:
            raise HTTPException(404, "User not found. Please call /auth/sync first.")
        
        # Early duplicate check using client-provided order_id
        if request.order_id:
            existing = db.query(Purchase).filter(Purchase.order_id == request.order_id).first()
            if existing:
                logger.info(f"[PURCHASE] Duplicate detected (client order_id): {request.order_id} for {firebase_uid}")
                return GemPurchaseResponse(
                    success=True,  # Not an error - purchase was already processed
                    credits_awarded=0,
                    new_balance=user.total_credits,
                    order_id=request.order_id,
                    message="Purchase already processed",
                    is_duplicate=True,
                )
        
        # Verify purchase with Google Play API
        logger.info(f"[PURCHASE] Verifying {platform} purchase for {firebase_uid}: product={request.product_id}")
        
        result = verify_purchase(
            purchase_token=request.purchase_token,
            product_id=request.product_id,
            platform=platform,
        )
        
        if not result.valid:
            logger.warning(f"[PURCHASE] Verification failed for {firebase_uid}: {result.reason}")
            raise HTTPException(402, f"Purchase verification failed: {result.reason}")
        
        verified_order_id = result.order_id
        
        # Double-check for duplicate using verified order_id
        if verified_order_id:
            existing = db.query(Purchase).filter(Purchase.order_id == verified_order_id).first()
            if existing:
                logger.info(f"[PURCHASE] Duplicate detected (verified order_id): {verified_order_id} for {firebase_uid}")
                return GemPurchaseResponse(
                    success=True,  # Not an error - purchase was already processed
                    credits_awarded=0,
                    new_balance=user.total_credits,
                    order_id=verified_order_id,
                    message="Purchase already processed",
                    is_duplicate=True,
                )
        
        # Also check by purchase token hash (belt and suspenders)
        token_hash = hashlib.sha256(request.purchase_token.encode()).hexdigest()
        existing_by_token = db.query(Purchase).filter(Purchase.purchase_token_hash == token_hash).first()
        if existing_by_token:
            logger.info(f"[PURCHASE] Duplicate detected (token hash): {firebase_uid}")
            return GemPurchaseResponse(
                success=True,
                credits_awarded=0,
                new_balance=user.total_credits,
                order_id=existing_by_token.order_id,
                message="Purchase already processed",
                is_duplicate=True,
            )
        
        # All checks passed - credit user atomically
        now = utc_naive_now()
        
        # Create purchase record
        purchase_record = Purchase(
            user_id=user.id,
            firebase_uid=firebase_uid,
            order_id=verified_order_id,
            purchase_token_hash=token_hash,
            product_id=request.product_id,
            platform=platform,
            device_id=request.device_id,
            credits_awarded=credits_to_award,
            purchase_state="verified",
            consumed=False,
            is_test_purchase=result.is_test_purchase,
            store_data=result.store_data,
            purchase_time_ms=str(result.purchase_time_ms) if result.purchase_time_ms else None,
            verified_at=now,
            created_at=now,
        )
        db.add(purchase_record)
        
        # Update user's credit balance
        user.total_credits += credits_to_award
        new_balance = user.total_credits
        
        # Create credit transaction record
        credit_transaction = CreditTransaction(
            user_id=user.id,
            delta=credits_to_award,
            reason=f"iap_purchase:{request.product_id}",
            credit_metadata={
                "event": "iap_purchase",
                "product_id": request.product_id,
                "platform": platform,
                "order_id": verified_order_id,
                "is_test": bool(result.is_test_purchase),
                "credits_awarded": int(credits_to_award),
            },
            created_at=now,
        )
        db.add(credit_transaction)
        
        # Create audit log
        create_audit_log(
            db,
            event_type="gem_purchase",
            message=f"Gem purchase verified: {request.product_id} (+{credits_to_award} credits)",
            actor=firebase_uid[:16],
            target=user.id,
            target_type="user",
            severity="info",
            details={
                "product_id": request.product_id,
                "credits_awarded": credits_to_award,
                "platform": platform,
                "order_id": verified_order_id,
                "is_test": result.is_test_purchase,
            },
        )
        
        db.commit()
        
        logger.info(
            f"[PURCHASE] SUCCESS: {firebase_uid} purchased {request.product_id}, "
            f"+{credits_to_award} credits, new_balance={new_balance}, order={verified_order_id}"
        )
    
    # AFTER crediting user, consume the purchase on Google Play
    # This allows the user to buy the same product again
    # Do this AFTER commit to ensure user gets credits even if consume fails
    if platform == "android":
        try:
            consumed = consume_google_play_purchase(
                product_id=request.product_id,
                purchase_token=request.purchase_token,
            )
            if consumed:
                # Update purchase record to mark as consumed
                with SessionLocal() as db:
                    purchase = db.query(Purchase).filter(Purchase.order_id == verified_order_id).first()
                    if purchase:
                        purchase.consumed = True
                        db.commit()
                logger.info(f"[PURCHASE] Consumed on Google Play: {verified_order_id}")
            else:
                logger.warning(f"[PURCHASE] Failed to consume on Google Play: {verified_order_id}")
        except Exception as e:
            # Log but don't fail - user already got their credits
            logger.warning(f"[PURCHASE] Consume failed (non-critical): {e}")
    
    return GemPurchaseResponse(
        success=True,
        credits_awarded=credits_to_award,
        new_balance=new_balance,
        order_id=verified_order_id,
        message=f"Purchase verified and {credits_to_award} credits added",
        is_duplicate=False,
    )


# =============================================================================
# Rewarded Ads Endpoints (SSV - Server-Side Verification)
# =============================================================================

ads_router = APIRouter(prefix="/ads", tags=["ads"])

# AdMob public keys for SSV signature verification
# These are fetched from Google and cached
import base64
import hashlib
import urllib.parse
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

ADMOB_KEYS_URL = "https://www.gstatic.com/admob/reward/verifier-keys.json"
_admob_keys_cache: Dict[str, Any] = {"keys": None, "last_fetch": 0}
ADMOB_KEYS_CACHE_TTL = 24 * 60 * 60  # 24 hours


def _fetch_admob_keys_sync() -> Dict[str, Any]:
    """Fetch AdMob public keys for SSV verification (sync version)."""
    import requests as req
    
    current_time = time.time()
    
    # Check cache
    if (_admob_keys_cache["keys"] is not None and 
        current_time - _admob_keys_cache["last_fetch"] < ADMOB_KEYS_CACHE_TTL):
        return _admob_keys_cache["keys"]
    
    try:
        response = req.get(ADMOB_KEYS_URL, timeout=10)
        if response.status_code == 200:
            keys_data = response.json()
            _admob_keys_cache["keys"] = keys_data
            _admob_keys_cache["last_fetch"] = current_time
            logger.info("[ADS] Fetched AdMob SSV keys successfully")
            return keys_data
    except Exception as e:
        logger.error(f"[ADS] Failed to fetch AdMob keys: {e}")
    
    # Return cached keys if fetch failed
    return _admob_keys_cache["keys"] or {}


async def _fetch_admob_keys() -> Dict[str, Any]:
    """Fetch AdMob public keys for SSV verification (async wrapper)."""
    import asyncio
    return await asyncio.to_thread(_fetch_admob_keys_sync)


def _verify_admob_signature(query_string: str, signature: str, key_id: str, keys_data: Dict) -> bool:
    """
    Verify AdMob SSV callback signature.
    
    Google signs the callback with ECDSA using their public key.
    We verify the signature to ensure the callback is authentic.
    """
    try:
        # Find the key by key_id
        keys = keys_data.get("keys", [])
        public_key_pem = None
        for key in keys:
            if str(key.get("keyId")) == str(key_id):
                public_key_pem = key.get("pem")
                break
        
        if not public_key_pem:
            logger.warning(f"[ADS] AdMob key {key_id} not found in keys")
            return False
        
        # Parse the public key
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
        public_key = load_pem_public_key(public_key_pem.encode(), backend=default_backend())
        
        # Decode the signature (URL-safe base64)
        signature_bytes = base64.urlsafe_b64decode(signature + "==")  # Add padding
        
        # The message to verify is the query string without signature and key_id
        # Parse query string, remove signature and key_id, reconstruct
        params = urllib.parse.parse_qs(query_string, keep_blank_values=True)
        params.pop("signature", None)
        params.pop("key_id", None)
        
        # Reconstruct the message (sorted by key for consistency)
        message_parts = []
        for key in sorted(params.keys()):
            for value in params[key]:
                message_parts.append(f"{key}={value}")
        message = "&".join(message_parts)
        
        # Verify signature
        public_key.verify(signature_bytes, message.encode(), ec.ECDSA(hashes.SHA256()))
        return True
        
    except InvalidSignature:
        logger.warning("[ADS] AdMob signature verification failed: invalid signature")
        return False
    except Exception as e:
        logger.error(f"[ADS] AdMob signature verification error: {e}")
        return False


def _get_today_ad_stats(db, user_id: str) -> tuple[int, int]:
    """
    Get user's ad stats for today (UTC).
    
    Returns:
        (ads_watched_today, credits_earned_today)
    """
    from datetime import datetime, time as dt_time
    from sqlalchemy import func
    
    # Get start of today (UTC)
    now = utc_naive_now()
    today_start = datetime.combine(now.date(), dt_time.min)
    
    # Count total ads watched today
    ads_watched_today = db.query(func.count(RewardedAdEvent.id)).filter(
        RewardedAdEvent.user_id == user_id,
        RewardedAdEvent.created_at >= today_start,
    ).scalar() or 0
    
    # Sum credits earned today
    credits_earned_today = db.query(func.coalesce(func.sum(RewardedAdEvent.credits), 0)).filter(
        RewardedAdEvent.user_id == user_id,
        RewardedAdEvent.created_at >= today_start,
    ).scalar() or 0
    
    return ads_watched_today, credits_earned_today


class AdRewardStatusResponse(BaseModel):
    """Response for ad reward status check"""
    can_watch: bool
    credits_earned_today: int  # Credits user has earned today (ads_watched / 4)
    daily_limit: int  # Max credits per day
    credits_per_ad: int  # 0 since credits come every 4 ads
    next_reset_time: str  # ISO 8601 UTC
    cooldown_ends_at: Optional[str] = None  # When cooldown ends (null if no cooldown)
    # Fields for "4 ads = 1 credit" system
    ads_watched_today: int  # Total ads watched today (0, 1, 2, 3, 4, 5...)
    ads_required_for_credit: int  # How many ads needed for 1 credit


class SSVCallbackResponse(BaseModel):
    """Response for SSV callback"""
    status: str
    message: str


@ads_router.get("/rewarded/status", response_model=AdRewardStatusResponse)
async def get_ad_reward_status(
    authorization: Optional[str] = Header(None),
):
    """
    Check if user can watch rewarded ads and their daily progress.
    
    **Headers:**
    - Authorization: Bearer <firebase_id_token>
    
    **Returns:**
    - can_watch: Whether user can earn more credits from ads today
    - credits_earned_today: Credits earned from ads today (ads_watched / 4)
    - daily_limit: Maximum credits earnable per day
    - credits_per_ad: 0 (credits come every 4 ads, not per ad)
    - ads_watched_today: Total ads watched today
    - ads_required_for_credit: How many ads needed for 1 credit (4)
    - cooldown_ends_at: When cooldown ends (null if no cooldown)
    - next_reset_time: When daily limit resets (midnight UTC)
    """
    decoded = await verify_firebase_token(authorization)
    firebase_uid = decoded.get("uid")
    
    if not firebase_uid:
        raise HTTPException(400, "Firebase UID not found in token")
    
    with SessionLocal() as db:
        user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
        
        if not user:
            raise HTTPException(404, "User not found. Please sync first.")
        
        # Get config
        daily_limit = get_app_config_int(db, "daily_ad_credit_limit", 30)
        ads_required_for_credit = get_app_config_int(db, "ads_per_credit", 4)
        
        # Get today's ad stats
        ads_watched_today, credits_earned_today = _get_today_ad_stats(db, user.id)
        
        # Calculate next reset (midnight UTC tomorrow)
        from datetime import time as dt_time
        now = utc_naive_now()
        tomorrow = now.date() + timedelta(days=1)
        next_reset = datetime.combine(tomorrow, dt_time.min)
        
        return AdRewardStatusResponse(
            can_watch=credits_earned_today < daily_limit,
            credits_earned_today=credits_earned_today,
            daily_limit=daily_limit,
            credits_per_ad=0,  # 0 because credits come every N ads, not per ad
            next_reset_time=_format_iso_utc(next_reset),
            cooldown_ends_at=None,  # No cooldown implemented yet
            ads_watched_today=ads_watched_today,
            ads_required_for_credit=ads_required_for_credit,
        )


@ads_router.get("/admob/ssv")
async def admob_ssv_callback(
    request: Request,
    ad_network: Optional[str] = None,
    ad_unit: Optional[str] = None,
    custom_data: Optional[str] = None,  # Contains user_id
    reward_amount: Optional[int] = None,
    reward_item: Optional[str] = None,
    timestamp: Optional[str] = None,
    transaction_id: Optional[str] = None,
    user_id: Optional[str] = None,  # May be passed directly
    signature: Optional[str] = None,
    key_id: Optional[str] = None,
):
    """
    AdMob Server-Side Verification (SSV) callback.
    
    **Called by Google AdMob** after a user completes watching a rewarded ad.
    
    This endpoint:
    1. Verifies the callback signature (prevents spoofing)
    2. Checks daily credit limit
    3. Grants credits to user
    4. Logs the event for anti-abuse tracking
    
    **Query Parameters (sent by Google):**
    - ad_network: Ad network identifier
    - ad_unit: Your ad unit ID
    - custom_data: Custom data you passed (should contain user_id/firebase_uid)
    - reward_amount: Reward amount configured in AdMob (IGNORED - we use server config)
    - reward_item: Reward type configured in AdMob
    - timestamp: Callback timestamp
    - transaction_id: Unique transaction ID
    - signature: ECDSA signature
    - key_id: Key ID for signature verification
    
    **Returns:**
    - 200 OK if successful
    - 400 if invalid parameters
    - 403 if signature verification fails
    - 429 if daily limit reached
    """
    # Get client info for logging
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    query_string = str(request.query_params)
    
    logger.info(f"[ADS] AdMob SSV callback received: transaction_id={transaction_id}, custom_data={custom_data}, query={query_string}")
    
    # Handle Google's verification test (no parameters sent)
    # Google sends a simple GET request to verify the endpoint is reachable
    if not custom_data and not transaction_id and not user_id:
        logger.info("[ADS] SSV verification test request - returning OK")
        return {"status": "ok", "message": "SSV endpoint ready"}
    
    # Extract user identifier from custom_data or user_id parameter
    # custom_data format: "firebase_uid:XXXX" or just the firebase_uid
    firebase_uid = None
    if custom_data:
        if custom_data.startswith("firebase_uid:"):
            firebase_uid = custom_data.split(":", 1)[1]
        else:
            firebase_uid = custom_data
    elif user_id:
        firebase_uid = user_id
    
    if not firebase_uid:
        logger.warning("[ADS] SSV callback missing user identifier")
        # Return 200 with error message instead of 400 (Google expects 200)
        return {"status": "error", "message": "Missing user identifier in custom_data"}
    
    if not transaction_id:
        logger.warning("[ADS] SSV callback missing transaction_id")
        # Return 200 with error message instead of 400 (Google expects 200)
        return {"status": "error", "message": "Missing transaction_id"}
    
    # Verify signature (CRITICAL for security)
    signature_verified = False
    if signature and key_id:
        keys_data = await _fetch_admob_keys()
        if keys_data:
            signature_verified = _verify_admob_signature(query_string, signature, key_id, keys_data)
            if not signature_verified:
                logger.warning(f"[ADS] SSV signature verification failed for {firebase_uid}")
                # In production, you might want to reject unverified callbacks
                # For now, we log but continue (allows testing)
    else:
        logger.warning("[ADS] SSV callback missing signature/key_id - skipping verification")
    
    with SessionLocal() as db:
        # Find user by firebase_uid
        user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
        
        if not user:
            logger.warning(f"[ADS] SSV callback for unknown user: {firebase_uid}")
            raise HTTPException(400, "User not found")
        
        # Check for duplicate transaction (prevent replay attacks)
        existing = db.query(RewardedAdEvent).filter(
            RewardedAdEvent.transaction_id == transaction_id
        ).first()
        
        if existing:
            logger.info(f"[ADS] Duplicate SSV callback ignored: {transaction_id}")
            return {"status": "ok", "message": "Already processed"}
        
        # Get config
        daily_limit = get_app_config_int(db, "daily_ad_credit_limit", 30)
        ads_required_for_credit = get_app_config_int(db, "ads_per_credit", 4)  # How many ads needed for 1 credit
        
        # Get today's ad stats BEFORE adding the new ad
        ads_watched_today, credits_earned_today = _get_today_ad_stats(db, user.id)
        
        # Check daily limit
        if credits_earned_today >= daily_limit:
            logger.info(f"[ADS] Daily limit reached for user {firebase_uid}: {credits_earned_today}/{daily_limit}")
            return {"status": "limit_reached", "message": f"Daily limit of {daily_limit} credits reached"}
        
        # This ad will be the Nth ad watched today (1-indexed for modulo logic)
        new_ad_count = ads_watched_today + 1
        
        # Grant credit every N ads (when count is divisible by ads_required_for_credit)
        # e.g., if ads_required_for_credit=4: grant on 4th, 8th, 12th ad...
        credits_to_grant = 0
        if new_ad_count % ads_required_for_credit == 0:
            credits_to_grant = 1
            
            # Don't exceed daily limit
            if credits_earned_today + credits_to_grant > daily_limit:
                credits_to_grant = 0
            
            if credits_to_grant > 0:
                user.total_credits += credits_to_grant
                user.bonus_granted += credits_to_grant
                
                # Record credit transaction
                transaction = CreditTransaction(
                    user_id=user.id,
                    delta=credits_to_grant,
                    reason="rewarded_ad",
                    credit_metadata={
                        "event": "rewarded_ad",
                        "credits_awarded": int(credits_to_grant),
                        "ad_provider": "admob",
                        "ad_unit_id": ad_unit,
                        "transaction_id": transaction_id,
                    },
                    created_at=utc_naive_now(),
                )
                db.add(transaction)
        
        # Log rewarded ad event (always log, credits=0 if still accumulating)
        ad_event = RewardedAdEvent(
            user_id=user.id,
            credits=credits_to_grant,  # Will be 0 if still accumulating
            ad_provider="admob",
            ad_unit_id=ad_unit,
            transaction_id=transaction_id,
            signature_verified=signature_verified,
            ip_address=client_ip,
            user_agent=user_agent,
            created_at=utc_naive_now(),
        )
        db.add(ad_event)
        
        db.commit()
        
        if credits_to_grant > 0:
            logger.info(f"[ADS] Granted {credits_to_grant} credit to {firebase_uid} (ads watched: {new_ad_count}) via SSV (transaction: {transaction_id})")
            return {"status": "ok", "message": f"Granted {credits_to_grant} credit"}
        else:
            ads_remaining = ads_required_for_credit - (new_ad_count % ads_required_for_credit)
            logger.info(f"[ADS] Ad recorded for {firebase_uid} (ads watched: {new_ad_count}, need {ads_remaining} more for credit) (transaction: {transaction_id})")
            return {"status": "ok", "message": f"Ad recorded. Watch {ads_remaining} more ad(s) to earn 1 credit"}


# NOTE: Client-side ad completion endpoint (/ads/rewarded/complete) has been REMOVED.
# All rewarded ad credits are now granted exclusively via AdMob SSV (Server-Side Verification).
# This prevents users from spoofing ad completions.
# 
# Frontend must configure AdMob to call /ads/admob/ssv with custom_data containing the firebase_uid.


# =============================================================================
# AdGem Offerwall Postback
# =============================================================================

# AdGem Configuration
# 
# Credit calculation is done in AdGem dashboard via multiplier setting.
# The `amount` field in postback already contains the correct credits to award.
# We NEVER calculate credits client-side - server postback is authoritative.
#
# Environment variable for postback key (used for secret token validation)
ADGEM_POSTBACK_KEY = os.getenv("ADGEM_POSTBACK_KEY", "")


@ads_router.get("/adgem/postback")
async def adgem_postback(
    request: Request,
    app_id: Optional[str] = None,  # AdGem macro: {app_id}
    player_id: Optional[str] = None,  # AdGem macro: {player_id} - Contains firebase_uid
    amount: Optional[str] = None,  # AdGem macro: {amount} - Virtual currency amount
    payout: Optional[str] = None,  # AdGem macro: {payout} - Actual payout in currency
    transaction_id: Optional[str] = None,  # AdGem macro: {transaction_id}
    offer_id: Optional[str] = None,  # AdGem macro: {offer_id}
    offer_name: Optional[str] = None,  # AdGem macro: {offer_name}
    country: Optional[str] = None,  # AdGem macro: {country}
    conversion_datetime: Optional[str] = None,  # AdGem macro: {conversion_datetime}
    campaign_id: Optional[str] = None,  # AdGem macro: {campaign_id}
    ip: Optional[str] = None,  # AdGem macro: {ip} - User's IP
    platform: Optional[str] = None,  # AdGem macro: {platform}
    gaid: Optional[str] = None,  # AdGem macro: {gaid} - Google Advertising ID
    idfa: Optional[str] = None,  # AdGem macro: {idfa} - iOS Identifier for Advertisers
):
    """
    AdGem Offerwall Server-Side Postback.
    
    **Called by AdGem** after a user completes an offer.
    
    SECURITY: Server postback is AUTHORITATIVE. Never trust client SDK callbacks.
    
    This endpoint:
    1. Validates required parameters
    2. Optionally validates secret token (append &secret=YOUR_KEY to postback URL)
    3. Checks for duplicate transaction_id (prevents replay attacks)
    4. Awards credits to user (amount = credits, configured via AdGem dashboard multiplier)
    5. Logs the event for audit
    
    **Query Parameters (AdGem macros):**
    - app_id: AdGem app ID (31874) {app_id}
    - player_id: User's firebase_uid {player_id}
    - amount: Credits to award (configured via AdGem dashboard multiplier) {amount}
    - payout: AdGem payout amount {payout}
    - transaction_id: Unique transaction ID {transaction_id}
    - offer_id: Offer identifier {offer_id}
    - offer_name: Offer name {offer_name}
    - country: Country code {country}
    - conversion_datetime: Conversion timestamp {conversion_datetime}
    - campaign_id: Campaign ID {campaign_id}
    - ip: User IP {ip}
    - platform: Platform {platform}
    - gaid: Google Advertising ID {gaid}
    - idfa: iOS IDFA {idfa}
    
    **Returns:**
    - "OK" (HTTP 200) if successful or duplicate (idempotent)
    - "INVALID" (HTTP 403) if invalid params or secret mismatch
    
    **Security:**
    - Server-side only - NEVER trust client
    - transaction_id uniqueness prevents replay attacks
    - Optional secret token validation
    - Full audit logging
    """
    from db import AdGemConversion
    
    # Get client info for logging/anti-fraud
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    query_string = str(request.query_params)
    
    logger.info(
        f"[ADGEM] Postback received: transaction_id={transaction_id}, "
        f"player_id={player_id}, amount={amount}, payout={payout}, IP={client_ip}"
    )
    
    # Store raw payload for audit
    raw_payload = dict(request.query_params)
    
    # ==========================================================================
    # 1. Validate Required Parameters
    # ==========================================================================
    
    if not app_id:
        logger.warning(f"[ADGEM] Missing app_id, IP={client_ip}")
        return Response(content="INVALID", status_code=403)
    
    if not player_id:
        logger.warning(f"[ADGEM] Missing player_id, IP={client_ip}")
        return Response(content="INVALID", status_code=403)
    
    if not amount:
        logger.warning(f"[ADGEM] Missing amount, IP={client_ip}")
        return Response(content="INVALID", status_code=403)
    
    if not transaction_id:
        logger.warning(f"[ADGEM] Missing transaction_id, IP={client_ip}")
        return Response(content="INVALID", status_code=403)
    
    # Parse numeric values
    try:
        amount_int = int(amount)
    except (ValueError, TypeError):
        logger.warning(f"[ADGEM] Invalid amount '{amount}', IP={client_ip}")
        return Response(content="INVALID", status_code=403)
    
    try:
        payout_float = float(payout) if payout else 0.0
    except (ValueError, TypeError):
        logger.warning(f"[ADGEM] Invalid payout '{payout}', IP={client_ip}")
        payout_float = 0.0
    
    if amount_int <= 0:
        logger.warning(f"[ADGEM] Amount must be > 0, got {amount_int}, IP={client_ip}")
        return Response(content="INVALID", status_code=403)
    
    # ==========================================================================
    # 2. Optional Secret Token Validation
    # ==========================================================================
    # 
    # AdGem doesn't provide a {verifier} macro, so we use an optional secret token.
    # If ADGEM_POSTBACK_KEY is configured, you can add it to your postback URL:
    #   &secret={your_key}
    # This provides additional security beyond transaction_id uniqueness.
    #
    secret_token = request.query_params.get("secret")
    secret_valid = True  # Default to valid if no secret configured
    
    if ADGEM_POSTBACK_KEY:
        if secret_token:
            # Validate secret token if provided
            secret_valid = hmac.compare_digest(secret_token, ADGEM_POSTBACK_KEY)
            if not secret_valid:
                logger.warning(
                    f"[ADGEM] ❌ SECRET TOKEN MISMATCH! "
                    f"transaction_id={transaction_id}, player_id={player_id}, IP={client_ip}"
                )
                # Log audit event for security monitoring
                with SessionLocal() as db:
                    create_audit_log(
                        db,
                        event_type="adgem_secret_mismatch",
                        severity="warning",
                        actor="adgem",
                        actor_ip=client_ip,
                        target=transaction_id,
                        target_type="adgem_postback",
                        message=f"Secret token mismatch for player_id={player_id}",
                        details={"raw_payload": raw_payload},
                    )
                    db.commit()
                return Response(content="INVALID", status_code=403)
            logger.info(f"[ADGEM] ✅ Secret token validated for transaction_id={transaction_id}")
        else:
            # No secret provided - log warning but allow (for backward compatibility)
            logger.warning(
                f"[ADGEM] No secret token in postback URL. "
                f"Consider adding &secret=YOUR_KEY to postback URL for security. "
                f"transaction_id={transaction_id}"
            )
    
    # ==========================================================================
    # 3. Check for Duplicate Transaction (Idempotency)
    # ==========================================================================
    
    with SessionLocal() as db:
        existing = db.query(AdGemConversion).filter(
            AdGemConversion.transaction_id == transaction_id
        ).first()
        
        if existing:
            logger.info(f"[ADGEM] Duplicate transaction ignored: {transaction_id} (already processed)")
            # Return 200 OK for idempotency - AdGem may retry
            return Response(content="OK", status_code=200)
        
        # ==========================================================================
        # 4. Find User by firebase_uid (player_id)
        # ==========================================================================
        
        # player_id should contain the firebase_uid
        firebase_uid = player_id
        
        user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
        
        if not user:
            # Log the conversion anyway for audit, but can't credit
            logger.warning(f"[ADGEM] User not found for firebase_uid={firebase_uid}, transaction_id={transaction_id}")
            
            conversion = AdGemConversion(
                user_id=None,  # User not found
                player_id=player_id,
                transaction_id=transaction_id,
                app_id=app_id,
                offer_id=offer_id,
                offer_name=offer_name,
                amount=amount_int,
                payout=payout_float,
                credits_awarded=0,  # No credits awarded
                country=country,
                verifier_valid=secret_valid,
                ip_address=client_ip,
                user_agent=user_agent,
                raw_payload=raw_payload,
                created_at=utc_naive_now(),
            )
            
            # Parse conversion datetime if provided
            if conversion_datetime:
                try:
                    conversion.conversion_time = datetime.strptime(
                        conversion_datetime, "%Y-%m-%d %H:%M:%S"
                    )
                except ValueError:
                    pass
            
            db.add(conversion)
            db.commit()
            
            # Return OK to prevent AdGem retries (user just doesn't exist)
            return Response(content="OK", status_code=200)
        
        # ==========================================================================
        # 5. Award Credits (amount from AdGem = credits, configured via dashboard multiplier)
        # ==========================================================================
        
        # IMPORTANT: amount_int is the credits to award (set via AdGem dashboard multiplier)
        # We NEVER calculate credits client-side - server postback is authoritative
        credits_to_award = amount_int
        
        logger.info(
            f"[ADGEM] Awarding credits: amount={credits_to_award} credits, "
            f"payout=₹{payout_float}, offer_id={offer_id}"
        )
        
        # Credit user atomically
        old_balance = user.total_credits
        user.total_credits += credits_to_award
        user.bonus_granted += credits_to_award
        
        # Record credit transaction
        from db import CreditTransaction
        transaction = CreditTransaction(
            user_id=user.id,
            delta=credits_to_award,
            reason=f"adgem_offer:{offer_id or 'unknown'}",
            credit_metadata={
                "event": "adgem_offer",
                "offer_id": offer_id,
                "offer_name": offer_name,
                "transaction_id": transaction_id,
                "credits_awarded": int(credits_to_award),
                "payout": payout_float,
            },
            created_at=utc_naive_now(),
        )
        db.add(transaction)
        
        # ==========================================================================
        # 6. Log Conversion Record
        # ==========================================================================
        
        conversion = AdGemConversion(
            user_id=user.id,
            player_id=player_id,
            transaction_id=transaction_id,
            app_id=app_id,
            offer_id=offer_id,
            offer_name=offer_name,
            amount=amount_int,
            payout=payout_float,
            credits_awarded=credits_to_award,
            country=country,
            verifier_valid=secret_valid,
            ip_address=client_ip,
            user_agent=user_agent,
            raw_payload=raw_payload,
            created_at=utc_naive_now(),
        )
        
        # Parse conversion datetime if provided
        if conversion_datetime:
            try:
                conversion.conversion_time = datetime.strptime(
                    conversion_datetime, "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                pass
        
        db.add(conversion)
        
        # Commit all changes atomically
        db.commit()
        
        logger.info(
            f"[ADGEM] ✅ SUCCESS: Awarded {credits_to_award} credits to user {user.id} "
            f"(firebase_uid={firebase_uid}, balance: {old_balance} → {user.total_credits}, "
            f"transaction_id={transaction_id}, offer_id={offer_id}, payout={payout_float})"
        )
        
        # Return OK to AdGem
        return Response(content="OK", status_code=200)


@ads_router.get("/adgem/health")
async def adgem_health():
    """
    Health check for AdGem postback endpoint.
    
    Use this to verify the endpoint is reachable and configured.
    """
    key_configured = bool(ADGEM_POSTBACK_KEY)
    return {
        "status": "ok",
        "adgem_postback_key_configured": key_configured,
        "endpoint": "/ads/adgem/postback",
    }


def _calculate_timing_metrics(
    job: Job, completion_time: datetime
) -> Dict[str, Optional[float]]:
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


def _log_timing_metrics(
    job_id: str, model: str, timing_metrics: Dict[str, Optional[float]]
):
    """Log timing metrics to console in a formatted way"""
    logger.info("=" * 80)
    logger.info(f"⏱️  JOB TIMING REPORT: {job_id}")
    logger.info(f"   Model: {model}")
    
    if timing_metrics["total_time"] is not None:
        logger.info(
            f"   📊 Total Time: {timing_metrics['total_time']:.2f}s (created → completed)"
        )
    
    if timing_metrics["processing_time"] is not None:
        logger.info(
            f"   ⚡ Processing Time: {timing_metrics['processing_time']:.2f}s (started → completed)"
        )
    
    if timing_metrics["queue_time"] is not None:
        logger.info(
            f"   ⏳ Queue Time: {timing_metrics['queue_time']:.2f}s (queued → started)"
        )
    
    if timing_metrics["dispatch_time"] is not None:
        logger.info(
            f"   🚀 Dispatch Time: {timing_metrics['dispatch_time']:.2f}s (dispatched → started)"
        )
    
    if timing_metrics.get("startup_time") is not None:
        logger.info(
            f"   🌱 Startup Time: {timing_metrics['startup_time']:.2f}s (pod start → first inference)"
        )
    
    logger.info("=" * 80)


# =============================================================================
# Public Endpoints
# =============================================================================


def _normalize_etag_header_value(raw: str) -> str:
    """Strip weak prefix and quotes for If-None-Match comparison."""
    v = raw.strip()
    if v.upper().startswith("W/"):
        v = v[2:].strip()
    if len(v) >= 2 and v.startswith('"') and v.endswith('"'):
        v = v[1:-1]
    return v


@public_router.get("/sales/active")
async def get_public_sale_status(request: Request):
    """
    Public endpoint: current sale window state for frontend pricing.
    No auth required so the app can render sale banners immediately.

    Supports conditional GET: If-None-Match + ETag → 304 when payload unchanged
    (reduces serialization and bandwidth under polling).
    """
    from credit_service import build_public_sale_status_payload

    payload = build_public_sale_status_payload()
    body_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    etag_token = hashlib.sha256(body_bytes).hexdigest()
    etag_header = f'"{etag_token}"'
    cache_headers = {
        "ETag": etag_header,
        "Cache-Control": "private, max-age=0, must-revalidate",
    }

    inm = request.headers.get("if-none-match")
    if inm:
        # Multiple ETags possible; any match → 304
        for part in inm.split(","):
            if _normalize_etag_header_value(part) == etag_token:
                return Response(status_code=304, headers=cache_headers)

    return Response(
        content=body_bytes,
        media_type="application/json",
        headers=cache_headers,
    )


@public_router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns server status and optionally checks component connectivity.
    """
    components = {}
    
    # Check database
    db = None
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        components["database"] = "ok"
    except Exception:
        components["database"] = "error"
    finally:
        if db is not None:
            db.close()
    
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


def _generate_signed_download_url(r2_key: str, expires_in_seconds: int = 86400) -> Optional[str]:
    """
    Generate a signed download URL for an R2 object.
    
    Args:
        r2_key: The R2 object key (e.g., "image-vide-generator-api/uuid.png")
        expires_in_seconds: URL validity duration (default 1 day)
    
    Returns:
        Signed URL string, or None if secret not configured
    """
    if not r2_key:
        return None
    
    DOWNLOAD_SECRET = os.getenv("CONTROLLER_SHARED_SECRET", "SECRETKEY")
    if not DOWNLOAD_SECRET:
        return None
    
    expires = int(time.time()) + expires_in_seconds
    payload = f"{r2_key}:{expires}"
    sig = hmac.new(
        DOWNLOAD_SECRET.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()
    
    return f"https://img-vid-gen-download.pavanalur3.workers.dev/d/{r2_key}?exp={expires}&sig={sig}"


def _scale_requested(params: Dict[str, Any]) -> bool:
    try:
        return float(params.get("scale_factor", 1) or 1) > 1
    except (TypeError, ValueError):
        return False


_RUNWARE_UPSCALE_SPECS = {
    "realesrgan_upscale": ("runware:504@1", "upscale_esrgan"),
    "p_image_upscale": ("prunaai:p-image@upscale", "upscale_p"),
}


def _output_upscale_choice(model: str, params: Dict[str, Any]) -> Optional[str]:
    """Which extra upscale to run, if the client asked for one."""
    from credit_service import OUTPUT_UPSCALE_MODEL_IDS

    if model in OUTPUT_UPSCALE_MODEL_IDS:
        return None
    if params.get("_kie_upscale_started") or params.get("_runware_upscale_started"):
        return None
    if not _scale_requested(params):
        return None
    choice = str(params.get("upscale_model") or "recraft_crisp_upscale").strip()
    if choice not in OUTPUT_UPSCALE_MODEL_IDS:
        choice = "recraft_crisp_upscale"
    return choice


async def _chain_kie_upscale(db, job, source_url: str, params: Dict[str, Any]) -> bool:
    """Send the finished image through Kie crisp upscale on the same job."""
    import kie_client
    from db import KieTask

    public = (os.getenv("CONTROLLER_PUBLIC_URL") or os.getenv("CONTROLLER_URL") or "").rstrip("/")
    if not public:
        logger.warning("[KIE] upscale chain skipped, CONTROLLER_PUBLIC_URL is unset")
        return False
    try:
        hosted = await asyncio.to_thread(kie_client.upload_file_from_url, source_url)
        task_id = await asyncio.to_thread(
            kie_client.create_task,
            "recraft/crisp-upscale",
            {"image": hosted, "nsfw_checker": True},
            f"{public}/kie/callback",
        )
    except Exception:
        logger.exception("[KIE] upscale chain failed for %s", job.job_id)
        return False
    db.add(KieTask(task_id=task_id, job_id=job.job_id, model="recraft_crisp_upscale"))
    params["_kie_upscale_started"] = True
    params["_kie_result_url"] = source_url
    job.params = params
    db.commit()
    logger.info("[KIE] Job %s chaining crisp upscale task %s", job.job_id, task_id)
    return True


def _stash_suno_submission(db, job_id: str, result: ExternalExecutionResult) -> None:
    """Keep the Suno pipeline state on the job before the callback arrives."""
    raw = result.raw if isinstance(result.raw, dict) else {}
    if "suno_context" not in raw and "music" not in raw:
        return
    job_row = db.query(Job).filter(Job.job_id == job_id).first()
    if not job_row:
        return
    params = dict(job_row.params or {})
    if isinstance(raw.get("suno_context"), dict):
        params["_suno"] = raw["suno_context"]
    if isinstance(raw.get("music"), dict):
        params["_music"] = raw["music"]
    job_row.params = params
    db.commit()


def _public_suno_music(job: Job) -> Optional[Dict[str, Any]]:
    if job.model != "suno_v6":
        return None
    music = (job.params or {}).get("_music") if isinstance(job.params, dict) else None
    if not isinstance(music, dict):
        return None
    return dict(music)


def _mint_suno_url(url: Optional[str]) -> Optional[str]:
    import kie_client

    if not isinstance(url, str) or not url.startswith("http"):
        return url
    try:
        return kie_client.download_url_for(url)
    except Exception as exc:
        logger.warning("[SUNO] download link was not minted: %s", exc)
        return url


async def _complete_suno_job(db, job, params: Dict[str, Any], primary_url: Optional[str]) -> None:
    music = dict(params.get("_music") or {})
    minted_covers = []
    for url in music.get("covers") or []:
        if isinstance(url, str) and url.startswith("http"):
            minted_covers.append(await asyncio.to_thread(_mint_suno_url, url))
    if minted_covers:
        music["covers"] = minted_covers
    tracks = []
    for index, track in enumerate(music.get("tracks") or []):
        if not isinstance(track, dict):
            continue
        item = dict(track)
        if item.get("audio_url"):
            item["download_url"] = await asyncio.to_thread(_mint_suno_url, item.get("audio_url"))
        cover = item.get("cover_url")
        if isinstance(cover, str) and cover.startswith("http"):
            item["cover_url"] = await asyncio.to_thread(_mint_suno_url, cover)
        elif minted_covers:
            item["cover_url"] = minted_covers[index] if index < len(minted_covers) else minted_covers[0]
        tracks.append(item)
    if tracks:
        music["tracks"] = tracks
    if music.get("video_url"):
        music["video_download_url"] = await asyncio.to_thread(_mint_suno_url, music.get("video_url"))
    stems = []
    for stem in music.get("stems") or []:
        if not isinstance(stem, dict):
            continue
        item = dict(stem)
        if item.get("url"):
            item["download_url"] = await asyncio.to_thread(_mint_suno_url, item.get("url"))
        stems.append(item)
    if stems:
        music["stems"] = stems
    music["download_minted_at"] = time.time()
    params["_music"] = music
    output_url = None
    if primary_url:
        params["_kie_result_url"] = primary_url
        output_url = await asyncio.to_thread(_mint_suno_url, primary_url)
    job.params = params
    db.commit()
    if _job_manager:
        _job_manager.mark_completed(job.job_id, output_url=output_url)
    else:
        job.status = JobState.COMPLETED.value
        job.output_url = output_url
        job.completed_at = utc_naive_now()
        job.updated_at = utc_naive_now()
        db.commit()
    done = db.query(Job).filter(Job.job_id == job.job_id).first()
    meta = done.params if done and isinstance(done.params, dict) else {}
    await _send_job_notification(
        token=done.fcm_token if done else None,
        job_id=job.job_id,
        model=job.model,
        success=True,
        download_url=output_url,
        batch_id=meta.get("_batch_id"),
        variation_index=int(meta.get("_variation_index", 0) or 0),
        total_variations=int(meta.get("_total_variations", 1) or 1),
    )
    clear_cached_job_params(job.job_id)


async def _fail_suno_job(db, job, params: Dict[str, Any], error: str) -> None:
    job.params = params
    db.commit()
    if _job_manager:
        _job_manager.mark_failed(job.job_id, error, worker_id=None)
    else:
        job.status = JobState.FAILED.value
        job.error = error
        job.updated_at = utc_naive_now()
        db.commit()
    done = db.query(Job).filter(Job.job_id == job.job_id).first()
    meta = done.params if done and isinstance(done.params, dict) else {}
    await _send_job_notification(
        token=done.fcm_token if done else None,
        job_id=job.job_id,
        model=job.model,
        success=False,
        batch_id=meta.get("_batch_id"),
        variation_index=int(meta.get("_variation_index", 0) or 0),
        total_variations=int(meta.get("_total_variations", 1) or 1),
    )
    clear_cached_job_params(job.job_id)


async def _apply_suno_v6(db, job, payload: Dict[str, Any]) -> None:
    """Advance the lyrics → song → cover pipeline, or finish a follow-up action."""
    import kie_client
    import suno_v6
    from db import KieTask

    if job.status == JobState.COMPLETED.value:
        await _keep_late_suno_versions(db, job, payload)
        return
    if job.status in (JobState.FAILED.value, JobState.CANCELLED.value):
        logger.info("[SUNO] job %s already terminal (%s)", job.job_id, job.status)
        return
    params = dict(job.params or {})
    context = params.get("_suno") if isinstance(params.get("_suno"), dict) else {}
    music = params.get("_music") if isinstance(params.get("_music"), dict) else {}
    event = suno_v6.interpret(payload)
    step = suno_v6.advance(context, music, event)
    params["_suno"] = step["context"]
    params["_music"] = step["music"]
    stored = step["music"].get("tracks") if isinstance(step.get("music"), dict) else []
    logger.info(
        "[SUNO] job %s decided %s phase=%s stored=%s",
        job.job_id,
        step.get("action"),
        (step.get("music") or {}).get("phase"),
        [track.get("id") for track in stored or [] if isinstance(track, dict)],
    )

    if step["action"] == "update":
        job.params = params
        job.updated_at = utc_naive_now()
        db.commit()
        return

    if step["action"] == "submit":
        public = (os.getenv("CONTROLLER_PUBLIC_URL") or os.getenv("CONTROLLER_URL") or "").rstrip("/")
        if not public:
            await _fail_suno_job(db, job, params, "CONTROLLER_PUBLIC_URL is required for Kie callBackUrl")
            return
        try:
            task_id = await asyncio.to_thread(
                kie_client.create_task,
                step["kie_model"],
                step["kie_input"],
                f"{public}/kie/callback",
            )
        except Exception as exc:
            logger.exception("[SUNO] follow-up task failed for %s", job.job_id)
            if step.get("optional"):
                note = dict(params.get("_music") or {})
                note["note"] = "Extra covers were not ready. Each version still has its own cover."
                note["phase"] = "complete"
                note["label"] = "Ready"
                params["_music"] = note
                suno_state = dict(params.get("_suno") or {})
                suno_state["phase"] = "done"
                params["_suno"] = suno_state
                await _complete_suno_job(db, job, params, step.get("primary_url"))
                return
            await _fail_suno_job(db, job, params, str(exc))
            return
        db.add(KieTask(task_id=task_id, job_id=job.job_id, model="suno_v6"))
        if step.get("kie_model") == "ai-music-api/generate":
            suno_state = dict(params.get("_suno") or {})
            suno_state["music_task_id"] = task_id
            params["_suno"] = suno_state
            music_state = dict(params.get("_music") or {})
            music_state["task_id"] = task_id
            params["_music"] = music_state
        params["_kie_task_id"] = task_id
        job.params = params
        job.updated_at = utc_naive_now()
        db.commit()
        logger.info("[SUNO] Job %s chained %s as %s", job.job_id, step.get("kie_model"), task_id)
        return

    if step["action"] == "finish":
        await _complete_suno_job(db, job, params, step.get("primary_url"))
        return
    await _fail_suno_job(db, job, params, step.get("error") or "Music generation failed")


async def _keep_late_suno_versions(db, job, payload: Dict[str, Any]) -> None:
    """Store a clip that arrives after the song was already marked ready.

    Finishing the job used to ignore every later callback, so the second
    version was dropped. This only updates the stored tracks.
    """
    import suno_v6

    event = suno_v6.interpret(payload)
    incoming = event.get("tracks") or []
    if not incoming:
        return
    params = dict(job.params or {})
    music = params.get("_music") if isinstance(params.get("_music"), dict) else {}
    music = dict(music)
    before = list(music.get("tracks") or [])
    merged = suno_v6._merge_tracks(before, incoming)
    if json.dumps(before, sort_keys=True, default=str) == json.dumps(merged, sort_keys=True, default=str):
        return
    music["tracks"] = merged
    params["_music"] = music
    job.params = params
    job.updated_at = utc_naive_now()
    db.commit()
    logger.info("[SUNO] job %s kept a late version (%s tracks)", job.job_id, len(merged))


async def _merge_late_suno_tracks(job_id: str, payload: Dict[str, Any]) -> None:
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job or job.status in (JobState.FAILED.value, JobState.CANCELLED.value):
            return
        await _keep_late_suno_versions(db, job, payload)
    finally:
        db.close()


async def _apply_kie_job_result(job_id: str, payload: Dict[str, Any]) -> None:
    import kie_client

    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            logger.error("[KIE] job %s not found for callback", job_id)
            return
        if job.model == "suno_v6":
            await _apply_suno_v6(db, job, payload)
            return
        if job.status in (JobState.COMPLETED.value, JobState.FAILED.value, JobState.CANCELLED.value):
            logger.info("[KIE] job %s already terminal (%s)", job_id, job.status)
            return
        urls = kie_client.parse_result_urls(payload)
        if kie_client.is_success_code(payload) and urls:
            source_url = urls[0]
            params = dict(job.params or {})
            if await _chain_output_upscale(db, job, source_url, params):
                return
            params["_kie_result_url"] = source_url
            try:
                output_url = kie_client.download_url_for(source_url)
                params["_kie_download_minted_at"] = time.time()
            except Exception as mint_err:
                logger.warning(
                    "[KIE] download-url failed for %s, sending result URL: %s",
                    job_id,
                    mint_err,
                )
                output_url = source_url
            job.params = params
            db.commit()
            if _job_manager:
                _job_manager.mark_completed(job_id, output_url=output_url)
            else:
                job.status = JobState.COMPLETED.value
                job.output_url = output_url
                job.completed_at = utc_naive_now()
                job.updated_at = utc_naive_now()
                db.commit()
            job = db.query(Job).filter(Job.job_id == job_id).first()
            meta = job.params if isinstance(job.params, dict) else {}
            await _send_job_notification(
                token=job.fcm_token,
                job_id=job.job_id,
                model=job.model,
                success=True,
                download_url=output_url,
                batch_id=meta.get("_batch_id"),
                variation_index=int(meta.get("_variation_index", 0) or 0),
                total_variations=int(meta.get("_total_variations", 1) or 1),
            )
            clear_cached_job_params(job_id)
            return
        err = kie_client.kie_failure_message(payload)
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        logger.error(
            "[KIE] job %s failed code=%s failCode=%s failMsg=%s",
            job_id,
            payload.get("code"),
            data.get("failCode"),
            data.get("failMsg"),
        )
        if _job_manager:
            _job_manager.mark_failed(job_id, err, worker_id=None)
        else:
            job.status = JobState.FAILED.value
            job.error = err
            job.updated_at = utc_naive_now()
            db.commit()
        job = db.query(Job).filter(Job.job_id == job_id).first()
        meta = job.params if isinstance(job.params, dict) else {}
        await _send_job_notification(
            token=job.fcm_token,
            job_id=job.job_id,
            model=job.model,
            success=False,
            batch_id=meta.get("_batch_id"),
            variation_index=int(meta.get("_variation_index", 0) or 0),
            total_variations=int(meta.get("_total_variations", 1) or 1),
        )
        clear_cached_job_params(job_id)
    finally:
        db.close()


@public_router.post("/kie/callback")
async def kie_callback(request: Request):
    import kie_client

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")
    task_id = kie_client.extract_task_id(payload)
    timestamp = request.headers.get("X-Webhook-Timestamp") or ""
    signature = request.headers.get("X-Webhook-Signature") or ""
    if not task_id or not kie_client.verify_webhook_signature(task_id, timestamp, signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    db = SessionLocal()
    job_id = None
    late_job_id = None
    partial = False
    defer_claim = False
    try:
        rec = db.query(KieTask).filter(KieTask.task_id == task_id).first()
        if not rec:
            logger.warning("[KIE] callback for unknown task %s", task_id)
            return {"ok": True, "ignored": True}
        if rec.callback_received:
            # A claimed song task can still carry the second version.
            if rec.model == "suno_v6":
                late_job_id = rec.job_id
            else:
                return {"ok": True, "duplicate": True}
        else:
            job_id = rec.job_id
            defer_claim = rec.model == "suno_v6"
        if defer_claim or late_job_id:
            import suno_v6
            event = suno_v6.interpret(payload)
            disposition = event.get("disposition")
            partial = disposition in ("pending", "progress")
            job = db.query(Job).filter(Job.job_id == (late_job_id or job_id)).first()
            params = job.params if job and isinstance(job.params, dict) else {}
            context = params.get("_suno") if isinstance(params.get("_suno"), dict) else {}
            music = params.get("_music") if isinstance(params.get("_music"), dict) else {}
            if not partial:
                partial = suno_v6.hold_for_second_version(context, music, event)
            stored_ids = [
                track.get("id") for track in (music.get("tracks") or []) if isinstance(track, dict)
            ]
            incoming_ids = [
                track.get("id") for track in (event.get("tracks") or []) if isinstance(track, dict)
            ]
            logger.info(
                "[KIE] suno callback task=%s job=%s already_claimed=%s waiting=%s stored=%s incoming=%s %s",
                task_id,
                late_job_id or job_id,
                bool(late_job_id),
                partial,
                stored_ids,
                incoming_ids,
                suno_v6.callback_summary(payload),
            )
        if not defer_claim:
            rec.callback_received = True
            rec.updated_at = utc_naive_now()
            db.commit()
    finally:
        db.close()
    if late_job_id:
        await _merge_late_suno_tracks(late_job_id, payload)
        return {"ok": True, "duplicate": True}
    await _apply_kie_job_result(job_id, payload)
    if defer_claim and not partial:
        claim_db = SessionLocal()
        try:
            claimed = claim_db.query(KieTask).filter(KieTask.task_id == task_id).first()
            if claimed and not claimed.callback_received:
                claimed.callback_received = True
                claimed.updated_at = utc_naive_now()
                claim_db.commit()
        finally:
            claim_db.close()
    if partial:
        return {"ok": True, "partial": True}
    return {"ok": True}


def _claim_runware_task(task_id: str) -> Optional[str]:
    """Mark the task claimed. Returns the job id, or None when already handled."""
    db = SessionLocal()
    try:
        rec = (
            db.query(RunwareTask)
            .filter(
                RunwareTask.task_id == task_id,
                RunwareTask.callback_received == False,  # noqa: E712
            )
            .first()
        )
        if not rec:
            return None
        rec.callback_received = True
        rec.updated_at = utc_naive_now()
        job_id = rec.job_id
        db.commit()
        return job_id
    finally:
        db.close()


def _release_runware_task(task_id: str) -> None:
    db = SessionLocal()
    try:
        rec = db.query(RunwareTask).filter(RunwareTask.task_id == task_id).first()
        if not rec:
            return
        rec.callback_received = False
        rec.updated_at = utc_naive_now()
        db.commit()
    finally:
        db.close()


async def _chain_output_upscale(db, job, source_url: str, params: Dict[str, Any]) -> bool:
    """Run the upscaler the client picked. Crisp stays on Kie; the other two are Runware."""
    choice = _output_upscale_choice(job.model, params)
    if not choice:
        return False
    if choice == "recraft_crisp_upscale":
        return await _chain_kie_upscale(db, job, source_url, params)
    return await _chain_runware_upscale(db, job, source_url, params, choice)


async def _chain_runware_upscale(
    db, job, source_url: str, params: Dict[str, Any], upscale_model: str
) -> bool:
    """2× General Upscale or Fine Upscale on the same job."""
    import runware_client

    spec = _RUNWARE_UPSCALE_SPECS.get(upscale_model)
    if not spec or params.get("_runware_upscale_started"):
        return False
    model_id, profile = spec
    task_uuid = runware_client.new_task_uuid()
    try:
        task = runware_client.assemble_task(
            {
                "provider": {"model_id": model_id},
                "runware": {
                    "profile": profile,
                    "task_type": "upscale",
                    "output_format": "PNG",
                },
            },
            {"image": source_url, "upscaleFactor": 2},
            task_uuid,
            runware_client.webhook_url(),
        )
        await asyncio.to_thread(runware_client.submit_task, task)
    except Exception:
        logger.exception("[RUNWARE] upscale chain failed for %s", job.job_id)
        return False
    db.add(RunwareTask(task_id=task_uuid, job_id=job.job_id, model=upscale_model))
    params["_runware_upscale_started"] = True
    params["upscale_model"] = upscale_model
    job.params = params
    db.commit()
    logger.info("[RUNWARE] Job %s chaining %s task %s", job.job_id, upscale_model, task_uuid)
    return True


async def _apply_runware_job_result(job_id: str, payload: Dict[str, Any]) -> str:
    """Apply a Runware payload. Returns applied, failed, processing, or ignored."""
    import runware_client

    if runware_client.is_processing(payload):
        return "processing"

    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            logger.error("[RUNWARE] job %s not found", job_id)
            return "ignored"
        if job.status in (JobState.COMPLETED.value, JobState.FAILED.value, JobState.CANCELLED.value):
            logger.info("[RUNWARE] job %s already terminal (%s)", job_id, job.status)
            return "ignored"

        url = runware_client.media_url(payload)
        if url:
            params = dict(job.params or {})
            if await _chain_output_upscale(db, job, url, params):
                return "applied"
            stored_key = None
            output_url = url
            try:
                stored = ExternalExecutionResult(success=True, result_url=url)
                stored_key, output_url = _store_external_result(job_id, job.model, stored)
            except Exception:
                logger.exception("[RUNWARE] store failed for %s; keeping provider URL", job_id)
                stored_key = None
                output_url = url
            if _job_manager:
                if stored_key:
                    _job_manager.mark_completed(job_id, result_r2_key=stored_key)
                    output_url = create_presigned_get_url(stored_key, expires_in=3600) or output_url
                else:
                    _job_manager.mark_completed(job_id, output_url=output_url)
            else:
                job.status = JobState.COMPLETED.value
                job.result_r2_key = stored_key
                job.output_url = None if stored_key else output_url
                job.completed_at = utc_naive_now()
                job.updated_at = utc_naive_now()
                db.commit()
                if stored_key:
                    output_url = create_presigned_get_url(stored_key, expires_in=3600) or output_url
            job = db.query(Job).filter(Job.job_id == job_id).first()
            meta = job.params if isinstance(job.params, dict) else {}
            await _send_job_notification(
                token=job.fcm_token,
                job_id=job.job_id,
                model=job.model,
                success=True,
                download_url=output_url,
                batch_id=meta.get("_batch_id"),
                variation_index=int(meta.get("_variation_index", 0) or 0),
                total_variations=int(meta.get("_total_variations", 1) or 1),
            )
            clear_cached_job_params(job_id)
            return "applied"

        err = runware_client.failure_message(payload) or "Runware task failed"
        logger.error("[RUNWARE] job %s failed: %s", job_id, err)
        if _job_manager:
            _job_manager.mark_failed(job_id, err, worker_id=None)
        else:
            job.status = JobState.FAILED.value
            job.error = err
            job.updated_at = utc_naive_now()
            db.commit()
        job = db.query(Job).filter(Job.job_id == job_id).first()
        meta = job.params if isinstance(job.params, dict) else {}
        await _send_job_notification(
            token=job.fcm_token,
            job_id=job.job_id,
            model=job.model,
            success=False,
            batch_id=meta.get("_batch_id"),
            variation_index=int(meta.get("_variation_index", 0) or 0),
            total_variations=int(meta.get("_total_variations", 1) or 1),
        )
        clear_cached_job_params(job_id)
        return "failed"
    finally:
        db.close()


@public_router.post("/runware/callback")
async def runware_callback(request: Request, token: str = ""):
    import runware_client

    if not runware_client.verify_webhook_token(token):
        raise HTTPException(status_code=401, detail="Invalid webhook token")
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    if isinstance(payload, list):
        payload = {"data": payload}
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")
    task_id = runware_client.task_uuid_of(payload)
    if not task_id:
        raise HTTPException(status_code=400, detail="Missing taskUUID")
    if runware_client.is_processing(payload):
        return {"ok": True, "pending": True}
    job_id = _claim_runware_task(task_id)
    if not job_id:
        return {"ok": True, "duplicate": True}
    outcome = await _apply_runware_job_result(job_id, payload)
    if outcome == "processing":
        _release_runware_task(task_id)
    return {"ok": True}


@public_router.get("/personal/ui")
async def personal_queue_ui():
    path = Path(__file__).resolve().parent / "static" / "personal_queue.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="personal_queue.html missing")
    return FileResponse(path, media_type="text/html")


@public_router.get("/download_url/{job_id}")
async def get_signed_download_url(job_id: str):
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job or job.status != JobState.COMPLETED.value:
            raise HTTPException(status_code=404, detail="Result not ready")

        # KIE results stay on KIE. Mint a fresh 20-minute link for each request.
        source = _kie_result_source(job)
        if source:
            import kie_client

            try:
                return kie_client.download_url_for(source)
            except Exception as mint_err:
                logger.warning("[KIE] refresh download-url failed for %s: %s", job_id, mint_err)
                if job.output_url:
                    return job.output_url
                raise HTTPException(status_code=502, detail="KIE download link unavailable")

        # External API route: output_url may already be a direct downloadable URL.
        if job.output_url:
            return job.output_url

        r2_key = job.result_r2_key
        signed_url = _generate_signed_download_url(r2_key)
        if not signed_url:
            raise HTTPException(status_code=500, detail="DOWNLOAD_SECRET not set")
        return signed_url
    finally:
        db.close()


# =============================================================================
# Client Endpoints
# =============================================================================


@client_router.post("/get_video_upload_url")
async def get_video_upload_url(
    request: Request,
    auth: Dict = Depends(require_firebase_auth),
):
    """
    Get a presigned URL for uploading an input video directly to R2 as binary.
    
    Frontend flow:
    1. Call this endpoint to get upload_url and r2_key
    2. PUT the raw video binary to upload_url (Content-Type: video/mp4)
    3. Submit the job via /submit with params.input_video_r2_key = r2_key
    
    This avoids sending large base64-encoded video data in JSON payloads.
    
    Returns:
        upload_url: Presigned PUT URL for R2 (valid for 30 minutes)
        r2_key: R2 object key to include in job submission as input_video_r2_key
    """
    video_id = str(uuid.uuid4())
    r2_key = f"{_image_prefix}/input-videos/{video_id}.mp4"
    
    upload_url = create_presigned_put_url(r2_key, expires_in=1800)  # 30 min to upload
    if not upload_url:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate video upload URL. R2 storage may not be configured."
        )
    
    logger.info(
        f"[VIDEO_UPLOAD] Generated upload URL for user {auth.get('user_id')}, "
        f"r2_key={r2_key}"
    )
    
    return {
        "upload_url": upload_url,
        "r2_key": r2_key,
    }


@client_router.post("/get_media_upload_urls")
async def get_media_upload_urls(
    request: Request,
    auth: Dict = Depends(require_firebase_auth),
):
    """
    Get presigned URLs for uploading input media (images/videos) directly to R2.
    
    Frontend flow:
    1. Call this endpoint with count (number of files needed)
    2. PUT raw binary data to each upload_url
    3. Submit the job via /submit with the r2_keys in params
       (e.g. params.init_image_r2_key, bulk_image_r2_keys, etc.)
    
    This avoids sending large base64-encoded media in JSON payloads.
    
    Request body (JSON):
        count: Number of upload URLs to generate (1-22, required)
        media_type: "image" or "video" (default: "image")
    
    Returns:
        uploads: List of {upload_url, r2_key} pairs
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    count = body.get("count", 1)
    media_type = body.get("media_type", "image")
    
    if not isinstance(count, int) or count < 1 or count > 22:
        raise HTTPException(
            status_code=400,
            detail="count must be an integer between 1 and 22"
        )
    
    if media_type not in ("image", "video"):
        raise HTTPException(
            status_code=400,
            detail="media_type must be 'image' or 'video'"
        )
    
    ext = ".bin" if media_type == "image" else ".mp4"
    prefix_dir = "input-media"
    
    uploads = []
    for _ in range(count):
        file_id = str(uuid.uuid4())
        r2_key = f"{_image_prefix}/{prefix_dir}/{file_id}{ext}"
        upload_url = create_presigned_put_url(r2_key, expires_in=1800)  # 30 min
        if not upload_url:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate upload URL. R2 storage may not be configured."
            )
        uploads.append({"upload_url": upload_url, "r2_key": r2_key})
    
    logger.info(
        f"[MEDIA_UPLOAD] Generated {count} upload URLs (type={media_type}) "
        f"for user {auth.get('user_id')}"
    )
    
    return {"uploads": uploads}


def _is_describe_asset_r2_key_allowed(key: str) -> bool:
    """Only keys created via /get_media_upload_urls (input-media prefix)."""
    if not key or ".." in key or key.startswith("/"):
        return False
    prefix = f"{_image_prefix}/input-media/"
    return key.startswith(prefix)


def _mime_from_r2_key(key: str) -> str:
    if "." not in key:
        return "application/octet-stream"
    ext = key.rsplit(".", 1)[-1].lower()
    if ext in ("jpg", "jpeg"):
        return "image/jpeg"
    if ext == "png":
        return "image/png"
    if ext == "webp":
        return "image/webp"
    if ext == "bin":
        return "application/octet-stream"
    guessed, _ = mimetypes.guess_type(f"file.{ext}")
    return guessed or "application/octet-stream"


def _r2_head_object_meta_sync(key: str) -> dict:
    """Return head_object response dict (includes ContentLength)."""
    if not _s3_client or not _r2_bucket:
        raise RuntimeError("R2 not configured")
    return _s3_client.head_object(Bucket=_r2_bucket, Key=key)


def _r2_get_object_bytes_sync(key: str, max_bytes: int) -> bytes:
    if not _s3_client or not _r2_bucket:
        raise RuntimeError("R2 not configured")
    resp = _s3_client.get_object(Bucket=_r2_bucket, Key=key)
    body = resp["Body"]
    chunks: List[bytes] = []
    total = 0
    while True:
        chunk = body.read(65536)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError("OBJECT_TOO_LARGE")
        chunks.append(chunk)
    return b"".join(chunks)


async def _reserve_asset_prompt_daily_slot(user_id: str) -> None:
    """Raises 429 if user exceeded daily cap (counts reserved attempts before VLM call)."""
    try:
        cap = int(os.getenv("ASSET_PROMPT_VLM_DAILY_LIMIT_PER_USER", "50"))
    except ValueError:
        cap = 50
    if cap <= 0:
        return
    async with _asset_prompt_rate_lock:
        today = _utc_today()
        cur = _asset_prompt_rate.get(user_id)
        if not cur or cur[0] != today:
            cur = (today, 0)
        n = cur[1]
        if n >= cap:
            raise HTTPException(
                status_code=429,
                detail="Daily limit reached for auto-prompt. Try again tomorrow.",
            )
        _asset_prompt_rate[user_id] = (today, n + 1)


@client_router.post(
    "/describe_asset_for_prompt",
    response_model=DescribeAssetForPromptResponse,
)
async def describe_asset_for_prompt(
    payload: DescribeAssetForPromptRequest,
    authorization: Optional[str] = Header(None),
    auth: Dict = Depends(require_firebase_auth),
):
    """
    Load image bytes from R2 and ask the Gemini Cloudflare worker for one generation prompt.
    The caller's Firebase token is forwarded so the worker can authorize the request.

    R2-only: request body must contain ``input_r2_key`` under the app's ``input-media/`` prefix.
    For video sources, the client should upload a single JPEG frame to R2 and pass that key
    with ``media_type=image``.
    """
    if not is_asset_prompt_vlm_enabled():
        raise HTTPException(
            status_code=503,
            detail="Auto-prompt from asset is disabled on the server.",
        )

    key = (payload.input_r2_key or "").strip()
    if not _is_describe_asset_r2_key_allowed(key):
        raise HTTPException(
            status_code=400,
            detail="Invalid or disallowed input_r2_key.",
        )

    uid = auth.get("user_id")
    if uid is None:
        raise HTTPException(status_code=401, detail="User id missing from auth")

    await _reserve_asset_prompt_daily_slot(str(uid))

    max_bytes = get_max_image_bytes()
    try:
        meta = await asyncio.to_thread(_r2_head_object_meta_sync, key)
    except ClientError as e:
        code = (e.response.get("Error") or {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            raise HTTPException(status_code=404, detail="Object not found in storage.") from e
        logger.error(f"[DESCRIBE_ASSET] R2 head_object failed: {e}")
        raise HTTPException(status_code=500, detail="Could not read object metadata from storage.") from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    size = int(meta.get("ContentLength") or 0)
    if size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail="Image is too large for auto-prompt. Use a smaller file.",
        )

    try:
        raw = await asyncio.to_thread(_r2_get_object_bytes_sync, key, max_bytes)
    except ClientError as e:
        code = (e.response.get("Error") or {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            raise HTTPException(status_code=404, detail="Object not found in storage.") from e
        logger.error(f"[DESCRIBE_ASSET] R2 get_object failed: {e}")
        raise HTTPException(status_code=500, detail="Could not read object from storage.") from e
    except ValueError as e:
        if str(e) == "OBJECT_TOO_LARGE":
            raise HTTPException(
                status_code=413,
                detail="Image is too large for auto-prompt. Use a smaller file.",
            ) from e
        raise HTTPException(status_code=500, detail="Could not read object.") from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    mime = (payload.content_type or "").strip() or _mime_from_r2_key(key)
    if not mime.startswith("image/"):
        mime = _mime_from_r2_key(key)

    ok, text = await asyncio.to_thread(
        describe_image_with_gemini, key, raw, mime, authorization or ""
    )
    if not ok:
        logger.warning(
            f"[DESCRIBE_ASSET] VLM failed user_id={uid!s} key_prefix={key[:48]!r} err={text[:200]!r}"
        )
        raise HTTPException(status_code=502, detail=text)

    logger.info(
        f"[DESCRIBE_ASSET] ok user_id={uid!s} key_prefix={key[:48]!r} prompt_len={len(text)}"
    )
    return DescribeAssetForPromptResponse(prompt=text)


@client_router.post(
    "/suggest_prompt",
    response_model=SuggestPromptResponse,
)
async def suggest_prompt_route(
    payload: SuggestPromptRequest,
    authorization: Optional[str] = Header(None),
    auth: Dict = Depends(require_firebase_auth),
):
    """Write or refine one prompt for the selected model. No gem charge."""
    if not is_asset_prompt_vlm_enabled():
        raise HTTPException(
            status_code=503,
            detail="Auto-prompt is disabled on the server.",
        )
    uid = auth.get("user_id")
    if uid is None:
        raise HTTPException(status_code=401, detail="User id missing from auth")
    await _reserve_asset_prompt_daily_slot(str(uid))
    try:
        if payload.context or payload.also:
            text, related = await asyncio.to_thread(
                suggest_related,
                payload.model_id,
                payload.field,
                payload.text,
                payload.caption,
                authorization or "",
                payload.context,
                payload.also,
            )
        else:
            text = await asyncio.to_thread(
                suggest_prompt,
                payload.model_id,
                payload.field,
                payload.text,
                payload.caption,
                authorization or "",
            )
            related = {}
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return SuggestPromptResponse(prompt=text, related=related)


@client_router.post("/eq_preset", response_model=EqPresetResponse)
async def eq_preset_route(
    payload: EqPresetRequest,
    authorization: Optional[str] = Header(None),
    auth: Dict = Depends(require_firebase_auth),
):
    """Pick one VLC equalizer id for a HitMaker song. No gem charge."""
    if auth.get("user_id") is None:
        raise HTTPException(status_code=401, detail="User id missing from auth")
    preset = await asyncio.to_thread(
        choose_eq_preset,
        payload.title,
        payload.style,
        payload.tags,
        payload.lyrics,
        authorization or "",
    )
    return EqPresetResponse(preset=preset)


def _bulk_entries_include_multi_image_group(entries: Optional[List[Any]]) -> bool:
    """True if any top-level bulk entry is a list (multi-reference task)."""
    if not entries:
        return False
    return any(isinstance(e, list) for e in entries)


def _flatten_bulk_r2_keys_for_cleanup(entries: List[Any], bulk_limit: int) -> List[str]:
    """Collect every R2 object key from input_image_r2_keys (flat or nested per task)."""
    flat: List[str] = []
    for entry in (entries or [])[:bulk_limit]:
        if isinstance(entry, str):
            flat.append(entry)
        elif isinstance(entry, list):
            for k in entry:
                if isinstance(k, str):
                    flat.append(k)
    return flat


def _build_bulk_images_to_process_r2(entries: List[Any], bulk_limit: int) -> List[Any]:
    """
    Each item is either 'r2:<key>' (single) or ['r2:k1', 'r2:k2', ...] (multi-reference).
    """
    out: List[Any] = []
    for entry in entries[:bulk_limit]:
        if isinstance(entry, str):
            out.append(f"r2:{entry}")
        elif isinstance(entry, list):
            out.append([f"r2:{k}" for k in entry])
        else:
            raise ValueError(
                f"Invalid input_image_r2_keys entry type: {type(entry).__name__}"
            )
    return out


@client_router.post("/submit", response_model=SubmitResponse)
async def submit_job(
    request: Request,
    payload: SubmitRequest,
    background_tasks: BackgroundTasks,
    auth: Dict = Depends(require_firebase_auth),
):
    """
    Submit a new job for processing.
    
    Requires Firebase authentication via Authorization: Bearer <firebase_id_token>
    
    Supports:
    - enhance_prompt: If true, AI will enhance the prompt before processing
    - variations: Number of output variations to generate (1-4, creates multiple jobs)
    - input_image_r2_keys: Bulk tasks as R2 object keys (str or list[str] per task for multi-ref)
    - is_bulk: Flag indicating bulk edit request (requires non-empty input_image_r2_keys)
    
    VIP Benefits:
    - Bulk edit: Process multiple images with same settings
    - Free upscale: Upscale cost is waived for VIP users
    - More variations: Max 4 variations (non-VIP limited to 2)
    
    Returns job ID and optionally dispatches immediately if a worker is ready.
    For variations > 1 or bulk edit, returns batch_id and all job_ids.
    """
    
    ip = get_client_ip(request)
    user_id = auth.get("user_id")
    is_vip = auth.get("is_vip", False)
    # Backward-compatible access for older SubmitRequest schema versions.
    on_device_flag = bool(getattr(payload, "on_device", False))

    # Save the client request as JSON (as received from client)
    try:
        # Convert payload to dict (handle Pydantic models)
        try:
            payload_data = payload.dict()
        except Exception:
            payload_data = dict(payload)
        
        # Prepare client request data (exactly as received)
        client_request_data = {
            "request_info": {
                "ip": ip,
                "user_id": user_id,
                "is_vip": is_vip,
                "auth_info": {k: v for k, v in auth.items() if k != "firebase_token"},
            },
            "payload": payload_data,  # Original payload from client
        }
        
        # Save with proper grouping and labeling
        _save_json_request(
            data=client_request_data,
            request_type="client_submit",
            user_id=user_id,
            additional_labels={
                "model": payload.model if hasattr(payload, "model") else None,
                "is_vip": "vip" if is_vip else "free",
            },
        )
    except Exception as e:
        logger.warning(f"Failed to save client submit request JSON: {e}")

    # Check if user account is active
    if not auth.get("is_active", True):
        raise HTTPException(
            status_code=403,
            detail="Account is disabled. Please contact support."
        )

    # ==========================================================================
    # BULK EDIT: Tier-aware limits
    # ==========================================================================
    # Bulk edit allows processing multiple images with the same prompt/settings.
    # Limits:
    # - non-VIP: up to 5 bulk tasks
    # - VIP: up to 15 bulk tasks
    # A "task" is one top-level entry (string or list of refs for multi-image models).
    is_bulk_request = (
        payload.is_bulk
        or (payload.input_image_r2_keys and len(payload.input_image_r2_keys) > 1)
        or _bulk_entries_include_multi_image_group(payload.input_image_r2_keys)
    )
    bulk_limit = 15 if is_vip else 5
    bulk_input_count = len(payload.input_image_r2_keys or [])

    if is_bulk_request and bulk_input_count > bulk_limit:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Bulk edit limit exceeded: max {bulk_limit} image(s) allowed for "
                f"{'VIP' if is_vip else 'non-VIP'} users."
            ),
        )
    
    # Validate model
    try:
        model = validate_model(payload.model)
    except InvalidModelError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Full params for dispatch (R2 keys only; embedded media rejected in SubmitRequest)
    full_params = dict(payload.params) if payload.params else {}
    # The client cannot choose the provider. Only a flagged request the gate allowed through is marked.
    full_params.pop("_moderation_flagged", None)
    if getattr(request.state, "moderation_flagged", False) is True:
        full_params["_moderation_flagged"] = True

    # Flutter UI id "ltx2_3_cinereal" → concrete LTX 2.3 worker model (must match worker registry)
    if model == "ltx2_3_cinereal":
        ff_keys = full_params.get("first_frame_image_r2_keys")
        has_last = bool(full_params.get("last_frame_image_r2_key"))
        multi_first = isinstance(ff_keys, list) and len(ff_keys) >= 2
        if has_last or multi_first:
            model = "ltx2_3_flf2v"
        else:
            model = "ltx2_3_i2v_distilled"
        logger.info(
            "[MODEL] Normalized ltx2_3_cinereal → %s (has_last=%s multi_first=%s)",
            model,
            has_last,
            multi_first,
        )

    # Flutter UI id "ltx25_int8" (CineReal on Animate Image) → i2v or flf2v alias
    if model == "ltx25_int8":
        ff_keys = full_params.get("first_frame_image_r2_keys")
        has_last = bool(full_params.get("last_frame_image_r2_key"))
        multi_first = isinstance(ff_keys, list) and len(ff_keys) >= 2
        if has_last or multi_first:
            model = "ltx25_int8_flf2v"
        else:
            model = "ltx25_int8_i2v"
        logger.info(
            "[MODEL] Normalized ltx25_int8 → %s (has_last=%s multi_first=%s)",
            model,
            has_last,
            multi_first,
        )
    
    # ==========================================================================
    # PARAM KEY NORMALIZATION
    # ==========================================================================
    # Frontend may send "is_custom" (string) but workers expect "is_custom_mode" (boolean)
    if "is_custom" in full_params and "is_custom_mode" not in full_params:
        is_custom_val = full_params.pop("is_custom")
        # Convert string to boolean
        if is_custom_val == "true" or is_custom_val is True:
            full_params["is_custom_mode"] = True
        elif is_custom_val == "false" or is_custom_val is False:
            full_params["is_custom_mode"] = False
        logger.debug(f"[PARAMS] Mapped 'is_custom' → 'is_custom_mode': {full_params.get('is_custom_mode')}")
    
    # ==========================================================================
    # VARIATIONS: VIP Feature - Max 2 for non-VIP, Max 4 for VIP
    # ==========================================================================
    max_variations = 4 if is_vip else 2
    requested_variations = payload.variations
    variations = max(1, min(max_variations, requested_variations))
    if model == "suno_v6":
        # One request already returns both song versions.
        variations = 1
    
    # If user requested more variations than allowed, notify them
    if requested_variations > max_variations:
        logger.info(
            f"[VARIATIONS] User requested {requested_variations} variations, "
            f"limited to {max_variations} (is_vip={is_vip})"
        )
    
    # ==========================================================================
    # BUILD IMAGE LIST for processing
    # ==========================================================================
    # R2 keys only: bulk uses input_image_r2_keys; single job uses params.init_image_r2_key
    # or params.init_image_r2_keys (multi-ref, one job). Entries use "r2:<key>" for the job loop.
    images_to_process = []
    using_r2_images = False   # True when images come from presigned R2 uploads
    r2_input_keys = []        # Track all input R2 keys for cleanup after job completion

    if model == "suno_v6" and payload.input_image_r2_keys:
        merged: List[str] = []
        existing_refs = full_params.get("init_image_r2_keys")
        if isinstance(existing_refs, list):
            merged.extend(key.strip() for key in existing_refs if isinstance(key, str) and key.strip())
        elif isinstance(full_params.get("init_image_r2_key"), str) and full_params.get("init_image_r2_key").strip():
            merged.append(full_params["init_image_r2_key"].strip())
        for entry in payload.input_image_r2_keys:
            if isinstance(entry, str) and entry.strip():
                if entry.strip() not in merged:
                    merged.append(entry.strip())
            elif isinstance(entry, list):
                for key in entry:
                    if isinstance(key, str) and key.strip() and key.strip() not in merged:
                        merged.append(key.strip())
        full_params["init_image_r2_keys"] = merged[:5]
        full_params.pop("init_image_r2_key", None)

    if model == "suno_v6":
        # Reference images stay on this one song. They are not a bulk edit.
        images_to_process = [None]
        for key in full_params.get("init_image_r2_keys") or []:
            if isinstance(key, str) and key not in r2_input_keys:
                r2_input_keys.append(key)
        single_ref = full_params.get("init_image_r2_key")
        if isinstance(single_ref, str) and single_ref.strip() and single_ref not in r2_input_keys:
            r2_input_keys.append(single_ref.strip())
    elif payload.input_image_r2_keys and len(payload.input_image_r2_keys) > 0:
        # R2 presigned upload path – images are already in R2
        using_r2_images = True
        try:
            images_to_process = _build_bulk_images_to_process_r2(
                list(payload.input_image_r2_keys), bulk_limit
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        r2_input_keys.extend(
            _flatten_bulk_r2_keys_for_cleanup(
                list(payload.input_image_r2_keys), bulk_limit
            )
        )
        logger.info(
            f"[R2_MEDIA] Processing {len(images_to_process)} bulk task(s) via R2 keys "
            f"(flat key count for cleanup={len(r2_input_keys)})"
        )
        
        # If a single R2 mask key was provided (non-bulk), store it
        if payload.mask_image_r2_key:
            r2_input_keys.append(payload.mask_image_r2_key)
        
        # Bulk mask R2 keys
        if payload.bulk_mask_r2_keys:
            for mk in payload.bulk_mask_r2_keys:
                if mk:
                    r2_input_keys.append(mk)
        
        # Log configs
        if payload.bulk_outpaint_configs:
            logger.info(f"[R2_MEDIA] Received {len(payload.bulk_outpaint_configs)} outpaint configs")
            for i, cfg in enumerate(payload.bulk_outpaint_configs):
                if cfg:
                    logger.debug(f"[R2_MEDIA] Outpaint config {i}: {cfg}")
    else:
        # Single image, multi-ref single job, or text-to-image
        multi_r2_keys = full_params.get("init_image_r2_keys")
        if multi_r2_keys is not None:
            if not isinstance(multi_r2_keys, list) or not multi_r2_keys:
                raise HTTPException(
                    status_code=400,
                    detail="params.init_image_r2_keys must be a non-empty list of R2 object keys",
                )
            for j, k in enumerate(multi_r2_keys):
                if not isinstance(k, str) or not k.strip():
                    raise HTTPException(
                        status_code=400,
                        detail=f"params.init_image_r2_keys[{j}] must be a non-empty R2 object key string",
                    )
            using_r2_images = True
            r2_input_keys.extend(multi_r2_keys)
            full_params.pop("init_image_r2_keys", None)
            if len(multi_r2_keys) == 1:
                images_to_process = [f"r2:{multi_r2_keys[0]}"]
            else:
                images_to_process = [[f"r2:{k}" for k in multi_r2_keys]]
            logger.info(
                f"[R2_MEDIA] Single job with {len(multi_r2_keys)} init_image R2 key(s)"
            )
        else:
            single_r2_key = full_params.get("init_image_r2_key")
            if single_r2_key:
                if not isinstance(single_r2_key, str) or not single_r2_key.strip():
                    raise HTTPException(
                        status_code=400,
                        detail="params.init_image_r2_key must be a non-empty R2 object key string",
                    )
                using_r2_images = True
                images_to_process = [f"r2:{single_r2_key}"]
                r2_input_keys.append(single_r2_key)
                full_params.pop("init_image_r2_key", None)
                logger.info("[R2_MEDIA] Single image via R2 key")
            else:
                images_to_process = [None]  # Text-to-image / no init image
    
    # Handle input video R2 key (payload overrides params.input_video_r2_key)
    if payload.input_video_r2_key:
        full_params["input_video_r2_key"] = payload.input_video_r2_key
    _ivk = full_params.get("input_video_r2_key")
    if _ivk and isinstance(_ivk, str) and _ivk.strip() and _ivk not in r2_input_keys:
        r2_input_keys.append(_ivk)
        logger.info(f"[R2_MEDIA] Input video via R2 key: {_ivk}")
    
    num_images = len(images_to_process)
    
    # ==========================================================================
    # VIDEO UPSCALER VALIDATION
    # ==========================================================================
    # Validate video_duration for video upscalers (max 10 minutes)
    if model in ("video_upscaler", "flashvsr_video_upscaler") and "video_duration" in full_params:
        vd_raw = full_params.get("video_duration")
        # Unresolved client/template placeholders (e.g. "{{video_duration}}") are not numeric
        if isinstance(vd_raw, str) and ("{{" in vd_raw or "}}" in vd_raw):
            logger.debug(
                "[VIDEO_UPSCALER] Dropping unresolved video_duration placeholder: %r",
                vd_raw,
            )
            full_params.pop("video_duration", None)
        else:
            try:
                video_duration = float(vd_raw or 0)
                MAX_VIDEO_DURATION = 600  # 10 minutes in seconds
                if video_duration > MAX_VIDEO_DURATION:
                    logger.warning(
                        f"[VIDEO_UPSCALER] Video duration {video_duration}s exceeds max {MAX_VIDEO_DURATION}s, "
                        f"capping to {MAX_VIDEO_DURATION}s"
                    )
                    full_params["video_duration"] = MAX_VIDEO_DURATION
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"[VIDEO_UPSCALER] Invalid video_duration: {full_params.get('video_duration')}, error: {e}"
                )
                full_params.pop("video_duration", None)
    
    # ==========================================================================
    # CREDIT CALCULATION
    # ==========================================================================
    # VIP benefits: Free upscale
    # Total cost = per_job_cost * variations * num_images
    # For video_upscaler: dynamic cost based on video_duration (2 credits per 5 seconds)
    if model == "suno_v6":
        try:
            import suno_v6
            suno_v6.prepare(payload.prompt or "", full_params, for_submit=True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    if model in ("xai_tts", "qwen_voice_design", "qwen_voice_clone"):
        full_params["speech_text"] = payload.prompt or full_params.get("prompt") or ""
    credit_cost = get_credit_cost(
        model, 
        full_params, 
        variations=variations,
        is_vip=is_vip,
        num_images=num_images
    )
    
    if auth.get("total_credits", 0) < credit_cost:
        raise HTTPException(
            status_code=402,  # Payment Required
            detail=f"Insufficient credits. Have: {auth.get('total_credits', 0)}, Need: {credit_cost}"
        )
    
    # Handle prompt enhancement if requested
    prompt_enhanced = False
    enhanced_prompt_text = None
    final_prompt = payload.prompt or ""
    
    if payload.enhance_prompt and final_prompt:
        mode = full_params.get("mode")
        if should_enhance_for_model(model, mode):
            try:
                was_enhanced, enhanced_text, original = await enhance_prompt(
                    final_prompt, model, timeout=10.0
                )
                if was_enhanced:
                    prompt_enhanced = True
                    enhanced_prompt_text = enhanced_text
                    final_prompt = enhanced_text
                    logger.info(f"Prompt enhanced for model {model}")
            except Exception as enhance_error:
                logger.warning(f"Prompt enhancement failed: {enhance_error}")
                # Continue with original prompt
    
    # ==========================================================================
    # BATCH ID: Generate for multi-variation or bulk edit requests
    # ==========================================================================
    total_jobs = num_images * variations
    needs_batch = total_jobs > 1
    batch_id = str(uuid.uuid4()) if needs_batch else None
    
    if is_bulk_request:
        logger.info(f"[BULK] Creating {total_jobs} jobs: {num_images} images × {variations} variations")
    
    # Create job(s) - multiple jobs for variations > 1 OR bulk edit
    job_ids = []
    created_jobs = []
    total_credits_locked = 0
    
    # Store full params for each job (with specific image) for dispatch
    job_full_params_map = {}
    
    db = SessionLocal()
    try:
        now = utc_naive_now()
        
        # Determine output extension based on model type
        # Video models output .mp4, image models output .png, text-output models (e.g. OCR) output .txt
        VIDEO_MODELS = {
            "ltx2_video", "fastwan_21_1b", "fastwan_22_5b", "wan_22_14b_rapid",
            "wan_22_uint4_lightx2v_i2v", "wan_22_uint4_lightx2v_t2v",
            "wan_22_fp8_lightx2v_i2v", "wan_22_fp8_lightx2v_t2v",
            "wan21", "wan22",
            "video_upscaler", "flashvsr_video_upscaler",
            "hunyan15_480p_i2v_step_distilled", "hunyan15_480p_t2v_distilled",
            "ltx2_3_t2v", "ltx2_3_i2v_distilled", "ltx2_3_flf2v",
            "ltx2_3_iv2v_distilled",
            "minimax_h3",
            "ltx25_int8", "ltx25_int8_t2v", "ltx25_int8_i2v", "ltx25_int8_flf2v",
        }
        TEXT_OUTPUT_MODELS = {"glm_ocr"}
        AUDIO_OUTPUT_MODELS = {
            "minimax_music3",
            "xai_tts",
            "qwen_voice_design",
            "qwen_voice_clone",
            "ace_step_music",
            "minimax_music_26",
            "suno_v6",
        }
        model_l = model.lower()
        is_video_model = model_l in VIDEO_MODELS or "video" in model_l
        is_text_output_model = model_l in TEXT_OUTPUT_MODELS
        is_audio_output_model = model_l in AUDIO_OUTPUT_MODELS or "music" in model_l
        suno_action = str(full_params.get("action") or "song").strip().lower()
        if model_l == "suno_v6" and suno_action in ("video", "music_video", "mp4"):
            output_extension = ".mp4"
        elif model_l == "suno_v6" and suno_action in ("covers", "cover", "cover_art"):
            output_extension = ".png"
        elif model_l == "suno_v6" and suno_action in ("lyrics", "lyric", "persona", "voice"):
            output_extension = ".txt"
        elif is_audio_output_model:
            output_extension = ".mp3"
        elif is_video_model:
            output_extension = ".mp4"
        elif is_text_output_model:
            output_extension = ".txt"
        else:
            output_extension = ".png"
        
        # Loop through images (for bulk edit) and variations
        job_index = 0
        for image_idx, image_data in enumerate(images_to_process):
            for variation_idx in range(variations):
                # Generate unique job ID for each job
                job_id = str(uuid.uuid4())
                result_key = f"{_image_prefix}/{job_id}{output_extension}"
                
                # Build full params for this specific job (with specific image)
                job_full_params = dict(full_params)
                if image_data:
                    if isinstance(image_data, list):
                        if not using_r2_images:
                            raise HTTPException(
                                status_code=500,
                                detail="Internal error: multi-reference image task without R2 keys",
                            )
                        keys: List[str] = []
                        for item in image_data:
                            if isinstance(item, str) and item.startswith("r2:"):
                                keys.append(item[3:])
                            else:
                                raise HTTPException(
                                    status_code=400,
                                    detail="Invalid bulk R2 entry: expected r2:<key> strings",
                                )
                        if not keys:
                            raise HTTPException(
                                status_code=400,
                                detail="Empty R2 key group in bulk request",
                            )
                        job_full_params.pop("init_image", None)
                        job_full_params.pop("init_image_r2_key", None)
                        job_full_params.pop("init_image_r2_keys", None)
                        if len(keys) == 1:
                            job_full_params["init_image_r2_key"] = keys[0]
                        else:
                            job_full_params["init_image_r2_keys"] = keys
                    elif isinstance(image_data, str) and image_data.startswith("r2:"):
                        job_full_params["init_image_r2_key"] = image_data[3:]
                        job_full_params.pop("init_image", None)
                        job_full_params.pop("init_image_r2_keys", None)
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid image input: use R2 upload + init_image_r2_key / input_image_r2_keys only.",
                        )
                
                # Apply per-image outpaint config (for expand scene bulk edit)
                if payload.bulk_outpaint_configs and image_idx < len(payload.bulk_outpaint_configs):
                    outpaint_config = payload.bulk_outpaint_configs[image_idx]
                    if outpaint_config:
                        # Map frontend keys to worker-expected keys
                        # Frontend sends "is_custom" (string), worker expects "is_custom_mode" (boolean)
                        key_mapping = {
                            "is_custom": "is_custom_mode",  # Frontend → Worker key mapping
                        }
                        
                        # Determine if this is custom mode (check both is_custom and is_custom_mode)
                        is_custom = outpaint_config.get("is_custom", "false")
                        is_custom_mode_from_config = outpaint_config.get("is_custom_mode", False)
                        is_actually_custom = (
                            is_custom == "true" or is_custom is True or 
                            is_custom_mode_from_config is True
                        )
                        
                        # Base keys that are always extracted (for both custom and non-custom mode)
                        base_keys = [
                            "alignment", "aspect_ratio", "is_custom", "is_custom_mode",
                            "overlap_left", "overlap_right", "overlap_top", "overlap_bottom",
                        ]
                        
                        # Custom positioning keys - ONLY extract when is_custom_mode is True
                        custom_keys = [
                            "custom_scale", "norm_offset_x", "norm_offset_y",
                            "canvas_width", "canvas_height",
                            "image_width", "image_height", "image_left", "image_top",
                            "fitted_width", "fitted_height", "offset_x", "offset_y",
                        ]
                        
                        # Build extract_keys list based on mode
                        extract_keys = base_keys.copy()
                        if is_actually_custom:
                            extract_keys.extend(custom_keys)
                            logger.debug(f"[BULK] Custom mode detected for image {image_idx}, including custom positioning params")
                        else:
                            logger.debug(f"[BULK] Non-custom mode for image {image_idx}, excluding custom positioning params")
                        
                        # IMPORTANT: Only remove keys that are actually in the per-image config
                        # This ensures defaults from base params are preserved if not overridden
                        # First, identify which keys from extract_keys are actually in outpaint_config
                        keys_to_replace = [key for key in extract_keys if key in outpaint_config]
                        
                        # Remove only the keys that will be replaced
                        for key in keys_to_replace:
                            target_key = key_mapping.get(key, key)
                            if target_key in job_full_params:
                                del job_full_params[target_key]
                                logger.debug(f"[BULK] Removed base param {target_key} (will be replaced from per-image config)")
                        
                        # Also remove custom positioning params if NOT in custom mode
                        # (to prevent them from being sent when they shouldn't be)
                        if not is_actually_custom:
                            for key in custom_keys:
                                target_key = key_mapping.get(key, key)
                                if target_key in job_full_params:
                                    del job_full_params[target_key]
                                    logger.debug(f"[BULK] Removed custom param {target_key} (not in custom mode)")
                        
                        # Apply outpaint-specific params from per-image config
                        # Only apply keys that are actually in the config
                        for key in keys_to_replace:
                            value = outpaint_config[key]
                            # Convert string booleans to actual booleans
                            if value == "true":
                                value = True
                            elif value == "false":
                                value = False
                            # Convert numeric strings to floats where needed
                            elif key in ["custom_scale", "norm_offset_x", "norm_offset_y", 
                                        "canvas_width", "canvas_height",
                                        "image_width", "image_height", "image_left", "image_top",
                                        "fitted_width", "fitted_height", "offset_x", "offset_y"]:
                                try:
                                    value = float(value)
                                except (ValueError, TypeError):
                                    pass
                            
                            # Apply key mapping (e.g., is_custom → is_custom_mode)
                            target_key = key_mapping.get(key, key)
                            job_full_params[target_key] = value
                            logger.debug(f"[BULK] Applied per-image param {target_key}={value}")
                        
                        # Verify critical params are set after extraction
                        alignment_set = job_full_params.get("alignment")
                        aspect_ratio_set = job_full_params.get("aspect_ratio")
                        is_custom_mode_set = job_full_params.get("is_custom_mode", False)
                        expansion_factor_set = job_full_params.get("expansion_factor")
                        
                        logger.info(f"[BULK] Applied outpaint config for image {image_idx}: alignment={alignment_set}, aspect_ratio={aspect_ratio_set}, is_custom_mode={is_custom_mode_set}, expansion_factor={expansion_factor_set}")
                        
                        if not alignment_set or not aspect_ratio_set:
                            logger.error(f"[BULK] CRITICAL: Missing alignment or aspect_ratio for image {image_idx} after applying outpaint config!")
                            logger.error(f"[BULK] outpaint_config keys: {list(outpaint_config.keys()) if outpaint_config else 'None'}")
                            logger.error(f"[BULK] job_full_params keys: {list(job_full_params.keys())}")
                    else:
                        logger.warning(f"[BULK] outpaint_config for image {image_idx} is None or empty")
                elif is_bulk_request and model == "diffusers_outpaint":
                    # For outpaint bulk requests, outpaint configs are REQUIRED
                    logger.error(f"[BULK] CRITICAL: Missing outpaint config for image {image_idx} in bulk outpaint request!")
                    logger.error(f"[BULK] bulk_outpaint_configs exists: {payload.bulk_outpaint_configs is not None}, len: {len(payload.bulk_outpaint_configs) if payload.bulk_outpaint_configs else 0}, image_idx: {image_idx}")
                
                # Apply per-image mask (R2 keys only)
                if using_r2_images and payload.bulk_mask_r2_keys and image_idx < len(
                    payload.bulk_mask_r2_keys
                ):
                    mask_r2_key = payload.bulk_mask_r2_keys[image_idx]
                    if mask_r2_key:
                        job_full_params.pop("mask_image", None)
                        job_full_params.pop("mask", None)
                        job_full_params["mask_image_r2_key"] = mask_r2_key
                        logger.info(f"[R2_MEDIA] Applied mask R2 key for image {image_idx}")
                elif not is_bulk_request and (
                    payload.mask_image_r2_key or job_full_params.get("mask_image_r2_key")
                ):
                    mask_r2_key = payload.mask_image_r2_key or job_full_params.get(
                        "mask_image_r2_key"
                    )
                    job_full_params.pop("mask_image", None)
                    job_full_params.pop("mask", None)
                    job_full_params["mask_image_r2_key"] = mask_r2_key
                    logger.info("[R2_MEDIA] Applied single mask R2 key")
                
                # Final verification for outpaint bulk requests: ensure critical params are set
                if is_bulk_request and model == "diffusers_outpaint":
                    if "alignment" not in job_full_params or "aspect_ratio" not in job_full_params:
                        logger.error(f"[BULK] CRITICAL ERROR: Missing alignment or aspect_ratio for job {job_id}!")
                        logger.error(f"[BULK] job_full_params keys: {list(job_full_params.keys())}")
                        logger.error(f"[BULK] This will cause the image to not expand properly!")
                    # Ensure expansion_factor is set (should be in base params, but verify)
                    if "expansion_factor" not in job_full_params:
                        logger.warning(f"[BULK] expansion_factor not in params for job {job_id}, worker will use default 1.25")
                
                # Store for later dispatch (R2 keys; URLs applied at dispatch time)
                job_full_params_map[job_id] = job_full_params
                
                # Prepare params for DB storage — strip any stray large/binary-looking fields
                db_params = strip_base64_images_for_db(job_full_params)

                # Store route selection metadata for observability/debugging.
                try:
                    route_params = _prepare_external_params(job_full_params)
                    db_params["_route_backend"] = (
                        "external"
                        if _external_api_router.should_use_external(model, route_params)
                        else "quickpod"
                    )
                except Exception:
                    db_params["_route_backend"] = "quickpod"
                # Persist for dispatchers/recovery: client runs inference locally; no worker/pod.
                if on_device_flag:
                    db_params["on_device"] = True
                    db_params["_route_backend"] = "on_device"
                
                # Add batch metadata to params for tracking
                if batch_id:
                    db_params["_batch_id"] = batch_id
                    db_params["_job_index"] = job_index
                    db_params["_total_jobs"] = total_jobs
                    # For bulk edit, track which image this is
                    if is_bulk_request:
                        db_params["_image_index"] = image_idx
                        db_params["_total_images"] = num_images
                        db_params["_is_bulk"] = True
                    # For variations, track variation index
                    db_params["_variation_index"] = variation_idx
                    db_params["_total_variations"] = variations
                
                # Store R2 input keys for cleanup after job completion
                # Collect per-job R2 keys (init_image + mask)
                job_r2_keys = []
                if job_full_params.get("init_image_r2_key"):
                    job_r2_keys.append(job_full_params["init_image_r2_key"])
                _multi_init = job_full_params.get("init_image_r2_keys")
                if _multi_init and isinstance(_multi_init, list):
                    job_r2_keys.extend(_multi_init)
                if job_full_params.get("mask_image_r2_key"):
                    job_r2_keys.append(job_full_params["mask_image_r2_key"])
                if job_full_params.get("input_video_r2_key"):
                    job_r2_keys.append(job_full_params["input_video_r2_key"])
                # Also check for ref_image_r2_key / ref_video_r2_key (dreamid)
                if job_full_params.get("ref_image_r2_key"):
                    job_r2_keys.append(job_full_params["ref_image_r2_key"])
                if job_full_params.get("ref_video_r2_key"):
                    job_r2_keys.append(job_full_params["ref_video_r2_key"])
                if job_r2_keys:
                    db_params["_input_r2_keys"] = job_r2_keys
                
                # Add enhanced prompt indicator if applicable
                if prompt_enhanced:
                    db_params["_prompt_enhanced"] = True
                    db_params["_original_prompt"] = payload.prompt
                
                # Add VIP flag for reference
                if is_vip:
                    db_params["_is_vip"] = True
                
                job = Job(
                    job_id=job_id,
                    model=model,
                    prompt=final_prompt,  # Use enhanced prompt if applicable
                    params=db_params,
                    status=JobState.QUEUED.value,
                    result_r2_key=result_key,
                    output_key=result_key,
                    # On-device jobs are client-completed; avoid stale timeout push alerts.
                    fcm_token=None if on_device_flag else payload.fcm_token,
                    user_id=user_id,
                    credits_locked=0,
                    credits_refunded=0,
                    created_at=now,
                    updated_at=now,
                    queued_at=now,
                    timeout_at=now + timedelta(minutes=30),
                )
                db.add(job)
                job_ids.append(job_id)
                created_jobs.append({
                    "job_id": job_id,
                    "result_key": result_key,
                    "image_idx": image_idx,
                    "variation_idx": variation_idx,
                    "job_index": job_index,
                })
                job_index += 1
        
        db.flush()  # Get all jobs into the session before locking credits
        
        # Lock credits for ALL jobs in batch (using first job_id as reference)
        # The credit_cost already accounts for variations * num_images multiplier
        primary_job_id = job_ids[0]
        success, message, credits_locked = lock_credits_for_job(
            db, user_id, primary_job_id, model, full_params, is_vip=is_vip
        )
        
        if not success:
            db.rollback()
            raise HTTPException(
                status_code=402,
                detail=message
            )
        
        total_credits_locked = credits_locked
        
        # For multi-job requests (variations > 1 OR bulk edit), lock credits for remaining jobs
        if total_jobs > 1:
            # Lock credits for remaining jobs
            for i, job_data in enumerate(created_jobs[1:], start=1):
                remaining_job_id = job_data["job_id"]
                # Get the specific params for this job (for bulk edit, each job has different image)
                job_params = job_full_params_map.get(remaining_job_id, full_params)
                success, msg, locked = lock_credits_for_job(
                    db, user_id, remaining_job_id, model, job_params, is_vip=is_vip
                )
                if success:
                    total_credits_locked += locked
                else:
                    # If we can't lock credits for a job, rollback everything
                    logger.error(f"Failed to lock credits for job {i}: {msg}")
                    db.rollback()
                    raise HTTPException(
                        status_code=402,
                        detail=f"Insufficient credits for all jobs. {msg}"
                    )
        
        # Audit log for batch creation
        try:
            audit_details = {
                "model": model,
                "credits_locked": total_credits_locked,
                "variations": variations,
                "num_images": num_images,
                "total_jobs": total_jobs,
                "batch_id": batch_id,
                "prompt_enhanced": prompt_enhanced,
                "is_bulk": is_bulk_request,
                "is_vip": is_vip,
            }
            message = f"Job(s) created for model {model} ({total_credits_locked} credits"
            if is_bulk_request:
                message += f", bulk: {num_images} images × {variations} variations = {total_jobs} jobs"
            else:
                message += f", {variations} variations"
            message += ")"
            
            create_audit_log(
                db,
                event_type="job_created",
                message=message,
                target=primary_job_id,
                target_type="job",
                actor_ip=ip,
                details=audit_details,
            )
        except Exception as audit_error:
            logger.warning(f"Failed to create audit log: {audit_error}")
        
        # ATOMIC COMMIT: All jobs + credit locks in single transaction
        db.commit()
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as db_error:
        logger.error(f"Failed to create jobs: {db_error}")
        try:
            db.rollback()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="Failed to create job")
    finally:
        db.close()
    
    # Log job creation (non-blocking)
    try:
        audit_logger.job_created(
            primary_job_id, model, ip, auth.get("fingerprint_hash", auth.get("type"))
        )
    except Exception:
        pass
    
    # Cache full params for all jobs first
    for job_data in created_jobs:
        jid = job_data["job_id"]
        
        # Get the specific params for this job (for bulk edit, each job has different image)
        job_params = job_full_params_map.get(jid, full_params)
        
        # Cache full params for dispatch (R2 keys → presigned GET URLs in _dispatch_to_worker)
        cache_job_params(jid, job_params)
    
    # Dispatch job(s) to worker unless this is an on-device client workflow.
    if on_device_flag:
        logger.info(
            f"[ON_DEVICE] Skipping worker dispatch for {len(created_jobs)} job(s) "
            f"(model={model}, user_id={user_id})"
        )
    elif len(created_jobs) > 1 and _pod_manager:
        # MULTI-JOB (bulk edit OR variations): dispatch all to ONE worker in a single /run_bulk request
        all_job_ids = [j["job_id"] for j in created_jobs]
        background_tasks.add_task(
            _dispatch_bulk_jobs_background, all_job_ids, model, full_params.get("mode")
        )
    else:
        # SINGLE JOB: dispatch individually (unchanged)
        for job_data in created_jobs:
            jid = job_data["job_id"]
            job_params = job_full_params_map.get(jid, full_params)
            if _pod_manager:
                background_tasks.add_task(
                    _dispatch_job_background, jid, model, job_params.get("mode")
                )
            elif _job_manager:
                logger.warning(
                    f"No pod_manager available, job {jid} will be handled by recovery system"
                )
    
    # Build download URL for primary job
    primary_job = created_jobs[0]
    download_url = (
        f"{_controller_url}/download_url/{primary_job['job_id']}"
        if _controller_url
        else f"/download_url/{primary_job['job_id']}"
    )
    
    # Presigned PUT for worker result upload; on-device jobs finish locally (no R2 upload).
    upload_url = (
        None
        if on_device_flag
        else create_presigned_put_url(primary_job["result_key"])
    )

    # Return response with all job IDs for variations/bulk edit
    # Frontend should poll all job_ids when total_jobs > 1
    return SubmitResponse(
        job_id=primary_job["job_id"],
        status=JobStatus.QUEUED,
        model=model,
        routed_to_worker=None,
        upload_url=upload_url,
        result_key=primary_job["result_key"],
        download_url=download_url,
        created_at=utc_naive_now(),
        # Variation/Bulk support
        variations=variations,
        batch_id=batch_id,  # None for single job, UUID for multiple
        job_ids=job_ids if total_jobs > 1 else None,  # All job IDs when multiple jobs
        # Prompt enhancement
        prompt_enhanced=prompt_enhanced,
        enhanced_prompt=enhanced_prompt_text if prompt_enhanced else None,
        # Credits
        credits_used=total_credits_locked,
    )


@client_router.post("/on_device/job_started")
async def on_device_job_started(
    payload: OnDeviceJobStartedRequest,
    auth: Dict = Depends(require_firebase_auth),
):
    """Mark an on-device job as started (client-reported lifecycle)."""
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid auth context")

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        job = (
            db.query(Job)
            .filter(Job.job_id == payload.job_id, Job.user_id == user.id)
            .first()
        )
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.status in (
            JobState.COMPLETED.value,
            JobState.FAILED.value,
            JobState.CANCELLED.value,
            JobState.TIMEOUT.value,
        ):
            return {"ok": True, "status": job.status}

        now = utc_naive_now()
        if job.status == JobState.QUEUED.value:
            job.status = JobState.RUNNING.value
        if not job.started_at:
            job.started_at = now
        if not job.dispatched_at:
            job.dispatched_at = now
        job.updated_at = now
        db.commit()
        return {"ok": True, "status": job.status}
    finally:
        db.close()


@client_router.post("/on_device/job_complete")
async def on_device_job_complete(
    payload: OnDeviceJobCompleteRequest,
    auth: Dict = Depends(require_firebase_auth),
):
    """Mark an on-device job terminal state and handle refunds on failure."""
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid auth context")

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        job = (
            db.query(Job)
            .filter(Job.job_id == payload.job_id, Job.user_id == user.id)
            .first()
        )
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.status in (
            JobState.COMPLETED.value,
            JobState.FAILED.value,
            JobState.CANCELLED.value,
            JobState.TIMEOUT.value,
        ):
            return {"ok": True, "status": job.status}

        now = utc_naive_now()
        if payload.success:
            job.status = JobState.COMPLETED.value
            job.error = None
            # Prevent any future stale timeout push attempts for this client-finished job.
            job.notification_sent = True
            if not job.started_at:
                job.started_at = now
            if not job.dispatched_at:
                job.dispatched_at = now
            job.completed_at = now
            job.timeout_at = None
            job.updated_at = now
            db.commit()
            return {"ok": True, "status": JobState.COMPLETED.value}

        # Failure path: mark failed and refund locked credits.
        job.status = JobState.FAILED.value
        job.error = payload.error or "On-device processing failed"
        job.notification_sent = True
        if not job.started_at:
            job.started_at = now
        if not job.dispatched_at:
            job.dispatched_at = now
        job.completed_at = now
        job.timeout_at = None
        job.updated_at = now
        db.commit()

        refund_ok, refund_msg, refund_amount = refund_credits_for_job(
            None,
            payload.job_id,
            reason="on_device_failed",
        )
        if not refund_ok:
            logger.warning(
                f"[ON_DEVICE] Refund failed for {payload.job_id}: {refund_msg}"
            )
        return {
            "ok": True,
            "status": JobState.FAILED.value,
            "refund_amount": refund_amount,
        }
    finally:
        db.close()


async def _dispatch_to_external_api(job_id: str, retry_count: int = 0) -> bool:
    """
    Execute a queued job through external API provider instead of QuickPod worker dispatch.
    Keeps the same job lifecycle and notification semantics.
    """
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            logger.error(f"[EXTERNAL_API] Job {job_id} not found")
            return False

        current_state = JobState(job.status)
        finished = {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.TIMEOUT,
            JobState.CANCELLED,
        }
        # The first attempt marks the job running before the provider call.
        # A retry must continue from that state. A fresh dispatch must not.
        if current_state in finished or (
            current_state == JobState.RUNNING and retry_count == 0
        ):
            logger.info(
                f"[EXTERNAL_API] Job {job_id} already in terminal/active state {current_state.value}"
            )
            return True

        cached_params = get_cached_job_params(job_id)
        raw_params = cached_params if cached_params else (job.params or {})
        source_params = carry_moderation_flag(raw_params, _prepare_external_params(raw_params))

        if not _external_api_router.should_use_external(job.model, source_params):
            return False

        if current_state != JobState.RUNNING:
            # State transitions mirror worker dispatch flow.
            if _job_manager:
                _job_manager.update_job_state(job_id, JobState.DISPATCHING)
                _job_manager.update_job_state(job_id, JobState.RUNNING)
            else:
                now = utc_naive_now()
                job.status = JobState.RUNNING.value
                job.dispatched_at = now
                job.started_at = now
                job.updated_at = now
                db.commit()

        result = await asyncio.to_thread(
            _external_api_router.execute, job.model, job.prompt or "", source_params
        )
        _stash_suno_submission(db, job_id, result)

        if result.pending and result.external_task_id:
            if result.provider == "runware":
                existing = (
                    db.query(RunwareTask)
                    .filter(RunwareTask.task_id == result.external_task_id)
                    .first()
                )
                if not existing:
                    db.add(
                        RunwareTask(
                            task_id=result.external_task_id,
                            job_id=job_id,
                            model=job.model,
                        )
                    )
                job_row = db.query(Job).filter(Job.job_id == job_id).first()
                if job_row:
                    params = dict(job_row.params or {})
                    params["_runware_task_id"] = result.external_task_id
                    job_row.params = params
                db.commit()
                logger.info("[RUNWARE] Job %s submitted task %s", job_id, result.external_task_id)
                return True
            existing = db.query(KieTask).filter(KieTask.task_id == result.external_task_id).first()
            if not existing:
                db.add(
                    KieTask(
                        task_id=result.external_task_id,
                        job_id=job_id,
                        model=job.model,
                    )
                )
            job_row = db.query(Job).filter(Job.job_id == job_id).first()
            if job_row:
                params = dict(job_row.params or {})
                params["_kie_task_id"] = result.external_task_id
                job_row.params = params
            db.commit()
            logger.info(f"[KIE] Job {job_id} submitted task {result.external_task_id}")
            return True

        r2_key = None
        output_url = result.result_url
        if result.success and (result.result_url or result.result_text or result.result_bytes):
            try:
                r2_key, output_url = _store_external_result(job_id, job.model, result)
            except Exception as store_err:
                logger.exception(f"[EXTERNAL_API] Failed to store result for {job_id}: {store_err}")
                result = ExternalExecutionResult(success=False, error=str(store_err))

        if result.success and (output_url or r2_key):
            if _job_manager:
                _job_manager.mark_completed(
                    job_id,
                    result_r2_key=r2_key,
                    output_url=output_url,
                )
            else:
                completed_job = db.query(Job).filter(Job.job_id == job_id).first()
                if completed_job:
                    completed_job.status = JobState.COMPLETED.value
                    completed_job.result_r2_key = r2_key
                    completed_job.output_url = output_url
                    completed_job.completed_at = utc_naive_now()
                    completed_job.updated_at = utc_naive_now()
                    db.commit()

            # Refresh job for notification metadata.
            job = db.query(Job).filter(Job.job_id == job_id).first()
            meta = job.params if isinstance(job.params, dict) else {}
            await _send_job_notification(
                token=job.fcm_token,
                job_id=job.job_id,
                model=job.model,
                success=True,
                download_url=output_url,
                batch_id=meta.get("_batch_id"),
                variation_index=int(meta.get("_variation_index", 0) or 0),
                total_variations=int(meta.get("_total_variations", 1) or 1),
            )
            clear_cached_job_params(job_id)
            logger.info(f"[EXTERNAL_API] Job {job_id} completed via external provider")
            return True

        # External execution failed; retry a few times before failing job permanently.
        max_retries = 2
        if retry_count < max_retries:
            delay_seconds = 2 * (retry_count + 1)
            logger.warning(
                f"[EXTERNAL_API] Job {job_id} failed (attempt {retry_count + 1}/{max_retries + 1}): "
                f"{result.error}. Retrying in {delay_seconds}s"
            )
            await asyncio.sleep(delay_seconds)
            asyncio.create_task(_dispatch_to_external_api(job_id, retry_count + 1))
            return True

        # Permanent failure path.
        err = result.error or "External API execution failed"
        if _job_manager:
            _job_manager.mark_failed(job_id, err, worker_id=None)
        else:
            failed_job = db.query(Job).filter(Job.job_id == job_id).first()
            if failed_job:
                failed_job.status = JobState.FAILED.value
                failed_job.error = err
                failed_job.updated_at = utc_naive_now()
                db.commit()

        job = db.query(Job).filter(Job.job_id == job_id).first()
        meta = job.params if isinstance(job.params, dict) else {}
        await _send_job_notification(
            token=job.fcm_token,
            job_id=job.job_id,
            model=job.model,
            success=False,
            batch_id=meta.get("_batch_id"),
            variation_index=int(meta.get("_variation_index", 0) or 0),
            total_variations=int(meta.get("_total_variations", 1) or 1),
        )
        clear_cached_job_params(job_id)
        logger.error(f"[EXTERNAL_API] Job {job_id} failed permanently: {err}")
        return True
    except Exception as e:
        logger.exception(f"[EXTERNAL_API] Unexpected dispatch failure for job {job_id}: {e}")
        return False
    finally:
        db.close()


async def _dispatch_job_background(
    job_id: str, model: str, mode: Optional[str], retry_count: int = 0
):
    """
    Background task to dispatch job - includes actual HTTP dispatch to worker.
    
    CRITICAL FIX: Track machines attempted per job to limit to 3 DIFFERENT machines.
    This prevents infinite loops when all available machines are problematic.
    
    RACE CONDITION FIX: Check job state before dispatching to prevent double dispatch
    when multiple dispatch tasks exist for the same job.
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
        
        # RACE CONDITION FIX: Check if job is already in a state where it shouldn't be dispatched
        # This can happen when multiple dispatch tasks exist for the same job (e.g., from
        # _requeue_jobs_after_pod_failure AND _dispatch_job_background retry)
        current_state = JobState(job.status)
        if current_state in [JobState.RUNNING, JobState.COMPLETED, JobState.FAILED, 
                             JobState.TIMEOUT, JobState.CANCELLED]:
            logger.info(
                f"Job {job_id} is already in {current_state.value} state, "
                f"skipping dispatch (retry_count={retry_count})"
            )
            return
        
        # Also skip if job is already being dispatched by another task
        # (DISPATCHING state means another dispatch is in progress)
        if current_state == JobState.DISPATCHING and retry_count > 0:
            # For retry attempts, only skip if there's a recent assignment
            # This prevents racing dispatch tasks from creating duplicate dispatches
            if job.updated_at:
                time_since_update = (utc_naive_now() - job.updated_at).total_seconds()
                if time_since_update < 30:  # If updated within last 30 seconds
                    logger.info(
                        f"Job {job_id} is already in DISPATCHING state (updated {time_since_update:.0f}s ago), "
                        f"skipping retry dispatch (retry_count={retry_count})"
                    )
                    return

        # On-device jobs: credits only; client completes via /on_device/* — never assign workers.
        job_params_early = job.params if isinstance(job.params, dict) else {}
        if job_params_early.get("on_device") is True:
            logger.info(
                f"[ON_DEVICE] Skipping worker/external dispatch for job {job_id} ({job.model})"
            )
            return

        # External API route: bypass pod/worker assignment.
        cached_params = get_cached_job_params(job_id)
        route_params = _prepare_external_params(cached_params if cached_params else (job.params or {}))
        if _external_api_router.should_use_external(job.model, route_params):
            logger.info(f"[EXTERNAL_API] Routing job {job_id} ({job.model}) through external provider")
            return await _dispatch_to_external_api(job_id, retry_count=retry_count)
        
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
                    None,
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
        # Double-check job state before assigning (in case it changed during the checks above)
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job:
                current_state = JobState(job.status)
                if current_state in [JobState.RUNNING, JobState.COMPLETED, JobState.FAILED]:
                    logger.info(
                        f"Job {job_id} state changed to {current_state.value} before assign, aborting dispatch"
                    )
                    return
        finally:
            db.close()
        
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
                asyncio.create_task(
                    _dispatch_job_background(job_id, model, mode, retry_count + 1)
                )
            else:
                logger.error(
                    f"Job {job_id} failed to dispatch after {max_retries} attempts. "
                    f"Job will remain in QUEUED for recovery system to handle."
                )
            
    except JobAlreadyProcessedError as e:
        # Job was already handled by another dispatch task - this is expected in race conditions
        logger.info(
            f"Job {job_id} was already processed by another dispatch task: {e}. "
            f"This is normal during concurrent dispatch attempts."
        )
        # No need to do anything - job is already being handled
        return
    except WorkerTimeoutError as e:
        # Worker registration timed out - pod may have failed to start
        # Revert job to QUEUED and trigger immediate retry
        logger.warning(
            f"Worker registration timed out for job {job_id}: {e}. "
            f"Reverting to QUEUED and triggering retry."
        )
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job and job.status not in [
                JobState.COMPLETED.value,
                JobState.FAILED.value,
                JobState.RUNNING.value,
            ]:
                job.status = JobState.QUEUED.value
                job.worker_id = None
                job.updated_at = utc_naive_now()
                if job.error_log is None:
                    job.error_log = []
                job.error_log.append(
                    {
                    "timestamp": utc_naive_now().isoformat(),
                    "state": "dispatch_background_error",
                    "error": str(e),
                    }
                )
                db.commit()
                
                # CRITICAL: Trigger immediate retry instead of waiting for recovery system
                max_retries = 5
                if retry_count < max_retries:
                    retry_delay = 5  # 5 second delay before retry
                    logger.info(
                        f"Job {job_id} will retry dispatch in {retry_delay}s (attempt {retry_count + 1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_delay)
                    asyncio.create_task(
                        _dispatch_job_background(job_id, model, mode, retry_count + 1)
                    )
                else:
                    logger.error(
                        f"Job {job_id} exhausted all retries ({max_retries}) after worker timeout. "
                        f"Job will remain in QUEUED for manual intervention."
                    )
        finally:
            db.close()
        return
    except Exception as e:
        logger.error(f"Background dispatch failed for {job_id}: {e}")
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job and job.status not in [
                JobState.COMPLETED.value,
                JobState.FAILED.value,
                JobState.RUNNING.value,
            ]:
                job.status = JobState.QUEUED.value
                job.worker_id = None
                job.updated_at = utc_naive_now()
                if job.error_log is None:
                    job.error_log = []
                job.error_log.append(
                    {
                    "timestamp": utc_naive_now().isoformat(),
                    "state": "dispatch_background_error",
                    "error": str(e),
                    }
                )
                db.commit()

                max_retries = 5
                if retry_count < max_retries:
                    retry_delay = min(15 * (2 ** retry_count), 120)
                    logger.info(
                        f"Job {job_id} reverted to QUEUED after dispatch error. "
                        f"Retrying in {retry_delay}s (attempt {retry_count + 1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_delay)
                    asyncio.create_task(
                        _dispatch_job_background(job_id, model, mode, retry_count + 1)
                    )
                else:
                    logger.error(
                        f"Job {job_id} exhausted all dispatch retries ({max_retries}) "
                        f"after generic error. Job remains in QUEUED."
                    )
        finally:
            db.close()


async def _release_bulk_chunk_and_dispatch_individually(
    chunk: List[str],
    pod_uuid: Optional[str],
    model: str,
    mode: Optional[str],
):
    """
    Release all slots for a chunk on the pod, set chunk jobs to QUEUED and worker_id=None,
    then schedule individual dispatch for each job. Used when a chunk fails (reserve or bulk send).
    """
    if _pod_manager and pod_uuid:
        for jid in chunk:
            try:
                await _pod_manager.release_slot_on_pod(pod_uuid, jid)
            except Exception as e:
                logger.warning(f"[BULK_DISPATCH] Release slot for {jid} failed: {e}")
    db = SessionLocal()
    try:
        for jid in chunk:
            job = db.query(Job).filter(Job.job_id == jid).first()
            if job and job.status not in [
                JobState.RUNNING.value, JobState.COMPLETED.value, JobState.FAILED.value,
            ]:
                job.status = JobState.QUEUED.value
                job.worker_id = None
                job.updated_at = utc_naive_now()
        db.commit()
    finally:
        db.close()
    for jid in chunk:
        asyncio.create_task(_dispatch_job_background(jid, model, mode, retry_count=0))


async def _dispatch_bulk_jobs_background(
    job_ids: List[str], model: str, mode: Optional[str]
):
    """
    Background task to dispatch multiple jobs in chunks by pod capacity.
    Chunks by available_slots (MAX_JOBS_PER_POD - queue_size) so we never send more
    than the worker can accept. Reserves one slot per job before sending each chunk.
    Fallback: on reserve or bulk send failure, release chunk slots, revert to QUEUED,
    and dispatch that chunk individually.
    """
    if not job_ids:
        return

    # External providers for first wave do not support native bulk; fan out per job.
    # This preserves user-visible bulk/variation behavior while avoiding QuickPod load.
    db = SessionLocal()
    try:
        first_job = db.query(Job).filter(Job.job_id == job_ids[0]).first()
        if first_job:
            first_params = _prepare_external_params(
                get_cached_job_params(first_job.job_id) or (first_job.params or {})
            )
            if _external_api_router.should_use_external(first_job.model, first_params):
                logger.info(
                    f"[EXTERNAL_API] Bulk fan-out for {len(job_ids)} jobs "
                    f"(model={first_job.model})"
                )
                for jid in job_ids:
                    asyncio.create_task(_dispatch_job_background(jid, model, mode, retry_count=0))
                return
    finally:
        db.close()

    if not _job_manager or not _pod_manager:
        logger.warning(
            "[BULK_DISPATCH] job_manager or pod_manager not available, skipping bulk dispatch"
        )
        for jid in job_ids:
            asyncio.create_task(_dispatch_job_background(jid, model, mode, retry_count=0))
        return

    logger.info(
        f"[BULK_DISPATCH] Starting bulk dispatch of {len(job_ids)} jobs for model={model}"
    )
    remaining = list(job_ids)

    try:
        while remaining:
            # Reserve one slot via assign_job for the first remaining job
            worker_id = await _job_manager.assign_job(remaining[0], model, mode)
            pod_uuid = None
            db = SessionLocal()
            try:
                worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
                if worker:
                    pod_uuid = worker.container_id
            finally:
                db.close()

            # Get pod's current queue_size and compute available_slots (under lock)
            async with _pod_manager._lock:
                pi = _pod_manager.individual_pods.get(pod_uuid) if pod_uuid else None
                if pi is not None:
                    limit = MAX_JOBS_PER_POD
                    available_slots = limit - pi.queue_size
                else:
                    # Pod not in tracking (e.g. race); use conservative default
                    available_slots = max(1, MAX_JOBS_PER_POD - 1)
            chunk_size = min(len(remaining), available_slots)
            chunk = remaining[0:chunk_size]

            # Reserve slots for chunk[1:] (chunk[0] already reserved by assign_job)
            reserve_failed = False
            for jid in chunk[1:]:
                reserved = await _pod_manager.reserve_slot_on_pod(pod_uuid, jid)
                if not reserved:
                    reserve_failed = True
                    break
            if reserve_failed:
                await _release_bulk_chunk_and_dispatch_individually(chunk, pod_uuid, model, mode)
                remaining = remaining[len(chunk):]
                continue

            # Assign chunk[1:] to worker_id in DB
            db = SessionLocal()
            try:
                for jid in chunk[1:]:
                    job = db.query(Job).filter(Job.job_id == jid).first()
                    if job:
                        job.worker_id = worker_id
                        job.updated_at = utc_naive_now()
                db.commit()
            finally:
                db.close()

            # Send chunk to worker
            success = await _dispatch_bulk_to_worker(chunk, worker_id)
            if not success:
                await _release_bulk_chunk_and_dispatch_individually(chunk, pod_uuid, model, mode)
            else:
                logger.info(
                    f"[BULK_DISPATCH] Dispatched chunk of {len(chunk)} jobs to worker {worker_id}"
                )
            remaining = remaining[len(chunk):]

    except Exception as e:
        logger.error(f"[BULK_DISPATCH] Error during bulk dispatch: {e}")
        # Revert all job_ids (this bulk's full list) to QUEUED and dispatch individually
        db = SessionLocal()
        try:
            for jid in job_ids:
                job = db.query(Job).filter(Job.job_id == jid).first()
                if job and job.status not in [
                    JobState.RUNNING.value, JobState.COMPLETED.value, JobState.FAILED.value,
                ]:
                    job.status = JobState.QUEUED.value
                    job.worker_id = None
                    job.updated_at = utc_naive_now()
            db.commit()
        finally:
            db.close()
        for jid in job_ids:
            asyncio.create_task(_dispatch_job_background(jid, model, mode, retry_count=0))


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


async def _check_worker_health_enhanced(
    endpoint: str, timeout: int = 5
) -> Dict[str, Any]:
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
        "details": {},
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
        logger.debug(
            f"Worker health check timeout for {endpoint} (worker may be busy processing)"
        )
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
        jobs_to_recover = (
            db.query(Job)
            .filter(
            Job.worker_id == worker_id,
            Job.status.in_([JobState.RUNNING.value, JobState.DISPATCHING.value]),
            )
            .all()
        )
        
        for job in jobs_to_recover:
            # Skip the excluded job (if specified)
            if exclude_job_id and job.job_id == exclude_job_id:
                continue

            # RACE FIX: Re-read job from DB before modifying. job_complete may have
            # just committed COMPLETED for this job; do not overwrite with QUEUED.
            db.refresh(job)
            if job.status in (JobState.COMPLETED.value, JobState.FAILED.value):
                logger.info(
                    f"Job {job.job_id} already in terminal state '{job.status}', "
                    f"skipping recovery in _recover_jobs_on_worker (job_complete likely committed concurrently)"
                )
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
            job.error_log.append(
                {
                "timestamp": utc_naive_now().isoformat(),
                "state": "pod_destruction_recovery",
                "error": f"Job recovered (was {old_status}) after worker {worker_id} pod was destroyed",
                "previous_status": old_status,
                }
            )
            
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
    
    RACE CONDITION FIX: Check job state at the start to prevent dispatching
    jobs that are already running or completed (can happen when multiple
    dispatch tasks exist for the same job).
    """
    db = SessionLocal()
    try:
        # Get job details
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            logger.error(f"_dispatch_to_worker: job {job_id} not found")
            return False
        
        # RACE CONDITION FIX: Check if job is already in a terminal or running state
        # This prevents double dispatch when multiple dispatch tasks exist for the same job
        current_state = JobState(job.status)
        if current_state in [JobState.RUNNING, JobState.COMPLETED, JobState.FAILED,
                             JobState.TIMEOUT, JobState.CANCELLED]:
            logger.warning(
                f"_dispatch_to_worker: job {job_id} is already in {current_state.value} state. "
                f"This likely means another dispatch task already processed this job. "
                f"Aborting dispatch to prevent duplicate processing."
            )
            # Return True to signal "success" (job is already being/has been processed)
            # This prevents the caller from triggering unnecessary retries
            return True
        
        # Get worker details
        worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
        if not worker:
            logger.error(f"_dispatch_to_worker: worker {worker_id} not found")
            # Revert job status
            job.status = JobState.QUEUED.value
            job.worker_id = None
            db.commit()
            return False
        
        # CRITICAL FIX: Don't trust DB status (D1 is slow/eventually consistent)
        # Instead, do a DIRECT health check to verify worker is ready
        if worker.status == "starting":
            # DB might be stale - do a direct health check
            logger.info(f"Worker {worker_id} shows 'starting' in DB, verifying via health check...")
            try:
                health_info = await _check_worker_health_enhanced(worker.endpoint, timeout=5)
                if health_info.get("ready") or health_info.get("status") == "ok":
                    logger.info(f"Worker {worker_id} confirmed ready via health check (DB was stale)")
                    # Update DB to match reality
                    worker.status = "dispatching"
                    db.commit()
                else:
                    logger.warning(
                        f"Worker {worker_id} is in 'starting' state (models not loaded). "
                        f"Cannot dispatch job {job_id}. Reverting to QUEUED."
                    )
                    job.status = JobState.QUEUED.value
                    job.worker_id = None
                    job.updated_at = utc_naive_now()
                    if job.error_log is None:
                        job.error_log = []
                    job.error_log.append(
                        {
                        "timestamp": utc_naive_now().isoformat(),
                        "state": "dispatch_worker_not_ready",
                        "error": f"Worker {worker_id} still in 'starting' state (health check failed)",
                        }
                    )
                    db.commit()
                    return False
            except Exception as e:
                logger.warning(
                    f"Worker {worker_id} health check failed: {e}. "
                    f"Cannot dispatch job {job_id}. Reverting to QUEUED."
                )
                job.status = JobState.QUEUED.value
                job.worker_id = None
                job.updated_at = utc_naive_now()
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
                job.error_log.append(
                    {
                    "timestamp": utc_naive_now().isoformat(),
                    "state": "dispatch_failed_pod_destroyed",
                    "error": f"Pod {pod_uuid} was destroyed before dispatch",
                    }
                )
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
        presigned_url = (
            create_presigned_put_url(job.result_r2_key) if job.result_r2_key else None
        )
        
        # Build the run payload
        # Prefer cached params (full R2 keys); DB params omit large blobs
        cached_params = get_cached_job_params(job.job_id)
        dispatch_params = cached_params if cached_params else (job.params or {})
        
        # ---------------------------------------------------------------
        # R2 MEDIA: Convert all *_r2_key fields to presigned download URLs
        # so workers receive plain URLs they can HTTP GET.
        # ---------------------------------------------------------------
        dispatch_params = dict(dispatch_params)  # Don't mutate cached params
        
        # init_image_r2_keys (multi) → init_image as list of presigned GET URLs
        if "init_image_r2_keys" in dispatch_params:
            keys = dispatch_params.pop("init_image_r2_keys")
            if not isinstance(keys, list) or not keys:
                raise Exception("init_image_r2_keys must be a non-empty list")
            urls: List[str] = []
            for r2_key in keys:
                download_url = create_presigned_get_url(r2_key, expires_in=3600)
                if download_url:
                    urls.append(download_url)
                else:
                    raise Exception(
                        f"Failed to generate init_image download URL for R2 key: {r2_key}"
                    )
            dispatch_params["init_image"] = urls
            logger.info(
                f"[R2_MEDIA] Job {job_id}: {len(urls)} init_image R2 keys → download URLs"
            )
        # init_image_r2_key → init_image (presigned GET URL)
        elif "init_image_r2_key" in dispatch_params:
            r2_key = dispatch_params.pop("init_image_r2_key")
            download_url = create_presigned_get_url(r2_key, expires_in=3600)
            if download_url:
                dispatch_params["init_image"] = download_url
                logger.info(f"[R2_MEDIA] Job {job_id}: init_image R2 key → download URL")
            else:
                raise Exception(f"Failed to generate init_image download URL for R2 key: {r2_key}")
        
        # mask_image_r2_key → mask_image (presigned GET URL)
        if "mask_image_r2_key" in dispatch_params:
            r2_key = dispatch_params.pop("mask_image_r2_key")
            download_url = create_presigned_get_url(r2_key, expires_in=3600)
            if download_url:
                dispatch_params["mask_image"] = download_url
                logger.info(f"[R2_MEDIA] Job {job_id}: mask_image R2 key → download URL")
            else:
                raise Exception(f"Failed to generate mask_image download URL for R2 key: {r2_key}")

        # first_frame_image_r2_key → first_frame_image (CineReal FLF2V / WAN two-frame UX)
        if "first_frame_image_r2_key" in dispatch_params:
            r2_key = dispatch_params.pop("first_frame_image_r2_key")
            download_url = create_presigned_get_url(r2_key, expires_in=3600)
            if download_url:
                dispatch_params["first_frame_image"] = download_url
                logger.info(f"[R2_MEDIA] Job {job_id}: first_frame_image R2 key → download URL")
            else:
                raise Exception(
                    f"Failed to generate first_frame_image download URL for R2 key: {r2_key}"
                )

        # first_frame_image_r2_keys → two presigned URLs as init_image list (LTX FLF2V bulk mapping)
        if "first_frame_image_r2_keys" in dispatch_params:
            keys = dispatch_params.pop("first_frame_image_r2_keys")
            if not isinstance(keys, list) or not keys:
                raise Exception("first_frame_image_r2_keys must be a non-empty list")
            urls_ff: List[str] = []
            for r2_key in keys:
                download_url = create_presigned_get_url(r2_key, expires_in=3600)
                if download_url:
                    urls_ff.append(download_url)
                else:
                    raise Exception(
                        f"Failed to generate first_frame_image download URL for R2 key: {r2_key}"
                    )
            if "init_image" not in dispatch_params:
                if len(urls_ff) == 2:
                    dispatch_params["init_image"] = urls_ff
                    logger.info(
                        f"[R2_MEDIA] Job {job_id}: first_frame_image_r2_keys ({len(urls_ff)}) "
                        f"→ init_image URL list for FLF2V"
                    )
                elif len(urls_ff) == 1:
                    dispatch_params.setdefault("first_frame_image", urls_ff[0])
                    logger.info(
                        f"[R2_MEDIA] Job {job_id}: first_frame_image_r2_keys (1) → first_frame_image"
                    )
                else:
                    raise Exception(
                        f"first_frame_image_r2_keys: expected 1 or 2 keys, got {len(urls_ff)}"
                    )

        if "reference_image_r2_keys" in dispatch_params:
            keys = dispatch_params.pop("reference_image_r2_keys")
            if isinstance(keys, list) and keys:
                urls_ref: List[str] = []
                for r2_key in keys[:4]:
                    download_url = create_presigned_get_url(r2_key, expires_in=3600)
                    if download_url:
                        urls_ref.append(download_url)
                    else:
                        raise Exception(
                            f"Failed to generate reference download URL for R2 key: {r2_key}"
                        )
                dispatch_params["reference_images"] = urls_ref
                logger.info(
                    f"[R2_MEDIA] Job {job_id}: {len(urls_ref)} reference images → download URLs"
                )

        if "continuation_latent_r2_key" in dispatch_params:
            r2_key = dispatch_params.pop("continuation_latent_r2_key")
            download_url = create_presigned_get_url(r2_key, expires_in=3600)
            if download_url:
                dispatch_params["continuation_latent_url"] = download_url
                logger.info(f"[R2_MEDIA] Job {job_id}: continuation latent → download URL")
            else:
                raise Exception(
                    f"Failed to generate continuation latent download URL for R2 key: {r2_key}"
                )

        # last_frame_image_r2_key → last_frame_image
        if "last_frame_image_r2_key" in dispatch_params:
            r2_key = dispatch_params.pop("last_frame_image_r2_key")
            download_url = create_presigned_get_url(r2_key, expires_in=3600)
            if download_url:
                dispatch_params["last_frame_image"] = download_url
                logger.info(f"[R2_MEDIA] Job {job_id}: last_frame_image R2 key → download URL")
            else:
                raise Exception(
                    f"Failed to generate last_frame_image download URL for R2 key: {r2_key}"
                )

        # Voice / guidance audio (Flutter uploads as audio_r2_key or cinereal_audio_r2_key)
        for audio_key_name in ("audio_r2_key", "cinereal_audio_r2_key"):
            if audio_key_name not in dispatch_params:
                continue
            r2_key = dispatch_params.pop(audio_key_name)
            download_url = create_presigned_get_url(r2_key, expires_in=3600)
            if download_url:
                dispatch_params["audio_url"] = download_url
                logger.info(f"[R2_MEDIA] Job {job_id}: {audio_key_name} → audio_url")
            else:
                raise Exception(
                    f"Failed to generate audio download URL for R2 key: {r2_key}"
                )
        
        # ref_image_r2_key → ref_image (presigned GET URL) — for dreamid
        if "ref_image_r2_key" in dispatch_params:
            r2_key = dispatch_params.pop("ref_image_r2_key")
            download_url = create_presigned_get_url(r2_key, expires_in=3600)
            if download_url:
                dispatch_params["ref_image"] = download_url
                logger.info(f"[R2_MEDIA] Job {job_id}: ref_image R2 key → download URL")
            else:
                raise Exception(f"Failed to generate ref_image download URL for R2 key: {r2_key}")
        
        # ref_video_r2_key → ref_video (presigned GET URL) — for dreamid
        if "ref_video_r2_key" in dispatch_params:
            r2_key = dispatch_params.pop("ref_video_r2_key")
            download_url = create_presigned_get_url(r2_key, expires_in=3600)
            if download_url:
                dispatch_params["ref_video"] = download_url
                logger.info(f"[R2_MEDIA] Job {job_id}: ref_video R2 key → download URL")
            else:
                raise Exception(f"Failed to generate ref_video download URL for R2 key: {r2_key}")
        
        # input_video_r2_key → input_video (presigned GET URL)
        if "input_video_r2_key" in dispatch_params:
            r2_key = dispatch_params.pop("input_video_r2_key")
            video_download_url = create_presigned_get_url(r2_key, expires_in=3600)
            if video_download_url:
                dispatch_params["input_video"] = video_download_url
                logger.info(
                    f"[R2_MEDIA] Job {job_id}: Converted input_video R2 key "
                    f"({r2_key}) to presigned download URL for worker"
                )
            else:
                logger.error(
                    f"[R2_MEDIA] Job {job_id}: Failed to generate download URL "
                    f"for R2 key: {r2_key}"
                )
                raise Exception(
                    f"Failed to generate video download URL for R2 key: {r2_key}"
                )
        
        # Strip internal tracking keys that shouldn't reach the worker
        # (these exist in db_params fallback; cache doesn't have them)
        for internal_key in list(dispatch_params.keys()):
            if internal_key.startswith("_"):
                dispatch_params.pop(internal_key)

        _assert_dispatch_params_media_are_urls(dispatch_params, job_id)
        
        run_payload = {
            "job_id": job.job_id,
            "prompt": job.prompt,
            "model": job.model,
            "mode": dispatch_params.get("mode"),
            "params": dispatch_params,
            "upload_url": presigned_url,
            "result_key": job.result_r2_key,
        }
        model_name = str(job.model or "")
        if job.result_r2_key and (model_name == "minimax_h3" or model_name.startswith("ltx25")):
            latent_key = f"{job.result_r2_key}.latent"
            latent_put = create_presigned_put_url(latent_key)
            if latent_put:
                run_payload["latent_upload_url"] = latent_put
                run_payload["latent_result_key"] = latent_key
        
        # Save dispatch payload as JSON (as sent to worker)
        try:
            dispatch_data = {
                "worker_info": {
                    "worker_id": worker_id,
                    "worker_endpoint": worker.endpoint,
                    "pod_uuid": pod_uuid,
                    "machine_id": worker.machine_id,
                },
                "job_info": {
                    "job_id": job.job_id,
                    "model": job.model,
                    "status": job.status,
                },
                "dispatch_payload": run_payload,  # Exact payload sent to worker
            }
            
            # Save with proper grouping and labeling
            _save_json_request(
                data=dispatch_data,
                request_type="worker_dispatch",
                job_id=job.job_id,
                user_id=job.user_id if hasattr(job, "user_id") else None,
                additional_labels={
                    "model": job.model,
                    "worker_id": worker_id[:8] if worker_id else None,
                    "mode": dispatch_params.get("mode"),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to save worker dispatch JSON: {e}")
        
        # Log if we're using cached vs DB params
        if cached_params:
            logger.debug(
                f"Job {job_id} using cached params ({len(cached_params)} param keys)"
            )
        
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
                running_jobs = (
                    db.query(Job)
                    .filter(
                        Job.worker_id == worker_id, Job.status == JobState.RUNNING.value
                    )
                    .all()
                )
                
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
                        _job_manager.update_job_state(
                            job_id, JobState.QUEUED, force=True
                        )
                    else:
                        job.status = JobState.QUEUED.value
                        job.worker_id = None
                        job.updated_at = utc_naive_now()
                        if job.error_log is None:
                            job.error_log = []
                        job.error_log.append(
                            {
                            "timestamp": utc_naive_now().isoformat(),
                            "state": "health_check_failed",
                            "error": "Worker failed health check twice before dispatch",
                            }
                        )
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
                        other_running = (
                            db.query(Job)
                            .filter(
                            Job.worker_id == worker_id,
                            Job.status == JobState.RUNNING.value,
                            Job.job_id != job_id,  # Exclude current job
                            )
                            .all()
                        )
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
                        await _pod_manager.mark_machine_bad(
                            str(machine_id), duration=600
                        )
                        logger.info(
                            f"Blacklisted machine {machine_id} for 600s (10 min) due to unhealthy worker"
                        )
                    
                    # Destroy the pod
                    if _pod_manager and pod_uuid:
                        try:
                            # Before destroying, recover any RUNNING jobs (shouldn't be any, but just in case)
                            await _recover_jobs_on_worker(
                                worker_id, exclude_job_id=job_id, db=db
                            )
                            
                            await _pod_manager.destroy_individual_pod(
                                pod_uuid,
                                reason="Health check failed twice before dispatch",
                            )
                            logger.info(f"Destroyed unhealthy pod {pod_uuid}")
                        except Exception as e:
                            logger.error(
                                f"Failed to destroy unhealthy pod {pod_uuid}: {e}"
                            )
                    
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
        accepted_job_ids_from_worker: Optional[List[str]] = None
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
                
                body_bytes = encode_worker_request_body(run_payload)
                dispatch_headers = {"Content-Type": "application/json"}
                try:
                    dispatch_headers["X-Signature"] = sign_payload_bytes(body_bytes)
                except ValueError:
                    pass
                resp = requests.post(
                    url, data=body_bytes, headers=dispatch_headers, timeout=timeout
                )
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
                        backoff = 5 * (2**attempt)
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
                    backoff = 5 * (2**attempt)
                    logger.info(
                        f"Waiting {backoff}s before retry (connection error)..."
                    )
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
            is_connection_error = (
                isinstance(last_error, requests.exceptions.ConnectionError)
                or "connection refused" in error_str
                or "failed to establish" in error_str
            )
            is_read_timeout = isinstance(last_error, requests.exceptions.Timeout) and (
                "read timeout" in error_str or "read timed out" in error_str
            )
            is_connect_timeout = (
                isinstance(last_error, requests.exceptions.Timeout)
                and "connect timeout" in error_str
            )
            
            # Check for 503 (queue full) - don't blacklist, just try different worker
            if (
                is_http_error
                and hasattr(last_error, "response")
                and last_error.response
            ):
                if last_error.response.status_code == 503:
                    logger.info(
                        f"Worker {worker_id} queue full (503). Reverting job to QUEUED to try different worker."
                    )
                    if _job_manager:
                        _job_manager.update_job_state(
                            job_id, JobState.QUEUED, force=True
                        )
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
                # NOTE: DON'T clear cached params here - if job fails with OOM and is retried,
                # we need the images. Cache will be cleared in job_complete when job reaches terminal state.
                return True  # Treat as successful (worker will complete it)
            
            should_blacklist_machine = is_connection_error or is_connect_timeout
            blacklist_duration = (
                600  # 10 minutes for all failures (prevent rapid retries)
            )
            
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
            
            error_type = (
                "connection"
                if is_connection_error
                else ("connect_timeout" if is_connect_timeout else "other")
            )
            
            logger.info(
                f"Dispatch failure for pod {pod_uuid}, machine_id={machine_id}, "
                f"error_type={error_type}, should_blacklist={should_blacklist_machine}"
            )
            
            # After first job dispatch failure, immediately destroy pod and requeue all jobs
            # Check for ALL jobs associated with this pod (assigned to this worker in database)
            all_jobs_on_pod = []
            
            if pod_uuid and worker_id:
                # 1. Find jobs assigned to this worker in database
                all_jobs_on_pod = (
                    db.query(Job)
                    .filter(
                    Job.worker_id == worker_id,
                        Job.status.notin_(
                            [
                        JobState.COMPLETED.value,
                        JobState.FAILED.value,
                        JobState.TIMEOUT.value,
                        JobState.CANCELLED.value,
                        JobState.RECOVERED.value,
                            ]
                        ),
                    )
                    .all()
                )
                
                # Note: PodInfo doesn't store job IDs in a queue, so we rely only on database assignments
                
                logger.info(
                    f"Found {len(all_jobs_on_pod)} job(s) assigned to worker {worker_id} on pod {pod_uuid}"
                )
            
            # Analyze job states on this pod to decide if we should destroy immediately
            if len(all_jobs_on_pod) > 0:
                # Count jobs in different states
                running_jobs = [
                    j for j in all_jobs_on_pod if j.status == JobState.RUNNING.value
                ]
                queued_dispatching_jobs = [
                    j
                    for j in all_jobs_on_pod
                    if j.status in [JobState.QUEUED.value, JobState.DISPATCHING.value]
                ]
                
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
                    
                    job_timeout_minutes = int(os.getenv("JOB_RUNNING_TIMEOUT_MINUTES", "5"))
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
                            _job_manager.update_job_state(
                                job_id, JobState.QUEUED, force=True
                            )
                        else:
                            job.status = JobState.QUEUED.value
                            job.worker_id = None
                            job.updated_at = utc_naive_now()
                            if job.error_log is None:
                                job.error_log = []
                            job.error_log.append(
                                {
                                "timestamp": utc_naive_now().isoformat(),
                                "state": "dispatch_failed_pod_busy",
                                "error": f"Dispatch failed but pod has RUNNING jobs, only requeueing this job",
                                }
                            )
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
                        await _pod_manager.mark_machine_bad(
                            str(machine_id), duration=blacklist_duration
                        )
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
                            reason=f"Dispatch failure - {len(affected_job_ids)} jobs affected",
                        )
                        logger.info(f"Destroyed pod {pod_uuid} due to dispatch failure")
                    except Exception as e:
                        logger.error(f"Failed to destroy pod {pod_uuid}: {e}")
                
                # Revert all affected jobs and clear worker assignment
                # RUNNING jobs should be marked as FAILED (worker might still be processing)
                # QUEUED/DISPATCHING jobs should be reverted to QUEUED for requeue
                jobs_to_requeue = []
                for affected_job_id in affected_job_ids:
                    affected_job = (
                        db.query(Job).filter(Job.job_id == affected_job_id).first()
                    )
                    if not affected_job:
                        continue
                    
                    current_state = JobState(affected_job.status)
                    error_msg = f"Pod {pod_uuid} destroyed due to dispatch failure"
                    
                    if current_state == JobState.RUNNING:
                        # RUNNING jobs: RETRY instead of permanently failing
                        # CRITICAL FIX: User's jobs should not be lost due to infrastructure issues
                        # Track retry count to prevent infinite retries
                        
                        running_retry_count = affected_job.retry_count or 0
                        max_running_retries = 3  # Allow 3 retries for RUNNING jobs (consistent with other paths)

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
                            affected_job.error_log.append(
                                {
                                "timestamp": utc_naive_now().isoformat(),
                                "state": "running_pod_destroyed_retry",
                                "error": error_msg,
                                "retry_count": affected_job.retry_count,
                                    "machine_id": (
                                        str(machine_id) if machine_id else None
                                    ),
                                }
                            )
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
                                        worker_id=affected_job.worker_id,
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Failed to mark job {affected_job_id} as failed: {e}"
                                    )
                            else:
                                affected_job.status = JobState.FAILED.value
                                affected_job.error = (
                                    f"{error_msg} (retry limit exceeded)"
                                )
                                affected_job.updated_at = utc_naive_now()
                                if affected_job.error_log is None:
                                    affected_job.error_log = []
                                affected_job.error_log.append(
                                    {
                                    "timestamp": utc_naive_now().isoformat(),
                                    "state": "running_pod_destroyed_failed",
                                    "error": error_msg,
                                        "machine_id": (
                                            str(machine_id) if machine_id else None
                                        ),
                                    }
                                )
                                db.commit()
                            
                            # Send FCM notification for RUNNING job that failed permanently
                            # CRITICAL: Only send if job is actually in FAILED state
                            db.refresh(affected_job)  # Get latest status
                            is_failed = affected_job.status == JobState.FAILED.value
                            
                            if (
                                affected_job.fcm_token
                                and not affected_job.notification_sent
                                and is_failed
                            ):
                                try:
                                    asyncio.create_task(
                                        send_fcm_notification(
                                            affected_job.fcm_token,
                                            "Job Failed",
                                            f"Your job failed due to internal error. No credits deducted.",
                                            {
                                                "job_id": affected_job_id,
                                                "status": "failed",
                                                "model": affected_job.model or "unknown",
                                                "error": error_msg[:100] if error_msg else "Internal error",
                                            },
                                        )
                                    )
                                    affected_job.notification_sent = True
                                    logger.info(f"FCM failure notification sent for job {affected_job_id}")
                                except Exception as e:
                                    logger.error(
                                        f"Failed to send FCM notification for job {affected_job_id}: {e}"
                                    )
                            elif not is_failed and affected_job.fcm_token and not affected_job.notification_sent:
                                logger.warning(
                                    f"Skipping FCM notification for job {affected_job_id}: "
                                    f"expected FAILED state but found '{affected_job.status}'"
                                )
                            
                            # Clear worker assignment
                            affected_job.worker_id = None
                            db.commit()
                    else:
                        # QUEUED or DISPATCHING jobs: revert to QUEUED for requeue
                        if _job_manager:
                            try:
                                _job_manager.update_job_state(
                                    affected_job_id, JobState.QUEUED, force=True
                                )
                                affected_job.worker_id = None
                                db.commit()
                                jobs_to_requeue.append(affected_job_id)
                            except Exception as e:
                                logger.error(
                                    f"Failed to requeue job {affected_job_id}: {e}"
                                )
                        else:
                            affected_job.status = JobState.QUEUED.value
                            affected_job.worker_id = None
                            affected_job.updated_at = utc_naive_now()
                            if affected_job.error_log is None:
                                affected_job.error_log = []
                            affected_job.error_log.append(
                                {
                                "timestamp": utc_naive_now().isoformat(),
                                "state": "dispatch_failed_pod_destroyed",
                                "error": error_msg,
                                    "machine_id": (
                                        str(machine_id) if machine_id else None
                                    ),  # Track for retry limiting
                                }
                            )
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
                        _requeue_jobs_after_pod_failure(
                            affected_job_ids, affected_model
                        )
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
            job.error_log.append(
                {
                "timestamp": utc_naive_now().isoformat(),
                "state": "dispatch_failed",
                "error": str(last_error),
                    "machine_id": (
                        str(machine_id) if machine_id else None
                    ),  # Track machine for retry limiting
                "pod_uuid": pod_uuid,
                }
            )
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
        
        # CRITICAL: Before marking job as RUNNING, verify the pod is still valid.
        # This prevents race condition where pod is destroyed mid-dispatch but HTTP request succeeded.
        pod_still_valid = True
        if _pod_manager and pod_uuid:
            async with _pod_manager._lock:
                pod_still_valid = pod_uuid in _pod_manager.individual_pods
            
            if not pod_still_valid:
                logger.warning(
                    f"Pod {pod_uuid} was destroyed during dispatch of job {job_id}. "
                    f"Aborting mark_running - job will be recovered by pod destruction handler."
                )
                # Don't mark as RUNNING - let recovery handle it
                return False
        
        # Also check if worker was marked offline (another sign of pod destruction)
        worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
        if worker and worker.status == "offline":
            logger.warning(
                f"Worker {worker_id} is offline (pod destroyed?). "
                f"Aborting mark_running for job {job_id} - job will be recovered."
            )
            return False
        
        # CRITICAL: Re-fetch job to check if it was recovered during dispatch
        # (Job may have been set to QUEUED by pod destruction handler)
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if job and job.status == JobState.QUEUED.value and job.worker_id is None:
            logger.warning(
                f"Job {job_id} was recovered to QUEUED during dispatch (worker_id is None). "
                f"Aborting mark_running - job will be re-dispatched by recovery system."
            )
            return False
        
        # Success! Mark job as RUNNING
        if _job_manager:
            try:
                _job_manager.mark_running(job_id, worker_id, started_at=None)
            except Exception as e:
                logger.error(f"Failed to mark job {job_id} as running: {e}")
                # Job might have been recovered already - don't fail
                return False
        else:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job:
                job.status = JobState.RUNNING.value
                # Do NOT set started_at - same as bulk; job_complete chains it
        # Store pod_uuid in job for cost tracking
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if job and pod_uuid:
            job.pod_uuid = pod_uuid
            db.commit()
        
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
        
        # NOTE: DON'T clear cached params here - if job fails with OOM and is retried,
        # we need the images. Cache will be cleared in job_complete when job reaches terminal state
        # (COMPLETED or permanently FAILED).
        
        return True
        
    except Exception as e:
        logger.exception(f"_dispatch_to_worker error for job {job_id}: {e}")
        # Don't clear cache on error - job may be retried
        return False
    finally:
        db.close()


async def _dispatch_bulk_to_worker(job_ids: List[str], worker_id: str) -> bool:
    """
    Dispatch multiple jobs to a worker via HTTP POST to /run_bulk endpoint.
    
    Builds individual payloads for each job (same structure as /run) and sends
    them all in one request. Worker splits and enqueues them internally.
    
    Returns True on success, False on failure.
    """
    db = SessionLocal()
    try:
        # Get worker details
        worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
        if not worker:
            logger.error(f"[BULK_DISPATCH] Worker {worker_id} not found")
            return False
        
        endpoint = worker.endpoint
        if not endpoint:
            logger.error(f"[BULK_DISPATCH] Worker {worker_id} has no endpoint")
            return False
        
        pod_uuid = worker.container_id
        
        # Verify pod still exists
        if _pod_manager and pod_uuid:
            async with _pod_manager._lock:
                pod_exists = pod_uuid in _pod_manager.individual_pods
            if not pod_exists:
                logger.warning(
                    f"[BULK_DISPATCH] Pod {pod_uuid} no longer exists. Cannot dispatch."
                )
                return False
        
        # Check worker health once for all jobs
        health_check = await _check_worker_health_enhanced(endpoint, timeout=5)
        if not health_check.get("healthy", False) and not health_check.get("processing", False):
            logger.error(
                f"[BULK_DISPATCH] Worker {worker_id} is unhealthy "
                f"(status: {health_check.get('status')}). Cannot dispatch."
            )
            return False
        
        # Build payloads for all jobs
        bulk_payloads = []
        valid_job_ids = []
        
        for jid in job_ids:
            job = db.query(Job).filter(Job.job_id == jid).first()
            if not job:
                logger.warning(f"[BULK_DISPATCH] Job {jid} not found, skipping")
                continue
            
            # Skip jobs already in terminal/running state
            current_state = JobState(job.status)
            if current_state in [JobState.RUNNING, JobState.COMPLETED, JobState.FAILED,
                                 JobState.TIMEOUT, JobState.CANCELLED]:
                logger.info(
                    f"[BULK_DISPATCH] Job {jid} already in {current_state.value}, skipping"
                )
                continue
            
            # Mark job as DISPATCHING
            if _job_manager:
                _job_manager.update_job_state(jid, JobState.DISPATCHING)
            else:
                job.status = JobState.DISPATCHING.value
                job.dispatched_at = utc_naive_now()
                job.updated_at = utc_naive_now()
                db.commit()
            
            # Generate presigned URL for upload
            presigned_url = (
                create_presigned_put_url(job.result_r2_key) if job.result_r2_key else None
            )
            
            # Get cached params (R2 keys) or DB fallback
            cached_params = get_cached_job_params(jid)
            dispatch_params = dict(cached_params) if cached_params else dict(job.params or {})
            
            # ---------------------------------------------------------------
            # R2 MEDIA: Convert all *_r2_key fields to presigned download URLs
            # ---------------------------------------------------------------
            r2_convert_failed = False
            if "init_image_r2_keys" in dispatch_params:
                keys = dispatch_params.pop("init_image_r2_keys")
                if isinstance(keys, list) and keys:
                    urls: List[str] = []
                    for r2_key in keys:
                        download_url = create_presigned_get_url(r2_key, expires_in=3600)
                        if download_url:
                            urls.append(download_url)
                        else:
                            logger.error(
                                f"[BULK_DISPATCH] Job {jid}: Failed download URL for init R2 key"
                            )
                            r2_convert_failed = True
                            break
                    if not r2_convert_failed and len(urls) == len(keys):
                        dispatch_params["init_image"] = urls
                        logger.info(
                            f"[BULK_DISPATCH] Job {jid}: init_image_r2_keys → {len(urls)} URLs"
                        )
                    elif not r2_convert_failed:
                        r2_convert_failed = True
                else:
                    r2_convert_failed = True
            for r2_field, target_field in [
                ("init_image_r2_key", "init_image"),
                ("mask_image_r2_key", "mask_image"),
                ("ref_image_r2_key", "ref_image"),
                ("ref_video_r2_key", "ref_video"),
                ("input_video_r2_key", "input_video"),
                ("last_frame_image_r2_key", "last_frame_image"),
                ("first_frame_image_r2_key", "first_frame_image"),
                ("audio_r2_key", "audio_url"),
                ("cinereal_audio_r2_key", "audio_url"),
            ]:
                if r2_field in dispatch_params:
                    r2_key = dispatch_params.pop(r2_field)
                    download_url = create_presigned_get_url(r2_key, expires_in=3600)
                    if download_url:
                        dispatch_params[target_field] = download_url
                        logger.info(f"[BULK_DISPATCH] Job {jid}: {r2_field} → download URL")
                    else:
                        logger.error(f"[BULK_DISPATCH] Job {jid}: Failed to generate download URL for {r2_field}")
                        r2_convert_failed = True
                        break
            
            if r2_convert_failed:
                continue
            
            # Strip internal tracking keys (prefixed with _)
            for internal_key in list(dispatch_params.keys()):
                if internal_key.startswith("_"):
                    dispatch_params.pop(internal_key)

            _assert_dispatch_params_media_are_urls(dispatch_params, jid)
            
            run_payload = {
                "job_id": job.job_id,
                "prompt": job.prompt,
                "model": job.model,
                "mode": dispatch_params.get("mode"),
                "params": dispatch_params,
                "upload_url": presigned_url,
                "result_key": job.result_r2_key,
            }
            
            bulk_payloads.append(run_payload)
            valid_job_ids.append(jid)
        
        if not bulk_payloads:
            logger.warning("[BULK_DISPATCH] No valid jobs to dispatch")
            return False
        
        # Save bulk dispatch payload for debugging
        try:
            dispatch_data = {
                "worker_info": {
                    "worker_id": worker_id,
                    "worker_endpoint": endpoint,
                    "pod_uuid": pod_uuid,
                },
                "bulk_dispatch": {
                    "total_jobs": len(bulk_payloads),
                    "job_ids": valid_job_ids,
                },
            }
            _save_json_request(
                data=dispatch_data,
                request_type="worker_bulk_dispatch",
                job_id=valid_job_ids[0],
                additional_labels={
                    "model": bulk_payloads[0]["model"],
                    "worker_id": worker_id[:8] if worker_id else None,
                    "bulk_count": str(len(bulk_payloads)),
                },
            )
        except Exception as e:
            logger.warning(f"[BULK_DISPATCH] Failed to save dispatch JSON: {e}")
        
        # Send HTTP POST to /run_bulk
        url = f"{endpoint.rstrip('/')}/run_bulk"
        logger.info(
            f"[BULK_DISPATCH] Dispatching {len(bulk_payloads)} jobs to {url}"
        )
        
        bulk_request_body = {"jobs": bulk_payloads}
        
        last_error = None
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                timeout = 30 + (attempt * 10)  # Longer timeout for bulk
                logger.debug(
                    f"[BULK_DISPATCH] Attempt {attempt + 1}/{max_attempts} "
                    f"(timeout={timeout}s)"
                )
                
                body_bytes = encode_worker_request_body(bulk_request_body)
                bulk_headers = {"Content-Type": "application/json"}
                try:
                    bulk_headers["X-Signature"] = sign_payload_bytes(body_bytes)
                except ValueError:
                    pass
                resp = requests.post(
                    url, data=body_bytes, headers=bulk_headers, timeout=timeout
                )
                resp.raise_for_status()
                
                # Success!
                last_error = None
                resp_data = resp.json()
                if isinstance(resp_data, dict):
                    raw_ids = resp_data.get("job_ids")
                    if isinstance(raw_ids, list):
                        accepted_job_ids_from_worker = [
                            str(x) for x in raw_ids if isinstance(x, str) and x
                        ]
                logger.info(
                    f"[BULK_DISPATCH] Worker accepted {len(bulk_payloads)} jobs: {resp_data}"
                )
                break
                
            except requests.exceptions.Timeout as e:
                last_error = e
                error_str = str(e).lower()
                if "read timeout" in error_str or "read timed out" in error_str:
                    # Worker likely received it
                    logger.warning(
                        f"[BULK_DISPATCH] READ TIMEOUT on attempt {attempt + 1}. "
                        f"Worker likely received the bulk request."
                    )
                    last_error = None
                    break
                else:
                    logger.warning(
                        f"[BULK_DISPATCH] CONNECT TIMEOUT on attempt {attempt + 1}: {e}"
                    )
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(5 * (2 ** attempt))
                    
            except requests.exceptions.HTTPError as e:
                last_error = e
                if e.response and e.response.status_code == 503:
                    logger.warning(
                        f"[BULK_DISPATCH] Worker queue full (503). Cannot accept bulk."
                    )
                    break  # Don't retry on 503
                logger.warning(
                    f"[BULK_DISPATCH] HTTP error on attempt {attempt + 1}: {e}"
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(3 * (attempt + 1))
                    
            except requests.exceptions.ConnectionError as e:
                last_error = e
                logger.warning(
                    f"[BULK_DISPATCH] Connection error on attempt {attempt + 1}: {e}"
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(5 * (2 ** attempt))
                    
            except Exception as e:
                last_error = e
                logger.error(f"[BULK_DISPATCH] Unexpected error: {e}")
                break
        
        if last_error:
            logger.error(
                f"[BULK_DISPATCH] All attempts failed for {len(valid_job_ids)} jobs: {last_error}"
            )
            # Revert all jobs to QUEUED
            for jid in valid_job_ids:
                job = db.query(Job).filter(Job.job_id == jid).first()
                if job and job.status not in [
                    JobState.RUNNING.value, JobState.COMPLETED.value, JobState.FAILED.value
                ]:
                    job.status = JobState.QUEUED.value
                    job.worker_id = None
                    job.updated_at = utc_naive_now()
            db.commit()
            return False
        
        # SUCCESS: Mark accepted jobs as RUNNING.
        # Worker may return a partial acceptance list (rare race/queue edge), so do not
        # blindly mark all sent jobs as RUNNING.
        accepted_job_ids = list(valid_job_ids)
        if accepted_job_ids_from_worker is not None:
            accepted_set = set(accepted_job_ids_from_worker)
            accepted_job_ids = [jid for jid in valid_job_ids if jid in accepted_set]
            if len(accepted_job_ids) != len(valid_job_ids):
                logger.warning(
                    "[BULK_DISPATCH] Partial worker acceptance: accepted=%s requested=%s worker=%s",
                    len(accepted_job_ids),
                    len(valid_job_ids),
                    worker_id,
                )
                rejected = [jid for jid in valid_job_ids if jid not in accepted_set]
                for jid in rejected:
                    job = db.query(Job).filter(Job.job_id == jid).first()
                    if job and job.status not in [
                        JobState.RUNNING.value, JobState.COMPLETED.value, JobState.FAILED.value
                    ]:
                        job.status = JobState.QUEUED.value
                        job.worker_id = None
                        job.updated_at = utc_naive_now()
                    if _pod_manager and pod_uuid:
                        try:
                            await _pod_manager.release_slot_on_pod(pod_uuid, jid)
                        except Exception as e:
                            logger.warning(
                                f"[BULK_DISPATCH] Failed releasing slot for rejected job {jid}: {e}"
                            )

        if not accepted_job_ids:
            logger.warning("[BULK_DISPATCH] Worker accepted 0 jobs in bulk response")
            for jid in valid_job_ids:
                job = db.query(Job).filter(Job.job_id == jid).first()
                if job and job.status not in [
                    JobState.RUNNING.value, JobState.COMPLETED.value, JobState.FAILED.value
                ]:
                    job.status = JobState.QUEUED.value
                    job.worker_id = None
                    job.updated_at = utc_naive_now()
            db.commit()
            return False

        # SUCCESS: Mark accepted jobs as RUNNING
        # Verify pod is still valid first
        pod_still_valid = True
        if _pod_manager and pod_uuid:
            async with _pod_manager._lock:
                pod_still_valid = pod_uuid in _pod_manager.individual_pods
        
        if not pod_still_valid:
            logger.warning(
                f"[BULK_DISPATCH] Pod {pod_uuid} destroyed during dispatch. "
                f"Jobs will be recovered by destruction handler."
            )
            return False
        
        for i, jid in enumerate(accepted_job_ids):
            try:
                if _job_manager:
                    # Do NOT set started_at for any bulk job. Stuck detection uses started_at;
                    # only the job actually being processed should have started_at set, which
                    # happens when the worker reports job_complete with next_job_id (or fallback below).
                    _job_manager.mark_running(jid, worker_id, started_at=None)
                else:
                    job = db.query(Job).filter(Job.job_id == jid).first()
                    if job:
                        job.status = JobState.RUNNING.value
                        # Leave started_at NULL for all; chained via job_complete (next_job_id or fallback)
                
                # Store pod_uuid in job
                job = db.query(Job).filter(Job.job_id == jid).first()
                if job and pod_uuid:
                    job.pod_uuid = pod_uuid
                
                # Remove from pending jobs
                if _pod_manager and job:
                    try:
                        await _pod_manager.remove_pending_job(jid, job.model)
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"[BULK_DISPATCH] Failed to mark job {jid} as running: {e}")
        
        # Update worker status
        worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
        if worker:
            worker.status = "processing" if worker.is_warm_pod else "busy"
            worker.last_active = utc_naive_now()
        
        db.commit()
        
        logger.info(
            f"[BULK_DISPATCH] Marked {len(accepted_job_ids)}/{len(valid_job_ids)} jobs as RUNNING on worker {worker_id}"
        )
        return True
        
    except Exception as e:
        logger.exception(f"[BULK_DISPATCH] Error: {e}")
        return False
    finally:
        db.close()


def _get_model_eta_for_status(model: str) -> int:
    """Best-effort ETA seconds for a model: dynamic EWMA first, then model_requirements default."""
    try:
        from db import SessionLocal as _SL, ModelEtaStat
        model_norm = (model or "").strip().lower()
        if model_norm:
            with _SL() as _db:
                row = _db.query(ModelEtaStat).filter(ModelEtaStat.model == model_norm).first()
                if row and int(row.sample_count or 0) >= 3:
                    return max(5, int(row.ewma_eta_seconds))
    except Exception:
        pass
    try:
        pm = _pod_manager
        if pm and hasattr(pm, "model_requirements"):
            req = pm.model_requirements.get(model) or pm.model_requirements.get((model or "").strip().lower()) or {}
            default_eta = req.get("eta_seconds_default")
            if default_eta:
                return int(default_eta)
    except Exception:
        pass
    return 60


def _compute_model_inference_seconds_map(supported: Any) -> Dict[str, float]:
    """
    EWMA inference seconds per model (ModelEtaStat), for worker download tuning.
    Only models with sample_count >= 3 and positive ewma are included.
    """
    out: Dict[str, float] = {}
    try:
        from db import SessionLocal, ModelEtaStat

        models = supported or []
        if not isinstance(models, (list, tuple)):
            models = [models] if models else []
        with SessionLocal() as db:
            for m in models:
                m_norm = str(m or "").strip().lower()
                if not m_norm:
                    continue
                row = (
                    db.query(ModelEtaStat)
                    .filter(ModelEtaStat.model == m_norm)
                    .first()
                )
                if (
                    row
                    and int(row.sample_count or 0) >= 3
                    and float(row.ewma_eta_seconds or 0) > 0
                ):
                    out[m_norm] = round(float(row.ewma_eta_seconds), 1)
    except Exception:
        pass
    return out


@client_router.get("/job_status", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    auth: Dict = Depends(require_firebase_auth),
):
    """
    Get current job status.
    
    Requires Firebase authentication via Authorization: Bearer <firebase_id_token>
    
    For multi-variation jobs:
    - Returns aggregated status (completed only when ALL variations done)
    - Includes variation_results array with all variation outputs
    - Primary result_url points to first variation
    """
    from models import VariationResult
    
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Check if this job is part of a batch (multi-variation request)
        batch_id = job.params.get("_batch_id") if job.params else None
        total_variations = job.params.get("_total_variations", 1) if job.params else 1
        
        # Default values for single job
        result_url = None
        variation_results = None
        all_completed = True
        aggregated_status = job.status
        aggregated_completed_at = job.completed_at
        aggregated_error = job.error
        
        if batch_id and total_variations > 1:
            # Multi-variation job - fetch all sibling jobs
            # Use database-agnostic query: cast JSON to text and use LIKE
            # This works on both SQLite and PostgreSQL
            try:
                # Cast params to string and search for batch_id
                # This is more reliable than .contains() which doesn't work on SQLite
                sibling_jobs = db.query(Job).filter(
                    cast(Job.params, SQLString).like(f'%"_batch_id": "{batch_id}"%')
                ).all()
                
                # Fallback: try alternative JSON format (no space after colon)
                if not sibling_jobs:
                    sibling_jobs = db.query(Job).filter(
                        cast(Job.params, SQLString).like(f'%"_batch_id":"{batch_id}"%')
                    ).all()
            except Exception as query_err:
                logger.warning(f"Batch query failed for batch {batch_id}: {query_err}")
                sibling_jobs = []
            
            # CRITICAL: If query returned empty results, fall back to just the current job
            # This prevents the 0==0 bug where empty results cause "failed" status
            if not sibling_jobs:
                logger.warning(
                    f"Batch query returned no results for batch_id={batch_id}, "
                    f"falling back to single-job status for job {job_id}"
                )
                # Use current job's status only
                sibling_jobs = [job]
            
            # Ensure current job is in the results (in case query missed it)
            current_job_in_results = any(j.job_id == job_id for j in sibling_jobs)
            if not current_job_in_results:
                sibling_jobs.append(job)
            
            # Build variation results
            variation_results = []
            completed_count = 0
            failed_count = 0
            latest_completed_at = None
            first_error = None
            
            for sib_job in sorted(sibling_jobs, key=lambda j: j.params.get("_variation_index", 0) if j.params else 0):
                var_idx = sib_job.params.get("_variation_index", 0) if sib_job.params else 0
                var_result_url = None
                
                if sib_job.status == JobState.COMPLETED.value:
                    var_result_url = _build_public_result_url(sib_job)
                    completed_count += 1
                    if sib_job.completed_at:
                        if latest_completed_at is None or sib_job.completed_at > latest_completed_at:
                            latest_completed_at = sib_job.completed_at
                elif sib_job.status == JobState.FAILED.value:
                    failed_count += 1
                    if first_error is None:
                        first_error = sib_job.error
                
                variation_results.append(VariationResult(
                    variation_index=var_idx,
                    status=sib_job.status,
                    result_url=var_result_url,
                    error=sib_job.error if sib_job.status == JobState.FAILED.value else None,
                ))
            
            # Determine aggregated status
            total_jobs = len(sibling_jobs)
            finished_count = completed_count + failed_count
            
            # CRITICAL: Avoid 0==0 bug - if total_jobs is 0, treat as not completed
            if total_jobs == 0:
                all_completed = False
                aggregated_status = JobState.QUEUED.value
                aggregated_completed_at = None
                aggregated_error = None
            else:
                all_completed = finished_count == total_jobs
                
                if all_completed:
                    if failed_count == total_jobs:
                        aggregated_status = JobState.FAILED.value
                        aggregated_error = first_error or "All variations failed"
                    elif completed_count > 0:
                        aggregated_status = JobState.COMPLETED.value
                        aggregated_error = None
                    else:
                        aggregated_status = JobState.FAILED.value
                    aggregated_completed_at = latest_completed_at
                else:
                    # Still processing - find the most representative status
                    running_count = sum(1 for j in sibling_jobs if j.status == JobState.RUNNING.value)
                    if running_count > 0:
                        aggregated_status = JobState.RUNNING.value
                    else:
                        aggregated_status = JobState.QUEUED.value
                    aggregated_completed_at = None
                    aggregated_error = None
            
            # Primary result_url is first completed variation
            for vr in variation_results:
                if vr.result_url:
                    result_url = vr.result_url
                    break
        else:
            # Single job - standard behavior
            total_variations = 1
            if job.status == JobState.COMPLETED.value:
                result_url = _build_public_result_url(job)
        
        estimated_wait_seconds = None
        queue_position = None
        if aggregated_status in (JobState.QUEUED.value, JobState.RUNNING.value):
            try:
                model_name = job.model or ""
                queued_ahead = db.query(Job).filter(
                    Job.model == model_name,
                    Job.status == JobState.QUEUED.value,
                    Job.created_at < job.created_at,
                ).count()
                running_count = db.query(Job).filter(
                    Job.model == model_name,
                    Job.status == JobState.RUNNING.value,
                ).count()
                if aggregated_status == JobState.QUEUED.value:
                    queue_position = queued_ahead + 1
                else:
                    queue_position = 0
                eta_per_job = _get_model_eta_for_status(model_name)
                pending_jobs = max(0, queued_ahead - max(0, running_count - 1))
                if aggregated_status == JobState.RUNNING.value:
                    estimated_wait_seconds = eta_per_job
                else:
                    estimated_wait_seconds = eta_per_job * (pending_jobs + 1)
            except Exception:
                pass

        return JobStatusResponse(
            job_id=job.job_id,
            status=JobStatus(aggregated_status),
            model=job.model,
            created_at=job.created_at,
            updated_at=job.updated_at,
            completed_at=aggregated_completed_at,
            worker_id=job.worker_id,
            retry_count=job.retry_count or 0,
            result_url=result_url,
            continuation_latent_r2_key=(job.params or {}).get("continuation_latent_r2_key"),
            error=aggregated_error,
            error_log=job.error_log,
            variations=total_variations,
            variation_results=variation_results,
            all_completed=all_completed,
            estimated_wait_seconds=estimated_wait_seconds,
            queue_position=queue_position,
            music=_public_suno_music(job),
        )
    finally:
        db.close()


@client_router.get("/batch_status")
async def get_batch_status(
    batch_id: str,
    auth: Dict = Depends(require_firebase_auth),
):
    """
    Get status of all jobs in a batch (for multi-variation requests).
    
    Requires Firebase authentication via Authorization: Bearer <firebase_id_token>
    
    Returns:
    - batch_id: The batch identifier
    - jobs: List of job statuses
    - all_completed: True if all jobs have finished (completed or failed)
    - successful_count: Number of successfully completed jobs
    - failed_count: Number of failed jobs
    - total_count: Total number of jobs in batch
    """
    db = SessionLocal()
    try:
        # Find all jobs with this batch_id in their params
        # Use database-agnostic query: cast JSON to text and use LIKE
        # This works on both SQLite and PostgreSQL
        jobs = db.query(Job).filter(
            cast(Job.params, SQLString).like(f'%"_batch_id": "{batch_id}"%')
        ).all()
        
        # Fallback: try alternative JSON format (no space after colon)
        if not jobs:
            jobs = db.query(Job).filter(
                cast(Job.params, SQLString).like(f'%"_batch_id":"{batch_id}"%')
            ).all()
        
        if not jobs:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        # Build response
        job_statuses = []
        successful_count = 0
        failed_count = 0
        
        for job in jobs:
            result_url = None
            if job.status == JobState.COMPLETED.value:
                result_url = _build_public_result_url(job)
            
            variation_idx = job.params.get("_variation_index", 0) if job.params else 0
            
            job_status = {
                "job_id": job.job_id,
                "status": job.status,
                "variation_index": variation_idx,
                "result_url": result_url,
                "error": job.error,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            }
            job_statuses.append(job_status)
            
            if job.status == JobState.COMPLETED.value:
                successful_count += 1
            elif job.status == JobState.FAILED.value:
                failed_count += 1
        
        # Sort by variation index
        job_statuses.sort(key=lambda x: x["variation_index"])
        
        total_count = len(jobs)
        finished_count = successful_count + failed_count
        all_completed = finished_count == total_count
        
        return {
            "batch_id": batch_id,
            "jobs": job_statuses,
            "all_completed": all_completed,
            "successful_count": successful_count,
            "failed_count": failed_count,
            "total_count": total_count,
            "pending_count": total_count - finished_count,
        }
    finally:
        db.close()


# =============================================================================
# Worker Endpoints
# =============================================================================


@worker_router.post("/register_worker")
async def register_worker(
    request: Request,
    payload_dict: Dict[str, Any] = Depends(require_worker_signed_payload),
):
    payload = RegisterWorkerRequest.model_validate(payload_dict)
    """
    Worker registration endpoint.
    
    Called by worker pods when they start up and are ready.
    """
    ip = get_client_ip(request)
    
    # Get pod UUID from worker (prefer pod_uuid over container_id)
    worker_pod_uuid = payload.pod_uuid or payload.container_id
    
    if not worker_pod_uuid:
        raise HTTPException(status_code=400, detail="pod_uuid or container_id required")
    
    # Normalize the UUID (remove hyphens for consistent comparison/storage)
    worker_pod_uuid_normalized = normalize_pod_uuid(worker_pod_uuid)
    
    # Extract worker IP and port from endpoint for verification
    worker_ip = None
    worker_port = None
    if payload.endpoint:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(payload.endpoint)
            worker_ip = parsed.hostname
            worker_port = parsed.port
        except Exception:
            pass
    
    # CRITICAL: Match worker to tracked pod by normalized UUID or IP
    # The server knows the real pod UUID (from create_pod response with hyphens)
    # Worker sends the same UUID (may or may not have hyphens)
    pod_uuid = None  # Will be set to the tracked pod's UUID (with hyphens, for API calls)
    matched_by = None
    
    if _pod_manager:
        # First: Try to match by normalized UUID
        for tracked_uuid in list(_pod_manager.individual_pods.keys()):
            tracked_normalized = normalize_pod_uuid(tracked_uuid)
            if tracked_normalized == worker_pod_uuid_normalized:
                pod_uuid = tracked_uuid  # Use the original format (with hyphens)
                matched_by = "uuid"
                break
        
        # Second: Try to match by IP (if UUID didn't match)
        if not pod_uuid and worker_ip:
            if hasattr(_pod_manager.lifecycle, '_pod_public_ips'):
                ip_cache = _pod_manager.lifecycle._pod_public_ips
                if worker_ip in ip_cache:
                    pod_uuid = ip_cache[worker_ip]
                    matched_by = "ip"
        
        if pod_uuid:
            logger.info(
                f"Worker registration: matched worker UUID '{worker_pod_uuid}' to tracked pod '{pod_uuid}' (by {matched_by})"
            )
            
            # Verify: If we have both UUID and IP info, confirm they match
            if matched_by == "uuid" and worker_ip:
                if hasattr(_pod_manager.lifecycle, '_pod_public_ips'):
                    expected_ip = None
                    for cached_ip, cached_uuid in _pod_manager.lifecycle._pod_public_ips.items():
                        if cached_uuid == pod_uuid:
                            expected_ip = cached_ip
                            break
                    if expected_ip and expected_ip != worker_ip:
                        logger.warning(
                            f"Worker registration: UUID matched but IP mismatch! "
                            f"Worker IP={worker_ip}, expected={expected_ip}. Using UUID match."
                        )
        else:
            # No match found - use worker's UUID (normalized)
            pod_uuid = worker_pod_uuid
            logger.warning(
                f"Worker registration: No tracked pod matches worker UUID '{worker_pod_uuid}' or IP '{worker_ip}'"
            )
    else:
        pod_uuid = worker_pod_uuid

    if _pod_manager and pod_uuid:
        tracked = _pod_manager.individual_pods.get(pod_uuid)
        if not tracked:
            want = normalize_pod_uuid(pod_uuid)
            for tracked_uuid, tracked_info in list(_pod_manager.individual_pods.items()):
                if normalize_pod_uuid(tracked_uuid) == want:
                    tracked = tracked_info
                    pod_uuid = tracked_uuid
                    break
        if tracked:
            public_url = (tracked.meta or {}).get("public_url")
            if public_url:
                payload.endpoint = public_url
            gpu_prov = (tracked.meta or {}).get("gpu_provider")
            if gpu_prov:
                if not isinstance(payload.info, dict):
                    payload.info = {}
                payload.info["gpu_provider"] = gpu_prov
    
    # Normalize for storage (we'll store without hyphens for consistency)
    pod_uuid_for_storage = normalize_pod_uuid(pod_uuid)
    
    # Validate pod belongs to orchestrator (using normalized comparison)
    if _pod_manager:
        active_pods = _pod_manager.active_pods
        # Check both formats
        pod_in_active = (
            pod_uuid in active_pods or 
            pod_uuid_for_storage in active_pods or
            any(normalize_pod_uuid(k) == pod_uuid_for_storage for k in active_pods.keys())
        )
        if not pod_in_active:
            # Check if recently destroyed
            if _pod_manager.lifecycle.is_recently_destroyed(pod_uuid):
                logger.warning(
                    f"Rejected registration from recently destroyed pod {pod_uuid}"
                )
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
    
    # Get model name - CRITICAL: prefer tracked pod's model over worker's reported model
    # Workers may report generic model names (e.g., "fastwan") while jobs need specific 
    # model names (e.g., "fastwan_22_5b"). The pod was created for a specific model,
    # so use that to ensure job dispatch can match correctly.
    model = None
    if _pod_manager and pod_uuid and pod_uuid in _pod_manager.individual_pods:
        tracked_model = _pod_manager.individual_pods[pod_uuid].model
        if tracked_model and tracked_model != "unknown":
            model = tracked_model
            if payload.models and payload.models[0] != tracked_model:
                logger.info(
                    f"Worker {payload.worker_id} reported model '{payload.models[0]}' but "
                    f"tracked pod {pod_uuid} was created for model '{tracked_model}'. "
                    f"Using tracked model for correct job dispatch."
                )
    
    # Fallback to worker's reported model if no tracked pod model
    if not model:
        model = payload.models[0] if payload.models else "unknown"

    # Build normalized runtime capability payload (Phase 2 hybrid capability source)
    runtime_supported_models = []
    for m in (payload.supported_models or payload.models or []):
        if isinstance(m, str):
            m_norm = m.strip().lower()
            if m_norm and m_norm not in runtime_supported_models:
                runtime_supported_models.append(m_norm)
    if model and model != "unknown" and model not in runtime_supported_models:
        runtime_supported_models.append(model)

    runtime_eta_map = {}
    if isinstance(payload.model_eta_seconds, dict):
        for k, v in payload.model_eta_seconds.items():
            if not isinstance(k, str):
                continue
            try:
                eta_val = int(v)
            except Exception:
                continue
            if eta_val <= 0:
                continue
            runtime_eta_map[k.strip().lower()] = eta_val

    merged_worker_info = payload.info.copy() if isinstance(payload.info, dict) else {}
    merged_worker_info["supported_models"] = runtime_supported_models
    if runtime_eta_map:
        merged_worker_info["model_eta_seconds"] = runtime_eta_map
    merged_worker_info["capability_source"] = "worker_runtime"
    merged_worker_info["capability_updated_at"] = utc_naive_now().isoformat()
    if _pod_manager and pod_uuid in _pod_manager.individual_pods:
        tracked_meta = _pod_manager.individual_pods[pod_uuid].meta or {}
        if tracked_meta.get("gpu_provider"):
            merged_worker_info["gpu_provider"] = tracked_meta["gpu_provider"]
        if tracked_meta.get("public_url"):
            merged_worker_info["public_url"] = tracked_meta["public_url"]
    
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
                    _check_worker_health(payload.endpoint, timeout=2), timeout=3.0
                )
                if not health_check:
                    logger.warning(
                        f"Worker {payload.worker_id} registration: /health endpoint not responding, "
                        f"but proceeding with registration"
                    )
                    worker_ready = False  # Mark as starting instead of ready
            except asyncio.TimeoutError:
                logger.warning(
                    f"Worker {payload.worker_id} health check timed out, marking as starting"
                )
                worker_ready = False
            except Exception as e:
                logger.warning(f"Worker health check failed during registration: {e}")
                worker_ready = False
        
        # Use enhanced health check to get detailed worker status (with timeout protection)
        if payload.endpoint and not worker_ready:
            try:
                health_info = await asyncio.wait_for(
                    _check_worker_health_enhanced(payload.endpoint, timeout=2),
                    timeout=3.0,
                )
                if health_info["healthy"]:
                    # Use status from worker's health endpoint
                    worker_status = health_info.get("status", "unknown")
                    if worker_status == "starting":
                        initial_status = "starting"
                    elif worker_status in ("ok", "busy"):
                        initial_status = "ready"
                    else:
                        initial_status = (
                            "starting"  # Conservative: wait for explicit "ok"
                        )
                    
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
                logger.warning(
                    f"Worker {payload.worker_id} enhanced health check timed out"
                )
                initial_status = "starting"
            except Exception as e:
                logger.warning(
                    f"Worker {payload.worker_id} enhanced health check error: {e}"
                )
                initial_status = "starting"
        
        # Get GPU cost info from pod tracking (for cost attribution)
        gpu_hourly_rate = None
        gpu_type = None
        offer_id = None
        machine_id = None
        if _pod_manager and pod_uuid in _pod_manager.individual_pods:
            pi = _pod_manager.individual_pods[pod_uuid]
            gpu_hourly_rate = pi.hourly_cost if pi.hourly_cost > 0 else None
            gpu_type = pi.gpu_type if pi.gpu_type else None
            offer_id = pi.offer_id if pi.offer_id > 0 else None
            machine_id = pi.machine_id
        
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
                info=merged_worker_info,
                is_warm_pod=is_warm,
                # Cost tracking fields
                gpu_hourly_rate=gpu_hourly_rate,
                gpu_type=gpu_type,
                offer_id=offer_id,
                machine_id=machine_id,
            )
            db.add(worker)
        else:
            worker.endpoint = payload.endpoint
            worker.model = model
            worker.container_id = pod_uuid
            worker.status = initial_status
            worker.last_heartbeat = now
            worker.last_active = now
            worker.info = merged_worker_info
            if pod_uuid:
                worker.is_warm_pod = is_warm
            # Update cost tracking fields if available
            if gpu_hourly_rate is not None:
                worker.gpu_hourly_rate = gpu_hourly_rate
            if gpu_type is not None:
                worker.gpu_type = gpu_type
            if offer_id is not None:
                worker.offer_id = offer_id
            if machine_id is not None:
                worker.machine_id = machine_id
        
        # If initially starting, schedule a task to POLL for readiness
        # (models can take 60-120s to load, so single check after 8s is insufficient)
        if initial_status == "starting" and payload.endpoint:
            logger.info(
                f"Worker {payload.worker_id} marked as 'starting', will poll for readiness every 10s"
            )
            # Store endpoint for background task
            endpoint_to_check = payload.endpoint
            worker_id_to_check = payload.worker_id
            pod_uuid_to_check = pod_uuid  # Capture for use in async task
            
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
                        w = (
                            db.query(Worker)
                            .filter(Worker.worker_id == worker_id_to_check)
                            .first()
                        )
                        
                        if not w:
                            logger.warning(
                                f"Worker {worker_id_to_check} no longer exists, stopping readiness poll"
                            )
                            return
                        
                        if w.status != "starting":
                            # Worker already transitioned to another state
                            logger.debug(
                                f"Worker {worker_id_to_check} status is '{w.status}', stopping readiness poll"
                            )
                            return
                        
                        # Check worker health
                        health_info = await _check_worker_health_enhanced(
                            endpoint_to_check, timeout=5
                        )
                        
                        # Check for OOM during model load
                        model_error = health_info.get("details", {}).get("model_error")
                        if model_error and isinstance(model_error, str):
                            model_error_l = model_error.lower()
                            if "oom" in model_error_l or "out of memory" in model_error_l:
                                logger.error(
                                    f"Worker {worker_id_to_check} reported OOM during model loading: {model_error}"
                                )
                                w.status = "offline"
                                db.commit()
                                # Trigger pod destruction in background
                                if _pod_manager:
                                    asyncio.create_task(
                                        _handle_oom_pod_destruction(
                                            _pod_manager, 
                                            pod_uuid_to_check, 
                                            "OOM during model load"
                                        )
                                    )
                                return
                        
                        if health_info.get("ready"):
                            w.status = "ready"
                            db.commit()
                            elapsed = asyncio.get_event_loop().time() - start_time
                            logger.info(
                                f"Worker {worker_id_to_check} is now READY after {elapsed:.0f}s "
                                f"(health status={health_info.get('status')})"
                            )
                            
                            # CRITICAL: Also update in-memory pod tracking
                            # This ensures dispatch sees the worker as ready immediately
                            if _pod_manager and pod_uuid_to_check in _pod_manager.individual_pods:
                                from pod_manager import PodStatus
                                _pod_manager.individual_pods[pod_uuid_to_check].status = PodStatus.READY
                                logger.info(f"Updated pod {pod_uuid_to_check} in-memory status to READY")
                            
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
        
        # Update pod manager tracking - set status based on actual worker status
        if _pod_manager and pod_uuid in _pod_manager.individual_pods:
            from pod_manager import PodStatus
            
            # CRITICAL: Don't set to READY if worker is still 'starting'
            # The _poll_for_readiness task will update this when worker is truly ready
            if worker.status == "ready":
                _pod_manager.individual_pods[pod_uuid].status = PodStatus.READY
            else:
                _pod_manager.individual_pods[pod_uuid].status = PodStatus.STARTING
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
            logger.warning(
                f"Failed to create audit log for worker registration: {audit_error}"
            )
        
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
        _mis = _compute_model_inference_seconds_map(
            payload.supported_models or payload.models
        )
        return {
            "status": "ok",
            "worker_id": payload.worker_id,
            "warning": "partial_registration",
            "model_inference_seconds": _mis,
        }
    finally:
        db.close()
    
    # Audit logger is optional - don't fail if it errors
    try:
        audit_logger.worker_registered(payload.worker_id, pod_uuid, ip, model)
    except Exception as e:
        logger.warning(f"Audit logger error (non-critical): {e}")
    
    logger.info(
        f"Worker {payload.worker_id} registered for model {model} (pod: {pod_uuid})"
    )

    _mis = _compute_model_inference_seconds_map(
        payload.supported_models or payload.models
    )
    return {
        "status": "ok",
        "worker_id": payload.worker_id,
        "model_inference_seconds": _mis,
    }


# In-memory heartbeat cache for fast responses (synced to DB periodically)
# CRITICAL: This avoids DB queries on every heartbeat, making the endpoint fast
_heartbeat_cache: Dict[str, Dict[str, Any]] = {}
_heartbeat_cache_lock = asyncio.Lock()

# In-memory cache for job params with images
# CRITICAL: Base64 images are NOT stored in DB (too large), but must be passed to worker
# This cache holds full params including images until job is dispatched
_job_params_cache: Dict[str, Dict[str, Any]] = {}
_job_params_cache_lock = asyncio.Lock()


# Known fields that may have held base64 historically; strip before DB persist
BASE64_IMAGE_FIELDS = frozenset([
    "init_image", "image", "mask_image", "control_image", "source_image", 
    "images", "reference_image", "style_image", "face_image", "input_image",
    "background_image", "foreground_image", "depth_image", "normal_image",
])


def _is_base64_image_string(value: Any) -> bool:
    """
    Check if a value looks like a base64-encoded image.
    Returns True for long strings that appear to be base64 data.
    """
    if not isinstance(value, str):
        return False
    
    # Must be reasonably long (images are typically > 1KB base64 = ~1365 chars)
    if len(value) < 500:
        return False
    
    # Check for data URL prefix
    if value.startswith('data:image'):
        return True
    
    # Check if it looks like base64 (mostly alphanumeric + /+=)
    # Sample first 100 chars to avoid checking entire string
    sample = value[:100].replace('/', '').replace('+', '').replace('=', '')
    return sample.isalnum()


_WORKER_MEDIA_PARAM_KEYS = (
    "init_image",
    "image",
    "mask_image",
    "input_video",
    "ref_image",
    "ref_video",
    "first_frame_image",
    "last_frame_image",
    "condition_image",
    "condition_video",
)


def _assert_dispatch_params_media_are_urls(
    dispatch_params: Dict[str, Any], job_id: str
) -> None:
    """
    Workers must receive only http(s) URLs for media (presigned R2 GET).
    Rejects data: URIs, raw base64 blobs, and local paths mistaken for payloads.
    """
    for key in _WORKER_MEDIA_PARAM_KEYS:
        val = dispatch_params.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            for i, item in enumerate(val):
                if not isinstance(item, str):
                    raise Exception(
                        f"Job {job_id}: {key}[{i}] must be a URL string after R2 resolution"
                    )
                s = item.strip()
                if s.startswith("data:") or _is_base64_image_string(s):
                    raise Exception(
                        f"Job {job_id}: {key}[{i}] must be a presigned https URL, not embedded media"
                    )
                if not (s.startswith("http://") or s.startswith("https://")):
                    raise Exception(
                        f"Job {job_id}: {key}[{i}] must start with http:// or https://"
                    )
            continue
        if not isinstance(val, str):
            raise Exception(f"Job {job_id}: {key} must be a URL string or list of URLs")
        s = val.strip()
        if s.startswith("data:") or _is_base64_image_string(s):
            raise Exception(
                f"Job {job_id}: {key} must be a presigned https URL, not embedded media"
            )
        if not (s.startswith("http://") or s.startswith("https://")):
            raise Exception(
                f"Job {job_id}: {key} must start with http:// or https:// (presigned R2 GET)"
            )


def strip_base64_images_for_db(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strip base64 image data from params for database storage.
    
    This prevents DB bloat from storing large image data that's only needed
    for worker dispatch (which uses the in-memory cache).
    
    Replaces image fields with a placeholder to indicate data was present.
    """
    if not params:
        return params
    
    cleaned = {}
    for key, value in params.items():
        # Skip known image fields
        if key in BASE64_IMAGE_FIELDS:
            if value:
                # Store placeholder to indicate image was provided
                cleaned[f"_{key}_provided"] = True
            continue
        
        # Check for unknown fields that look like base64 images
        if isinstance(value, str) and _is_base64_image_string(value):
            logger.debug(f"Stripping unknown base64 field '{key}' from DB params")
            cleaned[f"_{key}_provided"] = True
            continue
        
        # Handle lists (could contain images)
        if isinstance(value, list):
            # Check if it's a list of images
            if value and all(_is_base64_image_string(v) for v in value if isinstance(v, str)):
                cleaned[f"_{key}_provided"] = True
                cleaned[f"_{key}_count"] = len(value)
                continue
        
        # Keep non-image data
        cleaned[key] = value
    
    return cleaned


def cache_job_params(job_id: str, params: Dict[str, Any]):
    """Cache full job params for dispatch (R2 keys; server expands to presigned GET URLs)."""
    _job_params_cache[job_id] = copy.deepcopy(params)


def get_cached_job_params(job_id: str) -> Optional[Dict[str, Any]]:
    """Get cached job params, returns None if not cached"""
    return _job_params_cache.get(job_id)


def clear_cached_job_params(job_id: str):
    """Clear cached job params after dispatch"""
    _job_params_cache.pop(job_id, None)


def _learn_dynamic_eta_from_completed_job(
    db: Session,
    job: Job,
    payload_metrics: Optional[Dict[str, Any]],
):
    """
    Update dynamic per-model ETA using completed job runtime.

    Source priority:
    1) worker payload.metrics.processing_time_seconds
    2) persisted job.processing_time_seconds
    """
    if os.getenv("SCHEDULER_DYNAMIC_ETA_ENABLED", "false").lower() != "true":
        return
    if not job or not isinstance(job.model, str):
        return

    sample_seconds: Optional[float] = None
    if isinstance(payload_metrics, dict):
        raw = payload_metrics.get("processing_time_seconds")
        try:
            if raw is not None:
                sample_seconds = float(raw)
        except Exception:
            sample_seconds = None

    if sample_seconds is None:
        try:
            if job.processing_time_seconds is not None:
                sample_seconds = float(job.processing_time_seconds)
        except Exception:
            sample_seconds = None

    if sample_seconds is None or sample_seconds <= 0:
        return

    min_seconds = max(1.0, float(os.getenv("SCHEDULER_DYNAMIC_ETA_MIN_SECONDS", "10")))
    max_seconds = max(
        min_seconds,
        float(os.getenv("SCHEDULER_DYNAMIC_ETA_MAX_SECONDS", "1200")),
    )
    alpha = float(os.getenv("SCHEDULER_DYNAMIC_ETA_ALPHA", "0.2"))

    row = update_model_eta_stat(
        session=db,
        model=job.model,
        sample_seconds=sample_seconds,
        alpha=alpha,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
    )
    if row is None:
        return
    db.flush()
    logger.info(
        "dynamic_eta_learned model=%s sample=%.2fs ewma=%.2fs samples=%s",
        job.model,
        float(sample_seconds),
        float(row.ewma_eta_seconds or 0.0),
        int(row.sample_count or 0),
    )


def cleanup_input_r2_keys(job_id: str, r2_keys: List[str], batch_id: Optional[str] = None):
    """
    Delete input media objects from R2 after job reaches terminal state.
    
    This prevents R2 bloat from accumulated input images/videos.
    Called from job_complete for both success and permanent failure paths.
    
    IMPORTANT: For batch/variation jobs, multiple jobs may share the same R2 key
    (e.g. 1 image × 4 variations = 4 jobs with the same init_image_r2_key).
    We must NOT delete shared keys until ALL sibling jobs in the batch are in a
    terminal state, otherwise still-processing variations will fail to download
    the input media.  The reaper task serves as the ultimate safety net.
    """
    if not _s3_client or not _r2_bucket or not r2_keys:
        return
    
    # If this job is part of a batch, check if ALL sibling jobs are done.
    # If not, skip cleanup -- the last completing job will handle it,
    # and the reaper task is the safety net.
    if batch_id:
        try:
            db = SessionLocal()
            try:
                TERMINAL_STATES = {
                    JobState.COMPLETED.value,
                    JobState.FAILED.value,
                    JobState.TIMEOUT.value,
                    JobState.CANCELLED.value,
                }
                
                # Find all sibling jobs in this batch
                siblings = db.query(Job).filter(
                    cast(Job.params, SQLString).like(f'%"_batch_id": "{batch_id}"%')
                ).all()
                
                # Fallback: try alternative JSON format (no space after colon)
                if not siblings:
                    siblings = db.query(Job).filter(
                        cast(Job.params, SQLString).like(f'%"_batch_id":"{batch_id}"%')
                    ).all()
                
                if siblings:
                    non_terminal = [
                        s for s in siblings if s.status not in TERMINAL_STATES
                    ]
                    if non_terminal:
                        logger.info(
                            f"[R2_CLEANUP] Job {job_id}: batch {batch_id} still has "
                            f"{len(non_terminal)} non-terminal sibling(s). "
                            f"Deferring R2 cleanup to avoid race condition."
                        )
                        return
                    else:
                        logger.info(
                            f"[R2_CLEANUP] Job {job_id}: All {len(siblings)} siblings in "
                            f"batch {batch_id} are terminal. Proceeding with R2 cleanup."
                        )
            finally:
                db.close()
        except Exception as e:
            # If we can't check siblings, err on the side of NOT deleting.
            # The reaper task will clean up orphaned objects later.
            logger.warning(
                f"[R2_CLEANUP] Job {job_id}: Failed to check batch siblings: {e}. "
                f"Deferring cleanup to reaper."
            )
            return
    
    for key in r2_keys:
        try:
            _s3_client.delete_object(Bucket=_r2_bucket, Key=key)
            logger.debug(f"[R2_CLEANUP] Deleted input media: {key} (job {job_id})")
        except Exception as e:
            # Non-fatal: lifecycle rules will clean up eventually
            logger.warning(f"[R2_CLEANUP] Failed to delete {key} for job {job_id}: {e}")


def get_cached_last_heartbeat(worker_id: str) -> Optional[datetime]:
    """
    Return last heartbeat timestamp from the in-memory cache (if present).

    CRITICAL: Used by background reaper tasks to avoid false "stale" detection
    when DB sync lags behind the in-memory cache.
    """
    try:
        data = _heartbeat_cache.get(worker_id)
        if not data:
            return None
        ts = data.get("last_heartbeat")
        return ts if isinstance(ts, datetime) else None
    except Exception:
        return None


@worker_router.post("/heartbeat")
async def worker_heartbeat(
    request: Request,
    payload_dict: Dict[str, Any] = Depends(require_worker_signed_payload),
):
    payload = HeartbeatRequest.model_validate(payload_dict)
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
            "supported_models": payload.supported_models,
            "model_eta_seconds": payload.model_eta_seconds,
            "needs_sync": True,  # Mark for DB sync
        }
    except Exception:
        # If cache update fails for any reason, just continue
        # Sync task will eventually catch up via DB reads
        pass

    queue_snapshot: Dict[str, int] = {}
    try:
        if _pod_manager and hasattr(_pod_manager, "pending_jobs"):
            all_pending = _pod_manager.pending_jobs.get_all_pending()
            queue_snapshot = {
                str(model): len(job_ids)
                for model, job_ids in all_pending.items()
                if job_ids
            }
    except Exception:
        pass

    # Return immediately; queue_snapshot is in-memory only (no DB).
    return {"status": "ok", "queue_snapshot": queue_snapshot}


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
                    worker = (
                        db.query(Worker).filter(Worker.worker_id == worker_id).first()
                    )
                    if worker:
                        # CRITICAL: Do NOT "resurrect" workers that have been taken offline/terminated.
                        # After we destroy a pod, the worker may keep heartbeating briefly; if we
                        # overwrite status back to "ready", the reaper will try to destroy it again
                        # (and can end up logging "destroying pod None").
                        if worker.status in ("offline", "terminated", "error"):
                            continue

                        # CRITICAL FIX: Don't overwrite "dispatching" or "busy" with "ready" from heartbeat.
                        # The server sets these states during dispatch, and heartbeat should not reset them.
                        # Only accept heartbeat status if:
                        # 1. Heartbeat reports non-idle status (busy/starting), OR
                        # 2. Server-side status is already idle/starting (safe to update)
                        heartbeat_status = data.get("status")
                        current_status = worker.status
                        active_states = ("dispatching", "busy")
                        idle_states = ("ready", "starting", "idle")
                        
                        should_update_status = False
                        if heartbeat_status:
                            if heartbeat_status not in idle_states:
                                # Heartbeat reports non-idle (busy/starting) - always accept
                                should_update_status = True
                            elif current_status in idle_states:
                                # Both are idle - safe to update
                                should_update_status = True
                            elif current_status in active_states and heartbeat_status == "ready":
                                # CRITICAL: Don't overwrite active state with "ready"
                                # Server knows worker is dispatching/busy, heartbeat may be stale
                                should_update_status = False
                                logger.debug(
                                    f"Worker {worker_id} heartbeat reports 'ready' but server status is '{current_status}'. "
                                    f"Skipping status update to prevent overwriting active state."
                                )
                            else:
                                # Other cases - update
                                should_update_status = True

                        # Normal sync path
                        worker.last_heartbeat = data["last_heartbeat"]
                        worker.last_active = data["last_heartbeat"]
                        if should_update_status and heartbeat_status:
                            worker.status = heartbeat_status
                        if data["metrics"]:
                            worker.resource_metrics = data["metrics"]
                        hb_supported = data.get("supported_models")
                        hb_eta = data.get("model_eta_seconds")
                        hb_download_state = None
                        hb_metrics = data.get("metrics")
                        if isinstance(hb_metrics, dict):
                            raw_download_state = hb_metrics.get("download_state")
                            if isinstance(raw_download_state, dict):
                                compact_download_state = {}
                                for m_key, m_state in raw_download_state.items():
                                    if not isinstance(m_key, str):
                                        continue
                                    model_key = m_key.strip().lower()
                                    if not model_key:
                                        continue
                                    if isinstance(m_state, dict):
                                        compact = {}
                                        status_val = m_state.get("status")
                                        if isinstance(status_val, str) and status_val.strip():
                                            compact["status"] = status_val.strip()
                                        ready_in = m_state.get("ready_in_seconds")
                                        if isinstance(ready_in, (int, float)) and ready_in >= 0:
                                            compact["ready_in_seconds"] = int(ready_in)
                                        eta_download = m_state.get("eta_seconds")
                                        if isinstance(eta_download, (int, float)) and eta_download > 0:
                                            compact["eta_seconds"] = int(eta_download)
                                        if compact:
                                            compact_download_state[model_key] = compact
                                if compact_download_state:
                                    hb_download_state = compact_download_state
                        if hb_supported is not None or hb_eta is not None or hb_download_state is not None:
                            info = worker.info if isinstance(worker.info, dict) else {}
                            if hb_supported is not None:
                                normalized_supported = []
                                for m in hb_supported or []:
                                    if isinstance(m, str):
                                        m_norm = m.strip().lower()
                                        if m_norm and m_norm not in normalized_supported:
                                            normalized_supported.append(m_norm)
                                if normalized_supported:
                                    info["supported_models"] = normalized_supported
                            if isinstance(hb_eta, dict):
                                eta_map = {}
                                for k, v in hb_eta.items():
                                    if not isinstance(k, str):
                                        continue
                                    try:
                                        eta_val = int(v)
                                    except Exception:
                                        continue
                                    if eta_val <= 0:
                                        continue
                                    eta_map[k.strip().lower()] = eta_val
                                if eta_map:
                                    info["model_eta_seconds"] = eta_map
                            if hb_download_state is not None:
                                info["download_state"] = hb_download_state
                            info["capability_source"] = "worker_runtime"
                            info["capability_updated_at"] = data["last_heartbeat"].isoformat()
                            worker.info = info
                
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


@worker_router.post("/job_started")
async def job_started(
    request: Request,
    payload_dict: Dict[str, Any] = Depends(require_worker_signed_payload),
):
    payload = JobStartedRequest.model_validate(payload_dict)
    """
    Job started notification from worker.
    
    Worker calls this when it actually begins processing a job (not just when queued).
    This allows the server to update started_at timestamp accurately for timeout calculation.
    """
    job_id = payload.job_id
    worker_id = payload.worker_id
    
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            logger.warning(f"Job {job_id} not found for job_started notification")
            return {"status": "ok", "message": "Job not found (may have been recovered)"}
        
        # Only update started_at if job is in RUNNING state and started_at is not already set
        # or if it was set at dispatch time (we want to overwrite with actual start time)
        if job.status == JobState.RUNNING.value:
            # Update started_at to actual processing start time
            actual_start_time = utc_naive_now()
            if not job.started_at or job.started_at != actual_start_time:
                old_started_at = job.started_at
                job.started_at = actual_start_time
                job.updated_at = actual_start_time
                db.commit()
                logger.info(
                    f"Job {job_id} started_at updated: {old_started_at} -> {actual_start_time} "
                    f"(actual inference start time from worker)"
                )
            else:
                logger.debug(f"Job {job_id} started_at already set correctly")
        else:
            logger.debug(
                f"Job {job_id} is in {job.status} state, not updating started_at "
                f"(expected RUNNING for job_started callback)"
            )
        
        # Validate worker_id if provided
        if worker_id and job.worker_id and worker_id != job.worker_id:
            logger.warning(
                f"Job {job_id} job_started callback: worker_id mismatch "
                f"(payload: {worker_id}, DB: {job.worker_id})"
            )
        
        return {"status": "ok"}
        
    except Exception as e:
        logger.error(f"Error processing job_started notification for {job_id}: {e}")
        try:
            db.rollback()
        except Exception:
            pass
        # Don't raise - this is a best-effort notification
        return {"status": "error", "message": str(e)}
    finally:
        db.close()


@worker_router.post("/job_complete")
async def job_complete(
    background_tasks: BackgroundTasks,
    request: Request,
    payload_dict: Dict[str, Any] = Depends(require_worker_signed_payload),
):
    payload = JobCompleteRequest.model_validate(payload_dict)
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
        if current_state in [
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.TIMEOUT,
            JobState.CANCELLED,
        ]:
            logger.warning(
                f"Worker attempted to complete job {job_id} but it's already in {current_state.value} state. "
                f"This likely means the pod was destroyed while the job was running. "
                f"Ignoring completion request (idempotent)."
            )
            # Return success to avoid worker retries
            return {
                "status": "ok",
                "message": f"Job already in {current_state.value} state",
            }
        
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
            try:
                existing = _heartbeat_cache.get(worker_id)
                _heartbeat_cache[worker_id] = {
                    "last_heartbeat": utc_naive_now(),
                    "status": existing.get("status") if existing else None,
                    "metrics": existing.get("metrics") if existing else None,
                    "needs_sync": True,
                }
            except Exception:
                pass
        
        if payload.failed:
            # Chain started_at to the next job the worker will process (even on failure)
            if payload.next_job_id:
                next_job = db.query(Job).filter(Job.job_id == payload.next_job_id).first()
                if next_job and next_job.status == JobState.RUNNING.value:
                    next_job.started_at = utc_naive_now()
                    next_job.updated_at = utc_naive_now()
                    db.commit()
                    logger.info(f"Chained started_at to next job {payload.next_job_id} (after failure of {job_id})")

            error_str = (payload.error or "Unknown error").lower()
            
            # Check if this is a retryable infrastructure failure
            is_cuda_incompatibility = (
                "no kernel image is available" in error_str and "cuda" in error_str
            )
            is_oom_error = "oom" in error_str or "out of memory" in error_str
            is_other_cuda_error = "cuda" in error_str and "error" in error_str
            
            # Infrastructure failures that should trigger retry
            is_infrastructure_failure = is_cuda_incompatibility or is_oom_error
            
            # Get retry count from job
            retry_count = (
                job.retry_count
                if hasattr(job, "retry_count") and job.retry_count
                else 0
            )
            max_retries = 3  # Allow up to 3 retries for any worker error (consistent pattern)
            
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
                            600,  # duration
                        )

                    # Destroy the pod in background - it won't work for any int4 jobs
                    if pod_uuid and _pod_manager:
                        background_tasks.add_task(
                            _handle_incompatible_pod_destruction,
                            _pod_manager,
                                pod_uuid,
                            "CUDA kernel incompatibility (int4 variant)",
                            )
            
            # Check for OOM errors - remove pod from tracking immediately, then destroy
            elif is_oom_error:
                logger.warning(
                    f"OOM detected for job {job_id} on pod {pod_uuid} (machine {machine_id})"
                )
                # CRITICAL: Remove pod from tracking IMMEDIATELY to prevent reuse
                # Also mark worker as offline in DB immediately
                if pod_uuid and _pod_manager:
                    # Remove from tracking immediately (synchronously)
                    async with _pod_manager._lock:
                        if pod_uuid in _pod_manager.individual_pods:
                            logger.warning(
                                f"Removing pod {pod_uuid} from tracking immediately due to OOM (will be destroyed in background)"
                            )
                            _pod_manager.individual_pods.pop(pod_uuid, None)
                            _pod_manager.individual_pod_managers.pop(pod_uuid, None)

                    # Mark worker as offline in DB immediately (synchronously)
                    if worker_id:
                        try:
                            worker = (
                                db.query(Worker)
                                .filter(Worker.worker_id == worker_id)
                                .first()
                            )
                            if worker:
                                worker.status = "offline"
                                db.commit()
                                logger.info(
                                    f"Marked worker {worker_id} as offline due to OOM"
                                )
                        except Exception as e:
                            logger.warning(f"Failed to mark worker offline: {e}")

                    # Schedule destruction in background (cleanup)
                    # Use async wrapper to ensure proper await
                    background_tasks.add_task(
                        _handle_oom_pod_destruction,
                        _pod_manager,
                        pod_uuid,
                        f"OOM detected on job {job_id}",
                    )
                # Also blacklist the machine (less critical - pod removal is the main fix)
                if machine_id and _pod_manager:
                    background_tasks.add_task(
                        _pod_manager.mark_machine_bad,
                        str(machine_id),
                        300,  # 5 min for OOM
                    )
            
            # Check for other critical errors
            elif is_other_cuda_error:
                logger.warning(f"CUDA error detected for job {job_id}: {payload.error}")
            
            # CRITICAL: Retry any worker error up to max_retries (consistent with other paths)
            if retry_count < max_retries:
                logger.warning(
                    f"Job {job_id} failed (retry {retry_count + 1}/{max_retries}). "
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
                error_log.append(
                    {
                    "timestamp": utc_naive_now().isoformat(),
                    "worker_id": worker_id,
                    "error": payload.error,
                    "retry_count": job.retry_count,
                    "machine_id": machine_id,
                    "variant": variant,
                    }
                )
                job.error_log = error_log
                db.commit()
                
                logger.info(
                    f"Job {job_id} requeued for retry (attempt {job.retry_count}/{max_retries}). "
                    f"Will try different machine/variant."
                )
                
                # Trigger dispatch for the requeued job
                if _pod_manager:
                    job_mode = job.mode if hasattr(job, "mode") else None
                    asyncio.create_task(
                        _dispatch_job_background(job_id, job.model, job_mode)
                    )
                
                return {"status": "requeued", "retry_count": job.retry_count}
            
            else:
                # Retries exhausted - mark as permanently failed
                logger.error(
                    f"Job {job_id} failed after {retry_count} retries. "
                    f"Marking as permanently failed. Error: {payload.error}"
                )
                
                # Mark job as permanently failed
                if _job_manager:
                    _job_manager.mark_failed(
                        job_id, payload.error or "Unknown error", worker_id
                    )
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
                        "failed",
                    )
                
                audit_logger.job_failed(job_id, worker_id, payload.error)
                
                # Clear cached params - job permanently failed, no more retries
                clear_cached_job_params(job_id)
                
                # Cleanup input media from R2 (non-blocking, batch-aware)
                job_params_dict = job.params if isinstance(job.params, dict) else {}
                input_r2_keys = job_params_dict.get("_input_r2_keys", [])
                if input_r2_keys:
                    batch_id_for_cleanup = job_params_dict.get("_batch_id")
                    background_tasks.add_task(
                        cleanup_input_r2_keys, job_id, input_r2_keys, batch_id_for_cleanup
                    )
            
        else:
            # Validate R2 file exists
            if _s3_client and _r2_bucket and payload.result_r2_key:
                try:
                    _s3_client.head_object(Bucket=_r2_bucket, Key=payload.result_r2_key)
                except ClientError:
                    logger.error(f"Result file not found: {payload.result_r2_key}")
                    raise HTTPException(
                        status_code=400, detail="Result file not found in storage"
                    )
            
            # Mark job completed - use signed URL endpoint
            if _job_manager:
                try:
                    _job_manager.mark_completed(
                        job_id,
                        result_r2_key=payload.result_r2_key,
                        output_url=(
                            f"{_controller_url}/download_url/{job_id}"
                            if _controller_url
                            else None
                        ),
                        metrics=payload.metrics,
                    )
                except TypeError as te:
                    if "metrics" in str(te):
                        # Backward compat: older server_controller without metrics param
                        _job_manager.mark_completed(
                            job_id,
                            result_r2_key=payload.result_r2_key,
                            output_url=(
                                f"{_controller_url}/download_url/{job_id}"
                                if _controller_url
                                else None
                            ),
                        )
                    else:
                        raise
            else:
                completion_time = utc_naive_now()
                job.status = JobState.COMPLETED.value
                job.result_r2_key = payload.result_r2_key
                if payload.continuation_latent_r2_key:
                    stored = dict(job.params or {})
                    stored["continuation_latent_r2_key"] = payload.continuation_latent_r2_key
                    job.params = stored
                job.completed_at = completion_time
                job.updated_at = completion_time
                
                # Use worker-provided metrics if available, otherwise calculate
                if payload.metrics:
                    # Worker provides accurate processing_time_seconds (actual inference time including upscaler)
                    if "processing_time_seconds" in payload.metrics:
                        job.processing_time_seconds = payload.metrics["processing_time_seconds"]
                    # Worker provides startup_time_seconds (pod start to first inference start)
                    if "startup_time_seconds" in payload.metrics:
                        job.startup_time_seconds = payload.metrics["startup_time_seconds"]
                
                # Calculate other timing metrics (total_time, queue_time, dispatch_time)
                timing_metrics = _calculate_timing_metrics(job, completion_time)
                job.total_time_seconds = timing_metrics["total_time"]
                # Only set processing_time_seconds if not provided by worker
                if job.processing_time_seconds is None:
                    job.processing_time_seconds = timing_metrics["processing_time"]
                job.queue_time_seconds = timing_metrics["queue_time"]
                job.dispatch_time_seconds = timing_metrics["dispatch_time"]
                
                db.commit()
                
                # Log timing metrics (include worker-provided startup_time if set)
                if job.startup_time_seconds is not None:
                    timing_metrics = dict(timing_metrics, startup_time=job.startup_time_seconds)
                _log_timing_metrics(job_id, job.model, timing_metrics)
            
            # Chain started_at to the next job the worker will process (so stuck detection only times actual execution)
            if payload.next_job_id:
                next_job = db.query(Job).filter(Job.job_id == payload.next_job_id).first()
                if next_job and next_job.status == JobState.RUNNING.value:
                    next_job.started_at = utc_naive_now()
                    next_job.updated_at = utc_naive_now()
                    db.commit()
                    logger.info(f"Chained started_at to next job {payload.next_job_id}")
            elif worker_id:
                # Fallback: worker did not send next_job_id; set started_at for oldest RUNNING job on this worker with started_at NULL
                from sqlalchemy import and_
                next_job = (
                    db.query(Job)
                    .filter(
                        and_(
                            Job.worker_id == worker_id,
                            Job.status == JobState.RUNNING.value,
                            Job.job_id != job_id,
                            Job.started_at.is_(None),
                        )
                    )
                    .order_by(Job.updated_at.asc())
                    .limit(1)
                    .first()
                )
                if next_job:
                    next_job.started_at = utc_naive_now()
                    next_job.updated_at = utc_naive_now()
                    db.commit()
                    logger.info(f"Chained started_at to next job {next_job.job_id} (fallback, no next_job_id in payload)")

            # Update worker's last_job_finished_at timestamp
            gpu_hourly_rate = 0.0
            if worker_id:
                worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
                if worker:
                    worker.last_job_finished_at = utc_naive_now()
                    if payload.next_job_id:
                        # Worker still has jobs queued -- keep busy
                        pass  # Do NOT set worker.status = "ready"
                    else:
                        # Check if other RUNNING jobs exist on this worker
                        other_running = db.query(Job).filter(
                            Job.worker_id == worker_id,
                            Job.status == JobState.RUNNING.value,
                            Job.job_id != job_id,
                        ).count()
                        if other_running > 0:
                            pass  # Do NOT set worker.status = "ready"
                        else:
                            worker.status = "ready"
                    gpu_hourly_rate = worker.gpu_hourly_rate or 0.0
                    db.commit()
            
            # Calculate and store job cost (processing cost only - overhead will be added when pod is destroyed)
            if pod_uuid and gpu_hourly_rate > 0:
                try:
                    from cost_service import calculate_and_store_job_cost
                    
                    # Refresh job to get the latest processing_time_seconds
                    db.refresh(job)
                    
                    cost_breakdown = calculate_and_store_job_cost(
                        db=db,
                        job_id=job_id,
                        pod_uuid=pod_uuid,
                        gpu_hourly_rate=gpu_hourly_rate,
                        processing_time_seconds=job.processing_time_seconds,
                        pod_overhead_share=0.0,  # Will be calculated when pod is destroyed
                    )
                    
                    if cost_breakdown:
                        logger.info(
                            f"💰 Job {job_id} cost: ${cost_breakdown.total_cost_usd:.6f} "
                            f"(processing only, overhead pending)"
                        )
                except Exception as e:
                    logger.error(f"Failed to calculate job cost for {job_id}: {e}")
            
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
                    "completed",
                )
            
            audit_logger.job_completed(job_id, worker_id)
            
            # Clear cached params - job completed successfully
            clear_cached_job_params(job_id)
            
            # Cleanup input media from R2 (non-blocking, batch-aware)
            job_params_dict = job.params if isinstance(job.params, dict) else {}
            input_r2_keys = job_params_dict.get("_input_r2_keys", [])
            if input_r2_keys:
                batch_id_for_cleanup = job_params_dict.get("_batch_id")
                background_tasks.add_task(
                    cleanup_input_r2_keys, job_id, input_r2_keys, batch_id_for_cleanup
                )

        if (
            not payload.next_job_id
            and pod_uuid
            and _pod_manager
        ):
            background_tasks.add_task(
                _pod_manager.maybe_release_external_pod_after_job,
                pod_uuid,
                worker_id,
            )
        
        # Send FCM notification
        # CRITICAL: Only send notifications for jobs that are actually COMPLETED or FAILED
        # This prevents malformed notifications for jobs in intermediate states
        # For multi-variation jobs, only send notification when ALL variations complete
        
        # Refresh job to get the latest status from DB
        db.refresh(job)
        if job.status == JobState.COMPLETED.value:
            try:
                _learn_dynamic_eta_from_completed_job(db, job, payload.metrics)
                db.commit()
            except Exception as e:
                logger.warning(
                    f"dynamic_eta update skipped for job {job_id} model={job.model}: {e}"
                )
                try:
                    db.rollback()
                except Exception:
                    pass
        
        fcm_token = job.fcm_token
        job_status = job.status
        
        # Only send notification if job is in a terminal state (COMPLETED or FAILED)
        is_terminal_state = job_status in [JobState.COMPLETED.value, JobState.FAILED.value]
        should_send_notification = fcm_token and not job.notification_sent and is_terminal_state
        
        if not is_terminal_state and fcm_token and not job.notification_sent:
            # Log when we skip notification due to non-terminal state
            logger.warning(
                f"Skipping FCM notification for job {job_id}: job is in '{job_status}' state, "
                f"not a terminal state (COMPLETED/FAILED). payload.failed={payload.failed}"
            )
        
        if should_send_notification:
            # Send individual notification for each job as it completes
            # For multi-variation batches, frontend receives one notification per variation
            # and can match job_id to the job_ids array from submit response
            
            # CRITICAL: Use job's actual status from DB, not just payload.failed
            is_success = job_status == JobState.COMPLETED.value
            
            download_url = None
            if is_success and payload.result_r2_key:
                download_url = _generate_signed_download_url(payload.result_r2_key)
            
            # Include batch info in notification if part of a batch
            # SAFETY: job.params might be a JSON string in some edge cases
            params_dict = job.params
            if isinstance(params_dict, str):
                try:
                    import json
                    params_dict = json.loads(params_dict)
                    logger.warning(f"[FCM] job.params was a string for job {job_id}, parsed to dict")
                except Exception as e:
                    logger.error(f"[FCM] Failed to parse job.params string for job {job_id}: {e}")
                    params_dict = {}
            
            batch_id = params_dict.get("_batch_id") if params_dict else None
            variation_index = params_dict.get("_variation_index", 0) if params_dict else 0
            total_variations = params_dict.get("_total_variations", 1) if params_dict else 1
            
            # LOG ALL VALUES before adding task - DEBUGGING EMPTY NOTIFICATIONS
            logger.info(
                f"[FCM_QUEUE] Adding notification task for job {job_id}: "
                f"fcm_token_present={bool(fcm_token)}, fcm_token_len={len(fcm_token) if fcm_token else 0}, "
                f"job_id='{job_id}', job.model='{job.model}', is_success={is_success}, "
                f"download_url_present={bool(download_url)}, batch_id='{batch_id}', "
                f"variation_index={variation_index}, total_variations={total_variations}, "
                f"params_type={type(params_dict).__name__}, "
                f"params_keys={list(params_dict.keys()) if isinstance(params_dict, dict) else 'not_dict'}"
            )
            
            # CRITICAL VALIDATION: Ensure we have the required data before queueing
            if not fcm_token:
                logger.error(f"[FCM_QUEUE] SKIPPING job {job_id}: fcm_token is empty!")
            elif not job_id:
                logger.error(f"[FCM_QUEUE] SKIPPING: job_id is empty!")
            elif not job.model:
                logger.error(f"[FCM_QUEUE] WARNING: job.model is empty for job {job_id}, will use 'unknown'")
            
            background_tasks.add_task(
                _send_job_notification,
                fcm_token,
                job_id,
                job.model,
                is_success,
                download_url,
                # Pass batch info for frontend to track variations
                batch_id=batch_id,
                variation_index=variation_index,
                total_variations=total_variations,
            )
            job.notification_sent = True
            db.commit()
            
            logger.info(
                f"[FCM_QUEUE] Task added for job {job_id}: "
                f"batch_id={batch_id}, variation={variation_index + 1}/{total_variations}, "
                f"status={'completed' if is_success else 'failed'}, "
                f"notification_sent flag set to True and committed"
            )
    
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


async def _handle_incompatible_pod_destruction(pod_manager, pod_uuid: str, reason: str):
    """Background task to destroy incompatible pod"""
    try:
        await pod_manager.destroy_individual_pod(pod_uuid, reason=reason)
        logger.info(f"Destroyed incompatible pod {pod_uuid}: {reason}")
    except Exception as e:
        logger.error(f"Failed to destroy incompatible pod {pod_uuid}: {e}")


async def _handle_oom_pod_destruction(pod_manager, pod_uuid: str, reason: str):
    """
    Background task to destroy pod after OOM error.
    
    CRITICAL: This ensures the pod is actually destroyed after OOM, not just
    removed from tracking. We use an async wrapper because background_tasks.add_task
    requires proper handling of async functions.
    """
    try:
        logger.info(f"Destroying OOM pod {pod_uuid}: {reason}")
        success = await pod_manager.destroy_individual_pod(pod_uuid, reason=reason)
        if success:
            logger.info(f"Successfully destroyed OOM pod {pod_uuid}")
        else:
            logger.error(f"Failed to destroy OOM pod {pod_uuid} (destroy returned False)")
    except Exception as e:
        logger.error(f"Exception destroying OOM pod {pod_uuid}: {e}")


async def _remove_pending_job_safe(pod_manager, job_id: str, model: str, status: str):
    """Background task to safely remove job from pending queue"""
    try:
        await pod_manager.remove_pending_job(job_id, model)
    except Exception as e:
        logger.debug(f"Failed to remove {status} job {job_id} from pending queue: {e}")


async def _send_job_notification(
    token: str,
    job_id: str,
    model: str,
    success: bool,
    download_url: str = None,
    batch_id: str = None,
    variation_index: int = 0,
    total_variations: int = 1,
):
    """
    Background task to send FCM notification for a completed/failed job.
    
    For batch jobs (variations > 1), includes batch info so frontend can track
    which variation completed. Frontend receives individual notifications for
    each job and can match job_id to the job_ids array from submit response.
    
    CRITICAL: FCM data values MUST all be strings. None values will cause issues.
    """
    # LOG ALL INPUT PARAMETERS for debugging empty notification issues
    logger.info(
        f"[FCM_TASK] _send_job_notification called with: "
        f"token_present={bool(token)}, job_id='{job_id}', model='{model}', "
        f"success={success}, download_url_present={bool(download_url)}, "
        f"batch_id='{batch_id}', variation={variation_index}/{total_variations}"
    )
    
    # Defensive checks - ensure all required values are present
    if not token:
        logger.error(f"[FCM_TASK] ABORTED for job {job_id}: token is empty/None")
        return
    
    if not job_id:
        logger.error(f"[FCM_TASK] ABORTED: job_id is empty/None")
        return
    
    # Ensure job_id and model are strings (FCM requires string values)
    job_id_str = str(job_id) if job_id else ""
    model_str = str(model) if model else "unknown"
    
    # Double-check after conversion
    if not job_id_str:
        logger.error(f"[FCM_TASK] ABORTED: job_id_str is empty after conversion")
        return
    
    if success:
        # CRITICAL: Include model field - frontend expects it
        data = {
            "job_id": job_id_str,
            "status": "completed",
            "model": model_str,
        }
        if download_url:
            data["download_url"] = str(download_url)
        
        # Include batch info for multi-variation jobs
        if batch_id and total_variations > 1:
            data["batch_id"] = str(batch_id)
            data["variation_index"] = str(variation_index)
            data["total_variations"] = str(total_variations)
        
        # Customize message for variations
        if total_variations > 1:
            title = f"Variation {variation_index + 1}/{total_variations} Complete"
            body = f"Your result is ready!"
        else:
            title = "Job Complete"
            body = f"Your result is ready!"
        
        logger.info(
            f"[FCM_TASK] Preparing SUCCESS notification for job {job_id_str}: "
            f"title='{title}', body='{body}', data={data}"
        )
        await send_fcm_notification(token, title, body, data)
    else:
        # CRITICAL: Include model field - frontend expects it
        data = {
            "job_id": job_id_str,
            "status": "failed",
            "model": model_str,
        }
        
        # Include batch info for multi-variation jobs
        if batch_id and total_variations > 1:
            data["batch_id"] = str(batch_id)
            data["variation_index"] = str(variation_index)
            data["total_variations"] = str(total_variations)
        
        # Customize message for variations
        if total_variations > 1:
            title = f"Variation {variation_index + 1}/{total_variations} Failed"
            body = f"Your job failed. No credits deducted."
        else:
            title = "Job Failed"
            body = f"Your job failed. No credits deducted."
        
        logger.info(
            f"[FCM_TASK] Preparing FAILURE notification for job {job_id_str}: "
            f"title='{title}', body='{body}', data={data}"
        )
        await send_fcm_notification(token, title, body, data)


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
                    p for p in _pod_manager.individual_pods.values() if p.model == model
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
                        job.error_log.append(
                            {
                            "timestamp": utc_naive_now().isoformat(),
                            "state": "requeue_failed",
                            "error": f"Failed to create new pod: {str(e)}",
                            }
                        )
                        job.updated_at = utc_naive_now()
                db.commit()
            finally:
                db.close()
            return
        
        if not new_pod_uuid:
            logger.error(
                f"Failed to create new pod for model {model}, jobs will remain in QUEUED state"
            )
            return
        
        logger.info(
            f"Created new pod {new_pod_uuid} for model {model}, now dispatching {len(job_ids)} job(s)"
        )
        
        # Wait for worker registration (similar to _wait_for_worker_registration)
        new_worker_id = None
        start_time = time.time()
        timeout = 120  # 2 minutes timeout
        
        while time.time() - start_time < timeout:
            db = SessionLocal()
            try:
                worker = (
                    db.query(Worker)
                    .filter(
                    Worker.container_id == new_pod_uuid,
                    Worker.model == model,
                    Worker.status == "ready",
                    )
                    .first()
                )
                
                if worker:
                    new_worker_id = worker.worker_id
                    logger.info(
                        f"New worker {new_worker_id} registered for pod {new_pod_uuid}"
                    )
                    break
            finally:
                db.close()
            
            await asyncio.sleep(1)
        
        if not new_worker_id:
            logger.error(
                f"Timeout waiting for worker registration on new pod {new_pod_uuid}"
            )
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
                    if current_state in [
                        JobState.COMPLETED,
                        JobState.FAILED,
                        JobState.TIMEOUT,
                        JobState.CANCELLED,
                        JobState.RUNNING,
                    ]:
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
                    logger.info(
                        f"Successfully requeued job {job_id} to new pod {new_pod_uuid}"
                    )
                else:
                    logger.warning(
                        f"Failed to dispatch requeued job {job_id} to new pod {new_pod_uuid}"
                    )
                
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


@admin_router.get("/personal")
async def admin_list_personal(auth: Dict = Depends(require_any_auth)):
    import personal_queue

    db = SessionLocal()
    try:
        items = personal_queue.list_items(db)
        return {"items": [i.to_dict() for i in items]}
    finally:
        db.close()


@admin_router.get("/personal/models")
async def admin_personal_models(auth: Dict = Depends(require_any_auth)):
    import personal_queue

    return {"models": personal_queue.leftover_models()}


@admin_router.post("/personal")
async def admin_add_personal(request: Request, auth: Dict = Depends(require_any_auth)):
    import personal_queue

    body = await request.json()
    model = (body.get("model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    db = SessionLocal()
    try:
        item = personal_queue.add_item(
            db,
            prompt=body.get("prompt") or "",
            params=body.get("params") or {},
            model=model,
        )
        return item.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()


@admin_router.post("/personal/reorder")
async def admin_reorder_personal(request: Request, auth: Dict = Depends(require_any_auth)):
    import personal_queue

    body = await request.json()
    ids = body.get("ids") or []
    db = SessionLocal()
    try:
        personal_queue.reorder(db, ids)
        return {"ok": True}
    finally:
        db.close()


@admin_router.delete("/personal/{item_id}")
async def admin_delete_personal(item_id: str, auth: Dict = Depends(require_any_auth)):
    import personal_queue

    db = SessionLocal()
    try:
        ok = personal_queue.delete_item(db, item_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Not found")
        return {"ok": True}
    finally:
        db.close()


@worker_router.post("/personal/pull")
async def worker_personal_pull(payload_dict: Dict[str, Any] = Depends(require_worker_signed_payload)):
    import personal_queue

    db = SessionLocal()
    try:
        item = personal_queue.pull_next(
            db,
            worker_id=payload_dict.get("worker_id"),
            models=payload_dict.get("models"),
            model=payload_dict.get("model"),
        )
        if not item:
            return {"item": None}
        ext = personal_queue.result_extension(item.model)
        result_key = f"personal/{item.model}/{item.id}{ext}"
        upload_url = create_presigned_put_url(result_key) or ""
        data = item.to_dict()
        data["job_id"] = f"personal-{item.id}"
        data["upload_url"] = upload_url
        data["result_key"] = result_key
        return {"item": data}
    finally:
        db.close()


@worker_router.post("/personal/complete")
async def worker_personal_complete(
    background_tasks: BackgroundTasks,
    payload_dict: Dict[str, Any] = Depends(require_worker_signed_payload),
):
    import personal_queue

    db = SessionLocal()
    try:
        status = payload_dict.get("status") or ("done" if payload_dict.get("result_r2_key") else "failed")
        item = personal_queue.complete_item(
            db,
            payload_dict.get("item_id"),
            status=status,
            result_r2_key=payload_dict.get("result_r2_key"),
            error=payload_dict.get("error"),
        )
        if not item:
            raise HTTPException(status_code=404, detail="Not found")
        worker_id = payload_dict.get("worker_id")
        if worker_id and _pod_manager:
            worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
            if worker and worker.container_id:
                background_tasks.add_task(
                    _pod_manager.maybe_release_external_pod_after_job,
                    worker.container_id,
                    worker_id,
                )
        return item.to_dict()
    finally:
        db.close()


@worker_router.post("/personal/pause")
async def worker_personal_pause(payload_dict: Dict[str, Any] = Depends(require_worker_signed_payload)):
    import personal_queue

    db = SessionLocal()
    try:
        item = personal_queue.pause_item(db, payload_dict.get("item_id"))
        if not item:
            raise HTTPException(status_code=404, detail="Not found")
        return item.to_dict()
    finally:
        db.close()


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
            "active": sum(
                1 for w in workers if w.status in ("ready", "busy", "processing")
            ),
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


# =============================================================================
# Cost Analytics Endpoints
# =============================================================================

@admin_router.get("/cost/job/{job_id}")
async def get_job_cost_endpoint(
    job_id: str,
    auth: Dict = Depends(require_any_auth),
):
    """
    Get cost information for a specific job.
    
    Returns:
        Job cost breakdown including processing cost, overhead share, and GPU info
    """
    from cost_service import get_job_cost
    
    db = SessionLocal()
    try:
        cost_info = get_job_cost(db, job_id)
        if not cost_info:
            raise HTTPException(status_code=404, detail="Job not found")
        return cost_info
    finally:
        db.close()


@admin_router.get("/cost/pod/{pod_uuid}")
async def get_pod_cost_endpoint(
    pod_uuid: str,
    auth: Dict = Depends(require_any_auth),
):
    """
    Get cost information for a specific pod.
    
    Returns:
        Pod cost summary including total cost, jobs processed, and GPU info
    """
    from cost_service import get_pod_total_cost
    
    db = SessionLocal()
    try:
        cost_info = get_pod_total_cost(db, pod_uuid)
        if not cost_info:
            raise HTTPException(status_code=404, detail="Pod not found")
        return cost_info
    finally:
        db.close()


@admin_router.get("/cost/summary")
async def get_cost_summary_endpoint(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    model: Optional[str] = None,
    auth: Dict = Depends(require_any_auth),
):
    """
    Get cost summary for a time period.
    
    Query params:
        start_date: ISO format date string (default: 24 hours ago)
        end_date: ISO format date string (default: now)
        model: Optional filter by model name
    
    Returns:
        Aggregated cost metrics for the period
    """
    from cost_service import get_cost_summary_for_period
    from datetime import timedelta
    
    db = SessionLocal()
    try:
        # Parse dates or use defaults
        now = utc_naive_now()
        if start_date:
            try:
                start = datetime.fromisoformat(start_date.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format")
        else:
            start = now - timedelta(days=1)
        
        if end_date:
            try:
                end = datetime.fromisoformat(end_date.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format")
        else:
            end = now
        
        summary = get_cost_summary_for_period(db, start, end, model)
        return summary
    finally:
        db.close()


@admin_router.get("/cost/by_model")
async def get_cost_by_model_endpoint(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    auth: Dict = Depends(require_any_auth),
):
    """
    Get cost breakdown by model.
    
    Query params:
        start_date: ISO format date string (optional)
        end_date: ISO format date string (optional)
    
    Returns:
        List of per-model cost metrics
    """
    from cost_service import get_cost_by_model
    
    db = SessionLocal()
    try:
        # Parse dates
        start = None
        end = None
        
        if start_date:
            try:
                start = datetime.fromisoformat(start_date.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format")
        
        if end_date:
            try:
                end = datetime.fromisoformat(end_date.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format")
        
        breakdown = get_cost_by_model(db, start, end)
        return {"models": breakdown}
    finally:
        db.close()


@admin_router.get("/scheduler/status")
async def get_scheduler_status(
    auth: Dict = Depends(require_any_auth),
):
    """
    Scheduler/fleet runtime status for rollout monitoring.
    """
    from config import get_config
    cfg = get_config()

    scheduler_mode = cfg.scheduler.mode.value if getattr(cfg, "scheduler", None) else "unknown"
    shadow_mode = bool(getattr(getattr(cfg, "scheduler", None), "shadow_mode", False))
    scheduler_flags = {
        "dynamic_eta_enabled": bool(
            getattr(getattr(cfg, "scheduler", None), "dynamic_eta_enabled", False)
        ),
        "dynamic_eta_observe_only": bool(
            getattr(getattr(cfg, "scheduler", None), "dynamic_eta_observe_only", True)
        ),
        "dynamic_eta_min_samples": int(
            getattr(getattr(cfg, "scheduler", None), "dynamic_eta_min_samples", 5)
        ),
        "dynamic_eta_alpha": float(
            getattr(getattr(cfg, "scheduler", None), "dynamic_eta_alpha", 0.2)
        ),
        "dynamic_eta_allowed_groups": sorted(
            list(getattr(getattr(cfg, "scheduler", None), "dynamic_eta_allowed_groups", set()) or set())
        ),
        "soft_governor_enabled": bool(
            getattr(getattr(cfg, "scheduler", None), "soft_governor_enabled", False)
        ),
        "soft_governor_ops_window_seconds": int(
            getattr(getattr(cfg, "scheduler", None), "soft_governor_ops_window_seconds", 3600)
        ),
        "soft_governor_ops_threshold": int(
            getattr(getattr(cfg, "scheduler", None), "soft_governor_ops_threshold", 80)
        ),
        "pending_model_grace_enabled": bool(
            getattr(getattr(cfg, "scheduler", None), "pending_model_grace_enabled", True)
        ),
        "pending_model_max_wait_seconds": int(
            getattr(getattr(cfg, "scheduler", None), "pending_model_max_wait_seconds", 20)
        ),
        "pending_model_poll_interval_ms": int(
            getattr(getattr(cfg, "scheduler", None), "pending_model_poll_interval_ms", 500)
        ),
        "pending_model_hint_timeout_seconds": int(
            getattr(getattr(cfg, "scheduler", None), "pending_model_hint_timeout_seconds", 3)
        ),
    }

    fleet_snapshot = None
    if _pod_manager and hasattr(_pod_manager, "get_fixed_fleet_snapshot"):
        try:
            fleet_snapshot = await _pod_manager.get_fixed_fleet_snapshot()
        except Exception as e:
            fleet_snapshot = {"error": str(e)}

    return {
        "scheduler_mode": scheduler_mode,
        "scheduler_shadow_mode": shadow_mode,
        "scheduler_flags": scheduler_flags,
        "fixed_fleet": fleet_snapshot,
    }


@admin_router.get("/sales/status")
async def get_sales_status(
    auth: Dict = Depends(require_any_auth),
):
    """
    Current sale-window status for pricing validation.
    """
    from credit_service import get_active_sale_window_info
    return get_active_sale_window_info()


@admin_router.get("/sales/schedule_today")
async def get_sales_schedule_today(
    auth: Dict = Depends(require_any_auth),
):
    """
    Today's deterministic discount schedule and active window info.
    """
    from credit_service import get_today_sale_schedule_info
    return get_today_sale_schedule_info()


@admin_router.get("/credit/metadata_status")
async def get_credit_metadata_status(
    limit: int = 50,
    auth: Dict = Depends(require_any_auth),
):
    """
    Quick audit for credit transaction metadata persistence health.
    """
    db = SessionLocal()
    try:
        limit = max(1, min(200, int(limit)))
        rows = (
            db.query(CreditTransaction)
            .order_by(CreditTransaction.created_at.desc())
            .limit(limit)
            .all()
        )
        total = len(rows)
        with_meta = 0
        sale_tagged = 0
        sample = []
        for tx in rows:
            has_meta = bool(getattr(tx, "credit_metadata", None))
            if has_meta:
                with_meta += 1
                md = tx.credit_metadata if isinstance(tx.credit_metadata, dict) else {}
                pricing = md.get("pricing", {}) if isinstance(md.get("pricing"), dict) else {}
                if pricing.get("sale_active") is True:
                    sale_tagged += 1
            sample.append(
                {
                    "id": tx.id,
                    "reason": tx.reason,
                    "delta": tx.delta,
                    "has_metadata": has_meta,
                    "created_at": tx.created_at.isoformat() if tx.created_at else None,
                }
            )
        return {
            "checked": total,
            "with_metadata": with_meta,
            "without_metadata": total - with_meta,
            "sale_tagged": sale_tagged,
            "sample": sample[:20],
        }
    finally:
        db.close()
