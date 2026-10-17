# models.py
"""
Pydantic Models for API Request/Response Validation

Provides:
- Request validation with detailed error messages
- Response serialization
- Type safety across the application
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import re


# One bulk *task*: a single R2 object key, or a list of keys for multi-reference models.
# Example: ["key1", "key2"] → two tasks; [["a","b","c"]] → one task with three references.
BulkInputImageEntry = Union[str, List[str]]

# Params must not carry raw/base64 media; use *_r2_key fields and presigned uploads instead.
_FORBIDDEN_RAW_MEDIA_PARAM_KEYS = frozenset({
    "init_image",
    "image",
    "mask_image",
    "mask",
    "input_video",
    "ref_image",
    "ref_video",
    "control_image",
    "source_image",
    "reference_image",
    "style_image",
    "face_image",
    "input_image",
    "background_image",
    "foreground_image",
    "depth_image",
    "normal_image",
    "images",
})

# =============================================================================
# Enums
# =============================================================================

class JobStatus(str, Enum):
    """Job status states"""
    PENDING = "pending"
    QUEUED = "queued"
    DISPATCHING = "dispatching"
    RUNNING = "running"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RECOVERED = "recovered"
    CANCELLED = "cancelled"


class WorkerStatus(str, Enum):
    """Worker status states"""
    STARTING = "starting"  # Worker is starting up, model loading
    READY = "ready"
    DISPATCHING = "dispatching"
    BUSY = "busy"
    PROCESSING = "processing"
    IDLE = "idle"
    OFFLINE = "offline"
    ERROR = "error"


class HealthStatus(str, Enum):
    """Worker health status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class PodStatus(str, Enum):
    """Pod lifecycle status"""
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    STOPPING = "stopping"
    TERMINATED = "terminated"
    ERROR = "error"


class AuditEventType(str, Enum):
    """Audit log event types"""
    JOB_CREATED = "job_created"
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_TIMEOUT = "job_timeout"
    JOB_RECOVERED = "job_recovered"
    JOB_CANCELLED = "job_cancelled"
    POD_CREATED = "pod_created"
    POD_STARTED = "pod_started"
    POD_STOPPED = "pod_stopped"
    POD_DESTROYED = "pod_destroyed"
    POD_ERROR = "pod_error"
    WORKER_REGISTERED = "worker_registered"
    WORKER_HEARTBEAT = "worker_heartbeat"
    WORKER_OFFLINE = "worker_offline"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    IP_BANNED = "ip_banned"
    IP_UNBANNED = "ip_unbanned"
    RATE_LIMITED = "rate_limited"


# =============================================================================
# Request Models
# =============================================================================

class SubmitRequest(BaseModel):
    """
    Client job submission request.
    
    Example:
    ```json
    {
        "model": "flux_kontext_dev",
        "prompt": "A beautiful sunset over mountains",
        "params": {
            "mode": "txt2img",
            "width": 1024,
            "height": 1024,
            "cfg_scale": 3.5,
            "scale_image": true,
            "scale_factor": 2,
            "timeout": 3600
        },
        "fcm_token": "device-token-xyz",
        "enhance_prompt": true,
        "variations": 2
    }
    ```
    """
    model: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Model identifier to use for generation"
    )
    prompt: Optional[str] = Field(
        default="",
        max_length=10000,
        description="Text prompt for generation"
    )
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional parameters for the model (cfg_scale, width, height, scale_factor, etc.)"
    )
    fcm_token: Optional[str] = Field(
        default=None,
        max_length=512,
        description="Firebase Cloud Messaging token for push notifications"
    )
    enhance_prompt: bool = Field(
        default=False,
        description="If true, AI will enhance/improve the prompt before processing"
    )
    variations: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of output variations to generate (1-4, image models only)"
    )
    on_device: bool = Field(
        default=False,
        description="If true, server creates billable jobs but skips worker dispatch; "
                    "client reports started/completed lifecycle for on-device processing."
    )
    
    is_bulk: bool = Field(
        default=False,
        description="Flag indicating this is a bulk edit request"
    )
    
    # Per-image outpaint configs for bulk expand scene (each image needs placement)
    bulk_outpaint_configs: Optional[List[Optional[Dict[str, Any]]]] = Field(
        default=None,
        description="List of outpaint configurations for bulk expand scene. "
                    "Each config can contain: alignment, aspect_ratio, is_custom_mode, "
                    "custom_scale, norm_offset_x, norm_offset_y. Must align with input_image_r2_keys by index."
    )
    
    # -------------------------------------------------------------------------
    # R2 Presigned Media Upload (preferred over base64 for large payloads)
    # Frontend uploads media to R2 via /get_media_upload_urls, then sends only
    # the R2 keys here. Server converts keys to presigned GET URLs at dispatch.
    # -------------------------------------------------------------------------
    input_image_r2_keys: Optional[List[BulkInputImageEntry]] = Field(
        default=None,
        max_length=15,
        description="R2 keys per bulk task: each entry is one key (str) or a list of keys "
                    "for multi-reference jobs. Aligns with bulk_mask_r2_keys / bulk_outpaint_configs by task index."
    )
    mask_image_r2_key: Optional[str] = Field(
        default=None,
        max_length=512,
        description="R2 object key for pre-uploaded mask image (presigned upload)."
    )
    bulk_mask_r2_keys: Optional[List[Optional[str]]] = Field(
        default=None,
        description="R2 keys for pre-uploaded bulk masks. Must align with input_image_r2_keys. "
                    "Use null for images without masks."
    )
    input_video_r2_key: Optional[str] = Field(
        default=None,
        max_length=512,
        description="R2 object key for pre-uploaded input video."
    )
    
    @staticmethod
    def _normalize_bulk_media_entries(
        v: Any, field_name: str, max_refs_per_task: int = 8
    ) -> Optional[List[BulkInputImageEntry]]:
        """Ensure each bulk entry is a non-empty str or a non-empty list of non-empty str."""
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError(f"{field_name} must be a list")
        out: List[BulkInputImageEntry] = []
        for i, item in enumerate(v):
            if isinstance(item, str):
                if not item.strip():
                    raise ValueError(f"{field_name}[{i}] must be a non-empty string")
                out.append(item)
            elif isinstance(item, list):
                inner: List[str] = []
                for j, x in enumerate(item):
                    if not isinstance(x, str) or not x.strip():
                        raise ValueError(
                            f"{field_name}[{i}][{j}] must be a non-empty string"
                        )
                    inner.append(x)
                if not inner:
                    raise ValueError(f"{field_name}[{i}] must not be an empty list")
                if len(inner) > max_refs_per_task:
                    raise ValueError(
                        f"{field_name}[{i}]: at most {max_refs_per_task} images per task"
                    )
                out.append(inner)
            else:
                raise ValueError(
                    f"{field_name}[{i}] must be a string or a list of strings, not {type(item).__name__}"
                )
        return out

    @field_validator("input_image_r2_keys", mode="before")
    @classmethod
    def validate_input_image_r2_keys(cls, v: Any) -> Optional[List[BulkInputImageEntry]]:
        return cls._normalize_bulk_media_entries(v, "input_image_r2_keys")

    @field_validator("model")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name format"""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Model name can only contain alphanumeric characters, underscores, and hyphens")
        return v.lower()
    
    @field_validator("params")
    @classmethod
    def validate_params(cls, v: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate and sanitize params"""
        if v is None:
            return {}
        
        # Validate timeout if present
        if "timeout" in v:
            timeout = v["timeout"]
            if not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > 7200:
                raise ValueError("Timeout must be a positive number up to 7200 seconds")
        
        # Validate cfg_scale if present
        if "cfg_scale" in v:
            cfg_scale = v["cfg_scale"]
            if not isinstance(cfg_scale, (int, float)) or cfg_scale < 0.1 or cfg_scale > 15.0:
                raise ValueError("cfg_scale must be a number between 0.1 and 15.0")
        
        # Validate scale_factor if present
        if "scale_factor" in v:
            scale_factor = v["scale_factor"]
            if scale_factor not in [1, 2, 4]:
                raise ValueError("scale_factor must be 1, 2, or 4")
        
        # Validate width and height if present
        if "width" in v:
            width = v["width"]
            if not isinstance(width, int) or width < 32 or width > 2048:
                raise ValueError("width must be an integer between 32 and 2048")
        
        if "height" in v:
            height = v["height"]
            if not isinstance(height, int) or height < 32 or height > 2048:
                raise ValueError("height must be an integer between 32 and 2048")

        for key in _FORBIDDEN_RAW_MEDIA_PARAM_KEYS:
            if key not in v:
                continue
            val = v[key]
            if val is None or val == "":
                continue
            raise ValueError(
                f"Embedded media in params.{key} is not supported. Upload to R2 and use the "
                f"matching *_r2_key field (e.g. init_image_r2_key, mask_image_r2_key, input_video_r2_key)."
            )

        return v

    @model_validator(mode="after")
    def bulk_edit_requires_r2_keys(self) -> "SubmitRequest":
        if self.is_bulk:
            if not self.input_image_r2_keys:
                raise ValueError(
                    "Bulk jobs require non-empty input_image_r2_keys (R2 object keys after presigned upload)."
                )
        return self

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    
    @field_validator("variations")
    @classmethod
    def validate_variations(cls, v: int) -> int:
        """Validate variations count"""
        if v < 1 or v > 4:
            raise ValueError("variations must be between 1 and 4")
        return v


class RegisterWorkerRequest(BaseModel):
    """
    Worker registration request.
    
    Sent by worker pods when they start up and are ready to accept jobs.
    """
    worker_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique worker identifier"
    )
    models: List[str] = Field(
        ...,
        min_length=1,
        description="List of models this worker can handle"
    )
    endpoint: str = Field(
        default="",
        max_length=512,
        description="HTTP endpoint where worker accepts job requests"
    )
    container_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Container/pod identifier (legacy)"
    )
    pod_uuid: Optional[str] = Field(
        default=None,
        max_length=128,
        description="QuickPod pod UUID (preferred)"
    )
    public_ip: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Public IP address of the worker"
    )
    info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional worker information"
    )
    supported_models: Optional[List[str]] = Field(
        default=None,
        description="Optional runtime-reported supported models for grouped routing"
    )
    model_eta_seconds: Optional[Dict[str, int]] = Field(
        default=None,
        description="Optional runtime ETA hints per model in seconds"
    )
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        """Validate endpoint URL format"""
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("Endpoint must be a valid HTTP/HTTPS URL")
        return v
    
    @field_validator("models")
    @classmethod
    def validate_models(cls, v: List[str]) -> List[str]:
        """Normalize model names to lowercase"""
        return [m.lower().strip() for m in v if m.strip()]
    
    @model_validator(mode="after")
    def validate_pod_identifier(self):
        """Ensure at least one pod identifier is provided"""
        if not self.pod_uuid and not self.container_id:
            raise ValueError("Either pod_uuid or container_id must be provided")
        return self


class HeartbeatRequest(BaseModel):
    """
    Worker heartbeat request.
    
    Sent periodically by workers to indicate they're still alive.
    """
    worker_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Worker identifier"
    )
    status: Optional[WorkerStatus] = Field(
        default=WorkerStatus.READY,
        description="Current worker status"
    )
    metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Resource metrics (CPU, memory, GPU usage)"
    )
    current_jobs: Optional[List[str]] = Field(
        default=None,
        description="List of currently running job IDs"
    )
    supported_models: Optional[List[str]] = Field(
        default=None,
        description="Optional runtime-reported supported models refresh"
    )
    model_eta_seconds: Optional[Dict[str, int]] = Field(
        default=None,
        description="Optional runtime ETA hints refresh"
    )


class JobStartedRequest(BaseModel):
    """
    Job started notification from worker.
    
    Worker calls this when it actually begins processing a job (not just when queued).
    This allows the server to update started_at timestamp accurately.
    """
    job_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Job ID that started processing"
    )
    worker_id: Optional[str] = Field(
        None,
        description="Worker ID (optional, for validation)"
    )


class JobCompleteRequest(BaseModel):
    """
    Job completion notification from worker.
    """
    job_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Completed job ID"
    )
    result_r2_key: Optional[str] = Field(
        default=None,
        max_length=512,
        description="R2 storage key for the result file"
    )
    failed: Optional[bool] = Field(
        default=False,
        description="Whether the job failed"
    )
    error: Optional[str] = Field(
        default=None,
        max_length=4096,
        description="Error message if job failed"
    )
    continuation_latent_r2_key: Optional[str] = Field(
        default=None,
        max_length=512,
        description="R2 key for the video latent used to continue the next shot",
    )
    metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Job execution metrics (duration, memory usage, etc.)"
    )
    next_job_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="ID of the next job the worker will process (peeked from worker queue)"
    )


class BanIPRequest(BaseModel):
    """Admin request to ban an IP address"""
    ip: str = Field(
        ...,
        description="IP address to ban"
    )
    reason: str = Field(
        default="Manual ban",
        max_length=256,
        description="Reason for the ban"
    )
    duration: Optional[int] = Field(
        default=None,
        ge=0,
        le=2592000,  # 30 days max
        description="Ban duration in seconds (None for permanent)"
    )
    permanent: bool = Field(
        default=False,
        description="Whether this is a permanent ban"
    )
    
    @field_validator("ip")
    @classmethod
    def validate_ip(cls, v: str) -> str:
        """Basic IP validation"""
        v = v.strip()
        # Simple validation - more thorough validation can be added
        parts = v.split(".")
        if len(parts) == 4:
            try:
                if all(0 <= int(p) <= 255 for p in parts):
                    return v
            except ValueError:
                pass
        # Also allow IPv6
        if ":" in v:
            return v
        raise ValueError("Invalid IP address format")


class DescribeAssetForPromptRequest(BaseModel):
    """
    R2-only: client uploads bytes via presigned URL, then sends the object key here.
    v1 supports image bytes in R2 (including a single video frame exported as JPEG).
    """

    input_r2_key: str = Field(
        ...,
        min_length=8,
        max_length=512,
        description="R2 object key under the app's input-media prefix",
    )
    media_type: str = Field(
        ...,
        description="Must be 'image' for vision captioning (video: upload a frame as image).",
    )
    content_type: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional MIME hint from client (e.g. image/jpeg)",
    )

    @field_validator("media_type")
    @classmethod
    def media_type_image_only(cls, v: str) -> str:
        t = (v or "").strip().lower()
        if t != "image":
            raise ValueError(
                "media_type must be 'image'. For video, extract a frame client-side, "
                "upload it via presigned URL, and call with media_type=image."
            )
        return t


class DescribeAssetForPromptResponse(BaseModel):
    prompt: str = Field(..., description="VLM-generated prompt text for the composer")


# =============================================================================
# Response Models
# =============================================================================

class SubmitResponse(BaseModel):
    """
    Response for job submission.
    
    For single job (variations=1, no bulk):
    - job_id: The single job ID
    - job_ids: None
    - batch_id: None
    
    For multiple variations (variations>1):
    - job_id: The primary job ID (first job)
    - job_ids: List of ALL job IDs (including primary)
    - batch_id: Unique batch identifier for this group
    
    For bulk edit:
    - Multiple images with same prompt/settings
    - job_ids: List of ALL job IDs (images × variations)
    - batch_id: Unique batch identifier
    
    Frontend should poll ALL job_ids when job_ids is not None.
    """
    job_id: str = Field(..., description="Primary job identifier (first variation)")
    status: JobStatus = Field(..., description="Initial job status")
    model: str = Field(..., description="Model being used")
    routed_to_worker: Optional[str] = Field(None, description="Worker ID if immediately dispatched")
    upload_url: Optional[str] = Field(None, description="Presigned URL for result upload")
    result_key: Optional[str] = Field(None, description="R2 key where result will be stored")
    download_url: Optional[str] = Field(None, description="URL to download result when ready")
    created_at: datetime = Field(..., description="Job creation timestamp")
    
    # Variation support (for variations > 1)
    variations: int = Field(1, description="Number of variations requested")
    batch_id: Optional[str] = Field(None, description="Batch ID for multi-variation requests")
    job_ids: Optional[List[str]] = Field(None, description="List of ALL job IDs when variations > 1")
    
    # Prompt enhancement
    prompt_enhanced: bool = Field(False, description="Whether prompt was enhanced by AI")
    enhanced_prompt: Optional[str] = Field(None, description="The enhanced prompt if enhance_prompt was true")
    
    # Credits
    credits_used: int = Field(0, description="Total credits used for this request")


class VariationResult(BaseModel):
    """Result for a single variation"""
    variation_index: int = Field(..., description="0-based index of this variation")
    status: str = Field(..., description="Status of this variation job")
    result_url: Optional[str] = Field(None, description="Download URL if completed")
    error: Optional[str] = Field(None, description="Error message if failed")


class JobStatusResponse(BaseModel):
    """Response for job status query"""
    job_id: str
    status: JobStatus
    model: str
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    retry_count: int = 0
    result_url: Optional[str] = None
    continuation_latent_r2_key: Optional[str] = None
    error: Optional[str] = None
    error_log: Optional[List[Dict[str, Any]]] = None
    # Variation support - all results returned together
    variations: int = Field(1, description="Total variations requested")
    variation_results: Optional[List[VariationResult]] = Field(
        None,
        description="Results for all variations (only present if variations > 1)"
    )
    all_completed: bool = Field(True, description="True if all variations are complete")
    estimated_wait_seconds: Optional[int] = Field(None, description="Estimated seconds until job completes (processing jobs only)")
    queue_position: Optional[int] = Field(None, description="Position in worker queue (processing jobs only)")


class WorkerResponse(BaseModel):
    """Worker information response"""
    worker_id: str
    model: str
    status: WorkerStatus
    endpoint: Optional[str] = None
    pod_uuid: Optional[str] = None
    health_status: HealthStatus = HealthStatus.UNKNOWN
    last_heartbeat: Optional[datetime] = None
    last_active: Optional[datetime] = None
    is_warm_pod: bool = False
    total_jobs_processed: int = 0
    error_count: int = 0


class WorkerListResponse(BaseModel):
    """Response for worker list endpoint"""
    workers: List[WorkerResponse]
    total: int
    active: int
    idle: int


class JobListResponse(BaseModel):
    """Response for job list endpoint"""
    jobs: List[JobStatusResponse]
    total: int
    page: int = 1
    per_page: int = 50


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "ok"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: Optional[str] = None
    components: Optional[Dict[str, str]] = None


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: Dict[str, Any] = Field(
        ...,
        description="Error details including code and message"
    )


# =============================================================================
# Worker Metrics Models
# =============================================================================

class GPUMetrics(BaseModel):
    """GPU resource metrics"""
    index: int
    name: Optional[str] = None
    memory_total_mb: Optional[int] = None
    memory_used_mb: Optional[int] = None
    memory_free_mb: Optional[int] = None
    utilization_percent: Optional[float] = None
    temperature_celsius: Optional[float] = None


class ResourceMetrics(BaseModel):
    """Worker resource metrics"""
    cpu_percent: Optional[float] = Field(None, ge=0, le=100)
    memory_percent: Optional[float] = Field(None, ge=0, le=100)
    memory_used_mb: Optional[int] = None
    memory_total_mb: Optional[int] = None
    disk_percent: Optional[float] = Field(None, ge=0, le=100)
    gpus: Optional[List[GPUMetrics]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class JobMetrics(BaseModel):
    """Job execution metrics"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    peak_memory_mb: Optional[int] = None
    gpu_memory_peak_mb: Optional[int] = None
    steps_completed: Optional[int] = None
    total_steps: Optional[int] = None


# =============================================================================
# Error Log Models
# =============================================================================

class ErrorLogEntry(BaseModel):
    """Structured error log entry"""
    timestamp: datetime
    state: str
    error: str
    worker_id: Optional[str] = None
    retry_count: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


# =============================================================================
# Audit Log Models
# =============================================================================

class AuditLogEntry(BaseModel):
    """Audit log entry for database storage"""
    id: Optional[int] = None
    event_type: AuditEventType
    actor: Optional[str] = Field(None, description="Worker ID, system, or admin")
    target: Optional[str] = Field(None, description="Job ID, pod UUID, or IP")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, Any]] = None
    source_ip: Optional[str] = None


# =============================================================================
# FCM Notification Models
# =============================================================================

class FCMNotification(BaseModel):
    """Firebase Cloud Messaging notification payload"""
    title: str = Field(..., max_length=100)
    body: str = Field(..., max_length=500)
    data: Optional[Dict[str, str]] = None


class JobCompletionNotification(FCMNotification):
    """Notification for job completion"""
    title: str = "Job Complete"
    
    @classmethod
    def create(cls, job_id: str, model: str, success: bool = True) -> "JobCompletionNotification":
        if success:
            return cls(
                title="Job Complete",
                body=f"Your {model} job has finished successfully",
                data={
                    "job_id": job_id,
                    "status": "completed",
                    "model": model,
                }
            )
        else:
            return cls(
                title="Job Failed",
                body=f"Your {model} job has failed",
                data={
                    "job_id": job_id,
                    "status": "failed",
                    "model": model,
                }
            )


# =============================================================================
# Internal Models
# =============================================================================

class PodInfo(BaseModel):
    """Internal pod tracking information"""
    pod_uuid: str
    model: str
    mode: Optional[str] = None
    status: PodStatus = PodStatus.STARTING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    queue_size: int = 0
    error_count: int = 0
    machine_id: Optional[str] = None
    total_jobs_processed: int = 0
    variant: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class MachineInfo(BaseModel):
    """Machine metadata from cache"""
    machine_id: str
    gpu_name: Optional[str] = None
    gpu_count: int = 1
    tflops: float = 0
    is_priority: bool = False
    cached_images: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    last_seen: Optional[datetime] = None
    success_rate: Optional[float] = None


class JobDispatchInfo(BaseModel):
    """Information needed to dispatch a job to a worker"""
    job_id: str
    prompt: Optional[str] = None
    model: str
    mode: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    upload_url: Optional[str] = None
    result_key: Optional[str] = None


# =============================================================================
# Validation Helpers
# =============================================================================

def validate_model_name(model: str, available_models: List[str]) -> str:
    """
    Validate that a model name exists in available models.
    
    Args:
        model: Model name to validate
        available_models: List of available model names
        
    Returns:
        Normalized model name
        
    Raises:
        ValueError: If model is not available
    """
    model_lower = model.lower().strip()
    available_lower = [m.lower() for m in available_models]
    
    if model_lower not in available_lower:
        raise ValueError(
            f"Model '{model}' not found. Available models: {', '.join(available_models)}"
        )
    
    # Return the original casing from available_models
    idx = available_lower.index(model_lower)
    return available_models[idx]

