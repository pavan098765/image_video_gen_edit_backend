# exceptions.py
"""
Custom Exception Hierarchy

Provides consistent error handling and responses across the application.
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorCode(str, Enum):
    """Standard error codes for API responses"""
    
    # General errors (1xxx)
    INTERNAL_ERROR = "E1000"
    VALIDATION_ERROR = "E1001"
    NOT_FOUND = "E1002"
    CONFLICT = "E1003"
    RATE_LIMITED = "E1004"
    
    # Authentication/Authorization errors (2xxx)
    UNAUTHORIZED = "E2000"
    FORBIDDEN = "E2001"
    INVALID_API_KEY = "E2002"
    INVALID_SIGNATURE = "E2003"
    INVALID_DEVICE = "E2004"
    IP_BANNED = "E2005"
    
    # Job errors (3xxx)
    JOB_NOT_FOUND = "E3000"
    JOB_STATE_ERROR = "E3001"
    JOB_TIMEOUT = "E3002"
    JOB_DISPATCH_FAILED = "E3003"
    JOB_MAX_RETRIES = "E3004"
    INVALID_MODEL = "E3005"
    
    # Worker errors (4xxx)
    WORKER_NOT_FOUND = "E4000"
    WORKER_REGISTRATION_FAILED = "E4001"
    WORKER_TIMEOUT = "E4002"
    WORKER_OFFLINE = "E4003"
    
    # Pod errors (5xxx)
    POD_CREATION_FAILED = "E5000"
    POD_NOT_FOUND = "E5001"
    POD_STARTUP_TIMEOUT = "E5002"
    POD_DESTROY_FAILED = "E5003"
    NO_AVAILABLE_MACHINES = "E5004"
    MACHINE_BLACKLISTED = "E5005"
    
    # External service errors (6xxx)
    QUICKPOD_API_ERROR = "E6000"
    QUICKPOD_FATAL_ERROR = "E6001"
    FIREBASE_ERROR = "E6002"
    R2_ERROR = "E6003"
    
    # Storage errors (7xxx)
    FILE_NOT_FOUND = "E7000"
    UPLOAD_FAILED = "E7001"
    DOWNLOAD_FAILED = "E7002"


class BaseError(Exception):
    """
    Base exception class for all custom exceptions.
    
    Provides:
    - Error code for client identification
    - HTTP status code mapping
    - Detailed error messages
    - Optional context data
    """
    
    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    status_code: int = 500
    message: str = "An unexpected error occurred"
    
    def __init__(
        self, 
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.message = message or self.message
        self.details = details or {}
        self.cause = cause
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format"""
        result = {
            "error": {
                "code": self.code.value,
                "message": self.message,
            }
        }
        if self.details:
            result["error"]["details"] = self.details
        return result
    
    def __str__(self) -> str:
        base = f"[{self.code.value}] {self.message}"
        if self.cause:
            base += f" (caused by: {self.cause})"
        return base


# =============================================================================
# Authentication/Authorization Errors
# =============================================================================

class AuthenticationError(BaseError):
    """Base class for authentication errors"""
    code = ErrorCode.UNAUTHORIZED
    status_code = 401
    message = "Authentication required"


class InvalidAPIKeyError(AuthenticationError):
    """Invalid or missing API key"""
    code = ErrorCode.INVALID_API_KEY
    message = "Invalid or missing API key"


class InvalidSignatureError(AuthenticationError):
    """Invalid HMAC signature"""
    code = ErrorCode.INVALID_SIGNATURE
    message = "Invalid request signature"


class InvalidDeviceError(AuthenticationError):
    """Invalid device fingerprint or signature"""
    code = ErrorCode.INVALID_DEVICE
    message = "Invalid device authentication"


class ForbiddenError(BaseError):
    """Access forbidden"""
    code = ErrorCode.FORBIDDEN
    status_code = 403
    message = "Access forbidden"


class IPBannedError(ForbiddenError):
    """IP address is banned"""
    code = ErrorCode.IP_BANNED
    message = "Your IP address has been banned"
    
    def __init__(
        self, 
        ip: str,
        reason: Optional[str] = None,
        permanent: bool = False,
        **kwargs
    ):
        details = {"ip": ip, "permanent": permanent}
        if reason:
            details["reason"] = reason
        super().__init__(details=details, **kwargs)


class RateLimitError(BaseError):
    """Rate limit exceeded"""
    code = ErrorCode.RATE_LIMITED
    status_code = 429
    message = "Rate limit exceeded"
    
    def __init__(self, retry_after: Optional[int] = None, **kwargs):
        details = {}
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(details=details, **kwargs)


# =============================================================================
# Job Errors
# =============================================================================

class JobError(BaseError):
    """Base class for job-related errors"""
    status_code = 400


class JobNotFoundError(JobError):
    """Job not found"""
    code = ErrorCode.JOB_NOT_FOUND
    status_code = 404
    message = "Job not found"
    
    def __init__(self, job_id: str, **kwargs):
        super().__init__(message=f"Job not found: {job_id}", details={"job_id": job_id}, **kwargs)


class JobStateError(JobError):
    """Invalid job state transition"""
    code = ErrorCode.JOB_STATE_ERROR
    message = "Invalid job state transition"
    
    def __init__(
        self, 
        job_id: str,
        current_state: str,
        target_state: str,
        **kwargs
    ):
        details = {
            "job_id": job_id,
            "current_state": current_state,
            "target_state": target_state,
        }
        message = f"Cannot transition job {job_id} from {current_state} to {target_state}"
        super().__init__(message=message, details=details, **kwargs)


class JobTimeoutError(JobError):
    """Job execution timed out"""
    code = ErrorCode.JOB_TIMEOUT
    status_code = 408
    message = "Job execution timed out"
    
    def __init__(self, job_id: str, timeout_minutes: int = 30, **kwargs):
        details = {"job_id": job_id, "timeout_minutes": timeout_minutes}
        message = f"Job {job_id} timed out after {timeout_minutes} minutes"
        super().__init__(message=message, details=details, **kwargs)


class JobDispatchError(JobError):
    """Failed to dispatch job to worker"""
    code = ErrorCode.JOB_DISPATCH_FAILED
    message = "Failed to dispatch job"
    
    def __init__(
        self, 
        job_id: str,
        reason: Optional[str] = None,
        worker_id: Optional[str] = None,
        **kwargs
    ):
        details = {"job_id": job_id}
        if worker_id:
            details["worker_id"] = worker_id
        if reason:
            details["reason"] = reason
        message = f"Failed to dispatch job {job_id}"
        if reason:
            message += f": {reason}"
        super().__init__(message=message, details=details, **kwargs)


class JobMaxRetriesError(JobError):
    """Job exceeded maximum retry attempts"""
    code = ErrorCode.JOB_MAX_RETRIES
    message = "Job exceeded maximum retries"
    
    def __init__(self, job_id: str, max_retries: int = 3, **kwargs):
        details = {"job_id": job_id, "max_retries": max_retries}
        message = f"Job {job_id} exceeded maximum retry attempts ({max_retries})"
        super().__init__(message=message, details=details, **kwargs)


class InvalidModelError(JobError):
    """Invalid or unsupported model"""
    code = ErrorCode.INVALID_MODEL
    message = "Invalid or unsupported model"
    
    def __init__(self, model: str, available_models: Optional[list] = None, **kwargs):
        details = {"model": model}
        if available_models:
            details["available_models"] = available_models
        message = f"Invalid or unsupported model: {model}"
        super().__init__(message=message, details=details, **kwargs)


# =============================================================================
# Worker Errors
# =============================================================================

class WorkerError(BaseError):
    """Base class for worker-related errors"""
    status_code = 500


class WorkerNotFoundError(WorkerError):
    """Worker not found"""
    code = ErrorCode.WORKER_NOT_FOUND
    status_code = 404
    message = "Worker not found"
    
    def __init__(self, worker_id: str, **kwargs):
        super().__init__(
            message=f"Worker not found: {worker_id}", 
            details={"worker_id": worker_id}, 
            **kwargs
        )


class WorkerRegistrationError(WorkerError):
    """Worker registration failed"""
    code = ErrorCode.WORKER_REGISTRATION_FAILED
    message = "Worker registration failed"
    
    def __init__(
        self, 
        worker_id: str,
        reason: Optional[str] = None,
        pod_uuid: Optional[str] = None,
        **kwargs
    ):
        details = {"worker_id": worker_id}
        if pod_uuid:
            details["pod_uuid"] = pod_uuid
        if reason:
            details["reason"] = reason
        message = f"Worker registration failed for {worker_id}"
        if reason:
            message += f": {reason}"
        super().__init__(message=message, details=details, **kwargs)


class WorkerTimeoutError(WorkerError):
    """Worker registration timeout"""
    code = ErrorCode.WORKER_TIMEOUT
    status_code = 408
    message = "Worker registration timed out"
    
    def __init__(self, pod_uuid: str, timeout: int = 180, **kwargs):
        details = {"pod_uuid": pod_uuid, "timeout_seconds": timeout}
        message = f"Worker registration timed out for pod {pod_uuid}"
        super().__init__(message=message, details=details, **kwargs)


class WorkerOfflineError(WorkerError):
    """Worker is offline"""
    code = ErrorCode.WORKER_OFFLINE
    message = "Worker is offline"
    
    def __init__(self, worker_id: str, **kwargs):
        super().__init__(
            message=f"Worker is offline: {worker_id}",
            details={"worker_id": worker_id},
            **kwargs
        )


# =============================================================================
# Pod Errors
# =============================================================================

class PodError(BaseError):
    """Base class for pod-related errors"""
    status_code = 500


class PodCreationError(PodError):
    """Pod creation failed"""
    code = ErrorCode.POD_CREATION_FAILED
    message = "Failed to create pod"
    
    def __init__(
        self, 
        model: str,
        reason: Optional[str] = None,
        machine_id: Optional[str] = None,
        **kwargs
    ):
        details = {"model": model}
        if machine_id:
            details["machine_id"] = machine_id
        if reason:
            details["reason"] = reason
        message = f"Failed to create pod for model {model}"
        if reason:
            message += f": {reason}"
        super().__init__(message=message, details=details, **kwargs)


class PodNotFoundError(PodError):
    """Pod not found"""
    code = ErrorCode.POD_NOT_FOUND
    status_code = 404
    message = "Pod not found"
    
    def __init__(self, pod_uuid: str, **kwargs):
        super().__init__(
            message=f"Pod not found: {pod_uuid}",
            details={"pod_uuid": pod_uuid},
            **kwargs
        )


class PodStartupTimeoutError(PodError):
    """Pod startup timed out"""
    code = ErrorCode.POD_STARTUP_TIMEOUT
    status_code = 408
    message = "Pod startup timed out"
    
    def __init__(self, pod_uuid: str, timeout: int = 120, **kwargs):
        details = {"pod_uuid": pod_uuid, "timeout_seconds": timeout}
        message = f"Pod {pod_uuid} startup timed out after {timeout}s"
        super().__init__(message=message, details=details, **kwargs)


class PodDestroyError(PodError):
    """Pod destruction failed"""
    code = ErrorCode.POD_DESTROY_FAILED
    message = "Failed to destroy pod"
    
    def __init__(self, pod_uuid: str, reason: Optional[str] = None, **kwargs):
        details = {"pod_uuid": pod_uuid}
        if reason:
            details["reason"] = reason
        message = f"Failed to destroy pod {pod_uuid}"
        if reason:
            message += f": {reason}"
        super().__init__(message=message, details=details, **kwargs)


class NoAvailableMachinesError(PodError):
    """No machines available for pod creation"""
    code = ErrorCode.NO_AVAILABLE_MACHINES
    message = "No machines available"
    
    def __init__(self, model: str, **kwargs):
        details = {"model": model}
        message = f"No machines available for model {model}"
        super().__init__(message=message, details=details, **kwargs)


class MachineBlacklistedError(PodError):
    """Machine is temporarily blacklisted"""
    code = ErrorCode.MACHINE_BLACKLISTED
    message = "Machine is temporarily blacklisted"
    
    def __init__(self, machine_id: str, cooldown_remaining: Optional[int] = None, **kwargs):
        details = {"machine_id": machine_id}
        if cooldown_remaining:
            details["cooldown_remaining_seconds"] = cooldown_remaining
        message = f"Machine {machine_id} is temporarily blacklisted"
        super().__init__(message=message, details=details, **kwargs)


# =============================================================================
# External Service Errors
# =============================================================================

class ExternalServiceError(BaseError):
    """Base class for external service errors"""
    status_code = 502  # Bad Gateway


class QuickPodAPIError(ExternalServiceError):
    """QuickPod API error (transient)"""
    code = ErrorCode.QUICKPOD_API_ERROR
    message = "QuickPod API error"
    
    def __init__(self, operation: str, reason: Optional[str] = None, **kwargs):
        details = {"operation": operation}
        if reason:
            details["reason"] = reason
        message = f"QuickPod API error during {operation}"
        if reason:
            message += f": {reason}"
        super().__init__(message=message, details=details, **kwargs)


class QuickPodFatalError(ExternalServiceError):
    """QuickPod API fatal error (non-recoverable)"""
    code = ErrorCode.QUICKPOD_FATAL_ERROR
    message = "QuickPod API fatal error"
    
    def __init__(self, operation: str, reason: Optional[str] = None, **kwargs):
        details = {"operation": operation, "recoverable": False}
        if reason:
            details["reason"] = reason
        message = f"QuickPod API fatal error during {operation}"
        if reason:
            message += f": {reason}"
        super().__init__(message=message, details=details, **kwargs)


class FirebaseError(ExternalServiceError):
    """Firebase service error"""
    code = ErrorCode.FIREBASE_ERROR
    message = "Firebase error"
    
    def __init__(self, operation: str, reason: Optional[str] = None, **kwargs):
        details = {"operation": operation}
        if reason:
            details["reason"] = reason
        message = f"Firebase error during {operation}"
        if reason:
            message += f": {reason}"
        super().__init__(message=message, details=details, **kwargs)


class R2StorageError(ExternalServiceError):
    """R2 storage error"""
    code = ErrorCode.R2_ERROR
    message = "R2 storage error"
    
    def __init__(self, operation: str, key: Optional[str] = None, reason: Optional[str] = None, **kwargs):
        details = {"operation": operation}
        if key:
            details["key"] = key
        if reason:
            details["reason"] = reason
        message = f"R2 storage error during {operation}"
        if reason:
            message += f": {reason}"
        super().__init__(message=message, details=details, **kwargs)


# =============================================================================
# Storage Errors
# =============================================================================

class StorageError(BaseError):
    """Base class for storage errors"""
    status_code = 500


class FileNotFoundError(StorageError):
    """File not found in storage"""
    code = ErrorCode.FILE_NOT_FOUND
    status_code = 404
    message = "File not found"
    
    def __init__(self, key: str, **kwargs):
        super().__init__(
            message=f"File not found: {key}",
            details={"key": key},
            **kwargs
        )


class UploadFailedError(StorageError):
    """File upload failed"""
    code = ErrorCode.UPLOAD_FAILED
    message = "File upload failed"
    
    def __init__(self, key: str, reason: Optional[str] = None, **kwargs):
        details = {"key": key}
        if reason:
            details["reason"] = reason
        message = f"Failed to upload file: {key}"
        if reason:
            message += f": {reason}"
        super().__init__(message=message, details=details, **kwargs)


class DownloadFailedError(StorageError):
    """File download failed"""
    code = ErrorCode.DOWNLOAD_FAILED
    message = "File download failed"
    
    def __init__(self, key: str, reason: Optional[str] = None, **kwargs):
        details = {"key": key}
        if reason:
            details["reason"] = reason
        message = f"Failed to download file: {key}"
        if reason:
            message += f": {reason}"
        super().__init__(message=message, details=details, **kwargs)


# =============================================================================
# Validation Errors
# =============================================================================

class ValidationError(BaseError):
    """Request validation error"""
    code = ErrorCode.VALIDATION_ERROR
    status_code = 422
    message = "Validation error"
    
    def __init__(self, errors: list, **kwargs):
        details = {"validation_errors": errors}
        super().__init__(
            message="Request validation failed",
            details=details,
            **kwargs
        )

