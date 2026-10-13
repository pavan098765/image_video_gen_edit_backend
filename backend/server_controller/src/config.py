# config.py
"""
Configuration Management Module

Responsibilities:
- Load and validate environment variables
- Initialize Firebase Admin SDK
- Initialize S3/R2 client  
- Define all constants (timeouts, limits, paths)
- Support multiple deployment profiles (dev, staging, prod)
"""

import os
import json
import logging
from pathlib import Path
from datetime import timedelta
from typing import Optional, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

logger = logging.getLogger("config")


class DeploymentProfile(str, Enum):
    """Deployment environment profiles"""
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class PodStrategyType(str, Enum):
    """Pod allocation strategy types"""
    WARM_AND_INDIVIDUAL = "warm_and_individual"
    MACHINE_BASED = "machine_based"


class SchedulerMode(str, Enum):
    """Job scheduling strategy modes"""
    LEGACY_DYNAMIC_FLEET = "legacy_dynamic_fleet"
    FIXED_FLEET_WINDOWED = "fixed_fleet_windowed"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = "sqlite:///./controller.db"
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 1800  # 30 minutes
    echo: bool = False


@dataclass
class R2Config:
    """Cloudflare R2 Storage configuration"""
    endpoint: Optional[str] = None
    bucket: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    token: Optional[str] = None
    presigned_url_expiry: int = 3600  # 1 hour
    max_file_size: int = 500 * 1024 * 1024  # 500MB


@dataclass
class FirebaseConfig:
    """Firebase configuration"""
    credentials_path: Optional[str] = None
    project_id: Optional[str] = None
    initialized: bool = False


@dataclass
class QuickPodConfig:
    """QuickPod API configuration"""
    email: Optional[str] = None
    password: Optional[str] = None
    warm_pod_uuid: Optional[str] = None
    base_url: str = "https://api.quickpod.org"
    token_ttl_hours: int = 6
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 2.0


@dataclass
class SecurityConfig:
    """Security configuration"""
    shared_secret: str = ""
    api_keys: Set[str] = field(default_factory=set)
    device_signing_secret: str = ""
    device_timestamp_window: int = 300  # 5 minutes
    rate_limit_per_minute: int = 40
    ban_duration: int = 300  # 5 minutes
    max_body_size: int = 10_000_000  # 10MB
    suspicious_score_threshold: int = 5
    permanent_ban_on_threshold: bool = True
    trusted_worker_ip_ttl: int = 86400  # 24 hours


@dataclass
class WorkerConfig:
    """Worker management configuration"""
    heartbeat_interval: int = 20
    heartbeat_grace_multiplier: int = 2
    registration_timeout: int = 180
    max_jobs_per_pod: int = 30
    max_same_job_per_pod: int = 5
    heartbeat_stale_seconds: int = 90
    # Pod idle timeout: destroy pod after N seconds with no work.
    # Only applies when: pod has completed all assigned jobs AND no new jobs assigned to it.
    # Pods with active jobs (RUNNING/QUEUED/DISPATCHING) are never destroyed by idle reaper.
    idle_timeout_seconds: int = 90
    # Safety: Don't destroy pods younger than this (avoids thrashing on fresh startups)
    idle_min_age_seconds: int = 60
    # Quickpod leftover window (15 min billing). External providers use 0 unless personal leftover.
    quickpod_idle_timeout_seconds: int = 900
    external_pod_idle_timeout_seconds: int = 900


@dataclass
class JobConfig:
    """Job management configuration"""
    max_retries: int = 3
    timeout_minutes: int = 30  # General timeout for PENDING/QUEUED/DISPATCHING jobs
    running_timeout_minutes: int = 10  # Hard timeout for RUNNING jobs (shorter - detect stuck workers)
    auto_recover: bool = True
    queue_size_threshold: int = 5
    queue_fairness_enabled: bool = True
    queue_fairness_max_consecutive_per_model: int = 2


@dataclass
class PodConfig:
    """Pod management configuration"""
    strategy: PodStrategyType = PodStrategyType.MACHINE_BASED
    max_individual_pods: int = 50
    startup_timeout: int = 120
    bad_machine_cooldown: int = 600  # 10 minutes
    autoscale_cooldown: int = 600  # 10 minutes
    fallback_window_seconds: int = 600
    fallback_threshold: int = 3
    load_scale_threshold: int = 15


@dataclass
class SchedulerConfig:
    """Scheduler strategy configuration"""
    mode: SchedulerMode = SchedulerMode.LEGACY_DYNAMIC_FLEET
    shadow_mode: bool = False
    dynamic_eta_enabled: bool = False
    dynamic_eta_observe_only: bool = True
    dynamic_eta_min_samples: int = 5
    dynamic_eta_alpha: float = 0.2
    dynamic_eta_min_seconds: int = 10
    dynamic_eta_max_seconds: int = 1200
    dynamic_eta_allowed_groups: Set[str] = field(default_factory=set)
    soft_governor_enabled: bool = False
    soft_governor_ops_window_seconds: int = 3600
    soft_governor_ops_threshold: int = 80
    soft_governor_extra_safety_seconds: int = 30
    soft_governor_extra_drain_guard_seconds: int = 30
    pending_model_grace_enabled: bool = True
    pending_model_max_wait_seconds: int = 20
    pending_model_poll_interval_ms: int = 500
    pending_model_hint_timeout_seconds: int = 3


@dataclass
class AppConfig:
    """Main application configuration"""
    profile: DeploymentProfile = DeploymentProfile.DEV
    debug: bool = False
    public_url: str = ""
    image_prefix: str = "image-video-generator-api"
    
    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    r2: R2Config = field(default_factory=R2Config)
    firebase: FirebaseConfig = field(default_factory=FirebaseConfig)
    quickpod: QuickPodConfig = field(default_factory=QuickPodConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    job: JobConfig = field(default_factory=JobConfig)
    pod: PodConfig = field(default_factory=PodConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Template UUIDs
    template_uuids: Dict[str, Optional[str]] = field(default_factory=dict)
    template_disk_sizes: Dict[str, int] = field(default_factory=dict)


class ConfigurationError(Exception):
    """Raised when configuration validation fails"""
    pass


class ConfigManager:
    """
    Centralized configuration manager.
    Loads environment variables, validates configuration, and provides
    access to all application settings.
    """
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[AppConfig] = None
    _s3_client = None
    _firebase_initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._config = AppConfig()
    
    @classmethod
    def get_instance(cls) -> 'ConfigManager':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @property
    def config(self) -> AppConfig:
        """Get current configuration"""
        return self._config
    
    def load(self, env_file: Optional[str] = None) -> AppConfig:
        """
        Load configuration from environment variables.
        
        Args:
            env_file: Optional path to .env file
            
        Returns:
            Loaded AppConfig instance
        """
        # Load .env file if provided
        if env_file:
            env_path = Path(env_file)
            if env_path.exists():
                load_dotenv(env_path)
                logger.info(f"Loaded environment from {env_file}")
            else:
                logger.warning(f"Environment file not found: {env_file}")
        else:
            # Try default locations
            for default_path in ["config.env", ".env", "../config.env"]:
                if Path(default_path).exists():
                    load_dotenv(default_path)
                    logger.info(f"Loaded environment from {default_path}")
                    break
        
        # Load configuration
        self._load_profile()
        self._load_database_config()
        self._load_r2_config()
        self._load_firebase_config()
        self._load_quickpod_config()
        self._load_security_config()
        self._load_worker_config()
        self._load_job_config()
        self._load_pod_config()
        self._load_scheduler_config()
        self._load_template_config()
        
        return self._config
    
    def _load_profile(self):
        """Load deployment profile"""
        profile_str = os.getenv("DEPLOYMENT_PROFILE", "dev").lower()
        try:
            self._config.profile = DeploymentProfile(profile_str)
        except ValueError:
            self._config.profile = DeploymentProfile.DEV
            logger.warning(f"Unknown profile '{profile_str}', defaulting to DEV")
        
        self._config.debug = os.getenv("DEBUG", "false").lower() == "true"
        self._config.public_url = os.getenv("CONTROLLER_PUBLIC_URL", "")
        self._config.image_prefix = os.getenv("IMAGE_PREFIX", "image-video-generator-api")
    
    def _load_database_config(self):
        """Load database configuration"""
        db_config = self._config.database
        db_config.url = os.getenv("DATABASE_URL", "sqlite:///./controller.db")
        db_config.pool_size = int(os.getenv("DB_POOL_SIZE", "5"))
        db_config.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "10"))
        db_config.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        db_config.echo = os.getenv("DB_ECHO", "false").lower() == "true"
    
    def _load_r2_config(self):
        """Load R2 storage configuration"""
        r2_config = self._config.r2
        r2_config.endpoint = os.getenv("R2_ENDPOINT")
        r2_config.bucket = os.getenv("R2_BUCKET")
        r2_config.access_key_id = os.getenv("R2_ACCESS_KEY_ID")
        r2_config.secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
        r2_config.token = os.getenv("R2_TOKEN")
        r2_config.presigned_url_expiry = int(os.getenv("R2_PRESIGNED_URL_EXPIRY", "3600"))
    
    def _load_firebase_config(self):
        """Load Firebase configuration"""
        fb_config = self._config.firebase
        fb_config.credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        fb_config.project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    
    def _load_quickpod_config(self):
        """Load QuickPod configuration"""
        qp_config = self._config.quickpod
        qp_config.email = os.getenv("QUICKPOD_EMAIL")
        qp_config.password = os.getenv("QUICKPOD_PASSWORD")
        qp_config.warm_pod_uuid = os.getenv("QUICKPOD_POD_UUID")
        qp_config.base_url = os.getenv("QUICKPOD_BASE", "https://api.quickpod.org")
        qp_config.token_ttl_hours = int(os.getenv("QUICKPOD_TOKEN_TTL_HOURS", "6"))
        qp_config.request_timeout = int(os.getenv("QUICKPOD_REQUEST_TIMEOUT", "30"))
    
    def _load_security_config(self):
        """Load security configuration"""
        sec_config = self._config.security
        sec_config.shared_secret = os.getenv("CONTROLLER_SHARED_SECRET", "")
        
        api_keys_str = os.getenv("CONTROLLER_API_KEYS", "")
        sec_config.api_keys = set(k.strip() for k in api_keys_str.split(",") if k.strip())
        
        sec_config.device_signing_secret = os.getenv(
            "DEVICE_SIGNING_SECRET", 
            os.getenv("CONTROLLER_SHARED_SECRET", "")
        )
        sec_config.device_timestamp_window = int(os.getenv("DEVICE_TIMESTAMP_WINDOW", "300"))
        sec_config.rate_limit_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", "40"))
        sec_config.ban_duration = int(os.getenv("BAN_DURATION", "300"))
        sec_config.max_body_size = int(os.getenv("MAX_BODY_SIZE", "10000000"))
    
    def _load_worker_config(self):
        """Load worker configuration"""
        worker_config = self._config.worker
        worker_config.heartbeat_interval = int(os.getenv("HEARTBEAT_INTERVAL", "20"))
        worker_config.heartbeat_grace_multiplier = int(os.getenv("HEARTBEAT_GRACE_MULTIPLIER", "2"))
        worker_config.registration_timeout = int(os.getenv("WORKER_REGISTRATION_TIMEOUT", "180"))
        worker_config.max_jobs_per_pod = int(os.getenv("MAX_JOBS_PER_POD", "30"))
        worker_config.heartbeat_stale_seconds = int(os.getenv("HEARTBEAT_STALE_SECS", "90"))
        # Pod idle timeout: destroy after N seconds with no work assigned
        worker_config.idle_timeout_seconds = int(os.getenv("WORKER_IDLE_TIMEOUT", "90"))
        worker_config.idle_min_age_seconds = int(os.getenv("WORKER_IDLE_MIN_AGE", "60"))
        worker_config.quickpod_idle_timeout_seconds = int(
            os.getenv("QUICKPOD_IDLE_TIMEOUT_SECONDS", "900")
        )
        worker_config.external_pod_idle_timeout_seconds = int(
            os.getenv("EXTERNAL_POD_IDLE_TIMEOUT_SECONDS", "900")
        )
    
    def _load_job_config(self):
        """Load job configuration"""
        job_config = self._config.job
        job_config.max_retries = int(os.getenv("JOB_MAX_RETRIES", "3"))
        job_config.timeout_minutes = int(os.getenv("JOB_TIMEOUT_MINUTES", "60"))
        job_config.running_timeout_minutes = int(os.getenv("JOB_RUNNING_TIMEOUT_MINUTES", "10"))
        job_config.auto_recover = os.getenv("JOB_AUTO_RECOVER", "true").lower() == "true"
        job_config.queue_size_threshold = int(os.getenv("QUEUE_SIZE_THRESHOLD", "5"))
        job_config.queue_fairness_enabled = (
            os.getenv("QUEUE_FAIRNESS_ENABLED", "true").lower() == "true"
        )
        job_config.queue_fairness_max_consecutive_per_model = max(
            1,
            int(os.getenv("QUEUE_FAIRNESS_MAX_CONSECUTIVE_PER_MODEL", "2")),
        )
    
    def _load_pod_config(self):
        """Load pod configuration"""
        pod_config = self._config.pod
        
        strategy_str = os.getenv("POD_STRATEGY", "machine_based").lower()
        if strategy_str in ("individual", "machine", "machine_based"):
            pod_config.strategy = PodStrategyType.MACHINE_BASED
        elif strategy_str in ("warm", "warm_and_individual", "default"):
            pod_config.strategy = PodStrategyType.WARM_AND_INDIVIDUAL
        else:
            pod_config.strategy = PodStrategyType.MACHINE_BASED
        
        pod_config.max_individual_pods = int(os.getenv("MAX_INDIVIDUAL_PODS", "50"))
        pod_config.startup_timeout = int(os.getenv("POD_STARTUP_TIMEOUT", "120"))
        pod_config.bad_machine_cooldown = int(os.getenv("BAD_MACHINE_COOLDOWN", "600"))
        pod_config.autoscale_cooldown = int(os.getenv("AUTOSCALE_COOLDOWN", "600"))
        pod_config.fallback_window_seconds = int(os.getenv("FALLBACK_WINDOW_SECONDS", "600"))
        pod_config.fallback_threshold = int(os.getenv("FALLBACK_THRESHOLD", "3"))
        pod_config.load_scale_threshold = int(os.getenv("LOAD_SCALE_THRESHOLD", "15"))

    def _load_scheduler_config(self):
        """Load scheduler strategy configuration"""
        scheduler_config = self._config.scheduler
        mode_str = os.getenv("SCHEDULER_MODE", "legacy_dynamic_fleet").lower().strip()
        if mode_str == SchedulerMode.FIXED_FLEET_WINDOWED.value:
            scheduler_config.mode = SchedulerMode.FIXED_FLEET_WINDOWED
        else:
            scheduler_config.mode = SchedulerMode.LEGACY_DYNAMIC_FLEET

        scheduler_config.shadow_mode = (
            os.getenv("SCHEDULER_SHADOW_MODE", "false").lower() == "true"
        )
        scheduler_config.dynamic_eta_enabled = (
            os.getenv("SCHEDULER_DYNAMIC_ETA_ENABLED", "false").lower() == "true"
        )
        scheduler_config.dynamic_eta_observe_only = (
            os.getenv("SCHEDULER_DYNAMIC_ETA_OBSERVE_ONLY", "true").lower() == "true"
        )
        scheduler_config.dynamic_eta_min_samples = max(
            1, int(os.getenv("SCHEDULER_DYNAMIC_ETA_MIN_SAMPLES", "5"))
        )
        try:
            alpha = float(os.getenv("SCHEDULER_DYNAMIC_ETA_ALPHA", "0.2"))
        except Exception:
            alpha = 0.2
        scheduler_config.dynamic_eta_alpha = max(0.01, min(1.0, alpha))
        scheduler_config.dynamic_eta_min_seconds = max(
            1, int(os.getenv("SCHEDULER_DYNAMIC_ETA_MIN_SECONDS", "10"))
        )
        scheduler_config.dynamic_eta_max_seconds = max(
            scheduler_config.dynamic_eta_min_seconds,
            int(os.getenv("SCHEDULER_DYNAMIC_ETA_MAX_SECONDS", "1200")),
        )
        groups_raw = os.getenv("SCHEDULER_DYNAMIC_ETA_ALLOWED_GROUPS", "")
        scheduler_config.dynamic_eta_allowed_groups = set(
            x.strip().lower() for x in groups_raw.split(",") if x.strip()
        )
        scheduler_config.soft_governor_enabled = (
            os.getenv("SCHEDULER_SOFT_GOVERNOR_ENABLED", "false").lower() == "true"
        )
        scheduler_config.soft_governor_ops_window_seconds = max(
            60, int(os.getenv("SCHEDULER_SOFT_GOVERNOR_OPS_WINDOW_SECONDS", "3600"))
        )
        scheduler_config.soft_governor_ops_threshold = max(
            1, int(os.getenv("SCHEDULER_SOFT_GOVERNOR_OPS_THRESHOLD", "80"))
        )
        scheduler_config.soft_governor_extra_safety_seconds = max(
            0, int(os.getenv("SCHEDULER_SOFT_GOVERNOR_EXTRA_SAFETY_SECONDS", "30"))
        )
        scheduler_config.soft_governor_extra_drain_guard_seconds = max(
            0, int(os.getenv("SCHEDULER_SOFT_GOVERNOR_EXTRA_DRAIN_GUARD_SECONDS", "30"))
        )
        scheduler_config.pending_model_grace_enabled = (
            os.getenv("PENDING_MODEL_GRACE_ENABLED", "true").lower() == "true"
        )
        scheduler_config.pending_model_max_wait_seconds = max(
            0, int(os.getenv("PENDING_MODEL_MAX_WAIT_SECONDS", "20"))
        )
        scheduler_config.pending_model_poll_interval_ms = max(
            100, int(os.getenv("PENDING_MODEL_POLL_INTERVAL_MS", "500"))
        )
        scheduler_config.pending_model_hint_timeout_seconds = max(
            1, int(os.getenv("PENDING_MODEL_HINT_TIMEOUT_SECONDS", "3"))
        )
    
    def _load_template_config(self):
        """Load template UUIDs and disk sizes"""
        templates = {
            "sd15": os.getenv("SD15_TEMPLATE_UUID"),
            "sd35": os.getenv("SD35_TEMPLATE_UUID"),
            "sdxl": os.getenv("SDXL_TEMPLATE_UUID"),
            "diffusers_outpaint": os.getenv("DIFFUSERS_OUTPAINT_TEMPLATE_UUID"),
            "qwen_image_edit": os.getenv("QWEN_IMAGE_EDIT_TEMPLATE_UUID"),
            "qwen_image_edit_2509": os.getenv("QWEN_IMAGE_EDIT_2509_TEMPLATE_UUID"),
            "qwen_image_edit_2511_rapid_aio": os.getenv("QWEN_IMAGE_EDIT_2511_RAPID_AIO_TEMPLATE_UUID"),
            "flux_kontext_dev": os.getenv("FLUX_KONTEXT_DEV_TEMPLATE_UUID"),
            "qwen_image": os.getenv("QWEN_IMAGE_TEMPLATE_UUID"),
            "qwen_image_edit_int4": os.getenv("QWEN_IMAGE_EDIT_INT4_TEMPLATE_UUID"),
            "qwen_image_edit_2509_int4": os.getenv("QWEN_IMAGE_EDIT_2509_INT4_TEMPLATE_UUID"),
            "flux_kontext_dev_int4": os.getenv("FLUX_KONTEXT_DEV_INT4_TEMPLATE_UUID"),
            "fastwan_21_1b": os.getenv("FASTWAN_21_1B_TEMPLATE_UUID"),
            "fastwan_22_5b": os.getenv("FASTWAN_22_5B_TEMPLATE_UUID"),
            "wan_22_14b_rapid": os.getenv("WAN_22_14B_RAPID_TEMPLATE_UUID"),
            "wan_22_uint4_lightx2v_i2v": os.getenv("WAN_22_14B_I2V_LIGHTX2V"),
            "wan_22_uint4_lightx2v_t2v": os.getenv("WAN_22_14B_T2V_LIGHTX2V"),
            "wan_22_fp8_lightx2v_i2v": os.getenv("WAN_22_14B_I2V_LIGHTX2V"),
            "wan_22_fp8_lightx2v_t2v": os.getenv("WAN_22_14B_T2V_LIGHTX2V"),
            "video_upscaler": os.getenv("VIDEO_UPSCALER_REALESRGAN_TEMPLATE_UUID"),
            "flashvsr_video_upscaler": os.getenv("FLASHVSR_VIDEO_UPSCALER_TEMPLATE_UUID"),
            "dreamid": os.getenv("DREAMID_TEMPLATE_UUID"),
            "hunyan15_480p_t2v_distilled": os.getenv("HUNYUAN_480P_T2V_TEMPLATE_UUID"),
            "hunyan15_480p_i2v_step_distilled": os.getenv("HUNYUAN_480P_I2V_TEMPLATE_UUID"),
            "ltx2_3_t2v": os.getenv("LTX2_3_DIFFUSERS_MULTI_TEMPLATE_UUID"),
            "ltx2_3_i2v_distilled": os.getenv("LTX2_3_DIFFUSERS_MULTI_TEMPLATE_UUID"),
            "ltx2_3_flf2v": os.getenv("LTX2_3_DIFFUSERS_MULTI_TEMPLATE_UUID"),
            "glm_ocr": os.getenv("GLM_OCR_TEMPLATE_UUID"),
            "minimax_h3": os.getenv("MINIMAX_H3_COMFY_TEMPLATE_UUID"),
            "ltx25_int8": os.getenv("LTX25_INT8_COMFY_TEMPLATE_UUID"),
            "ltx25_int8_t2v": os.getenv("LTX25_INT8_COMFY_TEMPLATE_UUID"),
            "ltx25_int8_i2v": os.getenv("LTX25_INT8_COMFY_TEMPLATE_UUID"),
            "ltx25_int8_flf2v": os.getenv("LTX25_INT8_COMFY_TEMPLATE_UUID"),
            "qwen_image_21_turbo": os.getenv("QWEN_IMAGE_21_TURBO_COMFY_TEMPLATE_UUID"),
            "minimax_music3": os.getenv("MINIMAX_MUSIC3_COMFY_TEMPLATE_UUID"),
            "aurasrv2": os.getenv("AURASRV2_TEMPLATE_UUID"),
            "realesrgan": os.getenv("REALESRGAN_TEMPLATE_UUID"),
            "swin2sr": os.getenv("SWIN2SR_TEMPLATE_UUID"),
            "face_enhancer": os.getenv("REALESRGAN_TEMPLATE_UUID"),  # Uses same template as realesrgan
            "basic_utils": os.getenv("BASIC_UTILS_TEMPLATE_UUID"),
            "warm_pod": os.getenv("WARM_POD_TEMPLATE_UUID"),
            "default": os.getenv("DEFAULT_TEMPLATE_UUID"),
        }
        self._config.template_uuids = templates
        
        disk_sizes = {
            "sd15": 30,
            "sd35": 40,
            "sdxl": 30,
            "diffusers_outpaint": 50,
            "qwen_image": 80,
            "qwen_image_edit": 80,
            "qwen_image_edit_2509": 85,
            "qwen_image_edit_2511_rapid_aio": 90,
            "flux_kontext_dev": 45,
            "qwen_image_int4": 80,
            "qwen_image_edit_int4": 80,
            "qwen_image_edit_2509_int4": 85,
            "flux_kontext_dev_int4": 45,
            "fastwan_21_1b": 60,
            "fastwan_22_5b": 60,
            "wan_22_14b_rapid": 100,
            "wan_22_uint4_lightx2v_i2v": 60,
            "wan_22_uint4_lightx2v_t2v": 60,
            "wan_22_fp8_lightx2v_i2v": 80,
            "wan_22_fp8_lightx2v_t2v": 80,
            "video_upscaler": 40,  # RIFE + Real-ESRGAN models ~10GB + workspace
            "flashvsr_video_upscaler": 50,  # FlashVSR-Pro models ~7GB + workspace
            "dreamid": 70,
            "hunyan15_480p_t2v_distilled": 100,
            "hunyan15_480p_i2v_step_distilled": 100,
            "ltx2_3_t2v": 200,
            "ltx2_3_i2v_distilled": 200,
            "ltx2_3_flf2v": 200,
            "glm_ocr": 30,
            "minimax_h3": 250,
            "ltx25_int8": 120,
            "ltx25_int8_t2v": 120,
            "ltx25_int8_i2v": 120,
            "ltx25_int8_flf2v": 120,
            "qwen_image_21_turbo": 80,
            "minimax_music3": 80,
            "aurasrv2": 30,
            "realesrgan": 30,
            "swin2sr": 30,
            "face_enhancer": 30,
            "basic_utils": 30,
        }
        self._config.template_disk_sizes = disk_sizes
    
    def validate(self, strict: bool = False) -> bool:
        """
        Validate configuration.
        
        Args:
            strict: If True, raise ConfigurationError on validation failure
            
        Returns:
            True if validation passes
        """
        errors = []
        warnings = []
        
        # Security validation
        if not self._config.security.shared_secret:
            errors.append("CONTROLLER_SHARED_SECRET is required")
        
        if not self._config.security.api_keys:
            warnings.append("No API keys configured (CONTROLLER_API_KEYS)")
        
        # QuickPod validation
        if not self._config.quickpod.email or not self._config.quickpod.password:
            errors.append("QUICKPOD_EMAIL and QUICKPOD_PASSWORD are required")
        
        # Firebase validation (warning only)
        if not self._config.firebase.credentials_path:
            warnings.append("GOOGLE_APPLICATION_CREDENTIALS not set - FCM notifications disabled")
        elif not Path(self._config.firebase.credentials_path).exists():
            errors.append(f"Firebase credentials file not found: {self._config.firebase.credentials_path}")
        
        # R2 validation (warning only)
        r2 = self._config.r2
        if not all([r2.endpoint, r2.access_key_id, r2.secret_access_key]):
            warnings.append("R2 storage not fully configured - result upload disabled")
        
        # Log findings
        for warning in warnings:
            logger.warning(f"Config warning: {warning}")
        
        for error in errors:
            logger.error(f"Config error: {error}")
        
        if errors and strict:
            raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return len(errors) == 0
    
    def log_config(self):
        """Log configuration (non-sensitive parts only)"""
        logger.info("=" * 60)
        logger.info("APPLICATION CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Profile: {self._config.profile.value}")
        logger.info(f"Debug: {self._config.debug}")
        logger.info(f"Public URL: {self._config.public_url or '(not set)'}")
        logger.info(f"Pod Strategy: {self._config.pod.strategy.value}")
        logger.info(f"Scheduler Mode: {self._config.scheduler.mode.value}")
        logger.info(f"Scheduler Shadow Mode: {self._config.scheduler.shadow_mode}")
        logger.info(
            f"Dynamic ETA: enabled={self._config.scheduler.dynamic_eta_enabled}, "
            f"observe_only={self._config.scheduler.dynamic_eta_observe_only}, "
            f"min_samples={self._config.scheduler.dynamic_eta_min_samples}, "
            f"alpha={self._config.scheduler.dynamic_eta_alpha:.2f}"
        )
        logger.info(
            f"Soft Governor: enabled={self._config.scheduler.soft_governor_enabled}, "
            f"window={self._config.scheduler.soft_governor_ops_window_seconds}s, "
            f"threshold={self._config.scheduler.soft_governor_ops_threshold}, "
            f"extra_safety={self._config.scheduler.soft_governor_extra_safety_seconds}s, "
            f"extra_drain={self._config.scheduler.soft_governor_extra_drain_guard_seconds}s"
        )
        logger.info(
            f"Pending Model Grace: enabled={self._config.scheduler.pending_model_grace_enabled}, "
            f"max_wait={self._config.scheduler.pending_model_max_wait_seconds}s, "
            f"poll={self._config.scheduler.pending_model_poll_interval_ms}ms, "
            f"hint_timeout={self._config.scheduler.pending_model_hint_timeout_seconds}s"
        )
        logger.info(f"Max Individual Pods: {self._config.pod.max_individual_pods}")
        logger.info(f"Worker Registration Timeout: {self._config.worker.registration_timeout}s")
        logger.info(
            f"Pod Idle Timeout: {self._config.worker.idle_timeout_seconds}s "
            f"(destroy when idle with no jobs assigned)"
        )
        logger.info(f"Job Timeout: {self._config.job.timeout_minutes} minutes")
        logger.info(f"Job Max Retries: {self._config.job.max_retries}")
        logger.info(
            f"Queue Fairness: enabled={self._config.job.queue_fairness_enabled}, "
            f"max_consecutive={self._config.job.queue_fairness_max_consecutive_per_model}"
        )
        logger.info(f"R2 Configured: {bool(self._config.r2.endpoint)}")
        logger.info(f"Firebase Configured: {bool(self._config.firebase.credentials_path)}")
        logger.info(f"API Keys Count: {len(self._config.security.api_keys)}")
        logger.info("=" * 60)
    
    def get_s3_client(self):
        """
        Get or create S3/R2 client.
        
        Returns:
            boto3 S3 client or None if not configured
        """
        if self._s3_client is not None:
            return self._s3_client
        
        r2 = self._config.r2
        if not all([r2.endpoint, r2.access_key_id, r2.secret_access_key]):
            return None
        
        try:
            self._s3_client = boto3.client(
                "s3",
                endpoint_url=r2.endpoint,
                aws_access_key_id=r2.access_key_id,
                aws_secret_access_key=r2.secret_access_key,
                config=Config(signature_version="s3v4"),
            )
            logger.info("S3/R2 client initialized successfully")
            return self._s3_client
        except Exception as e:
            logger.error(f"Failed to initialize S3/R2 client: {e}")
            return None
    
    def check_r2_connectivity(self) -> bool:
        """
        Health check for R2 connectivity.
        
        Returns:
            True if R2 is accessible
        """
        client = self.get_s3_client()
        if not client:
            return False
        
        try:
            client.head_bucket(Bucket=self._config.r2.bucket)
            return True
        except ClientError as e:
            logger.error(f"R2 connectivity check failed: {e}")
            return False
    
    def initialize_firebase(self) -> bool:
        """
        Initialize Firebase Admin SDK.
        
        Returns:
            True if initialization succeeded
        """
        if self._firebase_initialized:
            return True
        
        fb_config = self._config.firebase
        if not fb_config.credentials_path:
            logger.warning("Firebase credentials not configured")
            return False
        
        try:
            import firebase_admin
            from firebase_admin import credentials
            
            sa_path = fb_config.credentials_path
            if not os.path.exists(sa_path):
                raise FileNotFoundError(f"Service account file not found: {sa_path}")
            
            with open(sa_path, "r", encoding="utf-8") as f:
                sa_json = json.load(f)
            
            project_id = (
                sa_json.get("project_id") or 
                fb_config.project_id or
                os.environ.get("GOOGLE_CLOUD_PROJECT")
            )
            
            if not project_id:
                raise ConfigurationError(
                    "No project_id found in service account JSON and GOOGLE_CLOUD_PROJECT not set"
                )
            
            cred = credentials.Certificate(sa_json)
            firebase_admin.initialize_app(cred, {"projectId": project_id})
            
            self._firebase_initialized = True
            self._config.firebase.initialized = True
            self._config.firebase.project_id = project_id
            
            logger.info(f"Firebase Admin initialized with projectId={project_id}")
            return True
            
        except Exception as e:
            logger.exception(f"Firebase Admin init failed: {e}")
            return False


# Global configuration instance
config_manager = ConfigManager.get_instance()


def get_config() -> AppConfig:
    """Get current application configuration"""
    return config_manager.config


def load_config(env_file: Optional[str] = None, strict: bool = False) -> AppConfig:
    """
    Load and validate configuration.
    
    Args:
        env_file: Optional path to .env file
        strict: If True, raise on validation errors
        
    Returns:
        Loaded AppConfig instance
    """
    config = config_manager.load(env_file)
    config_manager.validate(strict=strict)
    return config

