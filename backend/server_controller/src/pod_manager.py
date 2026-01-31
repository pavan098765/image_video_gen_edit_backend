# pod_manager.py
"""
Pod & Resource Management (Major Refactor)

Provides:
- Pod Lifecycle Management (create, monitor, destroy)
- Strategy Implementations (warm_and_individual, machine_based)
- Machine cache management
- Pod selection & allocation with variant fallback
- Pending job queue management
- Autoscaling based on load and fallback frequency
"""

import os
import time
import json
import asyncio
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List, Tuple, Set
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from quickpod_api import QuickPodManager, QuickPodErrorType
from exceptions import (
    PodCreationError,
    PodStartupTimeoutError,
    PodDestroyError,
    PodNotFoundError,
    NoAvailableMachinesError,
    MachineBlacklistedError,
    WorkerTimeoutError,
    QuickPodFatalError,
    QuickPodAPIError,
    JobAlreadyProcessedError,
)

logger = logging.getLogger("pod_manager")


def normalize_pod_uuid(pod_uuid: Optional[str]) -> Optional[str]:
    """Normalize pod UUID by removing hyphens for comparison."""
    if not pod_uuid:
        return None
    return pod_uuid.replace("-", "").lower()


def format_pod_uuid_with_hyphens(pod_uuid: Optional[str]) -> Optional[str]:
    """Format a pod UUID with hyphens (8-4-4-4-12 format) for QuickPod API calls."""
    if not pod_uuid:
        return None
    
    clean = pod_uuid.replace("-", "").lower()
    
    if len(pod_uuid) == 36 and "-" in pod_uuid:
        return pod_uuid.lower()
    
    if len(clean) == 32:
        try:
            int(clean, 16)
            return f"{clean[:8]}-{clean[8:12]}-{clean[12:16]}-{clean[16:20]}-{clean[20:]}"
        except ValueError:
            pass
    
    return pod_uuid


# =============================================================================
# Configuration & Types
# =============================================================================

class PodStrategyType(str, Enum):
    """Pod allocation strategy types"""
    WARM_AND_INDIVIDUAL = "warm_and_individual"
    MACHINE_BASED = "machine_based"


class PodStatus(str, Enum):
    """Pod lifecycle status"""
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    STOPPING = "stopping"
    TERMINATED = "terminated"
    ERROR = "error"


# Machine priority list (machine_id, description)
# MACHINE_PRIORITY_LIST removed - now relying solely on machine_cache.json
# which is dynamically updated based on successful pod creations and model caching.
# This prevents conflicts with the machine blacklist system.

MAX_JOBS_PER_POD = int(os.getenv("MAX_JOBS_PER_POD", "5"))
MAX_SAME_JOB_PER_POD = int(os.getenv("MAX_SAME_JOB_PER_POD", "5"))


@dataclass
class PodInfo:
    """Internal pod tracking information"""
    # Required fields (no defaults) - must come first
    pod_uuid: str
    model: str
    mode: Optional[str]
    status: PodStatus
    created_at: datetime
    last_active: datetime
    # Optional fields with defaults
    queue_size: int = 0
    error_count: int = 0
    machine_id: Optional[str] = None
    total_jobs_processed: int = 0
    unique_job_ids: Set[str] = field(default_factory=set)
    job_id_counts: Dict[str, int] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    job_start_times: Dict[str, datetime] = field(default_factory=dict)
    is_warm_pod: bool = False
    variant: Optional[str] = None
    # Cost tracking fields
    hourly_cost: float = 0.0  # $/hour from offer
    gpu_type: str = ""  # e.g., "NVIDIA GeForce RTX 4090"
    offer_id: int = 0


# =============================================================================
# Pod Lifecycle Manager
# =============================================================================

class PodLifecycleManager:
    """
    Manages individual pod lifecycle operations.
    
    Responsibilities:
    - Create pod with proper error handling
    - Wait for pod readiness
    - Destroy pod with graceful shutdown
    - Track recently destroyed pods
    """
    
    def __init__(self, qp_manager: QuickPodManager):
        self.qp_manager = qp_manager
        self._recently_destroyed: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    async def create_pod(
        self,
        offer_id: int,
        template_uuid: str,
        disk_size: int,
        model: str,
        mode: Optional[str] = None,
        variant: Optional[str] = None,
        machine_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create a new pod.
        
        Args:
            offer_id: QuickPod offer ID
            template_uuid: Template to use
            disk_size: Disk size in GB
            model: Model name
            mode: Optional mode
            variant: Optional variant (fp4, int4)
            machine_id: Machine ID for tracking
            
        Returns:
            pod_uuid on success, None on failure
        """
        logger.info(
            f"Creating pod: model={model}, variant={variant}, "
            f"template={template_uuid}, offer={offer_id}"
        )
        
        try:
            # Create pod via QuickPod API
            pod_data = await asyncio.to_thread(
                self.qp_manager.create_pod,
                offer_id,
                template_uuid,
                disk_size,
            )
            
            pod_uuid = (
                pod_data.get("pod_uuid") or 
                pod_data.get("uuid") or 
                pod_data.get("id")
            )
            
            if not pod_uuid:
                logger.error(f"No pod_uuid in response: {pod_data}")
                return None
            
            # CRITICAL: Cache the public IP for worker registration matching
            # Workers register with docker container ID (CONTAINER_LABEL), not pod UUID.
            # We need to match them by IP to get the correct pod UUID for destroy operations.
            public_ip = pod_data.get("public_ipaddress")
            if public_ip:
                if not hasattr(self, '_pod_public_ips'):
                    self._pod_public_ips = {}
                self._pod_public_ips[public_ip] = pod_uuid
                logger.info(f"Pod created: {pod_uuid} on machine {machine_id}, cached IP {public_ip}")
            else:
                logger.info(f"Pod created: {pod_uuid} on machine {machine_id}")
            
            return pod_uuid
            
        except QuickPodFatalError as e:
            logger.error(f"Fatal error creating pod: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating pod: {e}")
            raise PodCreationError(model, reason=str(e), cause=e)
    
    async def wait_for_ready(
        self,
        pod_uuid: str,
        timeout: int = 120,
        check_interval: float = 2.0,
        health_check_url: Optional[str] = None,
    ) -> bool:
        """
        Wait for pod to become ready.
        
        Args:
            pod_uuid: Pod UUID to check
            timeout: Maximum wait time in seconds
            check_interval: Time between checks
            health_check_url: Optional URL to check for readiness
            
        Returns:
            True if pod became ready, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if health_check_url:
                    # Check via HTTP health endpoint
                    resp = requests.get(health_check_url, timeout=5)
                    if resp.status_code == 200:
                        logger.info(f"Pod {pod_uuid} is ready (health check passed)")
                        return True
                else:
                    # Check via QuickPod API
                    # Note: This would need implementation in QuickPod API
                    # For now, we rely on worker registration
                    pass
                    
            except Exception as e:
                logger.debug(f"Pod {pod_uuid} not ready yet: {e}")
            
            await asyncio.sleep(check_interval)
        
        logger.warning(f"Pod {pod_uuid} did not become ready within {timeout}s")
        return False
    
    async def destroy_pod(
        self,
        pod_uuid: str,
        graceful: bool = False,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Destroy a pod.
        
        Args:
            pod_uuid: Pod UUID to destroy
            graceful: Whether to attempt graceful shutdown
            reason: Reason for destruction (for logging)
            
        Returns:
            True if destroyed successfully
        """
        # CRITICAL: Ensure UUID is in hyphenated format for QuickPod API
        original_uuid = pod_uuid
        pod_uuid = format_pod_uuid_with_hyphens(pod_uuid) or pod_uuid
        
        if pod_uuid != original_uuid:
            logger.debug(f"Formatted pod_uuid for API: {original_uuid} -> {pod_uuid}")
        
        pod_uuid_normalized = normalize_pod_uuid(pod_uuid)
        
        async with self._lock:
            # Check if recently destroyed (check both formats)
            if pod_uuid in self._recently_destroyed:
                logger.debug(f"Pod {pod_uuid} recently destroyed, skipping")
                return True
            # Also check normalized
            for destroyed_uuid in self._recently_destroyed.keys():
                if normalize_pod_uuid(destroyed_uuid) == pod_uuid_normalized:
                    logger.debug(f"Pod {pod_uuid} recently destroyed (normalized match), skipping")
                    return True
        
        logger.info(f"Destroying pod {pod_uuid}: {reason or 'no reason given'}")
        
        try:
            # Create manager for this pod (must use hyphenated format)
            pod_manager = QuickPodManager(
                self.qp_manager.email,
                self.qp_manager.password,
                pod_uuid,
            )
            
            if graceful:
                # Try to stop first
                try:
                    await asyncio.to_thread(pod_manager.stop_pod)
                    await asyncio.sleep(2)  # Brief wait
                except Exception as e:
                    logger.debug(f"Stop pod failed (continuing to destroy): {e}")
            
            # Destroy pod
            await asyncio.to_thread(pod_manager.destroy_pod)
            
            # Mark as recently destroyed
            async with self._lock:
                self._recently_destroyed[pod_uuid] = time.time()
            
            logger.info(f"Pod {pod_uuid} destroyed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to destroy pod {pod_uuid}: {e}")
            # Still mark as recently destroyed to prevent repeated attempts
            async with self._lock:
                self._recently_destroyed[pod_uuid] = time.time()
            return False
    
    def is_recently_destroyed(self, pod_uuid: str, window: int = 300) -> bool:
        """Check if pod was recently destroyed (uses normalized comparison)"""
        # Direct lookup first
        destroyed_at = self._recently_destroyed.get(pod_uuid)
        if destroyed_at:
            return time.time() - destroyed_at < window
        
        # Try normalized comparison
        pod_uuid_normalized = normalize_pod_uuid(pod_uuid)
        for destroyed_uuid, destroyed_at in self._recently_destroyed.items():
            if normalize_pod_uuid(destroyed_uuid) == pod_uuid_normalized:
                return time.time() - destroyed_at < window
        
        return False
    
    async def cleanup_destroyed_tracker(self, max_age: int = 3600):
        """Clean up old entries from recently destroyed tracker"""
        now = time.time()
        async with self._lock:
            expired = [
                uuid for uuid, ts in self._recently_destroyed.items()
                if now - ts > max_age
            ]
            for uuid in expired:
                del self._recently_destroyed[uuid]


# =============================================================================
# Machine Cache Manager
# =============================================================================

class MachineCacheManager:
    """
    Manages machine cache for performance optimization.
    
    Tracks:
    - Which images are cached on which machines
    - Machine success rates
    - Machine metadata (GPU, TFLOPS)
    """
    
    def __init__(self, cache_path: str = "machine_cache.json"):
        self.cache_path = cache_path
        self.cache: Dict[str, Any] = {
            "machines": {},
            "meta": {"version": 1, "updated_at": None}
        }
        self._lock = asyncio.Lock()
    
    async def load(self):
        """Load cache from file"""
        async with self._lock:
            if not os.path.exists(self.cache_path):
                return
            
            try:
                def _read():
                    with open(self.cache_path, "r") as f:
                        return json.load(f)
                
                self.cache = await asyncio.to_thread(_read)
                logger.info(
                    f"Loaded machine cache: {len(self.cache.get('machines', {}))} machines"
                )
            except Exception as e:
                logger.error(f"Failed to load machine cache: {e}")
    
    async def save(self):
        """Save cache to file"""
        async with self._lock:
            try:
                tmp_path = f"{self.cache_path}.tmp"
                
                def _write():
                    with open(tmp_path, "w") as f:
                        json.dump(self.cache, f, indent=2)
                    os.replace(tmp_path, self.cache_path)
                
                await asyncio.to_thread(_write)
                logger.debug(f"Saved machine cache to {self.cache_path}")
            except Exception as e:
                logger.error(f"Failed to save machine cache: {e}")
    
    async def mark_image_cached(self, machine_id: str, image_name: str):
        """Mark an image as cached on a machine"""
        async with self._lock:
            now = datetime.utcnow().isoformat() + "Z"
            machines = self.cache.setdefault("machines", {})
            machine = machines.setdefault(
                str(machine_id),
                {"cached_images": {}, "is_priority": False}
            )
            
            cached_images = machine.setdefault("cached_images", {})
            image_info = cached_images.get(image_name, {})
            
            cached_images[image_name] = {
                "added_at": image_info.get("added_at", now),
                "score": image_info.get("score", 0.0) + 1.0,
                "last_used": now,
            }
            
            machine["last_seen"] = now
            self.cache["meta"]["updated_at"] = now
        
        await self.save()
        logger.info(f"Marked {image_name} cached on machine {machine_id}")
    
    def get_cached_machines_for_model(self, model: str) -> List[Tuple[str, float]]:
        """
        Get machines that have a model cached, sorted by score.
        
        Returns:
            List of (machine_id, score) tuples
        """
        result = []
        machines = self.cache.get("machines", {})
        
        for mid, info in machines.items():
            cached = info.get("cached_images", {})
            if model in cached:
                score = cached[model].get("score", 0.0)
                result.append((mid, score))
        
        result.sort(key=lambda x: -x[1])  # Descending by score
        return result
    
    def get_machine_metadata(self, machine_id: str) -> Dict[str, Any]:
        """Get metadata for a machine"""
        machines = self.cache.get("machines", {})
        return machines.get(str(machine_id), {})
    
    async def update_machine_success(self, machine_id: str, success: bool):
        """Update machine success rate"""
        async with self._lock:
            machines = self.cache.setdefault("machines", {})
            machine = machines.setdefault(str(machine_id), {})
            
            stats = machine.setdefault("stats", {"success": 0, "failure": 0})
            if success:
                stats["success"] = stats.get("success", 0) + 1
            else:
                stats["failure"] = stats.get("failure", 0) + 1
            
            total = stats["success"] + stats["failure"]
            if total > 0:
                machine["success_rate"] = stats["success"] / total
        
        await self.save()


# =============================================================================
# Machine Blacklist Manager
# =============================================================================

class MachineBlacklistManager:
    """
    Manages temporary machine blacklisting.
    
    Machines are blacklisted when they fail to produce outputs.
    """
    
    def __init__(self, cooldown: int = 600):
        self._blacklist: Dict[str, float] = {}  # machine_id -> blacklist_until
        self._lock = asyncio.Lock()
        self.cooldown = cooldown
    
    def is_blacklisted(self, machine_id: str) -> bool:
        """Check if machine is currently blacklisted"""
        until = self._blacklist.get(str(machine_id))
        if not until:
            return False
        
        if time.time() < until:
            return True
        
        # Expired - clean up
        self._blacklist.pop(str(machine_id), None)
        return False
    
    async def blacklist(self, machine_id: str, duration: Optional[int] = None):
        """Blacklist a machine"""
        duration = duration or self.cooldown
        async with self._lock:
            self._blacklist[str(machine_id)] = time.time() + duration
        
        logger.warning(f"Machine {machine_id} blacklisted for {duration}s")
    
    async def unblacklist(self, machine_id: str):
        """Remove machine from blacklist"""
        async with self._lock:
            self._blacklist.pop(str(machine_id), None)
        logger.info(f"Machine {machine_id} removed from blacklist")
    
    async def cleanup_expired(self):
        """Remove expired blacklist entries"""
        now = time.time()
        async with self._lock:
            expired = [
                mid for mid, until in self._blacklist.items()
                if now >= until
            ]
            for mid in expired:
                del self._blacklist[mid]
    
    def get_cooldown_remaining(self, machine_id: str) -> Optional[int]:
        """Get remaining cooldown time for a blacklisted machine"""
        until = self._blacklist.get(str(machine_id))
        if not until:
            return None
        remaining = until - time.time()
        return int(remaining) if remaining > 0 else None
    
    def get_active_blacklist(self) -> List[str]:
        """
        Get list of currently blacklisted machine IDs (excluding expired ones).
        
        IMPORTANT: Use this instead of accessing _blacklist.keys() directly
        to ensure expired entries are filtered out.
        """
        now = time.time()
        active = [
            mid for mid, until in self._blacklist.items()
            if until > now  # Only include if blacklist hasn't expired
        ]
        return active


# =============================================================================
# Pending Job Queue Manager
# =============================================================================

class PendingJobQueueManager:
    """
    Manages pending job queues per model.
    """
    
    def __init__(self):
        self._queues: Dict[str, Set[str]] = {}  # model -> set of job_ids
        self._lock = asyncio.Lock()
    
    async def add_job(self, job_id: str, model: str):
        """Add job to pending queue"""
        async with self._lock:
            if model not in self._queues:
                self._queues[model] = set()
            self._queues[model].add(job_id)
        logger.debug(f"Added job {job_id} to pending queue for {model}")
    
    async def remove_job(self, job_id: str, model: str):
        """Remove job from pending queue"""
        async with self._lock:
            if model in self._queues:
                self._queues[model].discard(job_id)
    
    async def pop_job(self, model: str) -> Optional[str]:
        """Pop a job from the queue for a model"""
        async with self._lock:
            queue = self._queues.get(model, set())
            if queue:
                return queue.pop()
            return None
    
    def get_pending_count(self, model: str) -> int:
        """Get count of pending jobs for a model"""
        return len(self._queues.get(model, set()))
    
    def get_all_pending(self) -> Dict[str, Set[str]]:
        """Get all pending queues"""
        return dict(self._queues)
    
    async def clear_model(self, model: str):
        """Clear all pending jobs for a model (used when cache is stale)"""
        async with self._lock:
            if model in self._queues:
                cleared_count = len(self._queues[model])
                self._queues[model].clear()
                logger.info(f"Cleared {cleared_count} stale pending jobs for model {model}")


# =============================================================================
# Pod Manager (Main Class)
# =============================================================================

class PodManager:
    """
    Main pod orchestration manager.
    
    Implements:
    - Strategy-based pod allocation (warm_and_individual, machine_based)
    - Pod lifecycle management
    - Machine cache and blacklist management
    - Autoscaling
    """
    
    # Model aliases: maps variant model IDs to their base worker model
    # This allows multiple model IDs to use the same worker/template
    MODEL_ALIASES: Dict[str, str] = {
        "realesrgan_anime": "realesrgan",
        "realesrgan_face": "realesrgan",
        "realesrgan_denoise": "realesrgan",
    }
    
    @classmethod
    def resolve_model_alias(cls, model: str) -> str:
        """Resolve model alias to base model for worker routing.
        
        Returns the base model if an alias exists, otherwise returns the original model.
        """
        return cls.MODEL_ALIASES.get(model, model)
    
    def __init__(
        self,
        email: str,
        password: str,
        warm_pod_uuid: Optional[str] = None,
    ):
        self.email = email
        self.password = password
        self.warm_pod_uuid = warm_pod_uuid
        
        # Strategy
        strategy_str = os.getenv("POD_STRATEGY", "machine_based").lower()
        if strategy_str in ("warm", "warm_and_individual", "default"):
            self.strategy = PodStrategyType.WARM_AND_INDIVIDUAL
        else:
            self.strategy = PodStrategyType.MACHINE_BASED
        
        # QuickPod managers
        self.global_qp = QuickPodManager(email, password)
        self.warm_pod_manager = (
            QuickPodManager(email, password, warm_pod_uuid)
            if warm_pod_uuid
            else None
        )
        
        # Sub-managers
        self.lifecycle = PodLifecycleManager(self.global_qp)
        self.machine_cache = MachineCacheManager(
            os.getenv("MACHINE_CACHE_PATH", "machine_cache.json")
        )
        self.blacklist = MachineBlacklistManager(
            int(os.getenv("BAD_MACHINE_COOLDOWN", "600"))
        )
        self.pending_jobs = PendingJobQueueManager()
        
        # Pod tracking
        self.individual_pods: Dict[str, PodInfo] = {}
        self.individual_pod_managers: Dict[str, QuickPodManager] = {}
        
        # Locks
        self._lock = asyncio.Lock()
        self._pod_locks: Dict[str, asyncio.Lock] = {}
        self._model_locks: Dict[str, asyncio.Lock] = {}
        self._global_qp_lock = asyncio.Lock()
        self._global_qp_logged_in = False
        
        # Global pod creation semaphore - CRITICAL: prevents parallel pod creation storms
        # Limits to 2 concurrent pod creations globally to prevent runaway pod creation
        self._pod_creation_semaphore = asyncio.Semaphore(
            int(os.getenv("MAX_CONCURRENT_POD_CREATIONS", "2"))
        )
        # Track in-flight pod creations per model to prevent duplicates
        self._pod_creation_in_flight: Dict[str, bool] = {}
        self._pod_creation_backoff: Dict[str, float] = {}  # model -> last_failure_time
        
        # Configuration
        self.max_individual_pods = int(os.getenv("MAX_INDIVIDUAL_PODS", "50"))
        self.pod_startup_timeout = int(os.getenv("POD_STARTUP_TIMEOUT", "120"))
        self.worker_registration_timeout = int(os.getenv("WORKER_REGISTRATION_TIMEOUT", "180"))
        
        # Templates
        self.template_uuids = self._load_template_uuids()
        self.template_disk_sizes = self._load_disk_sizes()
        self.model_requirements = self._load_model_requirements()
        
        # Autoscaler state
        self.fallback_events: deque = deque()
        self.last_autoscale_ts = 0
        self.autoscale_cooldown = int(os.getenv("AUTOSCALE_COOLDOWN", "600"))
        
        # State
        self.own_warm_pod = False
        
        logger.info(
            f"PodManager initialized: strategy={self.strategy.value}, "
            f"warm_pod={warm_pod_uuid}"
        )
    
    def _load_template_uuids(self) -> Dict[str, Optional[str]]:
        """Load template UUIDs from environment"""
        return {
            
            "sd15": os.getenv("SD15_TEMPLATE_UUID"),
            "sd35": os.getenv("SD35_TEMPLATE_UUID"),
            "sdxl": os.getenv("SDXL_TEMPLATE_UUID"),
            
            "diffusers_outpaint": os.getenv("DIFFUSERS_OUTPAINT_TEMPLATE_UUID"),
            
            "qwen_image_edit": os.getenv("QWEN_IMAGE_EDIT_TEMPLATE_UUID"),
            "qwen_image_edit_int4": os.getenv("QWEN_IMAGE_EDIT_INT4_TEMPLATE_UUID"),

            "qwen_image_edit_2509": os.getenv("QWEN_IMAGE_EDIT_2509_TEMPLATE_UUID"),
            "qwen_image_edit_2509_int4": os.getenv("QWEN_IMAGE_EDIT_2509_INT4_TEMPLATE_UUID"),

            "qwen_image": os.getenv("QWEN_IMAGE_TEMPLATE_UUID"),
            "qwen_image_int4": os.getenv("QWEN_IMAGE_INT4_TEMPLATE_UUID"),

            "flux_kontext_dev": os.getenv("FLUX_KONTEXT_DEV_TEMPLATE_UUID"),
            "flux_kontext_dev_int4": os.getenv("FLUX_KONTEXT_DEV_INT4_TEMPLATE_UUID"),

            "z_image_turbo": os.getenv("Z_IMAGE_TURBO_TEMPLATE_UUID"),
            "z_image_turbo_int4": os.getenv("Z_IMAGE_TURBO_INT4_TEMPLATE_UUID"),

            "fastwan_21_1b": os.getenv("FASTWAN_21_1B_TEMPLATE_UUID"),
            "fastwan_22_5b": os.getenv("FASTWAN_22_5B_TEMPLATE_UUID"),

            "aurasrv2": os.getenv("AURASRV2_TEMPLATE_UUID"),
            "realesrgan": os.getenv("REALESRGAN_TEMPLATE_UUID"),
            "swin2sr": os.getenv("SWIN2SR_TEMPLATE_UUID"),

            "basic_utils": os.getenv("BASIC_UTILS_TEMPLATE_UUID"),
            "warm_pod": os.getenv("WARM_POD_TEMPLATE_UUID"),
            "default": os.getenv("DEFAULT_TEMPLATE_UUID"),
        }
    
    def _load_disk_sizes(self) -> Dict[str, int]:
        """Load disk sizes for templates"""
        return {
            "sd15": 30,
            "sd35": 40,
            "sdxl": 30,
            "diffusers_outpaint": 50,
            "qwen_image": 80,
            "qwen_image_edit": 80,
            "qwen_image_edit_2509": 85,
            "flux_kontext_dev": 45,
            "z_image_turbo": 80,
            "qwen_image_int4": 80,
            "qwen_image_edit_int4": 80,
            "qwen_image_edit_2509_int4": 85,
            "flux_kontext_dev_int4": 45,
            "z_image_turbo_int4": 80,
            "fastwan_21_1b": 60,
            "fastwan_22_5b": 60,
            "aurasrv2": 30,
            "realesrgan": 30,
            "swin2sr": 30,
            "basic_utils": 30,
        }
    
    def _load_model_requirements(self) -> Dict[str, Any]:
        """Load model requirements from JSON file"""
        path = os.getenv("MODEL_REQUIREMENTS_PATH", "model_requirements.json")
        try:
            with open(path, "r") as f:
                data = json.load(f)
            logger.info(f"Loaded model requirements for {len(data)} models")
            return data
        except Exception as e:
            logger.warning(f"Failed to load model_requirements.json: {e}")
            return {}
    
    # -------------------------------------------------------------------------
    # Locking Helpers
    # -------------------------------------------------------------------------
    
    def _get_pod_lock(self, pod_uuid: str) -> asyncio.Lock:
        """Get or create lock for a specific pod"""
        lock = self._pod_locks.get(pod_uuid)
        if lock:
            return lock
        new_lock = asyncio.Lock()
        return self._pod_locks.setdefault(pod_uuid, new_lock)
    
    def _get_model_lock(self, model: str) -> asyncio.Lock:
        """Get or create lock for a specific model"""
        lock = self._model_locks.get(model)
        if lock:
            return lock
        new_lock = asyncio.Lock()
        return self._model_locks.setdefault(model, new_lock)
    
    # -------------------------------------------------------------------------
    # Active Pods Property
    # -------------------------------------------------------------------------
    
    @property
    def active_pods(self) -> Dict[str, PodInfo]:
        """Get all active pods (for validation)"""
        pods = dict(self.individual_pods)
        if self.warm_pod_uuid and self.own_warm_pod:
            pods[self.warm_pod_uuid] = PodInfo(
                pod_uuid=self.warm_pod_uuid,
                model="warm",
                mode=None,
                status=PodStatus.READY,
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow(),
                is_warm_pod=True,
            )
        return pods
    
    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    
    async def initialize(self):
        """Initialize pod manager"""
        # Load machine cache
        await self.machine_cache.load()
        
        # Initialize warm pod if configured
        if self.warm_pod_manager:
            try:
                logger.info("Initializing warm pod...")
                await self._ensure_global_login()
                resp = await asyncio.to_thread(self.warm_pod_manager.start_pod)
                
                if isinstance(resp, dict) and "busy" in str(resp.get("error", "")).lower():
                    logger.warning("Warm pod busy, cannot acquire")
                    self.own_warm_pod = False
                else:
                    self.own_warm_pod = True
                    logger.info("Warm pod started successfully")
            except Exception as e:
                logger.exception(f"Failed to initialize warm pod: {e}")
                self.own_warm_pod = False
        else:
            logger.info("No warm pod configured")
            self.own_warm_pod = False
    
    async def _ensure_global_login(self, force_refresh: bool = False):
        """Ensure global QuickPod manager is logged in"""
        async with self._global_qp_lock:
            needs_login = (
                force_refresh or
                not self._global_qp_logged_in or
                not self.global_qp._is_token_valid()
            )
            
            if needs_login:
                try:
                    await asyncio.to_thread(self.global_qp.login)
                    self._global_qp_logged_in = True
                    logger.info("QuickPod global login successful")
                except Exception as e:
                    logger.error(f"QuickPod global login failed: {e}")
                    raise
    
    # -------------------------------------------------------------------------
    # Variant & Template Helpers
    # -------------------------------------------------------------------------
    
    def get_variant_plan(self, model: str) -> List[Tuple[Optional[str], List[str]]]:
        """
        Get variant fallback plan for a model.
        
        Returns list of (variant, allowed_gpus) tuples in priority order.
        Each variant has its own allowed GPU list:
        - fp4 variants are for 5090 only
        - int4 variants are for other GPUs (4090, 4080, 4070)
        """
        # Resolve model alias to base model for lookups
        base_model = self.resolve_model_alias(model)
        req = self.model_requirements.get(base_model, {})
        
        preferred = req.get("preferred_variant")
        fallbacks = req.get("fallback_variants", [])
        default_allowed_gpus = req.get("allowed_gpu_names", [])
        variant_allowed_gpus = req.get("variant_allowed_gpus", {})
        
        plan = []
        
        # Add preferred variant first with variant-specific GPU requirements
        if preferred:
            gpus_for_variant = variant_allowed_gpus.get(preferred, default_allowed_gpus)
            plan.append((preferred, gpus_for_variant))
            logger.debug(f"Variant plan for {model}: {preferred} -> GPUs: {gpus_for_variant}")
        
        # Add fallback variants with their specific GPU requirements
        for fb in fallbacks:
            gpus_for_variant = variant_allowed_gpus.get(fb, default_allowed_gpus)
            plan.append((fb, gpus_for_variant))
            logger.debug(f"Variant plan for {model}: {fb} -> GPUs: {gpus_for_variant}")
        
        # Add base variant (no variant) if no preferred
        if not preferred and not plan:
            plan.append((None, default_allowed_gpus))
        
        logger.info(f"Variant plan for {model}: {[(v, gpus) for v, gpus in plan]}")
        return plan
    
    def get_template_for_variant(self, model: str, variant: Optional[str]) -> Optional[str]:
        """Get template UUID for a model/variant combination"""
        # Resolve model alias to base model for lookups
        base_model = self.resolve_model_alias(model)
        req = self.model_requirements.get(base_model, {})
        
        if variant and "variant_template_key" in req:
            template_key = req["variant_template_key"].get(variant, base_model)
        else:
            template_key = base_model
        
        return self.template_uuids.get(template_key)
    
    def get_disk_size_for_variant(self, model: str, variant: Optional[str]) -> int:
        """Get disk size for a model/variant combination"""
        # Resolve model alias to base model for lookups
        base_model = self.resolve_model_alias(model)
        req = self.model_requirements.get(base_model, {})
        
        if variant and "variant_disk_size" in req:
            return req["variant_disk_size"].get(variant, self.template_disk_sizes.get(base_model, 50))
        
        return self.template_disk_sizes.get(base_model, 50)
    
    # -------------------------------------------------------------------------
    # Machine Management
    # -------------------------------------------------------------------------
    
    def is_machine_bad(self, machine_id: str) -> bool:
        """Check if machine is blacklisted"""
        return self.blacklist.is_blacklisted(machine_id)
    
    async def mark_machine_bad(self, machine_id: str, duration: Optional[int] = None):
        """Blacklist a machine"""
        duration = duration or self.blacklist.cooldown
        logger.warning(
            f"🚫 BLACKLISTING machine {machine_id} for {duration}s due to failure"
        )
        await self.blacklist.blacklist(machine_id, duration)
        await self.machine_cache.update_machine_success(machine_id, success=False)
        
        # Log current blacklist state
        blacklisted_machines = [
            mid for mid, until in self.blacklist._blacklist.items()
            if time.time() < until
        ]
        logger.info(f"Currently blacklisted machines: {blacklisted_machines}")
    
    async def mark_machine_good(self, machine_id: str):
        """Record successful use of a machine and remove from blacklist if present"""
        await self.machine_cache.update_machine_success(machine_id, success=True)
        
        # CRITICAL: If machine was blacklisted but now succeeded, remove from blacklist
        # This handles the case where a machine was blacklisted due to a failed pod,
        # but a new pod on the same machine succeeded
        if self.is_machine_bad(str(machine_id)):
            await self.blacklist.unblacklist(str(machine_id))
            logger.info(f"Machine {machine_id} removed from blacklist after successful pod startup")
    
    # -------------------------------------------------------------------------
    # Pod Slot Management
    # -------------------------------------------------------------------------
    
    async def reserve_slot_on_pod(self, pod_uuid: str, job_id: str) -> bool:
        """
        Reserve a slot on a pod for a job.
        
        Returns True if slot reserved, False if pod full or not tracked.
        """
        pod_lock = self._get_pod_lock(pod_uuid)
        async with pod_lock:
            pi = self.individual_pods.get(pod_uuid)
            if not pi:
                return False
            
            # Per-model concurrency cap: heavy models can set max_jobs_per_pod in model_requirements.json
            model_limit = self.model_requirements.get(pi.model, {}).get(
                "max_jobs_per_pod", MAX_JOBS_PER_POD
            )
            limit = max(1, int(model_limit))

            if pi.queue_size >= limit:
                return False
            
            pi.queue_size += 1
            pi.last_active = datetime.utcnow()
            pi.job_start_times[job_id] = datetime.utcnow()
            
            logger.debug(
                f"Reserved slot on {pod_uuid}: job={job_id}, queue_size={pi.queue_size}"
            )
            return True
    
    async def release_slot_on_pod(self, pod_uuid: str, job_id: str):
        """Release a reserved slot on a pod"""
        pod_lock = self._get_pod_lock(pod_uuid)
        async with pod_lock:
            pi = self.individual_pods.get(pod_uuid)
            if not pi:
                return
            
            pi.queue_size = max(0, pi.queue_size - 1)
            pi.last_active = datetime.utcnow()
            pi.job_start_times.pop(job_id, None)
    
    async def mark_job_completed_on_pod(self, pod_uuid: str, job_id: str):
        """Mark a job as completed on a pod"""
        pod_lock = self._get_pod_lock(pod_uuid)
        async with pod_lock:
            pi = self.individual_pods.get(pod_uuid)
            if not pi:
                return
            
            pi.queue_size = max(0, pi.queue_size - 1)
            pi.total_jobs_processed += 1
            pi.unique_job_ids.add(job_id)
            pi.job_id_counts[job_id] = pi.job_id_counts.get(job_id, 0) + 1
            pi.last_active = datetime.utcnow()
            pi.job_start_times.pop(job_id, None)
    
    async def record_job_start(
        self,
        pod_uuid: str,
        job_id: str,
        started_at: Optional[datetime] = None,
    ):
        """Record job start time on a pod"""
        pod_lock = self._get_pod_lock(pod_uuid)
        async with pod_lock:
            pi = self.individual_pods.get(pod_uuid)
            if not pi:
                return
            
            pi.job_start_times[job_id] = started_at or datetime.utcnow()
    
    # -------------------------------------------------------------------------
    # Pending Jobs
    # -------------------------------------------------------------------------
    
    async def add_pending_job(self, job_id: str, model: str):
        """Add job to pending queue"""
        await self.pending_jobs.add_job(job_id, model)
    
    async def remove_pending_job(self, job_id: str, model: str):
        """Remove job from pending queue"""
        await self.pending_jobs.remove_job(job_id, model)
    
    # -------------------------------------------------------------------------
    # Pod Health Check
    # -------------------------------------------------------------------------
    
    def verify_worker_network_ready(self, endpoint: str) -> bool:
        """Check if worker is reachable via health endpoint"""
        try:
            base = (endpoint or "").rstrip("/")
            if not base.startswith("http"):
                return False
            
            url = f"{base}/health"
            resp = requests.get(url, timeout=5)
            return resp.status_code == 200
        except Exception as e:
            logger.debug(f"Worker health check failed for {endpoint}: {e}")
            return False
    
    async def check_pod_ready(self, pod_uuid: str) -> bool:
        """Check if a pod is ready (has a registered worker)"""
        # Import here to avoid circular import
        from db import SessionLocal, Worker
        
        with SessionLocal() as db:
            worker = (
                db.query(Worker)
                .filter(Worker.container_id == pod_uuid)
                .first()
            )
            
            if worker and worker.status == "ready":
                if worker.endpoint:
                    return self.verify_worker_network_ready(worker.endpoint)
                return True
            
            return False
    
    # -------------------------------------------------------------------------
    # Job Dispatch
    # -------------------------------------------------------------------------
    
    async def dispatch_job(
        self,
        job_id: str,
        model: str,
        mode: Optional[str] = None,
    ) -> str:
        """
        Dispatch a job to a worker.
        
        Returns worker_id on success.
        """
        if self.strategy == PodStrategyType.WARM_AND_INDIVIDUAL:
            return await self._dispatch_warm_and_individual(job_id, model, mode)
        else:
            return await self._dispatch_machine_based(job_id, model, mode)
    
    async def _dispatch_warm_and_individual(
        self,
        job_id: str,
        model: str,
        mode: Optional[str],
    ) -> str:
        """
        Dispatch using warm + individual pod strategy.
        
        Priority:
        1. Use warm worker if available
        2. Wait for existing STARTING pod for same model (avoid duplicate pod creation)
        3. Create a new individual pod
        """
        from db import SessionLocal, Worker, Job
        from models import JobStatus
        
        await self.add_pending_job(job_id, model)
        
        # === PHASE 1: Try warm worker first ===
        with SessionLocal() as db:
            worker = (
                db.query(Worker)
                .filter(
                    Worker.is_warm_pod == True,
                    Worker.model == model,
                    Worker.status == "ready",  # ONLY ready workers - NOT starting!
                )
                .order_by(Worker.last_active.desc())
                .first()
            )
            
            if worker:
                worker.status = "dispatching"
                worker.last_active = datetime.utcnow()
                
                job = db.query(Job).filter(Job.job_id == job_id).first()
                if job:
                    job.status = JobStatus.DISPATCHING.value
                    job.updated_at = datetime.utcnow()
                
                db.commit()
                
                # Record start time
                container_id = worker.container_id or self.warm_pod_uuid
                if container_id:
                    asyncio.create_task(self.record_job_start(container_id, job_id))
                
                logger.info(f"Dispatching job {job_id} to warm worker {worker.worker_id}")
                return worker.worker_id
        
        # === PHASE 2: Check for STARTING pods for the same model ===
        # If another job already triggered pod creation, wait for it instead of creating duplicate
        starting_pod_uuid = None
        async with self._lock:
            for pod_uuid, info in self.individual_pods.items():
                if info.model == model and info.status == PodStatus.STARTING:
                    # Skip blacklisted machines
                    if info.machine_id and self.is_machine_bad(info.machine_id):
                        continue
                    starting_pod_uuid = pod_uuid
                    logger.info(
                        f"Found STARTING pod {pod_uuid} for model {model}, "
                        f"will wait instead of creating new pod for job {job_id}"
                    )
                    break
        
        if starting_pod_uuid:
            # Wait for this pod to become ready
            logger.info(f"Job {job_id} waiting for existing STARTING pod {starting_pod_uuid}")
            try:
                worker_id = await self._wait_for_worker_registration(
                    starting_pod_uuid, model, job_id
                )
                logger.info(f"Job {job_id} successfully assigned to pod {starting_pod_uuid} (waited for STARTING)")
                return worker_id
            except WorkerTimeoutError:
                logger.warning(
                    f"Timeout waiting for STARTING pod {starting_pod_uuid}, "
                    f"will create new pod for job {job_id}"
                )
                # Fall through to create a new pod
        
        # === PHASE 3: No warm worker and no STARTING pods - create individual pod ===
        logger.info(f"No warm worker or STARTING pods for model {model}, creating new pod for job {job_id}")
        pod_uuid = await self.create_individual_pod(model, mode)
        if not pod_uuid:
            raise PodCreationError(model, reason="Failed to create individual pod")
        
        # Wait for worker registration
        return await self._wait_for_worker_registration(pod_uuid, model, job_id)
    
    async def _dispatch_machine_based(
        self,
        job_id: str,
        model: str,
        mode: Optional[str],
    ) -> str:
        """
        Dispatch using machine-based strategy.
        
        Priority:
        1. Reuse existing READY pod for the same model (with available slot)
        2. Wait for existing STARTING pod for the same model (avoid duplicate pod creation)
        3. Wait for BUSY pod for the same model (max_jobs reached, will become free soon)
        4. Create a new pod if no existing pods for this model
        """
        from db import SessionLocal, Worker, Job
        from models import JobStatus
        
        await self.add_pending_job(job_id, model)
        
        # Collect candidates
        async with self._lock:
            candidates = list(self.individual_pods.items())
        
        # Track pods that exist for this model but have no available slots
        busy_pods_for_model = []
        
        # === PHASE 1: Try to find a READY pod to reuse immediately ===
        for pod_uuid, info in candidates:
            # Skip blacklisted machines
            if info.machine_id and self.is_machine_bad(info.machine_id):
                continue
            
            # CRITICAL: Skip pods that were recently destroyed (shouldn't be in candidates, but double-check)
            if self.lifecycle.is_recently_destroyed(pod_uuid):
                logger.debug(f"Skipping recently destroyed pod {pod_uuid}")
                continue
            
            if info.model == model and info.status == PodStatus.READY:
                # Try to reserve slot
                reserved = await self.reserve_slot_on_pod(pod_uuid, job_id)
                if not reserved:
                    # Pod exists for this model but has no available slots (busy)
                    busy_pods_for_model.append(pod_uuid)
                    continue
                
                # Update DB
                with SessionLocal() as db:
                    worker = (
                        db.query(Worker)
                        .filter(Worker.container_id == pod_uuid, Worker.model == model)
                        .first()
                    )
                    
                    if worker:
                        # CRITICAL: Never dispatch to workers that aren't truly ready.
                        # Some pods register as "starting" and only become ready after model load.
                        # If we flip them to "dispatching" here, routes._dispatch_to_worker won't detect
                        # the starting state and server will timeout hitting /run.
                        if worker.status != "ready":
                            logger.info(
                                f"Pod {pod_uuid} tracked as READY but DB worker {worker.worker_id} "
                                f"is '{worker.status}'. Skipping reuse for job {job_id}."
                            )
                            # CRITICAL: Treat this pod as STARTING so Phase-2 can wait for it,
                            # instead of falling through to "no existing pods" and creating a duplicate pod.
                            async with self._lock:
                                pi = self.individual_pods.get(pod_uuid)
                                if pi and pi.status == PodStatus.READY:
                                    pi.status = PodStatus.STARTING
                                    logger.info(
                                        f"Downgraded pod {pod_uuid} tracking status READY→STARTING "
                                        f"(DB worker='{worker.status}') to avoid duplicate pod creation"
                                    )
                            # Rollback reservation and try other pods / create new pod
                            # (must be immediate to avoid leaking queue slots)
                            await self.release_slot_on_pod(pod_uuid, job_id)
                            continue

                        worker.status = "dispatching"
                        worker.last_active = datetime.utcnow()
                        db.commit()
                        logger.info(f"Reusing READY pod {pod_uuid} for job {job_id}")
                        return worker.worker_id
                    else:
                        # Rollback reservation
                        await self.release_slot_on_pod(pod_uuid, job_id)
        
        # === PHASE 2: Check for STARTING pods for the same model ===
        # If another job already triggered pod creation, wait for it instead of creating duplicate
        starting_pod_uuid = None
        async with self._lock:
            for pod_uuid, info in self.individual_pods.items():
                if info.model == model and info.status == PodStatus.STARTING:
                    # Skip blacklisted machines
                    if info.machine_id and self.is_machine_bad(info.machine_id):
                        continue
                    starting_pod_uuid = pod_uuid
                    logger.info(
                        f"Found STARTING pod {pod_uuid} for model {model}, "
                        f"will wait instead of creating new pod for job {job_id}"
                    )
                    break
        
        if starting_pod_uuid:
            # Wait for this pod to become ready
            logger.info(f"Job {job_id} waiting for existing STARTING pod {starting_pod_uuid}")
            try:
                worker_id = await self._wait_for_worker_registration(
                    starting_pod_uuid, model, job_id
                )
                # Record that this job reused a pod that was being created
                async with self._lock:
                    pi = self.individual_pods.get(starting_pod_uuid)
                    if pi:
                        await self.record_job_start(starting_pod_uuid, job_id, started_at=pi.created_at)
                logger.info(f"Job {job_id} successfully assigned to pod {starting_pod_uuid} (waited for STARTING)")
                return worker_id
            except JobAlreadyProcessedError:
                # Job was processed by another dispatch task - this is OK, just return
                logger.info(
                    f"Job {job_id} was already processed by another dispatch task "
                    f"while waiting for STARTING pod {starting_pod_uuid}"
                )
                raise  # Re-raise to signal to caller
            except WorkerTimeoutError:
                logger.warning(
                    f"Timeout waiting for STARTING pod {starting_pod_uuid}, "
                    f"will re-check for READY pods before creating new pod for job {job_id}"
                )
                # CRITICAL FIX: Re-check for READY pods after timeout.
                # The pod we were waiting for may have been destroyed, but another pod
                # (that was BUSY when we started) may now be READY.
                async with self._lock:
                    candidates_after_timeout = list(self.individual_pods.items())
                
                for pod_uuid, info in candidates_after_timeout:
                    if info.machine_id and self.is_machine_bad(info.machine_id):
                        continue
                    if self.lifecycle.is_recently_destroyed(pod_uuid):
                        continue
                    if info.model == model and info.status == PodStatus.READY:
                        reserved = await self.reserve_slot_on_pod(pod_uuid, job_id)
                        if not reserved:
                            continue
                        with SessionLocal() as db:
                            worker = (
                                db.query(Worker)
                                .filter(Worker.container_id == pod_uuid, Worker.model == model)
                                .first()
                            )
                            if worker and worker.status == "ready":
                                worker.status = "dispatching"
                                worker.last_active = datetime.utcnow()
                                db.commit()
                                logger.info(
                                    f"After timeout, found READY pod {pod_uuid} for job {job_id} "
                                    f"(avoiding unnecessary pod creation)"
                                )
                                return worker.worker_id
                            else:
                                await self.release_slot_on_pod(pod_uuid, job_id)
                
                # No READY pods found after timeout, fall through to create new pod
        
        # === PHASE 2.5: Wait for BUSY pod to become available ===
        # If there are existing pods for this model but all are busy (max_jobs reached),
        # wait for one to become available instead of creating a new pod
        if busy_pods_for_model:
            busy_pod_uuid = busy_pods_for_model[0]
            logger.info(
                f"Found BUSY pod {busy_pod_uuid} for model {model} (max jobs reached), "
                f"waiting for slot to become available for job {job_id}"
            )
            try:
                worker_id = await self._wait_for_busy_pod_slot(
                    busy_pod_uuid, model, job_id
                )
                if worker_id:
                    logger.info(
                        f"Job {job_id} successfully assigned to pod {busy_pod_uuid} "
                        f"(waited for BUSY pod slot)"
                    )
                    return worker_id
            except JobAlreadyProcessedError:
                # Job was processed by another dispatch task - this is OK
                logger.info(
                    f"Job {job_id} was already processed by another dispatch task "
                    f"while waiting for BUSY pod {busy_pod_uuid}"
                )
                raise
            except Exception as e:
                logger.warning(
                    f"Failed to wait for BUSY pod {busy_pod_uuid}: {e}. "
                    f"Will create new pod for job {job_id}"
                )
                # Fall through to create a new pod
        
        # === PHASE 3: No existing pods - create a new one ===
        logger.info(f"No existing pods for model {model}, creating new pod for job {job_id}")
        pod_uuid = await self.create_machine_based_pod(model, mode)
        if not pod_uuid:
            # RACE CONDITION FIX: If pod creation returned None, check if it's because
            # another pod creation was already in-flight. If so, retry finding the STARTING pod.
            async with self._lock:
                in_flight = self._pod_creation_in_flight.get(model, False)
            
            if in_flight:
                logger.info(
                    f"Pod creation returned None (in-flight), retrying to find STARTING pod for job {job_id}"
                )
                # Wait a moment for the pod to be registered
                await asyncio.sleep(0.5)
                
                # Retry Phase 2 - look for STARTING pod (with retries)
                for retry_attempt in range(10):  # Try up to 10 times (5 seconds)
                    async with self._lock:
                        for pu, pi in self.individual_pods.items():
                            if pi.model == model and pi.status in [PodStatus.STARTING, PodStatus.READY]:
                                if pi.machine_id and self.is_machine_bad(pi.machine_id):
                                    continue
                                logger.info(
                                    f"Found {pi.status.name} pod {pu} for model {model} after in-flight retry, "
                                    f"waiting for it (job {job_id})"
                                )
                                break
                        else:
                            # No pod found yet, wait and retry
                            await asyncio.sleep(0.5)
                            continue
                        
                        # Found a pod, exit retry loop
                        pod_uuid = pu
                        break
                
                if pod_uuid:
                    # Found a STARTING/READY pod, wait for worker registration
                    try:
                        worker_id = await self._wait_for_worker_registration(
                            pod_uuid, model, job_id
                        )
                        async with self._lock:
                            pi = self.individual_pods.get(pod_uuid)
                            if pi:
                                await self.record_job_start(pod_uuid, job_id, started_at=pi.created_at)
                        logger.info(
                            f"Job {job_id} successfully assigned to pod {pod_uuid} "
                            f"(found after in-flight retry)"
                        )
                        return worker_id
                    except JobAlreadyProcessedError:
                        raise
                    except Exception as e:
                        logger.warning(
                            f"Failed waiting for pod {pod_uuid} after in-flight retry: {e}"
                        )
                        # Fall through to raise error
            
            # Log blacklist status for debugging
            blacklisted_machines = [
                f"{mid} (until {datetime.fromtimestamp(until).strftime('%H:%M:%S')})"
                for mid, until in self.blacklist._blacklist.items()
                if self.blacklist.is_blacklisted(mid)
            ]
            error_msg = "Failed to create machine-based pod"
            if blacklisted_machines:
                error_msg += f". All machines blacklisted: {blacklisted_machines}"
            raise PodCreationError(model, reason=error_msg)
        
        # Record start time from pod creation
        async with self._lock:
            pi = self.individual_pods.get(pod_uuid)
            if pi:
                await self.record_job_start(pod_uuid, job_id, started_at=pi.created_at)
        
        # Wait for worker registration
        try:
            return await self._wait_for_worker_registration(pod_uuid, model, job_id)
        except JobAlreadyProcessedError:
            # Job was processed by another dispatch task - this is OK
            logger.info(
                f"Job {job_id} was already processed by another dispatch task "
                f"while waiting for new pod {pod_uuid}"
            )
            raise
    
    async def _wait_for_worker_registration(
        self,
        pod_uuid: str,
        model: str,
        job_id: str,
    ) -> str:
        """
        Wait for a worker to register for a pod.
        
        CRITICAL: Only assigns jobs to workers in "ready" state, NOT "starting".
        Workers in "starting" state are still loading models and cannot process jobs.
        
        RACE CONDITION FIX: Check job state before assigning to prevent double dispatch
        when multiple dispatch tasks are waiting for the same pod.
        """
        from db import SessionLocal, Worker, Job
        from job_manager import JobState
        
        start = time.time()
        timeout = self.worker_registration_timeout
        worker_found_starting = False  # Track if we found a starting worker
        
        while time.time() - start < timeout:
            # CRITICAL: Check if pod was destroyed while we're waiting
            # This prevents waiting the full timeout for a pod that no longer exists
            if self.lifecycle.is_recently_destroyed(pod_uuid):
                elapsed = time.time() - start
                logger.warning(
                    f"Pod {pod_uuid} was destroyed while waiting for worker registration "
                    f"(waited {elapsed:.1f}s), raising timeout to re-check for other pods"
                )
                raise WorkerTimeoutError(f"Pod {pod_uuid} was destroyed during wait")
            
            # Also check if pod is still tracked (might be removed without going through destroy)
            async with self._lock:
                still_tracked = pod_uuid in self.individual_pods
            if not still_tracked:
                elapsed = time.time() - start
                logger.warning(
                    f"Pod {pod_uuid} no longer tracked while waiting for worker registration "
                    f"(waited {elapsed:.1f}s), raising timeout to re-check for other pods"
                )
                raise WorkerTimeoutError(f"Pod {pod_uuid} no longer tracked during wait")
            
            with SessionLocal() as db:
                # RACE CONDITION FIX: Check if job is still in a dispatchable state
                # Another dispatch task might have already assigned this job
                job = db.query(Job).filter(Job.job_id == job_id).first()
                if job:
                    current_state = JobState(job.status)
                    if current_state in [JobState.RUNNING, JobState.COMPLETED, JobState.FAILED,
                                         JobState.TIMEOUT, JobState.CANCELLED]:
                        logger.info(
                            f"Job {job_id} is already in {current_state.value} state, "
                            f"aborting wait for worker registration on pod {pod_uuid}"
                        )
                        # Raise an exception to signal the caller that job is already handled
                        raise JobAlreadyProcessedError(
                            f"Job {job_id} is already in {current_state.value} state"
                        )
                
                # CRITICAL FIX: Only accept READY workers, NOT starting
                # A worker in "starting" state is still loading models and will fail health checks
                worker = (
                    db.query(Worker)
                    .filter(
                        Worker.container_id == pod_uuid,
                        Worker.model == model,
                        Worker.status == "ready",  # ONLY ready workers - NOT starting!
                    )
                    .first()
                )
                
                if worker:
                    # Double-check job state right before assignment
                    job = db.query(Job).filter(Job.job_id == job_id).first()
                    if job:
                        current_state = JobState(job.status)
                        if current_state in [JobState.RUNNING, JobState.COMPLETED, JobState.FAILED]:
                            logger.info(
                                f"Job {job_id} changed to {current_state.value} before assignment, "
                                f"aborting wait for worker registration"
                            )
                            raise JobAlreadyProcessedError(
                                f"Job {job_id} is already in {current_state.value} state"
                            )
                    
                    worker.status = "dispatching"
                    worker.last_active = datetime.utcnow()
                    db.commit()
                    logger.info(
                        f"Worker {worker.worker_id} is READY, assigning job {job_id}"
                    )
                    return worker.worker_id
                
                # Check if there's a starting worker (for logging purposes)
                starting_worker = (
                    db.query(Worker)
                    .filter(
                        Worker.container_id == pod_uuid,
                        Worker.model == model,
                        Worker.status == "starting",
                    )
                    .first()
                )
                
                if starting_worker and not worker_found_starting:
                    worker_found_starting = True
                    logger.info(
                        f"Worker {starting_worker.worker_id} registered but in 'starting' state, "
                        f"waiting for models to load before assigning job {job_id}..."
                    )
            
            await asyncio.sleep(1)
        
        # Check if we timed out with a starting worker (vs no worker at all)
        with SessionLocal() as db:
            # Final job state check
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job:
                current_state = JobState(job.status)
                if current_state in [JobState.RUNNING, JobState.COMPLETED, JobState.FAILED]:
                    logger.info(
                        f"Job {job_id} is in {current_state.value} state at timeout, "
                        f"was likely handled by another dispatch task"
                    )
                    raise JobAlreadyProcessedError(
                        f"Job {job_id} is already in {current_state.value} state"
                    )
            
            final_worker = (
                db.query(Worker)
                .filter(
                    Worker.container_id == pod_uuid,
                    Worker.model == model,
                )
                .first()
            )
            
            if final_worker:
                if final_worker.status == "starting":
                    logger.warning(
                        f"Pod {pod_uuid} worker {final_worker.worker_id} timed out "
                        f"while still in 'starting' state - models may not have loaded"
                    )
                elif final_worker.status == "ready":
                    # Race condition - worker became ready just now
                    # But first check job state
                    job = db.query(Job).filter(Job.job_id == job_id).first()
                    if job:
                        current_state = JobState(job.status)
                        if current_state in [JobState.RUNNING, JobState.COMPLETED, JobState.FAILED]:
                            raise JobAlreadyProcessedError(
                                f"Job {job_id} is already in {current_state.value} state"
                            )
                    
                    logger.info(
                        f"Worker {final_worker.worker_id} became ready during timeout check, "
                        f"assigning job {job_id}"
                    )
                    final_worker.status = "dispatching"
                    final_worker.last_active = datetime.utcnow()
                    db.commit()
                    return final_worker.worker_id
        
        raise WorkerTimeoutError(pod_uuid, timeout)
    
    async def _wait_for_busy_pod_slot(
        self,
        pod_uuid: str,
        model: str,
        job_id: str,
        timeout: int = 120,
    ) -> Optional[str]:
        """
        Wait for a busy pod (max jobs reached) to have an available slot.
        
        This prevents creating unnecessary new pods when an existing pod is just
        temporarily full and will become available soon.
        
        Args:
            pod_uuid: The UUID of the busy pod
            model: The model name
            job_id: The job ID waiting for a slot
            timeout: Maximum time to wait (default 120s = 2 minutes)
            
        Returns:
            worker_id if slot became available and job was assigned
            None if timeout or pod was destroyed
        """
        from db import SessionLocal, Worker, Job
        from job_manager import JobState
        
        start = time.time()
        last_log_time = 0
        
        while time.time() - start < timeout:
            # RACE CONDITION FIX: Check if job is still in a dispatchable state
            with SessionLocal() as db:
                job = db.query(Job).filter(Job.job_id == job_id).first()
                if job:
                    current_state = JobState(job.status)
                    if current_state in [JobState.RUNNING, JobState.COMPLETED, JobState.FAILED,
                                         JobState.TIMEOUT, JobState.CANCELLED]:
                        logger.info(
                            f"Job {job_id} is already in {current_state.value} state, "
                            f"aborting wait for BUSY pod slot on {pod_uuid}"
                        )
                        raise JobAlreadyProcessedError(
                            f"Job {job_id} is already in {current_state.value} state"
                        )
            
            # Check if pod still exists
            async with self._lock:
                pi = self.individual_pods.get(pod_uuid)
                if not pi:
                    logger.warning(
                        f"Pod {pod_uuid} was destroyed while job {job_id} was waiting for slot"
                    )
                    return None
                
                if pi.model != model:
                    logger.warning(
                        f"Pod {pod_uuid} model changed from {model} to {pi.model}"
                    )
                    return None
            
            # Try to reserve a slot
            reserved = await self.reserve_slot_on_pod(pod_uuid, job_id)
            if reserved:
                # Got a slot, now try to assign the job
                with SessionLocal() as db:
                    # Double-check job state before assignment
                    job = db.query(Job).filter(Job.job_id == job_id).first()
                    if job:
                        current_state = JobState(job.status)
                        if current_state in [JobState.RUNNING, JobState.COMPLETED, JobState.FAILED]:
                            await self.release_slot_on_pod(pod_uuid, job_id)
                            raise JobAlreadyProcessedError(
                                f"Job {job_id} is already in {current_state.value} state"
                            )
                    
                    worker = (
                        db.query(Worker)
                        .filter(Worker.container_id == pod_uuid, Worker.model == model)
                        .first()
                    )
                    
                    if worker and worker.status == "ready":
                        worker.status = "dispatching"
                        worker.last_active = datetime.utcnow()
                        db.commit()
                        
                        elapsed = time.time() - start
                        logger.info(
                            f"Job {job_id} got slot on previously BUSY pod {pod_uuid} "
                            f"after waiting {elapsed:.1f}s"
                        )
                        return worker.worker_id
                    else:
                        # Worker not ready yet, release slot and keep waiting
                        await self.release_slot_on_pod(pod_uuid, job_id)
                        if worker:
                            logger.debug(
                                f"Pod {pod_uuid} has slot but worker status is '{worker.status}', "
                                f"waiting for ready state"
                            )
            
            # Log progress every 10 seconds
            elapsed = time.time() - start
            if elapsed - last_log_time >= 10:
                logger.debug(
                    f"Job {job_id} still waiting for slot on BUSY pod {pod_uuid} "
                    f"({elapsed:.0f}s elapsed)"
                )
                last_log_time = elapsed
            
            await asyncio.sleep(1)
        
        # Timeout
        elapsed = time.time() - start
        logger.warning(
            f"Timeout ({elapsed:.0f}s) waiting for slot on BUSY pod {pod_uuid} "
            f"for job {job_id}"
        )
        return None
    
    # -------------------------------------------------------------------------
    # Pod Creation
    # -------------------------------------------------------------------------
    
    async def create_individual_pod(
        self,
        model: str,
        mode: Optional[str] = None,
    ) -> Optional[str]:
        """Create an individual pod using simple template selection with cost tracking"""
        await self._ensure_global_login()
        
        # Resolve model alias to base model for template/config lookups
        base_model = self.resolve_model_alias(model)
        
        template_uuid = self.template_uuids.get(base_model)
        if not template_uuid:
            template_uuid = self.template_uuids.get("default")
        
        if not template_uuid:
            logger.error(f"No template found for model {model} (base: {base_model})")
            return None
        
        disk_size = self.template_disk_sizes.get(base_model, 50)
        
        try:
            # Get best offer with full details including pricing
            offer_info = await asyncio.to_thread(
                self.global_qp.get_best_offer_with_details
            )
            
            if not offer_info:
                raise NoAvailableMachinesError(model)
            
            # Create pod
            pod_uuid = await self.lifecycle.create_pod(
                offer_id=offer_info.offer_id,
                template_uuid=template_uuid,
                disk_size=disk_size,
                model=model,
                mode=mode,
                machine_id=str(offer_info.machine_id) if offer_info.machine_id else None,
            )
            
            if pod_uuid:
                await self._register_pod_locally(
                    pod_uuid, model, mode,
                    str(offer_info.machine_id) if offer_info.machine_id else None,
                    hourly_cost=offer_info.hourly_cost,
                    gpu_type=offer_info.gpu_type,
                    offer_id=offer_info.offer_id,
                )
            
            return pod_uuid
            
        except Exception as e:
            logger.error(f"Failed to create individual pod: {e}")
            return None
    
    async def create_machine_based_pod(
        self,
        model: str,
        mode: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create a machine-based pod with variant fallback.
        
        Tries:
        1. Priority machines
        2. Cached machines
        3. Global fallback
        
        Includes throttling to prevent runaway pod creation:
        - Global semaphore limits concurrent creations
        - Per-model in-flight tracking prevents duplicates
        - Backoff after failures
        """
        # CRITICAL: Check if pod creation is already in-flight for this model
        async with self._lock:
            if self._pod_creation_in_flight.get(model, False):
                logger.warning(
                    f"Pod creation already in-flight for model {model}, skipping duplicate request"
                )
                return None
            
            # Check backoff period (30s after failure)
            last_failure = self._pod_creation_backoff.get(model, 0)
            backoff_remaining = 30 - (time.time() - last_failure)
            if backoff_remaining > 0:
                logger.warning(
                    f"Pod creation for model {model} in backoff period ({backoff_remaining:.1f}s remaining)"
                )
                return None
            
            # Mark as in-flight
            self._pod_creation_in_flight[model] = True
        
        try:
            # Acquire global semaphore to limit concurrent pod creations
            async with self._pod_creation_semaphore:
                return await self._create_machine_based_pod_impl(model, mode)
        finally:
            # Always clear in-flight flag
            async with self._lock:
                self._pod_creation_in_flight[model] = False
    
    async def _create_machine_based_pod_impl(
        self,
        model: str,
        mode: Optional[str] = None,
    ) -> Optional[str]:
        """Internal implementation of pod creation (called with semaphore held)"""
        model_lock = self._get_model_lock(model)
        async with model_lock:
            await self._ensure_global_login()
            
            # Get variant plan
            variant_plan = self.get_variant_plan(model)
            
            # Build candidate machine list from machine_cache only
            # (MACHINE_PRIORITY_LIST was removed to avoid conflicts with blacklist)
            candidates = []
            
            # Get cached machines for this model (sorted by score/success rate)
            # Use base model for cache lookup (aliases share the same cache entry)
            base_model = self.resolve_model_alias(model)
            cached = self.machine_cache.get_cached_machines_for_model(base_model)
            for mid, score in cached:
                mid_str = str(mid)
                # Skip blacklisted machines
                if self.is_machine_bad(mid_str):
                    logger.debug(f"Skipping blacklisted machine {mid_str} from cache")
                    continue
                candidates.append((mid_str, f"cached (score={score})"))
            
            if model != base_model:
                logger.info(
                    f"Candidates for {model} (alias of {base_model}, from cache, blacklist-filtered): "
                    f"{[c[0] for c in candidates]}"
                )
            else:
                logger.info(
                    f"Candidates for {model} (from cache, blacklist-filtered): "
                    f"{[c[0] for c in candidates]}"
                )
            
            # If no cached candidates, we'll rely on global fallback
            
            # Try each variant with its specific GPU requirements
            for variant, allowed_gpus in variant_plan:
                logger.info(
                    f"Trying variant={variant} for model={model}, "
                    f"allowed_gpus={allowed_gpus}"
                )
                
                for mid, desc in candidates:
                    # Skip blacklisted
                    if self.is_machine_bad(mid):
                        continue
                    
                    # Check requirements (use base model for alias resolution)
                    base_model = self.resolve_model_alias(model)
                    req = self.model_requirements.get(base_model, {})
                    min_tflops = req.get("min_tflops", 0)
                    
                    # Get template and disk size
                    template_uuid = self.get_template_for_variant(model, variant)
                    disk_size = self.get_disk_size_for_variant(model, variant)
                    
                    if not template_uuid:
                        continue
                    
                    try:
                        # Get current ACTIVE blacklisted machines (excludes expired)
                        blacklisted = self.blacklist.get_active_blacklist()
                        
                        # Get offer with full details including pricing
                        offer_info = await asyncio.to_thread(
                            self.global_qp.get_best_offer_with_details,
                            int(mid),
                            min_tflops,
                            allowed_gpus,
                            1,  # min_gpu_count
                            blacklisted,  # Pass active blacklist to get_best_offer
                        )
                        
                        if not offer_info:
                            logger.debug(f"No offer available for machine {mid}")
                            continue
                        
                        # Double-check: Verify the returned machine is not blacklisted
                        # (Belt and suspenders approach)
                        final_machine_id = str(offer_info.machine_id or mid)
                        if self.is_machine_bad(final_machine_id):
                            logger.warning(
                                f"Skipping machine {final_machine_id} returned by get_best_offer: "
                                f"machine is blacklisted (double-check caught this)"
                            )
                            continue
                        
                        # Create pod with cost tracking info
                        pod_uuid = await self._try_create_pod_on_machine(
                            final_machine_id,
                            model,
                            mode,
                            variant,
                            offer_info.offer_id,
                            template_uuid,
                            disk_size,
                            hourly_cost=offer_info.hourly_cost,
                            gpu_type=offer_info.gpu_type,
                        )
                        
                        if pod_uuid:
                            return pod_uuid
                            
                    except Exception as e:
                        logger.warning(f"Failed on machine {mid}: {e}")
                        continue
                
                # If we reach here, this variant failed on all cached machines
                logger.warning(
                    f"Variant {variant} (GPUs: {allowed_gpus}) failed on all cached machines. "
                    f"Trying next variant..."
                )
            
            # Global fallback - try each variant's GPUs without preferred machine
            logger.warning(
                "All cached machines exhausted for all variants. "
                "Trying global fallback (searching all available machines)..."
            )
            
            # Get current ACTIVE blacklisted machines (excludes expired)
            blacklisted = self.blacklist.get_active_blacklist()
            logger.info(f"Global fallback with ACTIVE blacklist: {blacklisted}")
            
            # Try each variant with its specific GPU requirements
            for variant, allowed_gpus in variant_plan:
                logger.info(
                    f"Global fallback: trying variant={variant} with GPUs={allowed_gpus}"
                )
                
                try:
                    # Resolve model alias for requirements lookup
                    base_model = self.resolve_model_alias(model)
                    req = self.model_requirements.get(base_model, {})
                    min_tflops = req.get("min_tflops", 0)
                    
                    # Get offer with full details including pricing
                    offer_info = await asyncio.to_thread(
                        self.global_qp.get_best_offer_with_details,
                        None,  # no preferred machine
                        min_tflops,
                        allowed_gpus,  # Use variant-specific GPUs!
                        1,  # min_gpu_count
                        blacklisted,  # Pass blacklist to get_best_offer
                    )
                    
                    if not offer_info:
                        logger.debug(
                            f"Global fallback: no offer for variant={variant}, "
                            f"allowed_gpus={allowed_gpus}"
                        )
                        continue
                    
                    # Double-check: Verify the machine is not blacklisted
                    if offer_info.machine_id and self.is_machine_bad(str(offer_info.machine_id)):
                        logger.error(
                            f"Global fallback returned blacklisted machine {offer_info.machine_id}! "
                            f"This should not happen - blacklist was passed."
                        )
                        continue
                    
                    # Get template and disk size for this variant
                    template_uuid = self.get_template_for_variant(model, variant)
                    disk_size = self.get_disk_size_for_variant(model, variant)
                    
                    if not template_uuid:
                        template_uuid = self.template_uuids.get(base_model) or self.template_uuids.get("default")
                    
                    if template_uuid:
                        logger.info(
                            f"Global fallback: creating pod with variant={variant}, "
                            f"machine={offer_info.machine_id}, template={template_uuid}, "
                            f"hourly_cost=${offer_info.hourly_cost:.3f}"
                        )
                        pod_uuid = await self._try_create_pod_on_machine(
                            str(offer_info.machine_id) if offer_info.machine_id else "unknown",
                            model,
                            mode,
                            variant,  # Use the correct variant!
                            offer_info.offer_id,
                            template_uuid,
                            disk_size,
                            hourly_cost=offer_info.hourly_cost,
                            gpu_type=offer_info.gpu_type,
                        )
                        
                        if pod_uuid:
                            # Record fallback event for autoscaler
                            await self._record_fallback_event(model, str(offer_info.machine_id))
                            return pod_uuid
                            
                except Exception as e:
                    logger.warning(f"Global fallback failed for variant={variant}: {e}")
                    continue
            
            logger.error(
                f"Global fallback: no available machines for model {model}. "
                f"All machines exhausted or blacklisted."
            )
            
            # Log blacklist status for debugging
            blacklisted_machines = [
                f"{mid} (until {datetime.fromtimestamp(until).strftime('%H:%M:%S')})"
                for mid, until in self.blacklist._blacklist.items()
                if self.blacklist.is_blacklisted(mid)
            ]
            if blacklisted_machines:
                logger.warning(f"Currently blacklisted machines: {blacklisted_machines}")
            
            # Set backoff period on failure to prevent rapid retry storms
            async with self._lock:
                self._pod_creation_backoff[model] = time.time()
            
            return None
    
    async def _try_create_pod_on_machine(
        self,
        machine_id: str,
        model: str,
        mode: Optional[str],
        variant: Optional[str],
        offer_id: int,
        template_uuid: str,
        disk_size: int,
        hourly_cost: float = 0.0,
        gpu_type: str = "",
    ) -> Optional[str]:
        """
        Try to create a pod on a specific machine.
        
        Handles:
        - QuickPod API errors (fatal vs transient)
        - Pod startup timeout
        - Machine blacklisting
        - Cost tracking (hourly_cost, gpu_type)
        """
        logger.info(
            f"Attempting pod creation: model={model}, variant={variant}, "
            f"machine={machine_id}, template={template_uuid}, "
            f"hourly_cost=${hourly_cost:.3f}"
        )
        
        # Final safety check: never create pod on blacklisted machine
        if self.is_machine_bad(machine_id):
            logger.warning(
                f"Refusing to create pod on blacklisted machine {machine_id} "
                f"for model {model}"
            )
            return None
        
        try:
            pod_uuid = await self.lifecycle.create_pod(
                offer_id=offer_id,
                template_uuid=template_uuid,
                disk_size=disk_size,
                model=model,
                mode=mode,
                variant=variant,
                machine_id=machine_id,
            )
        except QuickPodFatalError as e:
            # Don't blacklist machine for API errors
            logger.error(f"QuickPod fatal error on {machine_id}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Pod creation failed on {machine_id}: {e}")
            await self.mark_machine_bad(machine_id)
            return None
        
        if not pod_uuid:
            await self.mark_machine_bad(machine_id)
            return None
        
        # Register pod locally with cost tracking info
        await self._register_pod_locally(
            pod_uuid, model, mode, machine_id, variant,
            hourly_cost=hourly_cost,
            gpu_type=gpu_type,
            offer_id=offer_id,
        )
        
        # Wait for pod to be ready
        start = time.time()
        while time.time() - start < self.pod_startup_timeout:
            try:
                if await self.check_pod_ready(pod_uuid):
                    async with self._lock:
                        if pod_uuid in self.individual_pods:
                            self.individual_pods[pod_uuid].status = PodStatus.READY
                    
                    # Mark image as cached
                    await self.machine_cache.mark_image_cached(machine_id, model)
                    await self.mark_machine_good(machine_id)
                    
                    logger.info(f"Pod {pod_uuid} ready on {machine_id}")
                    return pod_uuid
            except Exception as e:
                logger.debug(f"Pod readiness check error: {e}")
            
            await asyncio.sleep(2)
        
        # Timeout - check if worker registered anyway (race condition protection)
        # CRITICAL: Worker must be in "ready" state, not just registered!
        # Workers in "starting" state are still loading models
        if await self._check_worker_registered_for_pod(pod_uuid, require_ready=True):
            logger.info(
                f"Pod {pod_uuid} timed out but worker is READY - keeping alive"
            )
            async with self._lock:
                if pod_uuid in self.individual_pods:
                    self.individual_pods[pod_uuid].status = PodStatus.READY
            await self.machine_cache.mark_image_cached(machine_id, model)
            return pod_uuid
        
        # Check if worker registered but still starting - give it more time
        if await self._check_worker_registered_for_pod(pod_uuid, require_ready=False):
            logger.warning(
                f"Pod {pod_uuid} timed out but worker registered in 'starting' state. "
                f"Keeping alive - models may still be loading. "
                f"Worker will transition to 'ready' when models are loaded."
            )
            # Keep pod alive but mark as STARTING (not READY)
            async with self._lock:
                if pod_uuid in self.individual_pods:
                    self.individual_pods[pod_uuid].status = PodStatus.STARTING
            await self.machine_cache.mark_image_cached(machine_id, model)
            return pod_uuid
        
        # Truly timed out - worker never registered at all, destroy and blacklist
        logger.error(f"Pod {pod_uuid} startup timeout (no worker registration), destroying")
        await self.destroy_individual_pod(pod_uuid, reason="Startup timeout - no worker registration")
        await self.mark_machine_bad(machine_id)
        
        return None
    
    async def _register_pod_locally(
        self,
        pod_uuid: str,
        model: str,
        mode: Optional[str],
        machine_id: Optional[str],
        variant: Optional[str] = None,
        hourly_cost: float = 0.0,
        gpu_type: str = "",
        offer_id: int = 0,
    ):
        """Register a pod in local tracking with cost info"""
        async with self._lock:
            self.individual_pods[pod_uuid] = PodInfo(
                pod_uuid=pod_uuid,
                model=model,
                mode=mode,
                status=PodStatus.STARTING,
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow(),
                machine_id=machine_id,
                variant=variant,
                meta={"variant": variant},
                hourly_cost=hourly_cost,
                gpu_type=gpu_type,
                offer_id=offer_id,
            )
            
            self.individual_pod_managers[pod_uuid] = QuickPodManager(
                self.email,
                self.password,
                pod_uuid,
            )
            
            self._pod_locks[pod_uuid] = asyncio.Lock()
    
    async def _check_worker_registered_for_pod(self, pod_uuid: str, require_ready: bool = False) -> bool:
        """
        Check if any worker has registered for a pod.
        
        Args:
            pod_uuid: Pod UUID to check
            require_ready: If True, worker must be in "ready" state (not just registered)
        
        Returns:
            True if worker is registered (and ready, if require_ready=True) AND pod still exists
        """
        from db import SessionLocal, Worker
        
        # CRITICAL FIX: First check if pod still exists in our tracking
        # This prevents race condition where pod is destroyed but DB still shows worker as READY
        async with self._lock:
            pod_exists = pod_uuid in self.individual_pods
        
        if not pod_exists:
            logger.warning(
                f"Worker check for pod {pod_uuid}: pod NOT in tracking (destroyed?), "
                f"returning False regardless of DB state"
            )
            return False
        
        with SessionLocal() as db:
            if require_ready:
                # Only return True if worker is actually READY (not starting)
                worker = (
                    db.query(Worker)
                    .filter(
                        Worker.container_id == pod_uuid,
                        Worker.status == "ready",  # MUST be ready
                    )
                    .first()
                )
            else:
                worker = (
                    db.query(Worker)
                    .filter(Worker.container_id == pod_uuid)
                    .first()
                )
            
            if worker:
                logger.debug(
                    f"Worker check for pod {pod_uuid}: found worker {worker.worker_id}, "
                    f"status={worker.status}, require_ready={require_ready}"
                )
            
            return worker is not None
    
    # -------------------------------------------------------------------------
    # Pod Destruction
    # -------------------------------------------------------------------------
    
    async def destroy_individual_pod(
        self,
        pod_uuid: str,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Destroy an individual pod.
        
        CRITICAL: Before destroying, recover any RUNNING jobs on this pod
        so users don't see jobs stuck in "processing" state forever.
        
        Also finalizes pod costs and distributes overhead across all jobs that ran on this pod.
        """
        from db import SessionLocal, Worker, Job
        from job_manager import JobState
        import importlib
        
        original_pod_uuid = pod_uuid
        pod_uuid_normalized = normalize_pod_uuid(pod_uuid)
        
        # CRITICAL: Resolve pod_uuid to find it in tracking (may be stored with/without hyphens)
        # Also ensure we have the hyphenated format for QuickPod API calls
        tracked_pod_uuid = None  # The key in individual_pods
        async with self._lock:
            # Direct lookup first
            if pod_uuid in self.individual_pods:
                tracked_pod_uuid = pod_uuid
            else:
                # Try normalized comparison
                for tracked_uuid in self.individual_pods.keys():
                    if normalize_pod_uuid(tracked_uuid) == pod_uuid_normalized:
                        tracked_pod_uuid = tracked_uuid
                        logger.info(
                            f"Resolved pod_uuid '{original_pod_uuid}' to tracked '{tracked_pod_uuid}'"
                        )
                        break
        
        # Use tracked UUID if found, otherwise format the input
        if tracked_pod_uuid:
            pod_uuid = tracked_pod_uuid
        else:
            # Ensure hyphenated format for API
            pod_uuid = format_pod_uuid_with_hyphens(pod_uuid) or pod_uuid
            logger.warning(
                f"Pod {original_pod_uuid} not found in tracking, using formatted: {pod_uuid}"
            )
        
        # Capture pod info (for blacklisting and cost tracking) BEFORE removing from tracking
        machine_id: Optional[str] = None
        pod_created_at: Optional[datetime] = None
        pod_hourly_cost: float = 0.0
        pod_gpu_type: str = ""
        async with self._lock:
            pi = self.individual_pods.get(pod_uuid)
            if pi:
                machine_id = pi.machine_id
                pod_created_at = pi.created_at
                pod_hourly_cost = pi.hourly_cost
                pod_gpu_type = pi.gpu_type
        
        # CRITICAL: Recover any running jobs BEFORE destroying the pod
        # This prevents jobs from being lost when a pod dies
        recovered_jobs: List[Dict[str, Any]] = []
        try:
            with SessionLocal() as db:
                # Find the worker associated with this pod (try both formats)
                worker = db.query(Worker).filter(Worker.container_id == pod_uuid).first()
                if not worker:
                    # Try normalized comparison
                    workers = db.query(Worker).filter(Worker.container_id.isnot(None)).all()
                    for w in workers:
                        if normalize_pod_uuid(w.container_id) == pod_uuid_normalized:
                            worker = w
                            break
                
                if worker:
                    # Find all RUNNING or DISPATCHING jobs on this worker
                    jobs_to_recover = db.query(Job).filter(
                        Job.worker_id == worker.worker_id,
                        Job.status.in_([JobState.RUNNING.value, JobState.DISPATCHING.value]),
                    ).all()
                    
                    recovered_count = 0
                    for job in jobs_to_recover:
                        old_status = job.status
                        job.status = JobState.QUEUED.value
                        job.worker_id = None
                        job.updated_at = datetime.utcnow()
                        recovered_jobs.append(
                            {
                                "job_id": job.job_id,
                                "model": getattr(job, "model", None),
                                "mode": getattr(job, "mode", None),
                            }
                        )
                        
                        if job.error_log is None:
                            job.error_log = []
                        job.error_log.append({
                            "timestamp": datetime.utcnow().isoformat(),
                            "state": "pod_destruction_recovery",
                            "error": f"Job recovered (was {old_status}) after pod {pod_uuid} destroyed: {reason}",
                            "previous_status": old_status,
                        })
                        
                        recovered_count += 1
                        logger.warning(
                            f"Recovered job {job.job_id} (was {old_status}) from destroyed pod {pod_uuid}"
                        )
                    
                    if recovered_count > 0:
                        db.commit()
                        logger.info(
                            f"Recovered {recovered_count} job(s) from pod {pod_uuid} before destruction"
                        )
                    
                    # Mark worker as offline
                    worker.status = "offline"
                    db.commit()
                    
        except Exception as e:
            logger.error(f"Error recovering jobs before destroying pod {pod_uuid}: {e}")
        
        # CRITICAL: Remove pod from local tracking FIRST, before re-dispatching recovered jobs.
        # This ensures any in-flight dispatches or new dispatches won't try to use this dying pod.
        async with self._lock:
            self.individual_pods.pop(pod_uuid, None)
            self.individual_pod_managers.pop(pod_uuid, None)
        
        logger.info(f"Pod {pod_uuid} removed from tracking (recovered jobs will be re-dispatched)")
        
        # CRITICAL: Recovered jobs must be re-enqueued into the in-memory pending queue,
        # otherwise autoscaler/dispatcher won't see them and clients will be stuck in QUEUED forever.
        for r in recovered_jobs:
            try:
                if r.get("job_id") and r.get("model"):
                    await self.add_pending_job(r["job_id"], r["model"])
            except Exception as e:
                logger.warning(f"Failed to add recovered job {r.get('job_id')} to pending queue: {e}")

        # CRITICAL: Trigger background dispatch for recovered jobs (best-effort).
        # Avoid hard import at module import time (circular); import lazily here.
        # NOTE: Pod was already removed from tracking above, so new dispatches won't reuse it.
        if recovered_jobs:
            try:
                routes = importlib.import_module("routes")
                dispatch_fn = getattr(routes, "_dispatch_job_background", None)
                if dispatch_fn:
                    # Add a small delay before re-dispatching to allow old in-flight dispatches
                    # to complete and hit the "pod not in tracking" check. This reduces race conditions.
                    async def delayed_dispatch(job_id: str, model: str, mode: Optional[str]):
                        await asyncio.sleep(2)  # 2 second delay
                        await dispatch_fn(job_id, model, mode)
                    
                    for r in recovered_jobs:
                        if r.get("job_id") and r.get("model"):
                            asyncio.create_task(delayed_dispatch(r["job_id"], r["model"], r.get("mode")))
            except Exception as e:
                logger.warning(f"Failed to trigger dispatch for recovered jobs from pod {pod_uuid}: {e}")
        
        # Destroy via lifecycle manager
        success = await self.lifecycle.destroy_pod(pod_uuid, reason=reason)
        
        # Finalize pod costs and distribute overhead across all jobs that ran on this pod
        if success and pod_created_at and pod_hourly_cost > 0:
            try:
                from cost_service import finalize_pod_costs
                
                pod_destroyed_at = datetime.utcnow()
                
                with SessionLocal() as db:
                    cost_summary = finalize_pod_costs(
                        db=db,
                        pod_uuid=pod_uuid,
                        pod_created_at=pod_created_at,
                        pod_destroyed_at=pod_destroyed_at,
                        gpu_hourly_rate=pod_hourly_cost,
                        gpu_type=pod_gpu_type,
                        distribution_method="proportional",  # Distribute overhead by job runtime
                    )
                    
                    if cost_summary:
                        logger.info(
                            f"Pod {pod_uuid} cost finalized: "
                            f"${cost_summary.total_cost_usd:.6f} total, "
                            f"{cost_summary.jobs_processed} jobs processed"
                        )
            except Exception as e:
                logger.error(f"Failed to finalize pod costs for {pod_uuid}: {e}")

        # If pod died due to infrastructure/network issues, blacklist the machine to avoid immediate reuse.
        # (Do NOT blacklist for normal idle scale-down.)
        if machine_id and reason:
            reason_l = reason.lower()
            if (
                "stale heartbeat" in reason_l
                or "startup timeout" in reason_l
                or "dispatch failure" in reason_l
                or "stuck in starting" in reason_l
                or "stuck running" in reason_l  # Worker stuck during job execution
                or "running for" in reason_l    # Job exceeded running timeout
            ):
                try:
                    await self.mark_machine_bad(str(machine_id), duration=600)
                    logger.warning(
                        f"Machine {machine_id} blacklisted for 10 min after pod {pod_uuid} destruction: {reason}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to blacklist machine {machine_id} after pod {pod_uuid} destruction: {e}")
        
        # Remove pod lock
        self._pod_locks.pop(pod_uuid, None)
        
        return success
    
    # -------------------------------------------------------------------------
    # Autoscaler
    # -------------------------------------------------------------------------
    
    async def _record_fallback_event(self, model: str, machine_id: str):
        """Record a fallback event for autoscaler"""
        now = time.time()
        self.fallback_events.append((now, model, machine_id))
        
        # Clean old events
        window = int(os.getenv("FALLBACK_WINDOW_SECONDS", "600"))
        while self.fallback_events and now - self.fallback_events[0][0] > window:
            self.fallback_events.popleft()
    
    async def autoscaler_loop(self):
        """Background autoscaler task"""
        fallback_threshold = int(os.getenv("FALLBACK_THRESHOLD", "3"))
        load_threshold = int(os.getenv("LOAD_SCALE_THRESHOLD", "15"))
        
        while True:
            try:
                now = time.time()
                
                # Check cooldown
                if now - self.last_autoscale_ts < self.autoscale_cooldown:
                    await asyncio.sleep(5)
                    continue
                
                # Load-based scaling
                for model, jobs in self.pending_jobs.get_all_pending().items():
                    pending_count = len(jobs)
                    
                    if pending_count >= load_threshold:
                        # CRITICAL: Verify pending count against database
                        # The in-memory pending_jobs might be stale
                        from db import SessionLocal
                        from job_manager import JobState
                        try:
                            with SessionLocal() as db:
                                from db import Job
                                actual_pending = db.query(Job).filter(
                                    Job.model == model,
                                    Job.status.in_([JobState.QUEUED.value, JobState.PENDING.value]),
                                ).count()
                                
                                if actual_pending < load_threshold:
                                    # In-memory count is stale - clean it up
                                    logger.warning(
                                        f"Load autoscaler: model={model}, in-memory pending={pending_count}, "
                                        f"but DB shows only {actual_pending} pending - skipping scale up and cleaning cache"
                                    )
                                    # Clear stale entries from pending_jobs
                                    await self.pending_jobs.clear_model(model)
                                    continue
                        except Exception as e:
                            logger.error(f"Error verifying pending count from DB: {e}")
                            continue
                        
                        # Check if any pods exist for this model (READY or STARTING)
                        # Don't create new pod if one is already being created
                        has_pod = any(
                            p.model == model and p.status in [PodStatus.READY, PodStatus.STARTING]
                            for p in self.individual_pods.values()
                        )
                        
                        if not has_pod:
                            logger.info(
                                f"Load autoscaler: model={model}, pending={pending_count}, "
                                f"no existing pods - creating new pod"
                            )
                            self.last_autoscale_ts = now
                            await self._scale_up_for_model(model)
                            break
                        else:
                            logger.debug(
                                f"Load autoscaler: model={model}, pending={pending_count}, "
                                f"but pod already exists - skipping"
                            )
                
                # Fallback frequency scaling
                model_counts: Dict[str, int] = {}
                for ts, model, mid in self.fallback_events:
                    model_counts[model] = model_counts.get(model, 0) + 1
                
                for model, count in model_counts.items():
                    if count >= fallback_threshold:
                        logger.info(
                            f"Fallback autoscaler: model={model}, fallback_count={count}"
                        )
                        self.last_autoscale_ts = now
                        await self._scale_up_for_model(model)
                        break
                
            except Exception:
                logger.exception("Autoscaler error")
            
            await asyncio.sleep(5)
    
    async def _scale_up_for_model(self, model: str):
        """Scale up by creating a new pod for a model"""
        try:
            logger.info(f"Scaling up pod for model={model}")
            pod_uuid = await self.create_machine_based_pod(model)
            
            if pod_uuid:
                logger.info(f"Scaled up: created pod {pod_uuid} for {model}")
            else:
                logger.warning(
                    f"Scale up failed for {model} - no machines available. "
                    f"Will retry after cooldown period."
                )
        except Exception as e:
            logger.exception(f"Scale up error for {model}: {e}")
            # Don't re-raise - autoscaler will retry after cooldown

