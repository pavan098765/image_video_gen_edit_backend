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


def parse_fixed_fleet_gpu_tiers(raw: str) -> Dict[str, int]:
    """
    Parse FIXED_FLEET_GPU_TIERS: pipe-separated preference groups, comma-separated
    machine IDs per group. First group is tried before the second, etc.
    Returns machine_id -> tier index (lower is better).
    """
    result: Dict[str, int] = {}
    raw = (raw or "").strip()
    if not raw:
        return result
    for tier_idx, group in enumerate(raw.split("|")):
        for part in group.split(","):
            mid = part.strip()
            if mid:
                result[mid] = tier_idx
    return result


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

MAX_JOBS_PER_POD = int(os.getenv("MAX_JOBS_PER_POD", "30"))
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
        gpu_provider: Optional[str] = None,
        provider_meta: Optional[Dict[str, Any]] = None,
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
        provider = (gpu_provider or "quickpod").strip().lower()
        if provider and provider not in ("quickpod",):
            logger.info(
                "Destroying %s pod %s: %s",
                provider,
                pod_uuid,
                reason or "no reason given",
            )
            try:
                from gpu_providers import destroy_external

                ok = await asyncio.to_thread(
                    destroy_external, provider, pod_uuid, provider_meta or {}
                )
            except Exception as e:
                logger.error("Failed to destroy %s pod %s: %s", provider, pod_uuid, e)
                ok = False
            async with self._lock:
                self._recently_destroyed[pod_uuid] = time.time()
            return ok

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
    
    def list_all_cached_pairs(self) -> List[Tuple[str, str]]:
        """
        List all (machine_id, model) pairs from machines[*].cached_images.
        Used by the cache refresh service to know what to refresh.
        
        Returns:
            List of (machine_id, model) tuples
        """
        result = []
        machines = self.cache.get("machines", {})
        for mid, info in machines.items():
            cached = info.get("cached_images", {})
            for model in cached:
                result.append((mid, model))
        return result
    
    def get_refresh_blacklisted_streak(self, machine_id: str) -> int:
        """Get consecutive refresh runs this machine was blacklisted (0 if not set)."""
        machines = self.cache.get("machines", {})
        machine = machines.get(str(machine_id), {})
        return machine.get("refresh_blacklisted_streak", 0)
    
    async def increment_refresh_blacklisted_streak(self, machine_id: str) -> int:
        """Increment blacklist streak for a machine. Returns new streak value."""
        async with self._lock:
            machines = self.cache.setdefault("machines", {})
            machine = machines.setdefault(str(machine_id), {"cached_images": {}, "is_priority": False})
            streak = machine.get("refresh_blacklisted_streak", 0) + 1
            machine["refresh_blacklisted_streak"] = streak
            self.cache["meta"]["updated_at"] = datetime.utcnow().isoformat() + "Z"
        await self.save()
        return streak
    
    async def reset_refresh_blacklisted_streak(self, machine_id: str):
        """Reset blacklist streak to 0 when machine was successfully used."""
        async with self._lock:
            machines = self.cache.get("machines", {})
            machine = machines.get(str(machine_id))
            if machine is not None:
                machine["refresh_blacklisted_streak"] = 0
                self.cache["meta"]["updated_at"] = datetime.utcnow().isoformat() + "Z"
        await self.save()
    
    def get_machines_with_streak_ge_3(self) -> List[str]:
        """Get machine IDs that have refresh_blacklisted_streak >= 3 (to be removed)."""
        result = []
        machines = self.cache.get("machines", {})
        for mid, info in machines.items():
            if info.get("refresh_blacklisted_streak", 0) >= 3:
                result.append(mid)
        return result
    
    async def remove_machine(self, machine_id: str):
        """Remove a machine from the cache (e.g. after 3 consecutive blacklisted refresh runs)."""
        async with self._lock:
            machines = self.cache.get("machines", {})
            if str(machine_id) in machines:
                del machines[str(machine_id)]
                self.cache["meta"]["updated_at"] = datetime.utcnow().isoformat() + "Z"
        await self.save()
        logger.info(f"Removed machine {machine_id} from cache (3 consecutive blacklisted refresh runs)")
    
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

    # Full static grouped-model configuration (Phase remediation baseline).
    STATIC_GROUP_DEFINITIONS: Dict[str, List[str]] = {
        "image_aio_4x_utils": [
            "qwen_image_edit_2511_rapid_aio",
            "z_image_turbo",
            "flux_klein",
            "sdxl_t2i",
            "diffusers_outpaint",
            "basic_utils",
            "aurasrv2",
            "video_upscaler",
            "swin2sr",
            "face_enhancer",
            "glm_ocr",
        ],
        "wan_lightx2v": [
            "wan_22_fp8_lightx2v_i2v",
            "wan_22_fp8_lightx2v_t2v",
        ],
        "hunyuan_480p": [
            "hunyan15_480p_t2v_distilled",
            "hunyan15_480p_i2v_step_distilled",
        ],
        "ltx2_3_diffusers": [
            "ltx2_3_t2v",
            "ltx2_3_i2v_distilled",
            "ltx2_3_flf2v",
            "ltx2_3_iv2v_distilled",
            "ltx2_3_cinereal",
        ],
        "minimax_h3_comfy": [
            "minimax_h3",
        ],
        "ltx25_int8_comfy": [
            "ltx25_int8",
            "ltx25_int8_t2v",
            "ltx25_int8_i2v",
            "ltx25_int8_flf2v",
        ],
    }

    # Same ids as worker omni_config.OMNI_SHARED_BAKED_MODELS (fixed-fleet template pick).
    # Must stay in sync with
    # backend/quickpod_individual_worker/worker_multi_common/src/omni_config.py
    # (OMNI_SHARED_BAKED_MODELS). A startup drift check in main.py logs a warning on mismatch.
    FLEET_OMNI_SHARED_BAKE_MODELS: frozenset = frozenset(
        {
            "basic_utils",
            "aurasrv2",
            "swin2sr",
            "video_upscaler",
            "face_enhancer",
            "glm_ocr",
        }
    )
    
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
        self.scheduler_mode = os.getenv("SCHEDULER_MODE", "legacy_dynamic_fleet").strip().lower()
        self.fixed_fleet_enabled = self.scheduler_mode == "fixed_fleet_windowed"
        self.fixed_fleet_max_pods = int(os.getenv("FIXED_FLEET_MAX_PODS", "8"))
        machine_ids_raw = os.getenv("FIXED_FLEET_MACHINE_IDS", "")
        self.fixed_fleet_machine_ids: Set[str] = set(
            x.strip() for x in machine_ids_raw.split(",") if x.strip()
        )
        self.fixed_fleet_gpu_tier_rank: Dict[str, int] = parse_fixed_fleet_gpu_tiers(
            os.getenv("FIXED_FLEET_GPU_TIERS", "")
        )
        if self.fixed_fleet_machine_ids and self.fixed_fleet_gpu_tier_rank:
            max_tier = max(self.fixed_fleet_gpu_tier_rank.values(), default=-1)
            orphan_tier = max_tier + 1
            for mid in self.fixed_fleet_machine_ids:
                if mid not in self.fixed_fleet_gpu_tier_rank:
                    self.fixed_fleet_gpu_tier_rank[mid] = orphan_tier
                    logger.warning(
                        "FIXED_FLEET_MACHINE_IDS id %s not listed in FIXED_FLEET_GPU_TIERS; "
                        "using lowest-preference tier index %s",
                        mid,
                        orphan_tier,
                    )
        strict_raw = os.getenv("FIXED_FLEET_STRICT_MACHINES", "false").lower().strip()
        self.fixed_fleet_strict_machines = strict_raw in ("1", "true", "yes", "on")
        self.pod_max_lifetime_seconds = int(os.getenv("POD_MAX_LIFETIME_SECONDS", "840"))
        self.pod_drain_guard_seconds = int(os.getenv("POD_DRAIN_GUARD_SECONDS", "90"))
        self._fixed_fleet_machine_claims: Set[str] = set()
        self.fixed_fleet_metrics: Dict[str, int] = {
            "slot_claim_denied": 0,
            "drain_guard_reject": 0,
            "rejected_by_ttl": 0,
            "ttl_cutoff_destroyed": 0,
            "admission_existing_feasible": 0,
            "admission_created_new_pod": 0,
            "admission_fallback_legacy": 0,
            "admission_pending_wait_started": 0,
            "admission_pending_wait_success": 0,
            "admission_pending_wait_timeout": 0,
            "admission_pending_wait_error": 0,
            "admission_pending_wait_total_ms": 0,
            "fairness_dispatch_total": 0,
            "pod_create_total": 0,
            "pod_destroy_total": 0,
        }
        self._pod_create_events: deque = deque()
        self._pod_destroy_events: deque = deque()

        # Fixed-fleet windowed scheduler: larger disk + omni/multi QuickPod templates
        self.fleet_pod_disk_gb = max(1, int(os.getenv("FLEET_POD_DISK_GB", "500")))
        
        # Templates
        self.template_uuids = self._load_template_uuids()
        self.template_disk_sizes = self._load_disk_sizes()
        self._fleet_multi_template_uuids = self._load_fleet_multi_template_uuids()
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
        if self.fixed_fleet_enabled:
            n_fleet_templates = sum(
                1 for v in self._fleet_multi_template_uuids.values() if (v or "").strip()
            )
            logger.info(
                "Fixed fleet mode enabled: max_pods=%s (hard slot cap), primary_machine_ids=%s, "
                "strict_machines_only=%s, gpu_tier_groups=%s, offer_fallback=%s, pod_ttl=%ss, "
                "drain_guard=%ss, fleet_disk_gb=%s, fleet_multi_template_models=%s",
                self.fixed_fleet_max_pods,
                sorted(list(self.fixed_fleet_machine_ids)),
                self.fixed_fleet_strict_machines,
                len(self.fixed_fleet_gpu_tier_rank) if self.fixed_fleet_gpu_tier_rank else 0,
                "disabled" if self.fixed_fleet_strict_machines else "non_allowlisted_when_cap_allows",
                self.pod_max_lifetime_seconds,
                self.pod_drain_guard_seconds,
                self.fleet_pod_disk_gb,
                n_fleet_templates,
            )
    
    @staticmethod
    def _env_uuid(*names: str) -> Optional[str]:
        """First non-empty env value among ``names`` (strip whitespace)."""
        for n in names:
            raw = os.getenv(n)
            if raw is None:
                continue
            s = str(raw).strip()
            if s:
                return s
        return None

    def _load_fleet_multi_template_uuids(self) -> Dict[str, Optional[str]]:
        """
        Map base model id -> QuickPod template UUID for omni/multi images.
        Used only when SCHEDULER_MODE=fixed_fleet_windowed (see ``fixed_fleet_enabled``).
        Env names mirror ``config.env`` MULTI_TEMPLATE entries.
        Qwen omni: ``QWEN_IMAGE_EDIT_2511_RAPID_AIO_MULTI_TEMPLATE_UUID`` preferred;
        ``QWEN_IMAGE_EDIT_25011_RAPID_AIO_MULTI_TEMPLATE_UUID`` also accepted.
        """
        return {
            "qwen_image_edit_2511_rapid_aio": self._env_uuid(
                "QWEN_IMAGE_EDIT_2511_RAPID_AIO_MULTI_TEMPLATE_UUID",
                "QWEN_IMAGE_EDIT_25011_RAPID_AIO_MULTI_TEMPLATE_UUID",
            ),
            "flux_klein": self._env_uuid("FLUX_KLEIN_MULTI_TEMPLATE_UUID"),
            "z_image_turbo": self._env_uuid("Z_IMAGE_TURBO_MULTI_TEMPLATE_UUID"),
            "sdxl_t2i": self._env_uuid("SDXL_MULTI_TEMPLATE_UUID"),
            "diffusers_outpaint": self._env_uuid("DIFFUSERS_OUTPAINT_MULTI_TEMPLATE_UUID"),
            "basic_utils": self._env_uuid("BASIC_UTILS_MULTI_TEMPLATE_UUID"),
            "aurasrv2": self._env_uuid("AURASRV2_MULTI_TEMPLATE_UUID"),
            "video_upscaler": self._env_uuid("VIDEO_UPSCALER_MULTI_TEMPLATE_UUID"),
            "swin2sr": self._env_uuid("SWIN2SR_MULTI_TEMPLATE_UUID"),
            "face_enhancer": self._env_uuid("FACE_ENHANCER_MULTI_TEMPLATE_UUID"),
            "glm_ocr": self._env_uuid("GLM_OCR_MULTI_TEMPLATE_UUID"),
            "wan_22_fp8_lightx2v_i2v": self._env_uuid(
                "WAN_22_14B_I2V_LIGHTX2V_MULTI",
                "WAN_22_14B_I2V_LIGHTX2V_MULTI_TEMPLATE_UUID",
            ),
            "wan_22_fp8_lightx2v_t2v": self._env_uuid(
                "WAN_22_14B_T2V_LIGHTX2V_MULTI",
                "WAN_22_14B_T2V_LIGHTX2V_MULTI_TEMPLATE_UUID",
            ),
            "hunyan15_480p_t2v_distilled": self._env_uuid(
                "HUNYUAN_480P_T2V_MULTI_TEMPLATE_UUID",
            ),
            "hunyan15_480p_i2v_step_distilled": self._env_uuid(
                "HUNYUAN_480P_I2V_MULTI_TEMPLATE_UUID",
            ),
            "ltx2_3_t2v": self._env_uuid("LTX2_3_DIFFUSERS_MULTI_TEMPLATE_UUID"),
            "ltx2_3_i2v_distilled": self._env_uuid(
                "LTX2_3_DIFFUSERS_MULTI_TEMPLATE_UUID"
            ),
            "ltx2_3_flf2v": self._env_uuid("LTX2_3_DIFFUSERS_MULTI_TEMPLATE_UUID"),
            "ltx2_3_iv2v_distilled": self._env_uuid(
                "LTX2_3_DIFFUSERS_MULTI_TEMPLATE_UUID"
            ),
        }

    def _load_template_uuids(self) -> Dict[str, Optional[str]]:
        """Load template UUIDs from environment"""
        return {
            
            "sd15": os.getenv("SD15_TEMPLATE_UUID"),
            "sd35": os.getenv("SD35_TEMPLATE_UUID"),
            "sdxl_t2i": os.getenv("SDXL_TEMPLATE_UUID"),
            "sdxl_inpaint": os.getenv("SDXL_TEMPLATE_UUID"),
            "sdxl_i2i": os.getenv("SDXL_TEMPLATE_UUID"),
            
            "diffusers_outpaint": os.getenv("DIFFUSERS_OUTPAINT_TEMPLATE_UUID"),
            
            "qwen_image_edit": os.getenv("QWEN_IMAGE_EDIT_TEMPLATE_UUID"),
            "qwen_image_edit_int4": os.getenv("QWEN_IMAGE_EDIT_INT4_TEMPLATE_UUID"),

            "qwen_image_edit_2509": os.getenv("QWEN_IMAGE_EDIT_2509_TEMPLATE_UUID"),
            "qwen_image_edit_2509_int4": os.getenv("QWEN_IMAGE_EDIT_2509_INT4_TEMPLATE_UUID"),
            "qwen_image_edit_2511_rapid_aio": os.getenv("QWEN_IMAGE_EDIT_2511_RAPID_AIO_TEMPLATE_UUID"),

            "qwen_image": os.getenv("QWEN_IMAGE_TEMPLATE_UUID"),
            "qwen_image_int4": os.getenv("QWEN_IMAGE_INT4_TEMPLATE_UUID"),

            "flux_kontext_dev": os.getenv("FLUX_KONTEXT_DEV_TEMPLATE_UUID"),
            "flux_kontext_dev_int4": os.getenv("FLUX_KONTEXT_DEV_INT4_TEMPLATE_UUID"),
            "flux_klein": os.getenv("FLUX_KLEIN_TEMPLATE_UUID"),

            "z_image_turbo": os.getenv("Z_IMAGE_TURBO_TEMPLATE_UUID"),
            "z_image_turbo_int4": os.getenv("Z_IMAGE_TURBO_INT4_TEMPLATE_UUID"),

            "fastwan_21_1b": os.getenv("FASTWAN_21_1B_TEMPLATE_UUID"),
            "fastwan_22_5b": os.getenv("FASTWAN_22_5B_TEMPLATE_UUID"),
            "wan_22_14b_rapid": os.getenv("WAN_22_14B_RAPID_TEMPLATE_UUID"),
            "wan_22_uint4_lightx2v_i2v": os.getenv("WAN_22_14B_I2V_LIGHTX2V"),
            "wan_22_uint4_lightx2v_t2v": os.getenv("WAN_22_14B_T2V_LIGHTX2V"),
            "wan_22_fp8_lightx2v_i2v": os.getenv("WAN_22_14B_I2V_LIGHTX2V"),
            "wan_22_fp8_lightx2v_t2v": os.getenv("WAN_22_14B_T2V_LIGHTX2V"),
            "video_upscaler": os.getenv("VIDEO_UPSCALER_REALESRGAN_TEMPLATE_UUID"),
            "flashvsr_video_upscaler": os.getenv("FLASHVSR_VIDEO_UPSCALER_TEMPLATE_UUID"),

            "ltx2_video": os.getenv("LTX2_VIDEO_TEMPLATE_UUID"),
            "ltx2_3_t2v": os.getenv("LTX2_3_DIFFUSERS_MULTI_TEMPLATE_UUID"),
            "ltx2_3_i2v_distilled": os.getenv("LTX2_3_DIFFUSERS_MULTI_TEMPLATE_UUID"),
            "ltx2_3_flf2v": os.getenv("LTX2_3_DIFFUSERS_MULTI_TEMPLATE_UUID"),
            "ltx2_3_iv2v_distilled": os.getenv("LTX2_3_DIFFUSERS_MULTI_TEMPLATE_UUID"),
            "ltx2_3_cinereal": os.getenv("LTX2_3_DIFFUSERS_MULTI_TEMPLATE_UUID"),

            "dreamid": os.getenv("DREAMID_TEMPLATE_UUID"),
            "hunyan15_480p_t2v_distilled": os.getenv("HUNYUAN_480P_T2V_TEMPLATE_UUID"),
            "hunyan15_480p_i2v_step_distilled": os.getenv("HUNYUAN_480P_I2V_TEMPLATE_UUID"),
            "glm_ocr": os.getenv("GLM_OCR_TEMPLATE_UUID"),
            "minimax_h3": os.getenv("MINIMAX_H3_COMFY_TEMPLATE_UUID"),
            "ltx25_int8": os.getenv("LTX25_INT8_COMFY_TEMPLATE_UUID"),
            "ltx25_int8_t2v": os.getenv("LTX25_INT8_COMFY_TEMPLATE_UUID"),
            "ltx25_int8_i2v": os.getenv("LTX25_INT8_COMFY_TEMPLATE_UUID"),
            "ltx25_int8_flf2v": os.getenv("LTX25_INT8_COMFY_TEMPLATE_UUID"),

            "aurasrv2": os.getenv("AURASRV2_TEMPLATE_UUID"),
            "realesrgan": os.getenv("REALESRGAN_TEMPLATE_UUID"),
            "swin2sr": os.getenv("SWIN2SR_TEMPLATE_UUID"),
            "face_enhancer": os.getenv("REALESRGAN_TEMPLATE_UUID"),  # Uses same template as realesrgan

            "basic_utils": os.getenv("BASIC_UTILS_TEMPLATE_UUID"),
            "warm_pod": os.getenv("WARM_POD_TEMPLATE_UUID"),
            "default": os.getenv("DEFAULT_TEMPLATE_UUID"),
        }
    
    def _load_disk_sizes(self) -> Dict[str, int]:
        """Load disk sizes for templates"""
        return {
            "sd15": 30,
            "sd35": 40,
            "sdxl_t2i": 40,
            "sdxl_inpaint": 40,
            "sdxl_i2i": 40,
            "flux_klein": 40,
            "diffusers_outpaint": 50,
            "qwen_image": 80,
            "qwen_image_edit": 80,
            "qwen_image_edit_2509": 85,
            "qwen_image_edit_2511_rapid_aio": 85,
            "flux_kontext_dev": 45,
            "z_image_turbo": 80,
            "qwen_image_int4": 80,
            "qwen_image_edit_int4": 80,
            "qwen_image_edit_2509_int4": 85,
            "flux_kontext_dev_int4": 45,
            "z_image_turbo_int4": 80,
            "fastwan_21_1b": 60,
            "fastwan_22_5b": 60,
            "wan_22_14b_rapid": 100,  # 75GB model (60GB transformers + other components)
            "wan_22_uint4_lightx2v_i2v": 70,  # UINT4 quantized model ~40GB + LoRA + post-processing models
            "wan_22_uint4_lightx2v_t2v": 70,  # UINT4 quantized model ~40GB + LoRA + post-processing models
            "wan_22_fp8_lightx2v_i2v": 80,  # FP8 quantized model ~20-28GB + post-processing models (no LoRA needed)
            "wan_22_fp8_lightx2v_t2v": 80,  # FP8 quantized model ~20-28GB + post-processing models (no LoRA needed)
            "video_upscaler": 40,  # RIFE + Real-ESRGAN models ~10GB + workspace
            "flashvsr_video_upscaler": 50,  # FlashVSR-Pro models ~7GB + workspace
            "ltx2_video": 85,  # FP4 model ~20GB + Gemma ~2GB + workspace
            "ltx2_3_t2v": 200,
            "ltx2_3_i2v_distilled": 200,
            "ltx2_3_flf2v": 200,
            "ltx2_3_iv2v_distilled": 200,
            "ltx2_3_cinereal": 200,
            "dreamid": 60,
            "hunyan15_480p_t2v_distilled": 50,
            "hunyan15_480p_i2v_step_distilled": 50,
            "glm_ocr": 40,
            "minimax_h3": 250,
            "ltx25_int8": 120,
            "ltx25_int8_t2v": 120,
            "ltx25_int8_i2v": 120,
            "ltx25_int8_flf2v": 120,
            "aurasrv2": 30,
            "realesrgan": 30,
            "swin2sr": 30,
            "face_enhancer": 30,
            "basic_utils": 30,
        }
    
    def _load_model_requirements(self) -> Dict[str, Any]:
        """Load model requirements from JSON file"""
        path = os.getenv("MODEL_REQUIREMENTS_PATH", "model_requirements.json")
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.warning("model_requirements.json root is not an object; using empty config")
                return {}
            data = self._enrich_and_validate_model_requirements(data)
            logger.info(f"Loaded model requirements for {len(data)} models")
            return data
        except Exception as e:
            logger.warning(f"Failed to load model_requirements.json: {e}")
            return {}

    def _build_static_group_defaults(self, model_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Build full static defaults for group_id/supported_models/eta."""
        defaults: Dict[str, Dict[str, Any]] = {}
        model_set = {m.strip().lower() for m in model_names if isinstance(m, str)}
        base_eta = int(os.getenv("SCHEDULER_DEFAULT_MODEL_ETA_SECONDS", "180"))

        eta_overrides = {
            "qwen_image_edit": 85,
            "qwen_image": 75,
            "qwen_image_edit_2509": 85,
            "qwen_image_edit_2511_rapid_aio": 70,
            "z_image_turbo": 55,
            "flux_klein": 65,
            "sdxl_t2i": 70,
            "sdxl_i2i": 70,
            "sdxl_inpaint": 80,
            "diffusers_outpaint": 95,
            "wan_22_fp8_lightx2v_i2v": 170,
            "wan_22_fp8_lightx2v_t2v": 155,
            "video_upscaler": 220,
            "aurasrv2": 60,
            "realesrgan": 70,
            "realesrgan_anime": 70,
            "realesrgan_face": 75,
            "realesrgan_denoise": 75,
            "swin2sr": 80,
            "dreamid": 145,
            "hunyan15_480p_t2v_distilled": 210,
            "hunyan15_480p_i2v_step_distilled": 220,
            "glm_ocr": 45,
            "face_enhancer": 65,
            "ltx2_video": 180,
            "ltx2_3_t2v": 200,
            "ltx2_3_i2v_distilled": 220,
            "ltx2_3_flf2v": 200,
            "ltx2_3_iv2v_distilled": 220,
            "basic_utils": 40,
        }

        assigned_models: Set[str] = set()
        for group_id, members in self.STATIC_GROUP_DEFINITIONS.items():
            normalized_members = [
                m.strip().lower() for m in members if isinstance(m, str) and m.strip().lower() in model_set
            ]
            if not normalized_members:
                continue
            for model in normalized_members:
                defaults[model] = {
                    "group_id": group_id,
                    "supported_models": list(normalized_members),
                    "eta_seconds_default": int(eta_overrides.get(model, base_eta)),
                }
                assigned_models.add(model)

        # Any model not explicitly grouped becomes a self-group.
        for model in sorted(model_set):
            if model in assigned_models:
                continue
            defaults[model] = {
                "group_id": f"solo_{model}",
                "supported_models": [model],
                "eta_seconds_default": int(eta_overrides.get(model, base_eta)),
            }
        return defaults

    def _enrich_and_validate_model_requirements(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure every model has valid group_id/supported_models/eta_seconds_default.
        Missing/invalid values are backfilled from static defaults.
        """
        normalized_keys = {k.strip().lower(): k for k in data.keys() if isinstance(k, str)}
        defaults = self._build_static_group_defaults(list(normalized_keys.keys()))

        for norm_model, original_key in normalized_keys.items():
            req = data.get(original_key)
            if not isinstance(req, dict):
                req = {}
                data[original_key] = req

            default_cfg = defaults.get(norm_model, {})

            group_id = req.get("group_id")
            if not isinstance(group_id, str) or not group_id.strip():
                req["group_id"] = default_cfg.get("group_id", f"solo_{norm_model}")
            else:
                req["group_id"] = group_id.strip()

            supported = req.get("supported_models")
            normalized_supported: List[str] = []
            if isinstance(supported, list):
                for item in supported:
                    if isinstance(item, str) and item.strip():
                        item_norm = item.strip().lower()
                        if item_norm not in normalized_supported:
                            normalized_supported.append(item_norm)
            if not normalized_supported:
                normalized_supported = list(default_cfg.get("supported_models", [norm_model]))
            if norm_model not in normalized_supported:
                normalized_supported.append(norm_model)
            gid_key = req.get("group_id")
            if isinstance(gid_key, str) and gid_key.strip() in self.STATIC_GROUP_DEFINITIONS:
                grp_members = {
                    m.strip().lower()
                    for m in self.STATIC_GROUP_DEFINITIONS[gid_key.strip()]
                    if isinstance(m, str) and m.strip()
                }
                normalized_supported = sorted(set(normalized_supported) | grp_members)
            req["supported_models"] = normalized_supported

            eta = req.get("eta_seconds_default")
            try:
                eta_int = int(eta)
            except Exception:
                eta_int = 0
            if eta_int <= 0:
                eta_int = int(default_cfg.get("eta_seconds_default", 180))
            req["eta_seconds_default"] = eta_int

            data[original_key] = req

        return data
    
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
    
    def _resolve_any_omni_fleet_template(self) -> str:
        """First non-empty fleet multi template UUID from FLEET_OMNI_TEMPLATE_PRIORITY order."""
        raw = os.getenv("FLEET_OMNI_TEMPLATE_PRIORITY", "").strip()
        if raw:
            keys = [k.strip().lower() for k in raw.split(",") if k.strip()]
        else:
            keys = [
                "flux_klein",
                "z_image_turbo",
                "qwen_image_edit_2511_rapid_aio",
                "sdxl_t2i",
                "diffusers_outpaint",
                "wan_22_fp8_lightx2v_t2v",
                "wan_22_fp8_lightx2v_i2v",
                "hunyan15_480p_t2v_distilled",
                "hunyan15_480p_i2v_step_distilled",
            ]
        for k in keys:
            u = (self._fleet_multi_template_uuids.get(k) or "").strip()
            if u:
                return u
        return ""

    def get_template_for_variant(self, model: str, variant: Optional[str]) -> Optional[str]:
        """Get template UUID for a model/variant combination"""
        # Resolve model alias to base model for lookups
        base_model = self.resolve_model_alias(model)
        if self.fixed_fleet_enabled:
            fleet_t = (self._fleet_multi_template_uuids.get(base_model) or "").strip()
            if base_model in self.FLEET_OMNI_SHARED_BAKE_MODELS:
                any_t = self._resolve_any_omni_fleet_template()
                if any_t:
                    return any_t
                if fleet_t:
                    return fleet_t
                return None
            if fleet_t:
                return fleet_t

        req = self.model_requirements.get(base_model, {})
        
        if variant and "variant_template_key" in req:
            template_key = req["variant_template_key"].get(variant, base_model)
        else:
            template_key = base_model
        
        return self.template_uuids.get(template_key)
    
    def get_disk_size_for_variant(self, model: str, variant: Optional[str]) -> int:
        """Get disk size for a model/variant combination"""
        if self.fixed_fleet_enabled:
            return self.fleet_pod_disk_gb

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
            f"[BLACKLIST] machine {machine_id} for {duration}s due to failure"
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

    def _fixed_fleet_total_slots(self) -> int:
        """
        Hard cap on concurrent non-cache-refresh individual pods in fixed-fleet mode.
        Always FIXED_FLEET_MAX_PODS (not len(allowlist)): every rented machine counts
        toward the same cap (QuickPod offers do not expose reliable GPU identity).
        """
        return max(1, self.fixed_fleet_max_pods)

    def _machine_success_rate(self, machine_id: str) -> float:
        """Rolling success rate from machine_cache (0.0 if unknown)."""
        meta = self.machine_cache.get_machine_metadata(str(machine_id))
        sr = meta.get("success_rate")
        if sr is None:
            return 0.0
        try:
            return float(sr)
        except (TypeError, ValueError):
            return 0.0

    def _fixed_fleet_machine_tier(self, machine_id: str) -> int:
        """Lower tier index = higher preference (from FIXED_FLEET_GPU_TIERS)."""
        if not self.fixed_fleet_gpu_tier_rank:
            return 0
        return int(self.fixed_fleet_gpu_tier_rank.get(str(machine_id), 10**6))

    async def _fixed_fleet_claim_machine_for_creation(self, machine_id: str) -> bool:
        """
        Reserve a machine slot for pod creation in fixed-fleet mode.
        Prevents races from creating >1 pod per machine or exceeding slot count.

        When FIXED_FLEET_STRICT_MACHINES is true, pod creation only targets
        FIXED_FLEET_MACHINE_IDS; otherwise non-allowlisted offers may be used if
        allowlisted machines are exhausted (still under FIXED_FLEET_MAX_PODS).
        """
        if not self.fixed_fleet_enabled:
            return True

        machine_id = str(machine_id)
        async with self._lock:
            active_pods = [
                p for p in self.individual_pods.values()
                if not p.meta.get("is_cache_refresh_pod")
            ]
            active_machine_ids = set(str(p.machine_id) for p in active_pods if p.machine_id)
            total_slots = self._fixed_fleet_total_slots()

            if machine_id in active_machine_ids:
                logger.info(
                    "Fixed fleet: machine %s already has active pod, skipping",
                    machine_id,
                )
                self.fixed_fleet_metrics["slot_claim_denied"] += 1
                return False
            if machine_id in self._fixed_fleet_machine_claims:
                logger.info(
                    "Fixed fleet: machine %s already claimed by in-flight creation, skipping",
                    machine_id,
                )
                self.fixed_fleet_metrics["slot_claim_denied"] += 1
                return False
            if (len(active_pods) + len(self._fixed_fleet_machine_claims)) >= total_slots:
                logger.info(
                    "Fixed fleet: all slots occupied (active=%s claims=%s total=%s), skipping machine=%s",
                    len(active_pods),
                    len(self._fixed_fleet_machine_claims),
                    total_slots,
                    machine_id,
                )
                self.fixed_fleet_metrics["slot_claim_denied"] += 1
                return False

            self._fixed_fleet_machine_claims.add(machine_id)
            return True

    async def _fixed_fleet_release_machine_claim(self, machine_id: str):
        """Release in-flight machine claim."""
        if not self.fixed_fleet_enabled:
            return
        machine_id = str(machine_id)
        async with self._lock:
            self._fixed_fleet_machine_claims.discard(machine_id)

    def _prune_pod_operation_events(self, now: datetime):
        """Prune old create/destroy timestamps used for rolling metrics."""
        max_window = max(
            int(os.getenv("SCHEDULER_SOFT_GOVERNOR_OPS_WINDOW_SECONDS", "3600")),
            int(os.getenv("SCHEDULER_POD_OPS_24H_SECONDS", "86400")),
        )
        cutoff = now - timedelta(seconds=max(60, max_window))
        while self._pod_create_events and self._pod_create_events[0] < cutoff:
            self._pod_create_events.popleft()
        while self._pod_destroy_events and self._pod_destroy_events[0] < cutoff:
            self._pod_destroy_events.popleft()

    async def _record_pod_operation_event(self, op_type: str):
        """Track pod create/destroy operations for rolling risk controls."""
        now = datetime.utcnow()
        async with self._lock:
            self._prune_pod_operation_events(now)
            if op_type == "create":
                self._pod_create_events.append(now)
                self.fixed_fleet_metrics["pod_create_total"] = int(
                    self.fixed_fleet_metrics.get("pod_create_total", 0)
                ) + 1
            elif op_type == "destroy":
                self._pod_destroy_events.append(now)
                self.fixed_fleet_metrics["pod_destroy_total"] = int(
                    self.fixed_fleet_metrics.get("pod_destroy_total", 0)
                ) + 1

    async def get_pod_operation_metrics(self) -> Dict[str, int]:
        """Return rolling pod create/destroy counters."""
        now = datetime.utcnow()
        one_hour_cutoff = now - timedelta(hours=1)
        one_day_cutoff = now - timedelta(hours=24)
        async with self._lock:
            self._prune_pod_operation_events(now)
            creates_1h = sum(1 for ts in self._pod_create_events if ts >= one_hour_cutoff)
            destroys_1h = sum(1 for ts in self._pod_destroy_events if ts >= one_hour_cutoff)
            creates_24h = sum(1 for ts in self._pod_create_events if ts >= one_day_cutoff)
            destroys_24h = sum(1 for ts in self._pod_destroy_events if ts >= one_day_cutoff)
            return {
                "pod_creates_1h": int(creates_1h),
                "pod_destroys_1h": int(destroys_1h),
                "pod_ops_rolling_1h": int(creates_1h + destroys_1h),
                "pod_creates_24h": int(creates_24h),
                "pod_destroys_24h": int(destroys_24h),
                "pod_ops_rolling_24h": int(creates_24h + destroys_24h),
            }

    async def get_soft_governor_state(self) -> Dict[str, Any]:
        """
        Return admission soft-governor state based on rolling pod op counts.
        """
        enabled = os.getenv("SCHEDULER_SOFT_GOVERNOR_ENABLED", "false").lower() == "true"
        window_seconds = max(
            60, int(os.getenv("SCHEDULER_SOFT_GOVERNOR_OPS_WINDOW_SECONDS", "3600"))
        )
        threshold = max(1, int(os.getenv("SCHEDULER_SOFT_GOVERNOR_OPS_THRESHOLD", "80")))
        extra_safety = max(
            0, int(os.getenv("SCHEDULER_SOFT_GOVERNOR_EXTRA_SAFETY_SECONDS", "30"))
        )
        extra_drain_guard = max(
            0, int(os.getenv("SCHEDULER_SOFT_GOVERNOR_EXTRA_DRAIN_GUARD_SECONDS", "30"))
        )
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=window_seconds)
        async with self._lock:
            self._prune_pod_operation_events(now)
            creates_window = sum(1 for ts in self._pod_create_events if ts >= cutoff)
            destroys_window = sum(1 for ts in self._pod_destroy_events if ts >= cutoff)
            ops_window = int(creates_window + destroys_window)
            active = bool(enabled and ops_window >= threshold)
            return {
                "enabled": bool(enabled),
                "active": active,
                "window_seconds": int(window_seconds),
                "threshold": int(threshold),
                "ops_in_window": int(ops_window),
                "pod_creates_in_window": int(creates_window),
                "pod_destroys_in_window": int(destroys_window),
                "extra_safety_seconds": int(extra_safety if active else 0),
                "extra_drain_guard_seconds": int(extra_drain_guard if active else 0),
            }
    
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

            if self.fixed_fleet_enabled:
                age_seconds = (datetime.utcnow() - pi.created_at).total_seconds()
                remaining_seconds = self.pod_max_lifetime_seconds - age_seconds
                if remaining_seconds <= self.pod_drain_guard_seconds:
                    logger.info(
                        "Fixed fleet drain: refusing new job=%s on pod=%s remaining=%.1fs (guard=%ss)",
                        job_id,
                        pod_uuid,
                        remaining_seconds,
                        self.pod_drain_guard_seconds,
                    )
                    self.fixed_fleet_metrics["drain_guard_reject"] += 1
                    self.fixed_fleet_metrics["rejected_by_ttl"] += 1
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
        if (model or "").strip().lower() in {"studio", "studio_engine"}:
            raise ValueError(
                "Studio jobs are pulled by studio_engine on Vultr, not dispatched to QuickPod"
            )
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
                if info.meta.get("is_cache_refresh_pod"):
                    continue
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
        
        # === PHASE 1: Try to find a READY pod to reuse immediately (least-loaded first) ===
        # Collect all eligible READY pods for this model
        eligible_ready_pods = []
        for pod_uuid, info in candidates:
            if info.meta.get("is_cache_refresh_pod"):
                continue
            if info.machine_id and self.is_machine_bad(info.machine_id):
                continue
            if self.lifecycle.is_recently_destroyed(pod_uuid):
                continue
            if info.model == model and info.status == PodStatus.READY:
                eligible_ready_pods.append((pod_uuid, info))
        
        # Sort by queue_size ascending (least-loaded first)
        eligible_ready_pods.sort(key=lambda x: x[1].queue_size)
        
        for pod_uuid, info in eligible_ready_pods:
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
                        # BUG FIX: Before rejecting, check if the heartbeat cache shows
                        # the worker as "ready". The DB may lag behind the in-memory
                        # cache (heartbeat sync runs every 30s). If the worker is
                        # actually ready per heartbeat, update the DB and proceed.
                        worker_actually_ready = False
                        try:
                            from routes import _heartbeat_cache
                            cached = _heartbeat_cache.get(worker.worker_id)
                            if cached:
                                hb_status = cached.get("status")
                                last_hb = cached.get("last_heartbeat")
                                from db import utc_naive_now as _utc_now
                                _now = _utc_now()
                                hb_age = ((_now - last_hb).total_seconds() if last_hb else 9999)
                                if hb_status == "ready" and hb_age < 60:
                                    # Heartbeat confirms worker is ready; trust it over stale DB
                                    worker.status = "ready"
                                    db.commit()
                                    worker_actually_ready = True
                                    logger.info(
                                        f"Pod {pod_uuid}: DB worker was '{worker.status}' but heartbeat "
                                        f"shows 'ready' ({hb_age:.0f}s ago). Updated DB, proceeding with reuse."
                                    )
                        except Exception as e:
                            logger.debug(f"Heartbeat cache check failed for {worker.worker_id}: {e}")

                        if not worker_actually_ready:
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
                    logger.info(f"Reusing READY pod {pod_uuid} for job {job_id} (least-loaded, queue_size={info.queue_size})")
                    return worker.worker_id
                else:
                    # Rollback reservation
                    await self.release_slot_on_pod(pod_uuid, job_id)
        
        # === PHASE 2: Check for STARTING pods for the same model ===
        # If another job already triggered pod creation, wait for it instead of creating duplicate
        starting_pod_uuid = None
        async with self._lock:
            for pod_uuid, info in self.individual_pods.items():
                if info.meta.get("is_cache_refresh_pod"):
                    continue
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
                    if info.meta.get("is_cache_refresh_pod"):
                        continue
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
                            elif worker and worker.status != "ready":
                                # Check heartbeat cache - DB may be stale
                                try:
                                    from routes import _heartbeat_cache
                                    cached = _heartbeat_cache.get(worker.worker_id)
                                    if cached and cached.get("status") == "ready":
                                        last_hb = cached.get("last_heartbeat")
                                        from db import utc_naive_now as _utc_now
                                        hb_age = ((_utc_now() - last_hb).total_seconds() if last_hb else 9999)
                                        if hb_age < 60:
                                            worker.status = "dispatching"
                                            worker.last_active = datetime.utcnow()
                                            db.commit()
                                            logger.info(
                                                f"After timeout, pod {pod_uuid} DB was '{worker.status}' "
                                                f"but heartbeat says ready. Reusing for job {job_id}."
                                            )
                                            return worker.worker_id
                                except Exception:
                                    pass
                                await self.release_slot_on_pod(pod_uuid, job_id)
                            else:
                                await self.release_slot_on_pod(pod_uuid, job_id)
                
                # No READY pods found after timeout, fall through to create new pod
        
        # === PHASE 2.5: All pods full -- skip wait, fall through to create new pod ===
        if busy_pods_for_model:
            logger.info(
                f"All {len(busy_pods_for_model)} pod(s) for model {model} are at capacity, "
                f"creating new pod for job {job_id} (overflow)"
            )
            # Fall through to Phase 3 -- don't wait 120s
        
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
                            if pi.meta.get("is_cache_refresh_pod"):
                                continue
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
                    # CRITICAL FIX: Reserve slot on pod before assigning job
                    # This enforces max_jobs_per_pod limit even for STARTING-pod wait path
                    slot_reserved = await self.reserve_slot_on_pod(pod_uuid, job_id)
                    if not slot_reserved:
                        # Pod is at capacity (max_jobs_per_pod reached)
                        logger.debug(
                            f"Pod {pod_uuid} is at capacity (max_jobs_per_pod reached), "
                            f"continuing wait for slot availability for job {job_id}"
                        )
                        # Continue wait loop - don't assign yet
                        await asyncio.sleep(1)
                        continue
                    
                    # Double-check job state right before assignment
                    job = db.query(Job).filter(Job.job_id == job_id).first()
                    if job:
                        current_state = JobState(job.status)
                        if current_state in [JobState.RUNNING, JobState.COMPLETED, JobState.FAILED]:
                            logger.info(
                                f"Job {job_id} changed to {current_state.value} before assignment, "
                                f"aborting wait for worker registration"
                            )
                            # Release slot since we're not assigning
                            await self.release_slot_on_pod(pod_uuid, job_id)
                            raise JobAlreadyProcessedError(
                                f"Job {job_id} is already in {current_state.value} state"
                            )
                    
                    worker.status = "dispatching"
                    worker.last_active = datetime.utcnow()
                    db.commit()
                    logger.info(
                        f"Worker {worker.worker_id} is READY, assigning job {job_id} (slot reserved)"
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
                    # CRITICAL FIX: Reserve slot before assigning
                    slot_reserved = await self.reserve_slot_on_pod(pod_uuid, job_id)
                    if not slot_reserved:
                        # Pod is at capacity - this shouldn't happen at timeout, but handle gracefully
                        logger.warning(
                            f"Pod {pod_uuid} is at capacity during timeout check for job {job_id}. "
                            f"Will raise timeout error."
                        )
                        # Release any partial state and raise timeout
                        raise WorkerTimeoutError(pod_uuid, timeout)
                    
                    # But first check job state
                    job = db.query(Job).filter(Job.job_id == job_id).first()
                    if job:
                        current_state = JobState(job.status)
                        if current_state in [JobState.RUNNING, JobState.COMPLETED, JobState.FAILED]:
                            # Release slot since we're not assigning
                            await self.release_slot_on_pod(pod_uuid, job_id)
                            raise JobAlreadyProcessedError(
                                f"Job {job_id} is already in {current_state.value} state"
                            )
                    
                    logger.info(
                        f"Worker {final_worker.worker_id} became ready during timeout check, "
                        f"assigning job {job_id} (slot reserved)"
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
                    elif worker and worker.status != "ready":
                        # BUG FIX: Check heartbeat cache before giving up
                        try:
                            from routes import _heartbeat_cache
                            cached = _heartbeat_cache.get(worker.worker_id)
                            if cached and cached.get("status") == "ready":
                                last_hb = cached.get("last_heartbeat")
                                from db import utc_naive_now as _utc_now
                                hb_age = ((_utc_now() - last_hb).total_seconds() if last_hb else 9999)
                                if hb_age < 60:
                                    worker.status = "dispatching"
                                    worker.last_active = datetime.utcnow()
                                    db.commit()
                                    elapsed = time.time() - start
                                    logger.info(
                                        f"Job {job_id} got slot on BUSY pod {pod_uuid} after {elapsed:.1f}s "
                                        f"(DB was '{worker.status}', heartbeat says ready)"
                                    )
                                    return worker.worker_id
                        except Exception:
                            pass
                        # Worker not ready yet, release slot and keep waiting
                        await self.release_slot_on_pod(pod_uuid, job_id)
                        logger.debug(
                            f"Pod {pod_uuid} has slot but worker status is '{worker.status}', "
                            f"waiting for ready state"
                        )
                    else:
                        # No worker found, release slot and keep waiting
                        await self.release_slot_on_pod(pod_uuid, job_id)
            
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
            cache_score_by_mid: Dict[str, float] = {
                str(m): float(s) for m, s in cached
            }
            for mid, score in cached:
                mid_str = str(mid)
                # Skip blacklisted machines
                if self.is_machine_bad(mid_str):
                    logger.debug(f"Skipping blacklisted machine {mid_str} from cache")
                    continue
                candidates.append((mid_str, f"cached (score={score})"))

            if (
                self.fixed_fleet_enabled
                and self.fixed_fleet_strict_machines
                and self.fixed_fleet_machine_ids
            ):
                _n0 = len(candidates)
                candidates = [
                    c for c in candidates if c[0] in self.fixed_fleet_machine_ids
                ]
                if _n0 != len(candidates):
                    logger.info(
                        "Fixed fleet strict: cache candidates filtered %s -> %s",
                        _n0,
                        len(candidates),
                    )

            if self.fixed_fleet_enabled and self.fixed_fleet_machine_ids and candidates:

                def _fleet_candidate_sort_key(item: Tuple[str, str]):
                    mid = item[0]
                    nonfleet = 0 if mid in self.fixed_fleet_machine_ids else 1
                    tier = self._fixed_fleet_machine_tier(mid)
                    sr = self._machine_success_rate(mid)
                    score = cache_score_by_mid.get(mid, 0.0)
                    return (nonfleet, tier, -sr, -score)

                candidates.sort(key=_fleet_candidate_sort_key)
            
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
            
            # Try each variant with its specific GPU requirements
            # For each variant, try ALL available machines of that GPU type (not just cached)
            for variant, allowed_gpus in variant_plan:
                logger.info(
                    f"Trying variant={variant} for model={model}, "
                    f"allowed_gpus={allowed_gpus}"
                )
                
                # Check requirements (use base model for alias resolution)
                base_model = self.resolve_model_alias(model)
                req = self.model_requirements.get(base_model, {})
                min_tflops = req.get("min_tflops", 0)
                
                # Get template and disk size
                template_uuid = self.get_template_for_variant(model, variant)
                disk_size = self.get_disk_size_for_variant(model, variant)
                
                if not template_uuid:
                    logger.warning(f"No template found for variant {variant}, skipping")
                    continue
                
                # Track machines we've tried for this variant to avoid duplicates
                tried_machines = set()
                
                # ---------------------------------------------------------------
                # Fetch all offers ONCE per variant iteration to avoid API spam.
                # We then iterate through each GPU tier in order, trying cached then uncached.
                # ---------------------------------------------------------------
                _all_variant_offers: list = []
                try:
                    _all_variant_offers = await asyncio.to_thread(
                        self.global_qp.get_rentable_machines
                    )
                except Exception as _fetch_err:
                    logger.debug(f"Pre-fetch offers for GPU sorting failed: {_fetch_err}")

                def extract_machine_id(o: dict) -> str:
                    for k in ("machines_id", "machine_id", "machineId"):
                        if k in o and o[k] is not None:
                            try:
                                return str(int(o[k]))
                            except (ValueError, TypeError):
                                pass
                    return ""

                def extract_offer_id(o: dict):
                    return o.get("offer_id") or o.get("id")

                def extract_price(o: dict) -> float:
                    try:
                        return float(o.get("hourly_cost", 1e9))
                    except (ValueError, TypeError):
                        return 1e9

                def has_valid_tflops(o: dict) -> bool:
                    try:
                        offer_tflops = float(o.get("tflops", 0))
                        if isinstance(min_tflops, dict):
                            offer_name = str(o.get("offer_name", "")).upper()
                            required_tflops = float(min_tflops.get("default", 0))
                            for gpu, tflops_val in min_tflops.items():
                                if gpu.upper() in offer_name and gpu.upper() != "DEFAULT":
                                    required_tflops = float(tflops_val)
                                    break
                            return offer_tflops >= required_tflops
                        else:
                            return offer_tflops >= float(min_tflops)
                    except (ValueError, TypeError):
                        return False

                def is_multi_gpu_offer(o: dict) -> bool:
                    name = str(o.get("offer_name", "")).upper()
                    if name.startswith(("2X", "3X", "4X")):
                        return True
                    gpu_indexes = str(o.get("gpu_indexes", "") or "")
                    tokens = [t for t in gpu_indexes.split("|") if t.strip()]
                    return len(tokens) > 1

                def has_allowed_geolocation(o: dict) -> bool:
                    allowed = {"US", "NZ"}
                    machines = o.get("_machines", {})
                    if isinstance(machines, dict):
                        geo = machines.get("geolocation", "")
                        if geo in allowed:
                            return True
                    return o.get("geolocation", "") in allowed

                blacklisted = self.blacklist.get_active_blacklist()
                blacklist_set = set(blacklisted)

                # Iterate through EACH GPU tier specified in this variant's plan
                pod_created_for_variant = False
                
                for target_gpu in (allowed_gpus or []):
                    logger.info(f"Evaluating GPU tier '{target_gpu}' for variant {variant}")
                    
                    # --- 1. Filter ALL offers for THIS specific GPU tier ---
                    eligible_offers_for_gpu = []
                    for o in _all_variant_offers:
                        mid = extract_machine_id(o)
                        if not mid or mid in tried_machines or mid in blacklist_set:
                            continue
                            
                        # Must match the CURRENT target GPU tier
                        if target_gpu.upper() not in str(o.get("offer_name", "")).upper():
                            continue
                            
                        if is_multi_gpu_offer(o): continue
                        if not has_valid_tflops(o): continue
                        
                        # Memory > 30GB
                        try:
                            if float(o.get("memory", 0)) <= 30: continue
                        except (ValueError, TypeError):
                            continue
                            
                        if not has_allowed_geolocation(o): continue
                        
                        eligible_offers_for_gpu.append(o)

                    if (
                        self.fixed_fleet_enabled
                        and self.fixed_fleet_strict_machines
                        and self.fixed_fleet_machine_ids
                    ):
                        eligible_offers_for_gpu = [
                            o
                            for o in eligible_offers_for_gpu
                            if extract_machine_id(o) in self.fixed_fleet_machine_ids
                        ]
                    
                    if not eligible_offers_for_gpu:
                        logger.info(f"No eligible machines found for GPU tier '{target_gpu}'")
                        continue
                        
                    # --- 2. Sort eligible machines ---
                    # Default: cached (by score) then cheapest.
                    # Fixed fleet + allowlist: allowlisted machines first, ordered by GPU tier
                    # (FIXED_FLEET_GPU_TIERS), then machine_cache success_rate, then cache score,
                    # then price. Non-allowlisted offers follow when strict mode is off.
                    cached_mids = [c[0] for c in candidates]
                    fleet_ids = self.fixed_fleet_machine_ids

                    def offer_sort_key(o):
                        mid = extract_machine_id(o)
                        price = extract_price(o)
                        is_cached = mid in cached_mids
                        cache_idx = cached_mids.index(mid) if is_cached else 999999
                        tier = self._fixed_fleet_machine_tier(mid)
                        sr = self._machine_success_rate(mid)
                        if self.fixed_fleet_enabled and fleet_ids:
                            if mid in fleet_ids:
                                sub = 0 if is_cached else 1
                                return (0, tier, -sr, sub, cache_idx, price)
                            if is_cached:
                                return (1, -sr, cache_idx, price, 0)
                            return (2, price, 0, mid)
                        return (0 if is_cached else 1, cache_idx, price, 0)

                    eligible_offers_for_gpu.sort(key=offer_sort_key)
                    
                    logger.info(
                        f"Found {len(eligible_offers_for_gpu)} eligible machines for GPU '{target_gpu}' "
                        f"({sum(1 for o in eligible_offers_for_gpu if extract_machine_id(o) in cached_mids)} cached)"
                    )
                    
                    # --- 3. Try each eligible machine ---
                    for offer in eligible_offers_for_gpu:
                        machine_id_str = extract_machine_id(offer)
                        offer_id = extract_offer_id(offer)
                        if not offer_id: continue
                        
                        # Double-check blacklist (race condition)
                        if self.is_machine_bad(machine_id_str):
                            continue
                            
                        hourly_cost = extract_price(offer)
                        gpu_type = offer.get("gpu_type", offer.get("offer_name", "Unknown"))
                        is_cached = machine_id_str in cached_mids
                        if self.fixed_fleet_enabled and fleet_ids:
                            if machine_id_str in fleet_ids:
                                tier_lbl = "PRIMARY"
                            elif is_cached:
                                tier_lbl = "SECONDARY_CACHE"
                            else:
                                tier_lbl = "COLD_OFFER"
                        else:
                            tier_lbl = "CACHED" if is_cached else "UNCACHED"

                        logger.info(
                            f"Trying {tier_lbl} machine {machine_id_str} "
                            f"for variant {variant} (GPU: {target_gpu}, price=${hourly_cost:.3f}/hr)"
                        )
                        
                        pod_uuid = await self._try_create_pod_on_machine(
                            machine_id_str,
                            model,
                            mode,
                            variant,
                            offer_id,
                            template_uuid,
                            disk_size,
                            hourly_cost=hourly_cost,
                            gpu_type=gpu_type,
                        )
                        
                        if pod_uuid:
                            logger.info(f"Successfully created pod on machine {machine_id_str}")
                            return pod_uuid
                            
                        # If failed, mark as tried and continue
                        tried_machines.add(machine_id_str)
                
                # If we reach here, this variant failed on all available machines of this GPU type
                logger.warning(
                    f"Variant {variant} (GPUs: {allowed_gpus}) failed on all available machines. "
                    f"Trying next variant..."
                )
            
            # Global fallback - only if not strict fixed fleet (strict = allowlisted IDs only)
            skip_global_fallback = (
                self.fixed_fleet_enabled
                and self.fixed_fleet_strict_machines
                and bool(self.fixed_fleet_machine_ids)
            )
            if skip_global_fallback:
                logger.warning(
                    "All variants exhausted allowlisted fixed-fleet machines for this model "
                    "(or no matching GPU-tier offer on those machines). "
                    "FIXED_FLEET_STRICT_MACHINES=true: skipping global fallback."
                )
            else:
                logger.warning(
                    "All variants exhausted all available machines in their GPU tiers. "
                    "Trying global fallback (searching all available machines across all GPU types)..."
                )

                # Get current ACTIVE blacklisted machines (excludes expired)
                blacklisted = self.blacklist.get_active_blacklist()
                logger.info(f"Global fallback with ACTIVE blacklist: {blacklisted}")

                # Try each variant with its specific GPU requirements
                for variant, allowed_gpus in variant_plan:
                    logger.info(
                        f"Global fallback: trying variant={variant} with GPUs={allowed_gpus}"
                    )

                    # First, try with all allowed GPUs together
                    variant_pod_created = False
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

                        if offer_info and offer_info.machine_id:
                            # Double-check: Verify the machine is not blacklisted
                            if not self.is_machine_bad(str(offer_info.machine_id)):
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
                                        str(offer_info.machine_id),
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
                                        variant_pod_created = True
                                        # Update machine cache with newly found machine
                                        await self.machine_cache.mark_image_cached(
                                            str(offer_info.machine_id), base_model
                                        )
                                        logger.info(
                                            f"Added machine {offer_info.machine_id} to cache for model {base_model}"
                                        )

                                        # Record fallback event for autoscaler
                                        await self._record_fallback_event(model, str(offer_info.machine_id))
                                        return pod_uuid
                            else:
                                logger.warning(
                                    f"Global fallback returned blacklisted machine {offer_info.machine_id}, "
                                    f"trying individual GPU types..."
                                )

                        # If variant-level search failed, try each GPU type individually
                        if not variant_pod_created:
                            logger.info(
                                f"Global fallback: variant-level search failed for {variant}, "
                                f"trying individual GPU types: {allowed_gpus}"
                            )

                            for gpu_type in allowed_gpus:
                                try:
                                    logger.info(
                                        f"Global fallback: trying individual GPU type {gpu_type} "
                                        f"for variant={variant}"
                                    )

                                    # Try with single GPU type
                                    offer_info = await asyncio.to_thread(
                                        self.global_qp.get_best_offer_with_details,
                                        None,  # no preferred machine
                                        min_tflops,
                                        [gpu_type],  # Try single GPU type
                                        1,  # min_gpu_count
                                        blacklisted,  # Pass blacklist
                                    )

                                    if offer_info and offer_info.machine_id:
                                        # Double-check: Verify the machine is not blacklisted
                                        if self.is_machine_bad(str(offer_info.machine_id)):
                                            logger.debug(
                                                f"Skipping blacklisted machine {offer_info.machine_id} "
                                                f"for GPU type {gpu_type}"
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
                                                f"GPU={gpu_type}, machine={offer_info.machine_id}, "
                                                f"template={template_uuid}, hourly_cost=${offer_info.hourly_cost:.3f}"
                                            )
                                            pod_uuid = await self._try_create_pod_on_machine(
                                                str(offer_info.machine_id),
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
                                                # Update machine cache with newly found machine
                                                await self.machine_cache.mark_image_cached(
                                                    str(offer_info.machine_id), base_model
                                                )
                                                logger.info(
                                                    f"Added machine {offer_info.machine_id} to cache for model {base_model} "
                                                    f"(found via GPU type {gpu_type})"
                                                )

                                                # Record fallback event for autoscaler
                                                await self._record_fallback_event(model, str(offer_info.machine_id))
                                                return pod_uuid

                                except Exception as e:
                                    logger.warning(
                                        f"Global fallback failed for GPU type {gpu_type} "
                                        f"in variant {variant}: {e}"
                                    )
                                    continue

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

            if self._should_try_external_providers():
                logger.warning(
                    "Quickpod exhausted for model %s; searching RunPod/Vast/Salad",
                    model,
                )
                external_uuid = await self._create_external_provider_pod(model, mode)
                if external_uuid:
                    return external_uuid
            
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

        claimed_for_creation = await self._fixed_fleet_claim_machine_for_creation(machine_id)
        if not claimed_for_creation:
            return None

        try:
            base_model_cache = self.resolve_model_alias(model)

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

                        # Mark image as cached (base model so aliases share machine_cache)
                        await self.machine_cache.mark_image_cached(machine_id, base_model_cache)
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
                await self.machine_cache.mark_image_cached(machine_id, base_model_cache)
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
                await self.machine_cache.mark_image_cached(machine_id, base_model_cache)
                return pod_uuid

            # Truly timed out - worker never registered at all, destroy and blacklist
            logger.error(f"Pod {pod_uuid} startup timeout (no worker registration), destroying")
            await self.destroy_individual_pod(pod_uuid, reason="Startup timeout - no worker registration")
            await self.mark_machine_bad(machine_id)

            return None
        finally:
            await self._fixed_fleet_release_machine_claim(machine_id)
    
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
        is_cache_refresh_pod: bool = False,
        gpu_provider: str = "quickpod",
        public_url: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ):
        """Register a pod in local tracking with cost info"""
        provider = (gpu_provider or "quickpod").strip().lower() or "quickpod"
        meta: Dict[str, Any] = {"variant": variant, "gpu_provider": provider}
        if public_url:
            meta["public_url"] = public_url
        if extra_meta:
            meta.update(extra_meta)
            meta["gpu_provider"] = provider
        if is_cache_refresh_pod:
            meta["is_cache_refresh_pod"] = True
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
                meta=meta,
                hourly_cost=hourly_cost,
                gpu_type=gpu_type,
                offer_id=offer_id,
            )

            if provider in ("", "quickpod"):
                self.individual_pod_managers[pod_uuid] = QuickPodManager(
                    self.email,
                    self.password,
                    pod_uuid,
                )
            
            self._pod_locks[pod_uuid] = asyncio.Lock()
        if not is_cache_refresh_pod:
            await self._record_pod_operation_event("create")
    
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

    def _should_try_external_providers(self) -> bool:
        if os.getenv("GPU_PROVIDERS_ENABLED", "true").strip().lower() in ("0", "false", "off", "no"):
            return False
        try:
            from gpu_providers import any_external_configured

            return any_external_configured()
        except Exception:
            return False

    async def _apply_public_url(self, pod_uuid: str, url: str) -> None:
        if not url:
            return
        async with self._lock:
            pi = self.individual_pods.get(pod_uuid)
            if pi:
                pi.meta["public_url"] = url
        from db import SessionLocal, Worker

        with SessionLocal() as db:
            worker = db.query(Worker).filter(Worker.container_id == pod_uuid).first()
            if worker:
                worker.endpoint = url
                info = dict(worker.info or {})
                if pi:
                    info["gpu_provider"] = (pi.meta or {}).get("gpu_provider") or info.get("gpu_provider")
                worker.info = info
                db.commit()

    async def _create_external_provider_pod(
        self,
        model: str,
        mode: Optional[str] = None,
    ) -> Optional[str]:
        """Create a worker on RunPod, Vast, or Salad after Quickpod has no machines."""
        from gpu_providers import create_on_providers, refresh_public_url
        from gpu_providers.spec import build_worker_spec

        variant_plan = self.get_variant_plan(model)
        for variant, allowed_gpus in variant_plan:
            template_uuid = self.get_template_for_variant(model, variant)
            disk_size = self.get_disk_size_for_variant(model, variant)
            spec = build_worker_spec(model, variant, template_uuid, disk_size, allowed_gpus)
            if not spec:
                logger.warning(
                    "No Docker image in templates.json for model=%s variant=%s",
                    model,
                    variant,
                )
                continue
            created = await asyncio.to_thread(create_on_providers, spec)
            if not created:
                continue
            extra_meta = dict(created.meta or {})
            extra_meta["public_url"] = created.public_url
            extra_meta["gpu_provider"] = created.provider
            await self._register_pod_locally(
                created.pod_id,
                model,
                mode,
                created.machine_id,
                variant,
                hourly_cost=created.hourly_cost,
                gpu_type=created.gpu_type,
                offer_id=0,
                gpu_provider=created.provider,
                public_url=created.public_url,
                extra_meta=extra_meta,
            )
            start = time.time()
            while time.time() - start < self.pod_startup_timeout:
                async with self._lock:
                    pi = self.individual_pods.get(created.pod_id)
                    meta = dict(pi.meta or {}) if pi else {}
                if not meta.get("public_url"):
                    url = await asyncio.to_thread(
                        refresh_public_url, created.provider, meta
                    )
                    if url:
                        await self._apply_public_url(created.pod_id, url)
                try:
                    if await self.check_pod_ready(created.pod_id):
                        async with self._lock:
                            if created.pod_id in self.individual_pods:
                                self.individual_pods[created.pod_id].status = PodStatus.READY
                        logger.info(
                            "External %s pod %s ready for model=%s",
                            created.provider,
                            created.pod_id,
                            model,
                        )
                        return created.pod_id
                except Exception as e:
                    logger.debug("External pod ready check: %s", e)
                await asyncio.sleep(2)

            if await self._check_worker_registered_for_pod(created.pod_id, require_ready=True):
                async with self._lock:
                    if created.pod_id in self.individual_pods:
                        self.individual_pods[created.pod_id].status = PodStatus.READY
                return created.pod_id
            if await self._check_worker_registered_for_pod(created.pod_id, require_ready=False):
                logger.warning(
                    "External %s pod %s registered but still starting; keeping alive",
                    created.provider,
                    created.pod_id,
                )
                async with self._lock:
                    if created.pod_id in self.individual_pods:
                        self.individual_pods[created.pod_id].status = PodStatus.STARTING
                return created.pod_id

            logger.error(
                "External %s pod %s startup timeout, destroying",
                created.provider,
                created.pod_id,
            )
            await self.destroy_individual_pod(
                created.pod_id,
                reason="Startup timeout - no worker registration",
            )
        return None

    async def maybe_release_external_pod_after_job(
        self,
        pod_uuid: str,
        worker_id: Optional[str] = None,
    ) -> None:
        """Destroy RunPod/Vast/Salad pods after a job unless personal leftover exists."""
        if not pod_uuid:
            return
        provider = "quickpod"
        async with self._lock:
            pi = self.individual_pods.get(pod_uuid)
            if not pi:
                for tracked, info in self.individual_pods.items():
                    if normalize_pod_uuid(tracked) == normalize_pod_uuid(pod_uuid):
                        pod_uuid = tracked
                        pi = info
                        break
            if pi:
                provider = str((pi.meta or {}).get("gpu_provider") or "quickpod")
        if provider in ("", "quickpod"):
            return

        from db import SessionLocal, Worker, Job
        from job_manager import JobState
        from gpu_providers.ttl import should_keep_external_after_job

        db = SessionLocal()
        keep = False
        try:
            worker = None
            if worker_id:
                worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
            if not worker:
                worker = db.query(Worker).filter(Worker.container_id == pod_uuid).first()
            if worker:
                busy = (
                    db.query(Job)
                    .filter(
                        Job.worker_id == worker.worker_id,
                        Job.status.in_(
                            [
                                JobState.QUEUED.value,
                                JobState.RUNNING.value,
                                JobState.DISPATCHING.value,
                            ]
                        ),
                    )
                    .count()
                )
                if busy > 0:
                    return
                keep = should_keep_external_after_job(worker)
            if keep:
                logger.info(
                    "Keeping %s pod %s for personal leftover (15 min TTL)",
                    provider,
                    pod_uuid,
                )
                return
            logger.info(
                "Closing %s pod %s after job (no personal leftover)",
                provider,
                pod_uuid,
            )
        finally:
            db.close()
        await self.destroy_individual_pod(
            pod_uuid,
            reason="External provider job done, no personal leftover",
        )
    
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
        is_cache_refresh_pod = False
        gpu_provider = "quickpod"
        provider_meta: Dict[str, Any] = {}
        async with self._lock:
            pi = self.individual_pods.get(pod_uuid)
            if pi:
                machine_id = pi.machine_id
                pod_created_at = pi.created_at
                pod_hourly_cost = pi.hourly_cost
                pod_gpu_type = pi.gpu_type
                is_cache_refresh_pod = bool(pi.meta.get("is_cache_refresh_pod"))
                gpu_provider = str((pi.meta or {}).get("gpu_provider") or "quickpod")
                provider_meta = dict(pi.meta or {})
        
        # CRITICAL: Recover any running jobs BEFORE destroying the pod
        # This prevents jobs from being lost when a pod dies
        recovered_jobs: List[Dict[str, Any]] = []
        failed_jobs_to_remove: List[Tuple[str, str]] = []
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
                    # Find all QUEUED, RUNNING or DISPATCHING jobs on this worker
                    jobs_to_recover = db.query(Job).filter(
                        Job.worker_id == worker.worker_id,
                        Job.status.in_([JobState.QUEUED.value, JobState.RUNNING.value, JobState.DISPATCHING.value]),
                    ).order_by(Job.created_at.asc()).all()
                    
                    max_retries_default = int(os.getenv("JOB_MAX_RETRIES", "3"))
                    recovered_count = 0
                    failed_count = 0
                    for job in jobs_to_recover:
                        # RACE FIX: Re-read job from DB before modifying. job_complete may have
                        # just committed COMPLETED for this job; do not overwrite with QUEUED/FAILED.
                        db.refresh(job)
                        if job.status in (JobState.COMPLETED.value, JobState.FAILED.value):
                            logger.info(
                                f"Job {job.job_id} already in terminal state '{job.status}', "
                                f"skipping recovery (job_complete likely committed concurrently)"
                            )
                            continue

                        old_status = job.status
                        retry_count = job.retry_count or 0
                        max_retries = getattr(job, "max_retries", None) or max_retries_default

                        if retry_count >= max_retries:
                            # Exceeded retry limit - mark FAILED to stop infinite create/destroy loop
                            job.status = JobState.FAILED.value
                            job.worker_id = None
                            job.updated_at = datetime.utcnow()
                            job.error = (
                                f"Pod destroyed: {reason}. Max retries ({max_retries}) exceeded."
                            )
                            if job.error_log is None:
                                job.error_log = []
                            job.error_log.append({
                                "timestamp": datetime.utcnow().isoformat(),
                                "state": "pod_destruction_max_retries",
                                "error": f"Job marked FAILED (was {old_status}) - retry {retry_count} >= max {max_retries}",
                                "previous_status": old_status,
                            })
                            failed_count += 1
                            failed_jobs_to_remove.append(
                                (job.job_id, getattr(job, "model", "") or "")
                            )
                            logger.error(
                                f"Job {job.job_id} exceeded max retries ({retry_count}/{max_retries}) "
                                f"after pod {pod_uuid} destroyed. Marking FAILED to stop retry loop."
                            )
                        else:
                            # Under limit - recover and requeue
                            job.status = JobState.QUEUED.value
                            job.worker_id = None
                            job.started_at = None  # Clear stale started_at from previous run
                            job.updated_at = datetime.utcnow()
                            job.retry_count = retry_count + 1
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
                                "retry_count": job.retry_count,
                            })
                            recovered_count += 1
                            logger.warning(
                                f"Recovered job {job.job_id} (was {old_status}, retry {job.retry_count}/{max_retries}) "
                                f"from destroyed pod {pod_uuid}"
                            )
                    
                    if recovered_count > 0 or failed_count > 0:
                        db.commit()
                        if recovered_count > 0:
                            logger.info(
                                f"Recovered {recovered_count} job(s) from pod {pod_uuid} before destruction"
                            )
                        if failed_count > 0:
                            logger.info(
                                f"Marked {failed_count} job(s) as FAILED (max retries exceeded)"
                            )
                    
                    # Mark worker as offline
                    worker.status = "offline"
                    if isinstance(worker.info, dict) and worker.info.get("gpu_provider"):
                        gpu_provider = str(worker.info.get("gpu_provider") or gpu_provider)
                    db.commit()
                    
        except Exception as e:
            logger.error(f"Error recovering jobs before destroying pod {pod_uuid}: {e}")
        finally:
            # Remove failed jobs from pending queue (best-effort)
            for job_id, model in failed_jobs_to_remove:
                if job_id and model:
                    try:
                        await self.remove_pending_job(job_id, model)
                    except Exception as e:
                        logger.debug(f"Could not remove failed job {job_id} from pending: {e}")
        
        # BUG FIX: Mark pod as STOPPING first so dispatchers skip it, then call the
        # API destroy BEFORE removing from tracking.  Previously the pod was removed
        # from tracking first, and the API call happened later -- if the API call
        # failed the pod became a zombie (still running, but invisible to our system).
        async with self._lock:
            pi = self.individual_pods.get(pod_uuid)
            if pi:
                pi.status = PodStatus.STOPPING
        
        # Destroy via lifecycle manager (API call) BEFORE removing from tracking
        success = await self.lifecycle.destroy_pod(
            pod_uuid,
            reason=reason,
            gpu_provider=gpu_provider,
            provider_meta=provider_meta,
        )
        
        if not success:
            logger.error(
                f"API destroy call failed for pod {pod_uuid}. "
                f"Pod may still be running (zombie). Will still remove from tracking."
            )
        
        # NOW remove from tracking (after API destroy has been attempted)
        async with self._lock:
            self.individual_pods.pop(pod_uuid, None)
            self.individual_pod_managers.pop(pod_uuid, None)
        if success and not is_cache_refresh_pod:
            await self._record_pod_operation_event("destroy")
        
        logger.info(f"Pod {pod_uuid} removed from tracking (destroy success={success})")
        
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
        # Do NOT blacklist for "stuck running" / "running for" so recovered jobs can be re-dispatched to a new pod.
        if machine_id and reason and (gpu_provider in ("", "quickpod")):
            reason_l = reason.lower()
            if (
                "stale heartbeat" in reason_l
                or "startup timeout" in reason_l
                or "dispatch failure" in reason_l
                or "stuck in starting" in reason_l
                or "oom" in reason_l
                or "out of memory" in reason_l
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
                        
                        # Check if any pods exist with available capacity for this model
                        model_limit = self.model_requirements.get(model, {}).get(
                            "max_jobs_per_pod", MAX_JOBS_PER_POD
                        )
                        has_available_pod = any(
                            p.model == model
                            and p.status in [PodStatus.READY, PodStatus.STARTING]
                            and p.queue_size < model_limit
                            for p in self.individual_pods.values()
                        )
                        
                        if not has_available_pod:
                            logger.info(
                                f"Load autoscaler: model={model}, pending={pending_count}, "
                                f"no pods with available capacity - creating new pod"
                            )
                            self.last_autoscale_ts = now
                            await self._scale_up_for_model(model)
                            break
                        else:
                            logger.debug(
                                f"Load autoscaler: model={model}, pending={pending_count}, "
                                f"pod with capacity exists - skipping"
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

    # -------------------------------------------------------------------------
    # Cache Refresh Service
    # -------------------------------------------------------------------------

    async def run_cache_refresh(self) -> None:
        """
        Periodic cache refresh: start pods for each (machine_id, model) in cache,
        wait for worker registration, keep alive briefly, then destroy.
        Purpose: ensure Docker images stay cached on machines.
        Runs standalone (cron) and does not interfere with job-driven pod management.
        """
        refresh_registration_min = int(os.getenv("REFRESH_REGISTRATION_TIMEOUT_MINUTES", "8"))
        refresh_keep_alive_min = int(os.getenv("REFRESH_POD_KEEP_ALIVE_MINUTES", "5"))
        max_items = int(os.getenv("REFRESH_MAX_ITEMS_PER_RUN", "50"))
        registration_timeout_sec = refresh_registration_min * 60

        pairs = self.machine_cache.list_all_cached_pairs()
        fleet_allow = self.fixed_fleet_machine_ids
        if fleet_allow:
            before = len(pairs)
            pairs = [
                (mid, model)
                for mid, model in pairs
                if str(mid).strip() in fleet_allow
            ]
            logger.info(
                "Cache refresh: FIXED_FLEET_MACHINE_IDS allowlist (%s ids) — "
                "%s -> %s (machine, model) pairs",
                len(fleet_allow),
                before,
                len(pairs),
            )
            if not pairs and before:
                logger.warning(
                    "Cache refresh: no (machine, model) pairs left after fleet allowlist filter; "
                    "machine_cache may only list non-fleet machines"
                )
        pairs = pairs[:max_items]
        logger.info(f"Cache refresh: processing {len(pairs)} (machine, model) pairs (max {max_items})")

        blacklisted_machines = self.blacklist.get_active_blacklist()

        # Login once before the loop; if auth fails we exit immediately instead of retrying per pair
        await self._ensure_global_login()

        for machine_id, model in pairs:
            try:
                if self.is_machine_bad(machine_id):
                    streak = await self.machine_cache.increment_refresh_blacklisted_streak(machine_id)
                    logger.info(f"Cache refresh: skip blacklisted machine {machine_id}, streak={streak}")
                    continue

                template_uuid = self.get_template_for_variant(model, None)
                if not template_uuid:
                    logger.debug(f"Cache refresh: skip {model} (no template)")
                    continue

                disk_size = self.get_disk_size_for_variant(model, None)
                base_model = self.resolve_model_alias(model)
                req = self.model_requirements.get(base_model, {})
                min_tflops = req.get("min_tflops", 0)
                allowed_gpu_names = req.get("allowed_gpu_names")

                offer_info = self.global_qp.get_offer_for_machine_only(
                    machine_id=int(machine_id),
                    min_tflops=min_tflops,
                    allowed_gpu_names=allowed_gpu_names,
                    blacklisted_machines=blacklisted_machines,
                )
                if not offer_info:
                    logger.warning(f"Cache refresh: no offer for machine {machine_id}, skip")
                    continue

                pod_uuid = await self.lifecycle.create_pod(
                    offer_id=offer_info.offer_id,
                    template_uuid=template_uuid,
                    disk_size=disk_size,
                    model=model,
                    mode=None,
                    variant=None,
                    machine_id=machine_id,
                )
                if not pod_uuid:
                    logger.warning(f"Cache refresh: create pod failed for {model} on {machine_id}")
                    continue

                await self._register_pod_locally(
                    pod_uuid=pod_uuid,
                    model=model,
                    mode=None,
                    machine_id=machine_id,
                    variant=None,
                    hourly_cost=offer_info.hourly_cost,
                    gpu_type=offer_info.gpu_type,
                    offer_id=offer_info.offer_id,
                    is_cache_refresh_pod=True,
                )

                start = time.time()
                registered = False
                while time.time() - start < registration_timeout_sec:
                    if await self._check_worker_registered_for_pod(pod_uuid, require_ready=False):
                        registered = True
                        break
                    await asyncio.sleep(5)

                if registered:
                    await self.machine_cache.mark_image_cached(machine_id, model)
                    await self.machine_cache.reset_refresh_blacklisted_streak(machine_id)
                    logger.info(f"Cache refresh: {model} on {machine_id} registered, keeping alive {refresh_keep_alive_min} min")
                else:
                    logger.info(
                        f"Cache refresh: {model} on {machine_id} did not register in time "
                        f"(download in progress?), keeping alive {refresh_keep_alive_min} min"
                    )

                await asyncio.sleep(refresh_keep_alive_min * 60)
                await self.destroy_individual_pod(pod_uuid, reason="Cache refresh keep-alive done")

            except Exception as e:
                logger.exception(f"Cache refresh: error for {model} on {machine_id}: {e}")

        to_remove = self.machine_cache.get_machines_with_streak_ge_3()
        for mid in to_remove:
            await self.machine_cache.remove_machine(mid)
            logger.info(f"Cache refresh: removed machine {mid} from cache (3 consecutive blacklisted runs)")

        await self.machine_cache.save()
        logger.info("Cache refresh: completed")

    async def get_fixed_fleet_snapshot(self) -> Dict[str, Any]:
        """Runtime snapshot for observability/debugging."""
        from datetime import datetime

        now = datetime.utcnow()
        op_metrics = await self.get_pod_operation_metrics()
        soft_governor_state = await self.get_soft_governor_state()
        async with self._lock:
            pods = []
            for pod_uuid, pi in self.individual_pods.items():
                if pi.meta.get("is_cache_refresh_pod"):
                    continue
                age_seconds = max(0, int((now - pi.created_at).total_seconds()))
                remaining_seconds = max(0, int(self.pod_max_lifetime_seconds - age_seconds))
                pods.append(
                    {
                        "pod_uuid": pod_uuid,
                        "model": pi.model,
                        "machine_id": pi.machine_id,
                        "status": str(getattr(pi.status, "value", pi.status)),
                        "queue_size": int(pi.queue_size),
                        "age_seconds": age_seconds,
                        "remaining_seconds": remaining_seconds,
                        "drain_mode": bool(pi.meta.get("drain_mode")),
                    }
                )
            pods.sort(key=lambda x: x["remaining_seconds"])
            metrics_snapshot = dict(self.fixed_fleet_metrics)
            pending_success = int(metrics_snapshot.get("admission_pending_wait_success", 0))
            pending_timeout = int(metrics_snapshot.get("admission_pending_wait_timeout", 0))
            pending_outcomes = max(1, pending_success + pending_timeout)
            pending_total_ms = int(metrics_snapshot.get("admission_pending_wait_total_ms", 0))
            metrics_snapshot["admission_pending_wait_avg_ms"] = int(
                pending_total_ms / pending_outcomes
            )

            return {
                "enabled": bool(self.fixed_fleet_enabled),
                "scheduler_mode": self.scheduler_mode,
                "max_pods": self.fixed_fleet_max_pods,
                "allowlisted_machine_ids": sorted(list(self.fixed_fleet_machine_ids)),
                "strict_machines_only": bool(self.fixed_fleet_strict_machines),
                "gpu_tier_by_machine_id": dict(
                    sorted(self.fixed_fleet_gpu_tier_rank.items(), key=lambda kv: (kv[1], kv[0]))
                ),
                "ttl_seconds": self.pod_max_lifetime_seconds,
                "drain_guard_seconds": self.pod_drain_guard_seconds,
                "active_pods": pods,
                "active_pod_count": len(pods),
                "inflight_machine_claims": sorted(list(self._fixed_fleet_machine_claims)),
                "metrics": metrics_snapshot,
                "pod_operation_metrics": op_metrics,
                "soft_governor": soft_governor_state,
                "claim_logic": {
                    "one_active_pod_per_machine": True,
                    "slot_cap": (
                        self._fixed_fleet_total_slots() if self.fixed_fleet_enabled else None
                    ),
                    "slot_cap_source": (
                        "FIXED_FLEET_MAX_PODS" if self.fixed_fleet_enabled else "not_fixed_fleet"
                    ),
                    "primary_machine_preference": bool(
                        self.fixed_fleet_enabled and self.fixed_fleet_machine_ids
                    ),
                    "offer_fallback_when_primary_busy": bool(
                        self.fixed_fleet_enabled and not self.fixed_fleet_strict_machines
                    ),
                    "secondary_cache_ordering": (
                        "tier_then_success_rate_then_cache_then_price"
                        if self.fixed_fleet_enabled and self.fixed_fleet_gpu_tier_rank
                        else (
                            "primary_allowlist_then_machine_cache_secondary_then_cold"
                            if self.fixed_fleet_enabled
                            else None
                        )
                    ),
                    "fleet_omni_shared_bake_models": sorted(self.FLEET_OMNI_SHARED_BAKE_MODELS),
                    "fleet_omni_template_priority_env": "FLEET_OMNI_TEMPLATE_PRIORITY",
                },
            }

