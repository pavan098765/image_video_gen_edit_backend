import os
import time
import logging
import asyncio
import enum
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from quickpod_api import QuickPodManager  # assumed sync client

logger = logging.getLogger("pod_manager")

class PodStrategyType(str, enum.Enum):
    WARM_AND_INDIVIDUAL = "warm_and_individual"  # Original strategy
    MACHINE_BASED = "machine_based"  # New strategy with machine ID filtering

# Prioritized machine IDs for the machine-based strategy
MACHINE_PRIORITY_LIST: List[Tuple[str, str]] = [
    ("562", "4x 5090"),  # (machine_id, description)
    ("14774", "4x 4090"),
    ("14762", "1x 5090"),
]

# Job limits per pod
MAX_JOBS_PER_POD = 5  # Different job IDs
MAX_SAME_JOB_PER_POD = 20  # Same job ID


@dataclass
class PodInfo:
    pod_uuid: str
    model: str
    mode: Optional[str]
    status: str  # "starting", "ready", "error", "stopped"
    created_at: datetime
    last_active: datetime
    queue_size: int = 0
    error_count: int = 0
    machine_id: Optional[str] = None
    total_jobs_processed: int = 0
    unique_job_ids: set = field(default_factory=set)
    job_id_counts: Dict[str, int] = field(default_factory=dict)
    # NEW: optional metadata for quick lookup (public ip, ports, etc.)
    meta: Dict[str, Any] = field(default_factory=dict)


class PodManager:
    def __init__(self, email: str, password: str, warm_pod_uuid: Optional[str]):
        self.email = email
        self.password = password
        self.warm_pod_uuid = warm_pod_uuid
        self.warm_pod_manager = (
            QuickPodManager(email, password, warm_pod_uuid) if warm_pod_uuid else None
        )
        self.own_warm_pod: bool = False
        self.individual_pods: Dict[str, PodInfo] = {}
        self.individual_pod_managers: Dict[str, QuickPodManager] = {}
        self._lock = asyncio.Lock()
        
        # Track pending jobs by model
        self.pending_jobs: Dict[str, set] = {}  # model -> set of job_ids
        self._jobs_lock = asyncio.Lock()

        # Pod strategy configuration
        self.pod_strategy = PodStrategyType(os.getenv("POD_STRATEGY", PodStrategyType.WARM_AND_INDIVIDUAL.value))
        self.max_individual_pods = int(os.getenv("MAX_INDIVIDUAL_PODS", "3"))
        self.pod_startup_timeout = int(os.getenv("POD_STARTUP_TIMEOUT", "30"))
        self.template_uuids = {
            "sd15": os.getenv("SD15_TEMPLATE_UUID"),
            "sd35": os.getenv("SD35_TEMPLATE_UUID"),
            "sdxl": os.getenv("SDXL_TEMPLATE_UUID"),
            "diffusers_outpaint": os.getenv("DIFFUSERS_OUTPAINT_TEMPLATE_UUID"),
            "qwen_image_edit": os.getenv("QWEN_IMAGE_EDIT_TEMPLATE_UUID"),
            "flux_kontext_dev": os.getenv("FLUX_KONTEXT_DEV_TEMPLATE_UUID"),
            "fastwan_21_1b": os.getenv("FASTWAN_21_1B_TEMPLATE_UUID"),
            "fastwan_22_5b": os.getenv("FASTWAN_22_5B_TEMPLATE_UUID"),
            "image_upscaler_aurasrv2": os.getenv("AURASRV2_TEMPLATE_UUID"),
            "image_upscaler_realesrgan": os.getenv("REALESRGAN_TEMPLATE_UUID"),
            "image_upscaler_swin2sr": os.getenv("SWIN2SR_TEMPLATE_UUID"),
            "basic_utils": os.getenv("BASIC_UTILS_TEMPLATE_UUID"),
            "warm_pod": os.getenv("WARM_POD_TEMPLATE_UUID"),
            "default": os.getenv("DEFAULT_TEMPLATE_UUID"),
        }

    async def initialize(self) -> None:
        if not self.warm_pod_manager:
            logger.info("No warm pod configured.")
            self.own_warm_pod = False
            return
        try:
            logger.info("Logging into QuickPod for warm pod...")
            await asyncio.to_thread(self.warm_pod_manager.login)
            logger.info("Attempting to start warm pod...")
            resp = await asyncio.to_thread(self.warm_pod_manager.start_pod)
            if isinstance(resp, dict) and "busy" in resp.get("error", "").lower():
                logger.warning("Warm pod is allocated to someone else")
                self.own_warm_pod = False
            else:
                self.own_warm_pod = True
                logger.info("Warm pod started successfully")
        except Exception as e:
            logger.exception("Failed to initialize warm pod manager: %s", e)
            self.own_warm_pod = False

    async def _ensure_warm_pod_running(self) -> None:
        if not self.warm_pod_manager:
            return
        try:
            await asyncio.to_thread(self.warm_pod_manager.login)
            resp = await asyncio.to_thread(self.warm_pod_manager.start_pod)
            if isinstance(resp, dict) and resp.get("status") == "busy":
                logger.warning("Warm pod start attempt: offer busy")
                self.own_warm_pod = False
            else:
                logger.info("Warm pod started/restarted successfully")
                self.own_warm_pod = True
        except Exception:
            logger.exception("Failed to start/restart warm pod")
            self.own_warm_pod = False

    async def dispatch_job(self, job_id: str, model: str, mode: Optional[str] = None) -> str:
        """
        Dispatch job based on the selected pod strategy.
        Returns worker_id of the registered worker.
        """
        from db import SessionLocal, Worker, Job
        from job_manager import JobState

        if self.pod_strategy == PodStrategyType.WARM_AND_INDIVIDUAL:
            return await self._dispatch_job_warm_and_individual(job_id, model, mode)
        else:  # MACHINE_BASED
            return await self._dispatch_job_machine_based(job_id, model, mode)

    async def _dispatch_job_machine_based(self, job_id: str, model: str, mode: Optional[str] = None) -> str:
        """Handle job dispatch using machine-based strategy."""
        from db import SessionLocal, Worker, Job
        from job_manager import JobState

        # Track this job as pending until it's assigned
        await self.add_pending_job(job_id, model)

        # First try to find an existing suitable pod
        async with self._lock:
            for pod_uuid, info in self.individual_pods.items():
                if (info.model == model and 
                    info.status == "ready" and 
                    self.can_reuse_pod(info, job_id)):
                    
                    info.last_active = datetime.utcnow()
                    self.update_pod_job_counts(info, job_id)
                    
                    logger.info("Reusing machine-based pod %s for job %s", pod_uuid, job_id)
                    with SessionLocal() as db:
                        w = db.query(Worker).filter(
                            Worker.container_id == pod_uuid, 
                            Worker.model == model
                        ).first()
                        if w:
                            w.status = "dispatching"
                            w.last_active = datetime.utcnow()

                            job = db.query(Job).filter(Job.job_id == job_id).first()
                            if job:
                                job.status = JobState.DISPATCHING.value
                                job.updated_at = datetime.utcnow()

                            db.commit()
                            return w.worker_id

        # Create new pod if needed
        pod_uuid = await self.create_machine_based_pod(model, mode)
        if not pod_uuid:
            raise RuntimeError("Failed to create machine-based pod")

        # Wait for worker registration
        start = time.time()
        timeout = int(os.getenv("WORKER_REGISTRATION_TIMEOUT", "180"))
        while time.time() - start < timeout:
            with SessionLocal() as db:
                w = db.query(Worker).filter(
                    Worker.container_id == pod_uuid,
                    Worker.model == model,
                    Worker.status == "ready",
                ).first()
                if w:
                    w.status = "dispatching"
                    w.last_active = datetime.utcnow()

                    job = db.query(Job).filter(Job.job_id == job_id).first()
                    if job:
                        job.status = JobState.DISPATCHING.value
                        job.updated_at = datetime.utcnow()

                    # Initialize job counts
                    async with self._lock:
                        pod_info = self.individual_pods.get(pod_uuid)
                        if pod_info:
                            self.update_pod_job_counts(pod_info, job_id)

                    db.commit()
                    return w.worker_id
            await asyncio.sleep(1)
        
        raise RuntimeError(f"Worker registration timeout for pod {pod_uuid}")

    async def _dispatch_job_warm_and_individual(self, job_id: str, model: str, mode: Optional[str] = None) -> str:
        """Original warm pod + individual pods strategy."""
        from db import SessionLocal, Worker, Job
        from job_manager import JobState

        if self.warm_pod_manager and not self.own_warm_pod:
            logger.info("Warm pod exists but not running; attempting (re)start")
            await self._ensure_warm_pod_running()

        # --------------------
        # Try warm pod
        # --------------------
        if self.own_warm_pod and self.warm_pod_uuid:
            logger.info("Dispatching job %s to warm pod %s", job_id, self.warm_pod_uuid)
            start = time.time()
            timeout = int(os.getenv("WORKER_REGISTRATION_TIMEOUT", "180"))
            while time.time() - start < timeout:
                with SessionLocal() as db:
                    w = db.query(Worker).filter(
                        Worker.container_id == self.warm_pod_uuid,
                        Worker.model == model,
                        Worker.status == "ready",
                    ).first()
                    if w:
                        logger.info(
                            "Found registered warm worker %s for pod %s",
                            w.worker_id,
                            self.warm_pod_uuid,
                        )
                        w.status = "dispatching"
                        w.last_active = datetime.utcnow()

                        job = db.query(Job).filter(Job.job_id == job_id).first()
                        if job:
                            job.status = JobState.DISPATCHING.value
                            job.updated_at = datetime.utcnow()

                        db.commit()
                        return w.worker_id
                await asyncio.sleep(1)
            logger.warning(
                "Warm pod configured but worker not registered for pod %s after %s seconds",
                self.warm_pod_uuid,
                timeout,
            )

        # --------------------
        # Create new individual pod
        # --------------------
        pod_uuid = await self.create_individual_pod(model, mode)
        if not pod_uuid:
            raise RuntimeError("No pod available for job")

        logger.info("Waiting for worker registration for pod %s (job %s)", pod_uuid, job_id)
        start = time.time()
        timeout = int(os.getenv("WORKER_REGISTRATION_TIMEOUT", "180"))
        while time.time() - start < timeout:
            with SessionLocal() as db:
                w = db.query(Worker).filter(
                    Worker.container_id == pod_uuid,
                    Worker.model == model,
                    Worker.status == "ready",
                ).first()
                if w:
                    logger.info("Worker %s registered for pod %s", w.worker_id, pod_uuid)
                    w.status = "dispatching"
                    w.last_active = datetime.utcnow()

                    job = db.query(Job).filter(Job.job_id == job_id).first()
                    if job:
                        job.status = JobState.DISPATCHING.value
                        job.updated_at = datetime.utcnow()

                    db.commit()
                    return w.worker_id
            await asyncio.sleep(1)

        logger.error("Worker did not register for pod %s within timeout", pod_uuid)
        await self.stop_individual_pod(pod_uuid)
        raise RuntimeError(f"Worker for pod {pod_uuid} did not register")

        
    async def create_individual_pod(self, model: str, mode: Optional[str] = None) -> Optional[str]:
        """Create a new individual pod for a specific model/mode (auto-started)."""
        async with self._lock:
            if len(self.individual_pods) >= self.max_individual_pods:
                logger.warning(
                    "Maximum number of individual pods reached (%s/%s)",
                    len(self.individual_pods),
                    self.max_individual_pods,
                )
                raise RuntimeError("Maximum number of individual pods reached")

        pod_manager = None
        pod_uuid = None
        try:
            pod_manager = QuickPodManager(self.email, self.password)
            logger.info("Looking up best offer for pod (model=%s mode=%s)", model, mode)
            offer_id = await asyncio.to_thread(pod_manager.get_best_offer)
            if not offer_id:
                raise RuntimeError("No offer returned by QuickPod")

            template_name = f"{model}"
            logger.info("Looking for template '%s'", template_name)
            template_uuid = self.template_uuids.get(model)
            if not template_uuid:
                template_uuid = await asyncio.to_thread(pod_manager.get_template_by_name, template_name)
            if not template_uuid:
                logger.info("Falling back to default_startup_template")
                template_uuid = await asyncio.to_thread(
                    pod_manager.get_template_by_name, "default"
                )
                if not template_uuid:
                    raise RuntimeError(f"No suitable template found for {model}")

            logger.info("Creating pod (offer=%s template=%s)...", offer_id, template_uuid)
            pod_data = await asyncio.to_thread(
                pod_manager.create_pod,
                offer_id=offer_id,
                template_uuid=template_uuid,
                disk_size=50000,
            )
            if not isinstance(pod_data, dict):
                raise RuntimeError("Unexpected response creating pod")

            pod_uuid = pod_data.get("pod_uuid") or pod_data.get("uuid") or pod_data.get("id")
            if not pod_uuid:
                raise RuntimeError("No pod UUID returned from creation")

            logger.info("Created pod %s for model %s (mode=%s)", pod_uuid, model, mode)

            # Collect metadata for quick lookup
            pod_meta = {
                "public_ip": pod_data.get("public_ipaddr"),
                "ports": (pod_data.get("open_port_start"), pod_data.get("open_port_end")),
            }

            pod_info = PodInfo(
                pod_uuid=pod_uuid,
                model=model,
                mode=mode,
                status="starting",
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow(),
                meta=pod_meta,
            )
            async with self._lock:
                self.individual_pods[pod_uuid] = pod_info
                self.individual_pod_managers[pod_uuid] = pod_manager

            # Pods auto-start after creation
            return pod_uuid

        except Exception as e:
            logger.exception("Failed to create pod: %s", e)

        if pod_uuid:
            await self.stop_individual_pod(pod_uuid)
        return None

    async def start_individual_pod(self, pod_uuid: str) -> bool:
        """Attempt to start a pod if needed (normally unused since create_pod auto-starts)."""
        async with self._lock:
            pod_manager = self.individual_pod_managers.get(pod_uuid)
            pod_info = self.individual_pods.get(pod_uuid)

        if not pod_manager or not pod_info:
            logger.error("start_individual_pod: unknown pod_uuid %s", pod_uuid)
            return False

        try:
            logger.info("Starting pod %s (manual)...", pod_uuid)
            await asyncio.to_thread(pod_manager.start_pod)

            start_time = time.time()
            while time.time() - start_time < self.pod_startup_timeout:
                if await self.check_pod_ready(pod_uuid):
                    async with self._lock:
                        pod_info.status = "ready"
                        pod_info.last_active = datetime.utcnow()
                    logger.info("Pod %s is ready", pod_uuid)
                    return True
                await asyncio.sleep(5)

            raise RuntimeError("Pod startup timeout")
        except Exception as e:
            logger.exception("Failed to start pod %s: %s", pod_uuid, e)
            async with self._lock:
                if pod_info:
                    pod_info.status = "error"
                    pod_info.error_count += 1
            return False

    async def create_machine_based_pod(self, model: str, mode: Optional[str] = None) -> Optional[str]:
        """Create a pod on a specific machine based on priority list."""
        for machine_id, _desc in MACHINE_PRIORITY_LIST:
            try:
                template_id = self._get_template_for_model(model)
                if not template_id:
                    logger.error(f"No template found for model {model}")
                    continue

                pod_manager = QuickPodManager(self.email, self.password)
                await asyncio.to_thread(pod_manager.login)

                # Ask for an offer specifically on this machine_id
                offer_id = await asyncio.to_thread(pod_manager.get_best_offer, machine_id)
                if not offer_id:
                    logger.warning(f"No offer available on machine {machine_id}")
                    continue

                # Create pod using offer id and template uuid
                pod_data = await asyncio.to_thread(
                    pod_manager.create_pod,
                    offer_id,
                    template_id,
                    50000
                )

                # pod_data is expected to be dict (create_pod returns dict)
                if not isinstance(pod_data, dict):
                    logger.warning(f"Unexpected creation response for machine {machine_id}: {pod_data}")
                    continue

                pod_uuid = pod_data.get("pod_uuid") or pod_data.get("uuid") or pod_data.get("id")
                if not pod_uuid:
                    logger.warning(f"No pod_uuid returned for machine {machine_id} create call")
                    continue

                # Store pod info
                async with self._lock:
                    self.individual_pod_managers[pod_uuid] = pod_manager
                    self.individual_pods[pod_uuid] = PodInfo(
                        pod_uuid=pod_uuid,
                        model=model,
                        mode=mode,
                        status="starting",
                        created_at=datetime.utcnow(),
                        last_active=datetime.utcnow(),
                        machine_id=str(machine_id)
                    )

                # Wait for pod to be ready
                start_time = time.time()
                while time.time() - start_time < self.pod_startup_timeout:
                    if await self.check_pod_ready(pod_uuid):
                        async with self._lock:
                            self.individual_pods[pod_uuid].status = "ready"
                        return pod_uuid
                    await asyncio.sleep(5)

                logger.error(f"Pod {pod_uuid} startup timeout on machine {machine_id}")
                await self.stop_individual_pod(pod_uuid)

            except Exception as e:
                logger.exception(f"Failed to create pod on machine {machine_id}: {e}")
                continue

        return None

    def _get_template_for_model(self, model: str) -> Optional[str]:
        """Get template UUID for given model."""
        if model in self.template_uuids:
            return self.template_uuids[model]
        return self.template_uuids.get("default")

    def can_reuse_pod(self, pod_info: PodInfo, job_id: str) -> bool:
        """Check if a pod can be reused based on job limits."""
        if pod_info.total_jobs_processed >= MAX_JOBS_PER_POD:
            return False
            
        if job_id in pod_info.job_id_counts:
            return pod_info.job_id_counts[job_id] < MAX_SAME_JOB_PER_POD
            
        return len(pod_info.unique_job_ids) < MAX_JOBS_PER_POD
        
    def update_pod_job_counts(self, pod_info: PodInfo, job_id: str) -> None:
        """Update job counts for a pod."""
        pod_info.total_jobs_processed += 1
        pod_info.unique_job_ids.add(job_id)
        pod_info.job_id_counts[job_id] = pod_info.job_id_counts.get(job_id, 0) + 1

    async def check_pod_ready(self, pod_uuid: str) -> bool:
        """Check if a pod is ready (DB worker registration preferred)."""
        from db import SessionLocal, Worker

        with SessionLocal() as db:
            w = db.query(Worker).filter(Worker.container_id == pod_uuid).first()
            if w and w.status in ("ready", "processing"):
                return True

        async with self._lock:
            pod_manager = self.individual_pod_managers.get(pod_uuid)
        if not pod_manager:
            return False

        try:
            response = await asyncio.to_thread(pod_manager.get_logs)
            if isinstance(response, dict):
                return "ready" in str(response.get("status", "")).lower()
            elif isinstance(response, str):
                return "ready" in response.lower()
        except Exception as e:
            logger.exception("check_pod_ready failed for %s: %s", pod_uuid, e)
        return False

    async def stop_individual_pod(self, pod_uuid: str) -> None:
        """Stop and clean up an individual pod."""
        async with self._lock:
            pod_manager = self.individual_pod_managers.get(pod_uuid)
            exists = pod_uuid in self.individual_pods

        if not exists or not pod_manager:
            async with self._lock:
                self.individual_pods.pop(pod_uuid, None)
                self.individual_pod_managers.pop(pod_uuid, None)
            return

        try:
            logger.info("Stopping pod %s ...", pod_uuid)
            await asyncio.to_thread(pod_manager.stop_pod)
        except Exception:
            logger.exception("Error while stopping pod %s", pod_uuid)

        await asyncio.sleep(1)
        try:
            logger.info("Destroying pod %s ...", pod_uuid)
            await asyncio.to_thread(pod_manager.destroy_pod)
        except Exception:
            logger.exception("Error while destroying pod %s", pod_uuid)

        async with self._lock:
            self.individual_pods.pop(pod_uuid, None)
            self.individual_pod_managers.pop(pod_uuid, None)
        logger.info("Pod %s cleaned up", pod_uuid)

    async def add_pending_job(self, job_id: str, model: str):
        """Track a new pending job for a model."""
        async with self._jobs_lock:
            if model not in self.pending_jobs:
                self.pending_jobs[model] = set()
            self.pending_jobs[model].add(job_id)

    async def remove_pending_job(self, job_id: str, model: str):
        """Remove a job from pending tracking once assigned or completed."""
        async with self._jobs_lock:
            if model in self.pending_jobs:
                self.pending_jobs[model].discard(job_id)
                if not self.pending_jobs[model]:
                    del self.pending_jobs[model]

    async def has_pending_jobs_for_model(self, model: str) -> bool:
        """Check if there are pending jobs for a specific model."""
        async with self._jobs_lock:
            return model in self.pending_jobs and bool(self.pending_jobs[model])

    async def should_keep_pod_alive(self, pod_info: PodInfo) -> bool:
        """
        Decide whether to keep a pod alive based on pending jobs.
        Returns True if there are pending jobs for the pod's model.
        """
        if self.pod_strategy == PodStrategyType.WARM_AND_INDIVIDUAL:
            return True  # Original strategy keeps pods alive until idle timeout
        
        # For machine-based strategy, check pending jobs
        return await self.has_pending_jobs_for_model(pod_info.model)

    async def cleanup_inactive_pods(self) -> None:
        """Clean up inactive or errored pods."""
        now = datetime.utcnow()
        to_stop = []
        async with self._lock:
            for pod_uuid, info in list(self.individual_pods.items()):
                inactive = (now - info.last_active) > timedelta(minutes=30)
                starting_timed_out = (
                    info.status == "starting" and (now - info.created_at) > timedelta(minutes=10)
                )
                if info.error_count >= 3 or inactive or starting_timed_out:
                    to_stop.append(pod_uuid)

        if not to_stop:
            return

        logger.info("Cleaning up %d pods: %s", len(to_stop), to_stop)
        for pod_uuid in to_stop:
            try:
                await self.stop_individual_pod(pod_uuid)
            except Exception:
                logger.exception("Error cleaning up pod %s", pod_uuid)

    async def stop_warm_pod(self, pod_uuid: str) -> None:
        """Stop warm pod (never destroy)."""
        if not self.warm_pod_manager or pod_uuid != self.warm_pod_uuid:
            logger.warning("stop_warm_pod called for non-warm pod %s", pod_uuid)
            return
        try:
            logger.info("Stopping warm pod %s ...", pod_uuid)
            await asyncio.to_thread(self.warm_pod_manager.stop_pod)
            # mark as not owned locally; next dispatch will attempt restart
            self.own_warm_pod = False
        except Exception:
            logger.exception("Error stopping warm pod %s", pod_uuid)

    async def destroy_individual_pod(self, pod_uuid: str) -> None:
        """Destroy an individual pod (skip stop)."""
        async with self._lock:
            pod_manager = self.individual_pod_managers.get(pod_uuid)
            exists = pod_uuid in self.individual_pods

        if not exists or not pod_manager:
            return

        try:
            logger.info("Destroying pod %s ...", pod_uuid)
            await asyncio.to_thread(pod_manager.destroy_pod)
        except Exception:
            logger.exception("Error destroying pod %s", pod_uuid)

        async with self._lock:
            self.individual_pods.pop(pod_uuid, None)
            self.individual_pod_managers.pop(pod_uuid, None)
        logger.info("Individual pod %s destroyed", pod_uuid)
