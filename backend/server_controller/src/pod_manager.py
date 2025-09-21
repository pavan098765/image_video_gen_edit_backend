# pod_manager.py
import os
import time
import logging
import asyncio
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from quickpod_api import QuickPodManager  # assumed sync client

logger = logging.getLogger("pod_manager")


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

        self.max_individual_pods = int(os.getenv("MAX_INDIVIDUAL_PODS", "3"))
        self.pod_startup_timeout = int(os.getenv("POD_STARTUP_TIMEOUT", "300"))
        self.template_uuids = {
            "sd15": os.getenv("SD15_TEMPLATE_UUID"),
            "sd35": os.getenv("SD35_TEMPLATE_UUID"),
            "sdxl": os.getenv("SDXL_TEMPLATE_UUID"),
            "default": os.getenv("DEFAULT_TEMPLATE_UUID"),
        }

    async def initialize(self) -> None:
        """Initialize the pod manager and attempt to start warm pod (non-blocking)."""
        if not self.warm_pod_manager:
            logger.info("No warm pod configured.")
            self.own_warm_pod = False
            return

        try:
            logger.info("Logging into QuickPod for warm pod...")
            await asyncio.to_thread(self.warm_pod_manager.login)

            logger.info("Attempting to start warm pod...")
            resp = await asyncio.to_thread(self.warm_pod_manager.start_pod)

            if isinstance(resp, dict) and resp.get("status") == "busy":
                logger.warning("Warm pod is allocated to someone else")
                self.own_warm_pod = False
            else:
                self.own_warm_pod = True
                logger.info("Warm pod started successfully")
        except Exception as e:
            logger.exception("Failed to initialize warm pod manager: %s", e)
            self.own_warm_pod = False

    async def dispatch_job(self, job_id: str, model: str, mode: Optional[str] = None) -> str:
        """
        Assign a job to warm pod if available, else reuse a ready individual pod,
        else create a new individual pod. Returns worker_id of the registered worker.
        """
        if self.own_warm_pod and self.warm_pod_uuid:
            logger.info("Dispatching job %s to warm pod %s", job_id, self.warm_pod_uuid)
            start = time.time()
            timeout = int(os.getenv("WORKER_REGISTRATION_TIMEOUT", "180"))
            from db import SessionLocal, Worker
            while time.time() - start < timeout:
                with SessionLocal() as db:
                    w = db.query(Worker).filter(
                        Worker.container_id == self.warm_pod_uuid,
                        Worker.model == model,
                        Worker.status.in_(["ready", "processing"]),
                    ).first()
                    if w:
                        logger.info(
                            "Found registered warm worker %s for pod %s",
                            w.worker_id,
                            self.warm_pod_uuid,
                        )
                        return w.worker_id
                await asyncio.sleep(1)

            logger.warning(
                "Warm pod configured but worker not registered for pod %s after %s seconds",
                self.warm_pod_uuid,
                timeout,
            )

        async with self._lock:
            for pod_uuid, info in self.individual_pods.items():
                if info.model == model and info.status == "ready":
                    info.last_active = datetime.utcnow()
                    info.queue_size += 1
                    logger.info("Reusing ready individual pod %s for job %s", pod_uuid, job_id)
                    from db import SessionLocal, Worker
                    with SessionLocal() as db:
                        w = db.query(Worker).filter(
                            Worker.container_id == pod_uuid, Worker.model == model
                        ).first()
                        if w:
                            return w.worker_id

        pod_uuid = await self.create_individual_pod(model, mode)
        if not pod_uuid:
            raise RuntimeError("No pod available for job")

        logger.info("Waiting for worker registration for pod %s (job %s)", pod_uuid, job_id)
        start = time.time()
        timeout = int(os.getenv("WORKER_REGISTRATION_TIMEOUT", "180"))
        from db import SessionLocal, Worker
        while time.time() - start < timeout:
            with SessionLocal() as db:
                w = db.query(Worker).filter(
                    Worker.container_id == pod_uuid,
                    Worker.model == model,
                    Worker.status.in_(["ready", "processing"]),
                ).first()
                if w:
                    logger.info("Worker %s registered for pod %s", w.worker_id, pod_uuid)
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

            template_name = f"{model}_{mode}_startup_template" if mode else f"{model}_startup_template"
            logger.info("Looking for template '%s'", template_name)
            template_uuid = await asyncio.to_thread(pod_manager.get_template_by_name, template_name)
            if not template_uuid:
                logger.info("Falling back to default_startup_template")
                template_uuid = await asyncio.to_thread(
                    pod_manager.get_template_by_name, "default_startup_template"
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
