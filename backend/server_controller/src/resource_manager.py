# resource_manager.py
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional

from db import Worker, Job
from job_manager import JobState
from pod_manager import PodManager  # PodManager is optional at runtime (may be None)

logger = logging.getLogger("resource_manager")


class ResourceLimits:
    GPU_MEMORY_THRESHOLD = 0.9  # 90% GPU memory usage
    CPU_USAGE_THRESHOLD = 0.8   # 80% CPU usage
    MEMORY_THRESHOLD = 0.85     # 85% system memory
    MAX_ERROR_COUNT = 3         # Maximum errors before worker is marked unhealthy
    MAX_QUEUE_SIZE = 5          # Maximum jobs in queue per worker
    INACTIVE_TIMEOUT = timedelta(minutes=30)  # Time before worker is considered inactive


class ResourceManager:
    """
    Monitor worker resource metrics and mark unhealthy/critical workers.
    Optionally triggers PodManager cleanup for ephemeral workers.
    """

    def __init__(self, session_factory, pod_manager: Optional[PodManager] = None):
        """
        session_factory: callable that returns a DB session (e.g. SessionLocal)
        pod_manager: optional PodManager instance used to stop/destroy pods
        """
        self.session_factory = session_factory
        self.limits = ResourceLimits()
        self.pod_manager = pod_manager

    def check_worker_health(self, worker: Worker) -> str:
        """Check worker health based on metrics and return status string."""
        # If no metrics available, return 'unknown' so caller can decide.
        if not getattr(worker, "resource_metrics", None):
            return "unknown"

        metrics = worker.resource_metrics or {}
        status = "healthy"

        # Check GPU memory usage
        if metrics.get("gpu_memory_used", 0) > self.limits.GPU_MEMORY_THRESHOLD:
            status = "warning"
            logger.warning("Worker %s high GPU memory usage: %.1f%%", worker.worker_id,
                           metrics.get("gpu_memory_used", 0) * 100)

        # Check CPU usage
        if metrics.get("cpu_usage", 0) > self.limits.CPU_USAGE_THRESHOLD:
            status = "warning"
            logger.warning("Worker %s high CPU usage: %.1f%%", worker.worker_id,
                           metrics.get("cpu_usage", 0) * 100)

        # Check system memory
        if metrics.get("memory_usage", 0) > self.limits.MEMORY_THRESHOLD:
            status = "warning"
            logger.warning("Worker %s high memory usage: %.1f%%", worker.worker_id,
                           metrics.get("memory_usage", 0) * 100)

        # Check error count
        if getattr(worker, "error_count", 0) >= self.limits.MAX_ERROR_COUNT:
            status = "critical"
            logger.error("Worker %s has too many errors: %s", worker.worker_id, worker.error_count)

        # Check queue size
        if getattr(worker, "queue_size", 0) > self.limits.MAX_QUEUE_SIZE:
            # Queue too large is a warning-level issue
            status = "warning"
            logger.warning("Worker %s queue too large: %s", worker.worker_id, worker.queue_size)

        # Check inactivity
        last_active = getattr(worker, "last_active", None)
        if last_active and (datetime.utcnow() - last_active > self.limits.INACTIVE_TIMEOUT):
            status = "critical"
            logger.error("Worker %s inactive for too long", worker.worker_id)

        return status

    async def monitor_resources(self) -> None:
        """
        Async periodic monitor to check all workers and handle critical ones.
        Call regularly from a background task.
        """
        with self.session_factory() as db:
            workers = db.query(Worker).all()
            for worker in workers:
                try:
                    health_status = self.check_worker_health(worker)
                    if health_status != worker.health_status:
                        logger.info("Worker %s health changed: %s -> %s",
                                    worker.worker_id, worker.health_status, health_status)
                        worker.health_status = health_status
                        db.commit()

                    if health_status == "critical":
                        # handle_critical_worker is async because it may call pod_manager async methods
                        await self.handle_critical_worker(worker, db)
                except Exception as e:
                    logger.exception("Error monitoring worker %s: %s", worker.worker_id, e)

    async def handle_critical_worker(self, worker: Worker, db) -> None:
        """
        When a worker is critical: mark its active jobs as RECOVERED (so job_recovery can requeue),
        mark the worker offline/critical and optionally trigger pod cleanup for ephemeral pods.
        """
        try:
            # Find active jobs for this worker
            active_jobs = db.query(Job).filter(
                Job.worker_id == worker.worker_id,
                Job.status.in_([JobState.RUNNING.value, JobState.QUEUED.value])
            ).all()

            for job in active_jobs:
                # Use assignment instead of in-place append so SQLAlchemy detects the change
                job.status = JobState.RECOVERED.value
                job.error_log = (job.error_log or []) + [{
                    "timestamp": datetime.utcnow().isoformat(),
                    "event": "worker_critical",
                    "worker_id": worker.worker_id,
                    "health_status": worker.health_status
                }]

            # Mark worker offline/critical
            worker.status = "offline"
            worker.health_status = "critical"
            db.commit()

            # Trigger pod cleanup for ephemeral (non-warm) workers if PodManager is provided
            if getattr(worker, "container_id", None) and not getattr(worker, "is_warm_pod", False):
                if self.pod_manager:
                    logger.warning("Cleaning up critical worker %s (pod %s)", worker.worker_id, worker.container_id)
                    try:
                        # pod_manager.stop_individual_pod may be async; await if coroutine
                        maybe_coro = self.pod_manager.stop_individual_pod(worker.container_id)
                        if hasattr(maybe_coro, "__await__"):
                            await maybe_coro
                    except Exception as e:
                        logger.exception("Failed to cleanup pod %s for worker %s: %s",
                                         worker.container_id, worker.worker_id, e)
                else:
                    logger.info("PodManager not provided; skipping pod cleanup for worker %s", worker.worker_id)
            else:
                # Warm pods are kept but marked offline so operator can restart
                if getattr(worker, "is_warm_pod", False):
                    logger.info("Warm pod worker %s marked offline (manual restart may be required)", worker.worker_id)

        except Exception as e:
            logger.exception("Error handling critical worker %s: %s", worker.worker_id, e)

    def update_resource_metrics(self, worker_id: str, metrics: Dict[str, Any]) -> None:
        """
        Update worker resource metrics (called by controller's heartbeat handler or a metrics endpoint).
        Also updates last_active timestamp. Optionally update queue_size and error_count if supplied.
        """
        with self.session_factory() as db:
            worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
            if not worker:
                logger.debug("update_resource_metrics: worker not found %s", worker_id)
                return
            worker.resource_metrics = metrics
            worker.last_active = datetime.utcnow()

            # Optionally map common metrics to worker fields for easier queries
            if isinstance(metrics, dict):
                if "queue_size" in metrics:
                    try:
                        worker.queue_size = int(metrics.get("queue_size", 0))
                    except Exception:
                        pass
                if "error_count" in metrics:
                    try:
                        worker.error_count = int(metrics.get("error_count", worker.error_count or 0))
                    except Exception:
                        pass

            db.commit()

    def set_resource_limits(self, worker_id: str, limits: Dict[str, Any]) -> None:
        """Set custom resource limits for a worker (stored in DB)."""
        with self.session_factory() as db:
            worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
            if not worker:
                return
            worker.resource_limits = limits
            db.commit()

    def cleanup_resources(self) -> None:
        """
        Cleanup stale resources: mark stale workers offline and mark their active jobs as RECOVERED.
        Uses assignment for job.error_log to ensure persistence.
        """
        try:
            with self.session_factory() as db:
                stale_time = datetime.utcnow() - timedelta(minutes=5)
                stale_workers = db.query(Worker).filter(
                    Worker.last_heartbeat < stale_time,
                    Worker.status != "offline"
                ).all()

                for worker in stale_workers:
                    logger.warning("Worker %s appears stale, marking offline", worker.worker_id)
                    worker.status = "offline"
                    worker.health_status = "critical"

                    stale_jobs = db.query(Job).filter(
                        Job.worker_id == worker.worker_id,
                        Job.status.in_([JobState.RUNNING.value, JobState.QUEUED.value])
                    ).all()

                    for job in stale_jobs:
                        job.status = JobState.RECOVERED.value
                        job.error_log = (job.error_log or []) + [{
                            "timestamp": datetime.utcnow().isoformat(),
                            "event": "worker_stale",
                            "worker_id": worker.worker_id
                        }]

                db.commit()
        except Exception as e:
            logger.exception("Error in cleanup_resources: %s", e)

    def get_system_metrics(self) -> Dict[str, Any]:
        """Return aggregated system metrics useful for dashboards/monitoring."""
        try:
            with self.session_factory() as db:
                total_workers = db.query(Worker).count()
                active_workers = db.query(Worker).filter(Worker.status == "ready").count()
                busy_workers = db.query(Worker).filter(Worker.status == "busy").count()
                critical_workers = db.query(Worker).filter(Worker.health_status == "critical").count()

                active_jobs = db.query(Job).filter(
                    Job.status.in_([JobState.RUNNING.value, JobState.QUEUED.value])
                ).count()

                return {
                    "workers": {
                        "total": total_workers,
                        "active": active_workers,
                        "busy": busy_workers,
                        "critical": critical_workers
                    },
                    "jobs": {
                        "active": active_jobs
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
        except Exception as e:
            logger.exception("Error getting system metrics: %s", e)
            return {}
