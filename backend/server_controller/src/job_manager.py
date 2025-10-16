import os
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_
import logging
from db import Job, Worker
from pod_manager import PodManager

logger = logging.getLogger("job_manager")

MAX_RETRIES = int(os.getenv("JOB_MAX_RETRIES", "3"))


class JobState(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RECOVERED = "recovered"
    CANCELLED = "cancelled"
    DISPATCHING = "dispatching"



class JobStateError(Exception):
    pass


class JobManager:
    def __init__(self, session_factory, pod_manager: PodManager):
        self.session_factory = session_factory
        self.pod_manager = pod_manager
        self.job_timeout = timedelta(minutes=30)

        self.allowed_transitions = {
            JobState.PENDING: [JobState.QUEUED, JobState.CANCELLED],
            JobState.QUEUED: [JobState.RUNNING, JobState.CANCELLED, JobState.TIMEOUT, JobState.FAILED],
            JobState.RUNNING: [JobState.COMPLETED, JobState.FAILED, JobState.TIMEOUT, JobState.RECOVERED],
            JobState.COMPLETED: [],
            JobState.DISPATCHING: [JobState.RUNNING, JobState.FAILED],
            JobState.FAILED: [JobState.RECOVERED],
            JobState.TIMEOUT: [JobState.RECOVERED],
            JobState.RECOVERED: [JobState.QUEUED],
            JobState.CANCELLED: [],
        }

    def _validate_transition(self, current_state: JobState, new_state: JobState):
        if new_state not in self.allowed_transitions[current_state]:
            raise JobStateError(f"Invalid state transition from {current_state} to {new_state}")

    def _reset_worker_if_dispatching(self, db: Session, worker_id: Optional[str]):
        """Ensure worker doesn't stay stuck in 'dispatching' if job fails/cancels."""
        if not worker_id:
            return
        worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
        if worker and worker.status == "dispatching":
            logger.info(f"Resetting worker {worker.worker_id} from 'dispatching' → 'ready'")
            worker.status = "ready"
            worker.last_active = datetime.utcnow()

    def update_job_state(self, job_id: str, new_state: JobState, error_message: Optional[str] = None) -> Job:
        with self.session_factory() as db:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if not job:
                raise JobStateError(f"Job {job_id} not found")

            current_state = JobState(job.status)
            self._validate_transition(current_state, new_state)

            job.status = new_state.value
            job.updated_at = datetime.utcnow()

            # If job failed/cancelled/timeout, free up worker
            if new_state in [JobState.FAILED, JobState.TIMEOUT, JobState.CANCELLED]:
                self._reset_worker_if_dispatching(db, job.worker_id)

            if error_message:
                job.error_log = (job.error_log or []) + [{
                    "timestamp": datetime.utcnow().isoformat(),
                    "state": new_state.value,
                    "error": error_message,
                }]

            db.commit()
            db.refresh(job)
            return job

    async def assign_job(self, job_id: str, model: str, mode: Optional[str] = None) -> str:
        with self.session_factory() as db:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if not job:
                raise JobStateError(f"Job {job_id} not found")

            worker_id = await self.pod_manager.dispatch_job(job_id, model, mode)
            job.worker_id = worker_id
            job.updated_at = datetime.utcnow()
            job.status = JobState.QUEUED.value

            db.commit()
            logger.info(f"Job {job_id} assigned to worker {worker_id}")
            return worker_id

    async def check_for_stuck_jobs(self) -> None:
        now = datetime.utcnow()
        timeout_threshold = now - self.job_timeout

        with self.session_factory() as db:
            stuck_jobs = db.query(Job).filter(
                and_(
                    Job.status.in_([JobState.RUNNING.value, JobState.QUEUED.value]),
                    Job.updated_at < timeout_threshold,
                )
            ).all()

            for job in stuck_jobs:
                try:
                    logger.warning(
                        f"Job {job.job_id} appears stuck in state {job.status}. "
                        f"Last update: {job.updated_at}"
                    )
                    job.status = JobState.TIMEOUT.value
                    job.updated_at = datetime.utcnow()
                    job.error_log = (job.error_log or []) + [
                        {
                            "timestamp": datetime.utcnow().isoformat(),
                            "state": JobState.TIMEOUT.value,
                            "error": f"Job timed out after {self.job_timeout.total_seconds()/60} minutes",
                        }
                    ]

                    # Reset worker if it was still stuck in dispatching
                    self._reset_worker_if_dispatching(db, job.worker_id)

                    db.commit()
                    await self.recover_job(job.job_id)

                except Exception as e:
                    logger.error(f"Error handling stuck job {job.job_id}: {e}")

    async def recover_job(self, job_id: str) -> None:
        with self.session_factory() as db:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if not job:
                raise JobStateError(f"Job {job_id} not found")

            try:
                if (job.retry_count or 0) >= MAX_RETRIES:
                    job.status = JobState.FAILED.value
                    job.error = "max retries exceeded"
                    job.updated_at = datetime.utcnow()
                    # Reset worker if stuck
                    self._reset_worker_if_dispatching(db, job.worker_id)
                    db.commit()
                    logger.error(f"Job {job_id} failed: max retries exceeded")
                    return

                job.status = JobState.RECOVERED.value
                job.updated_at = datetime.utcnow()
                job.error_log = (job.error_log or []) + [
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "state": JobState.RECOVERED.value,
                        "error": "Job recovered for retry",
                    }
                ]

                # Reset worker if stuck
                self._reset_worker_if_dispatching(db, job.worker_id)

                job.status = JobState.QUEUED.value
                job.retry_count = (job.retry_count or 0) + 1
                job.worker_id = None
                db.commit()

                worker_id = await self.assign_job(job_id, job.model)
                logger.info(f"Successfully recovered and reassigned job {job_id} to worker {worker_id}")

            except Exception as e:
                logger.error(f"Failed to recover job {job_id}: {e}")
                job.status = JobState.FAILED.value
                job.error_log = (job.error_log or []) + [
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "state": JobState.FAILED.value,
                        "error": f"Recovery failed: {str(e)}",
                    }
                ]
                # Reset worker if stuck
                self._reset_worker_if_dispatching(db, job.worker_id)
                db.commit()

    def cancel_job(self, job_id: str, reason: str) -> None:
        with self.session_factory() as db:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if not job:
                raise JobStateError(f"Job {job_id} not found")

            current_state = JobState(job.status)
            if JobState.CANCELLED in self.allowed_transitions[current_state]:
                job.status = JobState.CANCELLED.value
                job.updated_at = datetime.utcnow()
                job.error_log = (job.error_log or []) + [
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "state": JobState.CANCELLED.value,
                        "error": reason,
                    }
                ]
                # Reset worker if stuck
                self._reset_worker_if_dispatching(db, job.worker_id)
                db.commit()
            else:
                raise JobStateError(f"Cannot cancel job in state {current_state}")

    def get_job_history(self, job_id: str) -> Dict[str, Any]:
        with self.session_factory() as db:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if not job:
                raise JobStateError(f"Job {job_id} not found")

            return {
                "job_id": job.job_id,
                "current_state": job.status,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "retry_count": job.retry_count or 0,
                "error_log": job.error_log or [],
                "worker_id": job.worker_id,
                "model": job.model,
            }
