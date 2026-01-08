# job_manager.py
"""
Job Orchestration Manager

Provides:
- Job state machine with strict transition validation
- Job lifecycle management (create, dispatch, complete, recover)
- Automatic retry with exponential backoff
- Stuck job detection and recovery
- Idempotent state transitions
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_

from db import SessionLocal, Job, Worker, create_audit_log, utc_naive_now
from exceptions import (
    JobNotFoundError,
    JobStateError,
    JobDispatchError,
    JobTimeoutError,
    JobMaxRetriesError,
)

logger = logging.getLogger("job_manager")


# =============================================================================
# Job States
# =============================================================================

class JobState(str, Enum):
    """
    Job lifecycle states.
    
    State Machine:
    
    PENDING → QUEUED → DISPATCHING → RUNNING → COMPLETED
                ↓          ↓            ↓
              FAILED ← TIMEOUT ← FAILED
                ↓
            RECOVERED (retry) → QUEUED
    """
    PENDING = "pending"
    QUEUED = "queued"
    DISPATCHING = "dispatching"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RECOVERED = "recovered"
    CANCELLED = "cancelled"


# Valid state transitions
ALLOWED_TRANSITIONS: Dict[JobState, List[JobState]] = {
    JobState.PENDING: [JobState.QUEUED, JobState.CANCELLED],
    JobState.QUEUED: [
        JobState.DISPATCHING,
        JobState.RUNNING,  # Direct dispatch
        JobState.CANCELLED,
        JobState.TIMEOUT,
        JobState.FAILED,
    ],
    JobState.DISPATCHING: [
        JobState.RUNNING,
        JobState.FAILED,
        JobState.QUEUED,  # Rollback on dispatch failure
    ],
    JobState.RUNNING: [
        JobState.COMPLETED,
        JobState.FAILED,
        JobState.TIMEOUT,
    ],
    JobState.COMPLETED: [],  # Terminal state
    JobState.FAILED: [JobState.RECOVERED, JobState.QUEUED],  # Can retry
    JobState.TIMEOUT: [JobState.RECOVERED, JobState.QUEUED],  # Can retry
    JobState.RECOVERED: [JobState.QUEUED],
    JobState.CANCELLED: [],  # Terminal state
}

# Terminal states that cannot transition
TERMINAL_STATES = {JobState.COMPLETED, JobState.CANCELLED}

# States that indicate job is active
ACTIVE_STATES = {JobState.QUEUED, JobState.DISPATCHING, JobState.RUNNING}


# =============================================================================
# Job Manager
# =============================================================================

class JobManager:
    """
    Manages job lifecycle with strict state machine enforcement.
    """
    
    def __init__(
        self,
        session_factory,
        pod_manager=None,
        max_retries: int = None,
        job_timeout_minutes: int = None,
    ):
        self.session_factory = session_factory
        self.pod_manager = pod_manager
        
        # Configuration
        self.max_retries = max_retries or int(os.getenv("JOB_MAX_RETRIES", "3"))
        self.job_timeout = timedelta(
            minutes=job_timeout_minutes or int(os.getenv("JOB_TIMEOUT_MINUTES", "30"))
        )
        self.auto_recover = os.getenv("JOB_AUTO_RECOVER", "false").lower() == "true"
    
    def _validate_transition(
        self,
        current_state: JobState,
        new_state: JobState,
        force: bool = False,
    ) -> bool:
        """
        Validate if a state transition is allowed.
        
        Args:
            current_state: Current job state
            new_state: Target state
            force: Allow forced transitions (recovery mode only)
            
        Returns:
            True if transition is valid
            
        Raises:
            JobStateError: If transition is not allowed
        """
        # Same state is always allowed (idempotent)
        if current_state == new_state:
            return True
        
        # Check allowed transitions
        allowed = ALLOWED_TRANSITIONS.get(current_state, [])
        
        if new_state in allowed:
            return True
        
        # Force mode for recovery
        if force:
            # Only allow force for specific recovery transitions
            recovery_allowed = {
                (JobState.FAILED, JobState.RUNNING),
                (JobState.FAILED, JobState.COMPLETED),
                (JobState.FAILED, JobState.QUEUED),
                (JobState.TIMEOUT, JobState.RUNNING),
                (JobState.TIMEOUT, JobState.COMPLETED),
                (JobState.TIMEOUT, JobState.QUEUED),
                (JobState.RUNNING, JobState.QUEUED),  # Requeue running job
            }
            
            if (current_state, new_state) in recovery_allowed:
                logger.warning(
                    f"Forced transition: {current_state.value} → {new_state.value}"
                )
                return True
        
        raise JobStateError(
            job_id="",  # Will be set by caller
            current_state=current_state.value,
            target_state=new_state.value,
        )
    
    def _reset_worker_if_dispatching(self, db: Session, worker_id: Optional[str]):
        """Reset worker from dispatching state if needed"""
        if not worker_id:
            return
        
        worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
        if worker and worker.status == "dispatching":
            logger.info(f"Resetting worker {worker.worker_id} from dispatching → ready")
            worker.status = "ready"
            worker.last_active = utc_naive_now()
    
    def _add_error_log(
        self,
        job: Job,
        state: str,
        error: str,
        worker_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Add entry to job's error log"""
        entry = {
            "timestamp": utc_naive_now().isoformat(),
            "state": state,
            "error": error,
        }
        if worker_id:
            entry["worker_id"] = worker_id
        if details:
            entry["details"] = details
        
        if job.error_log is None:
            job.error_log = []
        job.error_log.append(entry)
    
    # -------------------------------------------------------------------------
    # State Transitions
    # -------------------------------------------------------------------------
    
    def update_job_state(
        self,
        job_id: str,
        new_state: JobState,
        error_message: Optional[str] = None,
        force: bool = False,
        worker_id: Optional[str] = None,
    ) -> Job:
        """
        Update job to a new state.
        
        Args:
            job_id: Job identifier
            new_state: Target state
            error_message: Optional error message to log
            force: Allow forced transitions (recovery mode)
            worker_id: Optional worker ID for logging
            
        Returns:
            Updated Job instance
            
        Raises:
            JobNotFoundError: If job doesn't exist
            JobStateError: If transition is not allowed
        """
        with self.session_factory() as db:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if not job:
                raise JobNotFoundError(job_id)
            
            # Parse current state
            try:
                current_state = JobState(job.status)
            except ValueError:
                current_state = JobState.PENDING
            
            # Idempotent: same state is no-op
            if current_state == new_state:
                logger.debug(f"Job {job_id} already in state {new_state.value}")
                if error_message:
                    self._add_error_log(job, new_state.value, error_message, worker_id)
                    db.commit()
                    db.refresh(job)
                return job
            
            # Validate transition
            try:
                self._validate_transition(current_state, new_state, force)
            except JobStateError as e:
                e.details["job_id"] = job_id
                raise
            
            # Log forced transitions
            if force:
                logger.warning(
                    f"Forced job state change {job_id}: "
                    f"{current_state.value} → {new_state.value}"
                )
            
            # Update state
            job.status = new_state.value
            job.updated_at = utc_naive_now()
            now = utc_naive_now()
            
            # Track state transition timestamps for timing metrics
            if new_state == JobState.QUEUED and not job.queued_at:
                job.queued_at = now
            elif new_state == JobState.DISPATCHING and not job.dispatched_at:
                job.dispatched_at = now
            elif new_state == JobState.RUNNING and not job.started_at:
                job.started_at = now
            
            # Set completed_at for terminal states
            if new_state == JobState.COMPLETED:
                job.completed_at = now
            
            # Reset worker if job failed/cancelled/timeout
            if new_state in (JobState.FAILED, JobState.TIMEOUT, JobState.CANCELLED):
                self._reset_worker_if_dispatching(db, job.worker_id)
            
            # Add error log
            if error_message:
                self._add_error_log(job, new_state.value, error_message, worker_id)
            
            # Create audit log
            create_audit_log(
                db,
                event_type=f"job_{new_state.value}",
                message=f"Job {job_id} transitioned to {new_state.value}",
                target=job_id,
                target_type="job",
                actor=worker_id or "system",
                details={"from_state": current_state.value, "error": error_message},
            )
            
            db.commit()
            db.refresh(job)
            
            logger.info(f"Job {job_id}: {current_state.value} → {new_state.value}")
            return job
    
    def mark_running(
        self,
        job_id: str,
        worker_id: str,
        started_at: Optional[datetime] = None,
    ) -> Job:
        """Mark job as running with worker assignment"""
        with self.session_factory() as db:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if not job:
                raise JobNotFoundError(job_id)
            
            current_state = JobState(job.status) if job.status else JobState.PENDING
            self._validate_transition(current_state, JobState.RUNNING)
            
            job.status = JobState.RUNNING.value
            job.worker_id = worker_id
            job.started_at = started_at or utc_naive_now()
            job.updated_at = utc_naive_now()
            
            # Update worker
            worker = db.query(Worker).filter(Worker.worker_id == worker_id).first()
            if worker:
                worker.status = "busy" if not worker.is_warm_pod else "processing"
                worker.last_active = utc_naive_now()
            
            create_audit_log(
                db,
                event_type="job_started",
                message=f"Job {job_id} started on worker {worker_id}",
                target=job_id,
                target_type="job",
                actor=worker_id,
            )
            
            db.commit()
            db.refresh(job)
            
            logger.info(f"Job {job_id} running on worker {worker_id}")
            return job
    
    def mark_completed(
        self,
        job_id: str,
        result_r2_key: Optional[str] = None,
        output_url: Optional[str] = None,
    ) -> Job:
        """Mark job as completed with result and calculate timing metrics"""
        with self.session_factory() as db:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if not job:
                raise JobNotFoundError(job_id)
            
            current_state = JobState(job.status) if job.status else JobState.PENDING
            self._validate_transition(current_state, JobState.COMPLETED)
            
            completion_time = utc_naive_now()
            job.status = JobState.COMPLETED.value
            job.completed_at = completion_time
            job.updated_at = completion_time
            
            if result_r2_key:
                job.result_r2_key = result_r2_key
            if output_url:
                job.output_url = output_url
            
            # Calculate timing metrics
            timing_metrics = self._calculate_timing_metrics(job, completion_time)
            job.total_time_seconds = timing_metrics["total_time"]
            job.processing_time_seconds = timing_metrics["processing_time"]
            job.queue_time_seconds = timing_metrics["queue_time"]
            job.dispatch_time_seconds = timing_metrics["dispatch_time"]
            
            # Update worker
            if job.worker_id:
                worker = db.query(Worker).filter(Worker.worker_id == job.worker_id).first()
                if worker:
                    worker.status = "ready"
                    worker.last_active = completion_time
                    worker.last_job_finished_at = completion_time
                    worker.total_jobs_processed = (worker.total_jobs_processed or 0) + 1
            
            create_audit_log(
                db,
                event_type="job_completed",
                message=f"Job {job_id} completed",
                target=job_id,
                target_type="job",
                actor=job.worker_id or "system",
                details={
                    "result_key": result_r2_key,
                    "timing": timing_metrics,
                },
            )
            
            db.commit()
            db.refresh(job)
            
            # Log timing metrics to console
            self._log_timing_metrics(job_id, job.model, timing_metrics)
            
            logger.info(f"Job {job_id} completed")
            return job
    
    def _calculate_timing_metrics(
        self,
        job: Job,
        completion_time: datetime,
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
        self,
        job_id: str,
        model: str,
        timing_metrics: Dict[str, Optional[float]],
    ):
        """Log timing metrics to console in a formatted way"""
        logger.info("=" * 80)
        logger.info(f"⏱️  JOB TIMING REPORT: {job_id}")
        logger.info(f"   Model: {model}")
        
        if timing_metrics["total_time"] is not None:
            logger.info(f"   📊 Total Time: {timing_metrics['total_time']:.2f}s (created → completed)")
        
        if timing_metrics["processing_time"] is not None:
            logger.info(f"   ⚡ Processing Time: {timing_metrics['processing_time']:.2f}s (started → completed)")
        
        if timing_metrics["queue_time"] is not None:
            logger.info(f"   ⏳ Queue Time: {timing_metrics['queue_time']:.2f}s (queued → started)")
        
        if timing_metrics["dispatch_time"] is not None:
            logger.info(f"   🚀 Dispatch Time: {timing_metrics['dispatch_time']:.2f}s (dispatched → started)")
        
        logger.info("=" * 80)
    
    def mark_failed(
        self,
        job_id: str,
        error: str,
        worker_id: Optional[str] = None,
    ) -> Job:
        """Mark job as failed"""
        with self.session_factory() as db:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if not job:
                raise JobNotFoundError(job_id)
            
            current_state = JobState(job.status) if job.status else JobState.PENDING
            
            # Skip if already in terminal state
            if current_state in TERMINAL_STATES:
                logger.warning(
                    f"Job {job_id} already in terminal state {current_state.value}"
                )
                return job
            
            self._validate_transition(current_state, JobState.FAILED)
            
            job.status = JobState.FAILED.value
            job.error = error
            job.updated_at = utc_naive_now()
            
            self._add_error_log(job, JobState.FAILED.value, error, worker_id)
            self._reset_worker_if_dispatching(db, job.worker_id)
            
            create_audit_log(
                db,
                event_type="job_failed",
                message=f"Job {job_id} failed: {error}",
                target=job_id,
                target_type="job",
                actor=worker_id or job.worker_id or "system",
                severity="error",
                details={"error": error},
            )
            
            db.commit()
            
            # CREDIT REFUND: Refund locked credits when job fails
            # This happens AFTER the commit to avoid rollback issues
            # refund_credits_for_job uses its own session for atomicity
            if job.credits_locked > 0 and job.credits_refunded < job.credits_locked:
                try:
                    from credit_service import refund_credits_for_job
                    # Pass None for db - function uses its own session
                    success, msg, refunded = refund_credits_for_job(None, job_id, f"job_failed:{error[:50] if error else 'unknown'}")
                    if success:
                        logger.info(f"Refunded {refunded} credits for failed job {job_id}")
                    else:
                        logger.warning(f"Failed to refund credits for job {job_id}: {msg}")
                except Exception as refund_error:
                    logger.error(f"Credit refund error for job {job_id}: {refund_error}")
            
            db.refresh(job)
            
            logger.error(f"Job {job_id} failed: {error}")
            return job
    
    # -------------------------------------------------------------------------
    # Job Assignment & Dispatch
    # -------------------------------------------------------------------------
    
    async def assign_job(
        self,
        job_id: str,
        model: str,
        mode: Optional[str] = None,
    ) -> str:
        """
        Assign a job to a worker via pod manager.
        
        Args:
            job_id: Job to assign
            model: Model required
            mode: Optional mode
            
        Returns:
            Worker ID assigned
        """
        if not self.pod_manager:
            raise JobDispatchError(job_id, reason="No pod manager configured")
        
        with self.session_factory() as db:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if not job:
                raise JobNotFoundError(job_id)
            
            # Dispatch via pod manager
            worker_id = await self.pod_manager.dispatch_job(job_id, model, mode)
            
            # Update job
            job.worker_id = worker_id
            job.updated_at = utc_naive_now()
            job.status = JobState.QUEUED.value
            
            db.commit()
            
            logger.info(f"Job {job_id} assigned to worker {worker_id}")
            return worker_id
    
    # -------------------------------------------------------------------------
    # Recovery
    # -------------------------------------------------------------------------
    
    async def check_for_stuck_jobs(self):
        """
        Detect and handle stuck jobs.
        
        Checks for jobs in RUNNING or QUEUED that haven't been updated
        within the timeout window.
        """
        now = utc_naive_now()
        timeout_threshold = now - self.job_timeout
        
        with self.session_factory() as db:
            stuck_jobs = (
                db.query(Job)
                .filter(
                    and_(
                        Job.status.in_([
                            JobState.PENDING.value,      # Stuck after creation
                            JobState.QUEUED.value,       # Waiting for worker
                            JobState.DISPATCHING.value,  # Waiting for dispatch
                            JobState.RUNNING.value,      # Stuck during execution
                            JobState.RECOVERED.value,    # Recovery failed to dispatch
                        ]),
                        Job.updated_at < timeout_threshold,
                    )
                )
                .all()
            )
            
            for job in stuck_jobs:
                try:
                    logger.warning(
                        f"Job {job.job_id} stuck in {job.status}, "
                        f"last update: {job.updated_at}"
                    )
                    
                    # Mark as timeout
                    job.status = JobState.TIMEOUT.value
                    job.updated_at = now
                    
                    self._add_error_log(
                        job,
                        JobState.TIMEOUT.value,
                        f"Job timed out after {self.job_timeout.total_seconds() / 60} minutes",
                    )
                    self._reset_worker_if_dispatching(db, job.worker_id)
                    
                    db.commit()
                    
                    # Send FCM notification for timeout (if recovery is disabled or will fail)
                    if job.fcm_token and not job.notification_sent:
                        retry_count = job.retry_count or 0
                        # Only send notification if this is the final timeout (no more retries)
                        if retry_count >= self.max_retries or not self.auto_recover:
                            try:
                                from routes import send_fcm_notification
                                import asyncio
                                asyncio.create_task(
                                    send_fcm_notification(
                                        job.fcm_token,
                                        "Job Timeout",
                                        f"Your {job.model} job timed out after {self.job_timeout.total_seconds() / 60:.0f} minutes",
                                        {"job_id": job.job_id, "status": "timeout"},
                                    )
                                )
                                job.notification_sent = True
                                db.commit()
                            except Exception as e:
                                logger.error(f"Failed to send FCM notification for timeout job {job.job_id}: {e}")
                    
                    # Auto-recover if enabled
                    if self.auto_recover:
                        await self.recover_job(job.job_id)
                        
                except Exception as e:
                    logger.exception(f"Error handling stuck job {job.job_id}: {e}")
    
    async def recover_job(self, job_id: str):
        """
        Attempt to recover a failed/timed out job.
        
        Increments retry count and re-dispatches if under limit.
        """
        with self.session_factory() as db:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if not job:
                raise JobNotFoundError(job_id)
            
            current_retry = job.retry_count or 0
            
            # Check max retries
            if current_retry >= self.max_retries:
                job.status = JobState.FAILED.value
                job.error = "Maximum retries exceeded"
                job.updated_at = utc_naive_now()
                
                self._add_error_log(
                    job,
                    JobState.FAILED.value,
                    f"Max retries ({self.max_retries}) exceeded",
                )
                self._reset_worker_if_dispatching(db, job.worker_id)
                
                create_audit_log(
                    db,
                    event_type="job_max_retries",
                    message=f"Job {job_id} exceeded max retries",
                    target=job_id,
                    target_type="job",
                    severity="error",
                )
                
                db.commit()
                logger.error(f"Job {job_id} failed: max retries exceeded")
                
                # Send FCM notification for final failure
                if job.fcm_token and not job.notification_sent:
                    try:
                        from routes import send_fcm_notification
                        import asyncio
                        asyncio.create_task(
                            send_fcm_notification(
                                job.fcm_token,
                                "Job Failed",
                                f"Your {job.model} job failed after {self.max_retries} retry attempts",
                                {"job_id": job_id, "status": "failed", "error": "Max retries exceeded"},
                            )
                        )
                        job.notification_sent = True
                        db.commit()
                    except Exception as e:
                        logger.error(f"Failed to send FCM notification for failed job {job_id}: {e}")
                
                return
            
            try:
                # Mark as recovered
                job.status = JobState.RECOVERED.value
                job.updated_at = utc_naive_now()
                
                self._add_error_log(
                    job,
                    JobState.RECOVERED.value,
                    f"Job recovered for retry (attempt {current_retry + 1})",
                )
                self._reset_worker_if_dispatching(db, job.worker_id)
                
                # Reset for re-queue
                job.status = JobState.QUEUED.value
                job.retry_count = current_retry + 1
                job.worker_id = None
                job.error = None
                
                db.commit()
                
                # Re-dispatch via full dispatch flow (not just assign_job!)
                # CRITICAL: assign_job only assigns in DB, we need _dispatch_job_background
                # to actually send the HTTP request to the worker's /run endpoint.
                if self.pod_manager:
                    try:
                        # Import here to avoid circular import
                        import asyncio
                        import json
                        from routes import _dispatch_job_background
                        
                        # Get mode from job params if available
                        mode = None
                        if job.params:
                            try:
                                params = json.loads(job.params) if isinstance(job.params, str) else job.params
                                mode = params.get("mode")
                            except Exception:
                                pass
                        
                        # Trigger full dispatch flow (assign + HTTP dispatch)
                        asyncio.create_task(
                            _dispatch_job_background(job_id, job.model, mode)
                        )
                        logger.info(
                            f"Recovered job {job_id} (retry {current_retry + 1}), "
                            f"dispatching via background task"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to dispatch recovered job {job_id}: {e}. "
                            f"Job will remain in QUEUED and retry again later."
                        )
                        # Job is already in QUEUED state, so it will be picked up again
                else:
                    logger.info(
                        f"Recovered job {job_id} (retry {current_retry + 1}), "
                        f"awaiting manual dispatch"
                    )
                
            except Exception as e:
                logger.exception(f"Failed to recover job {job_id}: {e}")
                
                job.status = JobState.FAILED.value
                job.error = f"Recovery failed: {str(e)}"
                job.updated_at = utc_naive_now()
                
                self._add_error_log(
                    job,
                    JobState.FAILED.value,
                    f"Recovery failed: {str(e)}",
                )
                self._reset_worker_if_dispatching(db, job.worker_id)
                
                db.commit()
                
                # Send FCM notification for recovery failure
                if job.fcm_token and not job.notification_sent:
                    try:
                        from routes import send_fcm_notification
                        import asyncio
                        asyncio.create_task(
                            send_fcm_notification(
                                job.fcm_token,
                                "Job Failed",
                                f"Your {job.model} job failed during recovery",
                                {"job_id": job_id, "status": "failed", "error": str(e)},
                            )
                        )
                        job.notification_sent = True
                        db.commit()
                    except Exception as notify_error:
                        logger.error(f"Failed to send FCM notification: {notify_error}")
    
    # -------------------------------------------------------------------------
    # Cancellation
    # -------------------------------------------------------------------------
    
    def cancel_job(self, job_id: str, reason: str) -> Job:
        """
        Cancel a job.
        
        Args:
            job_id: Job to cancel
            reason: Cancellation reason
            
        Returns:
            Updated Job instance
        """
        with self.session_factory() as db:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if not job:
                raise JobNotFoundError(job_id)
            
            current_state = JobState(job.status) if job.status else JobState.PENDING
            
            # Check if cancellation is allowed
            if JobState.CANCELLED not in ALLOWED_TRANSITIONS.get(current_state, []):
                raise JobStateError(
                    job_id=job_id,
                    current_state=current_state.value,
                    target_state=JobState.CANCELLED.value,
                )
            
            job.status = JobState.CANCELLED.value
            job.updated_at = utc_naive_now()
            
            self._add_error_log(job, JobState.CANCELLED.value, reason)
            self._reset_worker_if_dispatching(db, job.worker_id)
            
            create_audit_log(
                db,
                event_type="job_cancelled",
                message=f"Job {job_id} cancelled: {reason}",
                target=job_id,
                target_type="job",
                details={"reason": reason},
            )
            
            db.commit()
            db.refresh(job)
            
            logger.info(f"Job {job_id} cancelled: {reason}")
            return job
    
    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        with self.session_factory() as db:
            return db.query(Job).filter(Job.job_id == job_id).first()
    
    def get_job_history(self, job_id: str) -> Dict[str, Any]:
        """Get job history and state"""
        with self.session_factory() as db:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if not job:
                raise JobNotFoundError(job_id)
            
            return {
                "job_id": job.job_id,
                "model": job.model,
                "current_state": job.status,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "completed_at": job.completed_at,
                "started_at": job.started_at,
                "retry_count": job.retry_count or 0,
                "error": job.error,
                "error_log": job.error_log or [],
                "worker_id": job.worker_id,
                "result_r2_key": job.result_r2_key,
                "output_url": job.output_url,
            }
    
    def get_active_jobs_for_worker(self, worker_id: str) -> List[Job]:
        """Get active jobs assigned to a worker"""
        with self.session_factory() as db:
            return (
                db.query(Job)
                .filter(
                    Job.worker_id == worker_id,
                    Job.status.in_([s.value for s in ACTIVE_STATES]),
                )
                .all()
            )
    
    def get_jobs_by_status(
        self,
        status: JobState,
        limit: int = 100,
    ) -> List[Job]:
        """Get jobs by status"""
        with self.session_factory() as db:
            return (
                db.query(Job)
                .filter(Job.status == status.value)
                .order_by(Job.created_at.desc())
                .limit(limit)
                .all()
            )


# =============================================================================
# Background Task
# =============================================================================

async def job_recovery_task(job_manager: JobManager, interval: int = 30):
    """
    Background task to check for and recover stuck jobs.
    
    Args:
        job_manager: JobManager instance
        interval: Check interval in seconds
    """
    logger.info(f"Starting job recovery task (interval={interval}s)")
    
    while True:
        try:
            await job_manager.check_for_stuck_jobs()
        except Exception:
            logger.exception("Job recovery task error")
        
        await asyncio.sleep(interval)

