# main.py
"""
Application Entry Point

Provides:
- FastAPI application factory
- Startup/shutdown lifecycle management
- Background task initialization
- Graceful shutdown handling
"""

import os
import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

from fastapi import FastAPI
from dotenv import load_dotenv

# Load environment first
env_path = Path(__file__).parent / "config.env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# Import modules after env load
from config import ConfigManager, load_config, get_config
from logging_config import setup_logging, get_logger
from db import init_db, get_db_manager
from middleware import setup_middleware
from routes import (
    public_router,
    client_router,
    worker_router,
    auth_router,
    ads_router,
    admin_router,
    set_dependencies,
    sync_heartbeat_cache_to_db,  # Background task for heartbeat caching
    get_cached_last_heartbeat,   # CRITICAL: avoid false stale detections
)
from pod_manager import PodManager
from job_manager import JobManager, job_recovery_task
from security import ip_ban_manager

# Initialize logging
setup_logging()
logger = get_logger("main")


# =============================================================================
# Application State
# =============================================================================

class AppState:
    """Global application state"""
    
    def __init__(self):
        self.config = None
        self.pod_manager: Optional[PodManager] = None
        self.job_manager: Optional[JobManager] = None
        self.s3_client = None
        self.background_tasks: list = []
        self.shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize all application components"""
        logger.info("=" * 60)
        logger.info("INITIALIZING APPLICATION")
        logger.info("=" * 60)
        
        # Load configuration
        self.config = load_config(strict=False)
        config_manager = ConfigManager.get_instance()
        config_manager.log_config()
        
        # Initialize database
        logger.info("Initializing database...")
        init_db()
        
        # Initialize Firebase
        logger.info("Initializing Firebase...")
        config_manager.initialize_firebase()
        
        # Initialize R2 client
        logger.info("Initializing R2 storage...")
        self.s3_client = config_manager.get_s3_client()
        
        # Initialize Pod Manager
        logger.info("Initializing Pod Manager...")
        self.pod_manager = PodManager(
            email=self.config.quickpod.email,
            password=self.config.quickpod.password,
            warm_pod_uuid=self.config.quickpod.warm_pod_uuid if self.config.pod.strategy.value == "warm_and_individual" else None,
        )
        await self.pod_manager.initialize()
        
        # Initialize Job Manager
        logger.info("Initializing Job Manager...")
        db_manager = get_db_manager()
        self.job_manager = JobManager(
            session_factory=db_manager.session_factory,
            pod_manager=self.pod_manager,
            max_retries=self.config.job.max_retries,
            job_timeout_minutes=self.config.job.timeout_minutes,
            running_job_timeout_minutes=self.config.job.running_timeout_minutes,
        )
        
        # Load IP bans
        logger.info("Loading IP bans...")
        await ip_ban_manager.load_from_db()
        await ip_ban_manager.load_from_file()
        
        # Set route dependencies
        logger.info("Configuring routes...")
        model_requirements = self.pod_manager.model_requirements if self.pod_manager else {}
        
        set_dependencies(
            pod_manager=self.pod_manager,
            job_manager=self.job_manager,
            s3_client=self.s3_client,
            r2_bucket=self.config.r2.bucket or "",
            r2_endpoint=self.config.r2.endpoint or "",
            model_requirements=model_requirements,
            image_prefix=self.config.image_prefix,
            controller_url=self.config.public_url,
        )
        
        logger.info("=" * 60)
        logger.info("APPLICATION INITIALIZED SUCCESSFULLY")
        logger.info("=" * 60)
    
    async def start_background_tasks(self):
        """Start all background tasks"""
        logger.info("Starting background tasks...")
        
        # Job recovery task
        if self.config.job.auto_recover:
            task = asyncio.create_task(
                job_recovery_task(self.job_manager, interval=30),
                name="job_recovery"
            )
            self.background_tasks.append(task)
            logger.info("Started job recovery task")
        
        # Idle worker reaper
        task = asyncio.create_task(
            self._idle_worker_reaper_task(),
            name="idle_worker_reaper"
        )
        self.background_tasks.append(task)
        logger.info("Started idle worker reaper task")
        
        # Pod cleanup task
        task = asyncio.create_task(
            self._pod_cleanup_task(),
            name="pod_cleanup"
        )
        self.background_tasks.append(task)
        logger.info("Started pod cleanup task")
        
        # Autoscaler
        if self.pod_manager:
            task = asyncio.create_task(
                self.pod_manager.autoscaler_loop(),
                name="autoscaler"
            )
            self.background_tasks.append(task)
            logger.info("Started autoscaler task")
        
        # IP ban cleanup
        task = asyncio.create_task(
            self._ban_cleanup_task(),
            name="ban_cleanup"
        )
        self.background_tasks.append(task)
        logger.info("Started ban cleanup task")
        
        # Job recovery task (CRITICAL - detects stuck jobs)
        if self.job_manager:
            task = asyncio.create_task(
                self._job_recovery_task(),
                name="job_recovery"
            )
            self.background_tasks.append(task)
            logger.info("Started job recovery task")
        
        # Queued job dispatcher task (CRITICAL - dispatches QUEUED jobs to available workers)
        # This handles jobs that were reverted to QUEUED after dispatch failures
        if self.pod_manager:
            task = asyncio.create_task(
                self._queued_job_dispatcher_task(),
                name="queued_job_dispatcher"
            )
            self.background_tasks.append(task)
            logger.info("Started queued job dispatcher task")
        
        # Heartbeat cache sync task (CRITICAL - makes heartbeat endpoint fast)
        task = asyncio.create_task(
            sync_heartbeat_cache_to_db(),
            name="heartbeat_sync"
        )
        self.background_tasks.append(task)
        logger.info("Started heartbeat cache sync task")
        
        # Stray pod cleanup task (catches pods that slipped through other cleanup)
        task = asyncio.create_task(
            self._stray_pod_cleanup_task(),
            name="stray_pod_cleanup"
        )
        self.background_tasks.append(task)
        logger.info("Started stray pod cleanup task")
        
        # R2 input media reaper (cleanup orphaned input-media objects)
        task = asyncio.create_task(
            self._r2_input_media_reaper_task(),
            name="r2_input_media_reaper"
        )
        self.background_tasks.append(task)
        logger.info("Started R2 input media reaper task")
        
        logger.info(f"Started {len(self.background_tasks)} background tasks")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("=" * 60)
        logger.info("SHUTTING DOWN APPLICATION")
        logger.info("=" * 60)
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Cancel background tasks
        logger.info("Cancelling background tasks...")
        for task in self.background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Allow in-flight requests to complete
        logger.info("Waiting for in-flight requests...")
        await asyncio.sleep(2)
        
        # Cleanup
        if self.pod_manager:
            # Save machine cache
            await self.pod_manager.machine_cache.save()
        
        # Save IP bans
        await ip_ban_manager.save_to_file()
        
        logger.info("=" * 60)
        logger.info("APPLICATION SHUTDOWN COMPLETE")
        logger.info("=" * 60)
    
    # -------------------------------------------------------------------------
    # Background Tasks
    # -------------------------------------------------------------------------
    
    async def _idle_worker_reaper_task(self):
        """Reap idle workers"""
        from db import SessionLocal, Worker, Job
        from datetime import datetime, timedelta
        from job_manager import JobState
        
        heartbeat_stale_secs = self.config.worker.heartbeat_stale_seconds
        idle_cfg = self.config.worker
        sweep_interval = 30
        
        logger.info(
            f"Idle worker reaper: stale>{heartbeat_stale_secs}s, "
            f"idle_timeout={idle_cfg.idle_timeout_seconds}s, min_age={idle_cfg.idle_min_age_seconds}s"
        )
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(sweep_interval)
                
                db = SessionLocal()
                now = datetime.utcnow()
                stale_cutoff = now - timedelta(seconds=heartbeat_stale_secs)
                
                try:
                    workers = db.query(Worker).filter(
                        # Include "starting" and "dispatching" for correctness
                        Worker.status.in_(["ready", "busy", "processing", "starting", "dispatching"])
                    ).all()
                    
                    for w in workers:
                        try:
                            # ------------------------------------------------------------------
                            # CRITICAL: Handle workers stuck in "starting" state FIRST.
                            # Do NOT treat "starting" as stale just because DB heartbeat is missing.
                            # ------------------------------------------------------------------
                            if w.status == "starting":
                                worker_age = (now - w.created_at).total_seconds() if w.created_at else 0
                                max_starting_time = 300  # 5 minutes
                                
                                if worker_age > max_starting_time:
                                    logger.error(
                                        f"Worker {w.worker_id} stuck in 'starting' state for {worker_age:.1f}s "
                                        f"(max: {max_starting_time}s). Destroying pod {w.container_id}"
                                    )
                                    if w.container_id and self.pod_manager:
                                        await self.pod_manager.destroy_individual_pod(
                                            w.container_id,
                                            reason=f"Stuck in starting state for {worker_age:.0f}s"
                                        )
                                    w.status = "offline"
                                    db.commit()
                                else:
                                    logger.debug(
                                        f"Worker {w.worker_id} is starting ({worker_age:.1f}s), "
                                        f"giving time to load models (max {max_starting_time}s)"
                                    )
                                continue

                            # Stale check
                            cached_hb = get_cached_last_heartbeat(w.worker_id)
                            effective_hb = cached_hb or w.last_heartbeat
                            
                            # If cache shows recent heartbeat, worker is NOT stale even if DB lags
                            if not effective_hb or effective_hb < stale_cutoff:
                                logger.warning(
                                    f"Worker {w.worker_id} stale, destroying pod {w.container_id}"
                                )
                                if w.container_id and self.pod_manager:
                                    try:
                                        await self.pod_manager.destroy_individual_pod(
                                            w.container_id,
                                            reason="Stale heartbeat"
                                        )
                                    except Exception as e:
                                        logger.error(f"Destroy failed: {e}")
                                
                                w.status = "offline"
                                db.commit()
                                continue
                            
                            # NEVER destroy workers that are busy/processing - they have active jobs
                            if w.status in ("busy", "processing"):
                                # Double-check: verify there are actually jobs assigned
                                active_jobs = db.query(Job).filter(
                                    Job.worker_id == w.worker_id,
                                    Job.status.in_([
                                        JobState.RUNNING.value,
                                        JobState.DISPATCHING.value
                                    ])
                                ).count()
                                
                                if active_jobs > 0:
                                    logger.debug(
                                        f"Worker {w.worker_id} is {w.status} with {active_jobs} job(s), "
                                        f"skipping idle check"
                                    )
                                    continue
                                
                                # BUG FIX: Worker says busy but no jobs — reset to ready and
                                # FALL THROUGH to the idle check below instead of `continue`.
                                # Previously `continue` here caused a thrashing loop: every 15s
                                # the reaper would reset busy→ready, but the next heartbeat
                                # would set it back to busy, and the idle timer never advanced.
                                logger.warning(
                                    f"Worker {w.worker_id} status is {w.status} but has no active jobs, "
                                    f"resetting to ready and checking idle timeout"
                                )
                                w.status = "ready"
                                db.commit()
                            
                            # Ready worker idle check
                            if w.status == "ready":
                                # CRITICAL: Check if this worker has any active jobs (RUNNING, QUEUED, DISPATCHING)
                                active_jobs = db.query(Job).filter(
                                    Job.worker_id == w.worker_id,
                                    Job.status.in_([
                                        JobState.RUNNING.value,
                                        JobState.QUEUED.value,
                                        JobState.DISPATCHING.value
                                    ])
                                ).count()
                                
                                if active_jobs > 0:
                                    logger.debug(
                                        f"Worker {w.worker_id} has {active_jobs} active job(s), "
                                        f"skipping idle check"
                                    )
                                    continue
                                
                                # Also check if pod has queued jobs (from pod_manager tracking)
                                if w.container_id and self.pod_manager:
                                    async with self.pod_manager._lock:
                                        pod_info = self.pod_manager.individual_pods.get(w.container_id)
                                        if pod_info and pod_info.queue_size > 0:
                                            logger.debug(
                                                f"Pod {w.container_id} has {pod_info.queue_size} "
                                                f"queued job(s), skipping idle check"
                                            )
                                            continue
                                
                                # Do not destroy if there are PENDING jobs for this model (e.g. next bulk batch)
                                if w.model:
                                    pending_for_model = db.query(Job).filter(
                                        Job.model == w.model,
                                        Job.status == JobState.PENDING.value,
                                    ).count()
                                    if pending_for_model > 0:
                                        logger.debug(
                                            f"Worker {w.worker_id} model {w.model} has {pending_for_model} "
                                            f"PENDING job(s), skipping idle destroy"
                                        )
                                        continue
                                
                                # Idle timeout: pod completed all assigned jobs, no new jobs came to it.
                                # Destroy only if idle for configured timeout (errors handled separately).
                                last_job = w.last_job_finished_at or w.created_at or w.last_active
                                idle_for = (now - last_job).total_seconds() if last_job else 0
                                worker_age = (now - w.created_at).total_seconds() if w.created_at else 0
                                
                                if worker_age < idle_cfg.idle_min_age_seconds:
                                    logger.debug(
                                        f"Worker {w.worker_id} is only {worker_age:.1f}s old, "
                                        f"min_age={idle_cfg.idle_min_age_seconds}s, skipping idle check"
                                    )
                                    continue
                                
                                timeout_secs = idle_cfg.idle_timeout_seconds
                                if idle_for > timeout_secs:
                                    logger.info(
                                        f"Worker {w.worker_id} idle for {idle_for:.1f}s "
                                        f"(timeout: {timeout_secs}s), destroying pod {w.container_id}"
                                    )
                                    if w.container_id and self.pod_manager:
                                        await self.pod_manager.destroy_individual_pod(
                                            w.container_id,
                                            reason=f"Idle for {idle_for:.0f}s"
                                        )
                                    w.status = "offline"
                                    db.commit()
                                else:
                                    # Every 60 seconds of idleness, do a health check to detect dead pods
                                    if idle_for > 60 and w.endpoint:
                                        try:
                                            import httpx
                                            async with httpx.AsyncClient() as client:
                                                health_url = f"{w.endpoint.rstrip('/')}/health"
                                                resp = await client.get(health_url, timeout=5.0)
                                                if resp.status_code != 200:
                                                    logger.warning(
                                                        f"Worker {w.worker_id} failed health check (status: {resp.status_code}), "
                                                        f"destroying pod"
                                                    )
                                                    if w.container_id and self.pod_manager:
                                                        await self.pod_manager.destroy_individual_pod(
                                                            w.container_id,
                                                            reason=f"Failed health check (status: {resp.status_code})"
                                                        )
                                                    w.status = "offline"
                                                    db.commit()
                                                    continue
                                        except Exception as health_err:
                                            logger.warning(
                                                f"Worker {w.worker_id} health check failed: {health_err}. "
                                                f"Destroying unresponsive pod."
                                            )
                                            if w.container_id and self.pod_manager:
                                                await self.pod_manager.destroy_individual_pod(
                                                    w.container_id,
                                                    reason=f"Health check failed: {health_err}"
                                                )
                                            w.status = "offline"
                                            db.commit()
                                            continue
                                    
                                    logger.debug(
                                        f"Worker {w.worker_id} idle for {idle_for:.1f}s, "
                                        f"within timeout ({timeout_secs}s)"
                                    )
                                    
                        except Exception as e:
                            logger.exception(f"Error processing worker {w.worker_id}: {e}")
                            
                finally:
                    db.close()
                    
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Idle worker reaper error")
    
    async def _pod_cleanup_task(self):
        """Clean up terminated/offline pods"""
        from db import SessionLocal, Worker
        
        while not self.shutdown_event.is_set():
            try:
                # INCREASED: 45s interval to reduce load on weak servers (was 15s)
                await asyncio.sleep(45)
                
                if not self.pod_manager:
                    continue
                
                db = SessionLocal()
                try:
                    # Find workers in terminal states with pods
                    workers = db.query(Worker).filter(
                        Worker.status.in_(["terminated", "offline", "error"]),
                        Worker.container_id != None,
                    ).all()
                    
                    for w in workers:
                        try:
                            # Skip warm pods
                            if w.is_warm_pod:
                                continue
                            
                            logger.info(
                                f"Cleaning up pod {w.container_id} "
                                f"(worker {w.worker_id}, status={w.status})"
                            )
                            
                            await self.pod_manager.destroy_individual_pod(
                                w.container_id,
                                reason=f"Worker status: {w.status}"
                            )
                            
                            w.container_id = None
                            db.commit()
                            
                        except Exception as e:
                            logger.debug(f"Pod cleanup error for {w.container_id}: {e}")
                            
                finally:
                    db.close()
                
                # Also cleanup lifecycle tracker
                await self.pod_manager.lifecycle.cleanup_destroyed_tracker()
                
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Pod cleanup task error")
    
    async def _stray_pod_cleanup_task(self):
        """
        Periodically check for stray pods that exist in QuickPod but not in our tracking.
        
        This catches pods that:
        - Failed to destroy but weren't properly tracked
        - Were created but never registered
        - Slipped through other cleanup mechanisms
        
        CRITICAL: Does NOT destroy pods that have active jobs!
        """
        from quickpod_api import QuickPodManager
        from db import SessionLocal, Worker, Job
        from job_manager import JobState
        
        # Run every 5 minutes
        cleanup_interval = 300
        
        logger.info(f"Stray pod cleanup task started (interval={cleanup_interval}s)")
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(cleanup_interval)
                
                if not self.pod_manager:
                    continue
                
                try:
                    # Fetch all pods from QuickPod
                    # Create a fresh session for each cleanup run to avoid session issues
                    from quickpod_api import QuickPodManager
                    qp = QuickPodManager(
                        self.pod_manager.email,
                        self.pod_manager.password,
                    )
                    await asyncio.to_thread(qp.login)
                    pods = await asyncio.to_thread(qp.get_my_pods)
                    
                    if pods is None:
                        logger.debug("Stray pod cleanup: Failed to fetch pods from QuickPod")
                        continue
                    
                    # Get set of tracked pod UUIDs
                    async with self.pod_manager._lock:
                        tracked_pods = set(self.pod_manager.individual_pods.keys())
                    
                    # Add warm pod if we have one
                    if self.pod_manager.warm_pod_uuid and self.pod_manager.own_warm_pod:
                        tracked_pods.add(self.pod_manager.warm_pod_uuid)
                    
                    # CRITICAL: Also get pods with active jobs from DB
                    # This protects against destroying pods after server restart
                    db = SessionLocal()
                    try:
                        # Find all workers with active jobs
                        active_workers = db.query(Worker).join(
                            Job, Worker.worker_id == Job.worker_id
                        ).filter(
                            Job.status.in_([
                                JobState.RUNNING.value,
                                JobState.DISPATCHING.value,
                            ])
                        ).all()
                        
                        # Add their container_ids to tracked set
                        for w in active_workers:
                            if w.container_id:
                                tracked_pods.add(w.container_id)
                                logger.debug(
                                    f"Stray pod cleanup: Protecting pod {w.container_id} "
                                    f"(has active jobs)"
                                )
                    finally:
                        db.close()
                    
                    # Find stray pods (exist in QuickPod but not tracked)
                    stray_count = 0
                    for pod in pods:
                        # Extract pod UUID from response
                        # Prefer the altname suffix (contains the true pod UUID). Avoid falling
                        # back to docker_ID because QuickPod destroy API expects the full UUID;
                        # using docker_ID leads to "permission" errors.
                        pod_uuid = None
                        altname = pod.get("altname", "")
                        if altname:
                            parts = altname.split("-")
                            if len(parts) >= 5:
                                pod_uuid = "-".join(parts[-5:])  # suffix is the actual pod UUID

                        # Secondary fallbacks
                        if not pod_uuid:
                            pod_uuid = pod.get("pod_uuid") or pod.get("uuid") or pod.get("id")
                        
                        # Convert to string (pod.get("id") can return an int)
                        if pod_uuid is not None:
                            pod_uuid = str(pod_uuid)

                        # If still not a UUID-looking string, skip to avoid bad destroy calls
                        if not pod_uuid or "-" not in pod_uuid:
                            logger.debug(
                                f"Stray pod cleanup: unable to extract pod_uuid from pod entry "
                                f"(altname={pod.get('altname')}, docker_ID={pod.get('docker_ID')}, id={pod.get('id')}). "
                                f"Skipping destroy to avoid permission errors."
                            )
                            continue
                        
                        # Skip if tracked (includes pods with active jobs)
                        if pod_uuid in tracked_pods:
                            continue
                        
                        # Skip if destroyed/exited
                        if pod.get("destroyed") or pod.get("state") == "exited":
                            continue
                        
                        # Skip if in recently destroyed tracker
                        if self.pod_manager.lifecycle.is_recently_destroyed(pod_uuid, window=600):
                            continue
                        
                        # DOUBLE CHECK: Query DB directly for any active jobs on this pod
                        db = SessionLocal()
                        try:
                            active_job_count = db.query(Job).join(
                                Worker, Job.worker_id == Worker.worker_id
                            ).filter(
                                Worker.container_id == pod_uuid,
                                Job.status.in_([
                                    JobState.RUNNING.value,
                                    JobState.DISPATCHING.value,
                                    JobState.QUEUED.value,
                                ])
                            ).count()
                            
                            if active_job_count > 0:
                                logger.info(
                                    f"Stray pod cleanup: Skipping pod {pod_uuid} - "
                                    f"has {active_job_count} active job(s) in DB"
                                )
                                continue
                        finally:
                            db.close()
                        
                        # This is a truly stray pod with no active jobs - destroy it
                        logger.warning(
                            f"Found stray pod {pod_uuid} (state={pod.get('state')}). "
                            f"Not in local tracking and no active jobs. "
                            f"Destroying to prevent cost leakage."
                        )
                        
                        try:
                            await self.pod_manager.lifecycle.destroy_pod(
                                pod_uuid,
                                graceful=False,
                                reason="Stray pod cleanup - not in local tracking, no active jobs"
                            )
                            stray_count += 1
                        except Exception as destroy_err:
                            logger.error(f"Failed to destroy stray pod {pod_uuid}: {destroy_err}")
                    
                    if stray_count > 0:
                        logger.info(f"Stray pod cleanup: destroyed {stray_count} stray pod(s)")
                    
                except Exception as e:
                    logger.warning(f"Stray pod cleanup error: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Stray pod cleanup task error")
    
    async def _ban_cleanup_task(self):
        """Clean up expired IP bans"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Every minute
                await ip_ban_manager.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Ban cleanup task error")
    
    async def _r2_input_media_reaper_task(self):
        """
        Safety-net reaper: delete orphaned objects under input-media/ and
        input-videos/ in R2 that are older than 2 hours.  This catches cases
        where job_complete cleanup was missed (e.g. server crash, timeout).
        
        Two prefixes are checked:
          - input-media/  : images and videos uploaded via /get_media_upload_urls
          - input-videos/ : videos uploaded via legacy /get_video_upload_url
        
        Runs every 60 minutes.
        """
        from datetime import datetime
        from routes import _s3_client, _r2_bucket, _image_prefix
        
        reaper_interval = 3600  # 60 minutes
        max_age_seconds = 7200  # 2 hours
        
        logger.info(
            f"R2 input media reaper started (interval={reaper_interval}s, "
            f"max_age={max_age_seconds}s)"
        )
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(reaper_interval)
                
                if not _s3_client or not _r2_bucket:
                    continue
                
                # Scan both input-media/ (new) and input-videos/ (legacy) prefixes
                prefixes = [
                    f"{_image_prefix}/input-media/",
                    f"{_image_prefix}/input-videos/",
                ]
                now = datetime.utcnow()
                total_deleted = 0
                
                for prefix in prefixes:
                    try:
                        paginator = _s3_client.get_paginator("list_objects_v2")
                        for page in paginator.paginate(Bucket=_r2_bucket, Prefix=prefix):
                            for obj in page.get("Contents", []):
                                last_modified = obj.get("LastModified")
                                if last_modified:
                                    # Make comparison timezone-naive
                                    if last_modified.tzinfo:
                                        last_modified = last_modified.replace(tzinfo=None)
                                    age = (now - last_modified).total_seconds()
                                    if age > max_age_seconds:
                                        try:
                                            _s3_client.delete_object(
                                                Bucket=_r2_bucket, Key=obj["Key"]
                                            )
                                            total_deleted += 1
                                        except Exception as e:
                                            logger.warning(
                                                f"[R2_REAPER] Failed to delete {obj['Key']}: {e}"
                                            )
                    except Exception as e:
                        logger.warning(f"[R2_REAPER] Failed to list/clean {prefix}: {e}")
                
                if total_deleted > 0:
                    logger.info(
                        f"[R2_REAPER] Cleaned up {total_deleted} orphaned input objects"
                    )
                    
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("R2 input media reaper task error")
    
    async def _job_recovery_task(self):
        """Check for stuck jobs and recover them"""
        sweep_interval = 30  # Check every 30 seconds
        
        logger.info(
            f"Job recovery task started (interval={sweep_interval}s, "
            f"running_timeout={self.job_manager.running_job_timeout.total_seconds() / 60:.0f} min, "
            f"general_timeout={self.job_manager.job_timeout.total_seconds() / 60:.0f} min)"
        )
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(sweep_interval)
                
                if self.job_manager:
                    await self.job_manager.check_for_stuck_jobs()
                    
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Job recovery task error")

    async def _queued_job_dispatcher_task(self):
        """
        CRITICAL: Dispatch QUEUED jobs to available workers.
        
        This task handles jobs that were reverted to QUEUED after:
        - Dispatch failures (e.g., all machines blacklisted)
        - Pod destruction during dispatch
        - Worker timeout during registration
        
        Without this task, QUEUED jobs would sit indefinitely waiting
        for the 30-minute timeout, even when workers are available.
        """
        from db import SessionLocal, Job, Worker
        from models import JobState
        from routes import _dispatch_job_background
        
        dispatch_interval = 5  # Check every 5 seconds
        
        # Track recently dispatched jobs to avoid re-dispatching too quickly
        # Maps job_id -> timestamp when dispatch was triggered
        recently_dispatched: dict = {}
        dispatch_cooldown = 10  # Don't re-dispatch same job within 10 seconds
        
        logger.info(
            f"Queued job dispatcher task started (interval={dispatch_interval}s)"
        )
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(dispatch_interval)
                
                if not self.pod_manager:
                    continue
                
                # Clean up old entries from recently_dispatched
                from utils import utc_naive_now
                from datetime import timedelta
                
                now = utc_naive_now()
                cutoff = now - timedelta(seconds=dispatch_cooldown * 2)
                recently_dispatched = {
                    k: v for k, v in recently_dispatched.items() 
                    if v > cutoff
                }
                
                # Get QUEUED jobs that need dispatch
                db = SessionLocal()
                try:
                    # Find QUEUED jobs without worker assignment
                    # Only get jobs that have been QUEUED for at least 3 seconds
                    # (to avoid racing with initial dispatch)
                    min_age = now - timedelta(seconds=3)
                    
                    queued_jobs = (
                        db.query(Job)
                        .filter(
                            Job.status == JobState.QUEUED.value,
                            Job.worker_id == None,  # noqa: E711
                            Job.updated_at < min_age,  # At least 3 seconds old
                        )
                        .order_by(Job.created_at.asc())  # FIFO
                        .limit(10)  # Process up to 10 jobs per cycle
                        .all()
                    )
                    
                    # Filter out recently dispatched jobs
                    cooldown_threshold = now - timedelta(seconds=dispatch_cooldown)
                    eligible_jobs = []
                    for job in queued_jobs:
                        last_dispatch = recently_dispatched.get(job.job_id)
                        if last_dispatch is None or last_dispatch < cooldown_threshold:
                            eligible_jobs.append(job)
                    
                    if eligible_jobs:
                        logger.info(
                            f"[DISPATCHER] Found {len(eligible_jobs)} QUEUED jobs to dispatch "
                            f"(filtered from {len(queued_jobs)})"
                        )
                    
                    # Group jobs by model for efficient dispatch
                    jobs_by_model: dict = {}
                    for job in eligible_jobs:
                        if job.model not in jobs_by_model:
                            jobs_by_model[job.model] = []
                        jobs_by_model[job.model].append(job)
                    
                    # Check for available workers for each model
                    for model, jobs in jobs_by_model.items():
                        # Always attempt dispatch for eligible QUEUED jobs.
                        # _dispatch_job_background handles finding/creating pods.
                        for job in jobs:
                            logger.info(
                                f"[DISPATCHER] Triggering dispatch for QUEUED job {job.job_id} "
                                f"(model={model}, queued_since={job.updated_at})"
                            )
                            
                            # Mark as recently dispatched
                            recently_dispatched[job.job_id] = now
                            
                            # Get job mode from params
                            job_mode = None
                            if job.params and isinstance(job.params, dict):
                                job_mode = job.params.get("mode")
                            
                            # Dispatch in background
                            asyncio.create_task(
                                _dispatch_job_background(job.job_id, model, job_mode)
                            )
                            
                            # Small delay between dispatches to avoid thundering herd
                            await asyncio.sleep(0.5)
                finally:
                    db.close()
                    
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Queued job dispatcher task error")


# =============================================================================
# Application Factory
# =============================================================================

app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await app_state.initialize()
    await app_state.start_background_tasks()
    
    yield
    
    # Shutdown
    await app_state.shutdown()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    # Get debug mode from env
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    # Create app
    app = FastAPI(
        title="Server Controller",
        description="Client-Server-Worker Orchestrator for AI Generation",
        version=os.getenv("APP_VERSION", "1.0.0"),
        debug=debug,
        lifespan=lifespan,
        # Disable docs in production
        docs_url="/docs" if debug else None,
        redoc_url="/redoc" if debug else None,
        openapi_url="/openapi.json" if debug else None,
    )
    
    # Setup middleware
    setup_middleware(app, debug=debug)
    
    # Include routers
    app.include_router(public_router)
    app.include_router(client_router)
    app.include_router(worker_router)
    app.include_router(admin_router)
    app.include_router(auth_router)
    app.include_router(ads_router)
    
    return app


# Create application instance
app = create_app()


# =============================================================================
# Signal Handlers
# =============================================================================

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    
    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        app_state.shutdown_event.set()
    
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    setup_signal_handlers()
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info",
    )

