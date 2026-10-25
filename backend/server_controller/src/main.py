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
from scheduler_strategy import build_scheduler_strategy
from security import ip_ban_manager
from studio_routes import studio_router

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
        self._ttl_destroy_inflight: set = set()
    
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
        
        # Crypto mode banner: both sides must agree. Mismatched modes fail on the first
        # register/heartbeat call, so emit the current mode loudly on boot.
        try:
            from worker_transport_crypto import (
                crypto_enabled as _wt_crypto_enabled,
                transport_crypto_explicitly_on as _wt_explicit_on,
                get_fernet as _wt_get_fernet,
            )
            _wt_env = os.getenv("WORKER_TRANSPORT_CRYPTO", "").strip()
            _wt_has_key = _wt_get_fernet() is not None
            logger.info(
                "WORKER_TRANSPORT_CRYPTO=%r explicit_on=%s fernet_key_available=%s payload_encryption=%s "
                "(controller and workers MUST match; set WORKER_TRANSPORT_CRYPTO=1 on both sides)",
                _wt_env,
                _wt_explicit_on(),
                _wt_has_key,
                _wt_crypto_enabled(),
            )
        except Exception as _crypto_log_err:
            logger.warning("Could not log worker transport crypto mode: %s", _crypto_log_err)

        # Initialize Pod Manager
        logger.info("Initializing Pod Manager...")
        self.pod_manager = PodManager(
            email=self.config.quickpod.email,
            password=self.config.quickpod.password,
            warm_pod_uuid=self.config.quickpod.warm_pod_uuid if self.config.pod.strategy.value == "warm_and_individual" else None,
        )
        await self.pod_manager.initialize()

        # Shared-bake drift check: controller FLEET_OMNI_SHARED_BAKE_MODELS must mirror
        # worker_multi_common/src/omni_config.py OMNI_SHARED_BAKED_MODELS. Worker advertises
        # them via register_worker + heartbeat; if the two diverge, the scheduler routes to
        # pods that are missing baked weights (or conservatively skips pods that have them).
        _WORKER_SHARED_BAKE_MIRROR = frozenset(
            {
                "basic_utils",
                "aurasrv2",
                "swin2sr",
                "video_upscaler",
                "face_enhancer",
                "glm_ocr",
            }
        )
        _ctrl_set = set(PodManager.FLEET_OMNI_SHARED_BAKE_MODELS)
        _worker_set = set(_WORKER_SHARED_BAKE_MIRROR)
        if _ctrl_set != _worker_set:
            missing_in_ctrl = sorted(_worker_set - _ctrl_set)
            extra_in_ctrl = sorted(_ctrl_set - _worker_set)
            logger.warning(
                "SHARED_BAKE_DRIFT FLEET_OMNI_SHARED_BAKE_MODELS=%s != worker OMNI_SHARED_BAKED_MODELS=%s "
                "(missing in controller: %s; extra in controller: %s) — "
                "sync backend/server_controller/src/pod_manager.py with "
                "backend/quickpod_individual_worker/worker_multi_common/src/omni_config.py",
                sorted(_ctrl_set),
                sorted(_worker_set),
                missing_in_ctrl,
                extra_in_ctrl,
            )
        else:
            logger.info(
                "Shared-bake parity OK (controller FLEET_OMNI_SHARED_BAKE_MODELS == worker OMNI_SHARED_BAKED_MODELS): %s",
                sorted(_ctrl_set),
            )
        
        # Initialize Job Manager
        logger.info("Initializing Job Manager...")
        db_manager = get_db_manager()
        scheduler_router = build_scheduler_strategy(
            pod_manager=self.pod_manager,
            scheduler_mode=self.config.scheduler.mode.value,
            shadow_mode=self.config.scheduler.shadow_mode,
        )
        self.job_manager = JobManager(
            session_factory=db_manager.session_factory,
            pod_manager=self.pod_manager,
            scheduler_router=scheduler_router,
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

        # Fixed-fleet TTL enforcer (hard pod cutoff + drain marker)
        if (
            self.pod_manager
            and self.config.scheduler.mode.value == "fixed_fleet_windowed"
        ):
            task = asyncio.create_task(
                self._fixed_fleet_ttl_enforcer_task(),
                name="fixed_fleet_ttl_enforcer",
            )
            self.background_tasks.append(task)
            logger.info("Started fixed fleet TTL enforcer task")
        
        # R2 input media reaper (cleanup orphaned input-media objects)
        task = asyncio.create_task(
            self._r2_input_media_reaper_task(),
            name="r2_input_media_reaper"
        )
        self.background_tasks.append(task)
        logger.info("Started R2 input media reaper task")
        
        # DB cleanup (delete old rows from workers, jobs, rewarded_ad_events, daily_spins)
        task = asyncio.create_task(
            self._db_cleanup_task(),
            name="db_cleanup"
        )
        self.background_tasks.append(task)
        logger.info("Started DB cleanup task")

        task = asyncio.create_task(
            self._kie_recordinfo_sweeper_task(),
            name="kie_recordinfo_sweeper",
        )
        self.background_tasks.append(task)
        logger.info("Started Kie recordInfo fallback sweeper")

        task = asyncio.create_task(
            self._runware_poll_sweeper_task(),
            name="runware_poll_sweeper",
        )
        self.background_tasks.append(task)
        logger.info("Started Runware getResponse sweeper")
        
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

    async def _kie_recordinfo_sweeper_task(self):
        """Once-per-task fallback if Kie never POSTs callBackUrl."""
        from datetime import timedelta
        from db import SessionLocal, Job, KieTask, utc_naive_now
        from routes import _apply_kie_job_result
        import kie_client

        await asyncio.sleep(60)
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(90)
                db = SessionLocal()
                try:
                    cutoff = utc_naive_now() - timedelta(minutes=8)
                    stale = (
                        db.query(KieTask)
                        .filter(
                            KieTask.callback_received == False,  # noqa: E712
                            KieTask.created_at < cutoff,
                        )
                        .limit(20)
                        .all()
                    )
                    for rec in stale:
                        try:
                            payload = kie_client.record_info(rec.task_id)
                        except Exception:
                            logger.exception("Kie recordInfo failed for %s", rec.task_id)
                            continue
                        if rec.model == "suno_v6":
                            import suno_v6
                            from datetime import timedelta as _td
                            event = suno_v6.interpret(payload)
                            disposition = event.get("disposition")
                            created = rec.created_at.replace(tzinfo=None) if rec.created_at and rec.created_at.tzinfo else rec.created_at
                            age = utc_naive_now() - created if created else _td(0)
                            job = db.query(Job).filter(Job.job_id == rec.job_id).first()
                            params = job.params if job and isinstance(job.params, dict) else {}
                            context = params.get("_suno") if isinstance(params.get("_suno"), dict) else {}
                            music = params.get("_music") if isinstance(params.get("_music"), dict) else {}
                            action = context.get("action") or music.get("action") or "song"
                            tracks = list(music.get("tracks") or [])
                            if event.get("tracks"):
                                tracks = suno_v6._merge_tracks(tracks, event.get("tracks") or [])
                            playable = sum(1 for track in tracks if suno_v6._track_has_audio(track))
                            # One clip is not a finished HitMaker song. Keep polling
                            # until the second arrives, then accept the single clip
                            # after a long wait instead of failing a playable song.
                            if (
                                action == "song"
                                and playable == 1
                                and age >= _td(minutes=12)
                                and disposition in ("pending", "progress", "success")
                                and not context.get("accept_single_track")
                                and job is not None
                            ):
                                context = dict(context)
                                context["accept_single_track"] = True
                                params = dict(params)
                                params["_suno"] = context
                                job.params = params
                                db.commit()
                            holding = suno_v6.hold_for_second_version(context, music, event)
                            if holding or (
                                disposition in ("pending", "progress")
                                and age < _td(minutes=20)
                                and not context.get("accept_single_track")
                            ):
                                if disposition in ("progress", "success") or holding:
                                    await _apply_kie_job_result(rec.job_id, payload)
                                continue
                            if disposition in ("pending", "progress") and playable < 1:
                                payload = {
                                    "code": 500,
                                    "msg": "Music generation timed out",
                                    "data": {"state": "fail", "failMsg": "Music generation timed out"},
                                }
                        rec.callback_received = True
                        rec.updated_at = utc_naive_now()
                        db.commit()
                        await _apply_kie_job_result(rec.job_id, payload)
                finally:
                    db.close()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Kie recordInfo sweeper error")

    async def _runware_poll_sweeper_task(self):
        """Poll getResponse when a webhook does not arrive."""
        from datetime import timedelta
        from db import SessionLocal, RunwareTask, utc_naive_now
        from routes import (
            _apply_runware_job_result,
            _claim_runware_task,
            _release_runware_task,
        )
        import runware_client

        await asyncio.sleep(20)
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(10)
                db = SessionLocal()
                try:
                    cutoff = utc_naive_now() - timedelta(seconds=8)
                    stale = (
                        db.query(RunwareTask)
                        .filter(
                            RunwareTask.callback_received == False,  # noqa: E712
                            RunwareTask.created_at < cutoff,
                        )
                        .limit(20)
                        .all()
                    )
                    pending = [(rec.task_id, rec.job_id, rec.created_at) for rec in stale]
                finally:
                    db.close()
                for task_id, job_id, created_at in pending:
                    try:
                        payload = await asyncio.to_thread(runware_client.get_response, task_id)
                    except Exception:
                        logger.exception("Runware getResponse failed for %s", task_id)
                        continue
                    created = created_at.replace(tzinfo=None) if created_at and created_at.tzinfo else created_at
                    age = utc_naive_now() - created if created else timedelta(0)
                    if runware_client.is_processing(payload):
                        if age > timedelta(minutes=25):
                            payload = {
                                "errors": [
                                    {
                                        "taskUUID": task_id,
                                        "message": "Runware task timed out",
                                    }
                                ]
                            }
                        else:
                            continue
                    claimed = _claim_runware_task(task_id)
                    if not claimed:
                        continue
                    outcome = await _apply_runware_job_result(claimed, payload)
                    if outcome == "processing":
                        _release_runware_task(task_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Runware poll sweeper error")

    async def _idle_worker_reaper_task(self):
        """Reap idle workers"""
        from db import SessionLocal, Worker, Job
        from datetime import datetime, timedelta
        from job_manager import JobState
        
        heartbeat_stale_secs = self.config.worker.heartbeat_stale_seconds
        idle_cfg = self.config.worker
        sweep_interval = 30
        fixed_fleet_mode = self.config.scheduler.mode.value == "fixed_fleet_windowed"
        
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
                            if (w.model or "").strip().lower() == "studio_engine":
                                cached_hb = get_cached_last_heartbeat(w.worker_id)
                                effective_hb = cached_hb or w.last_heartbeat
                                if effective_hb and effective_hb >= stale_cutoff:
                                    continue
                                logger.warning(
                                    f"Studio engine worker {w.worker_id} orphan heartbeat; "
                                    "deleting Vultr instance (not QuickPod)"
                                )
                                from vultr_pool import pool as vultr_pool

                                vid = None
                                if isinstance(w.info, dict):
                                    vid = w.info.get("vultr_id")
                                vid = vid or w.container_id
                                if vid:
                                    vultr_pool.destroy_instance(vid)
                                w.status = "offline"
                                db.commit()
                                continue
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
                                stale_age = (now - effective_hb).total_seconds() if effective_hb else 9999
                                max_stale_with_jobs = heartbeat_stale_secs * 3

                                active_running = db.query(Job).filter(
                                    Job.worker_id == w.worker_id,
                                    Job.status.in_([
                                        JobState.RUNNING.value,
                                        JobState.DISPATCHING.value
                                    ])
                                ).count()
                                if active_running > 0 and stale_age < max_stale_with_jobs:
                                    logger.warning(
                                        f"Worker {w.worker_id} heartbeat stale ({stale_age:.0f}s) but has "
                                        f"{active_running} active RUNNING/DISPATCHING job(s). "
                                        f"Skipping destroy (max grace: {max_stale_with_jobs}s)."
                                    )
                                    continue
                                elif active_running > 0:
                                    logger.error(
                                        f"Worker {w.worker_id} heartbeat stale for {stale_age:.0f}s "
                                        f"(>{max_stale_with_jobs}s grace) with {active_running} active job(s). "
                                        f"Worker likely dead, proceeding with pod destruction."
                                    )

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
                                
                                # Do not destroy if there are PENDING or orphaned QUEUED jobs for this model
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

                                    orphaned_queued = db.query(Job).filter(
                                        Job.model == w.model,
                                        Job.status == JobState.QUEUED.value,
                                        Job.worker_id.is_(None),
                                    ).count()
                                    if orphaned_queued > 0:
                                        logger.warning(
                                            f"Worker {w.worker_id} model {w.model} has {orphaned_queued} "
                                            f"orphaned QUEUED job(s) (worker_id=None). "
                                            f"Skipping idle destroy; triggering dispatch."
                                        )
                                        orphaned_jobs = db.query(Job).filter(
                                            Job.model == w.model,
                                            Job.status == JobState.QUEUED.value,
                                            Job.worker_id.is_(None),
                                        ).all()
                                        for oj in orphaned_jobs:
                                            op = oj.params if isinstance(oj.params, dict) else {}
                                            if op.get("on_device") is True:
                                                continue
                                            try:
                                                import importlib
                                                routes = importlib.import_module("routes")
                                                dispatch_fn = getattr(routes, "_dispatch_job_background", None)
                                                if dispatch_fn:
                                                    asyncio.create_task(dispatch_fn(oj.job_id, oj.model, None))
                                            except Exception as e:
                                                logger.warning(f"Failed to dispatch orphaned job {oj.job_id}: {e}")
                                        continue
                                
                                # Idle timeout: pod completed all assigned jobs, no new jobs came to it.
                                # Destroy only if idle for configured timeout (errors handled separately).
                                from gpu_providers.ttl import (
                                    idle_timeout_seconds_for_worker,
                                    provider_of_worker,
                                )
                                if fixed_fleet_mode and provider_of_worker(w) in ("", "quickpod"):
                                    logger.debug(
                                        f"Fixed fleet mode: skipping idle-time destroy for worker {w.worker_id}"
                                    )
                                    continue

                                last_job = w.last_job_finished_at or w.created_at or w.last_active
                                idle_for = (now - last_job).total_seconds() if last_job else 0
                                worker_age = (now - w.created_at).total_seconds() if w.created_at else 0

                                timeout_secs = idle_timeout_seconds_for_worker(
                                    w, idle_cfg.idle_timeout_seconds
                                )
                                min_age = (
                                    0
                                    if timeout_secs == 0
                                    else idle_cfg.idle_min_age_seconds
                                )
                                
                                if worker_age < min_age:
                                    logger.debug(
                                        f"Worker {w.worker_id} is only {worker_age:.1f}s old, "
                                        f"min_age={min_age}s, skipping idle check"
                                    )
                                    continue
                                
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

    async def _fixed_fleet_ttl_enforcer_task(self):
        """
        Enforce fixed-fleet pod lifecycle:
        - mark pods as drain mode near TTL
        - hard-destroy pods at/after TTL
        """
        from datetime import datetime

        sweep_interval = 1
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(sweep_interval)
                if not self.pod_manager:
                    continue

                now = datetime.utcnow()
                to_destroy = []
                ttl_seconds = self.pod_manager.pod_max_lifetime_seconds
                drain_guard = self.pod_manager.pod_drain_guard_seconds

                async with self.pod_manager._lock:
                    for pod_uuid, pi in self.pod_manager.individual_pods.items():
                        if pi.meta.get("is_cache_refresh_pod"):
                            continue
                        age_seconds = (now - pi.created_at).total_seconds()
                        remaining_seconds = ttl_seconds - age_seconds

                        if remaining_seconds <= drain_guard and not pi.meta.get("drain_mode"):
                            pi.meta["drain_mode"] = True
                            pi.meta["drain_since"] = now.isoformat()
                            logger.info(
                                "Fixed fleet drain mode: pod=%s machine=%s age=%.1fs remaining=%.1fs",
                                pod_uuid,
                                pi.machine_id,
                                age_seconds,
                                remaining_seconds,
                            )

                        if age_seconds >= ttl_seconds:
                            to_destroy.append((pod_uuid, age_seconds))

                # Deterministic ordering: oldest pods first, then UUID.
                to_destroy.sort(key=lambda x: (-x[1], x[0]))
                for pod_uuid, age_seconds in to_destroy:
                    if pod_uuid in self._ttl_destroy_inflight:
                        continue
                    self._ttl_destroy_inflight.add(pod_uuid)
                    try:
                        logger.warning(
                            "Fixed fleet hard TTL cutoff: destroying pod=%s age=%.1fs (ttl=%ss)",
                            pod_uuid,
                            age_seconds,
                            ttl_seconds,
                        )
                        if hasattr(self.pod_manager, "fixed_fleet_metrics"):
                            self.pod_manager.fixed_fleet_metrics["ttl_cutoff_destroyed"] = (
                                self.pod_manager.fixed_fleet_metrics.get("ttl_cutoff_destroyed", 0) + 1
                            )
                        await self.pod_manager.destroy_individual_pod(
                            pod_uuid,
                            reason=f"Fixed fleet hard TTL cutoff ({age_seconds:.0f}s)",
                        )
                    except Exception as e:
                        logger.error(
                            "Fixed fleet TTL destroy failed for pod=%s err=%s",
                            pod_uuid,
                            e,
                        )
                    finally:
                        self._ttl_destroy_inflight.discard(pod_uuid)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Fixed fleet TTL enforcer error")
    
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
                    # Fetch all pods from QuickPod.
                    # Reuse the shared manager so we don't force-login every sweep.
                    # QuickPodManager.get_my_pods() already refreshes auth only when needed.
                    qp = self.pod_manager.global_qp
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
    
    async def _db_cleanup_task(self):
        """
        Periodically delete old rows from workers, jobs, rewarded_ad_events,
        and daily_spins to prevent DB growth. Runs once per week.
        Uses batched deletes to avoid long-running transactions.
        """
        from db_cleanup import run_db_cleanup

        interval_days = int(os.getenv("DB_CLEANUP_INTERVAL_DAYS", "7"))
        retention_days = int(os.getenv("DB_CLEANUP_RETENTION_DAYS", "7"))
        batch_size = int(os.getenv("DB_CLEANUP_BATCH_SIZE", "500"))
        interval_seconds = interval_days * 24 * 3600

        logger.info(
            f"DB cleanup task started (interval={interval_days}d, "
            f"retention={retention_days}d, batch_size={batch_size})"
        )

        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(interval_seconds)

                try:
                    results = await asyncio.to_thread(
                        run_db_cleanup,
                        retention_days=retention_days,
                        batch_size=batch_size,
                    )
                    total = sum(results.values())
                    if total > 0:
                        logger.info(
                            f"[DB_CLEANUP] Deleted old rows: workers={results['workers']}, "
                            f"jobs={results['jobs']}, rewarded_ad_events={results['rewarded_ad_events']}, "
                            f"daily_spins={results['daily_spins']} (total={total})"
                        )
                except Exception as e:
                    logger.exception(f"[DB_CLEANUP] Failed: {e}")

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("DB cleanup task error")

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
        from db import SessionLocal, Job, Worker, utc_naive_now
        from datetime import timedelta
        from job_manager import JobState
        from routes import _dispatch_job_background
        
        dispatch_interval = 5  # Check every 5 seconds
        fairness_enabled = bool(self.config.job.queue_fairness_enabled)
        fairness_max_consecutive = max(
            1, int(self.config.job.queue_fairness_max_consecutive_per_model)
        )
        
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
                        jp = job.params if isinstance(job.params, dict) else {}
                        if jp.get("on_device") is True:
                            continue
                        if (job.model or "").strip().lower() in {"studio", "studio_engine"}:
                            continue
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
                    model_order: list = []
                    for job in eligible_jobs:
                        if job.model not in jobs_by_model:
                            jobs_by_model[job.model] = []
                            model_order.append(job.model)
                        jobs_by_model[job.model].append(job)

                    # Build fair dispatch sequence (bounded consecutive per model).
                    dispatch_sequence = []
                    if fairness_enabled:
                        cursor = 0
                        last_model = None
                        consecutive_for_last = 0

                        while True:
                            active_models = [m for m in model_order if jobs_by_model.get(m)]
                            if not active_models:
                                break

                            selected_model = None
                            for offset in range(len(model_order)):
                                candidate = model_order[(cursor + offset) % len(model_order)]
                                if not jobs_by_model.get(candidate):
                                    continue
                                if (
                                    candidate == last_model
                                    and consecutive_for_last >= fairness_max_consecutive
                                ):
                                    continue
                                selected_model = candidate
                                cursor = (cursor + offset + 1) % len(model_order)
                                break

                            if selected_model is None:
                                # All non-empty models were blocked by consecutive cap.
                                # Reset streak to preserve progress deterministically.
                                last_model = None
                                consecutive_for_last = 0
                                continue

                            dispatch_sequence.append((selected_model, jobs_by_model[selected_model].pop(0)))
                            if selected_model == last_model:
                                consecutive_for_last += 1
                            else:
                                last_model = selected_model
                                consecutive_for_last = 1
                    else:
                        for model in model_order:
                            for job in jobs_by_model.get(model, []):
                                dispatch_sequence.append((model, job))

                    for model, job in dispatch_sequence:
                        logger.info(
                            f"[DISPATCHER] Triggering dispatch for QUEUED job {job.job_id} "
                            f"(model={model}, queued_since={job.updated_at}, fairness={fairness_enabled})"
                        )
                        if fairness_enabled and self.pod_manager and hasattr(self.pod_manager, "fixed_fleet_metrics"):
                            metrics = self.pod_manager.fixed_fleet_metrics
                            metrics["fairness_dispatch_total"] = int(metrics.get("fairness_dispatch_total", 0)) + 1

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
    app.include_router(studio_router)
    
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

