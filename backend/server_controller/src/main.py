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
    admin_router,
    set_dependencies,
    sync_heartbeat_cache_to_db,  # Background task for heartbeat caching
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
        
        # Heartbeat cache sync task (CRITICAL - makes heartbeat endpoint fast)
        task = asyncio.create_task(
            sync_heartbeat_cache_to_db(),
            name="heartbeat_sync"
        )
        self.background_tasks.append(task)
        logger.info("Started heartbeat cache sync task")
        
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
        idle_timeout_secs = self.config.worker.idle_timeout
        # INCREASED: 60s interval to reduce load on weak servers (was 30s)
        sweep_interval = 60
        
        logger.info(
            f"Idle worker reaper: stale>{heartbeat_stale_secs}s, "
            f"idle>{idle_timeout_secs}s"
        )
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(sweep_interval)
                
                db = SessionLocal()
                now = datetime.utcnow()
                stale_cutoff = now - timedelta(seconds=heartbeat_stale_secs)
                
                try:
                    workers = db.query(Worker).filter(
                        Worker.status.in_(["ready", "busy", "processing"])
                    ).all()
                    
                    for w in workers:
                        try:
                            # Stale check
                            if not w.last_heartbeat or w.last_heartbeat < stale_cutoff:
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
                                
                                if active_jobs == 0:
                                    # Worker says busy but no jobs - might be stuck, reset to ready
                                    logger.warning(
                                        f"Worker {w.worker_id} status is {w.status} but has no active jobs, "
                                        f"resetting to ready"
                                    )
                                    w.status = "ready"
                                    db.commit()
                                else:
                                    logger.debug(
                                        f"Worker {w.worker_id} is {w.status} with {active_jobs} job(s), "
                                        f"skipping idle check"
                                    )
                                continue
                            
                            # CRITICAL: Handle workers stuck in "starting" state
                            # Workers in "starting" are loading models - give them time, but not forever
                            if w.status == "starting":
                                worker_age = (now - w.created_at).total_seconds() if w.created_at else 0
                                
                                # Allow 5 minutes for model loading (models can be large)
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
                                
                                # Check for pending jobs for this model (but not assigned yet)
                                pending = db.query(Job).filter(
                                    Job.model == w.model,
                                    Job.worker_id.is_(None),  # Not assigned yet
                                    Job.status.in_([JobState.QUEUED.value, JobState.DISPATCHING.value])
                                ).count()
                                
                                if pending > 0:
                                    logger.debug(
                                        f"Model {w.model} has {pending} pending job(s), "
                                        f"keeping worker {w.worker_id} alive"
                                    )
                                    continue
                                
                                # Check idle time
                                last_job = w.last_job_finished_at or w.created_at or w.last_active
                                idle_for = (now - last_job).total_seconds()
                                
                                # Add cooldown period after worker creation to avoid premature destruction
                                worker_age = (now - w.created_at).total_seconds() if w.created_at else 0
                                min_worker_age = 120  # 2 minutes minimum lifetime (increased from 60s)
                                
                                if worker_age < min_worker_age:
                                    logger.debug(
                                        f"Worker {w.worker_id} is only {worker_age:.1f}s old, "
                                        f"minimum age is {min_worker_age}s, skipping idle check"
                                    )
                                    continue
                                
                                # CRITICAL: Truly adaptive idle timeout balancing cost vs. readiness
                                # Goal: Aggressive destruction when no work, patient when work exists
                                
                                total_pending_jobs = db.query(Job).filter(
                                    Job.status.in_([JobState.QUEUED.value, JobState.PENDING.value])
                                ).count()
                                
                                # Check recent job submission rate (jobs created in last 2 minutes)
                                recent_cutoff = now - timedelta(minutes=2)
                                recent_jobs = db.query(Job).filter(
                                    Job.model == w.model,
                                    Job.created_at >= recent_cutoff
                                ).count()
                                
                                # SMART ADAPTIVE TIMEOUT:
                                # - No queue + no recent jobs = aggressive (90s) - save money!
                                # - Pending jobs OR recent activity = patient (300-600s) - avoid startup costs
                                
                                if total_pending_jobs > 5:
                                    # High load: 10 minute idle timeout (worth keeping worker ready)
                                    adaptive_idle_timeout = 300
                                    timeout_reason = f"high load ({total_pending_jobs} pending)"
                                elif total_pending_jobs > 0:
                                    # Medium load: 5 minute idle timeout
                                    adaptive_idle_timeout = 100
                                    timeout_reason = f"pending jobs ({total_pending_jobs})"
                                elif recent_jobs > 0:
                                    # Recent activity but no queue: 3 minute timeout
                                    # Keep worker alive in case more jobs come
                                    adaptive_idle_timeout = 30
                                    timeout_reason = f"recent activity ({recent_jobs} jobs in last 2min)"
                                else:
                                    # NO LOAD: aggressive timeout to save money
                                    # 90 seconds = enough for genuine brief gaps, but destroys truly idle
                                    adaptive_idle_timeout = 20
                                    timeout_reason = "no load (no pending, no recent)"
                                
                                logger.debug(
                                    f"Worker {w.worker_id} adaptive timeout: {adaptive_idle_timeout}s "
                                    f"({timeout_reason})"
                                )
                                
                                if idle_for > adaptive_idle_timeout:
                                    logger.info(
                                        f"Worker {w.worker_id} idle for {idle_for:.1f}s "
                                        f"(timeout: {adaptive_idle_timeout}s, pending_jobs: {total_pending_jobs}), "
                                        f"destroying pod {w.container_id}"
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
                                        f"but within timeout ({adaptive_idle_timeout}s)"
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
    
    async def _job_recovery_task(self):
        """Check for stuck jobs and recover them"""
        sweep_interval = 30  # Check every 30 seconds
        
        logger.info(
            f"Job recovery task started (interval={sweep_interval}s, "
            f"timeout={self.job_manager.job_timeout.total_seconds() / 60:.0f} min)"
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

