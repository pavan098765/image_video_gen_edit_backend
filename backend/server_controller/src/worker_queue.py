# worker_queue.py
from datetime import datetime
from sqlalchemy import select
from db import SessionLocal, Worker
import logging

logger = logging.getLogger(__name__)

async def atomic_increment_worker_queue(worker_id: str, pod_manager=None):
    """Atomically increment queue_size for worker_id in DB, return (new_q, container_id)."""
    db = SessionLocal()
    try:
        # Lock the row to avoid races
        w = db.query(Worker).filter(Worker.worker_id == worker_id).with_for_update().first()
        if not w:
            db.rollback()
            return None, None
        w.queue_size = (w.queue_size or 0) + 1
        w.last_active = datetime.utcnow()
        # mark dispatching/busy as caller expects
        w.status = "dispatching"
        db.commit()
        container = getattr(w, "container_id", None)
        new_q = w.queue_size
    except Exception:
        try:
            db.rollback()
        except:
            pass
        logger.exception("atomic_increment_worker_queue failed for %s", worker_id)
        return None, None
    finally:
        try:
            db.close()
        except:
            pass

    # Mirror into in-memory pod_manager if provided
    if pod_manager and container:
        async def _mirror():
            async with pod_manager._lock:
                pi = pod_manager.individual_pods.get(container)
                if pi:
                    pi.queue_size = getattr(pi, "queue_size", 0) + 1
                    pod_manager.update_pod_job_counts(pi, None)  # optional: track job ids elsewhere
        # schedule mirror (we are in async context)
        try:
            import asyncio
            asyncio.create_task(_mirror())
        except Exception:
            # if scheduling fails, it's okay; consistency will be re-balanced later
            pass

    return new_q, container


async def atomic_decrement_worker_queue(worker_id: str, pod_manager=None):
    """Atomically decrement queue_size for worker_id in DB, return (new_q, container_id)."""
    db = SessionLocal()
    try:
        w = db.query(Worker).filter(Worker.worker_id == worker_id).with_for_update().first()
        if not w:
            db.rollback()
            return None, None
        new_q = max(0, (w.queue_size or 0) - 1)
        w.queue_size = new_q
        w.last_active = datetime.utcnow()
        # If container removed and queue==0, set offline/ready appropriately is done by caller
        db.commit()
        container = getattr(w, "container_id", None)
    except Exception:
        try:
            db.rollback()
        except:
            pass
        logger.exception("atomic_decrement_worker_queue failed for %s", worker_id)
        return None, None
    finally:
        try:
            db.close()
        except:
            pass

    # Mirror into in-memory pod_manager if provided
    if pod_manager and container:
        async def _mirror():
            async with pod_manager._lock:
                pi = pod_manager.individual_pods.get(container)
                if pi:
                    pi.queue_size = max(0, (pi.queue_size or 0) - 1)
                    # optionally remove specific job id via pod_manager.update_pod_job_counts
        try:
            import asyncio
            asyncio.create_task(_mirror())
        except Exception:
            pass

    return new_q, container
