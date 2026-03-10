# db_cleanup.py
"""
Database Cleanup Service

Periodically deletes rows older than retention period from workers, jobs,
rewarded_ad_events, and daily_spins to prevent DB growth. Uses batched
deletes to avoid long-running transactions.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict

from sqlalchemy import and_, or_

from db import (
    Worker,
    Job,
    RewardedAdEvent,
    DailySpin,
    get_db_manager,
)
from job_manager import JobState

logger = logging.getLogger("db_cleanup")


# Terminal job states - safe to delete (never PENDING, QUEUED, DISPATCHING, RUNNING)
CLEANABLE_JOB_STATES = [
    JobState.COMPLETED.value,
    JobState.CANCELLED.value,
    JobState.FAILED.value,
    JobState.TIMEOUT.value,
    JobState.RECOVERED.value,
]

# Terminal worker statuses - safe to delete (never active pods)
CLEANABLE_WORKER_STATUSES = ["terminated", "offline", "error"]

# Batch sleep to avoid monopolizing thread pool
BATCH_SLEEP_SECONDS = 0.05


def _delete_workers_batch(session, cutoff: datetime, batch_size: int) -> int:
    """Delete a batch of old terminal workers. Returns count deleted."""
    # Workers in terminal states, old enough (by updated_at or destroyed_at)
    q = session.query(Worker.worker_id).filter(
        Worker.status.in_(CLEANABLE_WORKER_STATUSES),
        or_(
            Worker.updated_at < cutoff,
            and_(Worker.destroyed_at.isnot(None), Worker.destroyed_at < cutoff),
        ),
    )
    ids = [r[0] for r in q.limit(batch_size).all()]
    if not ids:
        return 0
    deleted = session.query(Worker).filter(Worker.worker_id.in_(ids)).delete(
        synchronize_session=False
    )
    return deleted


def _delete_jobs_batch(session, cutoff: datetime, batch_size: int) -> int:
    """Delete a batch of old terminal jobs. Returns count deleted."""
    q = session.query(Job.job_id).filter(
        Job.status.in_(CLEANABLE_JOB_STATES),
        Job.created_at < cutoff,
    )
    ids = [r[0] for r in q.limit(batch_size).all()]
    if not ids:
        return 0
    deleted = session.query(Job).filter(Job.job_id.in_(ids)).delete(
        synchronize_session=False
    )
    return deleted


def _delete_rewarded_ad_events_batch(
    session, cutoff: datetime, batch_size: int
) -> int:
    """Delete a batch of old rewarded_ad_events. Returns count deleted."""
    q = session.query(RewardedAdEvent.id).filter(
        RewardedAdEvent.created_at < cutoff,
    )
    ids = [r[0] for r in q.limit(batch_size).all()]
    if not ids:
        return 0
    deleted = (
        session.query(RewardedAdEvent)
        .filter(RewardedAdEvent.id.in_(ids))
        .delete(synchronize_session=False)
    )
    return deleted


def _delete_daily_spins_batch(session, cutoff: datetime, batch_size: int) -> int:
    """Delete a batch of old daily_spins. Returns count deleted."""
    q = session.query(DailySpin.id).filter(DailySpin.created_at < cutoff)
    ids = [r[0] for r in q.limit(batch_size).all()]
    if not ids:
        return 0
    deleted = (
        session.query(DailySpin)
        .filter(DailySpin.id.in_(ids))
        .delete(synchronize_session=False)
    )
    return deleted


def run_db_cleanup(
    retention_days: int = 7,
    batch_size: int = 500,
) -> Dict[str, int]:
    """
    Delete rows older than retention_days from workers, jobs, rewarded_ad_events,
    and daily_spins. Uses batched deletes with commit + sleep between batches.

    Designed to run in asyncio.to_thread() - creates its own session.

    Returns:
        Dict mapping table name to total rows deleted.
    """
    cutoff = datetime.utcnow() - timedelta(days=retention_days)
    results: Dict[str, int] = {
        "workers": 0,
        "jobs": 0,
        "rewarded_ad_events": 0,
        "daily_spins": 0,
    }
    db_manager = get_db_manager()

    with db_manager.get_session() as session:
        try:
            # Workers
            while True:
                deleted = _delete_workers_batch(session, cutoff, batch_size)
                if deleted == 0:
                    break
                session.commit()
                results["workers"] += deleted
                time.sleep(BATCH_SLEEP_SECONDS)

            # Jobs
            while True:
                deleted = _delete_jobs_batch(session, cutoff, batch_size)
                if deleted == 0:
                    break
                session.commit()
                results["jobs"] += deleted
                time.sleep(BATCH_SLEEP_SECONDS)

            # Rewarded ad events
            while True:
                deleted = _delete_rewarded_ad_events_batch(
                    session, cutoff, batch_size
                )
                if deleted == 0:
                    break
                session.commit()
                results["rewarded_ad_events"] += deleted
                time.sleep(BATCH_SLEEP_SECONDS)

            # Daily spins
            while True:
                deleted = _delete_daily_spins_batch(session, cutoff, batch_size)
                if deleted == 0:
                    break
                session.commit()
                results["daily_spins"] += deleted
                time.sleep(BATCH_SLEEP_SECONDS)

        except Exception as e:
            logger.exception("DB cleanup failed: %s", e)
            session.rollback()
            raise

    return results
