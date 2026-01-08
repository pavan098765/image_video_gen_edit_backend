# cost_service.py
"""
Job Cost Tracking Service

Provides:
- GPU infrastructure cost calculation per job
- Pod overhead cost distribution
- Cost analytics and reporting

Cost calculation rules:
- Job cost = (job_processing_time_hours) * (gpu_hourly_rate)
- Pod overhead = (pod_startup_time + pod_destroy_buffer) * (gpu_hourly_rate)
- Overhead is distributed across all jobs that ran on the pod

For single-job pods:
- Full pod cost (create → destroy) is assigned to that job

For multi-job pods:
- Individual job processing costs are calculated per job
- Pod creation/destruction overhead is distributed proportionally based on job runtime
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from sqlalchemy.orm import Session
from sqlalchemy import and_

logger = logging.getLogger("cost_service")


# =============================================================================
# Cost Data Classes
# =============================================================================

@dataclass
class JobCostBreakdown:
    """Detailed cost breakdown for a job"""
    job_id: str
    processing_cost_usd: float  # Cost for actual job processing time
    overhead_share_usd: float   # Share of pod startup/shutdown overhead
    total_cost_usd: float       # processing_cost + overhead_share
    gpu_hourly_rate: float      # $/hour for the GPU used
    processing_time_hours: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "processing_cost_usd": round(self.processing_cost_usd, 6),
            "overhead_share_usd": round(self.overhead_share_usd, 6),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "gpu_hourly_rate": self.gpu_hourly_rate,
            "processing_time_hours": round(self.processing_time_hours, 6),
        }


@dataclass
class PodCostSummary:
    """Cost summary for a pod's lifetime"""
    pod_uuid: str
    gpu_hourly_rate: float
    gpu_type: str
    total_runtime_hours: float
    total_cost_usd: float
    jobs_processed: int
    startup_overhead_usd: float
    processing_cost_usd: float
    job_costs: List[JobCostBreakdown]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pod_uuid": self.pod_uuid,
            "gpu_hourly_rate": self.gpu_hourly_rate,
            "gpu_type": self.gpu_type,
            "total_runtime_hours": round(self.total_runtime_hours, 6),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "jobs_processed": self.jobs_processed,
            "startup_overhead_usd": round(self.startup_overhead_usd, 6),
            "processing_cost_usd": round(self.processing_cost_usd, 6),
            "job_costs": [jc.to_dict() for jc in self.job_costs],
        }


# =============================================================================
# Cost Calculation Functions
# =============================================================================

def calculate_job_processing_cost(
    processing_time_seconds: float,
    gpu_hourly_rate: float,
) -> float:
    """
    Calculate the cost for a job's processing time.
    
    Args:
        processing_time_seconds: Time in seconds the job was processing
        gpu_hourly_rate: GPU cost in $/hour
        
    Returns:
        Cost in USD
    """
    if processing_time_seconds <= 0 or gpu_hourly_rate <= 0:
        return 0.0
    
    processing_hours = processing_time_seconds / 3600.0
    cost = processing_hours * gpu_hourly_rate
    
    return cost


def calculate_pod_total_cost(
    pod_created_at: datetime,
    pod_destroyed_at: datetime,
    gpu_hourly_rate: float,
) -> Tuple[float, float]:
    """
    Calculate total pod cost and runtime.
    
    Args:
        pod_created_at: When pod was created (in QuickPod)
        pod_destroyed_at: When pod was destroyed
        gpu_hourly_rate: GPU cost in $/hour
        
    Returns:
        Tuple of (total_cost_usd, runtime_hours)
    """
    if not pod_created_at or not pod_destroyed_at:
        return 0.0, 0.0
    
    runtime_delta = pod_destroyed_at - pod_created_at
    runtime_hours = runtime_delta.total_seconds() / 3600.0
    
    # QuickPod bills by the hour, but we track granular for analytics
    total_cost = runtime_hours * gpu_hourly_rate
    
    return total_cost, runtime_hours


def distribute_overhead_proportionally(
    total_overhead_usd: float,
    job_processing_times: Dict[str, float],
) -> Dict[str, float]:
    """
    Distribute pod overhead cost across jobs proportionally based on runtime.
    
    Args:
        total_overhead_usd: Total overhead cost to distribute
        job_processing_times: Dict mapping job_id -> processing_time_seconds
        
    Returns:
        Dict mapping job_id -> overhead_share_usd
    """
    if total_overhead_usd <= 0:
        return {job_id: 0.0 for job_id in job_processing_times}
    
    total_processing_time = sum(job_processing_times.values())
    
    if total_processing_time <= 0:
        # Distribute evenly if no processing time recorded
        num_jobs = len(job_processing_times)
        if num_jobs == 0:
            return {}
        share = total_overhead_usd / num_jobs
        return {job_id: share for job_id in job_processing_times}
    
    # Distribute proportionally based on processing time
    overhead_shares = {}
    for job_id, processing_time in job_processing_times.items():
        proportion = processing_time / total_processing_time
        overhead_shares[job_id] = total_overhead_usd * proportion
    
    return overhead_shares


def distribute_overhead_evenly(
    total_overhead_usd: float,
    num_jobs: int,
) -> float:
    """
    Distribute overhead evenly across jobs.
    
    Args:
        total_overhead_usd: Total overhead cost to distribute
        num_jobs: Number of jobs to distribute across
        
    Returns:
        Overhead share per job
    """
    if num_jobs <= 0:
        return 0.0
    return total_overhead_usd / num_jobs


# =============================================================================
# Job Cost Tracking (Database Operations)
# =============================================================================

def calculate_and_store_job_cost(
    db: Session,
    job_id: str,
    pod_uuid: str,
    gpu_hourly_rate: float,
    processing_time_seconds: Optional[float] = None,
    pod_overhead_share: float = 0.0,
) -> Optional[JobCostBreakdown]:
    """
    Calculate and store cost for a completed job.
    
    This should be called when a job completes.
    
    Args:
        db: Database session
        job_id: Job identifier
        pod_uuid: Pod that processed the job
        gpu_hourly_rate: GPU cost in $/hour
        processing_time_seconds: Override processing time (uses job.processing_time_seconds if None)
        pod_overhead_share: Pre-calculated overhead share (0 if not yet calculated)
        
    Returns:
        JobCostBreakdown with cost details, or None if job not found
    """
    from db import Job  # Import here to avoid circular imports
    
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if not job:
        logger.warning(f"Job {job_id} not found for cost calculation")
        return None
    
    # Get processing time
    if processing_time_seconds is None:
        processing_time_seconds = job.processing_time_seconds or 0.0
    
    # Calculate processing cost
    processing_cost = calculate_job_processing_cost(
        processing_time_seconds,
        gpu_hourly_rate,
    )
    
    # Total cost
    total_cost = processing_cost + pod_overhead_share
    
    # Update job record
    job.pod_uuid = pod_uuid
    job.gpu_hourly_rate = gpu_hourly_rate
    job.job_cost_usd = total_cost
    job.pod_overhead_share_usd = pod_overhead_share
    
    db.commit()
    
    processing_hours = processing_time_seconds / 3600.0 if processing_time_seconds > 0 else 0.0
    
    cost_breakdown = JobCostBreakdown(
        job_id=job_id,
        processing_cost_usd=processing_cost,
        overhead_share_usd=pod_overhead_share,
        total_cost_usd=total_cost,
        gpu_hourly_rate=gpu_hourly_rate,
        processing_time_hours=processing_hours,
    )
    
    logger.info(
        f"💰 Job {job_id} cost calculated: "
        f"${total_cost:.6f} (processing=${processing_cost:.6f}, overhead=${pod_overhead_share:.6f})"
    )
    
    return cost_breakdown


def finalize_pod_costs(
    db: Session,
    pod_uuid: str,
    pod_created_at: datetime,
    pod_destroyed_at: datetime,
    gpu_hourly_rate: float,
    gpu_type: str = "",
    distribution_method: str = "proportional",  # "proportional" or "even"
) -> Optional[PodCostSummary]:
    """
    Calculate and distribute final pod costs when a pod is destroyed.
    
    This should be called when a pod is destroyed. It:
    1. Calculates total pod cost
    2. Sums up all job processing costs
    3. Calculates overhead (total - processing)
    4. Distributes overhead across all jobs
    
    Args:
        db: Database session
        pod_uuid: Pod identifier
        pod_created_at: When pod was created
        pod_destroyed_at: When pod was destroyed
        gpu_hourly_rate: GPU cost in $/hour
        gpu_type: GPU type string
        distribution_method: "proportional" (by runtime) or "even" (equal split)
        
    Returns:
        PodCostSummary with all cost details
    """
    from db import Job, Worker  # Import here to avoid circular imports
    
    # Calculate total pod cost
    total_cost, runtime_hours = calculate_pod_total_cost(
        pod_created_at,
        pod_destroyed_at,
        gpu_hourly_rate,
    )
    
    # Get all jobs that ran on this pod
    jobs = db.query(Job).filter(Job.pod_uuid == pod_uuid).all()
    
    if not jobs:
        logger.info(f"No jobs found for pod {pod_uuid}, total cost: ${total_cost:.6f}")
        
        # Still update worker record with pod cost
        worker = db.query(Worker).filter(Worker.container_id == pod_uuid).first()
        if worker:
            worker.destroyed_at = pod_destroyed_at
            worker.total_pod_cost_usd = total_cost
            db.commit()
        
        return PodCostSummary(
            pod_uuid=pod_uuid,
            gpu_hourly_rate=gpu_hourly_rate,
            gpu_type=gpu_type,
            total_runtime_hours=runtime_hours,
            total_cost_usd=total_cost,
            jobs_processed=0,
            startup_overhead_usd=total_cost,  # All cost is overhead if no jobs ran
            processing_cost_usd=0.0,
            job_costs=[],
        )
    
    # Calculate total processing cost from all jobs
    job_processing_times: Dict[str, float] = {}
    total_processing_cost = 0.0
    
    for job in jobs:
        processing_time = job.processing_time_seconds or 0.0
        job_processing_times[job.job_id] = processing_time
        
        processing_cost = calculate_job_processing_cost(
            processing_time,
            gpu_hourly_rate,
        )
        total_processing_cost += processing_cost
    
    # Calculate overhead (pod cost minus processing cost)
    # This includes startup time, idle time between jobs, and shutdown time
    total_overhead = max(0.0, total_cost - total_processing_cost)
    
    # Distribute overhead across jobs
    if distribution_method == "proportional":
        overhead_shares = distribute_overhead_proportionally(total_overhead, job_processing_times)
    else:
        per_job_overhead = distribute_overhead_evenly(total_overhead, len(jobs))
        overhead_shares = {job.job_id: per_job_overhead for job in jobs}
    
    # Update each job with final cost including overhead
    job_costs = []
    for job in jobs:
        overhead_share = overhead_shares.get(job.job_id, 0.0)
        processing_time = job_processing_times.get(job.job_id, 0.0)
        processing_cost = calculate_job_processing_cost(processing_time, gpu_hourly_rate)
        
        job.pod_overhead_share_usd = overhead_share
        job.job_cost_usd = processing_cost + overhead_share
        job.gpu_hourly_rate = gpu_hourly_rate
        
        job_costs.append(JobCostBreakdown(
            job_id=job.job_id,
            processing_cost_usd=processing_cost,
            overhead_share_usd=overhead_share,
            total_cost_usd=job.job_cost_usd,
            gpu_hourly_rate=gpu_hourly_rate,
            processing_time_hours=processing_time / 3600.0 if processing_time > 0 else 0.0,
        ))
    
    # Update worker record with pod cost info
    worker = db.query(Worker).filter(Worker.container_id == pod_uuid).first()
    if worker:
        worker.destroyed_at = pod_destroyed_at
        worker.total_pod_cost_usd = total_cost
        db.commit()
    else:
        db.commit()
    
    summary = PodCostSummary(
        pod_uuid=pod_uuid,
        gpu_hourly_rate=gpu_hourly_rate,
        gpu_type=gpu_type,
        total_runtime_hours=runtime_hours,
        total_cost_usd=total_cost,
        jobs_processed=len(jobs),
        startup_overhead_usd=total_overhead,
        processing_cost_usd=total_processing_cost,
        job_costs=job_costs,
    )
    
    logger.info("=" * 80)
    logger.info(f"💰 POD COST REPORT: {pod_uuid}")
    logger.info(f"   GPU: {gpu_type} @ ${gpu_hourly_rate:.3f}/hour")
    logger.info(f"   Runtime: {runtime_hours:.4f} hours")
    logger.info(f"   Total Cost: ${total_cost:.6f}")
    logger.info(f"   Processing Cost: ${total_processing_cost:.6f}")
    logger.info(f"   Overhead: ${total_overhead:.6f} (distributed across {len(jobs)} jobs)")
    for jc in job_costs:
        logger.info(
            f"   → Job {jc.job_id}: ${jc.total_cost_usd:.6f} "
            f"(processing=${jc.processing_cost_usd:.6f}, overhead=${jc.overhead_share_usd:.6f})"
        )
    logger.info("=" * 80)
    
    return summary


def get_job_cost(db: Session, job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get cost information for a specific job.
    
    Args:
        db: Database session
        job_id: Job identifier
        
    Returns:
        Dict with cost details or None if job not found
    """
    from db import Job
    
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if not job:
        return None
    
    return {
        "job_id": job.job_id,
        "pod_uuid": job.pod_uuid,
        "gpu_hourly_rate": job.gpu_hourly_rate,
        "job_cost_usd": job.job_cost_usd,
        "pod_overhead_share_usd": job.pod_overhead_share_usd,
        "processing_time_seconds": job.processing_time_seconds,
    }


def get_pod_total_cost(db: Session, pod_uuid: str) -> Optional[Dict[str, Any]]:
    """
    Get total cost information for a pod.
    
    Args:
        db: Database session
        pod_uuid: Pod identifier
        
    Returns:
        Dict with cost details or None if no worker found
    """
    from db import Worker, Job
    
    worker = db.query(Worker).filter(Worker.container_id == pod_uuid).first()
    
    # Get all jobs on this pod
    jobs = db.query(Job).filter(Job.pod_uuid == pod_uuid).all()
    total_jobs = len(jobs)
    total_job_cost = sum(j.job_cost_usd or 0.0 for j in jobs)
    
    return {
        "pod_uuid": pod_uuid,
        "gpu_hourly_rate": worker.gpu_hourly_rate if worker else None,
        "gpu_type": worker.gpu_type if worker else None,
        "total_pod_cost_usd": worker.total_pod_cost_usd if worker else None,
        "created_at": worker.created_at.isoformat() if worker and worker.created_at else None,
        "destroyed_at": worker.destroyed_at.isoformat() if worker and worker.destroyed_at else None,
        "jobs_processed": total_jobs,
        "total_jobs_cost_usd": total_job_cost,
    }


# =============================================================================
# Analytics Functions
# =============================================================================

def get_cost_summary_for_period(
    db: Session,
    start_date: datetime,
    end_date: datetime,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get cost summary for a time period.
    
    Args:
        db: Database session
        start_date: Start of period
        end_date: End of period
        model: Optional filter by model
        
    Returns:
        Dict with aggregated cost metrics
    """
    from db import Job
    
    query = db.query(Job).filter(
        and_(
            Job.completed_at >= start_date,
            Job.completed_at <= end_date,
            Job.job_cost_usd.isnot(None),
        )
    )
    
    if model:
        query = query.filter(Job.model == model)
    
    jobs = query.all()
    
    if not jobs:
        return {
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "model_filter": model,
            "total_jobs": 0,
            "total_cost_usd": 0.0,
            "total_processing_cost_usd": 0.0,
            "total_overhead_cost_usd": 0.0,
            "avg_cost_per_job_usd": 0.0,
            "avg_processing_time_seconds": 0.0,
        }
    
    total_cost = sum(j.job_cost_usd or 0.0 for j in jobs)
    total_overhead = sum(j.pod_overhead_share_usd or 0.0 for j in jobs)
    total_processing_cost = total_cost - total_overhead
    total_processing_time = sum(j.processing_time_seconds or 0.0 for j in jobs)
    
    return {
        "period_start": start_date.isoformat(),
        "period_end": end_date.isoformat(),
        "model_filter": model,
        "total_jobs": len(jobs),
        "total_cost_usd": round(total_cost, 6),
        "total_processing_cost_usd": round(total_processing_cost, 6),
        "total_overhead_cost_usd": round(total_overhead, 6),
        "avg_cost_per_job_usd": round(total_cost / len(jobs), 6),
        "avg_processing_time_seconds": round(total_processing_time / len(jobs), 2),
    }


def get_cost_by_model(
    db: Session,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Get cost breakdown by model.
    
    Args:
        db: Database session
        start_date: Optional start date filter
        end_date: Optional end date filter
        
    Returns:
        List of dicts with per-model cost metrics
    """
    from db import Job
    from sqlalchemy import func
    
    query = db.query(
        Job.model,
        func.count(Job.job_id).label("job_count"),
        func.sum(Job.job_cost_usd).label("total_cost"),
        func.avg(Job.job_cost_usd).label("avg_cost"),
        func.sum(Job.processing_time_seconds).label("total_processing_time"),
    ).filter(Job.job_cost_usd.isnot(None))
    
    if start_date:
        query = query.filter(Job.completed_at >= start_date)
    if end_date:
        query = query.filter(Job.completed_at <= end_date)
    
    results = query.group_by(Job.model).all()
    
    return [
        {
            "model": r.model,
            "job_count": r.job_count,
            "total_cost_usd": round(r.total_cost or 0.0, 6),
            "avg_cost_per_job_usd": round(r.avg_cost or 0.0, 6),
            "total_processing_hours": round((r.total_processing_time or 0.0) / 3600.0, 4),
        }
        for r in results
    ]

