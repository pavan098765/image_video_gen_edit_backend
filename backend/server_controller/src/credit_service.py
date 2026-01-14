# credit_service.py
"""
Credit Service - Atomic, server-controlled credit operations.

This module handles all credit operations with proper locking and transaction safety.
Client cannot directly manipulate credits - only server-side operations are allowed.

Credit Flow:
- Job Submit: Lock credits atomically (deduct from user, record in job)
- Job Success: Credits already consumed, no action needed
- Job Failed: Refund locked credits to user
- Ad Reward: Add credits after server-side verification (deduplicated)
- Topup: Add credits after payment verification
- Subscription: Monthly credit grant

All operations use database transactions to ensure atomicity.
"""

import logging
import time
import threading
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text
import requests

from db import User, Job, CreditTransaction, Subscription, SessionLocal, utc_naive_now

logger = logging.getLogger("credit_service")


# =============================================================================
# Tool Config Cache (Dynamic Credit Costs)
# =============================================================================

TOOL_CONFIG_URL = "https://pub-67f874d172874988b5754734cc4fcb7e.r2.dev/tool_config.json"
TOOL_CONFIG_CACHE_TTL = 6 * 60 * 60  # 6 hours in seconds

# Thread-safe cache for tool config
_tool_config_cache: Dict[str, Any] = {
    "data": None,
    "last_fetch": 0,
    "lock": threading.Lock(),
}

# Fallback costs if tool_config.json is unavailable
FALLBACK_CREDIT_COSTS: Dict[str, Dict[str, int]] = {
    "default": {"cost": 10, "cost_upscale": 5, "offer": 0},
}


def _fetch_tool_config() -> Optional[Dict[str, Any]]:
    """
    Fetch tool_config.json from R2.
    Returns None on failure.
    """
    try:
        response = requests.get(TOOL_CONFIG_URL, timeout=10)
        response.raise_for_status()
        config = response.json()
        logger.info(f"[CREDIT] Successfully fetched tool_config.json from {TOOL_CONFIG_URL}")
        return config
    except requests.RequestException as e:
        logger.error(f"[CREDIT] Failed to fetch tool_config.json: {e}")
        return None
    except ValueError as e:
        logger.error(f"[CREDIT] Failed to parse tool_config.json: {e}")
        return None


def _build_model_cost_map(config: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """
    Build a model -> cost info mapping from tool_config.json.
    
    Returns dict like:
    {
        "qwen_image_edit_2509": {"cost": 10, "cost_upscale": 5, "offer": 0},
        ...
    }
    """
    model_costs = {}
    
    try:
        categories = config.get("categories", [])
        for category in categories:
            tools = category.get("tools", [])
            for tool in tools:
                model = tool.get("model")
                if model:
                    model_costs[model] = {
                        "cost": tool.get("cost", 10),
                        "cost_upscale": tool.get("cost_upscale", 5),
                        "offer": tool.get("offer", 0),
                    }
        
        logger.info(f"[CREDIT] Built cost map for {len(model_costs)} models")
    except Exception as e:
        logger.error(f"[CREDIT] Error building model cost map: {e}")
    
    return model_costs


def get_tool_config() -> Dict[str, Dict[str, int]]:
    """
    Get the cached tool config (model -> cost info mapping).
    Refreshes cache if TTL expired.
    Thread-safe.
    """
    global _tool_config_cache
    
    current_time = time.time()
    
    with _tool_config_cache["lock"]:
        # Check if cache is still valid
        if (_tool_config_cache["data"] is not None and 
            current_time - _tool_config_cache["last_fetch"] < TOOL_CONFIG_CACHE_TTL):
            return _tool_config_cache["data"]
        
        # Cache expired or empty - fetch fresh data
        logger.info("[CREDIT] Tool config cache expired or empty, fetching fresh config...")
        config = _fetch_tool_config()
        
        if config:
            model_costs = _build_model_cost_map(config)
            _tool_config_cache["data"] = model_costs
            _tool_config_cache["last_fetch"] = current_time
            return model_costs
        
        # Fetch failed - use existing cache if available, otherwise fallback
        if _tool_config_cache["data"] is not None:
            logger.warning("[CREDIT] Fetch failed, using stale cache")
            return _tool_config_cache["data"]
        
        logger.warning("[CREDIT] Fetch failed, no cache available, using fallback costs")
        return FALLBACK_CREDIT_COSTS


def get_model_cost_info(model: str) -> Dict[str, int]:
    """
    Get cost info for a specific model.
    
    Returns:
        Dict with keys: cost, cost_upscale, offer
    """
    tool_config = get_tool_config()
    return tool_config.get(model, FALLBACK_CREDIT_COSTS["default"])


def calculate_job_cost(
    model: str,
    params: Optional[Dict[str, Any]] = None,
    variations: int = 1,
) -> int:
    """
    Calculate the total cost for a job (or batch of jobs).
    
    Formula:
        base_cost = model's base cost
        upscale_cost = model's upscale cost (if scale_factor > 1)
        offer_discount = base_cost * (offer / 100)
        per_job_cost = base_cost + upscale_cost - offer_discount
        total = per_job_cost * variations
    
    Args:
        model: The model name (e.g., "qwen_image_edit_2509")
        params: Job params dict, may contain "scale_factor"
        variations: Number of variations to generate (multiplies cost)
    
    Returns:
        Total credit cost (integer, minimum 1)
    """
    cost_info = get_model_cost_info(model)
    
    base_cost = cost_info.get("cost", 10)
    upscale_cost = cost_info.get("cost_upscale", 5)
    offer_percent = cost_info.get("offer", 0)
    
    # Check if upscaling is enabled
    scale_factor = 1
    if params:
        # Handle various scale_factor formats
        sf = params.get("scale_factor", 1)
        if isinstance(sf, str):
            try:
                sf = float(sf)
            except ValueError:
                sf = 1
        scale_factor = int(sf) if sf else 1
    
    # Calculate per-job cost
    per_job_cost = base_cost
    
    # Add upscale cost if scale_factor > 1
    if scale_factor > 1:
        per_job_cost += upscale_cost
        logger.debug(f"[CREDIT] Adding upscale cost: {upscale_cost} (scale_factor={scale_factor})")
    
    # Apply offer discount
    if offer_percent > 0:
        discount = int(base_cost * offer_percent / 100)
        per_job_cost -= discount
        logger.debug(f"[CREDIT] Applying {offer_percent}% discount: -{discount}")
    
    # Ensure minimum cost of 1 per job
    per_job_cost = max(1, per_job_cost)
    
    # Multiply by variations count
    variations = max(1, min(4, variations))  # Clamp to 1-4
    total = per_job_cost * variations
    
    logger.info(
        f"[CREDIT] Cost for {model}: base={base_cost}, upscale={upscale_cost if scale_factor > 1 else 0}, "
        f"offer={offer_percent}%, per_job={per_job_cost}, variations={variations}, total={total}"
    )
    
    return total


def get_credit_cost(
    model: str,
    params: Optional[Dict[str, Any]] = None,
    variations: int = 1,
) -> int:
    """
    Get credit cost for a model (with optional params for upscale calculation).
    
    This is the main entry point for cost calculation.
    
    Args:
        model: Model name
        params: Job params (may contain scale_factor)
        variations: Number of variations to generate (1-4)
    
    Returns:
        Total credit cost
    """
    return calculate_job_cost(model, params, variations)


# =============================================================================
# Core Credit Operations
# =============================================================================

def lock_credits_for_job(
    db: Session,
    user_id: str,
    job_id: str,
    model: str,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, int]:
    """
    Atomically lock credits for a job.
    
    This is called during job submission to ensure the user has enough credits.
    The credits are deducted immediately and recorded in the job.
    
    CRITICAL: This runs within the caller's transaction (from submit endpoint).
    The caller is responsible for commit/rollback.
    
    Args:
        db: Database session
        user_id: User ID
        job_id: Job ID
        model: Model name
        params: Job params (used to calculate upscale cost)
    
    Returns:
        Tuple of (success, message, credits_locked)
    """
    try:
        # Get credit cost (includes upscale cost calculation)
        cost = get_credit_cost(model, params)
        logger.info(f"[CREDIT] Attempting to lock {cost} credits for job {job_id}, user {user_id}, model {model}")
        
        # Get user (SQLite uses transaction-level locking, no with_for_update needed)
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            logger.error(f"[CREDIT] User {user_id} not found in database!")
            return False, "User not found", 0
        
        logger.info(f"[CREDIT] Found user {user_id}, current balance: {user.total_credits}")
        
        if not user.is_active:
            logger.warning(f"[CREDIT] User {user_id} account is disabled")
            return False, "User account is disabled", 0
        
        if user.total_credits < cost:
            logger.warning(f"[CREDIT] User {user_id} has insufficient credits: {user.total_credits} < {cost}")
            return False, f"Insufficient credits. Have: {user.total_credits}, Need: {cost}", 0
        
        # Deduct credits atomically
        old_balance = user.total_credits
        user.total_credits -= cost
        logger.info(f"[CREDIT] Deducted {cost} from user {user_id}: {old_balance} -> {user.total_credits}")
        
        # Update job with locked credits
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if job:
            job.credits_locked = cost
            job.user_id = user_id
            logger.info(f"[CREDIT] Updated job {job_id} with credits_locked={cost}")
        else:
            logger.warning(f"[CREDIT] Job {job_id} not found - cannot set credits_locked")
        
        # Record transaction
        transaction = CreditTransaction(
            user_id=user_id,
            delta=-cost,
            reason="job_submitted",
            job_id=job_id,
            created_at=utc_naive_now(),
        )
        db.add(transaction)
        logger.info(f"[CREDIT] Added CreditTransaction: user={user_id}, delta=-{cost}, job={job_id}")
        
        # DON'T commit here - caller handles the transaction
        # This allows rollback if job creation fails
        db.flush()  # Ensure changes are written to DB within transaction
        
        logger.info(f"[CREDIT] ✅ Locked {cost} credits for job {job_id} (user: {user_id}, balance: {user.total_credits})")
        return True, "Credits locked", cost
        
    except Exception as e:
        logger.error(f"[CREDIT] ❌ Failed to lock credits for job {job_id}: {e}", exc_info=True)
        # Don't rollback here - let caller handle it
        return False, f"Credit lock failed: {str(e)}", 0


def refund_credits_for_job(
    db: Optional[Session],  # Ignored - uses own session for atomicity
    job_id: str,
    reason: str = "job_failed",
) -> Tuple[bool, str, int]:
    """
    Refund locked credits for a failed job.
    
    This is called when a job fails to ensure the user gets their credits back.
    Prevents double-refund by checking credits_refunded.
    
    CRITICAL: Uses its own transaction to ensure refund is always committed.
    The db parameter is ignored - function manages its own session.
    
    Returns:
        Tuple of (success, message, credits_refunded)
    """
    # Always use fresh session to avoid transaction conflicts
    refund_db = SessionLocal()
    try:
        job = refund_db.query(Job).filter(Job.job_id == job_id).first()
        
        if not job:
            return False, "Job not found", 0
        
        if not job.user_id:
            return False, "Job has no associated user", 0
        
        # Check for double-refund (atomic check)
        if job.credits_refunded >= job.credits_locked:
            logger.debug(f"Job {job_id} already refunded ({job.credits_refunded}/{job.credits_locked})")
            return False, "Already refunded", 0
        
        refund_amount = job.credits_locked - job.credits_refunded
        if refund_amount <= 0:
            return False, "Nothing to refund", 0
        
        # Get user
        user = refund_db.query(User).filter(User.id == job.user_id).first()
        if not user:
            return False, "User not found", 0
        
        # Refund credits atomically
        user.total_credits += refund_amount
        job.credits_refunded = job.credits_locked
        
        # Record transaction
        transaction = CreditTransaction(
            user_id=job.user_id,
            delta=refund_amount,
            reason=reason,
            job_id=job_id,
            created_at=utc_naive_now(),
        )
        refund_db.add(transaction)
        
        refund_db.commit()
        
        logger.info(f"Refunded {refund_amount} credits for job {job_id} (user: {job.user_id}, balance: {user.total_credits})")
        return True, "Credits refunded", refund_amount
        
    except Exception as e:
        logger.error(f"Failed to refund credits for job {job_id}: {e}")
        try:
            refund_db.rollback()
        except Exception:
            pass
        return False, f"Refund failed: {str(e)}", 0
    finally:
        refund_db.close()


# =============================================================================
# Credit Addition Operations (Server-Verified Only)
# =============================================================================

def add_topup_credits(
    db: Session,
    user_id: str,
    amount: int,
    payment_ref: str,
    provider: str = "stripe",
) -> Tuple[bool, str]:
    """
    Add credits from a verified payment.
    
    CRITICAL: payment_ref should be verified with payment provider before calling.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        # Check for duplicate payment (by transaction reason containing payment_ref)
        existing = db.query(CreditTransaction).filter(
            CreditTransaction.user_id == user_id,
            CreditTransaction.reason.contains(payment_ref),
        ).first()
        
        if existing:
            logger.warning(f"Duplicate topup attempt: {payment_ref}")
            return False, "Payment already processed"
        
        # Get user
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False, "User not found"
        
        # Add credits atomically
        user.total_credits += amount
        
        # Record transaction (reason includes payment_ref for deduplication)
        transaction = CreditTransaction(
            user_id=user_id,
            delta=amount,
            reason=f"topup:{provider}:{payment_ref}",
            created_at=utc_naive_now(),
        )
        db.add(transaction)
        
        db.commit()
        
        logger.info(f"Added {amount} credits for topup (user: {user_id}, ref: {payment_ref})")
        return True, f"Added {amount} credits"
        
    except Exception as e:
        logger.error(f"Failed to add topup credits for user {user_id}: {e}")
        db.rollback()
        return False, f"Failed: {str(e)}"


def grant_subscription_credits(
    db: Session,
    user_id: str,
    subscription_id: int,
    amount: int,
) -> Tuple[bool, str]:
    """
    Grant monthly credits for an active subscription.
    
    CRITICAL: Should only be called once per billing cycle.
    Use subscription.renewed_at to track.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        # Get subscription
        subscription = db.query(Subscription).filter(
            Subscription.id == subscription_id,
            Subscription.user_id == user_id,
            Subscription.status == "active",
        ).first()
        
        if not subscription:
            return False, "Active subscription not found"
        
        # Get user
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False, "User not found"
        
        # Add credits atomically
        user.total_credits += amount
        user.bonus_granted += amount
        
        # Update subscription renewal timestamp (prevents double-grant in same cycle)
        subscription.renewed_at = utc_naive_now()
        
        # Record transaction
        transaction = CreditTransaction(
            user_id=user_id,
            delta=amount,
            reason=f"subscription:{subscription.plan_id}",
            created_at=utc_naive_now(),
        )
        db.add(transaction)
        
        db.commit()
        
        logger.info(f"Granted {amount} subscription credits (user: {user_id}, plan: {subscription.plan_id})")
        return True, f"Added {amount} credits"
        
    except Exception as e:
        logger.error(f"Failed to grant subscription credits for user {user_id}: {e}")
        db.rollback()
        return False, f"Failed: {str(e)}"


# =============================================================================
# Balance Check
# =============================================================================

def get_user_balance(db: Session, user_id: str) -> Tuple[int, int]:
    """
    Get user's current balance.
    
    Returns:
        Tuple of (total_credits, bonus_granted)
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return 0, 0
    return user.total_credits, user.bonus_granted


def has_sufficient_credits(db: Session, user_id: str, model: str) -> Tuple[bool, int, int]:
    """
    Check if user has enough credits for a model.
    
    Returns:
        Tuple of (has_enough, current_balance, required_credits)
    """
    cost = get_credit_cost(model)
    balance, _ = get_user_balance(db, user_id)
    return balance >= cost, balance, cost
