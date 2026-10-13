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
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Tuple, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text
import requests

from db import User, Job, CreditTransaction, Subscription, SessionLocal, utc_naive_now
from sale_scheduler import SaleScheduler

logger = logging.getLogger("credit_service")
_sale_scheduler = SaleScheduler()


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
    "studio": {"cost": 25, "cost_upscale": 0, "offer": 0},
    "qwen2_1_t2i": {"cost": 18, "cost_upscale": 0, "offer": 0},
    "qwen2_1_i2i": {"cost": 18, "cost_upscale": 0, "offer": 0},
    "grok_imagine_t2i": {"cost": 18, "cost_upscale": 0, "offer": 0},
    "grok_imagine_edit": {"cost": 18, "cost_upscale": 0, "offer": 0},
    "recraft_crisp_upscale": {"cost": 8, "cost_upscale": 0, "offer": 0},
    "recraft_remove_background": {"cost": 8, "cost_upscale": 0, "offer": 0},
    "gemini_ocr": {"cost": 5, "cost_upscale": 0, "offer": 0},
}

# After sale discount, never charge below these list floors.
IMAGE_SALE_FLOORS: Dict[str, int] = {
    "qwen2_1_t2i": 15,
    "qwen2_1_i2i": 15,
    "grok_imagine_t2i": 15,
    "grok_imagine_edit": 15,
    "recraft_crisp_upscale": 7,
    "recraft_remove_background": 7,
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


def _build_model_cost_map(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build a model -> cost info mapping from tool_config.json.
    
    Reads from:
    1. config.capabilities[].models[] (model id, cost, costUpscale, offer, baseCost5Sec, maxDuration)
    2. config.categories[].tools[] (model, cost, cost_upscale, offer) - fallback
    
    Returns dict like:
    {
        "qwen_image_edit_2509": {"cost": 10, "cost_upscale": 5, "offer": 0},
        "wan_22_fp8_lightx2v_t2v": {"cost": 15, "cost_upscale": 3, "offer": 0, "baseCost5Sec": 15, "maxDuration": 5},
        ...
    }
    """
    model_costs = {}
    
    try:
        # 1. capabilities[].models[] (primary source - has id, cost, costUpscale, baseCost5Sec, maxDuration)
        capabilities = config.get("capabilities", [])
        for capability in capabilities:
            models = capability.get("models", [])
            for m in models:
                model_id = m.get("id")
                if not model_id:
                    continue
                entry = {
                    "cost": m.get("cost", 10),
                    "cost_upscale": m.get("costUpscale", m.get("cost_upscale", 5)),
                    "offer": m.get("offer", 0),
                }
                # Duration-based video cost (create_video, animate_image)
                base_cost_5 = m.get("baseCost5Sec", m.get("base_cost_5_sec"))
                if base_cost_5 is not None:
                    entry["baseCost5Sec"] = int(base_cost_5)
                max_dur = m.get("maxDuration", m.get("max_duration"))
                if max_dur is not None:
                    entry["maxDuration"] = int(max_dur)
                model_costs[model_id] = entry
        
        # 2. categories[].tools[] (fallback for configs with older structure)
        categories = config.get("categories", [])
        for category in categories:
            tools = category.get("tools", [])
            for tool in tools:
                model = tool.get("model")
                if model and model not in model_costs:
                    model_costs[model] = {
                        "cost": tool.get("cost", 10),
                        "cost_upscale": tool.get("cost_upscale", 5),
                        "offer": tool.get("offer", 0),
                    }
        
        logger.info(f"[CREDIT] Built cost map for {len(model_costs)} models")
    except Exception as e:
        logger.error(f"[CREDIT] Error building model cost map: {e}")

    if "studio" not in model_costs:
        model_costs["studio"] = dict(FALLBACK_CREDIT_COSTS["studio"])
    
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
    if model in tool_config:
        return tool_config[model]
    return FALLBACK_CREDIT_COSTS.get(model, FALLBACK_CREDIT_COSTS["default"])


def _resolve_effective_offer(model_offer_percent: int) -> Tuple[int, Dict[str, Any]]:
    """
    Resolve effective discount policy.

    Policy: sale_overrides
    - If sale window is active, sale discount overrides model offer.
    - Otherwise, model offer applies.
    """
    try:
        sale_state = _sale_scheduler.get_current_window_state()
        if sale_state.is_active:
            return int(sale_state.discount_percent), {
                "offer_source": "sale_window_override",
                "sale_active": True,
                "sale_discount_percent": int(sale_state.discount_percent),
                "sale_window_id": sale_state.window_id,
                "sale_window_index": sale_state.window_index,
                "sale_day_key": sale_state.day_key,
                "sale_timezone": sale_state.timezone,
                "sale_window_start_local": sale_state.window_start_local,
                "sale_window_end_local": sale_state.window_end_local,
                "model_offer_percent": int(model_offer_percent or 0),
            }
    except Exception as e:
        logger.warning(f"[CREDIT] Sale scheduler check failed, falling back to model offer: {e}")

    return int(model_offer_percent or 0), {
        "offer_source": "model_offer",
        "sale_active": False,
        "sale_discount_percent": 0,
        "sale_window_id": None,
        "sale_window_index": None,
        "sale_day_key": None,
        "sale_timezone": getattr(_sale_scheduler, "timezone_name", "UTC"),
        "sale_window_start_local": None,
        "sale_window_end_local": None,
        "model_offer_percent": int(model_offer_percent or 0),
    }


def get_active_sale_window_info() -> Dict[str, Any]:
    """Expose current sale window state for observability/admin endpoints."""
    state = _sale_scheduler.get_current_window_state()
    return {
        "is_active": state.is_active,
        "discount_percent": int(state.discount_percent),
        "day_key": state.day_key,
        "window_index": state.window_index,
        "window_id": state.window_id,
        "window_start_local": state.window_start_local,
        "window_end_local": state.window_end_local,
        "timezone": state.timezone,
    }


def build_public_sale_status_payload() -> Dict[str, Any]:
    """
    Public JSON body for GET /sales/active (banners / client pricing preview).

    Single builder for response bytes and ETag so they cannot drift.
    Authoritative charges still use _sale_scheduler inside calculate_job_cost / submit paths.
    """
    info = get_active_sale_window_info()
    next_window = None
    try:
        sched = _sale_scheduler
        snapshot = sched.get_today_schedule_snapshot()
        tz_name = snapshot.get("timezone") or "UTC"
        tz = ZoneInfo(tz_name)
        slot_minutes = int(snapshot.get("slot_minutes") or 15)
        slot_interval_hours = int(snapshot.get("slot_interval_hours") or 3)
        discount_schedule = snapshot.get("discount_schedule") or []
        now_utc = datetime.utcnow()
        local_now = now_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
        day_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        for idx, discount in enumerate(discount_schedule):
            slot_start = day_start + timedelta(hours=idx * slot_interval_hours)
            slot_end = slot_start + timedelta(minutes=slot_minutes)
            if slot_start > local_now:
                next_window = {
                    "start_utc": slot_start.astimezone(ZoneInfo("UTC")).isoformat(),
                    "discount_percent": int(discount),
                }
                break
        if next_window is None and discount_schedule:
            tomorrow_start = day_start + timedelta(days=1)
            tomorrow_local = tomorrow_start
            tomorrow_schedule = sched.get_daily_schedule(tomorrow_local)
            if tomorrow_schedule:
                first_slot = tomorrow_start
                next_window = {
                    "start_utc": first_slot.astimezone(ZoneInfo("UTC")).isoformat(),
                    "discount_percent": int(tomorrow_schedule[0]),
                }
    except Exception:
        pass
    return {
        "is_active": info.get("is_active", False),
        "discount_percent": info.get("discount_percent", 0),
        "window_start": info.get("window_start_local"),
        "window_end": info.get("window_end_local"),
        "timezone": info.get("timezone"),
        "next_window": next_window,
    }


def get_today_sale_schedule_info() -> Dict[str, Any]:
    """Expose today's configured sale schedule and current window."""
    return _sale_scheduler.get_today_schedule_snapshot()


def calculate_job_cost(
    model: str,
    params: Optional[Dict[str, Any]] = None,
    variations: int = 1,
    is_vip: bool = False,
    num_images: int = 1,
    return_meta: bool = False,
) -> Any:
    """
    Calculate the total cost for a job (or batch of jobs).
    
    Formula:
        base_cost = model's base cost
        upscale_cost = model's upscale cost (if scale_factor > 1 AND not VIP)
        offer_discount = base_cost * (offer / 100)
        per_job_cost = base_cost + upscale_cost - offer_discount
        total = per_job_cost * variations * num_images
    
    Special handling for video_upscaler and flashvsr_video_upscaler:
        - Dynamic cost based on video_duration: 2 credits per 5 seconds
        - Max video duration: 10 minutes (600 seconds)
        - Max cost: 240 credits (2 * 600 / 5)
    
    Duration-based cost for create_video / animate_image models (with baseCost5Sec in tool_config):
        - params.duration required (seconds from front end)
        - Formula: cost = baseCost5Sec * min(duration, maxDuration) / 5  (3 credits/sec when baseCost5Sec=15)
        - E.g. 5s→15, 8s→24, 10s→30
        - Capped by maxDuration from tool_config
    
    VIP Benefits:
        - FREE upscale (upscale_cost = 0 for VIP users)
    
    Args:
        model: The model name (e.g., "qwen_image_edit_2509")
        params: Job params dict, may contain "scale_factor" or "video_duration"
        variations: Number of variations to generate (multiplies cost)
        is_vip: Whether the user has active VIP subscription (free upscale)
        num_images: Number of images in bulk edit (multiplies cost)
    
    Returns:
        Total credit cost (integer, minimum 1)
    """
    # Studio: duration × tier, not a forever-flat 25.
    if model == "studio":
        prefs = (params or {}).get("prefs") or params or {}
        duration_seconds = float(prefs.get("duration_seconds") or 30)
        duration_seconds = max(5.0, min(duration_seconds, 1200.0))
        tier = str(prefs.get("quality_tier") or "studio")
        minutes = max(1, int((duration_seconds + 59) // 60))
        per_min = {"draft": 8, "studio": 15, "hero": 28}.get(tier, 15)
        extra = 10 if prefs.get("emit_shorts") or prefs.get("native_short") else 0
        total = max(10, minutes * per_min + extra)
        if return_meta:
            return total, {
                "model": model,
                "pricing_type": "studio_duration_tier",
                "quality_tier": tier,
                "minutes": minutes,
            }
        return total
    cost_info = get_model_cost_info(model)
    model_offer_percent = int(cost_info.get("offer", 0) or 0)
    effective_offer_percent, offer_meta = _resolve_effective_offer(model_offer_percent)

    # Special handling for video upscalers: dynamic cost based on video duration
    if model in ("video_upscaler", "flashvsr_video_upscaler") and params:
        video_duration = params.get("video_duration")
        if video_duration is not None:
            try:
                # Parse video_duration (can be float or string)
                duration_seconds = float(video_duration)
                
                # Cap at 10 minutes (600 seconds)
                MAX_VIDEO_DURATION = 600  # 10 minutes
                duration_seconds = min(duration_seconds, MAX_VIDEO_DURATION)
                
                # Calculate cost: 2 credits per 5 seconds
                # Round up to nearest 5-second increment
                credits_per_5_sec = 2
                duration_5sec_blocks = max(1, int((duration_seconds + 4) // 5))  # Round up
                per_job_cost = credits_per_5_sec * duration_5sec_blocks
                if effective_offer_percent > 0:
                    per_job_cost -= int(per_job_cost * effective_offer_percent / 100)
                    per_job_cost = max(1, per_job_cost)
                
                # Multiply by variations count
                variations = max(1, min(4, variations))  # Clamp to 1-4
                total = per_job_cost * variations
                
                logger.info(
                    f"[CREDIT] Video upscaler cost: duration={duration_seconds:.1f}s, "
                    f"5sec_blocks={duration_5sec_blocks}, per_job={per_job_cost}, "
                    f"variations={variations}, total={total}"
                )
                
                total = max(1, total)  # Ensure minimum cost of 1
                if return_meta:
                    return total, {
                        "model": model,
                        "pricing_type": "video_upscaler_duration",
                        "effective_offer_percent": effective_offer_percent,
                        **offer_meta,
                    }
                return total
            except (ValueError, TypeError) as e:
                logger.warning(f"[CREDIT] Invalid video_duration for {model}: {video_duration}, error: {e}")
                # Fall through to default calculation
    
    # Duration-based cost for create_video / animate_image models (with baseCost5Sec in tool_config)
    base_cost_5_sec = cost_info.get("baseCost5Sec")
    duration_val = (params or {}).get("duration") or (params or {}).get("duration_seconds")
    if base_cost_5_sec is not None and duration_val is not None:
        try:
            duration_seconds = float(duration_val)
            if duration_seconds > 0:
                max_duration = cost_info.get("maxDuration", 10)
                effective_duration = min(duration_seconds, max_duration)
                # Formula: cost = baseCost5Sec * effective_duration / 5 (matches frontend: 5s→15, 8s→24, 10s→30)
                per_job_cost = int(base_cost_5_sec * effective_duration / 5)
                per_job_cost = max(1, per_job_cost)
                # Apply effective offer discount (sale_overrides policy)
                if effective_offer_percent > 0:
                    per_job_cost -= int(per_job_cost * effective_offer_percent / 100)
                    per_job_cost = max(1, per_job_cost)
                variations_count = max(1, min(4, variations))
                num_images_count = max(1, num_images)
                total = per_job_cost * variations_count * num_images_count
                logger.info(
                    f"[CREDIT] Duration-based cost for {model}: duration={duration_seconds:.1f}s, "
                    f"effective={effective_duration:.1f}s, baseCost5Sec={base_cost_5_sec}, "
                    f"per_job={per_job_cost}, variations={variations_count}, total={total}"
                )
                total = max(1, total)
                if return_meta:
                    return total, {
                        "model": model,
                        "pricing_type": "duration_based",
                        "effective_offer_percent": effective_offer_percent,
                        **offer_meta,
                    }
                return total
        except (ValueError, TypeError) as e:
            logger.warning(f"[CREDIT] Invalid duration for {model}: {duration_val}, error: {e}")
            # Fall through to standard calculation
    
    # Standard cost calculation for other models
    
    base_cost = cost_info.get("cost", 10)
    upscale_cost = cost_info.get("cost_upscale", 5)
    
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
    # VIP BENEFIT: Free upscale for VIP users
    if scale_factor > 1:
        if is_vip:
            logger.debug(f"[CREDIT] VIP user - FREE upscale (scale_factor={scale_factor})")
        else:
            per_job_cost += upscale_cost
            logger.debug(f"[CREDIT] Adding upscale cost: {upscale_cost} (scale_factor={scale_factor})")
    
    # Apply offer discount
    if effective_offer_percent > 0:
        discount = int(base_cost * effective_offer_percent / 100)
        per_job_cost -= discount
        logger.debug(f"[CREDIT] Applying {effective_offer_percent}% discount: -{discount}")
    
    # Ensure minimum cost of 1 per job, then sale-safe image floors
    per_job_cost = max(1, per_job_cost)
    floor = IMAGE_SALE_FLOORS.get(model)
    if floor is not None:
        per_job_cost = max(floor, per_job_cost)
    
    # Multiply by variations count and number of images (bulk edit)
    variations = max(1, min(4, variations))  # Clamp to 1-4
    num_images = max(1, num_images)  # At least 1 image
    total = per_job_cost * variations * num_images
    
    upscale_applied = upscale_cost if (scale_factor > 1 and not is_vip) else 0
    logger.info(
        f"[CREDIT] Cost for {model}: base={base_cost}, upscale={upscale_applied}, "
        f"offer={effective_offer_percent}%, per_job={per_job_cost}, variations={variations}, "
        f"num_images={num_images}, is_vip={is_vip}, total={total}"
    )
    if return_meta:
        return total, {
            "model": model,
            "pricing_type": "standard",
            "base_cost": int(base_cost),
            "upscale_cost_applied": int(upscale_applied),
            "effective_offer_percent": int(effective_offer_percent),
            **offer_meta,
        }
    return total


def get_credit_cost(
    model: str,
    params: Optional[Dict[str, Any]] = None,
    variations: int = 1,
    is_vip: bool = False,
    num_images: int = 1,
) -> int:
    """
    Get credit cost for a model (with optional params for upscale calculation).
    
    This is the main entry point for cost calculation.
    
    Args:
        model: Model name
        params: Job params (may contain scale_factor)
        variations: Number of variations to generate (1-4)
        is_vip: Whether user has active VIP (free upscale benefit)
        num_images: Number of images in bulk edit
    
    Returns:
        Total credit cost
    """
    return calculate_job_cost(model, params, variations, is_vip, num_images)


def get_credit_cost_breakdown(
    model: str,
    params: Optional[Dict[str, Any]] = None,
    variations: int = 1,
    is_vip: bool = False,
    num_images: int = 1,
) -> Tuple[int, Dict[str, Any]]:
    """Return cost and pricing metadata for auditing."""
    total, meta = calculate_job_cost(
        model,
        params,
        variations,
        is_vip,
        num_images,
        return_meta=True,
    )
    return int(total), dict(meta or {})


# =============================================================================
# Core Credit Operations
# =============================================================================

def lock_credits_for_job(
    db: Session,
    user_id: str,
    job_id: str,
    model: str,
    params: Optional[Dict[str, Any]] = None,
    is_vip: bool = False,
) -> Tuple[bool, str, int]:
    """
    Atomically lock credits for a job.
    
    This is called during job submission to ensure the user has enough credits.
    The credits are deducted immediately and recorded in the job.
    
    CRITICAL: This runs within the caller's transaction (from submit endpoint).
    The caller is responsible for commit/rollback.
    
    VIP Benefits applied here:
    - Free upscale (upscale cost = 0 for VIP users)
    
    Args:
        db: Database session
        user_id: User ID
        job_id: Job ID
        model: Model name
        params: Job params (used to calculate upscale cost)
        is_vip: Whether user has active VIP subscription (free upscale)
    
    Returns:
        Tuple of (success, message, credits_locked)
    """
    try:
        # Get credit cost + pricing metadata (includes sale window override policy)
        cost, pricing_meta = get_credit_cost_breakdown(
            model, params, variations=1, is_vip=is_vip
        )
        logger.info(f"[CREDIT] Attempting to lock {cost} credits for job {job_id}, user {user_id}, model {model}, is_vip={is_vip}")
        
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
            if isinstance(job.params, dict):
                meta_copy = dict(job.params)
                meta_copy["_pricing_meta"] = {
                    "credits_locked": int(cost),
                    "locked_at": utc_naive_now().isoformat(),
                    **pricing_meta,
                }
                job.params = meta_copy
            logger.info(f"[CREDIT] Updated job {job_id} with credits_locked={cost}")
        else:
            logger.warning(f"[CREDIT] Job {job_id} not found - cannot set credits_locked")
        
        # Record transaction
        reason = "job_submitted"
        if pricing_meta.get("sale_active"):
            sale_pct = pricing_meta.get("sale_discount_percent", 0)
            window_id = pricing_meta.get("sale_window_id") or "unknown"
            reason = f"job_submitted:sale_{sale_pct}pct:{window_id}"

        transaction = CreditTransaction(
            user_id=user_id,
            delta=-cost,
            reason=reason[:255],
            credit_metadata={
                "event": "job_submitted",
                "pricing": pricing_meta,
                "model": model,
                "is_vip": bool(is_vip),
                "locked_credits": int(cost),
            },
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
            credit_metadata={
                "event": "job_refund",
                "refund_reason": reason,
                "refund_amount": int(refund_amount),
            },
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
            credit_metadata={
                "event": "topup",
                "provider": provider,
                "payment_ref": payment_ref,
                "amount": int(amount),
            },
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
            credit_metadata={
                "event": "subscription_credit",
                "plan_id": subscription.plan_id,
                "subscription_id": int(subscription_id),
                "amount": int(amount),
            },
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
