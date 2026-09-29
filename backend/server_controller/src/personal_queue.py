#!/usr/bin/env python3
"""Leftover-pod personal queue. Client jobs always preempt these items.

Works for any leftover worker (image or video). Workers must send the model
keys they can run; pull only returns a matching item.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy.orm import Session

from db import PersonalQueueItem, utc_naive_now

logger = logging.getLogger("personal_queue")

STATES = ("queued", "running", "paused", "done", "failed")

VIDEO_MODELS = (
    "minimax_h3",
    "ltx25_int8",
    "ltx25_int8_t2v",
    "ltx25_int8_i2v",
    "ltx25_int8_flf2v",
    "ltx2_3_t2v",
    "ltx2_3_i2v_distilled",
    "ltx2_3_flf2v",
    "ltx2_3_iv2v_distilled",
    "ltx2_3_cinereal",
    "wan_22_fp8_lightx2v_t2v",
    "wan_22_fp8_lightx2v_i2v",
    "fastwan_21_1b",
    "fastwan_22_5b",
    "hunyan15_480p_t2v_distilled",
    "hunyan15_480p_i2v_step_distilled",
    "video_upscaler",
    "flashvsr_video_upscaler",
)

IMAGE_MODELS = (
    "sdxl_t2i",
    "sdxl_i2i",
    "sdxl_inpaint",
    "sd15",
    "sd35",
    "qwen_image",
    "qwen_image_edit",
    "qwen_image_edit_2509",
    "qwen_image_edit_2511_rapid_aio",
    "flux_kontext_dev",
    "flux_klein",
    "z_image_turbo",
    "diffusers_outpaint",
    "aurasrv2",
    "swin2sr",
    "realesrgan",
    "face_enhancer",
    "basic_utils",
)

LEFTOVER_MODELS = [
    {"id": mid, "kind": "video", "label": mid}
    for mid in VIDEO_MODELS
] + [
    {"id": mid, "kind": "image", "label": mid}
    for mid in IMAGE_MODELS
]


def leftover_models() -> List[Dict[str, str]]:
    return list(LEFTOVER_MODELS)


def has_leftover_work(db: Session, models: Optional[Sequence[str]] = None) -> bool:
    """True if personal dashboard has queued, paused, or running work for these models."""
    q = db.query(PersonalQueueItem).filter(
        PersonalQueueItem.status.in_(("queued", "paused", "running"))
    )
    allowed = _as_model_list(models, None)
    if allowed:
        q = q.filter(PersonalQueueItem.model.in_(allowed))
    return q.first() is not None


def is_video_model(model: str) -> bool:
    key = (model or "").strip().lower()
    if key in VIDEO_MODELS:
        return True
    return any(
        token in key
        for token in ("t2v", "i2v", "flf", "wan", "ltx", "hunyan", "minimax", "video")
    )


def result_extension(model: str) -> str:
    return ".mp4" if is_video_model(model) else ".png"


def _next_sort(db: Session) -> int:
    row = (
        db.query(PersonalQueueItem)
        .order_by(PersonalQueueItem.sort_order.desc())
        .first()
    )
    return int(row.sort_order) + 10 if row else 10


def list_items(db: Session) -> List[PersonalQueueItem]:
    return (
        db.query(PersonalQueueItem)
        .order_by(PersonalQueueItem.sort_order.asc(), PersonalQueueItem.created_at.asc())
        .all()
    )


def add_item(
    db: Session,
    prompt: str,
    params: Optional[Dict[str, Any]] = None,
    model: str = "",
) -> PersonalQueueItem:
    model = (model or "").strip()
    if not model:
        raise ValueError("model is required")
    item = PersonalQueueItem(
        id=str(uuid.uuid4()),
        prompt=prompt or "",
        params=params or {},
        model=model,
        status="queued",
        sort_order=_next_sort(db),
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def reorder(db: Session, ordered_ids: List[str]) -> None:
    for i, item_id in enumerate(ordered_ids):
        row = db.query(PersonalQueueItem).filter(PersonalQueueItem.id == item_id).first()
        if row:
            row.sort_order = (i + 1) * 10
            row.updated_at = utc_naive_now()
    db.commit()


def delete_item(db: Session, item_id: str) -> bool:
    row = db.query(PersonalQueueItem).filter(PersonalQueueItem.id == item_id).first()
    if not row:
        return False
    if row.status == "running":
        row.status = "paused"
        row.updated_at = utc_naive_now()
        db.commit()
        return True
    db.delete(row)
    db.commit()
    return True


def _as_model_list(models: Optional[Sequence[str]], model: Optional[str]) -> List[str]:
    out: List[str] = []
    if models:
        out.extend(str(m).strip() for m in models if m)
    if model:
        out.append(str(model).strip())
    # unique, keep order
    seen = set()
    uniq = []
    for m in out:
        if m and m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq


def pull_next(
    db: Session,
    worker_id: Optional[str] = None,
    models: Optional[Sequence[str]] = None,
    model: Optional[str] = None,
) -> Optional[PersonalQueueItem]:
    q = (
        db.query(PersonalQueueItem)
        .filter(PersonalQueueItem.status.in_(("queued", "paused")))
        .order_by(PersonalQueueItem.sort_order.asc(), PersonalQueueItem.created_at.asc())
    )
    allowed = _as_model_list(models, model)
    if allowed:
        q = q.filter(PersonalQueueItem.model.in_(allowed))
    row = q.first()
    if not row:
        return None
    row.status = "running"
    row.worker_id = worker_id
    row.updated_at = utc_naive_now()
    db.commit()
    db.refresh(row)
    return row


def complete_item(
    db: Session,
    item_id: str,
    status: str,
    result_r2_key: Optional[str] = None,
    error: Optional[str] = None,
) -> Optional[PersonalQueueItem]:
    row = db.query(PersonalQueueItem).filter(PersonalQueueItem.id == item_id).first()
    if not row:
        return None
    if status not in STATES:
        status = "failed"
    row.status = status
    if result_r2_key:
        row.result_r2_key = result_r2_key
    if error:
        row.error = error
    row.updated_at = utc_naive_now()
    if status in ("done", "failed"):
        row.completed_at = utc_naive_now()
    db.commit()
    db.refresh(row)
    return row


def pause_item(db: Session, item_id: str) -> Optional[PersonalQueueItem]:
    row = db.query(PersonalQueueItem).filter(PersonalQueueItem.id == item_id).first()
    if not row:
        return None
    row.status = "paused"
    row.worker_id = None
    row.updated_at = utc_naive_now()
    db.commit()
    db.refresh(row)
    return row
