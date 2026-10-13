"""
Sale scheduler for fixed daily discount windows.

Rules implemented:
- 8 windows/day
- Each window lasts 15 minutes
- Windows start every 3 hours (00:00, 03:00, ..., 21:00 in configured timezone)
- Discount set is [5, 5, 10, 10, 15, 15, 20, 20]
- No adjacent equal discounts within a day
"""

from __future__ import annotations

import os
import random
import threading
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo


DEFAULT_DISCOUNT_SET = [5, 5, 10, 10, 15, 15, 20, 20]


@dataclass(frozen=True)
class SaleWindowState:
    is_active: bool
    discount_percent: int
    day_key: str
    window_index: Optional[int]
    window_id: Optional[str]
    window_start_local: Optional[str]
    window_end_local: Optional[str]
    timezone: str


class SaleScheduler:
    """Deterministic, timezone-aware daily sale window scheduler."""

    def __init__(self):
        self.timezone_name = os.getenv("SALES_TIMEZONE", "UTC")
        self.slot_minutes = int(os.getenv("SALES_SLOT_MINUTES", "15"))
        self.slots_per_day = int(os.getenv("SALES_SLOTS_PER_DAY", "8"))
        self.slot_interval_hours = int(os.getenv("SALES_SLOT_INTERVAL_HOURS", "3"))
        self.seed_salt = os.getenv("SALES_DAILY_SEED_SALT", "sales-seed")

        if self.slots_per_day != 8:
            self.slots_per_day = 8
        if self.slot_interval_hours != 3:
            self.slot_interval_hours = 3
        if self.slot_minutes <= 0:
            self.slot_minutes = 15

        try:
            self._tz = ZoneInfo(self.timezone_name)
        except Exception:
            self.timezone_name = "UTC"
            self._tz = ZoneInfo("UTC")

        self._cache_lock = threading.Lock()
        self._daily_schedule_cache: Dict[str, List[int]] = {}

    def _daily_seed(self, day_key: str) -> int:
        raw = f"{self.seed_salt}:{self.timezone_name}:{day_key}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return int(digest, 16)

    @staticmethod
    def _is_no_adjacent(sequence: List[int]) -> bool:
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i - 1]:
                return False
        return True

    def _build_daily_schedule(self, day_key: str) -> List[int]:
        # Deterministic shuffle + adjacency guard.
        rng = random.Random(self._daily_seed(day_key))
        items = list(DEFAULT_DISCOUNT_SET)

        for _ in range(300):
            rng.shuffle(items)
            if self._is_no_adjacent(items):
                return list(items)

        # Deterministic fallback construction (always succeeds for this multiset).
        groups: Dict[int, List[int]] = {}
        for x in DEFAULT_DISCOUNT_SET:
            groups.setdefault(x, []).append(x)
        ordered_keys = sorted(groups.keys(), key=lambda k: (len(groups[k]), k), reverse=True)
        out: List[int] = []
        while len(out) < len(DEFAULT_DISCOUNT_SET):
            placed = False
            for k in ordered_keys:
                if not groups[k]:
                    continue
                if out and out[-1] == k:
                    continue
                out.append(groups[k].pop())
                placed = True
                break
            if not placed:
                # Swap with previous when blocked.
                out[-1], out[-2] = out[-2], out[-1]
        return out

    def get_daily_schedule(self, local_now: datetime) -> List[int]:
        day_key = local_now.strftime("%Y-%m-%d")
        with self._cache_lock:
            cached = self._daily_schedule_cache.get(day_key)
            if cached:
                return cached
            schedule = self._build_daily_schedule(day_key)
            self._daily_schedule_cache[day_key] = schedule
            # Keep cache tiny (today + around).
            if len(self._daily_schedule_cache) > 10:
                keys = sorted(self._daily_schedule_cache.keys())
                for old in keys[:-10]:
                    self._daily_schedule_cache.pop(old, None)
            return schedule

    def get_current_window_state(self, now_utc: Optional[datetime] = None) -> SaleWindowState:
        now_utc = now_utc or datetime.utcnow()
        local_now = now_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(self._tz)
        day_key = local_now.strftime("%Y-%m-%d")
        schedule = self.get_daily_schedule(local_now)

        start_hour = (local_now.hour // self.slot_interval_hours) * self.slot_interval_hours
        slot_index = start_hour // self.slot_interval_hours
        if slot_index < 0 or slot_index >= self.slots_per_day:
            return SaleWindowState(
                is_active=False,
                discount_percent=0,
                day_key=day_key,
                window_index=None,
                window_id=None,
                window_start_local=None,
                window_end_local=None,
                timezone=self.timezone_name,
            )

        slot_start = local_now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        slot_end = slot_start + timedelta(minutes=self.slot_minutes)
        is_active = slot_start <= local_now < slot_end
        discount = int(schedule[slot_index]) if is_active else 0
        window_id = f"{day_key}_{slot_index}" if is_active else None

        return SaleWindowState(
            is_active=is_active,
            discount_percent=discount,
            day_key=day_key,
            window_index=slot_index if is_active else None,
            window_id=window_id,
            window_start_local=slot_start.isoformat() if is_active else None,
            window_end_local=slot_end.isoformat() if is_active else None,
            timezone=self.timezone_name,
        )

    def get_today_schedule_snapshot(self, now_utc: Optional[datetime] = None) -> Dict[str, object]:
        now_utc = now_utc or datetime.utcnow()
        local_now = now_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(self._tz)
        day_key = local_now.strftime("%Y-%m-%d")
        schedule = self.get_daily_schedule(local_now)
        state = self.get_current_window_state(now_utc=now_utc)
        return {
            "day_key": day_key,
            "timezone": self.timezone_name,
            "slot_minutes": self.slot_minutes,
            "slots_per_day": self.slots_per_day,
            "slot_interval_hours": self.slot_interval_hours,
            "discount_schedule": schedule,
            "current_window": {
                "is_active": state.is_active,
                "discount_percent": state.discount_percent,
                "window_index": state.window_index,
                "window_id": state.window_id,
                "window_start_local": state.window_start_local,
                "window_end_local": state.window_end_local,
            },
        }

