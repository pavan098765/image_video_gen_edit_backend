#!/usr/bin/env python3
"""
Cache Refresh Service - Standalone Script

Runs periodically (e.g. via cron) to refresh Docker image cache on machines
in machine_cache.json. For each (machine_id, model) pair, starts a pod, waits for
worker registration, keeps it alive briefly, then destroys. Ensures images stay
cached so future pod starts are faster.

When FIXED_FLEET_MACHINE_IDS is set in config.env, only those machines are refreshed
(same allowlist as fixed-fleet scheduling).

Minimum interval: CACHE_REFRESH_MIN_INTERVAL_DAYS (default 20). If the state file
is newer than that, the script exits without work — so you can keep a frequent cron
and still run at most once per N days.

Run: python run_cache_refresh.py
Cron example (invoke often; script enforces 20-day gap): 0 3 * * 0
Force a run: CACHE_REFRESH_FORCE=1 python run_cache_refresh.py

Uses config.env for QUICKPOD_EMAIL, QUICKPOD_PASSWORD, template UUIDs, etc.
Use the SAME config.env and Python/venv as the main server so credentials and
env match (avoids 401 when main server uses cached token but script does fresh login).
"""

import os
import sys
import asyncio
from datetime import datetime, timezone
from pathlib import Path

# Ensure we load env from src/config.env
_script_dir = Path(__file__).parent.resolve()
_env_path = _script_dir / "config.env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)
    print(f"Loaded environment from {_env_path}")
else:
    from dotenv import load_dotenv
    load_dotenv()
    print("Env file not found, Loaded environment from .env")

# Change to script dir so relative paths (machine_cache.json, etc.) work
os.chdir(_script_dir)

from logging_config import setup_logging, get_logger
from config import load_config
from db import init_db
from pod_manager import PodManager

setup_logging()
logger = get_logger("run_cache_refresh")


async def main():
    """Run one cache refresh cycle."""
    logger.info("=" * 60)
    logger.info("CACHE REFRESH SERVICE - START")
    logger.info("=" * 60)

    force = os.getenv("CACHE_REFRESH_FORCE", "").strip().lower() in ("1", "true", "yes")
    min_days = int(
        os.getenv(
            "CACHE_REFRESH_MIN_INTERVAL_DAYS",
            os.getenv("REFRESH_RUN_INTERVAL_DAYS", "20"),
        )
    )
    min_days = max(1, min_days)
    state_raw = os.getenv("CACHE_REFRESH_STATE_PATH", "").strip()
    state_path = Path(state_raw) if state_raw else (_script_dir / ".cache_refresh_last_run")

    if not force and state_path.exists():
        mtime = datetime.fromtimestamp(state_path.stat().st_mtime, tz=timezone.utc)
        age_days = (datetime.now(timezone.utc) - mtime).total_seconds() / 86400.0
        if age_days < min_days:
            logger.info(
                "Cache refresh skipped: last run %.2f days ago (min interval %s days). "
                "State file: %s. Set CACHE_REFRESH_FORCE=1 to run anyway.",
                age_days,
                min_days,
                state_path,
            )
            return

    config = load_config(env_file=str(_env_path), strict=False)
    if not config.quickpod.email or not config.quickpod.password:
        logger.error("QUICKPOD_EMAIL and QUICKPOD_PASSWORD required in config.env")
        sys.exit(1)

    # Strip whitespace/CRLF so credentials match what main server uses (avoids 401 from trailing \\r)
    email = (config.quickpod.email or "").strip()
    password = (config.quickpod.password or "").strip()
    if not email or not password:
        logger.error("QUICKPOD_EMAIL and QUICKPOD_PASSWORD are empty after strip")
        sys.exit(1)

    init_db()

    logger.info("Using config from %s", _env_path)
    logger.info("Using QuickPod email: %s", email)
    pod_manager = PodManager(
        email=email,
        password=password,
        warm_pod_uuid=None,
    )
    await pod_manager.initialize()
    await pod_manager.run_cache_refresh()

    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.touch()
        logger.info("Updated cache refresh state: %s", state_path)
    except OSError as e:
        logger.warning("Could not write cache refresh state file %s: %s", state_path, e)

    logger.info("=" * 60)
    logger.info("CACHE REFRESH SERVICE - COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
