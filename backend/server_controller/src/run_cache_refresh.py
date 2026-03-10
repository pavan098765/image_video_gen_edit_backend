#!/usr/bin/env python3
"""
Cache Refresh Service - Standalone Script

Runs periodically (e.g. weekly via cron) to refresh Docker image cache on machines
in machine_cache.json. For each (machine_id, model) pair, starts a pod, waits for
worker registration, keeps it alive briefly, then destroys. Ensures images stay
cached so future pod starts are faster.

Run: python run_cache_refresh.py
Cron: 0 3 * * 0  (e.g. Sunday 3am weekly)

Uses config.env for QUICKPOD_EMAIL, QUICKPOD_PASSWORD, template UUIDs, etc.
"""

import os
import sys
import asyncio
from pathlib import Path

# Ensure we load env from src/config.env
_script_dir = Path(__file__).parent.resolve()
_env_path = _script_dir / "config.env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)
else:
    from dotenv import load_dotenv
    load_dotenv()

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

    config = load_config(strict=False)
    if not config.quickpod.email or not config.quickpod.password:
        logger.error("QUICKPOD_EMAIL and QUICKPOD_PASSWORD required in config.env")
        sys.exit(1)

    init_db()

    pod_manager = PodManager(
        email=config.quickpod.email,
        password=config.quickpod.password,
        warm_pod_uuid=None,
    )
    await pod_manager.initialize()
    await pod_manager.run_cache_refresh()

    logger.info("=" * 60)
    logger.info("CACHE REFRESH SERVICE - COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
