"""
Background lifecycle scheduler.

Runs the maintenance cycle (decay → consolidate → supersede → forget → prune)
for every active agent on a fixed interval, so recall quality is maintained
without an external cron. Uses APScheduler in-process; start it from the app
lifespan when ``scheduler_enabled`` is set. For multi-process deployments, run a
single dedicated scheduler process (or a real cron via scripts/run_lifecycle.py)
rather than enabling it in every worker.
"""

from __future__ import annotations

import logging

from sqlalchemy import text

from thinkingmemory.config.settings import get_settings
from thinkingmemory.core.database import get_engine
from thinkingmemory.engine import lifecycle

logger = logging.getLogger("thinkingmemory.scheduler")

_scheduler = None


def _active_agents() -> list[tuple[str, str]]:
    with get_engine().connect() as conn:
        return [
            (r[0], r[1])
            for r in conn.execute(
                text("SELECT DISTINCT agent_id, tenant_id FROM memory WHERE valid_to IS NULL")
            )
        ]


def run_maintenance_cycle() -> dict:
    """Run lifecycle.run_all for every active agent; return aggregate counts."""
    totals: dict[str, int] = {}
    agents = _active_agents()
    for agent_id, tenant_id in agents:
        counts = lifecycle.run_all(agent_id, tenant_id)
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
    logger.info("Lifecycle cycle over %d agents: %s", len(agents), totals)
    return totals


def start_scheduler():
    """Start the background scheduler (idempotent). Returns the scheduler."""
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    from apscheduler.schedulers.background import BackgroundScheduler

    minutes = get_settings().scheduler_interval_minutes
    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        run_maintenance_cycle,
        "interval",
        minutes=minutes,
        id="lifecycle",
        max_instances=1,
        coalesce=True,
    )
    _scheduler.start()
    logger.info("Lifecycle scheduler started (every %d min)", minutes)
    return _scheduler


def shutdown_scheduler():
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None


__all__ = ["start_scheduler", "shutdown_scheduler", "run_maintenance_cycle"]
