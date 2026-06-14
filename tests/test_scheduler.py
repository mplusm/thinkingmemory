"""Tests for the lifecycle scheduler."""

import time

from thinkingmemory.engine import store, scheduler


def test_run_maintenance_cycle(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    store.remember(agent_id, {"text": "a memory to maintain"}, mtype="episodic")
    totals = scheduler.run_maintenance_cycle()
    # aggregate counts dict has the lifecycle pass keys
    assert {"decayed", "consolidated", "forgotten"} <= set(totals)


def test_scheduler_start_and_fire(db_available, monkeypatch):
    """Start with a tiny interval and confirm the job is registered and runs."""
    from thinkingmemory.config.settings import get_settings

    get_settings().scheduler_interval_minutes  # ensure settings load
    fired = {"n": 0}
    monkeypatch.setattr(scheduler, "run_maintenance_cycle", lambda: fired.__setitem__("n", fired["n"] + 1))
    sch = scheduler.start_scheduler()
    try:
        assert sch.get_job("lifecycle") is not None
        # trigger the job immediately rather than waiting for the interval
        sch.get_job("lifecycle").modify(next_run_time=__import__("datetime").datetime.now())
        time.sleep(1.5)
        assert fired["n"] >= 1
    finally:
        scheduler.shutdown_scheduler()
