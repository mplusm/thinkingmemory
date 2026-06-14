"""Tests for Phase 3: bitemporal timeline, as_of recall, recursive trace, audit."""

import time

from thinkingmemory.engine import store, recall as recall_engine, lifecycle, audit
from thinkingmemory.engine.temporal import timeline
from thinkingmemory.core.timeutils import utcnow


def test_timeline_reflects_belief_at_time(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    old = store.remember(agent_id, {"text": "The office WiFi password is sunflower2024"}, mtype="semantic")
    t_mid = utcnow()
    time.sleep(0.01)
    # the fact changes; supersede the old one
    new = store.remember(agent_id, {"text": "The office WiFi password is moonbeam2025"}, mtype="semantic")
    lifecycle.resolve_duplicates(agent_id, distance=0.4)  # close the near-duplicate old fact

    # as of t_mid, only the old belief existed and was still valid
    past = timeline(agent_id, as_of=t_mid)
    past_ids = {m["id"] for m in past["memories"]}
    assert old["id"] in past_ids
    assert new["id"] not in past_ids  # not yet created at t_mid


def test_as_of_recall(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    m = store.remember(agent_id, {"text": "deploy target is staging"}, mtype="semantic")
    t_after = utcnow()
    # now forget it (soft close)
    store.forget(m["id"])
    # current recall sees nothing; as-of recall before the close still sees it
    assert recall_engine.recall(agent_id, "deploy target")["items"] == []
    past = recall_engine.recall(agent_id, "deploy target", as_of=t_after)
    assert any(it["id"] == m["id"] for it in past["items"])


def test_recursive_trace(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    store.remember_many(
        [
            {"agent_id": agent_id, "content": {"text": "Deployed billing to prod"}, "mtype": "episodic"},
            {"agent_id": agent_id, "content": {"text": "Deployed billing service to production"}, "mtype": "episodic"},
            {"agent_id": agent_id, "content": {"text": "Billing prod deployment finished"}, "mtype": "episodic"},
        ]
    )
    lifecycle.consolidate(agent_id, distance=0.4, min_cluster=3)
    # find the consolidated summary and trace it back to its sources
    res = recall_engine.recall(agent_id, "billing deployment", mtypes=["semantic"], track=False)
    summary_id = next(it["id"] for it in res["items"]
                      if it["provenance"] and it["provenance"].get("source") == "consolidation")
    tree = store.trace(summary_id)
    assert tree["provenance"]["source"] == "consolidation"
    assert len(tree["derived_from"]) >= 3   # expanded to its source memories


def test_audit_records_operations(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    store.remember(agent_id, {"text": "auditable event"}, mtype="episodic")
    recall_engine.recall(agent_id, "auditable event")
    entries = audit.query(agent_id=agent_id)
    actions = {e["action"] for e in entries}
    assert "remember" in actions and "recall" in actions
