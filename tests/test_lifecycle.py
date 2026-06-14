"""Tests for the lifecycle engine: decay, consolidation, forgetting, supersession."""

from thinkingmemory.engine import store, lifecycle
from thinkingmemory.engine.store import get as get_memory


def test_decay_reduces_salience(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    m = store.remember(agent_id, {"text": "an episodic event"}, mtype="episodic")
    assert m["decay_rate"] > 0  # episodic gets a default decay rate
    lifecycle.apply_decay(interval_days=10, agent_id=agent_id)
    after = get_memory(m["id"])
    assert after["salience"] < m["salience"]


def test_recall_bumps_then_decay_lowers_salience(db_available, agent_id, cleanup_agent):
    """The lifecycle feedback loop: recall raises salience, decay lowers it."""
    cleanup_agent(agent_id)
    m = store.remember(agent_id, {"text": "the api gateway timeout is 30 seconds"}, mtype="episodic")
    from thinkingmemory.engine import recall as recall_engine
    recall_engine.recall(agent_id, "what is the api gateway timeout?")  # bumps salience
    after_recall = get_memory(m["id"])
    assert after_recall["recall_count"] >= 1
    assert after_recall["salience"] > m["salience"]  # recall raised it above baseline
    lifecycle.apply_decay(interval_days=10, agent_id=agent_id)
    after_decay = get_memory(m["id"])
    assert after_decay["salience"] < after_recall["salience"]  # decay pulled it back down


def test_consolidation_creates_summary(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    # a tight cluster of near-identical episodic memories
    store.remember_many(
        [
            {"agent_id": agent_id, "content": {"text": "Deployed the billing service to production"}, "mtype": "episodic"},
            {"agent_id": agent_id, "content": {"text": "Deployed billing service to prod successfully"}, "mtype": "episodic"},
            {"agent_id": agent_id, "content": {"text": "Billing service deployment to production completed"}, "mtype": "episodic"},
        ]
    )
    created = lifecycle.consolidate(agent_id, distance=0.35, min_cluster=3)
    assert created >= 1
    # a semantic summary now exists with provenance linking sources
    from thinkingmemory.engine.recall import recall
    res = recall(agent_id, "billing service deployment", mtypes=["semantic"])
    assert any(it["provenance"] and it["provenance"].get("source") == "consolidation" for it in res["items"])


def test_forget_decayed_soft_closes(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    m = store.remember(agent_id, {"text": "low value note"}, mtype="episodic")
    # drive salience below threshold
    lifecycle.apply_decay(interval_days=200, agent_id=agent_id)
    n = lifecycle.forget_decayed(agent_id, salience_threshold=0.2, idle_days=0)
    assert n >= 1
    after = get_memory(m["id"])
    assert after["valid_to"] is not None  # soft-closed


def test_supersede_near_duplicates(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    old = store.remember(agent_id, {"text": "The office WiFi password is sunflower2024"}, mtype="semantic")
    new = store.remember(agent_id, {"text": "The office WiFi password is sunflower2024"}, mtype="semantic")
    superseded = lifecycle.resolve_duplicates(agent_id, distance=0.05)
    assert superseded >= 1
    # the older one is closed and points at the survivor
    older = get_memory(old["id"])
    assert older["valid_to"] is not None
    assert older["provenance"]["superseded_by"] == new["id"]
