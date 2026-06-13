"""Tests for the unified memory engine: remember, recall, packing, isolation."""

from thinkingmemory.engine import store, recall as recall_engine


def test_remember_and_get(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    m = store.remember(agent_id, {"text": "hello world"}, mtype="episodic")
    assert m["id"] and m["embedding"] and len(m["embedding"]) > 0
    assert store.get(m["id"])["text"] == "hello world"


def test_recall_surfaces_relevant_memory(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    store.remember_many(
        [
            {"agent_id": agent_id, "content": {"text": "The deploy key rotates every 90 days"}, "mtype": "semantic"},
            {"agent_id": agent_id, "content": {"text": "Lunch was good today"}, "mtype": "episodic"},
            {"agent_id": agent_id, "content": {"text": "It rained on Tuesday"}, "mtype": "episodic"},
        ]
    )
    res = recall_engine.recall(agent_id, "when does the deploy key rotate?", k=3)
    assert res["items"], "recall returned nothing"
    assert "deploy key" in res["items"][0]["text"]
    assert res["items"][0]["why"]  # explains why it was retrieved


def test_recall_respects_token_budget(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    store.remember_many(
        [
            {"agent_id": agent_id, "content": {"text": f"Memory number {i} about topic {i}"}, "mtype": "episodic"}
            for i in range(30)
        ]
    )
    res = recall_engine.recall(agent_id, "topic", token_budget=40, k=20)
    assert res["tokens_used"] <= 40
    assert res["dump_tokens"] > res["tokens_used"]
    assert res["tokens_saved_vs_dump"] > 0


def test_recall_tenant_isolation(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    store.remember(agent_id, {"text": "tenant A secret"}, tenant_id="tenantA")
    store.remember(agent_id, {"text": "tenant B secret"}, tenant_id="tenantB")
    a = recall_engine.recall(agent_id, "secret", tenant_id="tenantA")
    b = recall_engine.recall(agent_id, "secret", tenant_id="tenantB")
    assert all("tenant A" in it["text"] for it in a["items"])
    assert all("tenant B" in it["text"] for it in b["items"])


def test_recall_hybrid_beats_recency(db_available, agent_id, cleanup_agent):
    """A keyword-strong but old memory should still be recalled over recent noise."""
    cleanup_agent(agent_id)
    gold = store.remember(agent_id, {"text": "The kubernetes ingress controller config lives in infra/ingress.yaml"}, mtype="semantic")
    # add many newer, irrelevant memories
    store.remember_many(
        [{"agent_id": agent_id, "content": {"text": f"random unrelated note {i}"}, "mtype": "episodic"} for i in range(15)]
    )
    res = recall_engine.recall(agent_id, "where is the kubernetes ingress config?", k=5)
    assert gold["id"] in {it["id"] for it in res["items"]}


def test_forget_soft_then_excluded(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    m = store.remember(agent_id, {"text": "ephemeral note about widgets"}, mtype="episodic")
    assert recall_engine.recall(agent_id, "widgets")["items"]
    store.forget(m["id"])  # soft
    assert recall_engine.recall(agent_id, "widgets")["items"] == []
