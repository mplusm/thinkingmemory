"""Tests that the memory table is partitioned and works across partitions."""

from sqlalchemy import text

from thinkingmemory.engine import store, recall as recall_engine
from thinkingmemory.core.database import get_engine


def test_memory_table_is_partitioned(db_available):
    with get_engine().connect() as conn:
        kind = conn.execute(
            text("SELECT relkind FROM pg_class WHERE relname = 'memory'")
        ).scalar()
        parts = conn.execute(
            text("SELECT count(*) FROM pg_inherits WHERE inhparent = 'memory'::regclass")
        ).scalar()
    assert kind == "p", "memory should be a partitioned table"
    assert parts >= 2


def test_store_and_recall_across_tenants(db_available, agent_id, cleanup_agent):
    """Different tenants hash to (likely) different partitions; recall still works."""
    cleanup_agent(agent_id)
    for t in ("ptA", "ptB", "ptC", "ptD"):
        store.remember(agent_id, {"text": f"the deploy region for {t} is {t}-region"},
                       mtype="semantic", tenant_id=t)
    for t in ("ptA", "ptB", "ptC", "ptD"):
        res = recall_engine.recall(agent_id, "what is the deploy region?", tenant_id=t)
        assert res["items"], f"no recall for tenant {t}"
        assert all(it["text"].startswith(f"the deploy region for {t}") for it in res["items"])


def test_id_sequence_continues(db_available, agent_id, cleanup_agent):
    """Identity ids keep incrementing after the migration (no collisions)."""
    cleanup_agent(agent_id)
    a = store.remember(agent_id, {"text": "first"}, tenant_id="ptseq")
    b = store.remember(agent_id, {"text": "second"}, tenant_id="ptseq")
    assert b["id"] > a["id"]
