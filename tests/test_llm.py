"""Tests for pluggable LLM features (offline NoLLM default)."""

from thinkingmemory.engine import store, lifecycle, recall as recall_engine
from thinkingmemory.engine.llm import get_llm, NoLLM


def test_default_provider_is_offline():
    assert isinstance(get_llm(), NoLLM)


def test_offline_extract_facts():
    llm = NoLLM()
    facts = llm.extract_facts("The DB is on port 5432. It uses pgvector. Is it fast?")
    assert "The DB is on port 5432" in facts
    assert any("pgvector" in f for f in facts)
    assert not any(f.endswith("?") for f in facts)  # questions dropped


def test_offline_contradiction_polarity():
    llm = NoLLM()
    assert llm.is_contradiction("The service is enabled", "The service is not enabled")
    assert not llm.is_contradiction("The sky is blue", "Postgres uses MVCC")


def test_extract_semantic_creates_linked_facts(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    src = store.remember(
        agent_id,
        {"text": "We migrated billing to Stripe. The cutover happened on Friday. Dual-write ran for two weeks."},
        mtype="episodic",
    )
    n = lifecycle.extract_semantic(agent_id)
    assert n >= 1
    res = recall_engine.recall(agent_id, "billing migration to stripe", mtypes=["semantic"], track=False)
    assert any(it["provenance"] and it["provenance"].get("source") == "extraction"
               and src["id"] in it["provenance"].get("derived_from", []) for it in res["items"])


def test_resolve_contradictions_closes_older(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    old = store.remember(agent_id, {"text": "The deployment pipeline is enabled for production"}, mtype="semantic")
    new = store.remember(agent_id, {"text": "The deployment pipeline is not enabled for production"}, mtype="semantic")
    resolved = lifecycle.resolve_contradictions(agent_id, sim_threshold=0.6)
    assert resolved >= 1
    older = store.get(old["id"])
    assert older["valid_to"] is not None
    assert older["provenance"]["contradicted_by"] == new["id"]
