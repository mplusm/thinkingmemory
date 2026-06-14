"""Tests for cross-encoder reranking."""

from thinkingmemory.engine import store, recall as recall_engine
from thinkingmemory.engine.rerank import get_reranker


def test_reranker_scores_relevant_higher():
    rr = get_reranker()
    scores = rr.rerank(
        "how do I restart the analytics pipeline?",
        ["To restart the pipeline: stop workers, flush redis, replay kafka.",
         "The cafeteria serves lunch at noon."],
    )
    assert scores[0] > scores[1]


def test_recall_with_rerank_returns_relevant_first(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    store.remember_many(
        [
            {"agent_id": agent_id, "content": {"text": "Random note about office plants"}, "mtype": "episodic"},
            {"agent_id": agent_id, "content": {"text": "To restart the analytics pipeline, stop workers then replay kafka offsets"}, "mtype": "procedural"},
            {"agent_id": agent_id, "content": {"text": "The weather was sunny on Friday"}, "mtype": "episodic"},
        ]
    )
    res = recall_engine.recall(agent_id, "how to restart the analytics pipeline", rerank=True, k=3)
    assert res["items"]
    assert "restart the analytics pipeline" in res["items"][0]["text"]
    assert "rerank" in res["items"][0]["why"]
