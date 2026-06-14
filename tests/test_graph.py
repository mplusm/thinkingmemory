"""Tests for graph-hop recall (relational edges + recursive CTE)."""

from thinkingmemory.engine import store, graph, recall as recall_engine


def test_link_and_neighbors(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    a = store.remember(agent_id, {"text": "incident: checkout latency spiked"}, mtype="episodic")
    b = store.remember(agent_id, {"text": "root cause: a slow database index"}, mtype="semantic")
    c = store.remember(agent_id, {"text": "fix: added a composite index on orders"}, mtype="procedural")
    graph.link(a["id"], b["id"], relation="caused_by", agent_id=agent_id)
    graph.link(b["id"], c["id"], relation="fixed_by", agent_id=agent_id)

    one = graph.neighbors([a["id"]], depth=1)
    assert b["id"] in one and c["id"] not in one
    two = graph.neighbors([a["id"]], depth=2)
    assert two[c["id"]] == 2  # reachable in two hops


def test_graph_hops_surfaces_connected_memory(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    # The "fix" memory shares no keywords with the incident query, so lexical and
    # vector retrieval alone would miss it — the graph edge is what connects them.
    incident = store.remember(agent_id, {"text": "checkout latency spiked during the sale"}, mtype="episodic")
    fix = store.remember(agent_id, {"text": "added composite index on orders(customer_id, created_at)"}, mtype="procedural")
    graph.link(incident["id"], fix["id"], relation="resolved_by", agent_id=agent_id)
    # distractors more related to the query than `fix`, so `fix` falls outside
    # the vector/keyword seed set and is only reachable through the graph edge.
    store.remember_many(
        [{"agent_id": agent_id, "content": {"text": f"checkout latency note number {i} about page load and spikes"}, "mtype": "episodic"}
         for i in range(12)]
    )

    withg = recall_engine.recall(agent_id, "why did checkout latency spike", k=20, graph_hops=1, track=False)
    fix_item = next((it for it in withg["items"] if it["id"] == fix["id"]), None)
    assert fix_item is not None, "graph_hops should surface the connected memory"
    assert "graph" in fix_item["why"], "the graph retriever should be credited (not just lexical/vector)"


def test_neighbors_respect_tenant(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    a = store.remember(agent_id, {"text": "A"}, tenant_id="gtA")
    b = store.remember(agent_id, {"text": "B"}, tenant_id="gtA")
    graph.link(a["id"], b["id"], agent_id=agent_id, tenant_id="gtA")
    # querying under a different tenant sees no edges
    assert graph.neighbors([a["id"]], depth=1, tenant_id="gtB") == {}
