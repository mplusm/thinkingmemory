"""Smoke tests for the MCP server tools (called in-process, backed by CRUD)."""

import asyncio
import json

import thinkingmemory.mcp_server as srv


def test_all_tools_registered():
    tools = asyncio.run(srv.mcp.list_tools())
    names = {t.name for t in tools}
    expected = {
        "remember", "recall", "recall_similar", "forget_old_memories",
        "memory_stats", "store_fact", "recall_facts", "store_procedure",
        "recall_procedures", "set_working_memory", "get_working_memory",
        "list_working_memory",
    }
    assert expected <= names


def test_remember_and_recall(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    stored = json.loads(srv.remember(agent_id, {"event": "did a thing"}))
    assert stored["content"] == {"event": "did a thing"}

    recalled = json.loads(srv.recall(agent_id, limit=5))
    assert len(recalled) == 1
    assert recalled[0]["content"] == {"event": "did a thing"}


def test_mcp_tenant_isolation(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    srv.remember(agent_id, {"t": "A"}, tenant_id="tenantA")
    assert len(json.loads(srv.recall(agent_id, tenant_id="tenantA"))) == 1
    assert len(json.loads(srv.recall(agent_id, tenant_id="tenantB"))) == 0


def test_mcp_rejects_wrong_embedding_dim(db_available, agent_id):
    import pytest

    with pytest.raises(ValueError):
        srv.remember(agent_id, {"x": 1}, embedding=[0.1, 0.2])
