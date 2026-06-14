"""Smoke tests for the MCP server tools (called in-process, backed by the engine)."""

import asyncio
import json

import thinkingmemory.mcp_server as srv


def test_all_tools_registered():
    tools = asyncio.run(srv.mcp.list_tools())
    names = {t.name for t in tools}
    expected = {
        "remember", "recall", "remember_fact", "remember_procedure",
        "get_memory", "forget", "set_working_memory", "get_working_memory",
        "list_working_memory",
    }
    assert expected <= names


def test_remember_and_recall(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    srv.remember_fact(agent_id, "The cache TTL is 300 seconds")
    res = json.loads(srv.recall(agent_id, "how long is the cache TTL?"))
    assert res["items"]
    assert "300 seconds" in res["context"]


def test_mcp_tenant_isolation(db_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    srv.remember(agent_id, {"text": "alpha"}, tenant_id="tenantA")
    a = json.loads(srv.recall(agent_id, "alpha", tenant_id="tenantA"))
    b = json.loads(srv.recall(agent_id, "alpha", tenant_id="tenantB"))
    assert a["items"] and not b["items"]
