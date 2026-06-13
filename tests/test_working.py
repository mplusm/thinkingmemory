"""Tests for working (Redis-backed) memory and its tenant key isolation."""

from thinkingmemory.memory.working import redis_client as working


def test_store_get_delete(redis_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    assert working.store_working_memory(agent_id, "k", {"v": 1}, ttl=60) is True
    assert working.retrieve_working_memory(agent_id, "k") == {"v": 1}
    assert working.working_memory_exists(agent_id, "k") is True
    assert working.delete_working_memory(agent_id, "k") is True
    assert working.retrieve_working_memory(agent_id, "k") is None


def test_update_preserves_ttl(redis_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    working.store_working_memory(agent_id, "k", {"v": 1}, ttl=120)
    working.update_working_memory(agent_id, "k", {"v": 2})  # no ttl -> preserve
    assert working.retrieve_working_memory(agent_id, "k") == {"v": 2}
    assert 0 < working.get_ttl(agent_id, "k") <= 120


def test_tenant_key_isolation(redis_available, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    working.store_working_memory(agent_id, "k", {"t": "A"}, tenant_id="tenantA")
    working.store_working_memory(agent_id, "k", {"t": "B"}, tenant_id="tenantB")

    assert working.retrieve_working_memory(agent_id, "k", tenant_id="tenantA") == {"t": "A"}
    assert working.retrieve_working_memory(agent_id, "k", tenant_id="tenantB") == {"t": "B"}
    # Single-tenant lookup must not see tenant-scoped keys.
    assert working.retrieve_working_memory(agent_id, "k") is None
