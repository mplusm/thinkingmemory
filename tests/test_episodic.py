"""Tests for episodic memory: storage, retrieval, similarity, tenant isolation."""


def test_store_and_retrieve(client, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    r = client.post("/memory/store", json={"agent_id": agent_id, "content": {"note": "hello"}})
    assert r.status_code == 200
    stored = r.json()
    assert stored["content"] == {"note": "hello"}
    assert stored["memory_type"] == "episodic"

    r = client.get(f"/memory/retrieve/{agent_id}")
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 1
    assert items[0]["content"] == {"note": "hello"}


def test_memory_type_is_persisted(client, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    r = client.post(
        "/memory/store",
        json={"agent_id": agent_id, "content": {"x": 1}, "memory_type": "reflection"},
    )
    assert r.status_code == 200
    assert r.json()["memory_type"] == "reflection"


def test_tenant_isolation(client, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    client.post(
        "/memory/store",
        json={"agent_id": agent_id, "content": {"t": "A"}},
        headers={"X-Tenant-ID": "tenantA"},
    )
    client.post(
        "/memory/store",
        json={"agent_id": agent_id, "content": {"t": "B"}},
        headers={"X-Tenant-ID": "tenantB"},
    )

    a = client.get(f"/memory/retrieve/{agent_id}", headers={"X-Tenant-ID": "tenantA"}).json()
    b = client.get(f"/memory/retrieve/{agent_id}", headers={"X-Tenant-ID": "tenantB"}).json()

    assert {m["tenant_id"] for m in a} == {"tenantA"}
    assert {m["tenant_id"] for m in b} == {"tenantB"}


def test_similarity_search_uses_vector(client, agent_id, cleanup_agent, embedding_dim):
    cleanup_agent(agent_id)
    emb = [0.01] * embedding_dim
    client.post("/memory/store", json={"agent_id": agent_id, "content": {"n": 1}, "embedding": emb})

    r = client.post(f"/memory/similar/{agent_id}", json={"embedding": emb, "limit": 5})
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_wrong_embedding_dim_rejected(client, agent_id, embedding_dim):
    # Wrong dimension is a client error (422), not a server error.
    r = client.post(f"/memory/similar/{agent_id}", json={"embedding": [0.1, 0.2], "limit": 5})
    assert r.status_code == 422


def test_forget_old(client, agent_id, cleanup_agent):
    cleanup_agent(agent_id)
    client.post("/memory/store", json={"agent_id": agent_id, "content": {"n": 1}})
    # days=-1 makes the cutoff in the future, so everything is "old".
    r = client.delete(f"/memory/forget/old/{agent_id}?days=-1")
    assert r.status_code == 200
    assert client.get(f"/memory/retrieve/{agent_id}").json() == []
