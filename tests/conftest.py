"""
Shared pytest fixtures.

The tests exercise the real storage layer against a live Postgres + Redis (the
same services the app uses). If those services are unreachable the DB/Redis
fixtures skip rather than fail, so the suite degrades gracefully in CI without
infrastructure.

Each test gets a unique agent id and all rows/keys created under it are cleaned
up afterward, so tests never collide with each other or with real data.
"""

import uuid

import pytest


@pytest.fixture(scope="session")
def db_available() -> bool:
    """True if the configured Postgres is reachable; skip dependent tests if not."""
    from sqlalchemy import text
    from thinkingmemory.core.database import get_engine

    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Postgres not available: {exc}")


@pytest.fixture(scope="session")
def redis_available() -> bool:
    """True if the configured Redis is reachable; skip dependent tests if not."""
    from thinkingmemory.core.database import get_redis

    try:
        get_redis().ping()
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Redis not available: {exc}")


@pytest.fixture
def agent_id() -> str:
    """A unique agent id per test to isolate data."""
    return f"test_agent_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def cleanup_agent(db_available, redis_available):
    """Track agent ids and delete all their rows/keys after the test."""
    agents: list[str] = []

    def _register(agent: str) -> str:
        agents.append(agent)
        return agent

    yield _register

    from sqlalchemy import text
    from thinkingmemory.core.database import get_engine, get_redis

    tables = [
        "memory",        # unified table
        "memory_audit",  # audit log
        "memoryitem", "fact", "procedure", "datasource", "datatable",
        "datacolumn", "knowledgeentity", "userpreference", "workflowhabit",
    ]
    engine = get_engine()
    with engine.begin() as conn:
        for agent in agents:
            for table in tables:
                conn.execute(
                    text(f"DELETE FROM {table} WHERE agent_id = :a"), {"a": agent}
                )
    r = get_redis()
    for agent in agents:
        for pattern in (f"{agent}:*", f"*:{agent}:*"):
            keys = r.keys(pattern)
            if keys:
                r.delete(*keys)


@pytest.fixture
def client(db_available):
    """A FastAPI TestClient that returns error responses instead of raising."""
    from fastapi.testclient import TestClient
    from thinkingmemory.api.main import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def embedding_dim() -> int:
    from thinkingmemory.core.embeddings import EMBEDDING_DIM

    return EMBEDDING_DIM
