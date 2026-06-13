"""
ThinkingMemory MCP server.

Exposes the memory platform as Model Context Protocol tools so any MCP-capable
agent (Claude Desktop, IDE assistants, custom agents) can store and recall
memory with zero glue code. Tools call the CRUD layer directly — no HTTP hop —
so the server can run as a standalone process next to the database.

Run it over stdio:

    thinkingmemory-mcp
    # or
    python -m thinkingmemory.mcp_server

Configuration is read from the same settings/.env as the REST API
(DATABASE_URL, REDIS_URL, EMBEDDING_DIM). An optional default tenant can be set
with the THINKINGMEMORY_TENANT_ID environment variable; individual tool calls
may override it via the `tenant_id` argument.

Note: similarity tools require a precomputed embedding vector of EMBEDDING_DIM
floats — this server does not generate embeddings. The recency-based `recall`
tool needs no embedding and is the common path.
"""

import json
import logging
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from thinkingmemory.core.embeddings import EMBEDDING_DIM
from thinkingmemory.memory.episodic import crud as episodic
from thinkingmemory.memory.semantic import crud as semantic
from thinkingmemory.memory.procedural import crud as procedural
from thinkingmemory.memory.working import redis_client as working

logger = logging.getLogger("thinkingmemory.mcp")

# Optional default tenant for this server process; per-call argument wins.
DEFAULT_TENANT_ID: Optional[str] = os.getenv("THINKINGMEMORY_TENANT_ID") or None

mcp = FastMCP("thinkingmemory")


def _tenant(tenant_id: Optional[str]) -> Optional[str]:
    """Resolve the effective tenant: explicit arg overrides the process default."""
    return tenant_id if tenant_id is not None else DEFAULT_TENANT_ID


def _dump(value) -> str:
    """Serialize a CRUD result to a JSON string (datetimes -> ISO strings)."""
    return json.dumps(value, default=str, indent=2)


def _check_embedding(embedding: Optional[list[float]]) -> None:
    if embedding is not None and len(embedding) != EMBEDDING_DIM:
        raise ValueError(
            f"embedding must have {EMBEDDING_DIM} dimensions, got {len(embedding)}"
        )


# =============================================================================
# Episodic memory (experiences / events)
# =============================================================================


@mcp.tool()
def remember(
    agent_id: str,
    content: dict,
    memory_type: str = "episodic",
    embedding: Optional[list[float]] = None,
    extra_data: Optional[dict] = None,
    tenant_id: Optional[str] = None,
) -> str:
    """Store an episodic memory (an experience, action, or observation).

    `content` is an arbitrary JSON object describing what happened. Pass an
    `embedding` (EMBEDDING_DIM floats) only if you want it retrievable via
    `recall_similar`. Returns the stored memory as JSON.
    """
    _check_embedding(embedding)
    result = episodic.store_memory(
        agent_id=agent_id,
        content=content,
        embedding=embedding,
        extra_data=extra_data,
        memory_type=memory_type,
        tenant_id=_tenant(tenant_id),
    )
    return _dump(result)


@mcp.tool()
def recall(
    agent_id: str,
    limit: int = 10,
    tenant_id: Optional[str] = None,
) -> str:
    """Recall an agent's most recent episodic memories (newest first).

    This is the common, embedding-free retrieval path. Returns a JSON list.
    """
    result = episodic.retrieve_memories(
        agent_id=agent_id, limit=limit, tenant_id=_tenant(tenant_id)
    )
    return _dump(result)


@mcp.tool()
def recall_similar(
    agent_id: str,
    embedding: list[float],
    limit: int = 10,
    tenant_id: Optional[str] = None,
) -> str:
    """Recall episodic memories most similar to a query `embedding`.

    Requires a precomputed embedding of EMBEDDING_DIM floats. Returns a JSON
    list ordered by vector (L2) similarity.
    """
    _check_embedding(embedding)
    result = episodic.retrieve_similar_memories(
        agent_id=agent_id, embedding=embedding, limit=limit, tenant_id=_tenant(tenant_id)
    )
    return _dump(result)


@mcp.tool()
def forget_old_memories(
    agent_id: str,
    days: int = 30,
    tenant_id: Optional[str] = None,
) -> str:
    """Forget episodic memories older than `days`. Returns the count deleted."""
    deleted = episodic.forget_old_memories(
        agent_id=agent_id, days=days, tenant_id=_tenant(tenant_id)
    )
    return _dump({"deleted": deleted})


@mcp.tool()
def memory_stats(agent_id: str, tenant_id: Optional[str] = None) -> str:
    """Get statistics about an agent's episodic memory as JSON."""
    return _dump(episodic.get_memory_stats(agent_id, tenant_id=_tenant(tenant_id)))


# =============================================================================
# Semantic memory (durable facts)
# =============================================================================


@mcp.tool()
def store_fact(
    agent_id: str,
    fact: str,
    confidence: float = 1.0,
    source: Optional[str] = None,
    embedding: Optional[list[float]] = None,
    tenant_id: Optional[str] = None,
) -> str:
    """Store a durable semantic fact the agent should know long-term."""
    _check_embedding(embedding)
    result = semantic.store_fact(
        agent_id=agent_id,
        fact=fact,
        embedding=embedding,
        confidence=confidence,
        source=source,
        tenant_id=_tenant(tenant_id),
    )
    return _dump(result)


@mcp.tool()
def recall_facts(
    agent_id: str,
    limit: int = 10,
    tenant_id: Optional[str] = None,
) -> str:
    """Recall stored semantic facts for an agent as a JSON list."""
    return _dump(
        semantic.retrieve_facts(agent_id, limit=limit, tenant_id=_tenant(tenant_id))
    )


# =============================================================================
# Procedural memory (reusable how-to / workflows)
# =============================================================================


@mcp.tool()
def store_procedure(
    agent_id: str,
    name: str,
    steps: list[dict],
    description: Optional[str] = None,
    success_rate: float = 1.0,
    tenant_id: Optional[str] = None,
) -> str:
    """Store a reusable procedure (an ordered list of step objects)."""
    result = procedural.store_procedure(
        agent_id=agent_id,
        name=name,
        description=description,
        steps=steps,
        success_rate=success_rate,
        tenant_id=_tenant(tenant_id),
    )
    return _dump(result)


@mcp.tool()
def recall_procedures(
    agent_id: str,
    limit: int = 10,
    tenant_id: Optional[str] = None,
) -> str:
    """Recall stored procedures for an agent as a JSON list."""
    return _dump(
        procedural.retrieve_procedures(agent_id, limit=limit, tenant_id=_tenant(tenant_id))
    )


# =============================================================================
# Working memory (short-term, TTL-backed scratchpad)
# =============================================================================


@mcp.tool()
def set_working_memory(
    agent_id: str,
    key: str,
    value: dict,
    ttl: int = 300,
    tenant_id: Optional[str] = None,
) -> str:
    """Set a short-term working-memory key (expires after `ttl` seconds)."""
    working.store_working_memory(
        agent_id=agent_id, key=key, value=value, ttl=ttl, tenant_id=_tenant(tenant_id)
    )
    return _dump({"stored": key, "ttl": ttl})


@mcp.tool()
def get_working_memory(
    agent_id: str,
    key: str,
    tenant_id: Optional[str] = None,
) -> str:
    """Get a working-memory value by key. Returns JSON null if missing/expired."""
    value = working.retrieve_working_memory(agent_id, key, tenant_id=_tenant(tenant_id))
    return _dump(value)


@mcp.tool()
def list_working_memory(
    agent_id: str,
    tenant_id: Optional[str] = None,
) -> str:
    """List all current working-memory entries for an agent as a JSON object."""
    return _dump(working.get_all_working_memory(agent_id, tenant_id=_tenant(tenant_id)))


def main() -> None:
    """Console-script entry point: run the MCP server over stdio."""
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    mcp.run("stdio")


if __name__ == "__main__":
    main()
