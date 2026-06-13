"""
ThinkingMemory MCP server.

Exposes the unified memory database as Model Context Protocol tools so any
MCP-capable agent (Claude Desktop, IDE assistants, custom agents) can store and
recall memory with zero glue. Tools call the engine directly — no HTTP hop — and
embeddings are generated server-side, so agents never deal with vectors.

The headline tool is ``recall``: give it an *intent string* and it returns a
ranked, token-budget-packed context window with citations.

Run it over stdio:

    thinkingmemory-mcp
    # or
    python -m thinkingmemory.mcp_server

Configuration comes from the same settings/.env as the REST API (DATABASE_URL,
REDIS_URL, EMBEDDING_PROVIDER, EMBEDDING_DIM). An optional default tenant can be
set with THINKINGMEMORY_TENANT_ID; individual tool calls may override it.
"""

import json
import logging
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from thinkingmemory.engine import store, recall as recall_engine
from thinkingmemory.memory.working import redis_client as working

logger = logging.getLogger("thinkingmemory.mcp")

DEFAULT_TENANT_ID: Optional[str] = os.getenv("THINKINGMEMORY_TENANT_ID") or None

mcp = FastMCP("thinkingmemory")


def _tenant(tenant_id: Optional[str]) -> Optional[str]:
    return tenant_id if tenant_id is not None else DEFAULT_TENANT_ID


def _dump(value) -> str:
    return json.dumps(value, default=str, indent=2)


# =============================================================================
# Unified memory: remember / recall / forget
# =============================================================================


@mcp.tool()
def remember(
    agent_id: str,
    content: dict,
    mtype: str = "episodic",
    scope: str = "private",
    salience: float = 1.0,
    confidence: float = 1.0,
    tenant_id: Optional[str] = None,
) -> str:
    """Store a memory. `content` is any JSON object; `mtype` is one of
    episodic | semantic | procedural | working. The text is embedded
    server-side. Returns the stored memory as JSON.
    """
    return _dump(
        store.remember(
            agent_id=agent_id,
            content=content,
            mtype=mtype,
            scope=scope,
            salience=salience,
            confidence=confidence,
            tenant_id=_tenant(tenant_id),
        )
    )


@mcp.tool()
def recall(
    agent_id: str,
    intent: str,
    token_budget: int = 4000,
    mtypes: Optional[list[str]] = None,
    k: int = 20,
    tenant_id: Optional[str] = None,
) -> str:
    """Recall the most useful context for an `intent`.

    Returns a JSON object with a ready-to-use `context` string (ranked,
    deduped, packed to `token_budget` tokens, with [n] citations), the
    contributing `items`, and `tokens_saved_vs_dump`. Optionally restrict to
    certain `mtypes` (e.g. ["semantic"]).
    """
    return _dump(
        recall_engine.recall(
            agent_id=agent_id,
            intent=intent,
            tenant_id=_tenant(tenant_id),
            mtypes=mtypes,
            token_budget=token_budget,
            k=k,
        )
    )


@mcp.tool()
def remember_fact(
    agent_id: str,
    fact: str,
    confidence: float = 1.0,
    tenant_id: Optional[str] = None,
) -> str:
    """Convenience: store a durable semantic fact (mtype=semantic)."""
    return _dump(
        store.remember(
            agent_id=agent_id,
            content={"text": fact},
            mtype="semantic",
            confidence=confidence,
            tenant_id=_tenant(tenant_id),
        )
    )


@mcp.tool()
def remember_procedure(
    agent_id: str,
    name: str,
    steps: list,
    description: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> str:
    """Convenience: store a reusable procedure (mtype=procedural)."""
    return _dump(
        store.remember(
            agent_id=agent_id,
            content={"name": name, "steps": steps, "description": description},
            text=f"{name}: {description or ''}".strip(),
            mtype="procedural",
            tenant_id=_tenant(tenant_id),
        )
    )


@mcp.tool()
def get_memory(memory_id: int, tenant_id: Optional[str] = None) -> str:
    """Fetch a single memory by id (JSON), or null if not found."""
    return _dump(store.get(memory_id, tenant_id=_tenant(tenant_id)))


@mcp.tool()
def forget(memory_id: int, hard: bool = False, tenant_id: Optional[str] = None) -> str:
    """Forget a memory. Soft (default) is recoverable; hard deletes."""
    ok = store.forget(memory_id, hard=hard, tenant_id=_tenant(tenant_id))
    return _dump({"forgotten": memory_id if ok else None, "hard": hard})


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
def get_working_memory(agent_id: str, key: str, tenant_id: Optional[str] = None) -> str:
    """Get a working-memory value by key. Returns JSON null if missing/expired."""
    return _dump(working.retrieve_working_memory(agent_id, key, tenant_id=_tenant(tenant_id)))


@mcp.tool()
def list_working_memory(agent_id: str, tenant_id: Optional[str] = None) -> str:
    """List all current working-memory entries for an agent as a JSON object."""
    return _dump(working.get_all_working_memory(agent_id, tenant_id=_tenant(tenant_id)))


def main() -> None:
    """Console-script entry point: run the MCP server over stdio."""
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    mcp.run("stdio")


if __name__ == "__main__":
    main()
