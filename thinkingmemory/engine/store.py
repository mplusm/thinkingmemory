"""
Writes to the unified Memory substrate: remember / forget / get / trace.

``remember`` embeds the memory's text server-side (via the configured provider)
before inserting, so callers never deal with vectors. ``forget`` is soft by
default (closes the bitemporal window) and recoverable; ``hard=True`` deletes.
"""

from __future__ import annotations

import json
from typing import Optional

from sqlalchemy import text
from sqlmodel import select

from thinkingmemory.core.database import get_engine, get_session_context
from thinkingmemory.core.embeddings import embedding_to_list
from thinkingmemory.core.timeutils import utcnow
from thinkingmemory.engine import audit
from thinkingmemory.engine.embeddings import get_embedder
from thinkingmemory.engine.models import Memory

# Default per-mtype salience decay rates (per day). Episodic experience fades
# fastest; durable semantic/procedural knowledge fades slowly; working memory is
# transient. Used when a caller doesn't specify decay_rate. Recall counteracts
# decay by bumping salience, so frequently-useful memories persist.
DECAY_DEFAULTS = {
    "working": 0.30,     # ~2-day half-life
    "episodic": 0.05,    # ~14-day half-life
    "semantic": 0.005,   # ~140-day half-life
    "procedural": 0.005,
}


def default_decay_rate(mtype: str) -> float:
    return DECAY_DEFAULTS.get(mtype, 0.0)


def render_text(content: dict, text: Optional[str] = None) -> str:
    """Derive the embeddable/packable text for a memory."""
    if text:
        return text
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        # Flatten a dict into readable "key: value" lines.
        return "\n".join(f"{k}: {v}" for k, v in content.items())
    return str(content)


def memory_to_dict(m: Memory) -> dict:
    """Serialize a Memory row to a JSON-friendly dict."""
    return {
        "id": m.id,
        "tenant_id": m.tenant_id,
        "agent_id": m.agent_id,
        "scope": m.scope,
        "mtype": m.mtype,
        "content": m.content,
        "text": m.text,
        "embedding": embedding_to_list(m.embedding),
        "salience": m.salience,
        "confidence": m.confidence,
        "decay_rate": m.decay_rate,
        "recall_count": m.recall_count,
        "last_recalled_at": m.last_recalled_at,
        "valid_from": m.valid_from,
        "valid_to": m.valid_to,
        "created_at": m.created_at,
        "superseded_at": m.superseded_at,
        "provenance": m.provenance,
    }


def remember(
    agent_id: str,
    content: dict,
    *,
    text: Optional[str] = None,
    mtype: str = "episodic",
    scope: str = "private",
    salience: float = 1.0,
    confidence: float = 1.0,
    decay_rate: Optional[float] = None,
    provenance: Optional[dict] = None,
    tenant_id: Optional[str] = None,
    embed: bool = True,
) -> dict:
    """Store one memory, embedding its text server-side."""
    resolved_text = render_text(content, text)
    embedding = get_embedder().embed([resolved_text])[0] if embed else None

    item = Memory(
        agent_id=agent_id,
        content=content,
        text=resolved_text,
        embedding=embedding,
        mtype=mtype,
        scope=scope,
        salience=salience,
        confidence=confidence,
        decay_rate=decay_rate if decay_rate is not None else default_decay_rate(mtype),
        provenance=provenance,
    )
    if tenant_id is not None:
        item.tenant_id = tenant_id

    with get_session_context(tenant_id) as session:
        session.add(item)
        session.commit()
        session.refresh(item)
        result = memory_to_dict(item)

    audit.record("remember", agent_id, tenant_id=tenant_id,
                 target_id=result["id"], details={"mtype": mtype})
    return result


def remember_many(items: list[dict], tenant_id: Optional[str] = None) -> list[dict]:
    """Store many memories with a single batched embedding call.

    Each item is a dict accepting the same keys as ``remember`` (``agent_id`` and
    ``content`` required).
    """
    if not items:
        return []
    texts = [render_text(it["content"], it.get("text")) for it in items]
    vectors = get_embedder().embed(texts)

    rows: list[Memory] = []
    for it, resolved_text, vec in zip(items, texts, vectors):
        row = Memory(
            agent_id=it["agent_id"],
            content=it["content"],
            text=resolved_text,
            embedding=vec,
            mtype=it.get("mtype", "episodic"),
            scope=it.get("scope", "private"),
            salience=it.get("salience", 1.0),
            confidence=it.get("confidence", 1.0),
            decay_rate=it["decay_rate"] if it.get("decay_rate") is not None
            else default_decay_rate(it.get("mtype", "episodic")),
            provenance=it.get("provenance"),
        )
        tid = it.get("tenant_id", tenant_id)
        if tid is not None:
            row.tenant_id = tid
        rows.append(row)

    with get_session_context(tenant_id) as session:
        session.add_all(rows)
        session.commit()
        for row in rows:
            session.refresh(row)
        results = [memory_to_dict(r) for r in rows]

    by_agent: dict = {}
    for r in results:
        by_agent[r["agent_id"]] = by_agent.get(r["agent_id"], 0) + 1
    for ag, count in by_agent.items():
        audit.record("remember_batch", ag, tenant_id=tenant_id, details={"count": count})
    return results


def _fetch(session, memory_id: int, tenant_id: Optional[str]) -> Optional[Memory]:
    """Fetch one Memory by id (composite PK means we can't use session.get(id))."""
    stmt = select(Memory).where(Memory.id == memory_id)
    if tenant_id is not None:
        stmt = stmt.where(Memory.tenant_id == tenant_id)
    return session.exec(stmt).first()


def get(memory_id: int, tenant_id: Optional[str] = None) -> Optional[dict]:
    """Fetch a single memory by id (tenant-scoped if tenant_id given)."""
    with get_session_context(tenant_id) as session:
        item = _fetch(session, memory_id, tenant_id)
        return memory_to_dict(item) if item is not None else None


def forget(memory_id: int, hard: bool = False, tenant_id: Optional[str] = None) -> bool:
    """Forget a memory. Soft (default) closes its bitemporal window; hard deletes."""
    with get_session_context(tenant_id) as session:
        item = _fetch(session, memory_id, tenant_id)
        if item is None:
            return False
        agent_id = item.agent_id
        if hard:
            session.delete(item)
        else:
            now = utcnow()
            item.valid_to = now
            item.superseded_at = now
        session.commit()

    audit.record("forget", agent_id, tenant_id=tenant_id,
                 target_id=memory_id, details={"hard": hard})
    return True


def _clear_working_memory(agent_id: str, tenant_id: Optional[str]) -> int:
    """Best-effort wipe of an agent's Redis working-memory scratchpad.

    Lazily imported so the engine doesn't hard-depend on Redis, and never raises:
    erasure of the durable store must succeed even if Redis is down/unconfigured.
    """
    try:
        from thinkingmemory.memory.working.redis_client import clear_working_memory
        return clear_working_memory(agent_id, tenant_id=tenant_id)
    except Exception:  # pragma: no cover - erasure must not break on Redis errors
        return 0


def forget_agent(agent_id: str, hard: bool = False,
                 tenant_id: Optional[str] = None) -> dict:
    """Forget *all* memory for one agent, across every mtype.

    Soft (default) closes each currently-valid row's bitemporal window so it is
    no longer recalled but remains for audit; hard deletes the rows outright,
    removes the agent's graph edges, and wipes its Redis working-memory
    scratchpad. Idempotent: a second call reports 0. Used to honor a user's
    "erase this project's memory" request (GDPR/CCPA deletion).
    """
    if hard:
        where = ["agent_id = :agent_id"]
        params = {"agent_id": agent_id}
        if tenant_id is not None:
            where.append("tenant_id = :tenant_id"); params["tenant_id"] = tenant_id
        clause = " AND ".join(where)
        with get_engine().begin() as conn:
            forgotten = conn.execute(
                text("DELETE FROM memory WHERE " + clause), params
            ).rowcount
            edges_removed = conn.execute(
                text("DELETE FROM memory_edge WHERE " + clause), params
            ).rowcount
        working_cleared = _clear_working_memory(agent_id, tenant_id)
    else:
        where = ["valid_to IS NULL", "agent_id = :agent_id"]
        params = {"agent_id": agent_id, "now": utcnow()}
        if tenant_id is not None:
            where.append("tenant_id = :tenant_id"); params["tenant_id"] = tenant_id
        sql = (
            "UPDATE memory SET valid_to = :now, superseded_at = :now "
            "WHERE " + " AND ".join(where)
        )
        with get_engine().begin() as conn:
            forgotten = conn.execute(text(sql), params).rowcount
        edges_removed = 0
        working_cleared = 0

    audit.record("forget_agent", agent_id, tenant_id=tenant_id,
                 details={"hard": hard, "forgotten": forgotten,
                          "edges_removed": edges_removed,
                          "working_cleared": working_cleared})
    return {"agent_id": agent_id, "hard": hard, "forgotten": forgotten,
            "edges_removed": edges_removed, "working_cleared": working_cleared}


def forget_tenant(tenant_id: str, hard: bool = False) -> dict:
    """Erase *all* memory for an entire tenant — every agent, every mtype.

    The "delete my account / all my data" path (GDPR Art. 17). Hard deletes all
    memory rows + graph edges + the tenant's entire Redis working memory; soft
    closes every currently-valid row's window. Always writes an audit row (kept
    even on hard delete as the compliance trail).
    """
    if not tenant_id:
        raise ValueError("tenant_id is required")

    if hard:
        with get_engine().begin() as conn:
            forgotten = conn.execute(
                text("DELETE FROM memory WHERE tenant_id = :t"), {"t": tenant_id}
            ).rowcount
            edges_removed = conn.execute(
                text("DELETE FROM memory_edge WHERE tenant_id = :t"), {"t": tenant_id}
            ).rowcount
        try:
            from thinkingmemory.memory.working.redis_client import clear_tenant_working_memory
            working_cleared = clear_tenant_working_memory(tenant_id)
        except Exception:  # pragma: no cover
            working_cleared = 0
    else:
        with get_engine().begin() as conn:
            forgotten = conn.execute(
                text("UPDATE memory SET valid_to = :now, superseded_at = :now "
                     "WHERE tenant_id = :t AND valid_to IS NULL"),
                {"now": utcnow(), "t": tenant_id},
            ).rowcount
        edges_removed = 0
        working_cleared = 0

    audit.record("forget_tenant", tenant_id, tenant_id=tenant_id,
                 details={"hard": hard, "forgotten": forgotten,
                          "edges_removed": edges_removed,
                          "working_cleared": working_cleared})
    return {"tenant_id": tenant_id, "hard": hard, "forgotten": forgotten,
            "edges_removed": edges_removed, "working_cleared": working_cleared}


def export_agent(agent_id: str, tenant_id: Optional[str] = None) -> dict:
    """Full data export for one agent (GDPR Art. 15 access / Art. 20 portability).

    Returns every memory row (including closed/superseded ones), the agent's
    graph edges, and its Redis working memory — a complete, machine-readable dump.
    """
    with get_session_context(tenant_id) as session:
        stmt = select(Memory).where(Memory.agent_id == agent_id)
        if tenant_id is not None:
            stmt = stmt.where(Memory.tenant_id == tenant_id)
        memories = [memory_to_dict(m) for m in session.exec(stmt).all()]

    edges = _export_edges(agent_id=agent_id, tenant_id=tenant_id)
    working = _export_working(agent_id, tenant_id)
    audit.record("export_agent", agent_id, tenant_id=tenant_id,
                 details={"memories": len(memories)})
    return {"agent_id": agent_id, "tenant_id": tenant_id,
            "exported_at": utcnow().isoformat(),
            "memories": memories, "edges": edges, "working_memory": working}


def export_tenant(tenant_id: str) -> dict:
    """Full data export for an entire tenant (all agents)."""
    if not tenant_id:
        raise ValueError("tenant_id is required")
    with get_session_context(tenant_id) as session:
        memories = [
            memory_to_dict(m)
            for m in session.exec(
                select(Memory).where(Memory.tenant_id == tenant_id)
            ).all()
        ]
    edges = _export_edges(tenant_id=tenant_id)
    audit.record("export_tenant", tenant_id, tenant_id=tenant_id,
                 details={"memories": len(memories)})
    return {"tenant_id": tenant_id, "exported_at": utcnow().isoformat(),
            "memories": memories, "edges": edges}


def _export_edges(agent_id: Optional[str] = None,
                  tenant_id: Optional[str] = None) -> list[dict]:
    where, params = [], {}
    if agent_id is not None:
        where.append("agent_id = :a"); params["a"] = agent_id
    if tenant_id is not None:
        where.append("tenant_id = :t"); params["t"] = tenant_id
    sql = "SELECT src_id, dst_id, relation, weight, agent_id, tenant_id FROM memory_edge"
    if where:
        sql += " WHERE " + " AND ".join(where)
    with get_engine().begin() as conn:
        return [dict(r._mapping) for r in conn.execute(text(sql), params)]


def _export_working(agent_id: str, tenant_id: Optional[str]) -> dict:
    try:
        from thinkingmemory.memory.working.redis_client import get_all_working_memory
        return get_all_working_memory(agent_id, tenant_id=tenant_id)
    except Exception:  # pragma: no cover
        return {}


def _trace_node(session, memory_id, tenant_id, depth, seen) -> Optional[dict]:
    """Recursively expand a memory's provenance chain into a tree."""
    if memory_id in seen or depth < 0:
        return None
    seen.add(memory_id)
    item = _fetch(session, memory_id, tenant_id)
    if item is None:
        return None
    prov = item.provenance or {}
    node = {
        "id": item.id,
        "mtype": item.mtype,
        "text": item.text,
        "provenance": prov,
        "derived_from": [],
        "superseded_by": None,
    }
    if depth > 0:
        for src_id in prov.get("derived_from", []):
            child = _trace_node(session, src_id, tenant_id, depth - 1, seen)
            if child:
                node["derived_from"].append(child)
        if prov.get("superseded_by"):
            node["superseded_by"] = _trace_node(
                session, prov["superseded_by"], tenant_id, depth - 1, seen
            )
    return node


def trace(memory_id: int, tenant_id: Optional[str] = None, depth: int = 3) -> Optional[dict]:
    """Why-do-I-know-this: the recursive provenance tree for a memory."""
    with get_session_context(tenant_id) as session:
        return _trace_node(session, memory_id, tenant_id, depth, set())


__all__ = [
    "remember",
    "remember_many",
    "get",
    "forget",
    "forget_agent",
    "forget_tenant",
    "export_agent",
    "export_tenant",
    "trace",
    "render_text",
    "memory_to_dict",
]
