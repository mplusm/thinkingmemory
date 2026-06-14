"""
Writes to the unified Memory substrate: remember / forget / get / trace.

``remember`` embeds the memory's text server-side (via the configured provider)
before inserting, so callers never deal with vectors. ``forget`` is soft by
default (closes the bitemporal window) and recoverable; ``hard=True`` deletes.
"""

from __future__ import annotations

import json
from typing import Optional

from sqlmodel import select

from thinkingmemory.core.database import get_session_context
from thinkingmemory.core.embeddings import embedding_to_list
from thinkingmemory.core.timeutils import utcnow
from thinkingmemory.engine.embeddings import get_embedder
from thinkingmemory.engine.models import Memory


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
        provenance=provenance,
    )
    if tenant_id is not None:
        item.tenant_id = tenant_id

    with get_session_context() as session:
        session.add(item)
        session.commit()
        session.refresh(item)
        return memory_to_dict(item)


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
            provenance=it.get("provenance"),
        )
        tid = it.get("tenant_id", tenant_id)
        if tid is not None:
            row.tenant_id = tid
        rows.append(row)

    with get_session_context() as session:
        session.add_all(rows)
        session.commit()
        for row in rows:
            session.refresh(row)
        return [memory_to_dict(r) for r in rows]


def get(memory_id: int, tenant_id: Optional[str] = None) -> Optional[dict]:
    """Fetch a single memory by id (tenant-scoped if tenant_id given)."""
    with get_session_context() as session:
        item = session.get(Memory, memory_id)
        if item is None or (tenant_id is not None and item.tenant_id != tenant_id):
            return None
        return memory_to_dict(item)


def forget(memory_id: int, hard: bool = False, tenant_id: Optional[str] = None) -> bool:
    """Forget a memory. Soft (default) closes its bitemporal window; hard deletes."""
    with get_session_context() as session:
        item = session.get(Memory, memory_id)
        if item is None or (tenant_id is not None and item.tenant_id != tenant_id):
            return False
        if hard:
            session.delete(item)
        else:
            now = utcnow()
            item.valid_to = now
            item.superseded_at = now
        session.commit()
        return True


def trace(memory_id: int, tenant_id: Optional[str] = None) -> Optional[dict]:
    """Return a memory's provenance and the memories it was derived from."""
    with get_session_context() as session:
        item = session.get(Memory, memory_id)
        if item is None or (tenant_id is not None and item.tenant_id != tenant_id):
            return None
        derived_from = (item.provenance or {}).get("derived_from", [])
        sources = []
        if derived_from:
            stmt = select(Memory).where(Memory.id.in_(derived_from))
            sources = [memory_to_dict(s) for s in session.exec(stmt).all()]
        return {
            "id": item.id,
            "provenance": item.provenance,
            "derived_from": sources,
        }


__all__ = [
    "remember",
    "remember_many",
    "get",
    "forget",
    "trace",
    "render_text",
    "memory_to_dict",
]
