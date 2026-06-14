"""
Bitemporal queries — belief over time.

Every memory records when it was true (``valid_from``/``valid_to``) and when we
learned/closed it (``created_at``/``superseded_at``). That lets us answer "what
did this agent believe at time T?" — for recall as-of a past moment, for an
audit timeline, and for reconstructing state after a contradiction was resolved.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import select

from thinkingmemory.core.database import get_session_context
from thinkingmemory.core.timeutils import to_naive_utc
from thinkingmemory.engine.models import Memory
from thinkingmemory.engine.store import memory_to_dict


def temporal_conditions(as_of: Optional[datetime]):
    """Conditions selecting memories believed at ``as_of`` (or now if None)."""
    if as_of is None:
        return [Memory.valid_to.is_(None)]
    as_of = to_naive_utc(as_of)
    return [
        Memory.created_at <= as_of,                     # we'd learned it by then
        Memory.valid_from <= as_of,                     # it was true by then
        (Memory.valid_to.is_(None)) | (Memory.valid_to > as_of),  # not yet closed
    ]


def timeline(
    agent_id: str,
    as_of: datetime,
    tenant_id: Optional[str] = None,
    mtypes: Optional[list[str]] = None,
    limit: int = 200,
) -> dict:
    """Return the set of memories an agent believed at a point in time."""
    as_of = to_naive_utc(as_of)
    with get_session_context() as session:
        conds = [Memory.agent_id == agent_id, *temporal_conditions(as_of)]
        if tenant_id is not None:
            conds.append(Memory.tenant_id == tenant_id)
        if mtypes:
            conds.append(Memory.mtype.in_(mtypes))
        rows = session.exec(
            select(Memory).where(*conds).order_by(Memory.created_at.desc()).limit(limit)
        ).all()
        return {
            "agent_id": agent_id,
            "as_of": as_of,
            "count": len(rows),
            "memories": [memory_to_dict(m) for m in rows],
        }


__all__ = ["temporal_conditions", "timeline"]
