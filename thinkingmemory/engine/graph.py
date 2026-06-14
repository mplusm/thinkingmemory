"""
The entity graph — typed relationships between memories, traversed for recall.

Agents' knowledge is relational ("incident X was caused by deploy Y", "task A
depends on procedure B"). A relational edge table plus a recursive CTE gives
multi-hop traversal on plain Postgres (no graph extension needed). Recall uses
it to pull in memories connected to the strongest lexical/vector hits.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import text
from sqlmodel import select

from thinkingmemory.core.database import get_session_context
from thinkingmemory.engine.models import MemoryEdge


def link(
    src_id: int,
    dst_id: int,
    relation: str = "relates_to",
    agent_id: str = "",
    weight: float = 1.0,
    tenant_id: Optional[str] = None,
    bidirectional: bool = False,
) -> dict:
    """Create a directed edge src -> dst (optionally both directions)."""
    with get_session_context(tenant_id) as session:
        edges = [MemoryEdge(src_id=src_id, dst_id=dst_id, relation=relation,
                            agent_id=agent_id, weight=weight)]
        if bidirectional:
            edges.append(MemoryEdge(src_id=dst_id, dst_id=src_id, relation=relation,
                                    agent_id=agent_id, weight=weight))
        for e in edges:
            if tenant_id is not None:
                e.tenant_id = tenant_id
        session.add_all(edges)
        session.commit()
        return {"linked": len(edges), "src_id": src_id, "dst_id": dst_id, "relation": relation}


def neighbors(
    seed_ids: list[int],
    depth: int = 1,
    tenant_id: Optional[str] = None,
) -> dict[int, int]:
    """Return {memory_id: min_hop} reachable from the seeds within `depth` hops.

    Uses a recursive CTE over ``memory_edge``. Seeds themselves are excluded.
    """
    if not seed_ids or depth < 1:
        return {}
    tenant_clause = "AND e.tenant_id = :tenant" if tenant_id is not None else ""
    sql = text(
        f"""
        WITH RECURSIVE g(id, depth) AS (
            SELECT e.dst_id, 1
            FROM memory_edge e
            WHERE e.src_id = ANY(:seeds) {tenant_clause}
          UNION ALL
            SELECT e.dst_id, g.depth + 1
            FROM memory_edge e
            JOIN g ON e.src_id = g.id
            WHERE g.depth < :maxdepth {tenant_clause}
        )
        SELECT id, min(depth) AS hop FROM g GROUP BY id
        """
    )
    params = {"seeds": list(seed_ids), "maxdepth": depth}
    if tenant_id is not None:
        params["tenant"] = tenant_id
    with get_session_context(tenant_id) as session:
        rows = session.execute(sql, params).fetchall()
    seeds = set(seed_ids)
    return {r[0]: r[1] for r in rows if r[0] not in seeds}


def neighbor_list(memory_id: int, depth: int = 1, tenant_id: Optional[str] = None) -> list[dict]:
    """Human-friendly neighbor listing for /v1/neighbors."""
    hops = neighbors([memory_id], depth=depth, tenant_id=tenant_id)
    return [{"id": mid, "hop": hop} for mid, hop in sorted(hops.items(), key=lambda x: x[1])]


__all__ = ["link", "neighbors", "neighbor_list"]
