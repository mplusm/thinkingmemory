"""
The lifecycle engine — the moat.

Background maintenance that makes recall improve over time, the part a
hand-rolled pgvector setup can't easily match:

- **decay**: salience fades with time (``salience *= e^(-decay_rate·Δt)``);
  recall counteracts it, so useful memories persist and stale ones sink.
- **consolidate** ("sleep"): cluster similar episodic memories and write a
  single semantic summary linked to its sources via provenance.
- **forget**: soft-close low-salience, long-idle memories (recoverable), then
  hard-prune ones that have been closed for a while.
- **supersede** (contradiction-lite): when two semantic memories are near
  duplicates, keep the newer and close the older, linking provenance.

Each function is a callable "pass" usable from a scheduler, the
``/v1/maintenance/run`` endpoint, or ``scripts/run_lifecycle.py``. LLM-based
fact extraction and NLI contradiction detection are deliberate follow-ups; the
passes here need no LLM.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

import numpy as np
from sqlalchemy import text
from sqlmodel import select

from thinkingmemory.core.database import get_engine, get_session_context
from thinkingmemory.core.timeutils import utcnow
from thinkingmemory.engine.embeddings import get_embedder
from thinkingmemory.engine.models import Memory

# Defaults (overridable per call)
FORGET_SALIENCE_THRESHOLD = 0.2
FORGET_IDLE_DAYS = 30
PRUNE_AFTER_DAYS = 30
CONSOLIDATE_DISTANCE = 0.25      # cosine distance; lower = more similar
CONSOLIDATE_MIN_CLUSTER = 3
SUPERSEDE_DISTANCE = 0.05        # near-duplicate cosine distance


def _scope(conds, agent_id, tenant_id):
    if agent_id is not None:
        conds.append(Memory.agent_id == agent_id)
    if tenant_id is not None:
        conds.append(Memory.tenant_id == tenant_id)
    return conds


def apply_decay(
    interval_days: float = 1.0,
    agent_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> int:
    """Decay salience for one maintenance interval. Returns rows affected.

    Run on a fixed cadence (e.g. daily with interval_days=1). Only currently
    valid rows with a positive decay_rate are touched.
    """
    where = ["valid_to IS NULL", "decay_rate > 0"]
    params = {"days": interval_days}
    if agent_id is not None:
        where.append("agent_id = :agent_id"); params["agent_id"] = agent_id
    if tenant_id is not None:
        where.append("tenant_id = :tenant_id"); params["tenant_id"] = tenant_id
    sql = (
        "UPDATE memory SET salience = salience * exp(-decay_rate * :days) "
        "WHERE " + " AND ".join(where)
    )
    with get_engine().begin() as conn:
        return conn.execute(text(sql), params).rowcount


def forget_decayed(
    agent_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    salience_threshold: float = FORGET_SALIENCE_THRESHOLD,
    idle_days: int = FORGET_IDLE_DAYS,
) -> int:
    """Soft-close low-salience memories that haven't been recalled recently."""
    cutoff = utcnow() - timedelta(days=idle_days)
    where = [
        "valid_to IS NULL",
        "salience < :thr",
        "(last_recalled_at IS NULL OR last_recalled_at < :cutoff)",
        "created_at < :cutoff",
    ]
    params = {"thr": salience_threshold, "cutoff": cutoff, "now": utcnow()}
    if agent_id is not None:
        where.append("agent_id = :agent_id"); params["agent_id"] = agent_id
    if tenant_id is not None:
        where.append("tenant_id = :tenant_id"); params["tenant_id"] = tenant_id
    sql = (
        "UPDATE memory SET valid_to = :now, superseded_at = :now "
        "WHERE " + " AND ".join(where)
    )
    with get_engine().begin() as conn:
        return conn.execute(text(sql), params).rowcount


def prune_superseded(
    agent_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    older_than_days: int = PRUNE_AFTER_DAYS,
) -> int:
    """Hard-delete rows that have been closed (superseded/forgotten) for a while."""
    cutoff = utcnow() - timedelta(days=older_than_days)
    where = ["superseded_at IS NOT NULL", "superseded_at < :cutoff"]
    params = {"cutoff": cutoff}
    if agent_id is not None:
        where.append("agent_id = :agent_id"); params["agent_id"] = agent_id
    if tenant_id is not None:
        where.append("tenant_id = :tenant_id"); params["tenant_id"] = tenant_id
    sql = "DELETE FROM memory WHERE " + " AND ".join(where)
    with get_engine().begin() as conn:
        return conn.execute(text(sql), params).rowcount


def _already_consolidated_ids(session, agent_id, tenant_id) -> set:
    """Ids that are already a source of some consolidation summary.

    ``provenance`` is a JSON (not JSONB) column, so we filter in Python rather
    than with JSONB operators.
    """
    conds = [Memory.mtype == "semantic", Memory.valid_to.is_(None)]
    _scope(conds, agent_id, tenant_id)
    used = set()
    for m in session.exec(select(Memory).where(*conds)).all():
        prov = m.provenance or {}
        if prov.get("source") == "consolidation":
            used.update(prov.get("derived_from", []))
    return used


def consolidate(
    agent_id: str,
    tenant_id: Optional[str] = None,
    distance: float = CONSOLIDATE_DISTANCE,
    min_cluster: int = CONSOLIDATE_MIN_CLUSTER,
) -> int:
    """Cluster similar episodic memories into semantic summaries ("sleep").

    Returns the number of summary memories created. Each summary's provenance
    records its source ids; sources are down-weighted (their content now lives in
    the summary) but kept for traceability.
    """
    embedder = get_embedder()
    with get_session_context() as session:
        conds = [
            Memory.mtype == "episodic",
            Memory.valid_to.is_(None),
            Memory.embedding.isnot(None),
        ]
        _scope(conds, agent_id, tenant_id)
        rows = session.exec(select(Memory).where(*conds)).all()

        used = _already_consolidated_ids(session, agent_id, tenant_id)
        rows = [r for r in rows if r.id not in used]
        if len(rows) < min_cluster:
            return 0

        vecs = {r.id: np.asarray(list(r.embedding), dtype=float) for r in rows}
        by_id = {r.id: r for r in rows}
        claimed: set = set()
        created = 0

        for seed in rows:
            if seed.id in claimed:
                continue
            sv = vecs[seed.id]
            sv_n = sv / (np.linalg.norm(sv) or 1.0)
            cluster = [seed.id]
            for other in rows:
                if other.id == seed.id or other.id in claimed:
                    continue
                ov = vecs[other.id]
                cos_dist = 1.0 - float(sv_n @ (ov / (np.linalg.norm(ov) or 1.0)))
                if cos_dist <= distance:
                    cluster.append(other.id)
            if len(cluster) < min_cluster:
                continue

            members = [by_id[i] for i in cluster]
            summary_text = _summarize([m.text for m in members])
            embedding = embedder.embed([summary_text])[0]
            avg_salience = float(np.mean([m.salience for m in members]))

            summary = Memory(
                tenant_id=seed.tenant_id,
                agent_id=seed.agent_id,
                mtype="semantic",
                scope=seed.scope,
                content={"summary": summary_text, "source_count": len(members)},
                text=summary_text,
                embedding=embedding,
                salience=max(1.0, avg_salience),
                confidence=1.0,
                provenance={"source": "consolidation", "derived_from": cluster},
            )
            session.add(summary)
            for m in members:
                m.salience *= 0.5  # represented by the summary now
                claimed.add(m.id)
            created += 1

        session.commit()
        return created


def _summarize(texts: list[str]) -> str:
    """Extractive summary (no LLM): dedupe and join the cluster's memories."""
    seen, unique = set(), []
    for t in texts:
        key = t.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(t.strip())
    head = "; ".join(unique[:5])
    return f"Summary of {len(texts)} related memories: {head}"


def resolve_duplicates(
    agent_id: str,
    tenant_id: Optional[str] = None,
    distance: float = SUPERSEDE_DISTANCE,
) -> int:
    """Supersede near-duplicate semantic memories, keeping the newest.

    Contradiction-lite: collapses repeated/updated facts. The older row is
    soft-closed and linked via provenance to the survivor. Returns the number
    superseded. (Full NLI-based contradiction detection is a follow-up.)
    """
    with get_session_context() as session:
        conds = [Memory.mtype == "semantic", Memory.valid_to.is_(None), Memory.embedding.isnot(None)]
        _scope(conds, agent_id, tenant_id)
        rows = session.exec(select(Memory).where(*conds).order_by(Memory.created_at.desc())).all()
        if len(rows) < 2:
            return 0

        vecs = {r.id: np.asarray(list(r.embedding), dtype=float) for r in rows}
        norms = {i: (v / (np.linalg.norm(v) or 1.0)) for i, v in vecs.items()}
        superseded = 0
        survivors: list[int] = []  # newest-first kept rows
        now = utcnow()

        for r in rows:  # newest first
            dup_of = None
            for keep in survivors:
                if 1.0 - float(norms[r.id] @ norms[keep]) <= distance:
                    dup_of = keep
                    break
            if dup_of is not None:
                prov = dict(r.provenance or {})
                prov["superseded_by"] = dup_of
                r.provenance = prov
                r.valid_to = now
                r.superseded_at = now
                superseded += 1
            else:
                survivors.append(r.id)

        session.commit()
        return superseded


def run_all(
    agent_id: str,
    tenant_id: Optional[str] = None,
    interval_days: float = 1.0,
) -> dict:
    """Run the full maintenance cycle for an agent and return per-pass counts."""
    from thinkingmemory.engine import audit

    counts = {
        "decayed": apply_decay(interval_days, agent_id, tenant_id),
        "consolidated": consolidate(agent_id, tenant_id),
        "superseded": resolve_duplicates(agent_id, tenant_id),
        "forgotten": forget_decayed(agent_id, tenant_id),
        "pruned": prune_superseded(agent_id, tenant_id),
    }
    audit.record("maintenance", agent_id, tenant_id=tenant_id, details=counts)
    return counts


__all__ = [
    "apply_decay",
    "forget_decayed",
    "prune_superseded",
    "consolidate",
    "resolve_duplicates",
    "run_all",
]
