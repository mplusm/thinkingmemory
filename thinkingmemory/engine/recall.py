"""
The recall engine — the database's query primitive.

`recall(agent_id, intent, ...)` takes an intent string and returns a ranked,
token-budget-packed context window with citations, rather than raw rows:

  1. embed the intent;
  2. generate candidates three ways — vector (cosine), keyword (Postgres
     full-text), and recency;
  3. fuse them with Reciprocal Rank Fusion, weighted by salience;
  4. pack the top results' text into the token budget, with `[n]` citations;
  5. record the recall (bumps recall_count / last_recalled_at / salience) so
     frequently-useful memories rise over time.

It also reports `tokens_saved_vs_dump`: how much smaller the packed context is
than naively dumping every memory — the product's ROI metric.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import func
from sqlmodel import select

from thinkingmemory.core.database import get_session_context
from thinkingmemory.core.timeutils import utcnow
from thinkingmemory.engine import audit
from thinkingmemory.engine.embeddings import embed_one
from thinkingmemory.engine.models import Memory
from thinkingmemory.engine.temporal import temporal_conditions
from thinkingmemory.engine.tokens import count_tokens

# RRF dampening constant — larger = flatter contribution from rank position.
_RRF_K = 60
# Per-retriever weights. Relevance signals (vector, keyword) outrank pure
# recency so that recent-but-irrelevant memories don't crowd out good matches.
_SOURCE_WEIGHTS = {"vector": 1.0, "keyword": 1.0, "recency": 0.5}
# How strongly salience boosts the fused score.
_SALIENCE_WEIGHT = 0.15
# Salience increment applied to memories each time they are recalled.
_RECALL_SALIENCE_BUMP = 0.1


def _base_conditions(agent_id, tenant_id, scopes, mtypes, as_of=None):
    """Shared WHERE conditions: agent, tenant, belief-as-of, scope, type."""
    conds = [Memory.agent_id == agent_id, *temporal_conditions(as_of)]
    if tenant_id is not None:
        conds.append(Memory.tenant_id == tenant_id)
    if scopes:
        conds.append(Memory.scope.in_(scopes))
    if mtypes:
        conds.append(Memory.mtype.in_(mtypes))
    return conds


def _rrf_accumulate(scores: dict, why: dict, ranked_ids: list[int], source: str):
    """Add (weighted) Reciprocal Rank Fusion contributions from one candidate list."""
    weight = _SOURCE_WEIGHTS.get(source, 1.0)
    for rank, mem_id in enumerate(ranked_ids):
        scores[mem_id] = scores.get(mem_id, 0.0) + weight / (_RRF_K + rank)
        why.setdefault(mem_id, set()).add(source)


def recall(
    agent_id: str,
    intent: str,
    *,
    tenant_id: Optional[str] = None,
    scopes: Optional[list[str]] = None,
    mtypes: Optional[list[str]] = None,
    token_budget: int = 4000,
    k: int = 20,
    candidate_limit: int = 50,
    as_of: Optional[datetime] = None,
    track: bool = True,
) -> dict:
    """Retrieve a packed, ranked, cited context window for an intent.

    Pass ``as_of`` to recall against what the agent believed at a past moment
    (bitemporal recall); omit it for the current state.
    """
    qvec = embed_one(intent)

    with get_session_context(tenant_id) as session:
        base = _base_conditions(agent_id, tenant_id, scopes, mtypes, as_of)

        # --- Candidate generation (three retrievers) ---
        vec_ids = list(
            session.exec(
                select(Memory.id)
                .where(*base, Memory.embedding.isnot(None))
                .order_by(Memory.embedding.cosine_distance(qvec))
                .limit(candidate_limit)
            ).all()
        )

        tsv = func.to_tsvector("english", Memory.text)
        tsq = func.plainto_tsquery("english", intent)
        kw_ids = list(
            session.exec(
                select(Memory.id)
                .where(*base, tsv.op("@@")(tsq))
                .order_by(func.ts_rank(tsv, tsq).desc())
                .limit(candidate_limit)
            ).all()
        )

        rec_ids = list(
            session.exec(
                select(Memory.id)
                .where(*base)
                .order_by(Memory.created_at.desc())
                .limit(candidate_limit)
            ).all()
        )

        # --- Fusion (RRF + salience) ---
        scores: dict[int, float] = {}
        why: dict[int, set] = {}
        _rrf_accumulate(scores, why, vec_ids, "vector")
        _rrf_accumulate(scores, why, kw_ids, "keyword")
        _rrf_accumulate(scores, why, rec_ids, "recency")

        if not scores:
            return _empty_result(session, base)

        candidate_ids = list(scores.keys())
        rows = {
            m.id: m
            for m in session.exec(select(Memory).where(Memory.id.in_(candidate_ids))).all()
        }
        for mem_id, mem in rows.items():
            scores[mem_id] *= 1.0 + _SALIENCE_WEIGHT * (mem.salience or 0.0)

        ranked = sorted(candidate_ids, key=lambda i: scores[i], reverse=True)

        # --- Token-budget packing ---
        items, parts, tokens_used = [], [], 0
        for mem_id in ranked:
            if len(items) >= k:
                break
            mem = rows[mem_id]
            piece_tokens = count_tokens(mem.text)
            if tokens_used + piece_tokens > token_budget:
                continue  # skip and try smaller items so the budget is well used
            citation = len(items) + 1
            parts.append(f"[{citation}] {mem.text}")
            tokens_used += piece_tokens
            items.append(
                {
                    "citation": citation,
                    "id": mem.id,
                    "mtype": mem.mtype,
                    "score": round(scores[mem_id], 6),
                    "why": sorted(why[mem_id]),
                    "text": mem.text,
                    "provenance": mem.provenance,
                }
            )

        context = "\n\n".join(parts)

        # --- ROI metric: how much smaller than dumping everything? ---
        dump_tokens = sum(
            count_tokens(t)
            for t in session.exec(select(Memory.text).where(*base)).all()
        )

        # --- Record the recall (lifecycle feedback) ---
        if track and items:
            now = utcnow()
            for it in items:
                mem = rows[it["id"]]
                mem.recall_count += 1
                mem.last_recalled_at = now
                mem.salience = (mem.salience or 0.0) + _RECALL_SALIENCE_BUMP
            session.commit()

        audit.record(
            "recall",
            agent_id,
            tenant_id=tenant_id,
            details={
                "intent": intent,
                "n_items": len(items),
                "tokens_used": tokens_used,
                "tokens_saved_vs_dump": max(0, dump_tokens - tokens_used),
            },
        )

        return {
            "intent": intent,
            "context": context,
            "items": items,
            "tokens_used": tokens_used,
            "tokens_saved_vs_dump": max(0, dump_tokens - tokens_used),
            "dump_tokens": dump_tokens,
            "candidates_considered": len(candidate_ids),
        }


def _empty_result(session, base) -> dict:
    dump_tokens = sum(
        count_tokens(t) for t in session.exec(select(Memory.text).where(*base)).all()
    )
    return {
        "intent": None,
        "context": "",
        "items": [],
        "tokens_used": 0,
        "tokens_saved_vs_dump": 0,
        "dump_tokens": dump_tokens,
        "candidates_considered": 0,
    }


__all__ = ["recall"]
