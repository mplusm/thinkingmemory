#!/usr/bin/env python3
"""
Recall evaluation harness — the make-or-break chart.

Seeds a synthetic corpus for a throwaway agent, then for each query compares:

  - **Hybrid recall** (this engine: vector + keyword + recency, fused, packed)
  - **Naive recency** baseline (most-recent-k, i.e. "just show the latest")

and reports recall@k (did the gold memory surface?) plus how many tokens the
packed context saves versus dumping the whole corpus.

Usage:
    python eval/run_eval.py [--k 5] [--token-budget 200]
"""

import argparse
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from thinkingmemory.core.database import get_engine, init_db
from thinkingmemory.engine import store, recall as recall_engine
from thinkingmemory.engine.tokens import count_tokens
from eval.dataset import CORPUS, QUERIES


def _seed(agent_id: str) -> dict:
    """Insert the corpus; return {key: memory_id}."""
    items = [
        {"agent_id": agent_id, "content": {"text": t}, "text": t, "mtype": m}
        for (_key, m, t) in CORPUS
    ]
    stored = store.remember_many(items)
    return {key: row["id"] for (key, _m, _t), row in zip(CORPUS, stored)}


def _recency_topk(agent_id: str, k: int) -> list[int]:
    """Naive baseline: the k most recently created memory ids."""
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id FROM memory WHERE agent_id = :a "
                "ORDER BY created_at DESC LIMIT :k"
            ),
            {"a": agent_id, "k": k},
        ).fetchall()
    return [r[0] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--token-budget", type=int, default=200)
    ap.add_argument("--rerank", action="store_true", help="enable cross-encoder rerank")
    args = ap.parse_args()

    init_db()
    agent_id = f"eval_{uuid.uuid4().hex[:8]}"
    key_to_id = _seed(agent_id)
    dump_tokens = sum(count_tokens(t) for (_k, _m, t) in CORPUS)

    hybrid_hits = recency_hits = 0
    used_tokens_total = latency_total = 0.0

    print(f"\nCorpus: {len(CORPUS)} memories ({dump_tokens} tokens if dumped whole)")
    print(f"k={args.k}, token_budget={args.token_budget}\n")
    print(f"{'intent':52} {'hybrid':7} {'recency':8} {'tok_used':8}")
    print("-" * 80)

    for intent, gold_keys in QUERIES:
        gold_ids = {key_to_id[k] for k in gold_keys}

        t0 = time.time()
        res = recall_engine.recall(
            agent_id, intent, token_budget=args.token_budget, k=args.k,
            rerank=args.rerank, track=False,
        )
        latency_total += time.time() - t0

        hybrid_ids = {it["id"] for it in res["items"][: args.k]}
        recency_ids = set(_recency_topk(agent_id, args.k))

        h = bool(gold_ids & hybrid_ids)
        r = bool(gold_ids & recency_ids)
        hybrid_hits += h
        recency_hits += r
        used_tokens_total += res["tokens_used"]

        print(f"{intent[:52]:52} {'HIT' if h else 'miss':7} {'HIT' if r else 'miss':8} {res['tokens_used']:8}")

    n = len(QUERIES)
    avg_used = used_tokens_total / n
    print("-" * 80)
    print(f"\nRecall@{args.k}:   hybrid = {hybrid_hits}/{n} ({hybrid_hits/n:.0%})"
          f"   |   naive recency = {recency_hits}/{n} ({recency_hits/n:.0%})")
    print(f"Context size:  packed avg = {avg_used:.0f} tokens   |   full dump = {dump_tokens} tokens"
          f"   ->  {1 - avg_used/dump_tokens:.0%} smaller")
    print(f"Avg recall latency: {latency_total/n*1000:.0f} ms\n")

    # cleanup
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM memory WHERE agent_id = :a"), {"a": agent_id})


if __name__ == "__main__":
    main()
