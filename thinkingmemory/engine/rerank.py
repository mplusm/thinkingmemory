"""
Cross-encoder reranking of recall candidates.

Hybrid fusion (vector + keyword + recency) is cheap and high-recall but coarse.
A cross-encoder scores each (intent, candidate) pair jointly for far better
precision on the top results. It runs locally on CPU via fastembed and is
optional (off by default) because it adds latency and a model download.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from thinkingmemory.config.settings import get_settings


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        """Return a relevance score per doc (higher = more relevant)."""
        ...


class LocalReranker:
    """Local CPU cross-encoder via fastembed (default ms-marco-MiniLM-L-6-v2)."""

    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # lazy import

        self.name = f"local:{model_name}"
        self._model = TextCrossEncoder(model_name)

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        return [float(s) for s in self._model.rerank(query, docs)]


@lru_cache
def get_reranker() -> Reranker:
    """Return the configured reranker (cached for the process lifetime)."""
    return LocalReranker(model_name=get_settings().rerank_model)


__all__ = ["Reranker", "LocalReranker", "get_reranker"]
