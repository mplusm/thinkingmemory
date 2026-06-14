"""
Token counting for budget packing and the token-savings metric.

Uses tiktoken's ``cl100k_base`` when available (offline once installed); falls
back to a ~4-chars-per-token heuristic otherwise. The exact tokenizer of the
consuming model doesn't have to match — this is used for relative budgeting and
to quantify how much context `recall` saves versus dumping everything.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - depends on optional dep
        return None


def count_tokens(text: str) -> int:
    """Approximate token count for a string."""
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


__all__ = ["count_tokens"]
