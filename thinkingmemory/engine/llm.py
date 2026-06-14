"""
Pluggable LLM for fact extraction, consolidation summaries, and contradiction
(NLI) judgments.

There is no required LLM dependency: the default ``NoLLM`` provider uses offline
heuristics (sentence splitting, embedding similarity + negation polarity) so the
lifecycle features work with no API key. Set ``llm_provider`` to ``openai`` or
``anthropic`` (with a key) to upgrade extraction/summarization/NLI quality.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol

import numpy as np

from thinkingmemory.config.settings import get_settings

_NEGATIONS = {"not", "no", "never", "n't", "without", "cannot", "can't", "won't", "isn't", "aren't"}
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class LLM(Protocol):
    name: str

    def summarize(self, texts: list[str]) -> str: ...
    def extract_facts(self, text: str) -> list[str]: ...
    def is_contradiction(self, a: str, b: str) -> bool: ...


def _negation_set(text: str) -> set:
    toks = re.findall(r"[a-z']+", text.lower())
    return {t for t in toks if t in _NEGATIONS}


class NoLLM:
    """Offline heuristics — no API key required."""

    name = "none"

    def summarize(self, texts: list[str]) -> str:
        seen, uniq = set(), []
        for t in texts:
            k = t.strip().lower()
            if k and k not in seen:
                seen.add(k)
                uniq.append(t.strip())
        return f"Summary of {len(texts)} related memories: " + "; ".join(uniq[:5])

    def extract_facts(self, text: str) -> list[str]:
        """Keep declarative, non-question sentences of reasonable length."""
        facts = []
        for s in _SENTENCE_SPLIT.split(text.strip()):
            s = s.strip()
            if 3 <= len(s.split()) <= 40 and not s.endswith("?"):
                facts.append(s.rstrip("."))
        return facts

    def is_contradiction(self, a: str, b: str) -> bool:
        """Heuristic: very similar wording but divergent negation polarity."""
        from thinkingmemory.engine.embeddings import get_embedder

        va, vb = (np.asarray(v, dtype=float) for v in get_embedder().embed([a, b]))
        cos = float(va @ vb / ((np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0))
        if cos < 0.6:  # unrelated -> not a contradiction
            return False
        na, nb = _negation_set(a), _negation_set(b)
        return (bool(na) != bool(nb)) or (na != nb and (na or nb))


class _APILLM:
    """Shared OpenAI/Anthropic chat-completion wrapper."""

    def __init__(self, name: str, complete):
        self.name = name
        self._complete = complete  # callable(prompt:str)->str

    def summarize(self, texts: list[str]) -> str:
        joined = "\n- ".join(texts)
        return self._complete(
            "Summarize these related memories into one concise factual statement:\n- " + joined
        ).strip()

    def extract_facts(self, text: str) -> list[str]:
        out = self._complete(
            "Extract the durable, atomic facts from this text, one per line, no numbering:\n" + text
        )
        return [ln.strip("-• ").strip() for ln in out.splitlines() if ln.strip()]

    def is_contradiction(self, a: str, b: str) -> bool:
        out = self._complete(
            f"Do these two statements contradict each other? Answer yes or no.\nA: {a}\nB: {b}"
        )
        return out.strip().lower().startswith("y")


def _openai_llm(model: str) -> _APILLM:
    from openai import OpenAI

    client = OpenAI(api_key=get_settings().openai_api_key or None)
    mdl = model or "gpt-4o-mini"

    def complete(prompt: str) -> str:
        r = client.chat.completions.create(
            model=mdl, messages=[{"role": "user", "content": prompt}], temperature=0
        )
        return r.choices[0].message.content or ""

    return _APILLM(f"openai:{mdl}", complete)


def _anthropic_llm(model: str) -> _APILLM:
    import anthropic

    client = anthropic.Anthropic(api_key=get_settings().anthropic_api_key or None)
    mdl = model or "claude-haiku-4-5-20251001"

    def complete(prompt: str) -> str:
        r = client.messages.create(
            model=mdl, max_tokens=1024, messages=[{"role": "user", "content": prompt}]
        )
        return "".join(block.text for block in r.content if getattr(block, "type", "") == "text")

    return _APILLM(f"anthropic:{mdl}", complete)


@lru_cache
def get_llm() -> LLM:
    """Return the configured LLM provider (cached)."""
    s = get_settings()
    provider = s.llm_provider.lower()
    if provider == "openai":
        return _openai_llm(s.llm_model)
    if provider == "anthropic":
        return _anthropic_llm(s.llm_model)
    return NoLLM()


__all__ = ["LLM", "NoLLM", "get_llm"]
