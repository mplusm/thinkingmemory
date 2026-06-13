"""
Pluggable embedding providers.

The platform generates embeddings server-side (unlike the legacy API where the
caller supplied vectors). The default is a local CPU model via ``fastembed`` so
the product works offline with no API key or per-call cost; OpenAI is an opt-in
alternative. Provider and dimension are driven by settings — see
``config.settings`` and ``core.embeddings.EMBEDDING_DIM``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from thinkingmemory.config.settings import get_settings


class Embedder(Protocol):
    """Minimal embedding interface used by the store and recall engine."""

    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding (list[float]) per input text."""
        ...


class LocalEmbedder:
    """Local CPU embeddings via fastembed (default: BAAI/bge-small-en-v1.5)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dim: int = 384):
        from fastembed import TextEmbedding  # lazy import; heavy dependency

        self.name = f"local:{model_name}"
        self.dim = dim
        self._model = TextEmbedding(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # fastembed returns numpy arrays; normalize to native python floats.
        return [vec.tolist() for vec in self._model.embed(texts)]


class OpenAIEmbedder:
    """OpenAI embeddings (opt-in; requires OPENAI_API_KEY)."""

    def __init__(self, model_name: str = "text-embedding-3-small", dim: int = 1536):
        from openai import OpenAI  # lazy import

        settings = get_settings()
        self.name = f"openai:{model_name}"
        self.dim = dim
        self._model_name = model_name
        self._client = OpenAI(api_key=settings.openai_api_key or None)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self._model_name, input=texts)
        return [d.embedding for d in resp.data]


@lru_cache
def get_embedder() -> Embedder:
    """Return the configured embedder (cached for the process lifetime)."""
    settings = get_settings()
    provider = settings.embedding_provider.lower()
    if provider == "openai":
        return OpenAIEmbedder(model_name=settings.embedding_model, dim=settings.embedding_dim)
    return LocalEmbedder(model_name=settings.embedding_model, dim=settings.embedding_dim)


def embed_one(text: str) -> list[float]:
    """Embed a single string."""
    return get_embedder().embed([text])[0]


__all__ = ["Embedder", "LocalEmbedder", "OpenAIEmbedder", "get_embedder", "embed_one"]
