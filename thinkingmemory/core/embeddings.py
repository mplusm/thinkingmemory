"""
Shared embedding configuration.

The vector dimension is read once at import time so that all SQLModel table
definitions declare a fixed-size ``Vector(EMBEDDING_DIM)`` column. A fixed size
is required for pgvector to build HNSW / IVFFlat indexes — an unsized ``vector``
column can only be scanned sequentially.

Override with the ``EMBEDDING_DIM`` environment variable (or ``embedding_dim``
in settings) and recreate the tables if you change embedding models.
"""

from typing import Optional

from thinkingmemory.config.settings import get_settings

# Resolved at import time; models declare Vector(EMBEDDING_DIM) columns.
EMBEDDING_DIM: int = get_settings().embedding_dim


def embedding_to_list(embedding) -> Optional[list[float]]:
    """
    Convert a stored embedding to a JSON-serializable list of native floats.

    pgvector returns embeddings as numpy arrays, whose elements are
    ``numpy.float32`` and not directly JSON-serializable. This normalizes both
    numpy arrays and plain lists to native Python floats, and passes ``None``
    through unchanged.
    """
    if embedding is None:
        return None
    if hasattr(embedding, "tolist"):
        return embedding.tolist()
    return [float(x) for x in embedding]


__all__ = ["EMBEDDING_DIM", "embedding_to_list"]
