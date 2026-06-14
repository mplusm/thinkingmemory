"""Tests for the pluggable embedder and token counter."""

from thinkingmemory.engine.embeddings import get_embedder, embed_one
from thinkingmemory.engine.tokens import count_tokens
from thinkingmemory.core.embeddings import EMBEDDING_DIM


def test_embedder_dim_matches_setting():
    emb = get_embedder()
    assert emb.dim == EMBEDDING_DIM
    vec = embed_one("a quick test sentence")
    assert len(vec) == EMBEDDING_DIM
    assert all(isinstance(x, float) for x in vec[:5])


def test_batch_embed_shape():
    vecs = get_embedder().embed(["one", "two", "three"])
    assert len(vecs) == 3
    assert all(len(v) == EMBEDDING_DIM for v in vecs)


def test_count_tokens():
    assert count_tokens("") == 0
    assert count_tokens("hello world") >= 1
