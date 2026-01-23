"""Semantic memory layer."""

# Import models directly (avoid circular imports with crud)
from thinkingmemory.memory.semantic.models import Fact

__all__ = ["Fact"]
