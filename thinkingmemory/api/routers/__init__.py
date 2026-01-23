"""API Routers for ThinkingMemory."""

from thinkingmemory.api.routers.episodic import router as episodic_router
from thinkingmemory.api.routers.semantic import router as semantic_router
from thinkingmemory.api.routers.procedural import router as procedural_router
from thinkingmemory.api.routers.working import router as working_router

__all__ = [
    "episodic_router",
    "semantic_router",
    "procedural_router",
    "working_router",
]
