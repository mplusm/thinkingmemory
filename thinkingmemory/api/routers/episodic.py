"""
Episodic Memory API Router.

Handles storage, retrieval, and forgetting of episodic memories.

Every endpoint resolves an optional tenant id from the ``X-Tenant-ID`` header
and passes it to the CRUD layer for tenant isolation. Unhandled exceptions
propagate to the application-wide error handler (see ``api.errors``); only
domain-level 404s are raised explicitly here.
"""

from typing import Optional

from fastapi import APIRouter, Depends

from thinkingmemory.api.dependencies import get_tenant_id
from thinkingmemory.api.schemas import MemoryStoreRequest, SimilarityRequest
from thinkingmemory.memory.episodic.crud import (
    store_memory,
    retrieve_memories,
    forget_old_memories,
    forget_low_relevance_memories,
    retrieve_similar_memories,
    compress_similar_memories,
    delete_compressed_originals,
    get_memory_stats,
)

router = APIRouter(prefix="/memory", tags=["episodic"])


@router.post("/store")
async def store_memory_endpoint(
    request: MemoryStoreRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return store_memory(
        agent_id=request.agent_id,
        content=request.content,
        embedding=request.embedding,
        extra_data=request.extra_data,
        memory_type=request.memory_type,
        tenant_id=tenant_id,
    )


@router.get("/retrieve/{agent_id}")
async def retrieve_memory_endpoint(
    agent_id: str,
    limit: int = 10,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return retrieve_memories(agent_id, limit, tenant_id=tenant_id)


@router.post("/similar/{agent_id}")
async def retrieve_similar_memories_endpoint(
    agent_id: str,
    request: SimilarityRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return retrieve_similar_memories(
        agent_id=agent_id,
        embedding=request.embedding,
        limit=request.limit,
        similarity_threshold=request.similarity_threshold,
        tenant_id=tenant_id,
    )


@router.delete("/forget/old/{agent_id}")
async def forget_old_memories_endpoint(
    agent_id: str,
    days: int = 30,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    deleted_count = forget_old_memories(agent_id, days, tenant_id=tenant_id)
    return {"message": f"Deleted {deleted_count} old memories for agent {agent_id}"}


@router.delete("/forget/low-relevance/{agent_id}")
async def forget_low_relevance_memories_endpoint(
    agent_id: str,
    min_access_count: int = 1,
    days_since_access: int = 7,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """
    Delete memories with low relevance based on access patterns.

    Memories are deleted if they have been accessed fewer than min_access_count times
    AND haven't been accessed in the last days_since_access days.
    """
    deleted_count = forget_low_relevance_memories(
        agent_id, min_access_count, days_since_access, tenant_id=tenant_id
    )
    return {
        "message": f"Deleted {deleted_count} low-relevance memories for agent {agent_id}"
    }


@router.post("/compress/{agent_id}")
async def compress_memories_endpoint(
    agent_id: str,
    similarity_threshold: float = 0.3,
    min_cluster_size: int = 3,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """
    Compress similar episodic memories into consolidated entries.

    This groups similar memories based on embedding similarity and creates
    compressed summary entries. Original memories are marked as compressed
    but retained for audit purposes.

    Args:
        similarity_threshold: L2 distance threshold for grouping (lower = more similar)
        min_cluster_size: Minimum number of memories to form a compression cluster
    """
    compressed_count = compress_similar_memories(
        agent_id, similarity_threshold, min_cluster_size, tenant_id=tenant_id
    )
    return {
        "message": f"Created {compressed_count} compressed memory clusters for agent {agent_id}",
        "clusters_created": compressed_count,
    }


@router.delete("/compress/cleanup/{agent_id}")
async def delete_compressed_originals_endpoint(
    agent_id: str,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """
    Delete original memories that have been compressed.

    Call this after compression to free up storage space.
    The compressed summaries will remain.
    """
    deleted_count = delete_compressed_originals(agent_id, tenant_id=tenant_id)
    return {
        "message": f"Deleted {deleted_count} compressed original memories for agent {agent_id}",
        "deleted_count": deleted_count,
    }


@router.get("/stats/{agent_id}")
async def memory_stats_endpoint(
    agent_id: str,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """Get statistics about an agent's episodic memory."""
    return get_memory_stats(agent_id, tenant_id=tenant_id)
