"""
Episodic Memory API Router.

Handles storage, retrieval, and forgetting of episodic memories.
"""

from fastapi import APIRouter, HTTPException

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
async def store_memory_endpoint(request: MemoryStoreRequest):
    try:
        return store_memory(
            agent_id=request.agent_id,
            content=request.content,
            embedding=request.embedding,
            extra_data=request.extra_data,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retrieve/{agent_id}")
async def retrieve_memory_endpoint(agent_id: str, limit: int = 10):
    try:
        return retrieve_memories(agent_id, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/similar/{agent_id}")
async def retrieve_similar_memories_endpoint(agent_id: str, request: SimilarityRequest):
    try:
        return retrieve_similar_memories(
            agent_id=agent_id,
            embedding=request.embedding,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/forget/old/{agent_id}")
async def forget_old_memories_endpoint(agent_id: str, days: int = 30):
    try:
        deleted_count = forget_old_memories(agent_id, days)
        return {"message": f"Deleted {deleted_count} old memories for agent {agent_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/forget/low-relevance/{agent_id}")
async def forget_low_relevance_memories_endpoint(
    agent_id: str,
    min_access_count: int = 1,
    days_since_access: int = 7,
):
    """
    Delete memories with low relevance based on access patterns.

    Memories are deleted if they have been accessed fewer than min_access_count times
    AND haven't been accessed in the last days_since_access days.
    """
    try:
        deleted_count = forget_low_relevance_memories(
            agent_id, min_access_count, days_since_access
        )
        return {
            "message": f"Deleted {deleted_count} low-relevance memories for agent {agent_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compress/{agent_id}")
async def compress_memories_endpoint(
    agent_id: str,
    similarity_threshold: float = 0.3,
    min_cluster_size: int = 3,
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
    try:
        compressed_count = compress_similar_memories(
            agent_id, similarity_threshold, min_cluster_size
        )
        return {
            "message": f"Created {compressed_count} compressed memory clusters for agent {agent_id}",
            "clusters_created": compressed_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/compress/cleanup/{agent_id}")
async def delete_compressed_originals_endpoint(agent_id: str):
    """
    Delete original memories that have been compressed.

    Call this after compression to free up storage space.
    The compressed summaries will remain.
    """
    try:
        deleted_count = delete_compressed_originals(agent_id)
        return {
            "message": f"Deleted {deleted_count} compressed original memories for agent {agent_id}",
            "deleted_count": deleted_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{agent_id}")
async def memory_stats_endpoint(agent_id: str):
    """Get statistics about an agent's episodic memory."""
    try:
        return get_memory_stats(agent_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
