from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.memory.episodic.crud import (
    store_memory, retrieve_memories, forget_old_memories, forget_low_relevance_memories,
    retrieve_similar_memories, compress_similar_memories, delete_compressed_originals, get_memory_stats
)
from src.memory.semantic.crud import store_fact, retrieve_facts, forget_low_confidence_facts, retrieve_similar_facts
from src.memory.procedural.crud import store_procedure, retrieve_procedures, update_procedure_success_rate, forget_low_success_procedures, retrieve_similar_procedures
from src.memory.working.redis_client import (
    store_working_memory, retrieve_working_memory, delete_working_memory,
    list_working_memory_keys, get_all_working_memory, clear_working_memory,
    update_working_memory, extend_ttl, get_ttl, working_memory_exists, get_working_memory_stats
)

app = FastAPI(
    title="ThinkingMemory API",
    description="Agent-agnostic memory platform",
    version="0.1.0"
)

class MemoryStoreRequest(BaseModel):
    agent_id: str
    content: dict
    embedding: Optional[list[float]] = None
    extra_data: Optional[dict] = None

class FactStoreRequest(BaseModel):
    agent_id: str
    fact: str
    embedding: Optional[list[float]] = None
    confidence: float = 1.0
    source: Optional[str] = None

class ProcedureStoreRequest(BaseModel):
    agent_id: str
    name: str
    description: Optional[str] = None
    steps: list[dict]
    success_rate: float = 1.0
    version: int = 1

class ProcedureUpdateRequest(BaseModel):
    success_rate: float

@app.get("/")
async def root():
    return {"message": "ThinkingMemory API is running!"}

@app.post("/memory/store")
async def store_memory_endpoint(request: MemoryStoreRequest):
    try:
        return store_memory(
            agent_id=request.agent_id,
            content=request.content,
            embedding=request.embedding,
            extra_data=request.extra_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/retrieve/{agent_id}")
async def retrieve_memory_endpoint(agent_id: str, limit: int = 10):
    try:
        return retrieve_memories(agent_id, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/semantic/store")
async def store_fact_endpoint(request: FactStoreRequest):
    try:
        return store_fact(
            agent_id=request.agent_id,
            fact=request.fact,
            embedding=request.embedding,
            confidence=request.confidence,
            source=request.source
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/semantic/retrieve/{agent_id}")
async def retrieve_facts_endpoint(agent_id: str, limit: int = 10):
    try:
        return retrieve_facts(agent_id, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/procedural/store")
async def store_procedure_endpoint(request: ProcedureStoreRequest):
    try:
        return store_procedure(
            agent_id=request.agent_id,
            name=request.name,
            description=request.description,
            steps=request.steps,
            success_rate=request.success_rate,
            version=request.version
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/procedural/retrieve/{agent_id}")
async def retrieve_procedures_endpoint(agent_id: str, limit: int = 10):
    try:
        return retrieve_procedures(agent_id, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/procedural/update/{procedure_id}")
async def update_procedure_endpoint(procedure_id: int, request: ProcedureUpdateRequest):
    try:
        return update_procedure_success_rate(procedure_id, request.success_rate)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/forget/old/{agent_id}")
async def forget_old_memories_endpoint(agent_id: str, days: int = 30):
    try:
        deleted_count = forget_old_memories(agent_id, days)
        return {"message": f"Deleted {deleted_count} old memories for agent {agent_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/forget/low-relevance/{agent_id}")
async def forget_low_relevance_memories_endpoint(
    agent_id: str,
    min_access_count: int = 1,
    days_since_access: int = 7
):
    """
    Delete memories with low relevance based on access patterns.

    Memories are deleted if they have been accessed fewer than min_access_count times
    AND haven't been accessed in the last days_since_access days.
    """
    try:
        deleted_count = forget_low_relevance_memories(agent_id, min_access_count, days_since_access)
        return {"message": f"Deleted {deleted_count} low-relevance memories for agent {agent_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/semantic/forget/low-confidence/{agent_id}")
async def forget_low_confidence_facts_endpoint(agent_id: str, confidence_threshold: float = 0.5):
    try:
        deleted_count = forget_low_confidence_facts(agent_id, confidence_threshold)
        return {"message": f"Deleted {deleted_count} low-confidence facts for agent {agent_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/procedural/forget/low-success/{agent_id}")
async def forget_low_success_procedures_endpoint(agent_id: str, success_threshold: float = 0.5):
    try:
        deleted_count = forget_low_success_procedures(agent_id, success_threshold)
        return {"message": f"Deleted {deleted_count} low-success procedures for agent {agent_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SimilarityRequest(BaseModel):
    embedding: list[float]
    limit: int = 10
    similarity_threshold: float = 0.5

@app.post("/memory/similar/{agent_id}")
async def retrieve_similar_memories_endpoint(agent_id: str, request: SimilarityRequest):
    try:
        return retrieve_similar_memories(
            agent_id=agent_id,
            embedding=request.embedding,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/semantic/similar/{agent_id}")
async def retrieve_similar_facts_endpoint(agent_id: str, request: SimilarityRequest):
    try:
        return retrieve_similar_facts(
            agent_id=agent_id,
            embedding=request.embedding,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/procedural/similar/{agent_id}")
async def retrieve_similar_procedures_endpoint(agent_id: str, request: SimilarityRequest):
    try:
        return retrieve_similar_procedures(
            agent_id=agent_id,
            embedding=request.embedding,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Working Memory Endpoints (Redis-backed short-term memory)
# =============================================================================

class WorkingMemoryStoreRequest(BaseModel):
    agent_id: str
    key: str
    value: dict
    ttl: int = 300  # Default 5 minutes

class WorkingMemoryUpdateRequest(BaseModel):
    value: dict
    ttl: Optional[int] = None  # If None, preserves existing TTL

class ExtendTTLRequest(BaseModel):
    additional_seconds: int

@app.post("/working/store")
async def store_working_memory_endpoint(request: WorkingMemoryStoreRequest):
    """Store a key-value pair in working memory with TTL."""
    try:
        store_working_memory(
            agent_id=request.agent_id,
            key=request.key,
            value=request.value,
            ttl=request.ttl
        )
        return {
            "message": f"Stored key '{request.key}' for agent {request.agent_id}",
            "ttl": request.ttl
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/working/retrieve/{agent_id}/{key}")
async def retrieve_working_memory_endpoint(agent_id: str, key: str):
    """Retrieve a specific key from working memory."""
    try:
        value = retrieve_working_memory(agent_id, key)
        if value is None:
            raise HTTPException(status_code=404, detail=f"Key '{key}' not found for agent {agent_id}")
        return {"key": key, "value": value, "ttl": get_ttl(agent_id, key)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/working/retrieve/{agent_id}")
async def retrieve_all_working_memory_endpoint(agent_id: str):
    """Retrieve all working memory entries for an agent."""
    try:
        return get_all_working_memory(agent_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/working/keys/{agent_id}")
async def list_working_memory_keys_endpoint(agent_id: str):
    """List all keys in working memory for an agent."""
    try:
        keys = list_working_memory_keys(agent_id)
        return {"agent_id": agent_id, "keys": keys, "count": len(keys)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/working/update/{agent_id}/{key}")
async def update_working_memory_endpoint(agent_id: str, key: str, request: WorkingMemoryUpdateRequest):
    """Update an existing working memory entry."""
    try:
        success = update_working_memory(agent_id, key, request.value, request.ttl)
        if not success:
            raise HTTPException(status_code=404, detail=f"Key '{key}' not found for agent {agent_id}")
        return {"message": f"Updated key '{key}' for agent {agent_id}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/working/extend-ttl/{agent_id}/{key}")
async def extend_ttl_endpoint(agent_id: str, key: str, request: ExtendTTLRequest):
    """Extend the TTL of a working memory entry."""
    try:
        success = extend_ttl(agent_id, key, request.additional_seconds)
        if not success:
            raise HTTPException(status_code=404, detail=f"Key '{key}' not found for agent {agent_id}")
        new_ttl = get_ttl(agent_id, key)
        return {"message": f"Extended TTL for key '{key}'", "new_ttl": new_ttl}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/working/delete/{agent_id}/{key}")
async def delete_working_memory_endpoint(agent_id: str, key: str):
    """Delete a specific key from working memory."""
    try:
        success = delete_working_memory(agent_id, key)
        if not success:
            raise HTTPException(status_code=404, detail=f"Key '{key}' not found for agent {agent_id}")
        return {"message": f"Deleted key '{key}' for agent {agent_id}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/working/clear/{agent_id}")
async def clear_working_memory_endpoint(agent_id: str):
    """Clear all working memory for an agent."""
    try:
        deleted_count = clear_working_memory(agent_id)
        return {"message": f"Cleared {deleted_count} keys for agent {agent_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/working/exists/{agent_id}/{key}")
async def working_memory_exists_endpoint(agent_id: str, key: str):
    """Check if a working memory key exists."""
    try:
        exists = working_memory_exists(agent_id, key)
        return {"key": key, "exists": exists}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/working/stats/{agent_id}")
async def working_memory_stats_endpoint(agent_id: str):
    """Get statistics about an agent's working memory."""
    try:
        return get_working_memory_stats(agent_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Compression Endpoints (Episodic Memory Consolidation)
# =============================================================================

@app.post("/memory/compress/{agent_id}")
async def compress_memories_endpoint(
    agent_id: str,
    similarity_threshold: float = 0.3,
    min_cluster_size: int = 3
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
        compressed_count = compress_similar_memories(agent_id, similarity_threshold, min_cluster_size)
        return {
            "message": f"Created {compressed_count} compressed memory clusters for agent {agent_id}",
            "clusters_created": compressed_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/compress/cleanup/{agent_id}")
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
            "deleted_count": deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/stats/{agent_id}")
async def memory_stats_endpoint(agent_id: str):
    """Get statistics about an agent's episodic memory."""
    try:
        return get_memory_stats(agent_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))