"""
Working Memory API Router.

Handles Redis-backed short-term memory operations.
"""

from fastapi import APIRouter, HTTPException

from thinkingmemory.api.schemas import (
    WorkingMemoryStoreRequest,
    WorkingMemoryUpdateRequest,
    ExtendTTLRequest,
)
from thinkingmemory.memory.working.redis_client import (
    store_working_memory,
    retrieve_working_memory,
    delete_working_memory,
    list_working_memory_keys,
    get_all_working_memory,
    clear_working_memory,
    update_working_memory,
    extend_ttl,
    get_ttl,
    working_memory_exists,
    get_working_memory_stats,
)

router = APIRouter(prefix="/working", tags=["working"])


@router.post("/store")
async def store_working_memory_endpoint(request: WorkingMemoryStoreRequest):
    """Store a key-value pair in working memory with TTL."""
    try:
        store_working_memory(
            agent_id=request.agent_id,
            key=request.key,
            value=request.value,
            ttl=request.ttl,
        )
        return {
            "message": f"Stored key '{request.key}' for agent {request.agent_id}",
            "ttl": request.ttl,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retrieve/{agent_id}/{key}")
async def retrieve_working_memory_endpoint(agent_id: str, key: str):
    """Retrieve a specific key from working memory."""
    try:
        value = retrieve_working_memory(agent_id, key)
        if value is None:
            raise HTTPException(
                status_code=404, detail=f"Key '{key}' not found for agent {agent_id}"
            )
        return {"key": key, "value": value, "ttl": get_ttl(agent_id, key)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retrieve/{agent_id}")
async def retrieve_all_working_memory_endpoint(agent_id: str):
    """Retrieve all working memory entries for an agent."""
    try:
        return get_all_working_memory(agent_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/keys/{agent_id}")
async def list_working_memory_keys_endpoint(agent_id: str):
    """List all keys in working memory for an agent."""
    try:
        keys = list_working_memory_keys(agent_id)
        return {"agent_id": agent_id, "keys": keys, "count": len(keys)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/update/{agent_id}/{key}")
async def update_working_memory_endpoint(
    agent_id: str, key: str, request: WorkingMemoryUpdateRequest
):
    """Update an existing working memory entry."""
    try:
        success = update_working_memory(agent_id, key, request.value, request.ttl)
        if not success:
            raise HTTPException(
                status_code=404, detail=f"Key '{key}' not found for agent {agent_id}"
            )
        return {"message": f"Updated key '{key}' for agent {agent_id}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/extend-ttl/{agent_id}/{key}")
async def extend_ttl_endpoint(agent_id: str, key: str, request: ExtendTTLRequest):
    """Extend the TTL of a working memory entry."""
    try:
        success = extend_ttl(agent_id, key, request.additional_seconds)
        if not success:
            raise HTTPException(
                status_code=404, detail=f"Key '{key}' not found for agent {agent_id}"
            )
        new_ttl = get_ttl(agent_id, key)
        return {"message": f"Extended TTL for key '{key}'", "new_ttl": new_ttl}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/{agent_id}/{key}")
async def delete_working_memory_endpoint(agent_id: str, key: str):
    """Delete a specific key from working memory."""
    try:
        success = delete_working_memory(agent_id, key)
        if not success:
            raise HTTPException(
                status_code=404, detail=f"Key '{key}' not found for agent {agent_id}"
            )
        return {"message": f"Deleted key '{key}' for agent {agent_id}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear/{agent_id}")
async def clear_working_memory_endpoint(agent_id: str):
    """Clear all working memory for an agent."""
    try:
        deleted_count = clear_working_memory(agent_id)
        return {"message": f"Cleared {deleted_count} keys for agent {agent_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exists/{agent_id}/{key}")
async def working_memory_exists_endpoint(agent_id: str, key: str):
    """Check if a working memory key exists."""
    try:
        exists = working_memory_exists(agent_id, key)
        return {"key": key, "exists": exists}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{agent_id}")
async def working_memory_stats_endpoint(agent_id: str):
    """Get statistics about an agent's working memory."""
    try:
        return get_working_memory_stats(agent_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
