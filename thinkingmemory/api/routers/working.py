"""
Working Memory API Router.

Handles Redis-backed short-term memory operations. Every endpoint resolves an
optional tenant id from the ``X-Tenant-ID`` header and passes it to the storage
layer, which prefixes Redis keys with the tenant for isolation.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from thinkingmemory.api.dependencies import get_tenant_id
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
async def store_working_memory_endpoint(
    request: WorkingMemoryStoreRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """Store a key-value pair in working memory with TTL."""
    store_working_memory(
        agent_id=request.agent_id,
        key=request.key,
        value=request.value,
        ttl=request.ttl,
        tenant_id=tenant_id,
    )
    return {
        "message": f"Stored key '{request.key}' for agent {request.agent_id}",
        "ttl": request.ttl,
    }


@router.get("/retrieve/{agent_id}/{key}")
async def retrieve_working_memory_endpoint(
    agent_id: str,
    key: str,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """Retrieve a specific key from working memory."""
    value = retrieve_working_memory(agent_id, key, tenant_id=tenant_id)
    if value is None:
        raise HTTPException(
            status_code=404, detail=f"Key '{key}' not found for agent {agent_id}"
        )
    return {"key": key, "value": value, "ttl": get_ttl(agent_id, key, tenant_id=tenant_id)}


@router.get("/retrieve/{agent_id}")
async def retrieve_all_working_memory_endpoint(
    agent_id: str,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """Retrieve all working memory entries for an agent."""
    return get_all_working_memory(agent_id, tenant_id=tenant_id)


@router.get("/keys/{agent_id}")
async def list_working_memory_keys_endpoint(
    agent_id: str,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """List all keys in working memory for an agent."""
    keys = list_working_memory_keys(agent_id, tenant_id=tenant_id)
    return {"agent_id": agent_id, "keys": keys, "count": len(keys)}


@router.patch("/update/{agent_id}/{key}")
async def update_working_memory_endpoint(
    agent_id: str,
    key: str,
    request: WorkingMemoryUpdateRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """Update an existing working memory entry."""
    success = update_working_memory(
        agent_id, key, request.value, request.ttl, tenant_id=tenant_id
    )
    if not success:
        raise HTTPException(
            status_code=404, detail=f"Key '{key}' not found for agent {agent_id}"
        )
    return {"message": f"Updated key '{key}' for agent {agent_id}"}


@router.patch("/extend-ttl/{agent_id}/{key}")
async def extend_ttl_endpoint(
    agent_id: str,
    key: str,
    request: ExtendTTLRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """Extend the TTL of a working memory entry."""
    success = extend_ttl(agent_id, key, request.additional_seconds, tenant_id=tenant_id)
    if not success:
        raise HTTPException(
            status_code=404, detail=f"Key '{key}' not found for agent {agent_id}"
        )
    new_ttl = get_ttl(agent_id, key, tenant_id=tenant_id)
    return {"message": f"Extended TTL for key '{key}'", "new_ttl": new_ttl}


@router.delete("/delete/{agent_id}/{key}")
async def delete_working_memory_endpoint(
    agent_id: str,
    key: str,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """Delete a specific key from working memory."""
    success = delete_working_memory(agent_id, key, tenant_id=tenant_id)
    if not success:
        raise HTTPException(
            status_code=404, detail=f"Key '{key}' not found for agent {agent_id}"
        )
    return {"message": f"Deleted key '{key}' for agent {agent_id}"}


@router.delete("/clear/{agent_id}")
async def clear_working_memory_endpoint(
    agent_id: str,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """Clear all working memory for an agent."""
    deleted_count = clear_working_memory(agent_id, tenant_id=tenant_id)
    return {"message": f"Cleared {deleted_count} keys for agent {agent_id}"}


@router.get("/exists/{agent_id}/{key}")
async def working_memory_exists_endpoint(
    agent_id: str,
    key: str,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """Check if a working memory key exists."""
    exists = working_memory_exists(agent_id, key, tenant_id=tenant_id)
    return {"key": key, "exists": exists}


@router.get("/stats/{agent_id}")
async def working_memory_stats_endpoint(
    agent_id: str,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """Get statistics about an agent's working memory."""
    return get_working_memory_stats(agent_id, tenant_id=tenant_id)
