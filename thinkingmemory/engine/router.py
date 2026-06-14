"""
The /v1 memory database API — the unified surface.

`remember` writes experience (embedded server-side); `recall` is the query
primitive (intent in, packed/cited context out). Tenant scoping reuses the
existing `X-Tenant-ID` dependency. Unhandled errors propagate to the app-wide
handler; only domain 404s are raised here.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from datetime import datetime

from fastapi import Query

from thinkingmemory.api.dependencies import get_tenant_id
from thinkingmemory.engine import store, recall as recall_engine, lifecycle, audit, graph
from thinkingmemory.engine.temporal import timeline
from thinkingmemory.engine.schemas import (
    RememberRequest,
    RememberManyRequest,
    RecallRequest,
    ForgetRequest,
    MaintenanceRequest,
    LinkRequest,
)

router = APIRouter(prefix="/v1", tags=["memory-db"])


@router.post("/remember")
async def remember_endpoint(
    request: RememberRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """Store a memory (embedded server-side)."""
    return store.remember(
        agent_id=request.agent_id,
        content=request.content,
        text=request.text,
        mtype=request.mtype,
        scope=request.scope,
        salience=request.salience,
        confidence=request.confidence,
        decay_rate=request.decay_rate,
        provenance=request.provenance,
        tenant_id=tenant_id,
    )


@router.post("/remember/batch")
async def remember_batch_endpoint(
    request: RememberManyRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """Store many memories with a single batched embedding call."""
    items = [it.model_dump() for it in request.items]
    return store.remember_many(items, tenant_id=tenant_id)


@router.post("/recall")
async def recall_endpoint(
    request: RecallRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """THE primitive: intent in, ranked + packed + cited context out."""
    return recall_engine.recall(
        agent_id=request.agent_id,
        intent=request.intent,
        tenant_id=tenant_id,
        scopes=request.scopes,
        mtypes=request.mtypes,
        token_budget=request.token_budget,
        k=request.k,
        as_of=request.as_of,
        rerank=request.rerank,
        graph_hops=request.graph_hops,
    )


@router.get("/memory/{memory_id}")
async def get_memory_endpoint(
    memory_id: int,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """Fetch a single memory by id."""
    result = store.get(memory_id, tenant_id=tenant_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    return result


@router.get("/trace/{memory_id}")
async def trace_memory_endpoint(
    memory_id: int,
    depth: int = 3,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """Why-do-I-know-this: the recursive provenance tree for a memory."""
    result = store.trace(memory_id, tenant_id=tenant_id, depth=depth)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    return result


@router.get("/timeline/{agent_id}")
async def timeline_endpoint(
    agent_id: str,
    as_of: datetime = Query(..., description="ISO timestamp; what was believed at this moment"),
    mtypes: Optional[list[str]] = Query(default=None),
    limit: int = 200,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """Bitemporal snapshot: what the agent believed at a point in time."""
    return timeline(agent_id, as_of=as_of, tenant_id=tenant_id, mtypes=mtypes, limit=limit)


@router.get("/audit")
async def audit_endpoint(
    agent_id: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 50,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """Recent audit-log entries (newest first)."""
    return audit.query(agent_id=agent_id, tenant_id=tenant_id, action=action, limit=limit)


@router.post("/forget")
async def forget_endpoint(
    request: ForgetRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """Forget a memory (soft by default; hard deletes)."""
    ok = store.forget(request.memory_id, hard=request.hard, tenant_id=tenant_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Memory {request.memory_id} not found")
    return {"forgotten": request.memory_id, "hard": request.hard}


@router.post("/link")
async def link_endpoint(
    request: LinkRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """Create a typed relationship (edge) between two memories."""
    return graph.link(
        src_id=request.src_id,
        dst_id=request.dst_id,
        relation=request.relation,
        agent_id=request.agent_id,
        weight=request.weight,
        bidirectional=request.bidirectional,
        tenant_id=tenant_id,
    )


@router.get("/neighbors/{memory_id}")
async def neighbors_endpoint(
    memory_id: int,
    depth: int = 1,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """List memories reachable from this one within `depth` hops."""
    return {"memory_id": memory_id, "neighbors": graph.neighbor_list(memory_id, depth=depth, tenant_id=tenant_id)}


@router.post("/maintenance/run")
async def maintenance_endpoint(
    request: MaintenanceRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    """Run the lifecycle cycle (decay, consolidate, supersede, forget, prune)
    for an agent. Returns per-pass counts."""
    return lifecycle.run_all(
        agent_id=request.agent_id,
        tenant_id=tenant_id,
        interval_days=request.interval_days,
    )
