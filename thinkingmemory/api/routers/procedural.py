"""
Procedural Memory API Router.

Handles storage, retrieval, and forgetting of procedural knowledge.
"""

from fastapi import APIRouter, HTTPException

from thinkingmemory.api.schemas import ProcedureStoreRequest, ProcedureUpdateRequest, SimilarityRequest
from thinkingmemory.memory.procedural.crud import (
    store_procedure,
    retrieve_procedures,
    update_procedure_success_rate,
    forget_low_success_procedures,
    retrieve_similar_procedures,
)

router = APIRouter(prefix="/procedural", tags=["procedural"])


@router.post("/store")
async def store_procedure_endpoint(request: ProcedureStoreRequest):
    try:
        return store_procedure(
            agent_id=request.agent_id,
            name=request.name,
            description=request.description,
            steps=request.steps,
            success_rate=request.success_rate,
            version=request.version,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retrieve/{agent_id}")
async def retrieve_procedures_endpoint(agent_id: str, limit: int = 10):
    try:
        return retrieve_procedures(agent_id, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/similar/{agent_id}")
async def retrieve_similar_procedures_endpoint(
    agent_id: str, request: SimilarityRequest
):
    try:
        return retrieve_similar_procedures(
            agent_id=agent_id,
            embedding=request.embedding,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/update/{procedure_id}")
async def update_procedure_endpoint(procedure_id: int, request: ProcedureUpdateRequest):
    try:
        return update_procedure_success_rate(procedure_id, request.success_rate)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/forget/low-success/{agent_id}")
async def forget_low_success_procedures_endpoint(
    agent_id: str, success_threshold: float = 0.5
):
    try:
        deleted_count = forget_low_success_procedures(agent_id, success_threshold)
        return {
            "message": f"Deleted {deleted_count} low-success procedures for agent {agent_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
