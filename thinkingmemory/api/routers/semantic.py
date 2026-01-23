"""
Semantic Memory API Router.

Handles storage, retrieval, and forgetting of semantic facts.
"""

from fastapi import APIRouter, HTTPException

from thinkingmemory.api.schemas import FactStoreRequest, SimilarityRequest
from thinkingmemory.memory.semantic.crud import (
    store_fact,
    retrieve_facts,
    forget_low_confidence_facts,
    retrieve_similar_facts,
)

router = APIRouter(prefix="/semantic", tags=["semantic"])


@router.post("/store")
async def store_fact_endpoint(request: FactStoreRequest):
    try:
        return store_fact(
            agent_id=request.agent_id,
            fact=request.fact,
            embedding=request.embedding,
            confidence=request.confidence,
            source=request.source,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retrieve/{agent_id}")
async def retrieve_facts_endpoint(agent_id: str, limit: int = 10):
    try:
        return retrieve_facts(agent_id, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/similar/{agent_id}")
async def retrieve_similar_facts_endpoint(agent_id: str, request: SimilarityRequest):
    try:
        return retrieve_similar_facts(
            agent_id=agent_id,
            embedding=request.embedding,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/forget/low-confidence/{agent_id}")
async def forget_low_confidence_facts_endpoint(
    agent_id: str, confidence_threshold: float = 0.5
):
    try:
        deleted_count = forget_low_confidence_facts(agent_id, confidence_threshold)
        return {
            "message": f"Deleted {deleted_count} low-confidence facts for agent {agent_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
