"""
Procedural Memory API Router.

Handles storage, retrieval, and forgetting of procedural knowledge,
user preferences, and workflow habits.

Every endpoint resolves an optional tenant id from the ``X-Tenant-ID`` header and
passes it to the CRUD layer. Unhandled exceptions propagate to the
application-wide error handler; only domain 404s are raised explicitly.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from thinkingmemory.api.dependencies import get_tenant_id
from thinkingmemory.api.schemas import (
    ProcedureStoreRequest,
    ProcedureUpdateRequest,
    SimilarityRequest,
    PreferenceStoreRequest,
    HabitStoreRequest,
    HabitUpdateRequest,
    HabitIncrementRequest,
)
from thinkingmemory.memory.procedural.crud import (
    store_procedure,
    retrieve_procedures,
    update_procedure_success_rate,
    forget_low_success_procedures,
    retrieve_similar_procedures,
    store_preference,
    retrieve_preferences,
    retrieve_preference_by_key,
    retrieve_similar_preferences,
    delete_preference,
    forget_low_confidence_preferences,
    store_habit,
    retrieve_habits,
    retrieve_similar_habits,
    increment_habit,
    update_habit,
    delete_habit,
    forget_unused_habits,
)

router = APIRouter(prefix="/procedural", tags=["procedural"])


@router.post("/store")
async def store_procedure_endpoint(
    request: ProcedureStoreRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return store_procedure(
        agent_id=request.agent_id,
        name=request.name,
        description=request.description,
        steps=request.steps,
        success_rate=request.success_rate,
        version=request.version,
        tenant_id=tenant_id,
    )


@router.get("/retrieve/{agent_id}")
async def retrieve_procedures_endpoint(
    agent_id: str,
    limit: int = 10,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return retrieve_procedures(agent_id, limit, tenant_id=tenant_id)


@router.post("/similar/{agent_id}")
async def retrieve_similar_procedures_endpoint(
    agent_id: str,
    request: SimilarityRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return retrieve_similar_procedures(
        agent_id=agent_id,
        embedding=request.embedding,
        limit=request.limit,
        similarity_threshold=request.similarity_threshold,
        tenant_id=tenant_id,
    )


@router.patch("/update/{procedure_id}")
async def update_procedure_endpoint(
    procedure_id: int,
    request: ProcedureUpdateRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    result = update_procedure_success_rate(
        procedure_id, request.success_rate, tenant_id=tenant_id
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"Procedure {procedure_id} not found")
    return result


@router.delete("/forget/low-success/{agent_id}")
async def forget_low_success_procedures_endpoint(
    agent_id: str,
    success_threshold: float = 0.5,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    deleted_count = forget_low_success_procedures(
        agent_id, success_threshold, tenant_id=tenant_id
    )
    return {
        "message": f"Deleted {deleted_count} low-success procedures for agent {agent_id}"
    }


# =============================================================================
# UserPreference Endpoints
# =============================================================================


@router.post("/preferences/store")
async def store_preference_endpoint(
    request: PreferenceStoreRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return store_preference(
        agent_id=request.agent_id,
        category=request.category,
        key=request.key,
        value=request.value,
        confidence=request.confidence,
        source=request.source,
        embedding=request.embedding,
        tenant_id=tenant_id,
    )


@router.get("/preferences/retrieve/{agent_id}")
async def retrieve_preferences_endpoint(
    agent_id: str,
    limit: int = 10,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return retrieve_preferences(agent_id, limit, tenant_id=tenant_id)


@router.get("/preferences/by-key/{agent_id}")
async def retrieve_preference_by_key_endpoint(
    agent_id: str,
    category: str,
    key: str,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    result = retrieve_preference_by_key(agent_id, category, key, tenant_id=tenant_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Preference not found for agent={agent_id}, category={category}, key={key}",
        )
    return result


@router.post("/preferences/similar/{agent_id}")
async def retrieve_similar_preferences_endpoint(
    agent_id: str,
    request: SimilarityRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return retrieve_similar_preferences(
        agent_id=agent_id,
        embedding=request.embedding,
        limit=request.limit,
        similarity_threshold=request.similarity_threshold,
        tenant_id=tenant_id,
    )


@router.delete("/preferences/delete/{preference_id}")
async def delete_preference_endpoint(
    preference_id: int,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    success = delete_preference(preference_id, tenant_id=tenant_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Preference {preference_id} not found")
    return {"message": f"Deleted preference {preference_id}"}


@router.delete("/preferences/forget/low-confidence/{agent_id}")
async def forget_low_confidence_preferences_endpoint(
    agent_id: str,
    confidence_threshold: float = 0.5,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    deleted_count = forget_low_confidence_preferences(
        agent_id, confidence_threshold, tenant_id=tenant_id
    )
    return {
        "message": f"Deleted {deleted_count} low-confidence preferences for agent {agent_id}"
    }


# =============================================================================
# WorkflowHabit Endpoints
# =============================================================================


@router.post("/habits/store")
async def store_habit_endpoint(
    request: HabitStoreRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return store_habit(
        agent_id=request.agent_id,
        habit_name=request.habit_name,
        pattern=request.pattern,
        description=request.description,
        embedding=request.embedding,
        tags=request.tags,
        tenant_id=tenant_id,
    )


@router.get("/habits/retrieve/{agent_id}")
async def retrieve_habits_endpoint(
    agent_id: str,
    limit: int = 10,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return retrieve_habits(agent_id, limit, tenant_id=tenant_id)


@router.post("/habits/similar/{agent_id}")
async def retrieve_similar_habits_endpoint(
    agent_id: str,
    request: SimilarityRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return retrieve_similar_habits(
        agent_id=agent_id,
        embedding=request.embedding,
        limit=request.limit,
        similarity_threshold=request.similarity_threshold,
        tenant_id=tenant_id,
    )


@router.post("/habits/increment/{habit_id}")
async def increment_habit_endpoint(
    habit_id: int,
    request: HabitIncrementRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    result = increment_habit(habit_id, request.success, tenant_id=tenant_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Habit {habit_id} not found")
    return result


@router.patch("/habits/update/{habit_id}")
async def update_habit_endpoint(
    habit_id: int,
    request: HabitUpdateRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    updates = request.model_dump(exclude_unset=True)
    result = update_habit(habit_id, tenant_id=tenant_id, **updates)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Habit {habit_id} not found")
    return result


@router.delete("/habits/delete/{habit_id}")
async def delete_habit_endpoint(
    habit_id: int,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    success = delete_habit(habit_id, tenant_id=tenant_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Habit {habit_id} not found")
    return {"message": f"Deleted habit {habit_id}"}


@router.delete("/habits/forget/unused/{agent_id}")
async def forget_unused_habits_endpoint(
    agent_id: str,
    max_frequency: int = 1,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    deleted_count = forget_unused_habits(agent_id, max_frequency, tenant_id=tenant_id)
    return {
        "message": f"Deleted {deleted_count} unused habits for agent {agent_id}"
    }
