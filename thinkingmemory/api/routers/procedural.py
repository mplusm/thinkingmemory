"""
Procedural Memory API Router.

Handles storage, retrieval, and forgetting of procedural knowledge,
user preferences, and workflow habits.
"""

from fastapi import APIRouter, HTTPException

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


# =============================================================================
# UserPreference Endpoints
# =============================================================================


@router.post("/preferences/store")
async def store_preference_endpoint(request: PreferenceStoreRequest):
    try:
        return store_preference(
            agent_id=request.agent_id,
            category=request.category,
            key=request.key,
            value=request.value,
            confidence=request.confidence,
            source=request.source,
            embedding=request.embedding,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preferences/retrieve/{agent_id}")
async def retrieve_preferences_endpoint(agent_id: str, limit: int = 10):
    try:
        return retrieve_preferences(agent_id, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preferences/by-key/{agent_id}")
async def retrieve_preference_by_key_endpoint(
    agent_id: str, category: str, key: str
):
    try:
        result = retrieve_preference_by_key(agent_id, category, key)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Preference not found for agent={agent_id}, category={category}, key={key}",
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preferences/similar/{agent_id}")
async def retrieve_similar_preferences_endpoint(agent_id: str, request: SimilarityRequest):
    try:
        return retrieve_similar_preferences(
            agent_id=agent_id,
            embedding=request.embedding,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/preferences/delete/{preference_id}")
async def delete_preference_endpoint(preference_id: int):
    try:
        success = delete_preference(preference_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Preference {preference_id} not found")
        return {"message": f"Deleted preference {preference_id}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/preferences/forget/low-confidence/{agent_id}")
async def forget_low_confidence_preferences_endpoint(
    agent_id: str, confidence_threshold: float = 0.5
):
    try:
        deleted_count = forget_low_confidence_preferences(agent_id, confidence_threshold)
        return {
            "message": f"Deleted {deleted_count} low-confidence preferences for agent {agent_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# WorkflowHabit Endpoints
# =============================================================================


@router.post("/habits/store")
async def store_habit_endpoint(request: HabitStoreRequest):
    try:
        return store_habit(
            agent_id=request.agent_id,
            habit_name=request.habit_name,
            pattern=request.pattern,
            description=request.description,
            embedding=request.embedding,
            tags=request.tags,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/habits/retrieve/{agent_id}")
async def retrieve_habits_endpoint(agent_id: str, limit: int = 10):
    try:
        return retrieve_habits(agent_id, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/habits/similar/{agent_id}")
async def retrieve_similar_habits_endpoint(agent_id: str, request: SimilarityRequest):
    try:
        return retrieve_similar_habits(
            agent_id=agent_id,
            embedding=request.embedding,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/habits/increment/{habit_id}")
async def increment_habit_endpoint(habit_id: int, request: HabitIncrementRequest):
    try:
        result = increment_habit(habit_id, request.success)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Habit {habit_id} not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/habits/update/{habit_id}")
async def update_habit_endpoint(habit_id: int, request: HabitUpdateRequest):
    try:
        updates = request.model_dump(exclude_unset=True)
        result = update_habit(habit_id, **updates)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Habit {habit_id} not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/habits/delete/{habit_id}")
async def delete_habit_endpoint(habit_id: int):
    try:
        success = delete_habit(habit_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Habit {habit_id} not found")
        return {"message": f"Deleted habit {habit_id}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/habits/forget/unused/{agent_id}")
async def forget_unused_habits_endpoint(
    agent_id: str, max_frequency: int = 1
):
    try:
        deleted_count = forget_unused_habits(agent_id, max_frequency)
        return {
            "message": f"Deleted {deleted_count} unused habits for agent {agent_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
