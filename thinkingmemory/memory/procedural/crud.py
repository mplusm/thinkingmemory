"""
Procedural Memory CRUD operations.

All functions accept an optional tenant_id parameter for multi-tenant deployments.
When tenant_id is None, no tenant filtering is applied (single-tenant mode).
"""

from typing import Optional

from sqlmodel import select, delete
from sqlalchemy import func

from thinkingmemory.core.database import get_session_context
from thinkingmemory.memory.procedural.models import Procedure, UserPreference, WorkflowHabit


def _procedure_to_dict(procedure: Procedure) -> dict:
    """Convert a Procedure to a serializable dict."""
    return {
        "id": procedure.id,
        "tenant_id": procedure.tenant_id,
        "agent_id": procedure.agent_id,
        "name": procedure.name,
        "description": procedure.description,
        "steps": procedure.steps,
        "embedding": list(procedure.embedding) if procedure.embedding else None,
        "success_rate": procedure.success_rate,
        "timestamp": procedure.timestamp,
        "version": procedure.version,
    }


def store_procedure(
    agent_id: str,
    name: str,
    description: str = None,
    steps: list[dict] = None,
    success_rate: float = 1.0,
    version: int = 1,
    tenant_id: Optional[str] = None,
):
    """Store a new procedure."""
    procedure = Procedure(
        agent_id=agent_id,
        name=name,
        description=description,
        steps=steps,
        success_rate=success_rate,
        version=version,
    )
    if tenant_id is not None:
        procedure.tenant_id = tenant_id

    with get_session_context() as session:
        session.add(procedure)
        session.commit()
        session.refresh(procedure)
        # Convert to dict before session closes
        return _procedure_to_dict(procedure)


def retrieve_procedures(agent_id: str, limit: int = 10, tenant_id: Optional[str] = None):
    """Retrieve procedures for an agent."""
    with get_session_context() as session:
        statement = select(Procedure).where(Procedure.agent_id == agent_id)
        if tenant_id is not None:
            statement = statement.where(Procedure.tenant_id == tenant_id)
        statement = statement.limit(limit)
        procedures = session.exec(statement).all()
        # Convert to dicts before session closes
        return [_procedure_to_dict(p) for p in procedures]


def update_procedure_success_rate(
    procedure_id: int,
    success_rate: float,
    tenant_id: Optional[str] = None,
):
    """Update a procedure's success rate."""
    with get_session_context() as session:
        procedure = session.get(Procedure, procedure_id)
        if procedure:
            # Verify tenant if provided
            if tenant_id is not None and hasattr(procedure, 'tenant_id'):
                if procedure.tenant_id != tenant_id:
                    return None  # Not authorized
            procedure.success_rate = success_rate
            session.commit()
            session.refresh(procedure)
            # Convert to dict before session closes
            return _procedure_to_dict(procedure)
        return None


def forget_low_success_procedures(
    agent_id: str,
    success_threshold: float = 0.5,
    tenant_id: Optional[str] = None,
):
    """Delete procedures with success rate below the threshold."""
    with get_session_context() as session:
        statement = delete(Procedure).where(
            Procedure.agent_id == agent_id,
            Procedure.success_rate < success_threshold,
        )
        if tenant_id is not None:
            statement = statement.where(Procedure.tenant_id == tenant_id)

        result = session.exec(statement)
        session.commit()
        return result.rowcount


def retrieve_similar_procedures(
    agent_id: str,
    embedding: list[float],
    limit: int = 10,
    similarity_threshold: float = 0.5,
    tenant_id: Optional[str] = None,
):
    """Retrieve procedures similar to the given embedding."""
    with get_session_context() as session:
        statement = select(Procedure).where(
            Procedure.agent_id == agent_id,
            Procedure.embedding.isnot(None),
        )
        if tenant_id is not None:
            statement = statement.where(Procedure.tenant_id == tenant_id)

        statement = statement.order_by(
            func.l2_distance(Procedure.embedding, embedding)
        ).limit(limit)
        procedures = session.exec(statement).all()
        # Convert to dicts before session closes
        return [_procedure_to_dict(p) for p in procedures]


# =============================================================================
# UserPreference CRUD
# =============================================================================


def _preference_to_dict(p: UserPreference) -> dict:
    """Convert a UserPreference to a serializable dict."""
    return {
        "id": p.id,
        "tenant_id": p.tenant_id,
        "agent_id": p.agent_id,
        "category": p.category,
        "key": p.key,
        "value": p.value,
        "confidence": p.confidence,
        "source": p.source,
        "observation_count": p.observation_count,
        "last_observed": p.last_observed,
        "embedding": list(p.embedding) if p.embedding else None,
        "timestamp": p.timestamp,
    }


def store_preference(
    agent_id: str,
    category: str,
    key: str,
    value: str,
    confidence: float = 1.0,
    source: str = None,
    observation_count: int = 1,
    last_observed: "datetime" = None,
    embedding: list[float] = None,
    tenant_id: Optional[str] = None,
):
    """Store or update a user preference (upsert by agent_id+category+key)."""
    from datetime import datetime as dt

    with get_session_context() as session:
        # Check for existing preference with same agent_id+category+key
        statement = select(UserPreference).where(
            UserPreference.agent_id == agent_id,
            UserPreference.category == category,
            UserPreference.key == key,
        )
        if tenant_id is not None:
            statement = statement.where(UserPreference.tenant_id == tenant_id)

        existing = session.exec(statement).first()

        if existing is not None:
            existing.value = value
            existing.confidence = confidence
            if source is not None:
                existing.source = source
            existing.observation_count = existing.observation_count + 1
            existing.last_observed = last_observed or dt.utcnow()
            if embedding is not None:
                existing.embedding = embedding
            session.commit()
            session.refresh(existing)
            return _preference_to_dict(existing)

        item = UserPreference(
            agent_id=agent_id,
            category=category,
            key=key,
            value=value,
            confidence=confidence,
            source=source,
            observation_count=observation_count,
            last_observed=last_observed,
            embedding=embedding,
        )
        if tenant_id is not None:
            item.tenant_id = tenant_id

        session.add(item)
        session.commit()
        session.refresh(item)
        return _preference_to_dict(item)


def retrieve_preferences(agent_id: str, limit: int = 10, tenant_id: Optional[str] = None):
    """Retrieve preferences for an agent."""
    with get_session_context() as session:
        statement = select(UserPreference).where(UserPreference.agent_id == agent_id)
        if tenant_id is not None:
            statement = statement.where(UserPreference.tenant_id == tenant_id)
        statement = statement.limit(limit)
        items = session.exec(statement).all()
        return [_preference_to_dict(i) for i in items]


def retrieve_preference_by_key(
    agent_id: str,
    category: str,
    key: str,
    tenant_id: Optional[str] = None,
):
    """Retrieve a specific preference by agent_id+category+key."""
    with get_session_context() as session:
        statement = select(UserPreference).where(
            UserPreference.agent_id == agent_id,
            UserPreference.category == category,
            UserPreference.key == key,
        )
        if tenant_id is not None:
            statement = statement.where(UserPreference.tenant_id == tenant_id)
        item = session.exec(statement).first()
        if item is None:
            return None
        return _preference_to_dict(item)


def retrieve_similar_preferences(
    agent_id: str,
    embedding: list[float],
    limit: int = 10,
    similarity_threshold: float = 0.5,
    tenant_id: Optional[str] = None,
):
    """Retrieve preferences similar to the given embedding."""
    with get_session_context() as session:
        statement = select(UserPreference).where(
            UserPreference.agent_id == agent_id,
            UserPreference.embedding.isnot(None),
        )
        if tenant_id is not None:
            statement = statement.where(UserPreference.tenant_id == tenant_id)
        statement = statement.order_by(
            func.l2_distance(UserPreference.embedding, embedding)
        ).limit(limit)
        items = session.exec(statement).all()
        return [_preference_to_dict(i) for i in items]


def delete_preference(preference_id: int, tenant_id: Optional[str] = None):
    """Delete a preference by ID."""
    with get_session_context() as session:
        item = session.get(UserPreference, preference_id)
        if item is None:
            return False
        if tenant_id is not None and item.tenant_id != tenant_id:
            return False
        session.delete(item)
        session.commit()
        return True


def forget_low_confidence_preferences(
    agent_id: str,
    confidence_threshold: float = 0.5,
    tenant_id: Optional[str] = None,
):
    """Delete preferences with confidence below the threshold."""
    with get_session_context() as session:
        statement = delete(UserPreference).where(
            UserPreference.agent_id == agent_id,
            UserPreference.confidence < confidence_threshold,
        )
        if tenant_id is not None:
            statement = statement.where(UserPreference.tenant_id == tenant_id)
        result = session.exec(statement)
        session.commit()
        return result.rowcount


# =============================================================================
# WorkflowHabit CRUD
# =============================================================================


def _habit_to_dict(h: WorkflowHabit) -> dict:
    """Convert a WorkflowHabit to a serializable dict."""
    return {
        "id": h.id,
        "tenant_id": h.tenant_id,
        "agent_id": h.agent_id,
        "habit_name": h.habit_name,
        "description": h.description,
        "pattern": h.pattern,
        "frequency_count": h.frequency_count,
        "last_performed": h.last_performed,
        "success_count": h.success_count,
        "failure_count": h.failure_count,
        "embedding": list(h.embedding) if h.embedding else None,
        "timestamp": h.timestamp,
        "tags": h.tags,
    }


def store_habit(
    agent_id: str,
    habit_name: str,
    pattern: dict,
    description: str = None,
    frequency_count: int = 1,
    last_performed: "datetime" = None,
    success_count: int = 0,
    failure_count: int = 0,
    embedding: list[float] = None,
    tags: list[str] = None,
    tenant_id: Optional[str] = None,
):
    """Store a new workflow habit."""
    item = WorkflowHabit(
        agent_id=agent_id,
        habit_name=habit_name,
        pattern=pattern,
        description=description,
        frequency_count=frequency_count,
        last_performed=last_performed,
        success_count=success_count,
        failure_count=failure_count,
        embedding=embedding,
        tags=tags,
    )
    if tenant_id is not None:
        item.tenant_id = tenant_id

    with get_session_context() as session:
        session.add(item)
        session.commit()
        session.refresh(item)
        return _habit_to_dict(item)


def retrieve_habits(agent_id: str, limit: int = 10, tenant_id: Optional[str] = None):
    """Retrieve habits for an agent, ordered by frequency descending."""
    with get_session_context() as session:
        statement = select(WorkflowHabit).where(WorkflowHabit.agent_id == agent_id)
        if tenant_id is not None:
            statement = statement.where(WorkflowHabit.tenant_id == tenant_id)
        statement = statement.order_by(WorkflowHabit.frequency_count.desc()).limit(limit)
        items = session.exec(statement).all()
        return [_habit_to_dict(i) for i in items]


def retrieve_similar_habits(
    agent_id: str,
    embedding: list[float],
    limit: int = 10,
    similarity_threshold: float = 0.5,
    tenant_id: Optional[str] = None,
):
    """Retrieve habits similar to the given embedding."""
    with get_session_context() as session:
        statement = select(WorkflowHabit).where(
            WorkflowHabit.agent_id == agent_id,
            WorkflowHabit.embedding.isnot(None),
        )
        if tenant_id is not None:
            statement = statement.where(WorkflowHabit.tenant_id == tenant_id)
        statement = statement.order_by(
            func.l2_distance(WorkflowHabit.embedding, embedding)
        ).limit(limit)
        items = session.exec(statement).all()
        return [_habit_to_dict(i) for i in items]


def increment_habit(habit_id: int, success: bool, tenant_id: Optional[str] = None):
    """Increment a habit's frequency and success/failure counts."""
    from datetime import datetime as dt

    with get_session_context() as session:
        item = session.get(WorkflowHabit, habit_id)
        if item is None:
            return None
        if tenant_id is not None and item.tenant_id != tenant_id:
            return None
        item.frequency_count += 1
        if success:
            item.success_count += 1
        else:
            item.failure_count += 1
        item.last_performed = dt.utcnow()
        session.commit()
        session.refresh(item)
        return _habit_to_dict(item)


def update_habit(
    habit_id: int,
    tenant_id: Optional[str] = None,
    **kwargs,
):
    """Update a habit's fields."""
    with get_session_context() as session:
        item = session.get(WorkflowHabit, habit_id)
        if item is None:
            return None
        if tenant_id is not None and item.tenant_id != tenant_id:
            return None
        for key, value in kwargs.items():
            if hasattr(item, key):
                setattr(item, key, value)
        session.commit()
        session.refresh(item)
        return _habit_to_dict(item)


def delete_habit(habit_id: int, tenant_id: Optional[str] = None):
    """Delete a habit by ID."""
    with get_session_context() as session:
        item = session.get(WorkflowHabit, habit_id)
        if item is None:
            return False
        if tenant_id is not None and item.tenant_id != tenant_id:
            return False
        session.delete(item)
        session.commit()
        return True


def forget_unused_habits(
    agent_id: str,
    max_frequency: int = 1,
    tenant_id: Optional[str] = None,
):
    """Delete habits with frequency_count at or below the threshold."""
    with get_session_context() as session:
        statement = delete(WorkflowHabit).where(
            WorkflowHabit.agent_id == agent_id,
            WorkflowHabit.frequency_count <= max_frequency,
        )
        if tenant_id is not None:
            statement = statement.where(WorkflowHabit.tenant_id == tenant_id)
        result = session.exec(statement)
        session.commit()
        return result.rowcount
