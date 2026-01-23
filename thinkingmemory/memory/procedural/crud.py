"""
Procedural Memory CRUD operations.

All functions accept an optional tenant_id parameter for multi-tenant deployments.
When tenant_id is None, no tenant filtering is applied (single-tenant mode).
"""

from typing import Optional

from sqlmodel import select, delete
from sqlalchemy import func

from thinkingmemory.core.database import get_session_context
from thinkingmemory.memory.procedural.models import Procedure


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
        return procedure


def retrieve_procedures(agent_id: str, limit: int = 10, tenant_id: Optional[str] = None):
    """Retrieve procedures for an agent."""
    with get_session_context() as session:
        statement = select(Procedure).where(Procedure.agent_id == agent_id)
        if tenant_id is not None:
            statement = statement.where(Procedure.tenant_id == tenant_id)
        statement = statement.limit(limit)
        return session.exec(statement).all()


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
        return procedure


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
        return session.exec(statement).all()
