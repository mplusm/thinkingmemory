"""
Semantic Memory CRUD operations.

All functions accept an optional tenant_id parameter for multi-tenant deployments.
When tenant_id is None, no tenant filtering is applied (single-tenant mode).
"""

from typing import Optional

from sqlmodel import select, delete
from sqlalchemy import func

from thinkingmemory.core.database import get_session_context
from thinkingmemory.memory.semantic.models import Fact


def _fact_to_dict(fact: Fact) -> dict:
    """Convert a Fact to a serializable dict."""
    return {
        "id": fact.id,
        "tenant_id": fact.tenant_id,
        "agent_id": fact.agent_id,
        "fact": fact.fact,
        "embedding": list(fact.embedding) if fact.embedding else None,
        "timestamp": fact.timestamp,
        "confidence": fact.confidence,
        "source": fact.source,
    }


def store_fact(
    agent_id: str,
    fact: str,
    embedding: list[float] = None,
    confidence: float = 1.0,
    source: str = None,
    tenant_id: Optional[str] = None,
):
    """Store a new semantic fact."""
    fact_item = Fact(
        agent_id=agent_id,
        fact=fact,
        embedding=embedding,
        confidence=confidence,
        source=source,
    )
    if tenant_id is not None:
        fact_item.tenant_id = tenant_id

    with get_session_context() as session:
        session.add(fact_item)
        session.commit()
        session.refresh(fact_item)
        # Convert to dict before session closes
        return _fact_to_dict(fact_item)


def retrieve_facts(agent_id: str, limit: int = 10, tenant_id: Optional[str] = None):
    """Retrieve facts for an agent."""
    with get_session_context() as session:
        statement = select(Fact).where(Fact.agent_id == agent_id)
        if tenant_id is not None:
            statement = statement.where(Fact.tenant_id == tenant_id)
        statement = statement.limit(limit)
        facts = session.exec(statement).all()
        # Convert to dicts before session closes
        return [_fact_to_dict(f) for f in facts]


def forget_low_confidence_facts(
    agent_id: str,
    confidence_threshold: float = 0.5,
    tenant_id: Optional[str] = None,
):
    """Delete facts with confidence below the threshold."""
    with get_session_context() as session:
        statement = delete(Fact).where(
            Fact.agent_id == agent_id,
            Fact.confidence < confidence_threshold,
        )
        if tenant_id is not None:
            statement = statement.where(Fact.tenant_id == tenant_id)

        result = session.exec(statement)
        session.commit()
        return result.rowcount


def retrieve_similar_facts(
    agent_id: str,
    embedding: list[float],
    limit: int = 10,
    similarity_threshold: float = 0.5,
    tenant_id: Optional[str] = None,
):
    """Retrieve facts similar to the given embedding."""
    with get_session_context() as session:
        statement = select(Fact).where(
            Fact.agent_id == agent_id,
            Fact.embedding.isnot(None),
        )
        if tenant_id is not None:
            statement = statement.where(Fact.tenant_id == tenant_id)

        statement = statement.order_by(
            func.l2_distance(Fact.embedding, embedding)
        ).limit(limit)
        facts = session.exec(statement).all()
        # Convert to dicts before session closes
        return [_fact_to_dict(f) for f in facts]
