from sqlmodel import select, delete
from sqlalchemy import func

from thinkingmemory.core.database import get_session_context
from thinkingmemory.memory.semantic.models import Fact

def store_fact(agent_id: str, fact: str, embedding: list[float] = None, confidence: float = 1.0, source: str = None):
    fact_item = Fact(
        agent_id=agent_id,
        fact=fact,
        embedding=embedding,
        confidence=confidence,
        source=source
    )
    with get_session_context() as session:
        session.add(fact_item)
        session.commit()
        session.refresh(fact_item)
        return fact_item

def retrieve_facts(agent_id: str, limit: int = 10):
    with get_session_context() as session:
        statement = select(Fact).where(Fact.agent_id == agent_id).limit(limit)
        return session.exec(statement).all()

def forget_low_confidence_facts(agent_id: str, confidence_threshold: float = 0.5):
    """Delete facts with confidence below the threshold."""
    with get_session_context() as session:
        statement = delete(Fact).where(
            Fact.agent_id == agent_id,
            Fact.confidence < confidence_threshold
        )
        result = session.exec(statement)
        session.commit()
        return result.rowcount

def retrieve_similar_facts(agent_id: str, embedding: list[float], limit: int = 10, similarity_threshold: float = 0.5):
    """Retrieve facts similar to the given embedding."""
    with get_session_context() as session:
        # Calculate cosine distance between the query embedding and stored embeddings
        # pgvector uses <-> for cosine distance (lower is more similar)
        # We'll use the <-> operator via func
        statement = select(Fact).where(
            Fact.agent_id == agent_id,
            Fact.embedding.isnot(None)
        ).order_by(
            func.l2_distance(Fact.embedding, embedding)
        ).limit(limit)
        return session.exec(statement).all()