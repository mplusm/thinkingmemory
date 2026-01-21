from sqlmodel import select, delete
from src.memory.semantic.models import Fact
from src.memory.semantic.database import get_session

def store_fact(agent_id: str, fact: str, embedding: list[float] = None, confidence: float = 1.0, source: str = None):
    fact_item = Fact(
        agent_id=agent_id,
        fact=fact,
        embedding=embedding,
        confidence=confidence,
        source=source
    )
    with next(get_session()) as session:
        session.add(fact_item)
        session.commit()
        session.refresh(fact_item)
        return fact_item

def retrieve_facts(agent_id: str, limit: int = 10):
    with next(get_session()) as session:
        statement = select(Fact).where(Fact.agent_id == agent_id).limit(limit)
        return session.exec(statement).all()

def forget_low_confidence_facts(agent_id: str, confidence_threshold: float = 0.5):
    """Delete facts with confidence below the threshold."""
    with next(get_session()) as session:
        statement = delete(Fact).where(
            Fact.agent_id == agent_id,
            Fact.confidence < confidence_threshold
        )
        result = session.exec(statement)
        session.commit()
        return result.rowcount