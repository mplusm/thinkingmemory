from sqlmodel import select, delete
from src.memory.episodic.models import MemoryItem
from src.memory.episodic.database import get_session
from datetime import datetime, timedelta

def store_memory(agent_id: str, content: dict, embedding: list[float] = None, extra_data: dict = None):
    item = MemoryItem(
        agent_id=agent_id,
        content=content,
        embedding=embedding,
        extra_data=extra_data
    )
    with next(get_session()) as session:
        session.add(item)
        session.commit()
        session.refresh(item)
        return item

def retrieve_memories(agent_id: str, limit: int = 10):
    with next(get_session()) as session:
        statement = select(MemoryItem).where(MemoryItem.agent_id == agent_id).limit(limit)
        return session.exec(statement).all()

def forget_old_memories(agent_id: str, days: int = 30):
    """Delete memories older than `days` for a specific agent."""
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    with next(get_session()) as session:
        statement = delete(MemoryItem).where(
            MemoryItem.agent_id == agent_id,
            MemoryItem.timestamp < cutoff_date
        )
        result = session.exec(statement)
        session.commit()
        return result.rowcount

def forget_low_relevance_memories(agent_id: str, relevance_threshold: float = 0.5):
    """Delete memories with low relevance (placeholder for future implementation)."""
    # Placeholder: In a real implementation, you would filter based on relevance scores.
    # For now, we'll just return 0 as no memories are deleted.
    return 0