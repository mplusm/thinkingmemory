from sqlmodel import select
from src.memory.episodic.models import MemoryItem
from src.memory.episodic.database import get_session

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