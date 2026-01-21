from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class MemoryItem(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    agent_id: str
    memory_type: str = "episodic"
    content: dict
    embedding: Optional[list[float]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[dict] = None