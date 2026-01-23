from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
from sqlalchemy import JSON
from pgvector.sqlalchemy import Vector

class MemoryItem(SQLModel, table=True):
    class Config:
        arbitrary_types_allowed = True

    id: Optional[int] = Field(default=None, primary_key=True)
    agent_id: str
    memory_type: str = "episodic"
    content: dict = Field(sa_type=JSON)
    embedding: Optional[Vector] = Field(default=None, sa_type=Vector)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    extra_data: Optional[dict] = Field(default=None, sa_type=JSON)
    # Relevance tracking fields
    access_count: int = Field(default=0)
    last_accessed: Optional[datetime] = Field(default=None)
    # Compression tracking
    is_compressed: bool = Field(default=False)
    source_memory_ids: Optional[list[int]] = Field(default=None, sa_type=JSON)