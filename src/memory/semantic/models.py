from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
from sqlalchemy import JSON
from pgvector.sqlalchemy import Vector

class Fact(SQLModel, table=True):
    class Config:
        arbitrary_types_allowed = True

    id: Optional[int] = Field(default=None, primary_key=True)
    agent_id: str
    fact: str
    embedding: Optional[Vector] = Field(default=None, sa_type=Vector)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = Field(default=1.0)
    source: Optional[str] = None