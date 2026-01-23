from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
from sqlalchemy import JSON
from pgvector.sqlalchemy import Vector

class Procedure(SQLModel, table=True):
    class Config:
        arbitrary_types_allowed = True

    id: Optional[int] = Field(default=None, primary_key=True)
    agent_id: str
    name: str
    description: Optional[str] = None
    steps: list[dict] = Field(sa_type=JSON)
    embedding: Optional[Vector] = Field(default=None, sa_type=Vector)
    success_rate: float = Field(default=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: int = Field(default=1)