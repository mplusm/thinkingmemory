from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
from sqlalchemy import JSON

class Fact(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    agent_id: str
    fact: str
    embedding: Optional[list[float]] = Field(default=None, sa_type=JSON)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = Field(default=1.0)
    source: Optional[str] = None