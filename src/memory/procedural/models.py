from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
from sqlalchemy import JSON

class Procedure(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    agent_id: str
    name: str
    description: Optional[str] = None
    steps: list[dict] = Field(sa_type=JSON)
    success_rate: float = Field(default=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: int = Field(default=1)