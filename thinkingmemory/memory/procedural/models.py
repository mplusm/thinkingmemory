from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
from sqlalchemy import JSON
from pgvector.sqlalchemy import Vector

class Procedure(SQLModel, table=True):
    class Config:
        arbitrary_types_allowed = True

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(default="default", index=True)  # Multi-tenant support
    agent_id: str
    name: str
    description: Optional[str] = None
    steps: list[dict] = Field(sa_type=JSON)
    embedding: Optional[Vector] = Field(default=None, sa_type=Vector)
    success_rate: float = Field(default=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: int = Field(default=1)


class UserPreference(SQLModel, table=True):
    __tablename__ = "userpreference"

    class Config:
        arbitrary_types_allowed = True

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(default="default", index=True)
    agent_id: str
    category: str
    key: str
    value: str
    confidence: float = Field(default=1.0)
    source: Optional[str] = None
    observation_count: int = Field(default=1)
    last_observed: Optional[datetime] = None
    embedding: Optional[Vector] = Field(default=None, sa_type=Vector)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WorkflowHabit(SQLModel, table=True):
    __tablename__ = "workflowhabit"

    class Config:
        arbitrary_types_allowed = True

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(default="default", index=True)
    agent_id: str
    habit_name: str
    description: Optional[str] = None
    pattern: dict = Field(sa_type=JSON)
    frequency_count: int = Field(default=1)
    last_performed: Optional[datetime] = None
    success_count: int = Field(default=0)
    failure_count: int = Field(default=0)
    embedding: Optional[Vector] = Field(default=None, sa_type=Vector)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tags: Optional[list[str]] = Field(default=None, sa_type=JSON)