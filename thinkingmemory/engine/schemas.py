"""Request/response schemas for the unified /v1 memory API."""

from typing import Optional

from pydantic import BaseModel, Field


class RememberRequest(BaseModel):
    agent_id: str
    content: dict
    text: Optional[str] = None          # defaults to a rendering of content
    mtype: str = "episodic"             # episodic | semantic | procedural | working
    scope: str = "private"             # private | shared | global
    salience: float = 1.0
    confidence: float = 1.0
    provenance: Optional[dict] = None


class RememberManyRequest(BaseModel):
    items: list[RememberRequest]


class RecallRequest(BaseModel):
    agent_id: str
    intent: str
    scopes: Optional[list[str]] = None
    mtypes: Optional[list[str]] = None
    token_budget: int = Field(default=4000, gt=0)
    k: int = Field(default=20, gt=0)


class ForgetRequest(BaseModel):
    memory_id: int
    hard: bool = False
