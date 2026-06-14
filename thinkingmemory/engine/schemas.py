"""Request/response schemas for the unified /v1 memory API."""

from datetime import datetime
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
    decay_rate: Optional[float] = None   # defaults per mtype if omitted
    provenance: Optional[dict] = None


class RememberManyRequest(BaseModel):
    items: list[RememberRequest]


class MaintenanceRequest(BaseModel):
    agent_id: str
    interval_days: float = 1.0


class RecallRequest(BaseModel):
    agent_id: str
    intent: str
    scopes: Optional[list[str]] = None
    mtypes: Optional[list[str]] = None
    token_budget: int = Field(default=4000, gt=0)
    k: int = Field(default=20, gt=0)
    as_of: Optional[datetime] = None   # recall against belief at a past moment
    rerank: Optional[bool] = None      # cross-encoder rerank (defaults to setting)
    graph_hops: int = 0                # expand via the entity graph (0 = off)


class ForgetRequest(BaseModel):
    memory_id: int
    hard: bool = False


class LinkRequest(BaseModel):
    src_id: int
    dst_id: int
    relation: str = "relates_to"
    agent_id: str = ""
    weight: float = 1.0
    bidirectional: bool = False
