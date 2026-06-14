"""
The unified ``Memory`` model — one substrate for all memory.

The legacy design used four tables (memoryitem/fact/procedure/...). Here the
"layer" becomes a *policy tag* (``mtype``) on a single row, so working/episodic/
semantic/procedural memories live together and are retrieved by one ``recall``
query. Fields beyond the content carry the signals the recall engine and (future)
lifecycle workers need: salience, confidence, bitemporal validity, and
provenance.
"""

from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field
from sqlalchemy import JSON
from pgvector.sqlalchemy import Vector

from thinkingmemory.core.embeddings import EMBEDDING_DIM
from thinkingmemory.core.timeutils import utcnow


class Memory(SQLModel, table=True):
    __tablename__ = "memory"

    class Config:
        arbitrary_types_allowed = True

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(default="default", index=True)
    agent_id: str = Field(index=True)

    # Policy tags (not separate tables)
    scope: str = Field(default="private")          # private | shared | global
    mtype: str = Field(default="episodic")         # episodic | semantic | procedural | working

    # Payload: structured content + the text that gets embedded and packed
    content: dict = Field(sa_type=JSON)
    text: str
    embedding: Optional[Vector] = Field(default=None, sa_type=Vector(EMBEDDING_DIM))

    # Retrieval / lifecycle signals
    salience: float = Field(default=1.0)
    confidence: float = Field(default=1.0)
    decay_rate: float = Field(default=0.0)
    recall_count: int = Field(default=0)
    last_recalled_at: Optional[datetime] = Field(default=None)

    # Bitemporal-lite: when it was true in the world vs. when we learned it
    valid_from: datetime = Field(default_factory=utcnow)
    valid_to: Optional[datetime] = Field(default=None)        # null = currently true
    created_at: datetime = Field(default_factory=utcnow)
    superseded_at: Optional[datetime] = Field(default=None)   # set when forgotten/replaced

    # Provenance: {source, derived_from: [ids], extractor, ...}
    provenance: Optional[dict] = Field(default=None, sa_type=JSON)


class AuditLog(SQLModel, table=True):
    """Append-only record of memory operations, for enterprise auditability."""

    __tablename__ = "memory_audit"

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(default="default", index=True)
    agent_id: str = Field(index=True)
    action: str = Field(index=True)            # remember | recall | forget | maintenance
    target_id: Optional[int] = Field(default=None)   # memory id, when applicable
    details: Optional[dict] = Field(default=None, sa_type=JSON)
    created_at: datetime = Field(default_factory=utcnow, index=True)
