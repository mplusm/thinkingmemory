"""
Request and response schemas for the ThinkingMemory API.

These Pydantic models define the API contract and can be reused
by wrapper applications.
"""

from pydantic import BaseModel
from typing import Optional


# =============================================================================
# Episodic Memory Schemas
# =============================================================================

class MemoryStoreRequest(BaseModel):
    agent_id: str
    content: dict
    embedding: Optional[list[float]] = None
    extra_data: Optional[dict] = None


class SimilarityRequest(BaseModel):
    embedding: list[float]
    limit: int = 10
    similarity_threshold: float = 0.5


# =============================================================================
# Semantic Memory Schemas
# =============================================================================

class FactStoreRequest(BaseModel):
    agent_id: str
    fact: str
    embedding: Optional[list[float]] = None
    confidence: float = 1.0
    source: Optional[str] = None


# =============================================================================
# Procedural Memory Schemas
# =============================================================================

class ProcedureStoreRequest(BaseModel):
    agent_id: str
    name: str
    description: Optional[str] = None
    steps: list[dict]
    success_rate: float = 1.0
    version: int = 1


class ProcedureUpdateRequest(BaseModel):
    success_rate: float


# =============================================================================
# Working Memory Schemas
# =============================================================================

class WorkingMemoryStoreRequest(BaseModel):
    agent_id: str
    key: str
    value: dict
    ttl: int = 300  # Default 5 minutes


class WorkingMemoryUpdateRequest(BaseModel):
    value: dict
    ttl: Optional[int] = None  # If None, preserves existing TTL


class ExtendTTLRequest(BaseModel):
    additional_seconds: int
