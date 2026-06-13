"""
Request and response schemas for the ThinkingMemory API.

These Pydantic models define the API contract and can be reused
by wrapper applications.
"""

from pydantic import BaseModel, field_validator
from typing import Optional

from thinkingmemory.core.embeddings import EMBEDDING_DIM


def _validate_embedding_dim(value: Optional[list[float]]) -> Optional[list[float]]:
    """Reject embeddings whose length does not match the configured dimension."""
    if value is not None and len(value) != EMBEDDING_DIM:
        raise ValueError(
            f"embedding must have {EMBEDDING_DIM} dimensions, got {len(value)}"
        )
    return value


# =============================================================================
# Episodic Memory Schemas
# =============================================================================

class MemoryStoreRequest(BaseModel):
    agent_id: str
    content: dict
    memory_type: str = "episodic"
    embedding: Optional[list[float]] = None
    extra_data: Optional[dict] = None

    _check_embedding = field_validator("embedding")(_validate_embedding_dim)


class SimilarityRequest(BaseModel):
    embedding: list[float]
    limit: int = 10
    similarity_threshold: float = 0.5

    _check_embedding = field_validator("embedding")(_validate_embedding_dim)


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
# Semantic Memory — Data Models Schemas
# =============================================================================

class DataSourceStoreRequest(BaseModel):
    agent_id: str
    source_name: str
    source_type: Optional[str] = None
    description: Optional[str] = None
    connection_alias: Optional[str] = None
    embedding: Optional[list[float]] = None
    metadata_: Optional[dict] = None


class DataSourceUpdateRequest(BaseModel):
    source_name: Optional[str] = None
    source_type: Optional[str] = None
    description: Optional[str] = None
    connection_alias: Optional[str] = None
    metadata_: Optional[dict] = None


class DataTableStoreRequest(BaseModel):
    agent_id: str
    table_name: str
    data_source_id: Optional[int] = None
    schema_name: Optional[str] = None
    table_type: Optional[str] = None
    description: Optional[str] = None
    row_count_estimate: Optional[int] = None
    embedding: Optional[list[float]] = None
    tags: Optional[list[str]] = None


class DataTableUpdateRequest(BaseModel):
    table_name: Optional[str] = None
    schema_name: Optional[str] = None
    table_type: Optional[str] = None
    description: Optional[str] = None
    row_count_estimate: Optional[int] = None
    tags: Optional[list[str]] = None


class DataColumnStoreRequest(BaseModel):
    agent_id: str
    column_name: str
    data_table_id: Optional[int] = None
    data_type: Optional[str] = None
    is_nullable: Optional[bool] = None
    is_primary_key: Optional[bool] = None
    is_foreign_key: Optional[bool] = None
    foreign_key_ref: Optional[str] = None
    description: Optional[str] = None
    sample_values: Optional[list[str]] = None
    lineage: Optional[dict] = None
    embedding: Optional[list[float]] = None
    tags: Optional[list[str]] = None


class DataColumnUpdateRequest(BaseModel):
    column_name: Optional[str] = None
    data_type: Optional[str] = None
    is_nullable: Optional[bool] = None
    is_primary_key: Optional[bool] = None
    is_foreign_key: Optional[bool] = None
    foreign_key_ref: Optional[str] = None
    description: Optional[str] = None
    sample_values: Optional[list[str]] = None
    lineage: Optional[dict] = None
    tags: Optional[list[str]] = None


class KnowledgeStoreRequest(BaseModel):
    agent_id: str
    entity_type: str
    name: str
    description: Optional[str] = None
    properties: Optional[dict] = None
    relationships: Optional[list[dict]] = None
    source: Optional[str] = None
    confidence: float = 1.0
    embedding: Optional[list[float]] = None
    tags: Optional[list[str]] = None


class KnowledgeUpdateRequest(BaseModel):
    entity_type: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    properties: Optional[dict] = None
    relationships: Optional[list[dict]] = None
    source: Optional[str] = None
    confidence: Optional[float] = None
    tags: Optional[list[str]] = None


# =============================================================================
# Procedural Memory — Preference & Habit Schemas
# =============================================================================

class PreferenceStoreRequest(BaseModel):
    agent_id: str
    category: str
    key: str
    value: str
    confidence: float = 1.0
    source: Optional[str] = None
    embedding: Optional[list[float]] = None


class HabitStoreRequest(BaseModel):
    agent_id: str
    habit_name: str
    pattern: dict
    description: Optional[str] = None
    embedding: Optional[list[float]] = None
    tags: Optional[list[str]] = None


class HabitUpdateRequest(BaseModel):
    habit_name: Optional[str] = None
    description: Optional[str] = None
    pattern: Optional[dict] = None
    tags: Optional[list[str]] = None


class HabitIncrementRequest(BaseModel):
    success: bool


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
