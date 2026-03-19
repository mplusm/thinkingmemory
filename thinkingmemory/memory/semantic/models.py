from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
from sqlalchemy import JSON
from pgvector.sqlalchemy import Vector

class Fact(SQLModel, table=True):
    class Config:
        arbitrary_types_allowed = True

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(default="default", index=True)  # Multi-tenant support
    agent_id: str
    fact: str
    embedding: Optional[Vector] = Field(default=None, sa_type=Vector)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = Field(default=1.0)
    source: Optional[str] = None


class DataSource(SQLModel, table=True):
    __tablename__ = "datasource"

    class Config:
        arbitrary_types_allowed = True

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(default="default", index=True)
    agent_id: str
    source_name: str
    source_type: Optional[str] = None
    description: Optional[str] = None
    connection_alias: Optional[str] = None
    embedding: Optional[Vector] = Field(default=None, sa_type=Vector)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata_: Optional[dict] = Field(default=None, sa_type=JSON)


class DataTable(SQLModel, table=True):
    __tablename__ = "datatable"

    class Config:
        arbitrary_types_allowed = True

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(default="default", index=True)
    agent_id: str
    data_source_id: Optional[int] = None
    schema_name: Optional[str] = None
    table_name: str
    table_type: Optional[str] = None
    description: Optional[str] = None
    row_count_estimate: Optional[int] = None
    embedding: Optional[Vector] = Field(default=None, sa_type=Vector)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tags: Optional[list[str]] = Field(default=None, sa_type=JSON)


class DataColumn(SQLModel, table=True):
    __tablename__ = "datacolumn"

    class Config:
        arbitrary_types_allowed = True

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(default="default", index=True)
    agent_id: str
    data_table_id: Optional[int] = None
    column_name: str
    data_type: Optional[str] = None
    is_nullable: Optional[bool] = None
    is_primary_key: Optional[bool] = None
    is_foreign_key: Optional[bool] = None
    foreign_key_ref: Optional[str] = None
    description: Optional[str] = None
    sample_values: Optional[list[str]] = Field(default=None, sa_type=JSON)
    lineage: Optional[dict] = Field(default=None, sa_type=JSON)
    embedding: Optional[Vector] = Field(default=None, sa_type=Vector)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tags: Optional[list[str]] = Field(default=None, sa_type=JSON)


class KnowledgeEntity(SQLModel, table=True):
    __tablename__ = "knowledgeentity"

    class Config:
        arbitrary_types_allowed = True

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(default="default", index=True)
    agent_id: str
    entity_type: str
    name: str
    description: Optional[str] = None
    properties: Optional[dict] = Field(default=None, sa_type=JSON)
    relationships: Optional[list[dict]] = Field(default=None, sa_type=JSON)
    source: Optional[str] = None
    confidence: float = Field(default=1.0)
    embedding: Optional[Vector] = Field(default=None, sa_type=Vector)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tags: Optional[list[str]] = Field(default=None, sa_type=JSON)