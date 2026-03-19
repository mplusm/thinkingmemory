"""
Semantic Memory CRUD operations.

All functions accept an optional tenant_id parameter for multi-tenant deployments.
When tenant_id is None, no tenant filtering is applied (single-tenant mode).
"""

from typing import Optional

from sqlmodel import select, delete
from sqlalchemy import func

from thinkingmemory.core.database import get_session_context
from thinkingmemory.memory.semantic.models import (
    Fact,
    DataSource,
    DataTable,
    DataColumn,
    KnowledgeEntity,
)


def _fact_to_dict(fact: Fact) -> dict:
    """Convert a Fact to a serializable dict."""
    return {
        "id": fact.id,
        "tenant_id": fact.tenant_id,
        "agent_id": fact.agent_id,
        "fact": fact.fact,
        "embedding": list(fact.embedding) if fact.embedding else None,
        "timestamp": fact.timestamp,
        "confidence": fact.confidence,
        "source": fact.source,
    }


def store_fact(
    agent_id: str,
    fact: str,
    embedding: list[float] = None,
    confidence: float = 1.0,
    source: str = None,
    tenant_id: Optional[str] = None,
):
    """Store a new semantic fact."""
    fact_item = Fact(
        agent_id=agent_id,
        fact=fact,
        embedding=embedding,
        confidence=confidence,
        source=source,
    )
    if tenant_id is not None:
        fact_item.tenant_id = tenant_id

    with get_session_context() as session:
        session.add(fact_item)
        session.commit()
        session.refresh(fact_item)
        # Convert to dict before session closes
        return _fact_to_dict(fact_item)


def retrieve_facts(agent_id: str, limit: int = 10, tenant_id: Optional[str] = None):
    """Retrieve facts for an agent."""
    with get_session_context() as session:
        statement = select(Fact).where(Fact.agent_id == agent_id)
        if tenant_id is not None:
            statement = statement.where(Fact.tenant_id == tenant_id)
        statement = statement.limit(limit)
        facts = session.exec(statement).all()
        # Convert to dicts before session closes
        return [_fact_to_dict(f) for f in facts]


def forget_low_confidence_facts(
    agent_id: str,
    confidence_threshold: float = 0.5,
    tenant_id: Optional[str] = None,
):
    """Delete facts with confidence below the threshold."""
    with get_session_context() as session:
        statement = delete(Fact).where(
            Fact.agent_id == agent_id,
            Fact.confidence < confidence_threshold,
        )
        if tenant_id is not None:
            statement = statement.where(Fact.tenant_id == tenant_id)

        result = session.exec(statement)
        session.commit()
        return result.rowcount


def retrieve_similar_facts(
    agent_id: str,
    embedding: list[float],
    limit: int = 10,
    similarity_threshold: float = 0.5,
    tenant_id: Optional[str] = None,
):
    """Retrieve facts similar to the given embedding."""
    with get_session_context() as session:
        statement = select(Fact).where(
            Fact.agent_id == agent_id,
            Fact.embedding.isnot(None),
        )
        if tenant_id is not None:
            statement = statement.where(Fact.tenant_id == tenant_id)

        statement = statement.order_by(
            func.l2_distance(Fact.embedding, embedding)
        ).limit(limit)
        facts = session.exec(statement).all()
        # Convert to dicts before session closes
        return [_fact_to_dict(f) for f in facts]


# =============================================================================
# DataSource CRUD
# =============================================================================


def _datasource_to_dict(ds: DataSource) -> dict:
    """Convert a DataSource to a serializable dict."""
    return {
        "id": ds.id,
        "tenant_id": ds.tenant_id,
        "agent_id": ds.agent_id,
        "source_name": ds.source_name,
        "source_type": ds.source_type,
        "description": ds.description,
        "connection_alias": ds.connection_alias,
        "embedding": list(ds.embedding) if ds.embedding else None,
        "timestamp": ds.timestamp,
        "metadata_": ds.metadata_,
    }


def store_datasource(
    agent_id: str,
    source_name: str,
    source_type: str = None,
    description: str = None,
    connection_alias: str = None,
    embedding: list[float] = None,
    metadata_: dict = None,
    tenant_id: Optional[str] = None,
):
    """Store a new data source."""
    item = DataSource(
        agent_id=agent_id,
        source_name=source_name,
        source_type=source_type,
        description=description,
        connection_alias=connection_alias,
        embedding=embedding,
        metadata_=metadata_,
    )
    if tenant_id is not None:
        item.tenant_id = tenant_id

    with get_session_context() as session:
        session.add(item)
        session.commit()
        session.refresh(item)
        return _datasource_to_dict(item)


def retrieve_datasources(agent_id: str, limit: int = 10, tenant_id: Optional[str] = None):
    """Retrieve data sources for an agent."""
    with get_session_context() as session:
        statement = select(DataSource).where(DataSource.agent_id == agent_id)
        if tenant_id is not None:
            statement = statement.where(DataSource.tenant_id == tenant_id)
        statement = statement.limit(limit)
        items = session.exec(statement).all()
        return [_datasource_to_dict(i) for i in items]


def retrieve_similar_datasources(
    agent_id: str,
    embedding: list[float],
    limit: int = 10,
    similarity_threshold: float = 0.5,
    tenant_id: Optional[str] = None,
):
    """Retrieve data sources similar to the given embedding."""
    with get_session_context() as session:
        statement = select(DataSource).where(
            DataSource.agent_id == agent_id,
            DataSource.embedding.isnot(None),
        )
        if tenant_id is not None:
            statement = statement.where(DataSource.tenant_id == tenant_id)
        statement = statement.order_by(
            func.l2_distance(DataSource.embedding, embedding)
        ).limit(limit)
        items = session.exec(statement).all()
        return [_datasource_to_dict(i) for i in items]


def update_datasource(
    datasource_id: int,
    tenant_id: Optional[str] = None,
    **kwargs,
):
    """Update a data source's fields."""
    with get_session_context() as session:
        item = session.get(DataSource, datasource_id)
        if item is None:
            return None
        if tenant_id is not None and item.tenant_id != tenant_id:
            return None
        for key, value in kwargs.items():
            if hasattr(item, key):
                setattr(item, key, value)
        session.commit()
        session.refresh(item)
        return _datasource_to_dict(item)


def delete_datasource(datasource_id: int, tenant_id: Optional[str] = None):
    """Delete a data source by ID."""
    with get_session_context() as session:
        item = session.get(DataSource, datasource_id)
        if item is None:
            return False
        if tenant_id is not None and item.tenant_id != tenant_id:
            return False
        session.delete(item)
        session.commit()
        return True


# =============================================================================
# DataTable CRUD
# =============================================================================


def _datatable_to_dict(dt: DataTable) -> dict:
    """Convert a DataTable to a serializable dict."""
    return {
        "id": dt.id,
        "tenant_id": dt.tenant_id,
        "agent_id": dt.agent_id,
        "data_source_id": dt.data_source_id,
        "schema_name": dt.schema_name,
        "table_name": dt.table_name,
        "table_type": dt.table_type,
        "description": dt.description,
        "row_count_estimate": dt.row_count_estimate,
        "embedding": list(dt.embedding) if dt.embedding else None,
        "timestamp": dt.timestamp,
        "tags": dt.tags,
    }


def store_datatable(
    agent_id: str,
    table_name: str,
    data_source_id: int = None,
    schema_name: str = None,
    table_type: str = None,
    description: str = None,
    row_count_estimate: int = None,
    embedding: list[float] = None,
    tags: list[str] = None,
    tenant_id: Optional[str] = None,
):
    """Store a new data table."""
    item = DataTable(
        agent_id=agent_id,
        table_name=table_name,
        data_source_id=data_source_id,
        schema_name=schema_name,
        table_type=table_type,
        description=description,
        row_count_estimate=row_count_estimate,
        embedding=embedding,
        tags=tags,
    )
    if tenant_id is not None:
        item.tenant_id = tenant_id

    with get_session_context() as session:
        session.add(item)
        session.commit()
        session.refresh(item)
        return _datatable_to_dict(item)


def retrieve_datatables(agent_id: str, limit: int = 10, tenant_id: Optional[str] = None):
    """Retrieve data tables for an agent."""
    with get_session_context() as session:
        statement = select(DataTable).where(DataTable.agent_id == agent_id)
        if tenant_id is not None:
            statement = statement.where(DataTable.tenant_id == tenant_id)
        statement = statement.limit(limit)
        items = session.exec(statement).all()
        return [_datatable_to_dict(i) for i in items]


def retrieve_similar_datatables(
    agent_id: str,
    embedding: list[float],
    limit: int = 10,
    similarity_threshold: float = 0.5,
    tenant_id: Optional[str] = None,
):
    """Retrieve data tables similar to the given embedding."""
    with get_session_context() as session:
        statement = select(DataTable).where(
            DataTable.agent_id == agent_id,
            DataTable.embedding.isnot(None),
        )
        if tenant_id is not None:
            statement = statement.where(DataTable.tenant_id == tenant_id)
        statement = statement.order_by(
            func.l2_distance(DataTable.embedding, embedding)
        ).limit(limit)
        items = session.exec(statement).all()
        return [_datatable_to_dict(i) for i in items]


def retrieve_table_with_columns(
    table_id: int,
    tenant_id: Optional[str] = None,
):
    """Retrieve a table and all its columns."""
    with get_session_context() as session:
        table = session.get(DataTable, table_id)
        if table is None:
            return None
        if tenant_id is not None and table.tenant_id != tenant_id:
            return None

        statement = select(DataColumn).where(DataColumn.data_table_id == table_id)
        if tenant_id is not None:
            statement = statement.where(DataColumn.tenant_id == tenant_id)
        columns = session.exec(statement).all()

        return {
            "table": _datatable_to_dict(table),
            "columns": [_datacolumn_to_dict(c) for c in columns],
        }


def update_datatable(
    datatable_id: int,
    tenant_id: Optional[str] = None,
    **kwargs,
):
    """Update a data table's fields."""
    with get_session_context() as session:
        item = session.get(DataTable, datatable_id)
        if item is None:
            return None
        if tenant_id is not None and item.tenant_id != tenant_id:
            return None
        for key, value in kwargs.items():
            if hasattr(item, key):
                setattr(item, key, value)
        session.commit()
        session.refresh(item)
        return _datatable_to_dict(item)


def delete_datatable(datatable_id: int, tenant_id: Optional[str] = None):
    """Delete a data table by ID."""
    with get_session_context() as session:
        item = session.get(DataTable, datatable_id)
        if item is None:
            return False
        if tenant_id is not None and item.tenant_id != tenant_id:
            return False
        session.delete(item)
        session.commit()
        return True


# =============================================================================
# DataColumn CRUD
# =============================================================================


def _datacolumn_to_dict(dc: DataColumn) -> dict:
    """Convert a DataColumn to a serializable dict."""
    return {
        "id": dc.id,
        "tenant_id": dc.tenant_id,
        "agent_id": dc.agent_id,
        "data_table_id": dc.data_table_id,
        "column_name": dc.column_name,
        "data_type": dc.data_type,
        "is_nullable": dc.is_nullable,
        "is_primary_key": dc.is_primary_key,
        "is_foreign_key": dc.is_foreign_key,
        "foreign_key_ref": dc.foreign_key_ref,
        "description": dc.description,
        "sample_values": dc.sample_values,
        "lineage": dc.lineage,
        "embedding": list(dc.embedding) if dc.embedding else None,
        "timestamp": dc.timestamp,
        "tags": dc.tags,
    }


def store_datacolumn(
    agent_id: str,
    column_name: str,
    data_table_id: int = None,
    data_type: str = None,
    is_nullable: bool = None,
    is_primary_key: bool = None,
    is_foreign_key: bool = None,
    foreign_key_ref: str = None,
    description: str = None,
    sample_values: list[str] = None,
    lineage: dict = None,
    embedding: list[float] = None,
    tags: list[str] = None,
    tenant_id: Optional[str] = None,
):
    """Store a new data column."""
    item = DataColumn(
        agent_id=agent_id,
        column_name=column_name,
        data_table_id=data_table_id,
        data_type=data_type,
        is_nullable=is_nullable,
        is_primary_key=is_primary_key,
        is_foreign_key=is_foreign_key,
        foreign_key_ref=foreign_key_ref,
        description=description,
        sample_values=sample_values,
        lineage=lineage,
        embedding=embedding,
        tags=tags,
    )
    if tenant_id is not None:
        item.tenant_id = tenant_id

    with get_session_context() as session:
        session.add(item)
        session.commit()
        session.refresh(item)
        return _datacolumn_to_dict(item)


def retrieve_datacolumns(agent_id: str, limit: int = 10, tenant_id: Optional[str] = None):
    """Retrieve data columns for an agent."""
    with get_session_context() as session:
        statement = select(DataColumn).where(DataColumn.agent_id == agent_id)
        if tenant_id is not None:
            statement = statement.where(DataColumn.tenant_id == tenant_id)
        statement = statement.limit(limit)
        items = session.exec(statement).all()
        return [_datacolumn_to_dict(i) for i in items]


def retrieve_similar_datacolumns(
    agent_id: str,
    embedding: list[float],
    limit: int = 10,
    similarity_threshold: float = 0.5,
    tenant_id: Optional[str] = None,
):
    """Retrieve data columns similar to the given embedding."""
    with get_session_context() as session:
        statement = select(DataColumn).where(
            DataColumn.agent_id == agent_id,
            DataColumn.embedding.isnot(None),
        )
        if tenant_id is not None:
            statement = statement.where(DataColumn.tenant_id == tenant_id)
        statement = statement.order_by(
            func.l2_distance(DataColumn.embedding, embedding)
        ).limit(limit)
        items = session.exec(statement).all()
        return [_datacolumn_to_dict(i) for i in items]


def retrieve_column_lineage(
    column_id: int,
    tenant_id: Optional[str] = None,
):
    """Retrieve lineage info for a specific column."""
    with get_session_context() as session:
        item = session.get(DataColumn, column_id)
        if item is None:
            return None
        if tenant_id is not None and item.tenant_id != tenant_id:
            return None
        return {
            "column_id": item.id,
            "column_name": item.column_name,
            "data_table_id": item.data_table_id,
            "lineage": item.lineage,
        }


def update_datacolumn(
    datacolumn_id: int,
    tenant_id: Optional[str] = None,
    **kwargs,
):
    """Update a data column's fields."""
    with get_session_context() as session:
        item = session.get(DataColumn, datacolumn_id)
        if item is None:
            return None
        if tenant_id is not None and item.tenant_id != tenant_id:
            return None
        for key, value in kwargs.items():
            if hasattr(item, key):
                setattr(item, key, value)
        session.commit()
        session.refresh(item)
        return _datacolumn_to_dict(item)


def delete_datacolumn(datacolumn_id: int, tenant_id: Optional[str] = None):
    """Delete a data column by ID."""
    with get_session_context() as session:
        item = session.get(DataColumn, datacolumn_id)
        if item is None:
            return False
        if tenant_id is not None and item.tenant_id != tenant_id:
            return False
        session.delete(item)
        session.commit()
        return True


# =============================================================================
# KnowledgeEntity CRUD
# =============================================================================


def _knowledge_to_dict(ke: KnowledgeEntity) -> dict:
    """Convert a KnowledgeEntity to a serializable dict."""
    return {
        "id": ke.id,
        "tenant_id": ke.tenant_id,
        "agent_id": ke.agent_id,
        "entity_type": ke.entity_type,
        "name": ke.name,
        "description": ke.description,
        "properties": ke.properties,
        "relationships": ke.relationships,
        "source": ke.source,
        "confidence": ke.confidence,
        "embedding": list(ke.embedding) if ke.embedding else None,
        "timestamp": ke.timestamp,
        "tags": ke.tags,
    }


def store_knowledge(
    agent_id: str,
    entity_type: str,
    name: str,
    description: str = None,
    properties: dict = None,
    relationships: list[dict] = None,
    source: str = None,
    confidence: float = 1.0,
    embedding: list[float] = None,
    tags: list[str] = None,
    tenant_id: Optional[str] = None,
):
    """Store a new knowledge entity."""
    item = KnowledgeEntity(
        agent_id=agent_id,
        entity_type=entity_type,
        name=name,
        description=description,
        properties=properties,
        relationships=relationships,
        source=source,
        confidence=confidence,
        embedding=embedding,
        tags=tags,
    )
    if tenant_id is not None:
        item.tenant_id = tenant_id

    with get_session_context() as session:
        session.add(item)
        session.commit()
        session.refresh(item)
        return _knowledge_to_dict(item)


def retrieve_knowledge(
    agent_id: str,
    entity_type: str = None,
    limit: int = 10,
    tenant_id: Optional[str] = None,
):
    """Retrieve knowledge entities for an agent, optionally filtered by type."""
    with get_session_context() as session:
        statement = select(KnowledgeEntity).where(KnowledgeEntity.agent_id == agent_id)
        if entity_type is not None:
            statement = statement.where(KnowledgeEntity.entity_type == entity_type)
        if tenant_id is not None:
            statement = statement.where(KnowledgeEntity.tenant_id == tenant_id)
        statement = statement.limit(limit)
        items = session.exec(statement).all()
        return [_knowledge_to_dict(i) for i in items]


def retrieve_similar_knowledge(
    agent_id: str,
    embedding: list[float],
    limit: int = 10,
    similarity_threshold: float = 0.5,
    tenant_id: Optional[str] = None,
):
    """Retrieve knowledge entities similar to the given embedding."""
    with get_session_context() as session:
        statement = select(KnowledgeEntity).where(
            KnowledgeEntity.agent_id == agent_id,
            KnowledgeEntity.embedding.isnot(None),
        )
        if tenant_id is not None:
            statement = statement.where(KnowledgeEntity.tenant_id == tenant_id)
        statement = statement.order_by(
            func.l2_distance(KnowledgeEntity.embedding, embedding)
        ).limit(limit)
        items = session.exec(statement).all()
        return [_knowledge_to_dict(i) for i in items]


def update_knowledge(
    knowledge_id: int,
    tenant_id: Optional[str] = None,
    **kwargs,
):
    """Update a knowledge entity's fields."""
    with get_session_context() as session:
        item = session.get(KnowledgeEntity, knowledge_id)
        if item is None:
            return None
        if tenant_id is not None and item.tenant_id != tenant_id:
            return None
        for key, value in kwargs.items():
            if hasattr(item, key):
                setattr(item, key, value)
        session.commit()
        session.refresh(item)
        return _knowledge_to_dict(item)


def delete_knowledge(knowledge_id: int, tenant_id: Optional[str] = None):
    """Delete a knowledge entity by ID."""
    with get_session_context() as session:
        item = session.get(KnowledgeEntity, knowledge_id)
        if item is None:
            return False
        if tenant_id is not None and item.tenant_id != tenant_id:
            return False
        session.delete(item)
        session.commit()
        return True


def forget_low_confidence_knowledge(
    agent_id: str,
    confidence_threshold: float = 0.5,
    tenant_id: Optional[str] = None,
):
    """Delete knowledge entities with confidence below the threshold."""
    with get_session_context() as session:
        statement = delete(KnowledgeEntity).where(
            KnowledgeEntity.agent_id == agent_id,
            KnowledgeEntity.confidence < confidence_threshold,
        )
        if tenant_id is not None:
            statement = statement.where(KnowledgeEntity.tenant_id == tenant_id)
        result = session.exec(statement)
        session.commit()
        return result.rowcount
