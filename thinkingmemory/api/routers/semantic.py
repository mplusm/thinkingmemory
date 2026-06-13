"""
Semantic Memory API Router.

Handles storage, retrieval, and forgetting of semantic facts,
data sources, data tables, data columns, and knowledge entities.

Every endpoint resolves an optional tenant id from the ``X-Tenant-ID`` header and
passes it to the CRUD layer. Unhandled exceptions propagate to the
application-wide error handler; only domain 404s are raised explicitly.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from thinkingmemory.api.dependencies import get_tenant_id
from thinkingmemory.api.schemas import (
    FactStoreRequest,
    SimilarityRequest,
    DataSourceStoreRequest,
    DataSourceUpdateRequest,
    DataTableStoreRequest,
    DataTableUpdateRequest,
    DataColumnStoreRequest,
    DataColumnUpdateRequest,
    KnowledgeStoreRequest,
    KnowledgeUpdateRequest,
)
from thinkingmemory.memory.semantic.crud import (
    store_fact,
    retrieve_facts,
    forget_low_confidence_facts,
    retrieve_similar_facts,
    store_datasource,
    retrieve_datasources,
    retrieve_similar_datasources,
    update_datasource,
    delete_datasource,
    store_datatable,
    retrieve_datatables,
    retrieve_similar_datatables,
    retrieve_table_with_columns,
    update_datatable,
    delete_datatable,
    store_datacolumn,
    retrieve_datacolumns,
    retrieve_similar_datacolumns,
    retrieve_column_lineage,
    update_datacolumn,
    delete_datacolumn,
    store_knowledge,
    retrieve_knowledge,
    retrieve_similar_knowledge,
    update_knowledge,
    delete_knowledge,
    forget_low_confidence_knowledge,
)

router = APIRouter(prefix="/semantic", tags=["semantic"])


@router.post("/store")
async def store_fact_endpoint(
    request: FactStoreRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return store_fact(
        agent_id=request.agent_id,
        fact=request.fact,
        embedding=request.embedding,
        confidence=request.confidence,
        source=request.source,
        tenant_id=tenant_id,
    )


@router.get("/retrieve/{agent_id}")
async def retrieve_facts_endpoint(
    agent_id: str,
    limit: int = 10,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return retrieve_facts(agent_id, limit, tenant_id=tenant_id)


@router.post("/similar/{agent_id}")
async def retrieve_similar_facts_endpoint(
    agent_id: str,
    request: SimilarityRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return retrieve_similar_facts(
        agent_id=agent_id,
        embedding=request.embedding,
        limit=request.limit,
        similarity_threshold=request.similarity_threshold,
        tenant_id=tenant_id,
    )


@router.delete("/forget/low-confidence/{agent_id}")
async def forget_low_confidence_facts_endpoint(
    agent_id: str,
    confidence_threshold: float = 0.5,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    deleted_count = forget_low_confidence_facts(
        agent_id, confidence_threshold, tenant_id=tenant_id
    )
    return {
        "message": f"Deleted {deleted_count} low-confidence facts for agent {agent_id}"
    }


# =============================================================================
# DataSource Endpoints
# =============================================================================


@router.post("/sources/store")
async def store_datasource_endpoint(
    request: DataSourceStoreRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return store_datasource(
        agent_id=request.agent_id,
        source_name=request.source_name,
        source_type=request.source_type,
        description=request.description,
        connection_alias=request.connection_alias,
        embedding=request.embedding,
        metadata_=request.metadata_,
        tenant_id=tenant_id,
    )


@router.get("/sources/retrieve/{agent_id}")
async def retrieve_datasources_endpoint(
    agent_id: str,
    limit: int = 10,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return retrieve_datasources(agent_id, limit, tenant_id=tenant_id)


@router.post("/sources/similar/{agent_id}")
async def retrieve_similar_datasources_endpoint(
    agent_id: str,
    request: SimilarityRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return retrieve_similar_datasources(
        agent_id=agent_id,
        embedding=request.embedding,
        limit=request.limit,
        similarity_threshold=request.similarity_threshold,
        tenant_id=tenant_id,
    )


@router.patch("/sources/update/{datasource_id}")
async def update_datasource_endpoint(
    datasource_id: int,
    request: DataSourceUpdateRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    updates = request.model_dump(exclude_unset=True)
    result = update_datasource(datasource_id, tenant_id=tenant_id, **updates)
    if result is None:
        raise HTTPException(status_code=404, detail=f"DataSource {datasource_id} not found")
    return result


@router.delete("/sources/delete/{datasource_id}")
async def delete_datasource_endpoint(
    datasource_id: int,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    success = delete_datasource(datasource_id, tenant_id=tenant_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"DataSource {datasource_id} not found")
    return {"message": f"Deleted DataSource {datasource_id}"}


# =============================================================================
# DataTable Endpoints
# =============================================================================


@router.post("/tables/store")
async def store_datatable_endpoint(
    request: DataTableStoreRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return store_datatable(
        agent_id=request.agent_id,
        table_name=request.table_name,
        data_source_id=request.data_source_id,
        schema_name=request.schema_name,
        table_type=request.table_type,
        description=request.description,
        row_count_estimate=request.row_count_estimate,
        embedding=request.embedding,
        tags=request.tags,
        tenant_id=tenant_id,
    )


@router.get("/tables/retrieve/{agent_id}")
async def retrieve_datatables_endpoint(
    agent_id: str,
    limit: int = 10,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return retrieve_datatables(agent_id, limit, tenant_id=tenant_id)


@router.post("/tables/similar/{agent_id}")
async def retrieve_similar_datatables_endpoint(
    agent_id: str,
    request: SimilarityRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return retrieve_similar_datatables(
        agent_id=agent_id,
        embedding=request.embedding,
        limit=request.limit,
        similarity_threshold=request.similarity_threshold,
        tenant_id=tenant_id,
    )


@router.get("/tables/with-columns/{table_id}")
async def retrieve_table_with_columns_endpoint(
    table_id: int,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    result = retrieve_table_with_columns(table_id, tenant_id=tenant_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"DataTable {table_id} not found")
    return result


@router.patch("/tables/update/{datatable_id}")
async def update_datatable_endpoint(
    datatable_id: int,
    request: DataTableUpdateRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    updates = request.model_dump(exclude_unset=True)
    result = update_datatable(datatable_id, tenant_id=tenant_id, **updates)
    if result is None:
        raise HTTPException(status_code=404, detail=f"DataTable {datatable_id} not found")
    return result


@router.delete("/tables/delete/{datatable_id}")
async def delete_datatable_endpoint(
    datatable_id: int,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    success = delete_datatable(datatable_id, tenant_id=tenant_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"DataTable {datatable_id} not found")
    return {"message": f"Deleted DataTable {datatable_id}"}


# =============================================================================
# DataColumn Endpoints
# =============================================================================


@router.post("/columns/store")
async def store_datacolumn_endpoint(
    request: DataColumnStoreRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return store_datacolumn(
        agent_id=request.agent_id,
        column_name=request.column_name,
        data_table_id=request.data_table_id,
        data_type=request.data_type,
        is_nullable=request.is_nullable,
        is_primary_key=request.is_primary_key,
        is_foreign_key=request.is_foreign_key,
        foreign_key_ref=request.foreign_key_ref,
        description=request.description,
        sample_values=request.sample_values,
        lineage=request.lineage,
        embedding=request.embedding,
        tags=request.tags,
        tenant_id=tenant_id,
    )


@router.get("/columns/retrieve/{agent_id}")
async def retrieve_datacolumns_endpoint(
    agent_id: str,
    limit: int = 10,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return retrieve_datacolumns(agent_id, limit, tenant_id=tenant_id)


@router.post("/columns/similar/{agent_id}")
async def retrieve_similar_datacolumns_endpoint(
    agent_id: str,
    request: SimilarityRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return retrieve_similar_datacolumns(
        agent_id=agent_id,
        embedding=request.embedding,
        limit=request.limit,
        similarity_threshold=request.similarity_threshold,
        tenant_id=tenant_id,
    )


@router.get("/columns/lineage/{column_id}")
async def retrieve_column_lineage_endpoint(
    column_id: int,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    result = retrieve_column_lineage(column_id, tenant_id=tenant_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"DataColumn {column_id} not found")
    return result


@router.patch("/columns/update/{datacolumn_id}")
async def update_datacolumn_endpoint(
    datacolumn_id: int,
    request: DataColumnUpdateRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    updates = request.model_dump(exclude_unset=True)
    result = update_datacolumn(datacolumn_id, tenant_id=tenant_id, **updates)
    if result is None:
        raise HTTPException(status_code=404, detail=f"DataColumn {datacolumn_id} not found")
    return result


@router.delete("/columns/delete/{datacolumn_id}")
async def delete_datacolumn_endpoint(
    datacolumn_id: int,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    success = delete_datacolumn(datacolumn_id, tenant_id=tenant_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"DataColumn {datacolumn_id} not found")
    return {"message": f"Deleted DataColumn {datacolumn_id}"}


# =============================================================================
# KnowledgeEntity Endpoints
# =============================================================================


@router.post("/knowledge/store")
async def store_knowledge_endpoint(
    request: KnowledgeStoreRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return store_knowledge(
        agent_id=request.agent_id,
        entity_type=request.entity_type,
        name=request.name,
        description=request.description,
        properties=request.properties,
        relationships=request.relationships,
        source=request.source,
        confidence=request.confidence,
        embedding=request.embedding,
        tags=request.tags,
        tenant_id=tenant_id,
    )


@router.get("/knowledge/retrieve/{agent_id}")
async def retrieve_knowledge_endpoint(
    agent_id: str,
    entity_type: str = None,
    limit: int = 10,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return retrieve_knowledge(
        agent_id, entity_type=entity_type, limit=limit, tenant_id=tenant_id
    )


@router.post("/knowledge/similar/{agent_id}")
async def retrieve_similar_knowledge_endpoint(
    agent_id: str,
    request: SimilarityRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    return retrieve_similar_knowledge(
        agent_id=agent_id,
        embedding=request.embedding,
        limit=request.limit,
        similarity_threshold=request.similarity_threshold,
        tenant_id=tenant_id,
    )


@router.patch("/knowledge/update/{knowledge_id}")
async def update_knowledge_endpoint(
    knowledge_id: int,
    request: KnowledgeUpdateRequest,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    updates = request.model_dump(exclude_unset=True)
    result = update_knowledge(knowledge_id, tenant_id=tenant_id, **updates)
    if result is None:
        raise HTTPException(status_code=404, detail=f"KnowledgeEntity {knowledge_id} not found")
    return result


@router.delete("/knowledge/delete/{knowledge_id}")
async def delete_knowledge_endpoint(
    knowledge_id: int,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    success = delete_knowledge(knowledge_id, tenant_id=tenant_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"KnowledgeEntity {knowledge_id} not found")
    return {"message": f"Deleted KnowledgeEntity {knowledge_id}"}


@router.delete("/knowledge/forget/low-confidence/{agent_id}")
async def forget_low_confidence_knowledge_endpoint(
    agent_id: str,
    confidence_threshold: float = 0.5,
    tenant_id: Optional[str] = Depends(get_tenant_id),
):
    deleted_count = forget_low_confidence_knowledge(
        agent_id, confidence_threshold, tenant_id=tenant_id
    )
    return {
        "message": f"Deleted {deleted_count} low-confidence knowledge entities for agent {agent_id}"
    }
