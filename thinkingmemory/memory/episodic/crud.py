"""
Episodic Memory CRUD operations.

All functions accept an optional tenant_id parameter for multi-tenant deployments.
When tenant_id is None, no tenant filtering is applied (single-tenant mode).
"""

from datetime import datetime, timedelta
from typing import Optional

from sqlmodel import select, delete
from sqlalchemy import func
import numpy as np

from thinkingmemory.core.database import get_session_context
from thinkingmemory.memory.episodic.models import MemoryItem


def _apply_tenant_filter(statement, model, tenant_id: Optional[str]):
    """Apply tenant filter to a statement if tenant_id is provided."""
    if tenant_id is not None:
        return statement.where(model.tenant_id == tenant_id)
    return statement


def _memory_to_dict(memory: MemoryItem) -> dict:
    """Convert a MemoryItem to a serializable dict."""
    return {
        "id": memory.id,
        "tenant_id": memory.tenant_id,
        "agent_id": memory.agent_id,
        "memory_type": memory.memory_type,
        "content": memory.content,
        "embedding": list(memory.embedding) if memory.embedding else None,
        "timestamp": memory.timestamp,
        "extra_data": memory.extra_data,
        "access_count": memory.access_count,
        "last_accessed": memory.last_accessed,
        "is_compressed": memory.is_compressed,
        "source_memory_ids": memory.source_memory_ids,
    }


def store_memory(
    agent_id: str,
    content: dict,
    embedding: list[float] = None,
    extra_data: dict = None,
    tenant_id: Optional[str] = None,
):
    """Store a new episodic memory."""
    item = MemoryItem(
        agent_id=agent_id,
        content=content,
        embedding=embedding,
        extra_data=extra_data,
    )
    # Set tenant_id if provided (column added via migration)
    if tenant_id is not None:
        item.tenant_id = tenant_id

    with get_session_context() as session:
        session.add(item)
        session.commit()
        session.refresh(item)
        # Convert to dict before session closes
        return _memory_to_dict(item)


def retrieve_memories(
    agent_id: str,
    limit: int = 10,
    track_access: bool = True,
    tenant_id: Optional[str] = None,
):
    """Retrieve recent memories and optionally track access for relevance scoring."""
    with get_session_context() as session:
        statement = select(MemoryItem).where(MemoryItem.agent_id == agent_id)
        statement = _apply_tenant_filter(statement, MemoryItem, tenant_id)
        statement = statement.order_by(MemoryItem.timestamp.desc()).limit(limit)

        memories = session.exec(statement).all()

        if track_access:
            for memory in memories:
                memory.access_count += 1
                memory.last_accessed = datetime.utcnow()
            session.commit()

        # Convert to dicts before session closes to avoid DetachedInstanceError
        return [_memory_to_dict(m) for m in memories]


def forget_old_memories(agent_id: str, days: int = 30, tenant_id: Optional[str] = None):
    """Delete memories older than `days` for a specific agent."""
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    with get_session_context() as session:
        statement = delete(MemoryItem).where(
            MemoryItem.agent_id == agent_id,
            MemoryItem.timestamp < cutoff_date,
        )
        if tenant_id is not None:
            statement = statement.where(MemoryItem.tenant_id == tenant_id)

        result = session.exec(statement)
        session.commit()
        return result.rowcount


def forget_low_relevance_memories(
    agent_id: str,
    min_access_count: int = 1,
    days_since_access: int = 7,
    tenant_id: Optional[str] = None,
):
    """
    Delete memories with low relevance based on access patterns.

    Relevance is determined by:
    - access_count: How many times the memory has been retrieved
    - last_accessed: When the memory was last accessed

    Memories are deleted if:
    - access_count < min_access_count AND
    - last_accessed is None OR last_accessed > days_since_access ago
    """
    cutoff_date = datetime.utcnow() - timedelta(days=days_since_access)
    with get_session_context() as session:
        statement = delete(MemoryItem).where(
            MemoryItem.agent_id == agent_id,
            MemoryItem.access_count < min_access_count,
            (MemoryItem.last_accessed.is_(None)) | (MemoryItem.last_accessed < cutoff_date),
        )
        if tenant_id is not None:
            statement = statement.where(MemoryItem.tenant_id == tenant_id)

        result = session.exec(statement)
        session.commit()
        return result.rowcount


def retrieve_similar_memories(
    agent_id: str,
    embedding: list[float],
    limit: int = 10,
    similarity_threshold: float = 0.5,
    track_access: bool = True,
    tenant_id: Optional[str] = None,
):
    """Retrieve memories similar to the given embedding and track access."""
    with get_session_context() as session:
        statement = select(MemoryItem).where(
            MemoryItem.agent_id == agent_id,
            MemoryItem.embedding.isnot(None),
        )
        statement = _apply_tenant_filter(statement, MemoryItem, tenant_id)
        statement = statement.order_by(
            func.l2_distance(MemoryItem.embedding, embedding)
        ).limit(limit)

        memories = session.exec(statement).all()

        if track_access:
            for memory in memories:
                memory.access_count += 1
                memory.last_accessed = datetime.utcnow()
            session.commit()

        # Convert to dicts before session closes
        return [_memory_to_dict(m) for m in memories]


def compress_similar_memories(
    agent_id: str,
    similarity_threshold: float = 0.3,
    min_cluster_size: int = 3,
    tenant_id: Optional[str] = None,
):
    """
    Compress similar memories into consolidated entries.

    This function:
    1. Finds clusters of similar memories based on embedding distance
    2. Creates a compressed memory that represents the cluster
    3. Marks original memories as compressed (keeping them for audit trail)

    Returns the number of compressed memories created.
    """
    with get_session_context() as session:
        # Get all uncompressed memories with embeddings
        statement = select(MemoryItem).where(
            MemoryItem.agent_id == agent_id,
            MemoryItem.embedding.isnot(None),
            MemoryItem.is_compressed == False,
        )
        statement = _apply_tenant_filter(statement, MemoryItem, tenant_id)
        statement = statement.order_by(MemoryItem.timestamp.desc())

        memories = session.exec(statement).all()

        if len(memories) < min_cluster_size:
            return 0

        # Simple clustering: group memories by similarity
        clusters = []
        used_ids = set()

        for i, memory in enumerate(memories):
            if memory.id in used_ids:
                continue

            cluster = [memory]
            used_ids.add(memory.id)

            # Find similar memories
            for j, other_memory in enumerate(memories):
                if other_memory.id in used_ids:
                    continue

                # Calculate L2 distance between embeddings
                if memory.embedding and other_memory.embedding:
                    dist = np.linalg.norm(
                        np.array(list(memory.embedding)) - np.array(list(other_memory.embedding))
                    )
                    if dist < similarity_threshold:
                        cluster.append(other_memory)
                        used_ids.add(other_memory.id)

            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

        # Create compressed memories for each cluster
        compressed_count = 0
        for cluster in clusters:
            # Aggregate content from cluster
            aggregated_content = {
                "type": "compressed",
                "original_count": len(cluster),
                "summary": [m.content for m in cluster],
                "time_range": {
                    "start": min(m.timestamp for m in cluster).isoformat(),
                    "end": max(m.timestamp for m in cluster).isoformat(),
                },
            }

            # Average the embeddings
            embeddings = [list(m.embedding) for m in cluster if m.embedding]
            avg_embedding = np.mean(embeddings, axis=0).tolist() if embeddings else None

            # Sum access counts
            total_access = sum(m.access_count for m in cluster)

            # Create compressed memory
            compressed_memory = MemoryItem(
                agent_id=agent_id,
                memory_type="episodic_compressed",
                content=aggregated_content,
                embedding=avg_embedding,
                extra_data={"compression_date": datetime.utcnow().isoformat()},
                access_count=total_access,
                is_compressed=False,
                source_memory_ids=[m.id for m in cluster],
            )
            if tenant_id is not None:
                compressed_memory.tenant_id = tenant_id

            session.add(compressed_memory)

            # Mark original memories as compressed
            for memory in cluster:
                memory.is_compressed = True

            compressed_count += 1

        session.commit()
        return compressed_count


def delete_compressed_originals(agent_id: str, tenant_id: Optional[str] = None):
    """
    Delete original memories that have been compressed.
    Call this after compression to free up space.
    """
    with get_session_context() as session:
        statement = delete(MemoryItem).where(
            MemoryItem.agent_id == agent_id,
            MemoryItem.is_compressed == True,
        )
        if tenant_id is not None:
            statement = statement.where(MemoryItem.tenant_id == tenant_id)

        result = session.exec(statement)
        session.commit()
        return result.rowcount


def get_memory_stats(agent_id: str, tenant_id: Optional[str] = None):
    """Get statistics about an agent's memories."""
    with get_session_context() as session:
        # Base conditions
        base_cond = [MemoryItem.agent_id == agent_id]
        if tenant_id is not None:
            base_cond.append(MemoryItem.tenant_id == tenant_id)

        total = session.exec(
            select(func.count(MemoryItem.id)).where(*base_cond)
        ).one()

        compressed = session.exec(
            select(func.count(MemoryItem.id)).where(
                *base_cond,
                MemoryItem.is_compressed == True,
            )
        ).one()

        with_embeddings = session.exec(
            select(func.count(MemoryItem.id)).where(
                *base_cond,
                MemoryItem.embedding.isnot(None),
            )
        ).one()

        avg_access = session.exec(
            select(func.avg(MemoryItem.access_count)).where(*base_cond)
        ).one()

        return {
            "total_memories": total,
            "compressed_memories": compressed,
            "active_memories": total - compressed,
            "memories_with_embeddings": with_embeddings,
            "average_access_count": float(avg_access) if avg_access else 0.0,
        }
