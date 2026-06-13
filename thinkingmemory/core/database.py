"""
Consolidated database module for ThinkingMemory.

This module provides a single source of truth for database connections,
replacing the duplicate database.py files in each memory module.

For multi-tenant wrappers: You can override get_session() dependency
to inject tenant context into queries.
"""

import logging
from typing import Generator
from contextlib import contextmanager

import redis
from sqlmodel import create_engine, Session, SQLModel
from sqlalchemy import text

from thinkingmemory.config.settings import get_settings

logger = logging.getLogger(__name__)

# Tables that carry an `embedding vector(N)` column we want an HNSW index on.
# Index name -> table name. Using L2 distance (vector_l2_ops) to match the
# func.l2_distance ordering used throughout the CRUD layer.
_VECTOR_TABLES = {
    "idx_memoryitem_embedding_hnsw": "memoryitem",
    "idx_fact_embedding_hnsw": "fact",
    "idx_procedure_embedding_hnsw": "procedure",
    "idx_datasource_embedding_hnsw": "datasource",
    "idx_datatable_embedding_hnsw": "datatable",
    "idx_datacolumn_embedding_hnsw": "datacolumn",
    "idx_knowledgeentity_embedding_hnsw": "knowledgeentity",
    "idx_userpreference_embedding_hnsw": "userpreference",
    "idx_workflowhabit_embedding_hnsw": "workflowhabit",
}

# Lazy initialization - engine created on first use
_engine = None
_redis_client = None


def get_engine():
    """
    Get or create the SQLAlchemy engine.

    The engine is created lazily and cached for the lifetime of the application.
    """
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(settings.database_url, echo=settings.debug)
    return _engine


def get_redis() -> redis.Redis:
    """
    Get or create the Redis client.

    The client is created lazily and cached for the lifetime of the application.
    """
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = redis.from_url(settings.redis_url)
    return _redis_client


def init_db():
    """
    Initialize all database tables.

    This creates all tables defined in SQLModel models (MemoryItem, Fact, Procedure).
    Safe to call multiple times - existing tables won't be modified.
    """
    # Import models here to avoid circular imports
    # These imports register the models with SQLModel.metadata
    from thinkingmemory.memory.episodic.models import MemoryItem  # noqa: F401
    from thinkingmemory.memory.semantic.models import Fact, DataSource, DataTable, DataColumn, KnowledgeEntity  # noqa: F401
    from thinkingmemory.memory.procedural.models import Procedure, UserPreference, WorkflowHabit  # noqa: F401

    engine = get_engine()

    # Ensure the pgvector extension exists before creating vector columns.
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    SQLModel.metadata.create_all(engine)

    create_vector_indexes()


def create_vector_indexes() -> None:
    """
    Create HNSW indexes on every embedding column for fast similarity search.

    Without these, `ORDER BY embedding <-> :q` falls back to a sequential scan.
    Safe to call repeatedly (IF NOT EXISTS). Index creation is best-effort: a
    failure on one table (e.g. extension missing) is logged but does not abort
    startup.
    """
    engine = get_engine()
    for index_name, table_name in _VECTOR_TABLES.items():
        stmt = text(
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON {table_name} USING hnsw (embedding vector_l2_ops)"
        )
        try:
            with engine.begin() as conn:
                conn.execute(stmt)
        except Exception as exc:  # pragma: no cover - depends on live DB state
            logger.warning(
                "Could not create vector index %s on %s: %s",
                index_name,
                table_name,
                exc,
            )


def migrate_vector_columns(dim: int | None = None) -> None:
    """
    Migrate existing unsized ``vector`` embedding columns to ``vector(dim)``.

    Older deployments created the embedding columns without a fixed dimension,
    which prevents building an HNSW index. This re-types each embedding column
    to the configured dimension and then (re)creates the indexes.

    Only columns that are currently *unsized* (atttypmod = -1) are altered, so
    already-sized columns are left untouched. Altering is only safe when the
    column has no rows of a different dimension; rows with NULL embeddings (the
    common case) are unaffected.

    Run this once against an existing database after upgrading:
        python scripts/migrate_vector_dims.py
    """
    target_dim = dim or get_settings().embedding_dim
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        for table_name in set(_VECTOR_TABLES.values()):
            # atttypmod == -1 means the vector column has no declared dimension.
            typmod = conn.execute(
                text(
                    "SELECT a.atttypmod FROM pg_attribute a "
                    "JOIN pg_class c ON a.attrelid = c.oid "
                    "WHERE c.relname = :t AND a.attname = 'embedding'"
                ),
                {"t": table_name},
            ).scalar()
            if typmod is None:
                continue  # table or column doesn't exist
            if typmod == -1:
                logger.info("Sizing %s.embedding to vector(%s)", table_name, target_dim)
                conn.execute(
                    text(
                        f"ALTER TABLE {table_name} "
                        f"ALTER COLUMN embedding TYPE vector({target_dim})"
                    )
                )

    create_vector_indexes()


def get_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session.

    Usage in FastAPI endpoints:
        @app.get("/items")
        def get_items(session: Session = Depends(get_session)):
            ...

    For multi-tenant wrappers: Override this dependency to inject
    tenant context or add query filters.
    """
    with Session(get_engine()) as session:
        yield session


@contextmanager
def get_session_context() -> Generator[Session, None, None]:
    """
    Context manager for getting a database session.

    Usage in non-FastAPI code (scripts, CRUD functions):
        with get_session_context() as session:
            session.exec(...)

    This replaces the pattern: with next(get_session()) as session
    """
    with Session(get_engine()) as session:
        yield session


__all__ = [
    "get_engine",
    "get_redis",
    "init_db",
    "create_vector_indexes",
    "migrate_vector_columns",
    "get_session",
    "get_session_context",
]
