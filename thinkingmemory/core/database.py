"""
Consolidated database module for ThinkingMemory.

This module provides a single source of truth for database connections,
replacing the duplicate database.py files in each memory module.

For multi-tenant wrappers: You can override get_session() dependency
to inject tenant context into queries.
"""

from typing import Generator
from contextlib import contextmanager

import redis
from sqlmodel import create_engine, Session, SQLModel

from thinkingmemory.config.settings import get_settings

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
    from thinkingmemory.memory.semantic.models import Fact  # noqa: F401
    from thinkingmemory.memory.procedural.models import Procedure  # noqa: F401

    SQLModel.metadata.create_all(get_engine())


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
    "get_session",
    "get_session_context",
]
