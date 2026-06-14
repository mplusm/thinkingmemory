"""
Centralized configuration using pydantic-settings.

This module provides a single source of truth for all configuration values.
The Settings class can be extended by wrapper applications (e.g., multi-tenant SaaS)
to add additional configuration options.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    This class is designed to be extensible. A multi-tenant wrapper can subclass
    this to add tenant-specific settings like TENANCY_MODE, rate limits, etc.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Allow extra env vars without validation errors
    )

    # Database settings
    database_url: str = "postgresql://memory_user:postgres@localhost:5432/thinkingmemory"

    # Redis settings
    redis_url: str = "redis://localhost:6379/0"

    # Embedding settings
    # The platform generates embeddings server-side via a pluggable provider.
    #   - "local"  : fastembed + BAAI/bge-small-en-v1.5 (384-dim, CPU, offline)
    #   - "openai" : text-embedding-3-small (1536-dim, requires OPENAI_API_KEY)
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Optional LLM for fact extraction, consolidation summaries, and
    # contradiction (NLI) judgments. "none" uses offline heuristics (no key
    # needed); "openai"/"anthropic" use the API when a key is set.
    llm_provider: str = "none"
    llm_model: str = ""
    # Dimension of stored embedding vectors. MUST match the provider/model above
    # (384 for bge-small, 1536 for text-embedding-3-small). Fixed at the column
    # level so pgvector can build HNSW indexes for fast similarity search.
    embedding_dim: int = 384

    # Cross-encoder reranking of recall candidates (local, CPU). Off by default
    # (adds latency + a model download); enable for higher recall precision.
    rerank_enabled: bool = False
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 30

    # Application settings
    app_name: str = "ThinkingMemory API"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    # Append-only audit logging of memory operations (enterprise auditability).
    audit_enabled: bool = True
    # Postgres Row-Level Security: when on, the DB enforces tenant isolation via
    # a per-session app.tenant_id GUC (defense-in-depth beneath app filtering).
    # Enable for multi-tenant deployments; run scripts/enable_rls.py once.
    rls_enabled: bool = False
    # Background lifecycle scheduler: periodically runs the maintenance cycle
    # (decay/consolidate/forget/...) for every active agent. Off by default.
    scheduler_enabled: bool = False
    scheduler_interval_minutes: int = 1440  # daily


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    This function is cached to avoid re-reading environment variables on every call.
    The cache can be cleared by calling get_settings.cache_clear() if needed.

    For multi-tenant wrappers: Override this function to return your custom Settings subclass.
    """
    return Settings()
