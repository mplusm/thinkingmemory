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
    # Dimension of stored embedding vectors. This MUST match the model used to
    # generate embeddings (e.g. 1536 for OpenAI text-embedding-3-small,
    # 3072 for text-embedding-3-large). It is fixed at the column level so that
    # pgvector can build HNSW indexes for fast similarity search.
    embedding_dim: int = 1536

    # Application settings
    app_name: str = "ThinkingMemory API"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    This function is cached to avoid re-reading environment variables on every call.
    The cache can be cleared by calling get_settings.cache_clear() if needed.

    For multi-tenant wrappers: Override this function to return your custom Settings subclass.
    """
    return Settings()
