"""
ThinkingMemory API - Main Application

This module provides the FastAPI application factory and default app instance.
It's designed to be extensible for wrapper applications (e.g., multi-tenant SaaS).

Usage for single-tenant (default):
    uvicorn src.api.main:app --reload

Usage for wrapper applications:
    from thinkingmemory.api.main import create_app, include_routers

    app = create_app(title="MyApp", version="1.0.0")
    # Add your middleware here
    include_routers(app)
    # Add your additional routes here
"""

from contextlib import asynccontextmanager
from typing import Optional, Callable, AsyncGenerator

from fastapi import FastAPI

from thinkingmemory.config.settings import get_settings
from thinkingmemory.core.database import init_db
from thinkingmemory.api.errors import configure_logging, install_error_handlers
from thinkingmemory.api.routers import working_router
from thinkingmemory.engine.router import router as memory_db_router


@asynccontextmanager
async def default_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Default lifespan context manager for the application.

    Initializes database tables on startup.
    Override this in wrapper applications if you need custom startup/shutdown logic.
    """
    # Startup
    init_db()
    yield
    # Shutdown (nothing to clean up by default)


def create_app(
    title: Optional[str] = None,
    description: Optional[str] = None,
    version: Optional[str] = None,
    lifespan: Optional[Callable] = None,
    **kwargs,
) -> FastAPI:
    """
    Create a FastAPI application instance.

    This factory function allows wrapper applications to customize the app
    before routers are included.

    Args:
        title: App title (defaults to settings.app_name)
        description: App description
        version: App version (defaults to settings.app_version)
        lifespan: Custom lifespan context manager
        **kwargs: Additional FastAPI constructor arguments

    Returns:
        FastAPI application instance (without routers - call include_routers separately)

    Example for wrapper:
        app = create_app(title="ThinkingMemory Cloud")
        app.add_middleware(AuthMiddleware)
        include_routers(app)
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=title or settings.app_name,
        description=description or "Agent-agnostic memory platform",
        version=version or settings.app_version,
        lifespan=lifespan or default_lifespan,
        **kwargs,
    )
    install_error_handlers(app)
    return app


def include_routers(app: FastAPI) -> None:
    """
    Include the memory routers in the application.

    The unified memory database lives under ``/v1`` (remember/recall/forget);
    working memory remains a separate Redis-backed TTL scratchpad under
    ``/working``. Call this after adding any middleware.

    Args:
        app: FastAPI application instance
    """
    app.include_router(memory_db_router)
    app.include_router(working_router)


def create_default_app() -> FastAPI:
    """
    Create the default single-tenant application with all routers included.

    This is the standard entry point for the open-source version.
    """
    app = create_app()

    @app.get("/")
    async def root():
        return {"message": "ThinkingMemory API is running!"}

    include_routers(app)
    return app


# Default application instance for single-tenant deployment
app = create_default_app()
