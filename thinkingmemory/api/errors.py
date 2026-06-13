"""
Centralized error handling and logging for the API.

Routers no longer wrap every handler in a broad ``try/except`` that returns
``str(e)`` to the client (which leaks internal details and stack-trace content).
Instead, unhandled exceptions propagate to a single handler that logs the full
traceback server-side and returns a generic 500 to the caller. ``HTTPException``
(e.g. 404s) is left untouched and handled by FastAPI as usual.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("thinkingmemory")


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once, idempotently."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def install_error_handlers(app: FastAPI) -> None:
    """Install a catch-all handler that logs and returns a generic 500."""

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception(
            "Unhandled error on %s %s", request.method, request.url.path
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )


__all__ = ["configure_logging", "install_error_handlers", "logger"]
