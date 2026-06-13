"""
Shared FastAPI dependencies.

The tenant dependency resolves the active tenant from the ``X-Tenant-ID`` request
header. When the header is absent the value is ``None`` which the CRUD layer
treats as single-tenant mode (no tenant filtering). Multi-tenant wrappers can
override ``get_tenant_id`` (e.g. to derive the tenant from an authenticated API
key) via FastAPI dependency overrides without touching the routers.
"""

from typing import Optional

from fastapi import Header

# Header carrying the tenant identifier for multi-tenant deployments.
TENANT_HEADER = "X-Tenant-ID"


async def get_tenant_id(
    x_tenant_id: Optional[str] = Header(default=None, alias=TENANT_HEADER),
) -> Optional[str]:
    """Resolve the active tenant id from the request, or None for single-tenant."""
    return x_tenant_id


__all__ = ["get_tenant_id", "TENANT_HEADER"]
