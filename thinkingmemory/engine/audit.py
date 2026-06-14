"""
Audit logging — an append-only record of memory operations.

Every remember/recall/forget/maintenance call can be recorded for enterprise
auditability and (later) usage metering. Recording is best-effort: a failure to
write an audit row never breaks the underlying operation. Toggle with the
``audit_enabled`` setting.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlmodel import select, desc

from thinkingmemory.config.settings import get_settings
from thinkingmemory.core.database import get_session_context
from thinkingmemory.engine.models import AuditLog

logger = logging.getLogger("thinkingmemory.audit")


def record(
    action: str,
    agent_id: str,
    tenant_id: Optional[str] = None,
    target_id: Optional[int] = None,
    details: Optional[dict] = None,
) -> None:
    """Append an audit entry (best-effort; never raises)."""
    if not get_settings().audit_enabled:
        return
    try:
        with get_session_context(tenant_id) as session:
            session.add(
                AuditLog(
                    action=action,
                    agent_id=agent_id,
                    tenant_id=tenant_id or "default",
                    target_id=target_id,
                    details=details,
                )
            )
            session.commit()
    except Exception:  # pragma: no cover - audit must not break the operation
        logger.exception("Failed to write audit log for %s/%s", action, agent_id)


def query(
    agent_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """Return recent audit entries, newest first."""
    with get_session_context(tenant_id) as session:
        stmt = select(AuditLog)
        if agent_id is not None:
            stmt = stmt.where(AuditLog.agent_id == agent_id)
        if tenant_id is not None:
            stmt = stmt.where(AuditLog.tenant_id == tenant_id)
        if action is not None:
            stmt = stmt.where(AuditLog.action == action)
        stmt = stmt.order_by(desc(AuditLog.created_at)).limit(limit)
        return [
            {
                "id": a.id,
                "tenant_id": a.tenant_id,
                "agent_id": a.agent_id,
                "action": a.action,
                "target_id": a.target_id,
                "details": a.details,
                "created_at": a.created_at,
            }
            for a in session.exec(stmt).all()
        ]


__all__ = ["record", "query"]
