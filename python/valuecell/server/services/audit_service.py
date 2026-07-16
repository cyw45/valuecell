"""Append-only audit logging for commercial and trading control-plane actions."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from valuecell.server.db.models.saas_control import AuditEvent


def record_audit_event(
    db: Session,
    *,
    action: str,
    target_type: str,
    target_id: str,
    outcome: str,
    tenant_id: str | None = None,
    actor_user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AuditEvent:
    """Persist one immutable event without accepting secret material."""

    event = AuditEvent(
        tenant_id=tenant_id,
        actor_user_id=actor_user_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        outcome=outcome,
        metadata_json=metadata or {},
    )
    db.add(event)
    return event
