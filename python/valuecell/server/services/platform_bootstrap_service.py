"""One-time deployment bootstrap for the first platform administrator."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from valuecell.server.api.auth import hash_password
from valuecell.server.db.models.tenant import SaaSUser, Tenant, TenantMembership


@dataclass(frozen=True)
class BootstrapResult:
    """Result of a safe, idempotent platform administrator bootstrap."""

    created: bool
    email: str | None


def bootstrap_platform_administrator(
    db: Session,
    email: str | None,
    password: str | None,
) -> BootstrapResult:
    """Create the deployment-defined administrator once; never reset an account."""

    if not email or not password:
        return BootstrapResult(created=False, email=None)
    canonical_email = email.strip().lower()
    existing = db.query(SaaSUser).filter(SaaSUser.email == canonical_email).first()
    if existing is not None:
        return BootstrapResult(created=False, email=canonical_email)

    user = SaaSUser(
        id=str(uuid.uuid4()),
        email=canonical_email,
        password_hash=hash_password(password),
    )
    tenant = Tenant(id=str(uuid.uuid4()), name="平台管理员工作区")
    membership = TenantMembership(tenant_id=tenant.id, user_id=user.id, role="owner")
    db.add_all([user, tenant, membership])
    db.commit()
    return BootstrapResult(created=True, email=canonical_email)
