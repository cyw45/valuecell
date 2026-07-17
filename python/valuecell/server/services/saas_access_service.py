"""Authorization and commercial-access checks for tenant-scoped SaaS actions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from fastapi import HTTPException
from sqlalchemy.orm import Session

from valuecell.server.db.models.saas_control import (
    EnterpriseAgreement,
    ServicePlan,
    TenantSubscription,
)

TenantRole = Literal[
    "owner", "admin", "strategist", "trader", "viewer", "billing_manager"
]

_PERMISSION_ROLES: dict[str, frozenset[TenantRole]] = {
    "tenant.read": frozenset(
        {"owner", "admin", "strategist", "trader", "viewer", "billing_manager"}
    ),
    "strategy.manage": frozenset({"owner", "admin", "strategist"}),
    "trade.execute": frozenset({"owner", "admin", "trader"}),
    "connection.manage": frozenset({"owner", "admin", "trader"}),
    "member.manage": frozenset({"owner", "admin"}),
    "billing.manage": frozenset({"owner", "admin", "billing_manager"}),
}


@dataclass(frozen=True)
class TenantAccess:
    """Commercial-access result evaluated at authentication time."""

    status: str
    commercial_model: str | None
    expires_at: datetime | None
    entitlements: dict[str, object]

    @property
    def active(self) -> bool:
        return self.status == "active"


class TenantAccessService:
    """Evaluate subscriptions and enterprise agreements without trusting JWT claims."""

    @staticmethod
    def access_for(db: Session, tenant_id: str) -> TenantAccess:
        now = datetime.now(timezone.utc)
        subscription = (
            db.query(TenantSubscription)
            .filter(
                TenantSubscription.tenant_id == tenant_id,
                TenantSubscription.status == "active",
                TenantSubscription.starts_at <= now,
                TenantSubscription.ends_at > now,
            )
            .order_by(TenantSubscription.ends_at.desc())
            .first()
        )
        if subscription is not None:
            plan = (
                db.query(ServicePlan)
                .filter(ServicePlan.id == subscription.plan_id)
                .first()
            )
            return TenantAccess(
                status="active",
                commercial_model="subscription",
                expires_at=subscription.ends_at,
                entitlements=dict(plan.entitlements or {}) if plan is not None else {},
            )

        agreement = (
            db.query(EnterpriseAgreement)
            .filter(
                EnterpriseAgreement.tenant_id == tenant_id,
                EnterpriseAgreement.status == "active",
                EnterpriseAgreement.starts_at <= now,
            )
            .first()
        )
        if agreement is not None and (
            agreement.ends_at is None or agreement.ends_at > now
        ):
            return TenantAccess(
                status="active",
                commercial_model="revenue_share",
                expires_at=agreement.ends_at,
                entitlements={"commercial_model": "revenue_share"},
            )
        return TenantAccess(
            status="pending_activation",
            commercial_model=None,
            expires_at=None,
            entitlements={},
        )


def require_tenant_permission(principal: object, permission: str) -> None:
    """Reject a role that does not hold a tenant permission."""

    if getattr(principal, "is_platform_admin", False):
        return
    allowed_roles = _PERMISSION_ROLES[permission]
    if getattr(principal, "role", "viewer") not in allowed_roles:
        raise HTTPException(status_code=403, detail="当前角色没有执行此操作的权限")


def require_active_tenant(principal: object) -> None:
    """Block expired customer workspaces while preserving platform administration."""

    if getattr(principal, "is_platform_admin", False):
        return
    if getattr(principal, "access_status", "pending_activation") != "active":
        raise HTTPException(status_code=403, detail="工作区尚未开通或服务已到期")


def require_platform_admin(principal: object) -> None:
    """Reject platform-control actions from customer workspaces."""

    if not getattr(principal, "is_platform_admin", False):
        raise HTTPException(status_code=403, detail="需要平台管理员权限")
