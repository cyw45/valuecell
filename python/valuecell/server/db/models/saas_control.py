"""Commercial tenancy, entitlement, settlement, and audit persistence models."""

from __future__ import annotations

import uuid

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from .base import Base


class ServicePlan(Base):
    """A platform-managed commercial plan with explicit tenant entitlements."""

    __tablename__ = "service_plans"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    code = Column(String(64), nullable=False, unique=True, index=True)
    name = Column(String(160), nullable=False)
    commercial_model = Column(String(32), nullable=False)
    duration_days = Column(Integer, nullable=True)
    price_cents = Column(Integer, nullable=True)
    currency = Column(String(8), nullable=False, default="CNY")
    entitlements = Column(JSON, nullable=False, default=dict)
    active = Column(String(16), nullable=False, default="active", index=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class TenantSubscription(Base):
    """Manual or external-payment subscription granting a tenant SaaS access."""

    __tablename__ = "tenant_subscriptions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(
        String(36),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    plan_id = Column(
        String(36), ForeignKey("service_plans.id", ondelete="RESTRICT"), nullable=False
    )
    status = Column(String(32), nullable=False, default="active", index=True)
    starts_at = Column(DateTime(timezone=True), nullable=False)
    ends_at = Column(DateTime(timezone=True), nullable=False, index=True)
    source = Column(String(32), nullable=False, default="manual")
    note = Column(String(1_000), nullable=True)
    granted_by_user_id = Column(
        String(36), ForeignKey("saas_users.id", ondelete="RESTRICT"), nullable=False
    )
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_tenant_subscriptions_access", "tenant_id", "status", "ends_at"),
    )


class EnterpriseAgreement(Base):
    """A signed revenue-share agreement that activates a business tenant."""

    __tablename__ = "enterprise_agreements"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(
        String(36),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    agreement_number = Column(String(100), nullable=False, unique=True, index=True)
    status = Column(String(32), nullable=False, default="active", index=True)
    revenue_share_rate = Column(String(16), nullable=False)
    settlement_cycle_days = Column(Integer, nullable=False, default=30)
    high_water_mark_quote = Column(String(32), nullable=False, default="0")
    starts_at = Column(DateTime(timezone=True), nullable=False)
    ends_at = Column(DateTime(timezone=True), nullable=True, index=True)
    note = Column(String(2_000), nullable=True)
    created_by_user_id = Column(
        String(36), ForeignKey("saas_users.id", ondelete="RESTRICT"), nullable=False
    )
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class ProfitSettlement(Base):
    """A reviewed account reconciliation and high-water-mark revenue-share statement."""

    __tablename__ = "profit_settlements"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(
        String(36),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    agreement_id = Column(
        String(36),
        ForeignKey("enterprise_agreements.id", ondelete="RESTRICT"),
        nullable=False,
    )
    connection_id = Column(
        String(36),
        ForeignKey("tenant_credentials.id", ondelete="RESTRICT"),
        nullable=False,
    )
    period_started_at = Column(DateTime(timezone=True), nullable=False)
    period_ended_at = Column(DateTime(timezone=True), nullable=False)
    ending_equity_quote = Column(String(32), nullable=False)
    net_external_cash_flow_quote = Column(String(32), nullable=False, default="0")
    gross_realized_pnl_quote = Column(String(32), nullable=False, default="0")
    fees_quote = Column(String(32), nullable=False, default="0")
    funding_quote = Column(String(32), nullable=False, default="0")
    high_water_mark_before_quote = Column(String(32), nullable=False)
    high_water_mark_after_quote = Column(String(32), nullable=False)
    eligible_profit_quote = Column(String(32), nullable=False)
    revenue_share_rate = Column(String(16), nullable=False)
    amount_due_quote = Column(String(32), nullable=False)
    status = Column(String(32), nullable=False, default="draft", index=True)
    reviewed_by_user_id = Column(
        String(36), ForeignKey("saas_users.id", ondelete="RESTRICT"), nullable=False
    )
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "agreement_id",
            "connection_id",
            "period_ended_at",
            name="uq_profit_settlement_period",
        ),
    )


class AuditEvent(Base):
    """Append-only platform and tenant audit event; this model has no update path."""

    __tablename__ = "audit_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(
        String(36),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    actor_user_id = Column(
        String(36),
        ForeignKey("saas_users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    action = Column(String(128), nullable=False, index=True)
    target_type = Column(String(64), nullable=False)
    target_id = Column(String(128), nullable=False)
    outcome = Column(String(32), nullable=False)
    metadata_json = Column(JSON, nullable=False, default=dict)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    __table_args__ = (
        Index("ix_audit_events_tenant_created", "tenant_id", "created_at"),
    )
