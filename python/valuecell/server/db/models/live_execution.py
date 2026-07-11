"""Tenant-scoped durable controls and audit records for gated live CEX execution."""

from __future__ import annotations

import uuid

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, JSON, String, UniqueConstraint
from sqlalchemy.sql import func

from .base import Base


class LiveRiskPolicy(Base):
    """Explicit tenant risk envelope required before a live order is considered."""

    __tablename__ = "live_risk_policies"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    allowed_symbols = Column(JSON, nullable=False, default=list)
    max_order_notional = Column(String(32), nullable=False)
    max_open_positions = Column(String(16), nullable=False)
    max_leverage = Column(String(16), nullable=False)
    max_total_notional = Column(String(32), nullable=False, default="1", server_default="1")
    max_daily_loss = Column(String(32), nullable=False, default="1", server_default="1")
    active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class LiveStrategyBinding(Base):
    """Binds one deterministic strategy to one live connection and risk policy."""

    __tablename__ = "live_strategy_bindings"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    strategy_id = Column(String(100), ForeignKey("rule_strategies.strategy_id", ondelete="CASCADE"), nullable=False, index=True)
    connection_id = Column(String(36), ForeignKey("tenant_credentials.id", ondelete="RESTRICT"), nullable=False, index=True)
    risk_policy_id = Column(String(36), ForeignKey("live_risk_policies.id", ondelete="RESTRICT"), nullable=False)
    active = Column(Boolean, nullable=False, default=True, index=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("tenant_id", "strategy_id", "connection_id", name="uq_live_strategy_binding"),
    )


class LiveExecutionOrder(Base):
    """Idempotent audit record created before any gated live exchange submission."""

    __tablename__ = "live_execution_orders"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    binding_id = Column(String(36), ForeignKey("live_strategy_bindings.id", ondelete="RESTRICT"), nullable=False)
    connection_id = Column(String(36), ForeignKey("tenant_credentials.id", ondelete="RESTRICT"), nullable=False)
    client_order_id = Column(String(128), nullable=False)
    provider = Column(String(16), nullable=False)
    market_type = Column(String(8), nullable=False)
    symbol = Column(String(32), nullable=False)
    side = Column(String(8), nullable=False)
    order_type = Column(String(8), nullable=False)
    requested_quote = Column(String(32), nullable=False)
    requested_quantity = Column(String(32), nullable=True)
    status = Column(String(32), nullable=False, default="pending")
    exchange_order_id = Column(String(128), nullable=True)
    response_metadata = Column(JSON, nullable=False, default=dict)
    reject_code = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("tenant_id", "client_order_id", name="uq_live_execution_order_client_id"),
        Index("ix_live_execution_orders_tenant_client", "tenant_id", "client_order_id"),
    )
