"""Shared-account persistence for concurrent strategy allocation.

Wallet rows are keyed by tenant and credential. Strategy reservations remain
separate so a shared OKX wallet is never mistaken for one strategy's account.
"""

from __future__ import annotations

import uuid

from sqlalchemy import Boolean, CheckConstraint, Column, DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.sql import func

from .base import Base


class StrategySharedAccount(Base):
    """One tenant-owned, exchange-backed shared wallet identity."""

    __tablename__ = "strategy_shared_accounts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False)
    credential_id = Column(String(36), ForeignKey("tenant_credentials.id", ondelete="RESTRICT"), nullable=False)
    environment = Column(String(16), nullable=False, default="okx_demo")
    wallet_equity_quote = Column(Float, nullable=True)
    available_quote = Column(Float, nullable=True)
    reserved_quote = Column(Float, nullable=False, default=0.0)
    occupied_notional_quote = Column(Float, nullable=False, default=0.0)
    pending_settlement_quote = Column(Float, nullable=False, default=0.0)
    reusable_quote = Column(Float, nullable=True)
    utilization_denominator_quote = Column(Float, nullable=True)
    sync_status = Column(String(16), nullable=False, default="unavailable")
    attribution_status = Column(String(16), nullable=False, default="unavailable")
    observed_at = Column(DateTime(timezone=True), nullable=True)
    version = Column(Integer, nullable=False, default=1)
    active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("tenant_id", "credential_id", "environment", name="uq_strategy_shared_account_scope"),
        CheckConstraint("reserved_quote >= 0", name="ck_strategy_shared_account_reserved"),
        CheckConstraint("occupied_notional_quote >= 0", name="ck_strategy_shared_account_occupied"),
        CheckConstraint("pending_settlement_quote >= 0", name="ck_strategy_shared_account_pending"),
        CheckConstraint("version >= 1", name="ck_strategy_shared_account_version"),
        Index("ix_strategy_shared_account_tenant_active", "tenant_id", "active"),
    )


class StrategyCapitalReservation(Base):
    """Atomic quote reservation belonging to one strategy intent."""

    __tablename__ = "strategy_capital_reservations"

    reservation_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id = Column(String(36), ForeignKey("strategy_shared_accounts.id", ondelete="RESTRICT"), nullable=False)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False)
    strategy_id = Column(String(100), ForeignKey("rule_strategies.strategy_id", ondelete="RESTRICT"), nullable=False)
    batch_id = Column(String(36), nullable=True, index=True)
    intent_id = Column(String(36), nullable=True, index=True)
    idempotency_key = Column(String(128), nullable=False)
    symbol = Column(String(32), nullable=False)
    side = Column(String(16), nullable=False)
    requested_quote = Column(Float, nullable=False)
    reserved_quote = Column(Float, nullable=False)
    consumed_quote = Column(Float, nullable=False, default=0.0)
    released_quote = Column(Float, nullable=False, default=0.0)
    status = Column(String(24), nullable=False, default="reserved", index=True)
    reason = Column(String(1_000), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("tenant_id", "idempotency_key", name="uq_strategy_reservation_idempotency"),
        CheckConstraint("requested_quote >= 0", name="ck_strategy_reservation_requested"),
        CheckConstraint("reserved_quote >= 0", name="ck_strategy_reservation_reserved"),
        CheckConstraint("consumed_quote >= 0", name="ck_strategy_reservation_consumed"),
        CheckConstraint("released_quote >= 0", name="ck_strategy_reservation_released"),
        CheckConstraint("consumed_quote + released_quote <= reserved_quote", name="ck_strategy_reservation_settled"),
        Index("ix_strategy_reservation_strategy_status", "strategy_id", "status"),
        Index("ix_strategy_reservation_account_status", "account_id", "status"),
    )
