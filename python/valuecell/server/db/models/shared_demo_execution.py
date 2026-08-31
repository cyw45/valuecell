"""Account-scoped, append-only evidence for shared OKX Demo execution.

These tables deliberately form a second, Demo-only evidence chain. They do
not read from or write to Paper accounts, fills, or PnL tables.
"""

from __future__ import annotations

import uuid

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    UniqueConstraint,
    event,
)
from sqlalchemy.sql import func

from .base import Base


_DEMO_ENVIRONMENT = "okx_demo"
_DEMO_VENUE = "okx"
_DEMO_DECIMAL = Numeric(38, 18)


class SharedDemoAccountSnapshot(Base):
    """Immutable exchange-authoritative wallet observation for one Demo account."""

    __tablename__ = "shared_demo_account_snapshots"

    snapshot_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id = Column(String(36), nullable=False)
    tenant_id = Column(String(36), nullable=False)
    credential_id = Column(String(36), nullable=False)
    environment = Column(String(16), nullable=False, default=_DEMO_ENVIRONMENT)
    source = Column(String(32), nullable=False, default="okx_account_sync")
    observed_at = Column(DateTime(timezone=True), nullable=False, index=True)
    wallet_equity_quote = Column(_DEMO_DECIMAL, nullable=True)
    available_quote = Column(_DEMO_DECIMAL, nullable=True)
    margin_used_quote = Column(_DEMO_DECIMAL, nullable=True)
    balances = Column(JSON, nullable=False, default=list)
    positions = Column(JSON, nullable=False, default=list)
    open_orders = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["account_id", "tenant_id", "credential_id", "environment"],
            [
                "strategy_shared_accounts.id",
                "strategy_shared_accounts.tenant_id",
                "strategy_shared_accounts.credential_id",
                "strategy_shared_accounts.environment",
            ],
            name="fk_shared_demo_snapshot_account_scope",
            ondelete="RESTRICT",
        ),
        CheckConstraint("environment = 'okx_demo'", name="ck_shared_demo_snapshot_environment"),
        CheckConstraint(
            "wallet_equity_quote IS NULL OR wallet_equity_quote >= 0",
            name="ck_shared_demo_snapshot_equity",
        ),
        CheckConstraint(
            "available_quote IS NULL OR available_quote >= 0",
            name="ck_shared_demo_snapshot_available",
        ),
        CheckConstraint(
            "margin_used_quote IS NULL OR margin_used_quote >= 0",
            name="ck_shared_demo_snapshot_margin_used",
        ),
        UniqueConstraint("account_id", "observed_at", name="uq_shared_demo_snapshot_account_observed"),
        Index("ix_shared_demo_snapshot_account_observed", "account_id", observed_at.desc()),
    )


class SharedDemoAccountSyncState(Base):
    """Mutable sync and reconciliation checkpoint for one Demo account scope."""

    __tablename__ = "shared_demo_account_sync_states"

    account_id = Column(String(36), primary_key=True)
    tenant_id = Column(String(36), nullable=False)
    credential_id = Column(String(36), nullable=False)
    environment = Column(String(16), nullable=False, default=_DEMO_ENVIRONMENT)
    latest_snapshot_id = Column(
        String(36),
        ForeignKey("shared_demo_account_snapshots.snapshot_id", ondelete="SET NULL"),
        nullable=True,
    )
    sync_status = Column(String(24), nullable=False, default="unavailable")
    reconciliation_status = Column(String(24), nullable=False, default="pending")
    last_attempt_at = Column(DateTime(timezone=True), nullable=True)
    last_success_at = Column(DateTime(timezone=True), nullable=True)
    last_reconciled_at = Column(DateTime(timezone=True), nullable=True)
    stale_after = Column(DateTime(timezone=True), nullable=True)
    consecutive_failures = Column(Integer, nullable=False, default=0)
    unresolved_submission_count = Column(Integer, nullable=False, default=0)
    last_error_code = Column(String(96), nullable=True)
    reconciliation_cursor = Column(String(128), nullable=True)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["account_id", "tenant_id", "credential_id", "environment"],
            [
                "strategy_shared_accounts.id",
                "strategy_shared_accounts.tenant_id",
                "strategy_shared_accounts.credential_id",
                "strategy_shared_accounts.environment",
            ],
            name="fk_shared_demo_sync_state_account_scope",
            ondelete="RESTRICT",
        ),
        CheckConstraint("environment = 'okx_demo'", name="ck_shared_demo_sync_state_environment"),
        CheckConstraint("consecutive_failures >= 0", name="ck_shared_demo_sync_state_failures"),
        CheckConstraint(
            "unresolved_submission_count >= 0",
            name="ck_shared_demo_sync_state_unresolved_submissions",
        ),
        CheckConstraint(
            "sync_status IN ('unavailable', 'healthy', 'stale', 'failed')",
            name="ck_shared_demo_sync_state_status",
        ),
        CheckConstraint(
            "reconciliation_status IN ('pending', 'reconciling', 'complete', 'blocked')",
            name="ck_shared_demo_sync_state_reconciliation",
        ),
        Index("ix_shared_demo_sync_state_reconcile", "reconciliation_status", "last_reconciled_at"),
    )


class SharedDemoExecutionReservation(Base):
    """One account-scoped Demo reservation, immutable after allocation."""

    __tablename__ = "shared_demo_execution_reservations"

    reservation_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id = Column(String(36), nullable=False)
    tenant_id = Column(String(36), nullable=False)
    credential_id = Column(String(36), nullable=False)
    environment = Column(String(16), nullable=False, default=_DEMO_ENVIRONMENT)
    strategy_id = Column(String(100), nullable=False)
    batch_id = Column(String(36), nullable=False)
    idempotency_key = Column(String(128), nullable=False)
    symbol = Column(String(32), nullable=False)
    side = Column(String(16), nullable=False)
    requested_quote = Column(_DEMO_DECIMAL, nullable=False)
    reserved_quote = Column(_DEMO_DECIMAL, nullable=False)
    fee_buffer_quote = Column(_DEMO_DECIMAL, nullable=False, default=0)
    slippage_buffer_quote = Column(_DEMO_DECIMAL, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["account_id", "tenant_id", "credential_id", "environment"],
            [
                "strategy_shared_accounts.id",
                "strategy_shared_accounts.tenant_id",
                "strategy_shared_accounts.credential_id",
                "strategy_shared_accounts.environment",
            ],
            name="fk_shared_demo_reservation_account_scope",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "strategy_id"],
            ["rule_strategies.tenant_id", "rule_strategies.strategy_id"],
            name="fk_shared_demo_reservation_strategy_tenant",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "strategy_id", "batch_id"],
            [
                "rule_strategy_execution_batches.tenant_id",
                "rule_strategy_execution_batches.strategy_id",
                "rule_strategy_execution_batches.batch_id",
            ],
            name="fk_shared_demo_reservation_batch_scope",
            ondelete="RESTRICT",
        ),
        CheckConstraint("environment = 'okx_demo'", name="ck_shared_demo_reservation_environment"),
        CheckConstraint("requested_quote > 0", name="ck_shared_demo_reservation_requested"),
        CheckConstraint("reserved_quote >= requested_quote", name="ck_shared_demo_reservation_reserved"),
        CheckConstraint("fee_buffer_quote >= 0", name="ck_shared_demo_reservation_fee_buffer"),
        CheckConstraint(
            "slippage_buffer_quote >= 0", name="ck_shared_demo_reservation_slippage_buffer"
        ),
        UniqueConstraint(
            "tenant_id", "idempotency_key", name="uq_shared_demo_reservation_idempotency"
        ),
        UniqueConstraint(
            "reservation_id",
            "account_id",
            "tenant_id",
            "credential_id",
            "environment",
            "strategy_id",
            "batch_id",
            name="uq_shared_demo_reservation_scope",
        ),
        Index(
            "ix_shared_demo_reservation_account_strategy_batch",
            "account_id",
            "strategy_id",
            "batch_id",
        ),
    )


class SharedDemoExecutionIntent(Base):
    """Immutable strict binding between a legacy intent and one Demo reservation."""

    __tablename__ = "shared_demo_execution_intents"

    intent_id = Column(String(36), primary_key=True)
    reservation_id = Column(String(36), nullable=False)
    account_id = Column(String(36), nullable=False)
    tenant_id = Column(String(36), nullable=False)
    credential_id = Column(String(36), nullable=False)
    environment = Column(String(16), nullable=False, default=_DEMO_ENVIRONMENT)
    strategy_id = Column(String(100), nullable=False)
    batch_id = Column(String(36), nullable=False)
    client_order_id = Column(String(128), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["intent_id", "tenant_id", "strategy_id", "batch_id"],
            [
                "rule_strategy_execution_intents.id",
                "rule_strategy_execution_intents.tenant_id",
                "rule_strategy_execution_intents.strategy_id",
                "rule_strategy_execution_intents.batch_id",
            ],
            name="fk_shared_demo_intent_legacy_scope",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            [
                "reservation_id",
                "account_id",
                "tenant_id",
                "credential_id",
                "environment",
                "strategy_id",
                "batch_id",
            ],
            [
                "shared_demo_execution_reservations.reservation_id",
                "shared_demo_execution_reservations.account_id",
                "shared_demo_execution_reservations.tenant_id",
                "shared_demo_execution_reservations.credential_id",
                "shared_demo_execution_reservations.environment",
                "shared_demo_execution_reservations.strategy_id",
                "shared_demo_execution_reservations.batch_id",
            ],
            name="fk_shared_demo_intent_reservation_scope",
            ondelete="RESTRICT",
        ),
        CheckConstraint("environment = 'okx_demo'", name="ck_shared_demo_intent_environment"),
        UniqueConstraint("reservation_id", name="uq_shared_demo_intent_reservation"),
        UniqueConstraint(
            "intent_id",
            "account_id",
            "tenant_id",
            "credential_id",
            "environment",
            "strategy_id",
            "batch_id",
            name="uq_shared_demo_intent_scope",
        ),
        UniqueConstraint(
            "tenant_id", "client_order_id", name="uq_shared_demo_intent_client_order"
        ),
        Index(
            "ix_shared_demo_intent_account_strategy_batch",
            "account_id",
            "strategy_id",
            "batch_id",
        ),
    )


class SharedDemoVenueOrder(Base):
    """Append-only accepted or reconciled OKX Demo venue order identity."""

    __tablename__ = "shared_demo_venue_orders"

    order_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    intent_id = Column(String(36), nullable=False)
    reservation_id = Column(String(36), nullable=False)
    account_id = Column(String(36), nullable=False)
    tenant_id = Column(String(36), nullable=False)
    credential_id = Column(String(36), nullable=False)
    environment = Column(String(16), nullable=False, default=_DEMO_ENVIRONMENT)
    strategy_id = Column(String(100), nullable=False)
    batch_id = Column(String(36), nullable=False)
    venue = Column(String(16), nullable=False, default=_DEMO_VENUE)
    client_order_id = Column(String(128), nullable=False)
    venue_order_id = Column(String(128), nullable=True)
    symbol = Column(String(32), nullable=False)
    side = Column(String(16), nullable=False)
    order_type = Column(String(16), nullable=False)
    leg_kind = Column(String(16), nullable=False)
    requested_price = Column(_DEMO_DECIMAL, nullable=True)
    requested_quantity = Column(_DEMO_DECIMAL, nullable=False)
    requested_quote = Column(_DEMO_DECIMAL, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            [
                "intent_id",
                "account_id",
                "tenant_id",
                "credential_id",
                "environment",
                "strategy_id",
                "batch_id",
            ],
            [
                "shared_demo_execution_intents.intent_id",
                "shared_demo_execution_intents.account_id",
                "shared_demo_execution_intents.tenant_id",
                "shared_demo_execution_intents.credential_id",
                "shared_demo_execution_intents.environment",
                "shared_demo_execution_intents.strategy_id",
                "shared_demo_execution_intents.batch_id",
            ],
            name="fk_shared_demo_order_intent_scope",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            [
                "reservation_id",
                "account_id",
                "tenant_id",
                "credential_id",
                "environment",
                "strategy_id",
                "batch_id",
            ],
            [
                "shared_demo_execution_reservations.reservation_id",
                "shared_demo_execution_reservations.account_id",
                "shared_demo_execution_reservations.tenant_id",
                "shared_demo_execution_reservations.credential_id",
                "shared_demo_execution_reservations.environment",
                "shared_demo_execution_reservations.strategy_id",
                "shared_demo_execution_reservations.batch_id",
            ],
            name="fk_shared_demo_order_reservation_scope",
            ondelete="RESTRICT",
        ),
        CheckConstraint("environment = 'okx_demo'", name="ck_shared_demo_order_environment"),
        CheckConstraint("venue = 'okx'", name="ck_shared_demo_order_venue"),
        CheckConstraint(
            "requested_price IS NULL OR requested_price > 0",
            name="ck_shared_demo_order_requested_price",
        ),
        CheckConstraint(
            "requested_quantity > 0", name="ck_shared_demo_order_requested_quantity"
        ),
        CheckConstraint("requested_quote >= 0", name="ck_shared_demo_order_requested_quote"),
        UniqueConstraint("venue", "client_order_id", name="uq_shared_demo_order_venue_client"),
        UniqueConstraint("venue", "venue_order_id", name="uq_shared_demo_order_venue_order"),
        UniqueConstraint(
            "order_id",
            "venue",
            "account_id",
            "tenant_id",
            "credential_id",
            "environment",
            "strategy_id",
            "batch_id",
            name="uq_shared_demo_order_scope",
        ),
        Index(
            "ix_shared_demo_order_account_strategy_batch",
            "account_id",
            "strategy_id",
            "batch_id",
        ),
    )


class SharedDemoOrderProjection(Base):
    """Current reconciliation projection derived from immutable order and fill facts."""

    __tablename__ = "shared_demo_order_projections"

    order_id = Column(String(36), primary_key=True)
    venue = Column(String(16), nullable=False, default=_DEMO_VENUE)
    account_id = Column(String(36), nullable=False)
    tenant_id = Column(String(36), nullable=False)
    credential_id = Column(String(36), nullable=False)
    environment = Column(String(16), nullable=False, default=_DEMO_ENVIRONMENT)
    strategy_id = Column(String(100), nullable=False)
    batch_id = Column(String(36), nullable=False)
    status = Column(String(32), nullable=False, default="pending")
    filled_quantity = Column(_DEMO_DECIMAL, nullable=False, default=0)
    remaining_quantity = Column(_DEMO_DECIMAL, nullable=False, default=0)
    filled_quote = Column(_DEMO_DECIMAL, nullable=False, default=0)
    fee_quote = Column(_DEMO_DECIMAL, nullable=False, default=0)
    last_reconciliation_source = Column(String(32), nullable=True)
    last_observed_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        ForeignKeyConstraint(
            [
                "order_id",
                "venue",
                "account_id",
                "tenant_id",
                "credential_id",
                "environment",
                "strategy_id",
                "batch_id",
            ],
            [
                "shared_demo_venue_orders.order_id",
                "shared_demo_venue_orders.venue",
                "shared_demo_venue_orders.account_id",
                "shared_demo_venue_orders.tenant_id",
                "shared_demo_venue_orders.credential_id",
                "shared_demo_venue_orders.environment",
                "shared_demo_venue_orders.strategy_id",
                "shared_demo_venue_orders.batch_id",
            ],
            name="fk_shared_demo_order_projection_order_scope",
            ondelete="RESTRICT",
        ),
        CheckConstraint("environment = 'okx_demo'", name="ck_shared_demo_projection_environment"),
        CheckConstraint("venue = 'okx'", name="ck_shared_demo_projection_venue"),
        CheckConstraint(
            "status IN ('pending', 'submitted', 'submission_unknown', 'open', "
            "'partially_filled', 'filled', 'cancelled', 'rejected', 'failed')",
            name="ck_shared_demo_projection_status",
        ),
        CheckConstraint("filled_quantity >= 0", name="ck_shared_demo_projection_filled_quantity"),
        CheckConstraint(
            "remaining_quantity >= 0", name="ck_shared_demo_projection_remaining_quantity"
        ),
        CheckConstraint("filled_quote >= 0", name="ck_shared_demo_projection_filled_quote"),
        CheckConstraint("fee_quote >= 0", name="ck_shared_demo_projection_fee_quote"),
        Index("ix_shared_demo_projection_account_status", "account_id", "status"),
    )


class SharedDemoFill(Base):
    """Append-only venue fill fact used to replay strategy-owned lots and PnL."""

    __tablename__ = "shared_demo_fills"

    fill_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    order_id = Column(String(36), nullable=False)
    venue = Column(String(16), nullable=False, default=_DEMO_VENUE)
    venue_fill_id = Column(String(128), nullable=False)
    account_id = Column(String(36), nullable=False)
    tenant_id = Column(String(36), nullable=False)
    credential_id = Column(String(36), nullable=False)
    environment = Column(String(16), nullable=False, default=_DEMO_ENVIRONMENT)
    strategy_id = Column(String(100), nullable=False)
    batch_id = Column(String(36), nullable=False)
    price = Column(_DEMO_DECIMAL, nullable=False)
    quantity = Column(_DEMO_DECIMAL, nullable=False)
    quote_amount = Column(_DEMO_DECIMAL, nullable=False)
    fee_quote = Column(_DEMO_DECIMAL, nullable=False, default=0)
    fee_quantity = Column(_DEMO_DECIMAL, nullable=False, default=0)
    fee_currency = Column(String(16), nullable=True)
    liquidity = Column(String(16), nullable=True)
    occurred_at = Column(DateTime(timezone=True), nullable=False, index=True)
    reconciliation_source = Column(String(32), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            [
                "order_id",
                "venue",
                "account_id",
                "tenant_id",
                "credential_id",
                "environment",
                "strategy_id",
                "batch_id",
            ],
            [
                "shared_demo_venue_orders.order_id",
                "shared_demo_venue_orders.venue",
                "shared_demo_venue_orders.account_id",
                "shared_demo_venue_orders.tenant_id",
                "shared_demo_venue_orders.credential_id",
                "shared_demo_venue_orders.environment",
                "shared_demo_venue_orders.strategy_id",
                "shared_demo_venue_orders.batch_id",
            ],
            name="fk_shared_demo_fill_order_scope",
            ondelete="RESTRICT",
        ),
        CheckConstraint("environment = 'okx_demo'", name="ck_shared_demo_fill_environment"),
        CheckConstraint("venue = 'okx'", name="ck_shared_demo_fill_venue"),
        CheckConstraint("price > 0", name="ck_shared_demo_fill_price"),
        CheckConstraint("quantity > 0", name="ck_shared_demo_fill_quantity"),
        CheckConstraint("quote_amount >= 0", name="ck_shared_demo_fill_quote"),
        CheckConstraint("fee_quote >= 0", name="ck_shared_demo_fill_fee_quote"),
        CheckConstraint("fee_quantity >= 0", name="ck_shared_demo_fill_fee_quantity"),
        UniqueConstraint("venue", "venue_fill_id", name="uq_shared_demo_fill_venue_fill"),
        Index(
            "ix_shared_demo_fill_account_strategy_batch_time",
            "account_id",
            "strategy_id",
            "batch_id",
            occurred_at,
        ),
    )


class SharedDemoReservationRecoveryEvent(Base):
    """Append-only audit evidence for reservation recovery and settlement decisions."""

    __tablename__ = "shared_demo_reservation_recovery_events"

    event_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    reservation_id = Column(String(36), nullable=False)
    account_id = Column(String(36), nullable=False)
    tenant_id = Column(String(36), nullable=False)
    credential_id = Column(String(36), nullable=False)
    environment = Column(String(16), nullable=False, default=_DEMO_ENVIRONMENT)
    strategy_id = Column(String(100), nullable=False)
    batch_id = Column(String(36), nullable=False)
    event_type = Column(String(48), nullable=False)
    outstanding_reserved_quote = Column(_DEMO_DECIMAL, nullable=False)
    occupied_quote = Column(_DEMO_DECIMAL, nullable=False, default=0)
    released_quote = Column(_DEMO_DECIMAL, nullable=False, default=0)
    reason_code = Column(String(96), nullable=True)
    payload = Column(JSON, nullable=False, default=dict)
    occurred_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            [
                "reservation_id",
                "account_id",
                "tenant_id",
                "credential_id",
                "environment",
                "strategy_id",
                "batch_id",
            ],
            [
                "shared_demo_execution_reservations.reservation_id",
                "shared_demo_execution_reservations.account_id",
                "shared_demo_execution_reservations.tenant_id",
                "shared_demo_execution_reservations.credential_id",
                "shared_demo_execution_reservations.environment",
                "shared_demo_execution_reservations.strategy_id",
                "shared_demo_execution_reservations.batch_id",
            ],
            name="fk_shared_demo_recovery_reservation_scope",
            ondelete="RESTRICT",
        ),
        CheckConstraint("environment = 'okx_demo'", name="ck_shared_demo_recovery_environment"),
        CheckConstraint(
            "outstanding_reserved_quote >= 0", name="ck_shared_demo_recovery_outstanding"
        ),
        CheckConstraint("occupied_quote >= 0", name="ck_shared_demo_recovery_occupied"),
        CheckConstraint("released_quote >= 0", name="ck_shared_demo_recovery_released"),
        Index("ix_shared_demo_recovery_reservation_time", "reservation_id", occurred_at),
    )


class SharedDemoStrategyAllocationCap(Base):
    """Account-level strategy allocation envelope for shared Demo funds."""

    __tablename__ = "shared_demo_strategy_allocation_caps"

    cap_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id = Column(String(36), nullable=False)
    tenant_id = Column(String(36), nullable=False)
    credential_id = Column(String(36), nullable=False)
    environment = Column(String(16), nullable=False, default=_DEMO_ENVIRONMENT)
    strategy_id = Column(String(100), nullable=False)
    max_reserved_quote = Column(_DEMO_DECIMAL, nullable=False)
    max_occupied_quote = Column(_DEMO_DECIMAL, nullable=False)
    active = Column(Integer, nullable=False, default=1)
    version = Column(Integer, nullable=False, default=1)
    effective_at = Column(DateTime(timezone=True), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["account_id", "tenant_id", "credential_id", "environment"],
            [
                "strategy_shared_accounts.id",
                "strategy_shared_accounts.tenant_id",
                "strategy_shared_accounts.credential_id",
                "strategy_shared_accounts.environment",
            ],
            name="fk_shared_demo_cap_account_scope",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "strategy_id"],
            ["rule_strategies.tenant_id", "rule_strategies.strategy_id"],
            name="fk_shared_demo_cap_strategy_tenant",
            ondelete="RESTRICT",
        ),
        CheckConstraint("environment = 'okx_demo'", name="ck_shared_demo_cap_environment"),
        CheckConstraint("max_reserved_quote >= 0", name="ck_shared_demo_cap_reserved"),
        CheckConstraint("max_occupied_quote >= 0", name="ck_shared_demo_cap_occupied"),
        CheckConstraint("active IN (0, 1)", name="ck_shared_demo_cap_active"),
        CheckConstraint("version >= 1", name="ck_shared_demo_cap_version"),
        CheckConstraint(
            "expires_at IS NULL OR expires_at > effective_at", name="ck_shared_demo_cap_expiry"
        ),
        UniqueConstraint(
            "account_id",
            "strategy_id",
            "version",
            name="uq_shared_demo_cap_account_strategy_version",
        ),
        Index("ix_shared_demo_cap_account_strategy_active", "account_id", "strategy_id", "active"),
    )


def _forbid_append_only_mutation(_mapper, _connection, _target) -> None:
    raise ValueError("shared Demo execution evidence is append-only")


for _append_only_model in (
    SharedDemoAccountSnapshot,
    SharedDemoExecutionReservation,
    SharedDemoExecutionIntent,
    SharedDemoVenueOrder,
    SharedDemoFill,
    SharedDemoReservationRecoveryEvent,
):
    event.listen(_append_only_model, "before_update", _forbid_append_only_mutation)
    event.listen(_append_only_model, "before_delete", _forbid_append_only_mutation)
