"""Persistence models for deterministic paper rule strategies."""

from __future__ import annotations

import uuid

from sqlalchemy import (
    Boolean, CheckConstraint, Column, DateTime, Float, ForeignKey, Index, Integer,
    JSON, String, UniqueConstraint,
)
from sqlalchemy.sql import func

from .base import Base


class RuleStrategy(Base):
    """A standalone, paper-only deterministic strategy configuration."""

    __tablename__ = "rule_strategies"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(String(100), unique=True, nullable=False, index=True)
    tenant_id = Column(String(36), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(String(1000), nullable=True)
    status = Column(String(20), nullable=False, default="stopped", index=True)
    paper_mode = Column(Boolean, nullable=False, default=True)
    execution_generation = Column(Integer, nullable=False, default=1, server_default="1")
    config = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    archived_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint("execution_generation >= 1", name="ck_rule_strategies_execution_generation"),
    )


class RuleStrategyEvaluationJournal(Base):
    """Durable explanation and paper-log record for a single evaluation."""

    __tablename__ = "rule_strategy_evaluation_journal"

    id = Column(Integer, primary_key=True, index=True)
    evaluation_id = Column(String(100), unique=True, nullable=False, index=True)
    strategy_id = Column(String(100), ForeignKey("rule_strategies.strategy_id", ondelete="CASCADE"), nullable=False, index=True)
    tenant_id = Column(String(36), nullable=False, index=True)
    result = Column(JSON, nullable=False)
    signals = Column(JSON, nullable=False, default=list)
    trades = Column(JSON, nullable=False, default=list)
    funding = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index(
            "ix_rule_strategy_journal_tenant_strategy_created",
            "tenant_id",
            "strategy_id",
            created_at.desc(),
        ),
    )


class RuleStrategyExecutionIntent(Base):
    """Durable, tenant-scoped request to execute one strategy evaluation."""

    __tablename__ = "rule_strategy_execution_intents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    strategy_id = Column(String(100), ForeignKey("rule_strategies.strategy_id", ondelete="RESTRICT"), nullable=False)
    evaluation_id = Column(String(100), ForeignKey("rule_strategy_evaluation_journal.evaluation_id", ondelete="RESTRICT"), nullable=False)
    execution_generation = Column(Integer, nullable=False)
    execution_source = Column(String(32), nullable=False, default="rule_strategy")
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False)
    credential_id = Column(String(36), ForeignKey("tenant_credentials.id", ondelete="RESTRICT"), nullable=True)
    idempotency_key = Column(String(128), nullable=False)
    symbol = Column(String(32), nullable=False)
    side = Column(String(8), nullable=False)
    order_type = Column(String(8), nullable=False)
    requested_quote = Column(String(32), nullable=False)
    requested_quantity = Column(String(32), nullable=True)
    accepted_quote = Column(String(32), nullable=True)
    accepted_quantity = Column(String(32), nullable=True)
    decision_price = Column(String(32), nullable=True)
    execution_target = Column(String(32), nullable=False, default="paper")
    leg_kind = Column(String(16), nullable=False, default="entry")
    lifecycle_state = Column(String(32), nullable=False, default="pending")
    status = Column(String(32), nullable=False, default="pending")
    attempt_count = Column(Integer, nullable=False, default=0, server_default="0")
    error_code = Column(String(64), nullable=True)
    error_message = Column(String(1000), nullable=True)
    submitted_at = Column(DateTime(timezone=True), nullable=True)
    terminal_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    request_payload = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        CheckConstraint("execution_generation >= 1", name="ck_rule_strategy_execution_intent_generation"),
        CheckConstraint("attempt_count >= 0", name="ck_rule_strategy_execution_intent_attempt_count"),
        UniqueConstraint("strategy_id", "evaluation_id", "execution_generation", name="uq_rule_strategy_execution_intent"),
        UniqueConstraint("tenant_id", "idempotency_key", name="uq_rule_strategy_execution_intent_tenant_idempotency"),
        Index("ix_rule_strategy_execution_intents_strategy_generation", "strategy_id", "execution_generation"),
        Index("ix_rule_strategy_execution_intents_strategy_status", "strategy_id", "status"),
        Index("ix_rule_strategy_execution_intents_lifecycle", "status", "updated_at"),
    )


class RuleStrategyAccount(Base):
    """Current isolated capital state for one tenant-owned strategy."""

    __tablename__ = "rule_strategy_accounts"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, index=True)
    strategy_id = Column(String(100), ForeignKey("rule_strategies.strategy_id", ondelete="CASCADE"), nullable=False, index=True)
    scope = Column(String(32), nullable=False, default="paper_virtual")
    credential_id = Column(String(36), ForeignKey("tenant_credentials.id", ondelete="RESTRICT"), nullable=True)
    allocation_quote = Column(Float, nullable=False)
    quote_balance = Column(Float, nullable=False)
    positions = Column(JSON, nullable=False, default=dict)
    realized_pnl_quote = Column(Float, nullable=False, default=0.0)
    unrealized_pnl_quote = Column(Float, nullable=False, default=0.0)
    equity_quote = Column(Float, nullable=False)
    version = Column(Integer, nullable=False, default=1)
    active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        CheckConstraint("allocation_quote > 0", name="ck_rule_strategy_account_allocation"),
        CheckConstraint("version >= 1", name="ck_rule_strategy_account_version"),
        UniqueConstraint("tenant_id", "strategy_id", name="uq_rule_strategy_account_tenant_strategy"),
    )


class RuleStrategyRiskState(Base):
    """Durable account-level admission state, never inferred from diagnostics."""

    __tablename__ = "rule_strategy_risk_states"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("rule_strategy_accounts.id", ondelete="CASCADE"), nullable=False, unique=True)
    tenant_id = Column(String(36), nullable=False, index=True)
    strategy_id = Column(String(100), nullable=False, index=True)
    state = Column(String(16), nullable=False, default="normal")
    daily_equity_baseline = Column(Float, nullable=False)
    high_water_equity = Column(Float, nullable=False)
    current_drawdown_pct = Column(Float, nullable=False, default=0.0)
    cooldown_until = Column(DateTime(timezone=True), nullable=True)
    reason_code = Column(String(96), nullable=True)
    reason_detail = Column(String(1000), nullable=True)
    version = Column(Integer, nullable=False, default=1)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        CheckConstraint("version >= 1", name="ck_rule_strategy_risk_state_version"),
        Index("ix_rule_strategy_risk_tenant_strategy", "tenant_id", "strategy_id"),
    )


class RuleStrategyMonitorSymbol(Base):
    """Current monitor-pool decision and audit-friendly reason per symbol."""

    __tablename__ = "rule_strategy_monitor_symbols"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, index=True)
    strategy_id = Column(String(100), ForeignKey("rule_strategies.strategy_id", ondelete="CASCADE"), nullable=False, index=True)
    symbol = Column(String(32), nullable=False)
    state = Column(String(16), nullable=False, default="candidate")
    reason_code = Column(String(96), nullable=True)
    reason_detail = Column(String(1000), nullable=True)
    evaluated_at = Column(DateTime(timezone=True), nullable=True)
    next_check_at = Column(DateTime(timezone=True), nullable=True)
    protected_held = Column(Boolean, nullable=False, default=False)
    consecutive_low_volume_days = Column(Integer, nullable=False, default=0)
    lease_owner = Column(String(100), nullable=True)
    lease_until = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("tenant_id", "strategy_id", "symbol", name="uq_rule_strategy_monitor_symbol"),
        Index("ix_rule_strategy_monitor_due", "state", "next_check_at"),
    )


class RuleStrategyOrderAttempt(Base):
    """Append-only remote venue submission observation for an execution intent."""

    __tablename__ = "rule_strategy_order_attempts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    intent_id = Column(String(36), ForeignKey("rule_strategy_execution_intents.id", ondelete="CASCADE"), nullable=False, index=True)
    tenant_id = Column(String(36), nullable=False, index=True)
    venue = Column(String(32), nullable=False)
    client_order_id = Column(String(128), nullable=False)
    venue_order_id = Column(String(128), nullable=True)
    requested_price = Column(String(32), nullable=True)
    requested_quantity = Column(String(32), nullable=False)
    status = Column(String(32), nullable=False)
    reconciliation_source = Column(String(32), nullable=True)
    error_code = Column(String(96), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("venue", "client_order_id", name="uq_rule_strategy_order_attempt_venue_client"),
    )


class RuleStrategyFill(Base):
    """Append-only normalized fill and cost evidence for an execution intent."""

    __tablename__ = "rule_strategy_fills"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    intent_id = Column(String(36), ForeignKey("rule_strategy_execution_intents.id", ondelete="CASCADE"), nullable=False, index=True)
    attempt_id = Column(String(36), ForeignKey("rule_strategy_order_attempts.id", ondelete="SET NULL"), nullable=True, index=True)
    tenant_id = Column(String(36), nullable=False, index=True)
    average_price = Column(String(32), nullable=False)
    quantity = Column(String(32), nullable=False)
    fee_quote = Column(String(32), nullable=False, default="0")
    remaining_quantity = Column(String(32), nullable=False, default="0")
    observed_slippage_pct = Column(String(32), nullable=False, default="0")
    reconciliation_source = Column(String(32), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class RuleStrategyEvent(Base):
    """Append-only, tenant-scoped strategy evidence stream."""

    __tablename__ = "rule_strategy_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, index=True)
    strategy_id = Column(String(100), ForeignKey("rule_strategies.strategy_id", ondelete="CASCADE"), nullable=False, index=True)
    account_id = Column(Integer, ForeignKey("rule_strategy_accounts.id", ondelete="SET NULL"), nullable=True, index=True)
    correlation_id = Column(String(100), nullable=False, index=True)
    evaluation_id = Column(String(100), nullable=True)
    intent_id = Column(String(36), nullable=True)
    order_attempt_id = Column(String(36), nullable=True)
    monitor_symbol_id = Column(Integer, nullable=True)
    actor = Column(String(16), nullable=False, default="system")
    reason_code = Column(String(96), nullable=False)
    payload_version = Column(Integer, nullable=False, default=1)
    before_state = Column(JSON, nullable=False, default=dict)
    after_state = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class RuleStrategyExecutionLease(Base):
    """Generation fence preventing duplicate evaluation across scheduler workers."""

    __tablename__ = "rule_strategy_execution_leases"

    strategy_id = Column(String(100), ForeignKey("rule_strategies.strategy_id", ondelete="CASCADE"), primary_key=True)
    execution_generation = Column(Integer, primary_key=True)
    owner_id = Column(String(100), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
