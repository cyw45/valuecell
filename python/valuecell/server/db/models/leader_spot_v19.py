"""Isolated persistence models for the V19 OKX leaderboard spot strategy."""

from __future__ import annotations

import uuid

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from .base import Base


class LeaderSpotV19Strategy(Base):
    """Tenant-owned V19 configuration that never shares legacy strategy tables."""

    __tablename__ = "leader_spot_v19_strategies"

    strategy_id = Column(String(100), primary_key=True)
    tenant_id = Column(
        String(36),
        ForeignKey("tenants.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    name = Column(String(200), nullable=False)
    description = Column(String(2_000), nullable=True)
    status = Column(String(16), nullable=False, default="stopped", index=True)
    environment = Column(String(16), nullable=False, default="paper")
    credential_id = Column(
        String(36),
        ForeignKey("tenant_credentials.id", ondelete="RESTRICT"),
        nullable=True,
    )
    execution_generation = Column(
        Integer,
        nullable=False,
        default=1,
        server_default="1",
    )
    current_batch_id = Column(String(36), nullable=True, index=True)
    config = Column(JSON, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    archived_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'stopped', 'paused', 'archived')",
            name="ck_leader_spot_v19_strategy_status",
        ),
        CheckConstraint(
            "environment IN ('paper', 'okx_demo')",
            name="ck_leader_spot_v19_strategy_environment",
        ),
        CheckConstraint(
            "execution_generation >= 1",
            name="ck_leader_spot_v19_strategy_generation",
        ),
        UniqueConstraint(
            "tenant_id",
            "strategy_id",
            name="uq_leader_spot_v19_strategy_tenant_id",
        ),
    )


class LeaderSpotV19ExecutionBatch(Base):
    """Immutable start-to-stop evidence boundary for one V19 execution run."""

    __tablename__ = "leader_spot_v19_execution_batches"

    batch_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), nullable=False, index=True)
    strategy_id = Column(String(100), nullable=False, index=True)
    strategy_name_snapshot = Column(String(200), nullable=False)
    execution_generation = Column(Integer, nullable=False)
    status = Column(String(16), nullable=False, default="running", index=True)
    started_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    stopped_at = Column(DateTime(timezone=True), nullable=True)
    config_snapshot = Column(JSON, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "strategy_id"],
            [
                "leader_spot_v19_strategies.tenant_id",
                "leader_spot_v19_strategies.strategy_id",
            ],
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "status IN ('running', 'stopped', 'archived')",
            name="ck_leader_spot_v19_batch_status",
        ),
        CheckConstraint(
            "execution_generation >= 1",
            name="ck_leader_spot_v19_batch_generation",
        ),
        UniqueConstraint(
            "tenant_id",
            "strategy_id",
            "batch_id",
            name="uq_leader_spot_v19_batch_scope",
        ),
        Index(
            "ix_leader_spot_v19_batches_tenant_strategy_started",
            "tenant_id",
            "strategy_id",
            started_at.desc(),
        ),
    )


class LeaderSpotV19Account(Base):
    """Batch-scoped accounting state; positions remain separately attributable."""

    __tablename__ = "leader_spot_v19_accounts"

    account_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), nullable=False, index=True)
    strategy_id = Column(String(100), nullable=False, index=True)
    batch_id = Column(String(36), nullable=False, index=True)
    scope = Column(String(16), nullable=False, default="paper")
    credential_id = Column(String(36), nullable=True)
    initial_equity_quote = Column(Float, nullable=False)
    quote_balance = Column(Float, nullable=False)
    realized_pnl_quote = Column(Float, nullable=False, default=0.0)
    unrealized_pnl_quote = Column(Float, nullable=False, default=0.0)
    equity_quote = Column(Float, nullable=False)
    version = Column(Integer, nullable=False, default=1, server_default="1")
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "strategy_id", "batch_id"],
            [
                "leader_spot_v19_execution_batches.tenant_id",
                "leader_spot_v19_execution_batches.strategy_id",
                "leader_spot_v19_execution_batches.batch_id",
            ],
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "scope IN ('paper', 'okx_demo')",
            name="ck_leader_spot_v19_account_scope",
        ),
        CheckConstraint(
            "initial_equity_quote > 0",
            name="ck_leader_spot_v19_account_initial_equity",
        ),
        CheckConstraint(
            "version >= 1",
            name="ck_leader_spot_v19_account_version",
        ),
        UniqueConstraint(
            "tenant_id",
            "strategy_id",
            "batch_id",
            name="uq_leader_spot_v19_account_scope",
        ),
    )


class LeaderSpotV19RiskState(Base):
    """Persisted V19 daily-loss and equity-halt facts, never derived from UI."""

    __tablename__ = "leader_spot_v19_risk_states"

    risk_state_id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    account_id = Column(
        String(36),
        ForeignKey("leader_spot_v19_accounts.account_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    tenant_id = Column(String(36), nullable=False, index=True)
    strategy_id = Column(String(100), nullable=False, index=True)
    batch_id = Column(String(36), nullable=False, index=True)
    state = Column(String(24), nullable=False, default="normal")
    daily_realized_pnl_quote = Column(Float, nullable=False, default=0.0)
    daily_loss_limit_quote = Column(Float, nullable=False)
    daily_loss_reset_at = Column(DateTime(timezone=True), nullable=False)
    prior_close_equity_quote = Column(Float, nullable=False)
    equity_drawdown_pct = Column(Float, nullable=False, default=0.0)
    halt_until = Column(DateTime(timezone=True), nullable=True)
    reason_code = Column(String(96), nullable=True)
    reason_detail = Column(String(1_000), nullable=True)
    version = Column(Integer, nullable=False, default=1, server_default="1")
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "strategy_id", "batch_id"],
            [
                "leader_spot_v19_execution_batches.tenant_id",
                "leader_spot_v19_execution_batches.strategy_id",
                "leader_spot_v19_execution_batches.batch_id",
            ],
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "state IN ('normal', 'daily_loss_halted', 'equity_halted')",
            name="ck_leader_spot_v19_risk_state",
        ),
        CheckConstraint(
            "daily_loss_limit_quote > 0",
            name="ck_leader_spot_v19_risk_daily_limit",
        ),
        CheckConstraint(
            "version >= 1",
            name="ck_leader_spot_v19_risk_version",
        ),
        UniqueConstraint(
            "tenant_id",
            "strategy_id",
            "batch_id",
            name="uq_leader_spot_v19_risk_scope",
        ),
    )


class LeaderSpotV19CandidateSnapshot(Base):
    """One persisted ranking candidate decision, including exclusions and ranking facts."""

    __tablename__ = "leader_spot_v19_candidate_snapshots"

    candidate_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), nullable=False, index=True)
    strategy_id = Column(String(100), nullable=False, index=True)
    batch_id = Column(String(36), nullable=False, index=True)
    snapshot_group_id = Column(String(36), nullable=False, index=True)
    source = Column(String(32), nullable=False)
    symbol = Column(String(32), nullable=False)
    source_rank = Column(Integer, nullable=True)
    market_state = Column(String(4), nullable=False)
    data_state = Column(String(16), nullable=False)
    funnel_stage = Column(String(48), nullable=False)
    accepted = Column(Boolean, nullable=False, default=False)
    score = Column(Float, nullable=True)
    reason_code = Column(String(96), nullable=True)
    facts = Column(JSON, nullable=False, default=dict)
    observed_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "strategy_id", "batch_id"],
            [
                "leader_spot_v19_execution_batches.tenant_id",
                "leader_spot_v19_execution_batches.strategy_id",
                "leader_spot_v19_execution_batches.batch_id",
            ],
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "market_state IN ('M0', 'M1', 'M2', 'M3', 'M4')",
            name="ck_leader_spot_v19_candidate_market_state",
        ),
        CheckConstraint(
            "data_state IN ('DATA_OK', 'DATA_DEGRADED', 'DATA_UNSAFE')",
            name="ck_leader_spot_v19_candidate_data_state",
        ),
        Index(
            "ix_leader_spot_v19_candidates_scope_observed",
            "tenant_id",
            "strategy_id",
            "batch_id",
            observed_at.desc(),
        ),
    )


class LeaderSpotV19MarketSnapshot(Base):
    """Raw, source-stamped market input captured before it drives a decision."""

    __tablename__ = "leader_spot_v19_market_snapshots"

    snapshot_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), nullable=False, index=True)
    strategy_id = Column(String(100), nullable=False, index=True)
    batch_id = Column(String(36), nullable=False, index=True)
    source = Column(String(32), nullable=False)
    snapshot_kind = Column(String(48), nullable=False)
    symbol = Column(String(32), nullable=True)
    payload = Column(JSON, nullable=False)
    freshness = Column(String(16), nullable=False)
    observed_at = Column(DateTime(timezone=True), nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "strategy_id", "batch_id"],
            [
                "leader_spot_v19_execution_batches.tenant_id",
                "leader_spot_v19_execution_batches.strategy_id",
                "leader_spot_v19_execution_batches.batch_id",
            ],
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "freshness IN ('fresh', 'stale', 'unsafe')",
            name="ck_leader_spot_v19_snapshot_freshness",
        ),
        CheckConstraint(
            "expires_at >= observed_at",
            name="ck_leader_spot_v19_snapshot_expiry",
        ),
        Index(
            "ix_leader_spot_v19_market_snapshots_scope_kind_observed",
            "tenant_id",
            "strategy_id",
            "batch_id",
            "snapshot_kind",
            observed_at.desc(),
        ),
    )


class LeaderSpotV19DataQualityReport(Base):
    """Durable quality gate result required before V19 entry evaluation."""
    __tablename__ = "leader_spot_v19_data_quality_reports"

    quality_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), nullable=False, index=True)
    strategy_id = Column(String(100), nullable=False, index=True)
    batch_id = Column(String(36), nullable=False, index=True)
    data_state = Column(String(16), nullable=False)
    accepted_for_entry = Column(Boolean, nullable=False, default=False)
    fresh_input_count = Column(Integer, nullable=False, default=0)
    required_input_count = Column(Integer, nullable=False, default=0)
    issues = Column(JSON, nullable=False, default=list)
    checked_symbols = Column(JSON, nullable=False, default=list)
    observed_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "strategy_id", "batch_id"],
            [
                "leader_spot_v19_execution_batches.tenant_id",
                "leader_spot_v19_execution_batches.strategy_id",
                "leader_spot_v19_execution_batches.batch_id",
            ],
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "data_state IN ('DATA_OK', 'DATA_DEGRADED', 'DATA_UNSAFE')",
            name="ck_leader_spot_v19_quality_data_state",
        ),
        CheckConstraint(
            "fresh_input_count >= 0 AND required_input_count >= 0",
            name="ck_leader_spot_v19_quality_input_counts",
        ),
        Index(
            "ix_leader_spot_v19_quality_scope_observed",
            "tenant_id",
            "strategy_id",
            "batch_id",
            observed_at.desc(),
        ),
    )


class LeaderSpotV19MarketStateDecision(Base):
    """Append-only M0–M4 decision and starvation policy evidence."""

    __tablename__ = "leader_spot_v19_market_state_decisions"

    decision_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), nullable=False, index=True)
    strategy_id = Column(String(100), nullable=False, index=True)
    batch_id = Column(String(36), nullable=False, index=True)
    data_state = Column(String(16), nullable=False)
    market_state = Column(String(4), nullable=False)
    entry_profile = Column(String(16), nullable=False)
    can_open = Column(Boolean, nullable=False, default=False)
    reason_codes = Column(JSON, nullable=False, default=list)
    input_facts = Column(JSON, nullable=False)
    conditions = Column(JSON, nullable=False, default=list)
    starvation = Column(JSON, nullable=False)
    observed_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "strategy_id", "batch_id"],
            [
                "leader_spot_v19_execution_batches.tenant_id",
                "leader_spot_v19_execution_batches.strategy_id",
                "leader_spot_v19_execution_batches.batch_id",
            ],
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "data_state IN ('DATA_OK', 'DATA_DEGRADED', 'DATA_UNSAFE')",
            name="ck_leader_spot_v19_market_decision_data_state",
        ),
        CheckConstraint(
            "market_state IN ('M0', 'M1', 'M2', 'M3', 'M4')",
            name="ck_leader_spot_v19_market_decision_state",
        ),
        CheckConstraint(
            "entry_profile IN ('halt', 'degraded', 'standard', 'strong_trend')",
            name="ck_leader_spot_v19_market_decision_profile",
        ),
        Index(
            "ix_leader_spot_v19_market_decision_scope_observed",
            "tenant_id",
            "strategy_id",
            "batch_id",
            observed_at.desc(),
        ),
    )


class LeaderSpotV19Position(Base):
    """Attributed position and durable V19 profit-protection state per symbol."""

    __tablename__ = "leader_spot_v19_positions"

    position_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), nullable=False, index=True)
    strategy_id = Column(String(100), nullable=False, index=True)
    batch_id = Column(String(36), nullable=False, index=True)
    symbol = Column(String(32), nullable=False)
    entry_intent_id = Column(String(36), nullable=True, index=True)
    entry_order_id = Column(String(128), nullable=True)
    entry_price = Column(String(32), nullable=False)
    entry_quantity = Column(String(32), nullable=False)
    entry_time = Column(DateTime(timezone=True), nullable=False, index=True)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    close_reason_code = Column(String(96), nullable=True)
    protection_status = Column(String(24), nullable=False, default="PROTECTION_NONE")
    protection_started_at = Column(DateTime(timezone=True), nullable=True)
    peak_price = Column(String(32), nullable=False)
    peak_profit_pct = Column(Float, nullable=False, default=0.0)
    moving_stop_price = Column(String(32), nullable=False)
    layered_exit_price = Column(String(32), nullable=True)
    loss_circuit_started_at = Column(DateTime(timezone=True), nullable=False)
    loss_circuit_active = Column(Boolean, nullable=False, default=True)
    trend_stop_active = Column(Boolean, nullable=False, default=False)
    trend_break_count = Column(Integer, nullable=False, default=0)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "strategy_id", "batch_id"],
            [
                "leader_spot_v19_execution_batches.tenant_id",
                "leader_spot_v19_execution_batches.strategy_id",
                "leader_spot_v19_execution_batches.batch_id",
            ],
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "protection_status IN ('PROTECTION_NONE', 'PROTECTION_PENDING', 'PROTECTION_ACTIVE')",
            name="ck_leader_spot_v19_position_protection",
        ),
        CheckConstraint(
            "trend_break_count >= 0",
            name="ck_leader_spot_v19_position_trend_breaks",
        ),
        Index(
            "ix_leader_spot_v19_positions_scope_symbol_open",
            "tenant_id",
            "strategy_id",
            "batch_id",
            "symbol",
            "closed_at",
        ),
    )


class LeaderSpotV19ExecutionIntent(Base):
    """Outbox-owned instruction that is durable before any venue submission."""

    __tablename__ = "leader_spot_v19_execution_intents"

    intent_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), nullable=False, index=True)
    strategy_id = Column(String(100), nullable=False, index=True)
    batch_id = Column(String(36), nullable=False, index=True)
    position_id = Column(String(36), nullable=True, index=True)
    credential_id = Column(String(36), nullable=True)
    execution_generation = Column(Integer, nullable=False)
    idempotency_key = Column(String(128), nullable=False)
    symbol = Column(String(32), nullable=False)
    side = Column(String(4), nullable=False)
    order_type = Column(String(8), nullable=False)
    leg_kind = Column(String(16), nullable=False)
    requested_quote = Column(String(32), nullable=True)
    requested_quantity = Column(String(32), nullable=True)
    requested_price = Column(String(32), nullable=True)
    lifecycle_state = Column(String(32), nullable=False, default="pending")
    status = Column(String(32), nullable=False, default="pending")
    attempt_count = Column(Integer, nullable=False, default=0, server_default="0")
    error_code = Column(String(96), nullable=True)
    error_detail = Column(String(1_000), nullable=True)
    submitted_at = Column(DateTime(timezone=True), nullable=True)
    terminal_at = Column(DateTime(timezone=True), nullable=True)
    request_payload = Column(JSON, nullable=False, default=dict)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "strategy_id", "batch_id"],
            [
                "leader_spot_v19_execution_batches.tenant_id",
                "leader_spot_v19_execution_batches.strategy_id",
                "leader_spot_v19_execution_batches.batch_id",
            ],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["position_id"],
            ["leader_spot_v19_positions.position_id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "execution_generation >= 1",
            name="ck_leader_spot_v19_intent_generation",
        ),
        CheckConstraint(
            "attempt_count >= 0",
            name="ck_leader_spot_v19_intent_attempt_count",
        ),
        CheckConstraint(
            "side IN ('buy', 'sell')",
            name="ck_leader_spot_v19_intent_side",
        ),
        CheckConstraint(
            "order_type IN ('limit', 'market')",
            name="ck_leader_spot_v19_intent_order_type",
        ),
        UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_leader_spot_v19_intent_idempotency",
        ),
        Index(
            "ix_leader_spot_v19_intents_scope_status",
            "tenant_id",
            "strategy_id",
            "batch_id",
            "status",
        ),
    )


class LeaderSpotV19OrderAttempt(Base):
    """Append-only observation of a remote order-submit or reconciliation attempt."""

    __tablename__ = "leader_spot_v19_order_attempts"

    attempt_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    intent_id = Column(
        String(36),
        ForeignKey("leader_spot_v19_execution_intents.intent_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tenant_id = Column(String(36), nullable=False, index=True)
    venue = Column(String(32), nullable=False)
    client_order_id = Column(String(128), nullable=False)
    venue_order_id = Column(String(128), nullable=True)
    requested_price = Column(String(32), nullable=True)
    requested_quantity = Column(String(32), nullable=False)
    status = Column(String(32), nullable=False)
    reconciliation_source = Column(String(32), nullable=True)
    error_code = Column(String(96), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "venue",
            "client_order_id",
            name="uq_leader_spot_v19_attempt_venue_client",
        ),
    )


class LeaderSpotV19Fill(Base):
    """Append-only fill facts used to replay positions and attributed PnL."""

    __tablename__ = "leader_spot_v19_fills"

    fill_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    intent_id = Column(
        String(36),
        ForeignKey("leader_spot_v19_execution_intents.intent_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attempt_id = Column(
        String(36),
        ForeignKey("leader_spot_v19_order_attempts.attempt_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    tenant_id = Column(String(36), nullable=False, index=True)
    venue_fill_id = Column(String(128), nullable=True)
    average_price = Column(String(32), nullable=False)
    quantity = Column(String(32), nullable=False)
    fee_quote = Column(String(32), nullable=False, default="0")
    remaining_quantity = Column(String(32), nullable=False, default="0")
    observed_slippage_pct = Column(String(32), nullable=False, default="0")
    reconciliation_source = Column(String(32), nullable=False)
    filled_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "intent_id",
            "venue_fill_id",
            name="uq_leader_spot_v19_fill_intent_venue",
        ),
    )


class LeaderSpotV19Event(Base):
    """Append-only strategy evidence stream for decisions and operational anomalies."""

    __tablename__ = "leader_spot_v19_events"

    event_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), nullable=False, index=True)
    strategy_id = Column(String(100), nullable=False, index=True)
    batch_id = Column(String(36), nullable=False, index=True)
    correlation_id = Column(String(100), nullable=False, index=True)
    position_id = Column(String(36), nullable=True, index=True)
    intent_id = Column(String(36), nullable=True, index=True)
    attempt_id = Column(String(36), nullable=True, index=True)
    actor = Column(String(16), nullable=False, default="system")
    reason_code = Column(String(96), nullable=False)
    payload_version = Column(Integer, nullable=False, default=1, server_default="1")
    before_state = Column(JSON, nullable=False, default=dict)
    after_state = Column(JSON, nullable=False, default=dict)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "strategy_id", "batch_id"],
            [
                "leader_spot_v19_execution_batches.tenant_id",
                "leader_spot_v19_execution_batches.strategy_id",
                "leader_spot_v19_execution_batches.batch_id",
            ],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["position_id"],
            ["leader_spot_v19_positions.position_id"],
            ondelete="SET NULL",
        ),
        ForeignKeyConstraint(
            ["intent_id"],
            ["leader_spot_v19_execution_intents.intent_id"],
            ondelete="SET NULL",
        ),
        ForeignKeyConstraint(
            ["attempt_id"],
            ["leader_spot_v19_order_attempts.attempt_id"],
            ondelete="SET NULL",
        ),
        CheckConstraint(
            "payload_version >= 1",
            name="ck_leader_spot_v19_event_payload_version",
        ),
        Index(
            "ix_leader_spot_v19_events_scope_created",
            "tenant_id",
            "strategy_id",
            "batch_id",
            created_at.desc(),
        ),
    )


class LeaderSpotV19ExecutionLease(Base):
    """Generation fence that keeps V19 scheduler workers mutually exclusive."""

    __tablename__ = "leader_spot_v19_execution_leases"

    strategy_id = Column(String(100), primary_key=True)
    execution_generation = Column(Integer, primary_key=True)
    owner_id = Column(String(100), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["strategy_id"],
            ["leader_spot_v19_strategies.strategy_id"],
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "execution_generation >= 1",
            name="ck_leader_spot_v19_lease_generation",
        ),
    )
