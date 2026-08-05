"""Immutable persistence records for reproducible rule-strategy validation."""

from __future__ import annotations

import uuid

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    event,
    inspect as sa_inspect,
)
from sqlalchemy.sql import func

from .base import Base


class RuleStrategyValidationRun(Base):
    """One tenant-owned, versioned validation request and its immutable result."""

    __tablename__ = "rule_strategy_validation_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(36), unique=True, nullable=False, index=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(
        String(36),
        ForeignKey("tenants.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    strategy_id = Column(
        String(100),
        ForeignKey("rule_strategies.strategy_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    status = Column(String(16), nullable=False, default="pending", index=True)
    source_preference = Column(String(64), nullable=False)
    selected_symbols = Column(JSON, nullable=False, default=list)
    config_json = Column(JSON, nullable=False)
    config_fingerprint = Column(String(64), nullable=False)
    assumptions = Column(JSON, nullable=False)
    assumptions_fingerprint = Column(String(64), nullable=False)
    data_fingerprint = Column(String(64), nullable=False)
    artifact_fingerprint = Column(String(64), nullable=True)
    initial_capital_quote = Column(Float, nullable=False)
    template_id = Column(String(96), nullable=True)
    template_version = Column(Integer, nullable=True)
    indicator_formula_version = Column(String(96), nullable=False, default="legacy_rule_engine")
    engine_version = Column(String(96), nullable=False)
    in_sample_start_at = Column(DateTime(timezone=True), nullable=False)
    in_sample_end_at_exclusive = Column(DateTime(timezone=True), nullable=False)
    out_of_sample_start_at = Column(DateTime(timezone=True), nullable=False)
    out_of_sample_end_at_exclusive = Column(DateTime(timezone=True), nullable=False)
    metrics = Column(JSON, nullable=True)
    error_code = Column(String(96), nullable=True)
    error_detail = Column(String(2_000), nullable=True)
    worker_id = Column(String(100), nullable=True)
    lease_expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    cancel_requested_at = Column(DateTime(timezone=True), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_rule_strategy_validation_run_status",
        ),
        CheckConstraint(
            "initial_capital_quote > 0",
            name="ck_rule_strategy_validation_initial_capital",
        ),
        CheckConstraint(
            "in_sample_start_at < in_sample_end_at_exclusive",
            name="ck_rule_strategy_validation_in_sample_window",
        ),
        CheckConstraint(
            "in_sample_end_at_exclusive = out_of_sample_start_at",
            name="ck_rule_strategy_validation_contiguous_window",
        ),
        CheckConstraint(
            "out_of_sample_start_at < out_of_sample_end_at_exclusive",
            name="ck_rule_strategy_validation_oos_window",
        ),
        Index(
            "ix_rule_strategy_validation_runs_tenant_strategy_created",
            "tenant_id",
            "strategy_id",
            created_at.desc(),
        ),
        Index(
            "ix_rule_strategy_validation_runs_claim",
            "status",
            "lease_expires_at",
        ),
    )


class RuleStrategyValidationDataset(Base):
    """Complete, normalized source bars and coverage manifest for one run slice."""

    __tablename__ = "rule_strategy_validation_datasets"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(String(36), unique=True, nullable=False, index=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(
        String(36),
        ForeignKey("rule_strategy_validation_runs.run_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    tenant_id = Column(
        String(36),
        ForeignKey("tenants.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    strategy_id = Column(
        String(100),
        ForeignKey("rule_strategies.strategy_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    source_provider = Column(String(64), nullable=False)
    symbol = Column(String(64), nullable=False)
    interval = Column(String(8), nullable=False)
    start_at = Column(DateTime(timezone=True), nullable=False)
    end_at_exclusive = Column(DateTime(timezone=True), nullable=False)
    bar_count = Column(Integer, nullable=False)
    bars = Column(JSON, nullable=False)
    page_manifest = Column(JSON, nullable=False, default=list)
    coverage_manifest = Column(JSON, nullable=False)
    content_hash = Column(String(64), nullable=False)
    retrieved_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        CheckConstraint("bar_count > 0", name="ck_rule_strategy_validation_dataset_bar_count"),
        CheckConstraint(
            "start_at < end_at_exclusive",
            name="ck_rule_strategy_validation_dataset_window",
        ),
        UniqueConstraint(
            "run_id",
            "symbol",
            "interval",
            name="uq_rule_strategy_validation_dataset_run_symbol_interval",
        ),
        Index(
            "ix_rule_strategy_validation_dataset_tenant_run",
            "tenant_id",
            "run_id",
        ),
    )


class RuleStrategyValidationPoint(Base):
    """Append-only account/equity point emitted during deterministic replay."""

    __tablename__ = "rule_strategy_validation_points"

    id = Column(Integer, primary_key=True, index=True)
    point_id = Column(String(36), unique=True, nullable=False, index=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(
        String(36),
        ForeignKey("rule_strategy_validation_runs.run_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    tenant_id = Column(
        String(36),
        ForeignKey("tenants.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    strategy_id = Column(
        String(100),
        ForeignKey("rule_strategies.strategy_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    sequence = Column(Integer, nullable=False)
    window = Column(String(16), nullable=False)
    observed_at = Column(DateTime(timezone=True), nullable=False)
    equity_quote = Column(Float, nullable=False)
    cash_quote = Column(Float, nullable=False)
    position_quote = Column(Float, nullable=False)
    drawdown_pct = Column(Float, nullable=False)
    account_snapshot = Column(JSON, nullable=False)
    decisions = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        CheckConstraint("sequence >= 0", name="ck_rule_strategy_validation_point_sequence"),
        CheckConstraint(
            "window IN ('in_sample', 'out_of_sample')",
            name="ck_rule_strategy_validation_point_window",
        ),
        CheckConstraint("equity_quote >= 0", name="ck_rule_strategy_validation_point_equity"),
        CheckConstraint("cash_quote >= 0", name="ck_rule_strategy_validation_point_cash"),
        CheckConstraint("position_quote >= 0", name="ck_rule_strategy_validation_point_position"),
        UniqueConstraint("run_id", "sequence", name="uq_rule_strategy_validation_point_sequence"),
        Index(
            "ix_rule_strategy_validation_points_tenant_run_window_time",
            "tenant_id",
            "run_id",
            "window",
            "observed_at",
        ),
    )


class RuleStrategyValidationFill(Base):
    """Append-only next-bar fill evidence from a validation run."""

    __tablename__ = "rule_strategy_validation_fills"

    id = Column(Integer, primary_key=True, index=True)
    fill_id = Column(String(36), unique=True, nullable=False, index=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(
        String(36),
        ForeignKey("rule_strategy_validation_runs.run_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    tenant_id = Column(
        String(36),
        ForeignKey("tenants.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    strategy_id = Column(
        String(100),
        ForeignKey("rule_strategies.strategy_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    sequence = Column(Integer, nullable=False)
    window = Column(String(16), nullable=False)
    symbol = Column(String(64), nullable=False)
    leg_kind = Column(String(16), nullable=False)
    side = Column(String(8), nullable=False)
    decision_at = Column(DateTime(timezone=True), nullable=False)
    filled_at = Column(DateTime(timezone=True), nullable=False)
    decision_price = Column(Float, nullable=False)
    fill_price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    quote_amount = Column(Float, nullable=False)
    fee_quote = Column(Float, nullable=False)
    slippage_pct = Column(Float, nullable=False)
    realized_pnl_quote = Column(Float, nullable=False, default=0.0)
    reason_code = Column(String(128), nullable=False)
    account_before = Column(JSON, nullable=False)
    account_after = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        CheckConstraint("sequence >= 0", name="ck_rule_strategy_validation_fill_sequence"),
        CheckConstraint(
            "window IN ('in_sample', 'out_of_sample')",
            name="ck_rule_strategy_validation_fill_window",
        ),
        CheckConstraint(
            "leg_kind IN ('entry', 'add', 'reduce', 'close')",
            name="ck_rule_strategy_validation_fill_leg",
        ),
        CheckConstraint("side IN ('buy', 'sell')", name="ck_rule_strategy_validation_fill_side"),
        CheckConstraint("decision_at <= filled_at", name="ck_rule_strategy_validation_fill_time"),
        CheckConstraint("decision_price > 0", name="ck_rule_strategy_validation_fill_decision_price"),
        CheckConstraint("fill_price > 0", name="ck_rule_strategy_validation_fill_price"),
        CheckConstraint("quantity > 0", name="ck_rule_strategy_validation_fill_quantity"),
        CheckConstraint("quote_amount > 0", name="ck_rule_strategy_validation_fill_quote"),
        CheckConstraint("fee_quote >= 0", name="ck_rule_strategy_validation_fill_fee"),
        UniqueConstraint("run_id", "sequence", name="uq_rule_strategy_validation_fill_sequence"),
        Index(
            "ix_rule_strategy_validation_fills_tenant_run_window_time",
            "tenant_id",
            "run_id",
            "window",
            "filled_at",
        ),
    )


_RUN_IMMUTABLE_FIELDS = (
    "tenant_id",
    "strategy_id",
    "source_preference",
    "selected_symbols",
    "config_json",
    "config_fingerprint",
    "assumptions",
    "assumptions_fingerprint",
    "data_fingerprint",
    "initial_capital_quote",
    "template_id",
    "template_version",
    "indicator_formula_version",
    "engine_version",
    "in_sample_start_at",
    "in_sample_end_at_exclusive",
    "out_of_sample_start_at",
    "out_of_sample_end_at_exclusive",
)


@event.listens_for(RuleStrategyValidationRun, "before_update")
def _forbid_validation_run_mutation_after_completion(_mapper, _connection, target) -> None:
    """Allow lifecycle progress only; immutable inputs never change after insert."""

    state = sa_inspect(target)
    status_history = state.attrs.status.history
    prior_status = status_history.deleted[0] if status_history.deleted else None
    if prior_status == "completed" or (prior_status is None and target.status == "completed"):
        raise ValueError("completed validation runs are immutable")
    for field in _RUN_IMMUTABLE_FIELDS:
        if state.attrs[field].history.has_changes():
            raise ValueError(f"validation run field '{field}' is immutable")
    if (
        state.attrs.metrics.history.has_changes()
        or state.attrs.artifact_fingerprint.history.has_changes()
    ) and target.status != "completed":
        raise ValueError("validation metrics are only written when a run completes")


@event.listens_for(RuleStrategyValidationRun, "before_delete")
def _forbid_completed_validation_run_delete(_mapper, _connection, target) -> None:
    if target.status == "completed":
        raise ValueError("completed validation runs are immutable")


def _forbid_append_only_mutation(_mapper, _connection, _target) -> None:
    raise ValueError("validation evidence is append-only")


for _append_only_model in (
    RuleStrategyValidationDataset,
    RuleStrategyValidationPoint,
    RuleStrategyValidationFill,
):
    event.listen(_append_only_model, "before_update", _forbid_append_only_mutation)
    event.listen(_append_only_model, "before_delete", _forbid_append_only_mutation)
