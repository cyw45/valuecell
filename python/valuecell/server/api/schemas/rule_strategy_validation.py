"""Contracts for reproducible, tenant-scoped rule-strategy validation."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .rule_strategy import RuleInterval, RuleStrategyCandle


ValidationRunStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
]
ValidationWindowName = Literal["in_sample", "out_of_sample"]


class RuleStrategyValidationModel(BaseModel):
    """Strict shared base for validation-only API contracts."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RuleStrategyValidationCandle(RuleStrategyCandle):
    """A closed UTC bar whose timestamp is the UTC bar-open instant."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RuleStrategyValidationCreateRequest(RuleStrategyValidationModel):
    """Inputs the eventual validation endpoint accepts from a tenant client.

    The service receives data separately through an injected materializer; this
    request intentionally contains no credentials or provider access settings.
    """

    oos_end_date: date
    source_preference: str = Field(default="injected", min_length=1, max_length=64)
    selected_symbols: list[str] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def normalize_symbols(self) -> "RuleStrategyValidationCreateRequest":
        symbols: list[str] = []
        for raw_symbol in self.selected_symbols:
            symbol = raw_symbol.strip().upper().replace("/", "-")
            if not symbol.endswith("-USDT"):
                raise ValueError("Only USDT crypto symbols are supported")
            if symbol not in symbols:
                symbols.append(symbol)
        if not symbols:
            raise ValueError("selected_symbols must contain at least one symbol")
        self.selected_symbols = symbols
        return self


class RuleStrategyValidationDatasetInput(RuleStrategyValidationModel):
    """Fully materialized bars supplied by a trusted in-process data adapter.

    The validation service never performs a network request. Callers must inject
    every bar (including page metadata) and provide timestamps at bar-open UTC.
    """

    source_provider: str = Field(min_length=1, max_length=64)
    symbol: str = Field(min_length=1, max_length=64)
    interval: RuleInterval
    bars: list[RuleStrategyValidationCandle] = Field(min_length=1)
    page_manifest: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def normalize_and_validate(self) -> "RuleStrategyValidationDatasetInput":
        symbol = self.symbol.strip().upper().replace("/", "-")
        if not symbol.endswith("-USDT"):
            raise ValueError("Only USDT crypto symbols are supported")
        self.symbol = symbol
        if self.retrieved_at.tzinfo is None:
            raise ValueError("retrieved_at must be timezone-aware")
        self.retrieved_at = self.retrieved_at.astimezone(UTC)
        timestamps = [bar.timestamp_ms for bar in self.bars]
        if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ValueError("dataset candle timestamps must be strictly increasing")
        return self


class RuleStrategyValidationWindow(RuleStrategyValidationModel):
    """Canonical non-overlapping in-sample and out-of-sample UTC boundaries."""

    in_sample_start_at: datetime
    in_sample_end_at_exclusive: datetime
    out_of_sample_start_at: datetime
    out_of_sample_end_at_exclusive: datetime

    @model_validator(mode="after")
    def validate_boundaries(self) -> "RuleStrategyValidationWindow":
        values = (
            self.in_sample_start_at,
            self.in_sample_end_at_exclusive,
            self.out_of_sample_start_at,
            self.out_of_sample_end_at_exclusive,
        )
        if any(value.tzinfo is None for value in values):
            raise ValueError("validation window boundaries must be timezone-aware")
        self.in_sample_start_at = self.in_sample_start_at.astimezone(UTC)
        self.in_sample_end_at_exclusive = self.in_sample_end_at_exclusive.astimezone(UTC)
        self.out_of_sample_start_at = self.out_of_sample_start_at.astimezone(UTC)
        self.out_of_sample_end_at_exclusive = self.out_of_sample_end_at_exclusive.astimezone(UTC)
        if not (
            self.in_sample_start_at < self.in_sample_end_at_exclusive
            == self.out_of_sample_start_at
            < self.out_of_sample_end_at_exclusive
        ):
            raise ValueError("validation windows must be contiguous and non-empty")
        return self


class RuleStrategyValidationRunSummary(RuleStrategyValidationModel):
    """Selection-safe, tenant-scoped summary of one immutable validation run."""

    run_id: str
    strategy_id: str
    status: ValidationRunStatus
    source_preference: str
    selected_symbols: list[str]
    window: RuleStrategyValidationWindow
    initial_capital_quote: float
    data_fingerprint: str
    config_fingerprint: str
    assumptions_fingerprint: str
    artifact_fingerprint: str | None = None
    metrics: dict[str, Any] | None = None
    error_code: str | None = None
    error_detail: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class RuleStrategyValidationRunDetail(RuleStrategyValidationRunSummary):
    """Complete immutable run facts intended for a validation detail screen."""

    config_snapshot: dict[str, Any]
    assumptions: dict[str, Any]
    template_id: str | None = None
    template_version: int | None = None
    indicator_formula_version: str | None = None
    engine_version: str


class RuleStrategyValidationDatasetSummary(RuleStrategyValidationModel):
    """Immutable materialized data manifest associated with a validation run."""

    dataset_id: str
    run_id: str
    source_provider: str
    symbol: str
    interval: RuleInterval
    start_at: datetime
    end_at_exclusive: datetime
    bar_count: int
    content_hash: str
    coverage_manifest: dict[str, Any]
    page_manifest: list[dict[str, Any]]
    retrieved_at: datetime


class RuleStrategyValidationPointView(RuleStrategyValidationModel):
    """One timestamped account/equity observation in a validation window."""

    sequence: int
    window: ValidationWindowName
    observed_at: datetime
    equity_quote: float
    cash_quote: float
    position_quote: float
    drawdown_pct: float
    account_snapshot: dict[str, Any]
    decisions: dict[str, Any]


class RuleStrategyValidationFillView(RuleStrategyValidationModel):
    """One deterministic next-bar simulated fill."""

    sequence: int
    window: ValidationWindowName
    symbol: str
    leg_kind: Literal["entry", "add", "reduce", "close"]
    side: Literal["buy", "sell"]
    decision_at: datetime
    filled_at: datetime
    decision_price: float
    fill_price: float
    quantity: float
    quote_amount: float
    fee_quote: float
    slippage_pct: float
    realized_pnl_quote: float
    reason_code: str
    account_before: dict[str, Any]
    account_after: dict[str, Any]
