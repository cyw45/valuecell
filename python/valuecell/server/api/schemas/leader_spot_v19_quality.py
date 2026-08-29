"""Data-quality and reconnect-recovery contracts for V19."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Protocol, Sequence

from pydantic import Field, model_validator

from .leader_spot_v19 import LeaderSpotV19Model
from .leader_spot_v19_snapshots import LeaderSpotV19MarketInput


LeaderSpotV19QualityState = Literal["DATA_OK", "DATA_DEGRADED", "DATA_UNSAFE"]
LeaderSpotV19IssueSeverity = Literal["degraded", "unsafe"]


class LeaderSpotV19QualityModel(LeaderSpotV19Model):
    """Mutable normalization shell for quality and recovery facts."""

    model_config = {"extra": "forbid", "allow_inf_nan": False, "frozen": False}


class LeaderSpotV19Candle(LeaderSpotV19QualityModel):
    """Validated OHLCV bar used by continuity and anomaly checks."""

    timestamp_ms: int = Field(gt=0)
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> "LeaderSpotV19Candle":
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("candle high must cover open, close, and low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("candle low must not exceed candle prices")
        return self


class LeaderSpotV19PriceObservation(LeaderSpotV19QualityModel):
    """One source-stamped price used for an independent source cross-check."""

    symbol: str = Field(min_length=1, max_length=32)
    source: str = Field(min_length=1, max_length=32)
    price: float = Field(gt=0)
    observed_at: datetime

    @model_validator(mode="after")
    def normalize(self) -> "LeaderSpotV19PriceObservation":
        if self.observed_at.tzinfo is None:
            raise ValueError("price observation must be timezone-aware")
        self.observed_at = self.observed_at.astimezone(UTC)
        self.symbol = self.symbol.strip().upper().replace("/", "-")
        if not self.symbol.endswith("-USDT"):
            raise ValueError("V19 price observations must use USDT symbols")
        return self


class LeaderSpotV19QualityIssue(LeaderSpotV19QualityModel):
    """Explainable quality failure; unsafe issues always disable new entries."""

    code: str = Field(min_length=1, max_length=96)
    severity: LeaderSpotV19IssueSeverity
    detail: str = Field(min_length=1, max_length=500)
    symbol: str | None = Field(default=None, max_length=32)


class LeaderSpotV19DataQualityReport(LeaderSpotV19QualityModel):
    """Complete quality decision persisted before a V19 evaluation can proceed."""

    data_state: LeaderSpotV19QualityState
    observed_at: datetime
    issues: list[LeaderSpotV19QualityIssue] = Field(default_factory=list)
    checked_symbols: list[str] = Field(default_factory=list)
    fresh_input_count: int = Field(ge=0)
    required_input_count: int = Field(ge=0)
    accepted_for_entry: bool

    @model_validator(mode="after")
    def validate_report(self) -> "LeaderSpotV19DataQualityReport":
        if self.observed_at.tzinfo is None:
            raise ValueError("quality report timestamp must be timezone-aware")
        self.observed_at = self.observed_at.astimezone(UTC)
        if self.fresh_input_count > self.required_input_count:
            raise ValueError("fresh input count cannot exceed required input count")
        if self.data_state == "DATA_OK" and self.issues:
            raise ValueError("DATA_OK cannot contain quality issues")
        if self.accepted_for_entry and self.data_state != "DATA_OK":
            raise ValueError("only DATA_OK can be accepted for entry")
        return self


class LeaderSpotV19RecoveryExit(LeaderSpotV19QualityModel):
    """A locally triggered exit confirmed absent from the reconciled venue order set."""

    symbol: str = Field(min_length=1, max_length=32)
    quantity: float = Field(gt=0)
    reason_code: str = Field(min_length=1, max_length=96)
    local_triggered_at: datetime
    venue_order_id: str | None = None

    @model_validator(mode="after")
    def normalize(self) -> "LeaderSpotV19RecoveryExit":
        if self.local_triggered_at.tzinfo is None:
            raise ValueError("recovery exit timestamp must be timezone-aware")
        self.local_triggered_at = self.local_triggered_at.astimezone(UTC)
        self.symbol = self.symbol.strip().upper().replace("/", "-")
        if not self.symbol.endswith("-USDT"):
            raise ValueError("V19 recovery exits must use USDT symbols")
        return self


class LeaderSpotV19RecoveryObservation(LeaderSpotV19QualityModel):
    """Venue state obtained before any recovery exit submission."""

    positions: list[dict[str, object]] = Field(default_factory=list)
    orders: list[dict[str, object]] = Field(default_factory=list)
    due_exits: list[LeaderSpotV19RecoveryExit] = Field(default_factory=list)
    observed_at: datetime

    @model_validator(mode="after")
    def validate_observation(self) -> "LeaderSpotV19RecoveryObservation":
        if self.observed_at.tzinfo is None:
            raise ValueError("recovery observation timestamp must be timezone-aware")
        self.observed_at = self.observed_at.astimezone(UTC)
        return self


class LeaderSpotV19RecoveryVenue(Protocol):
    """Injected venue boundary for reconnect reconciliation and exit completion."""

    async def reconcile(
        self, tenant_id: str, strategy_id: str, batch_id: str
    ) -> LeaderSpotV19RecoveryObservation: ...

    async def submit_recovery_exit(
        self,
        tenant_id: str,
        strategy_id: str,
        batch_id: str,
        exit_request: LeaderSpotV19RecoveryExit,
    ) -> str: ...


class LeaderSpotV19QualityInput(LeaderSpotV19QualityModel):
    """All source facts needed for one deterministic quality decision."""

    market_inputs: list[LeaderSpotV19MarketInput] = Field(default_factory=list)
    primary_prices: list[LeaderSpotV19PriceObservation] = Field(default_factory=list)
    secondary_prices: list[LeaderSpotV19PriceObservation] = Field(default_factory=list)
    btc_prices: list[LeaderSpotV19PriceObservation] = Field(default_factory=list)
    btc_secondary_prices: list[LeaderSpotV19PriceObservation] = Field(default_factory=list)
    required_symbols: list[str] = Field(default_factory=list)
    observed_at: datetime

    @model_validator(mode="after")
    def normalize(self) -> "LeaderSpotV19QualityInput":
        if self.observed_at.tzinfo is None:
            raise ValueError("quality input timestamp must be timezone-aware")
        self.observed_at = self.observed_at.astimezone(UTC)
        self.required_symbols = [
            symbol.strip().upper().replace("/", "-") for symbol in self.required_symbols
        ]
        return self
