"""Frozen-data V19 backtest and walk-forward contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, model_validator

from .leader_spot_v19_quality import LeaderSpotV19QualityModel


class LeaderSpotV19BacktestCandle(LeaderSpotV19QualityModel):
    symbol: str = Field(min_length=1, max_length=32)
    timestamp_ms: int = Field(gt=0)
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)


class LeaderSpotV19BacktestSignal(LeaderSpotV19QualityModel):
    symbol: str = Field(min_length=1, max_length=32)
    timestamp_ms: int = Field(gt=0)
    action: Literal["entry", "close"]
    reason_code: str = Field(min_length=1, max_length=128)


class LeaderSpotV19BacktestRequest(LeaderSpotV19QualityModel):
    initial_equity_quote: float = Field(gt=0)
    candles: list[LeaderSpotV19BacktestCandle] = Field(min_length=1)
    signals: list[LeaderSpotV19BacktestSignal] = Field(default_factory=list)
    config_snapshot: dict[str, object]
    data_source: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_coverage(self) -> "LeaderSpotV19BacktestRequest":
        timestamps = [item.timestamp_ms for item in self.candles]
        if len(timestamps) < 2:
            raise ValueError("backtest requires at least two candles")
        start = min(timestamps)
        end = max(timestamps)
        if end - start < 365 * 24 * 60 * 60 * 1_000:
            raise ValueError("backtest requires at least twelve months of frozen candles")
        return self


class LeaderSpotV19BacktestFill(LeaderSpotV19QualityModel):
    symbol: str
    side: Literal["buy", "sell"]
    decision_timestamp_ms: int
    fill_timestamp_ms: int
    decision_price: float
    fill_price: float
    quantity: float
    quote_amount: float
    fee_quote: float
    slippage_pct: float
    realized_pnl_quote: float
    reason_code: str


class LeaderSpotV19WalkForwardWindow(LeaderSpotV19QualityModel):
    train_start_ms: int
    train_end_ms: int
    test_start_ms: int
    test_end_ms: int
    test_metrics: dict[str, float | int]


class LeaderSpotV19BacktestResult(LeaderSpotV19QualityModel):
    data_fingerprint: str
    config_fingerprint: str
    assumptions_fingerprint: str
    fills: list[LeaderSpotV19BacktestFill]
    metrics: dict[str, float | int]
    walk_forward: list[LeaderSpotV19WalkForwardWindow]
