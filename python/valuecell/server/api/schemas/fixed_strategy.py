"""Pure signal contracts for the three code-owned strategy engines."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FixedStrategyKind = Literal["dual_ma_trend", "pair_rotation", "leader_breakout"]
FixedSignalAction = Literal[
    "long_entry",
    "short_entry",
    "exit",
    "hold",
    "blocked",
    "no_signal",
]


class FixedStrategyModel(BaseModel):
    """Strict finite contract shared by fixed-strategy pure functions."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class FixedCandle(FixedStrategyModel):
    """One closed UTC bar supplied by the platform market boundary."""

    symbol: str = Field(min_length=1, max_length=32)
    timestamp_ms: int = Field(gt=0)
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    quote_volume: float | None = Field(default=None, ge=0)
    is_closed: bool = True


class FixedPosition(FixedStrategyModel):
    """Strategy-owned position state used as pure-engine input."""

    symbol: str = Field(min_length=1, max_length=32)
    side: Literal["long", "short"]
    quantity: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    entry_timestamp_ms: int = Field(gt=0)
    peak_price: float | None = Field(default=None, gt=0)
    pair: str | None = Field(default=None, max_length=64)


class FixedCondition(FixedStrategyModel):
    """Human-readable, persisted decision fact."""

    code: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=255)
    state: Literal["triggered", "not_triggered", "blocked", "unavailable"]
    actual: float | str | bool | None = None
    threshold: float | str | bool | None = None
    operator: str | None = Field(default=None, max_length=16)
    detail: str = Field(min_length=1, max_length=1_000)
    data_timestamp_ms: int | None = Field(default=None, gt=0)


class FixedStrategySignal(FixedStrategyModel):
    """Strategy output; execution is performed by a separate platform layer."""

    kind: FixedStrategyKind
    symbol: str = Field(min_length=1, max_length=32)
    action: FixedSignalAction
    reason_code: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=2_000)
    conditions: list[FixedCondition] = Field(default_factory=list)
    indicators: dict[str, float | str | bool | None] = Field(default_factory=dict)
    observed_at: datetime
    pair: str | None = Field(default=None, max_length=64)
    execution_block_reason: str | None = Field(default=None, max_length=1_000)


class FixedEngineInput(FixedStrategyModel):
    """Common engine input with already-final platform candles."""

    candles: list[FixedCandle] = Field(min_length=1, max_length=5_000)
    position: FixedPosition | None = None
    observed_at: datetime
