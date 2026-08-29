"""V19 profit-protection and exit decision contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, model_validator

from .leader_spot_v19_quality import LeaderSpotV19QualityModel


class LeaderSpotV19PositionExitInput(LeaderSpotV19QualityModel):
    position_id: str
    symbol: str
    entry_price: float = Field(gt=0)
    quantity: float = Field(gt=0)
    entry_time: datetime
    protection_status: Literal["PROTECTION_NONE", "PROTECTION_PENDING", "PROTECTION_ACTIVE"]
    protection_started_at: datetime | None = None
    peak_price: float = Field(gt=0)
    peak_profit_pct: float = Field(ge=0)
    moving_stop_price: float = Field(gt=0)
    layered_exit_price: float | None = Field(default=None, gt=0)
    loss_circuit_active: bool
    hard_stop_two_source_confirmed: bool = False
    hard_stop_local_bid_persistent: bool = False
    trend_data_valid: bool = False
    trend_break_count: int = Field(ge=0)
    current_bid: float = Field(gt=0)
    closed_one_minute_high: float | None = Field(default=None, gt=0)
    fifteen_minute_closes: list[float] = Field(default_factory=list)
    market_state: Literal["M2", "M3", "M4"]
    observed_at: datetime

    @model_validator(mode="after")
    def normalize(self) -> "LeaderSpotV19PositionExitInput":
        if self.entry_time.tzinfo is None or self.observed_at.tzinfo is None:
            raise ValueError("exit timestamps must be timezone-aware")
        self.entry_time = self.entry_time.astimezone(UTC)
        self.observed_at = self.observed_at.astimezone(UTC)
        if self.protection_started_at is not None:
            if self.protection_started_at.tzinfo is None:
                raise ValueError("protection_started_at must be timezone-aware")
            self.protection_started_at = self.protection_started_at.astimezone(UTC)
        return self


class LeaderSpotV19ExitDecision(LeaderSpotV19QualityModel):
    position_id: str
    protection_status: Literal["PROTECTION_NONE", "PROTECTION_PENDING", "PROTECTION_ACTIVE"]
    peak_price: float
    peak_profit_pct: float
    moving_stop_price: float
    layered_exit_price: float | None
    loss_circuit_active: bool
    trend_break_count: int
    exit_reason_code: Literal[
        "STOP_LOSS_8PCT",
        "LOSS_CIRCUIT_7D",
        "MOVING_STOP",
        "LAYERED_RETRACEMENT",
        "TREND_EXIT",
    ] | None = None
    order_type: Literal["market", "limit"] | None = None
    limit_price: float | None = Field(default=None, gt=0)
    observed_at: datetime
