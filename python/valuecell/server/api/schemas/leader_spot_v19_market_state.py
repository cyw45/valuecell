"""V19 market-state and signal-starvation decision contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, model_validator

from .leader_spot_v19 import LeaderSpotV19Config
from .leader_spot_v19_quality import LeaderSpotV19QualityModel


LeaderSpotV19EntryProfile = Literal["halt", "degraded", "standard", "strong_trend"]


class LeaderSpotV19MarketStateInput(LeaderSpotV19QualityModel):
    """All persisted market and account facts required for a state decision."""

    data_state: Literal["DATA_OK", "DATA_DEGRADED", "DATA_UNSAFE"]
    up_ratio: float = Field(ge=0, le=1)
    volume_ratio_to_5d_average: float = Field(gt=0)
    fear_greed_index: int = Field(ge=0, le=100)
    funding_rate: float
    daily_loss_limit_reached: bool = False
    no_valid_candidate_since: datetime | None = None
    valid_candidate_count: int = Field(default=0, ge=0)
    observed_at: datetime

    @model_validator(mode="after")
    def normalize_timestamps(self) -> "LeaderSpotV19MarketStateInput":
        if self.observed_at.tzinfo is None:
            raise ValueError("market state observation must be timezone-aware")
        self.observed_at = self.observed_at.astimezone(UTC)
        if self.no_valid_candidate_since is not None:
            if self.no_valid_candidate_since.tzinfo is None:
                raise ValueError("no_valid_candidate_since must be timezone-aware")
            self.no_valid_candidate_since = self.no_valid_candidate_since.astimezone(UTC)
            if self.no_valid_candidate_since > self.observed_at:
                raise ValueError("no_valid_candidate_since cannot be in the future")
        return self


class LeaderSpotV19MarketCondition(LeaderSpotV19QualityModel):
    """One explainable market gate and its evaluated threshold."""

    code: str = Field(min_length=1, max_length=96)
    passed: bool
    actual: float | int | bool | str
    threshold: float | int | bool | str


class LeaderSpotV19SignalStarvationPolicy(LeaderSpotV19QualityModel):
    """The only candidate thresholds signal starvation may temporarily relax."""

    elapsed_hours: float = Field(ge=0)
    recovered: bool
    relative_strength_rank_pct: float = Field(gt=0, le=1)
    liquidity_quote: float = Field(gt=0)
    score_threshold: int = Field(ge=0)


class LeaderSpotV19MarketStateDecision(LeaderSpotV19QualityModel):
    """State and starvation result persisted before candidate selection."""

    market_state: Literal["M0", "M1", "M2", "M3", "M4"]
    entry_profile: LeaderSpotV19EntryProfile
    can_open: bool
    reason_codes: list[str] = Field(default_factory=list)
    conditions: list[LeaderSpotV19MarketCondition] = Field(default_factory=list)
    starvation: LeaderSpotV19SignalStarvationPolicy
    observed_at: datetime

    @model_validator(mode="after")
    def validate_state_permission(self) -> "LeaderSpotV19MarketStateDecision":
        if self.observed_at.tzinfo is None:
            raise ValueError("market decision timestamp must be timezone-aware")
        self.observed_at = self.observed_at.astimezone(UTC)
        if self.can_open != (self.market_state in {"M2", "M3", "M4"}):
            raise ValueError("opening permission must match the market state")
        return self


class LeaderSpotV19MarketStateRequest(LeaderSpotV19QualityModel):
    """Internal scheduler request; configuration is always the frozen V19 schema."""

    config: LeaderSpotV19Config
    inputs: LeaderSpotV19MarketStateInput
