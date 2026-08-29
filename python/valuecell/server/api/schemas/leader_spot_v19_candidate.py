"""Candidate-funnel contracts for the isolated V19 strategy."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, model_validator

from .leader_spot_v19_quality import LeaderSpotV19QualityModel
from .leader_spot_v19_snapshots import LeaderSpotV19OrderBookSnapshot


class LeaderSpotV19BoxBreakoutEvidence(LeaderSpotV19QualityModel):
    """Precomputed V16.1 box evidence; absent parameters fail closed."""

    parameter_source: Literal["V16.1"]
    parameter_fingerprint: str = Field(min_length=1, max_length=128)
    upper_bound: float = Field(gt=0)
    fifteen_minute_close_confirmed: bool
    five_minute_close_confirmations: int = Field(ge=0)
    second_five_minute_volume_confirmed: bool
    volume_multiplier: float = Field(gt=0)
    passed: bool


class LeaderSpotV19ScoreEvidence(LeaderSpotV19QualityModel):
    """A reproducible upstream score instead of an invented local formula."""

    formula_source: str = Field(min_length=1, max_length=64)
    formula_fingerprint: str = Field(min_length=1, max_length=128)
    total_score: float = Field(ge=0)
    factors: dict[str, float] = Field(min_length=1)


class LeaderSpotV19CandidateInput(LeaderSpotV19QualityModel):
    """Fully materialized facts for the ordered V19 candidate funnel."""

    symbol: str = Field(min_length=1, max_length=32)
    source_rank: int = Field(ge=1)
    market_state: Literal["M0", "M1", "M2", "M3", "M4"]
    data_state: Literal["DATA_OK", "DATA_DEGRADED", "DATA_UNSAFE"]
    quote_volume_24h: float = Field(ge=0)
    listing_at: datetime
    relative_strength_rank_pct: float = Field(ge=0, le=1)
    return_24h_pct: float
    high_pump_retest_confirmed: bool = False
    needle_detected: bool = False
    br_value: float = Field(ge=0)
    order_book: LeaderSpotV19OrderBookSnapshot | None = None
    box: LeaderSpotV19BoxBreakoutEvidence | None = None
    strict_new_coin_requirements_met: bool = False
    enhanced_depth_confirmed: bool = False
    score: LeaderSpotV19ScoreEvidence | None = None
    estimated_entry_slippage_pct: float | None = Field(default=None, ge=0)
    observed_at: datetime

    @model_validator(mode="after")
    def normalize(self) -> "LeaderSpotV19CandidateInput":
        if self.listing_at.tzinfo is None or self.observed_at.tzinfo is None:
            raise ValueError("candidate timestamps must be timezone-aware")
        self.listing_at = self.listing_at.astimezone(UTC)
        self.observed_at = self.observed_at.astimezone(UTC)
        if self.listing_at > self.observed_at:
            raise ValueError("listing_at cannot be after candidate observation")
        self.symbol = self.symbol.strip().upper().replace("/", "-")
        if not self.symbol.endswith("-USDT"):
            raise ValueError("V19 candidates must use USDT symbols")
        if self.order_book is not None and self.order_book.symbol != self.symbol:
            raise ValueError("candidate order book must match candidate symbol")
        return self


class LeaderSpotV19CandidateStep(LeaderSpotV19QualityModel):
    """One ordered funnel result, retained even after a later rejection."""

    stage: Literal[
        "entry_state",
        "liquidity",
        "new_coin",
        "relative_strength",
        "anomaly",
        "box_breakout",
        "score",
        "order_book",
    ]
    passed: bool
    reason_code: str | None = Field(default=None, max_length=96)
    facts: dict[str, float | int | bool | str | None] = Field(default_factory=dict)


class LeaderSpotV19CandidateDecision(LeaderSpotV19QualityModel):
    """Ordered candidate decision available to persistence and both clients."""

    symbol: str
    source_rank: int
    accepted: bool
    score: float | None
    reason_code: str | None
    steps: list[LeaderSpotV19CandidateStep]
    observed_at: datetime
