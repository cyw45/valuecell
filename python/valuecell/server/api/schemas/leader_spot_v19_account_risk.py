"""V19 account-risk contracts and pending-entry cancellation results."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, model_validator

from .leader_spot_v19_quality import LeaderSpotV19QualityModel


class LeaderSpotV19AccountRiskInput(LeaderSpotV19QualityModel):
    account_id: str
    daily_realized_pnl_quote: float
    daily_loss_limit_quote: float = Field(gt=0)
    daily_loss_reset_at: datetime
    prior_close_equity_quote: float = Field(gt=0)
    equity_quote: float = Field(ge=0)
    observed_at: datetime

    @model_validator(mode="after")
    def normalize(self) -> "LeaderSpotV19AccountRiskInput":
        if self.daily_loss_reset_at.tzinfo is None or self.observed_at.tzinfo is None:
            raise ValueError("risk timestamps must be timezone-aware")
        self.daily_loss_reset_at = self.daily_loss_reset_at.astimezone(UTC)
        self.observed_at = self.observed_at.astimezone(UTC)
        return self


class LeaderSpotV19AccountRiskDecision(LeaderSpotV19QualityModel):
    state: Literal["normal", "daily_loss_halted", "equity_halted"]
    can_open: bool
    daily_realized_pnl_quote: float
    daily_loss_reset_at: datetime
    equity_drawdown_pct: float = Field(ge=0)
    halt_until: datetime | None = None
    reason_code: str | None = None
    cancel_pending_entries: bool
    force_close_positions: bool
    observed_at: datetime


class LeaderSpotV19RiskCancellationResult(LeaderSpotV19QualityModel):
    cancelled_intent_ids: list[str] = Field(default_factory=list)
    preserved_intent_ids: list[str] = Field(default_factory=list)
