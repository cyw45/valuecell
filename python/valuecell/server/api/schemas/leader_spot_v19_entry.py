"""Fixed-amount, three-tier V19 entry execution contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import Field, model_validator

from .leader_spot_v19_candidate import LeaderSpotV19CandidateDecision
from .leader_spot_v19_quality import LeaderSpotV19QualityModel
from .leader_spot_v19_snapshots import LeaderSpotV19OrderBookSnapshot


class LeaderSpotV19EntryRequest(LeaderSpotV19QualityModel):
    """Scheduler-owned entry request; it cannot override V19 fixed-money limits."""

    signal_id: str = Field(min_length=1, max_length=100)
    candidate: LeaderSpotV19CandidateDecision
    confirmation_price: float = Field(gt=0)
    open_position_count: int = Field(ge=0, le=6)
    held_symbols: list[str] = Field(default_factory=list)
    cooldown_symbols: list[str] = Field(default_factory=list)
    observed_at: datetime

    @model_validator(mode="after")
    def validate_request(self) -> "LeaderSpotV19EntryRequest":
        if self.observed_at.tzinfo is None:
            raise ValueError("entry observation must be timezone-aware")
        self.observed_at = self.observed_at.astimezone(UTC)
        self.held_symbols = [item.strip().upper().replace("/", "-") for item in self.held_symbols]
        self.cooldown_symbols = [item.strip().upper().replace("/", "-") for item in self.cooldown_symbols]
        return self


class LeaderSpotV19EntryTier(LeaderSpotV19QualityModel):
    """One immutable V19 limit-entry tier."""

    tier: Literal[1, 2, 3]
    offset_pct: float = Field(gt=0)
    wait_seconds: int = Field(gt=0)


class LeaderSpotV19EntryOrderResult(LeaderSpotV19QualityModel):
    """Venue-normalized status after a limit submit, wait, or cancellation."""

    client_order_id: str = Field(min_length=1, max_length=128)
    venue_order_id: str | None = Field(default=None, max_length=128)
    status: Literal["filled", "open", "cancelled", "rejected", "submission_unknown"]
    filled_quantity: float = Field(default=0, ge=0)
    average_price: float | None = Field(default=None, gt=0)
    fee_quote: float = Field(default=0, ge=0)


class LeaderSpotV19EntryVenue(Protocol):
    """Injected venue boundary. Calls occur only from a background scheduler worker."""

    venue: str

    async def current_order_book(self, symbol: str) -> LeaderSpotV19OrderBookSnapshot: ...

    async def submit_limit_buy(
        self,
        *,
        client_order_id: str,
        symbol: str,
        quote_amount: float,
        price: float,
    ) -> LeaderSpotV19EntryOrderResult: ...

    async def wait_for_order(
        self, client_order_id: str, timeout_seconds: int
    ) -> LeaderSpotV19EntryOrderResult: ...

    async def cancel_order(self, client_order_id: str) -> LeaderSpotV19EntryOrderResult: ...


class LeaderSpotV19EntryDecision(LeaderSpotV19QualityModel):
    """Auditable result of all attempted V19 entry tiers."""

    accepted: bool
    reason_code: str | None
    symbol: str
    order_amount_quote: float
    tier_results: list[LeaderSpotV19EntryOrderResult] = Field(default_factory=list)
    position_id: str | None = None
    observed_at: datetime
