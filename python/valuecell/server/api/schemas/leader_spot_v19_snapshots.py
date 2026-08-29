"""Versioned V19 ranking and market-snapshot contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Protocol, Sequence

from pydantic import ConfigDict, Field, model_validator

from .leader_spot_v19 import LeaderSpotV19Model


class LeaderSpotV19SnapshotModel(LeaderSpotV19Model):
    """Mutable normalization shell for immutable persisted snapshot facts."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=False)


class LeaderSpotV19RankingItem(LeaderSpotV19SnapshotModel):
    """One provider-ranked, spot-USDT candidate before strategy filtering."""

    symbol: str = Field(min_length=1, max_length=32)
    rank: int = Field(ge=1)
    quote_volume_24h: float = Field(ge=0)
    listing_at: datetime | None = None
    spot_tradable: bool
    quote_asset: Literal["USDT"]
    provider_payload: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_symbol(self) -> "LeaderSpotV19RankingItem":
        normalized = self.symbol.strip().upper().replace("/", "-")
        if not normalized.endswith("-USDT"):
            raise ValueError("V19 ranking items must use USDT symbols")
        self.symbol = normalized
        if self.listing_at is not None:
            if self.listing_at.tzinfo is None:
                raise ValueError("listing_at must be timezone-aware")
            self.listing_at = self.listing_at.astimezone(UTC)
        return self


class LeaderSpotV19RankingSnapshot(LeaderSpotV19SnapshotModel):
    """Complete source snapshot; incomplete or expired snapshots cannot open trades."""

    source: Literal["okx"]
    observed_at: datetime
    expires_at: datetime
    items: list[LeaderSpotV19RankingItem] = Field(min_length=1)
    source_snapshot_id: str = Field(min_length=1, max_length=128)
    completeness: Literal["complete", "partial", "unsafe"] = "complete"

    @model_validator(mode="after")
    def validate_window_and_items(self) -> "LeaderSpotV19RankingSnapshot":
        if self.observed_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("ranking timestamps must be timezone-aware")
        self.observed_at = self.observed_at.astimezone(UTC)
        self.expires_at = self.expires_at.astimezone(UTC)
        if self.expires_at < self.observed_at:
            raise ValueError("ranking expiry cannot precede observation")
        ranks = [item.rank for item in self.items]
        symbols = [item.symbol for item in self.items]
        if len(ranks) != len(set(ranks)):
            raise ValueError("ranking ranks must be unique")
        if len(symbols) != len(set(symbols)):
            raise ValueError("ranking symbols must be unique")
        return self


class LeaderSpotV19BookLevel(LeaderSpotV19SnapshotModel):
    """One order-book price/quantity level captured for a candidate."""

    price: float = Field(gt=0)
    quantity: float = Field(ge=0)


class LeaderSpotV19OrderBookSnapshot(LeaderSpotV19SnapshotModel):
    """Bounded five-level book used by later depth and slippage gates."""

    symbol: str = Field(min_length=1, max_length=32)
    bids: list[LeaderSpotV19BookLevel] = Field(min_length=1, max_length=5)
    asks: list[LeaderSpotV19BookLevel] = Field(min_length=1, max_length=5)
    observed_at: datetime
    source: Literal["okx"]

    @model_validator(mode="after")
    def validate_book(self) -> "LeaderSpotV19OrderBookSnapshot":
        if self.observed_at.tzinfo is None:
            raise ValueError("order-book timestamp must be timezone-aware")
        self.observed_at = self.observed_at.astimezone(UTC)
        self.symbol = self.symbol.strip().upper().replace("/", "-")
        if not self.symbol.endswith("-USDT"):
            raise ValueError("V19 order books must use USDT symbols")
        if any(
            current.price > previous.price
            for previous, current in zip(self.bids, self.bids[1:])
        ):
            raise ValueError("bid levels must be descending")
        if any(
            current.price < previous.price
            for previous, current in zip(self.asks, self.asks[1:])
        ):
            raise ValueError("ask levels must be ascending")
        return self


class LeaderSpotV19MarketInput(LeaderSpotV19SnapshotModel):
    """One source-stamped candidate market payload for one required interval."""

    symbol: str = Field(min_length=1, max_length=32)
    interval: Literal["1m", "5m", "15m"]
    source: Literal["okx", "market_service"]
    candles: list[dict[str, float | int]] = Field(min_length=1)
    latest_price: float = Field(gt=0)
    order_book: LeaderSpotV19OrderBookSnapshot | None = None
    observed_at: datetime
    expires_at: datetime

    @model_validator(mode="after")
    def validate_market_window(self) -> "LeaderSpotV19MarketInput":
        if self.observed_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("market input timestamps must be timezone-aware")
        self.observed_at = self.observed_at.astimezone(UTC)
        self.expires_at = self.expires_at.astimezone(UTC)
        if self.expires_at < self.observed_at:
            raise ValueError("market input expiry cannot precede observation")
        self.symbol = self.symbol.strip().upper().replace("/", "-")
        if not self.symbol.endswith("-USDT"):
            raise ValueError("V19 market inputs must use USDT symbols")
        return self


class LeaderSpotV19MarketSnapshotProvider(Protocol):
    """Injected provider boundary; network access stays outside the collector."""

    async def fetch_market_inputs(
        self,
        symbols: Sequence[str],
        intervals: Sequence[Literal["1m", "5m", "15m"]],
    ) -> list[LeaderSpotV19MarketInput]: ...


class LeaderSpotV19RankingProvider(Protocol):
    """Injected OKX ranking boundary; no guessed endpoint is embedded in strategy code."""

    async def fetch_ranking(self) -> LeaderSpotV19RankingSnapshot: ...
