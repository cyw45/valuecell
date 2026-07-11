"""Public Polymarket research API contracts.

These contracts deliberately expose public market observations and paper-only
research data. They contain no credential, wallet, or order-execution field.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .base import SuccessResponse


class PredictionMarketOutcomeData(BaseModel):
    outcome: str
    token_id: str
    price: str | None = None


class PredictionMarketSummaryData(BaseModel):
    market_id: str
    slug: str = ""
    question: str
    active: bool
    closed: bool
    outcomes: list[PredictionMarketOutcomeData]


class PredictionMarketCatalogData(BaseModel):
    source: Literal["polymarket-public"] = "polymarket-public"
    mode: Literal["paper"] = "paper"
    source_timestamp_ms: int
    observed_at_ms: int
    freshness_age_ms: int
    freshness_status: Literal["fresh", "delayed", "stale", "unavailable"]
    markets: list[PredictionMarketSummaryData]
    next_cursor: str | None = None
    warnings: list[str] = Field(default_factory=list)


class PredictionMarketBookLevelData(BaseModel):
    price: str
    size: str


class PredictionMarketBookHealthData(BaseModel):
    status: Literal["valid", "crossed", "one_sided", "empty", "invalid", "stale"]
    reason: str | None = None
    crossed: bool = False
    one_sided: bool = False
    bid_levels: int = 0
    ask_levels: int = 0


class PredictionMarketOrderBookData(BaseModel):
    bids: list[PredictionMarketBookLevelData]
    asks: list[PredictionMarketBookLevelData]
    best_bid: str | None = None
    best_ask: str | None = None
    midpoint: str | None = None
    microprice: str | None = None
    health: PredictionMarketBookHealthData


class PredictionMarketSignalData(BaseModel):
    reference_price: str | None = None
    reference_method: Literal["microprice", "midpoint", "unavailable"] = "unavailable"
    volatility: str | None = None
    observation_count: int = 0
    volatility_status: Literal["available", "insufficient_history", "invalid_history"]


class PredictionMarketSnapshotData(BaseModel):
    source: Literal["polymarket-public"] = "polymarket-public"
    mode: Literal["paper"] = "paper"
    source_timestamp_ms: int
    observed_at_ms: int
    freshness_age_ms: int
    freshness_status: Literal["fresh", "delayed", "stale", "unavailable"]
    market_id: str
    question: str
    outcome: str
    token_id: str
    book: PredictionMarketOrderBookData
    signal: PredictionMarketSignalData | None = None
    warnings: list[str] = Field(default_factory=list)


PredictionMarketCatalogResponse = SuccessResponse[PredictionMarketCatalogData]
PredictionMarketSnapshotResponse = SuccessResponse[PredictionMarketSnapshotData]
