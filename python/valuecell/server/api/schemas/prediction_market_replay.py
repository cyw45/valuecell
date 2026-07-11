"""Deterministic, paper-only prediction-market order-book replay contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .base import SuccessResponse


class PredictionMarketReplayBookLevel(BaseModel):
    """One visible, frozen CLOB price level."""

    price: float
    size: float


class PredictionMarketReplaySnapshot(BaseModel):
    """An explicitly supplied public CLOB snapshot; it is never live-fetched."""

    source_timestamp_ms: int
    observed_at_ms: int
    bids: list[PredictionMarketReplayBookLevel]
    asks: list[PredictionMarketReplayBookLevel]


class PredictionMarketReplayOrder(BaseModel):
    """A paper IOC-style order replayed against visible liquidity."""

    side: Literal["buy", "sell"]
    size: float
    max_levels: int = Field(default=10)
    extra_slippage_bps: float = Field(default=0.0)


class PredictionMarketReplayPreviewRequest(BaseModel):
    """Frozen replay tape and the paper decision parameters."""

    decision_time_ms: int
    latency_ms: int
    order: PredictionMarketReplayOrder
    snapshots: list[PredictionMarketReplaySnapshot]


class PredictionMarketReplayAssumptions(BaseModel):
    """Explicit deterministic assumptions used by the paper simulation."""

    eligible_time_ms: int
    execution_snapshot_timestamp_ms: int | None
    max_levels: int
    extra_slippage_bps: float
    canceled_remainder: bool
    remainder_policy: Literal["cancel"] = "cancel"
    liquidity_scope: Literal["visible_frozen_levels"] = "visible_frozen_levels"


class PredictionMarketReplayFill(BaseModel):
    """Paper fills produced from the selected frozen visible book."""

    requested_size: float
    filled_size: float
    unfilled_size: float
    vwap: float | None
    levels_consumed: int


class PredictionMarketReplayMarkToBook(BaseModel):
    """Simulated PnL against the selected snapshot's visible mid-price."""

    mark_price: float | None
    pnl: float
    currency: Literal["quote"] = "quote"


class PredictionMarketReplayPreviewData(BaseModel):
    """Paper-only deterministic replay output; not an executable order result."""

    source: Literal["polymarket-public"] = "polymarket-public"
    mode: Literal["paper"] = "paper"
    simulation_mode: Literal["simulated"] = "simulated"
    source_timestamp_ms: int | None
    observed_at_ms: int | None
    freshness_age_ms: int | None
    freshness_status: Literal["fresh", "stale", "unavailable"]
    fingerprint: str = Field(
        description="SHA-256 of the canonical versioned frozen replay input."
    )
    assumptions: PredictionMarketReplayAssumptions
    fill: PredictionMarketReplayFill
    mark_to_book: PredictionMarketReplayMarkToBook


PredictionMarketReplayPreviewResponse = SuccessResponse[PredictionMarketReplayPreviewData]
