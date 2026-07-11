"""Deterministic paper replay of explicitly supplied prediction-market books."""

from __future__ import annotations

import hashlib
import json
import math

from valuecell.server.api.schemas.prediction_market_replay import (
    PredictionMarketReplayBookLevel,
    PredictionMarketReplayFill,
    PredictionMarketReplayMarkToBook,
    PredictionMarketReplayPreviewData,
    PredictionMarketReplayPreviewRequest,
    PredictionMarketReplaySnapshot,
    PredictionMarketReplayAssumptions,
)

_MAX_REPLAY_LEVELS = 100
_FRESHNESS_STALE_MS = 60_000


class PredictionMarketReplayService:
    """Replay a visible frozen order book without fetching or placing an order."""

    def preview(
        self, request: PredictionMarketReplayPreviewRequest
    ) -> PredictionMarketReplayPreviewData:
        """Create a deterministic IOC-style paper replay preview."""
        _validate_request(request)
        eligible_time_ms = request.decision_time_ms + request.latency_ms
        snapshot = next(
            (
                candidate
                for candidate in request.snapshots
                if candidate.source_timestamp_ms >= eligible_time_ms
            ),
            None,
        )
        fingerprint = _fingerprint(request)
        assumptions = PredictionMarketReplayAssumptions(
            eligible_time_ms=eligible_time_ms,
            execution_snapshot_timestamp_ms=(
                snapshot.source_timestamp_ms if snapshot is not None else None
            ),
            max_levels=request.order.max_levels,
            extra_slippage_bps=request.order.extra_slippage_bps,
            canceled_remainder=True,
        )
        if snapshot is None:
            return PredictionMarketReplayPreviewData(
                source_timestamp_ms=None,
                observed_at_ms=None,
                freshness_age_ms=None,
                freshness_status="unavailable",
                fingerprint=fingerprint,
                assumptions=assumptions,
                fill=PredictionMarketReplayFill(
                    requested_size=request.order.size,
                    filled_size=0.0,
                    unfilled_size=request.order.size,
                    vwap=None,
                    levels_consumed=0,
                ),
                mark_to_book=PredictionMarketReplayMarkToBook(
                    mark_price=None,
                    pnl=0.0,
                ),
            )

        filled_size, notional, levels_consumed = _walk_visible_levels(
            snapshot=snapshot,
            side=request.order.side,
            requested_size=request.order.size,
            max_levels=request.order.max_levels,
            extra_slippage_bps=request.order.extra_slippage_bps,
        )
        vwap = notional / filled_size if filled_size else None
        mark_price = (snapshot.bids[0].price + snapshot.asks[0].price) / 2
        pnl = _mark_to_book_pnl(request.order.side, filled_size, vwap, mark_price)
        freshness_age_ms = snapshot.observed_at_ms - snapshot.source_timestamp_ms
        canceled_remainder = filled_size < request.order.size
        assumptions.canceled_remainder = canceled_remainder
        return PredictionMarketReplayPreviewData(
            source_timestamp_ms=snapshot.source_timestamp_ms,
            observed_at_ms=snapshot.observed_at_ms,
            freshness_age_ms=freshness_age_ms,
            freshness_status=(
                "fresh" if freshness_age_ms <= _FRESHNESS_STALE_MS else "stale"
            ),
            fingerprint=fingerprint,
            assumptions=assumptions,
            fill=PredictionMarketReplayFill(
                requested_size=request.order.size,
                filled_size=filled_size,
                unfilled_size=request.order.size - filled_size,
                vwap=vwap,
                levels_consumed=levels_consumed,
            ),
            mark_to_book=PredictionMarketReplayMarkToBook(
                mark_price=mark_price,
                pnl=pnl,
            ),
        )


def _validate_request(request: PredictionMarketReplayPreviewRequest) -> None:
    _require_non_negative_integer(request.decision_time_ms, "decision_time_ms")
    _require_non_negative_integer(request.latency_ms, "latency_ms")
    if not request.snapshots:
        raise ValueError("snapshots must contain at least one frozen order book")
    if not _is_positive_finite(request.order.size):
        raise ValueError("order.size must be a positive finite number")
    if not isinstance(request.order.max_levels, int) or isinstance(
        request.order.max_levels, bool
    ):
        raise ValueError("order.max_levels must be an integer")
    if not 1 <= request.order.max_levels <= _MAX_REPLAY_LEVELS:
        raise ValueError(f"order.max_levels must be between 1 and {_MAX_REPLAY_LEVELS}")
    if not _is_non_negative_finite(request.order.extra_slippage_bps):
        raise ValueError("order.extra_slippage_bps must be non-negative and finite")
    if request.order.extra_slippage_bps > 10_000:
        raise ValueError("order.extra_slippage_bps must not exceed 10000")

    prior_timestamp: int | None = None
    for index, snapshot in enumerate(request.snapshots):
        _validate_snapshot(snapshot, index)
        if prior_timestamp is not None and snapshot.source_timestamp_ms <= prior_timestamp:
            raise ValueError("snapshots must be strictly ordered by source_timestamp_ms")
        prior_timestamp = snapshot.source_timestamp_ms


def _validate_snapshot(snapshot: PredictionMarketReplaySnapshot, index: int) -> None:
    _require_non_negative_integer(
        snapshot.source_timestamp_ms, f"snapshots[{index}].source_timestamp_ms"
    )
    _require_non_negative_integer(
        snapshot.observed_at_ms, f"snapshots[{index}].observed_at_ms"
    )
    if snapshot.observed_at_ms < snapshot.source_timestamp_ms:
        raise ValueError("snapshot observed_at_ms must not precede source_timestamp_ms")
    _validate_book_side(snapshot.bids, "bids", descending=True, snapshot_index=index)
    _validate_book_side(snapshot.asks, "asks", descending=False, snapshot_index=index)
    if snapshot.bids[0].price >= snapshot.asks[0].price:
        raise ValueError("snapshot best bid must be below best ask")


def _validate_book_side(
    levels: list[PredictionMarketReplayBookLevel],
    name: str,
    *,
    descending: bool,
    snapshot_index: int,
) -> None:
    if not levels:
        raise ValueError(f"snapshots[{snapshot_index}].{name} must be non-empty")
    prices: list[float] = []
    for level_index, level in enumerate(levels):
        if not _is_valid_probability_price(level.price):
            raise ValueError(
                f"snapshots[{snapshot_index}].{name}[{level_index}].price must be in (0, 1]"
            )
        if not _is_positive_finite(level.size):
            raise ValueError(
                f"snapshots[{snapshot_index}].{name}[{level_index}].size must be positive and finite"
            )
        prices.append(level.price)
    expected = sorted(prices, reverse=descending)
    if prices != expected or len(prices) != len(set(prices)):
        direction = "strictly descending" if descending else "strictly ascending"
        raise ValueError(f"snapshots[{snapshot_index}].{name} prices must be {direction}")


def _walk_visible_levels(
    *,
    snapshot: PredictionMarketReplaySnapshot,
    side: str,
    requested_size: float,
    max_levels: int,
    extra_slippage_bps: float,
) -> tuple[float, float, int]:
    levels = snapshot.asks if side == "buy" else snapshot.bids
    remaining = requested_size
    filled_size = 0.0
    notional = 0.0
    levels_consumed = 0
    adjustment = extra_slippage_bps / 10_000
    for level in levels[:max_levels]:
        if remaining <= 0:
            break
        fill_size = min(remaining, level.size)
        execution_price = (
            level.price * (1 + adjustment)
            if side == "buy"
            else level.price * (1 - adjustment)
        )
        filled_size += fill_size
        notional += fill_size * execution_price
        remaining -= fill_size
        levels_consumed += 1
    return filled_size, notional, levels_consumed


def _mark_to_book_pnl(
    side: str, filled_size: float, vwap: float | None, mark_price: float
) -> float:
    if vwap is None:
        return 0.0
    return (
        (mark_price - vwap) * filled_size
        if side == "buy"
        else (vwap - mark_price) * filled_size
    )


def _fingerprint(request: PredictionMarketReplayPreviewRequest) -> str:
    payload = json.dumps(
        {"version": 1, "request": request.model_dump(mode="json")},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_non_negative_integer(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _is_positive_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _is_non_negative_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _is_valid_probability_price(value: object) -> bool:
    return _is_positive_finite(value) and value <= 1
