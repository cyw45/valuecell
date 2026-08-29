"""Deterministic paper signal engine for the fixed SMA10/SMA20 strategy."""

from __future__ import annotations

from datetime import timezone

from valuecell.server.api.schemas.fixed_strategy import (
    FixedCondition,
    FixedEngineInput,
    FixedStrategySignal,
)

_KIND = "dual_ma_trend"
_FAST_PERIOD = 10
_SLOW_PERIOD = 20
_MIN_CANDLES = _SLOW_PERIOD + 2
_STOP_LOSS_PCT = 0.05
_MAX_HOLD_MS = 168 * 60 * 60 * 1000


def _observed_ms(request: FixedEngineInput) -> int:
    observed_at = request.observed_at
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    return int(observed_at.timestamp() * 1000)


def _condition(
    code: str,
    label: str,
    state: str,
    detail: str,
    *,
    actual: float | str | bool | None = None,
    threshold: float | str | bool | None = None,
    operator: str | None = None,
    timestamp_ms: int | None = None,
) -> FixedCondition:
    return FixedCondition(
        code=code,
        label=label,
        state=state,  # type: ignore[arg-type]
        actual=actual,
        threshold=threshold,
        operator=operator,
        detail=detail,
        data_timestamp_ms=timestamp_ms,
    )


def _signal(
    request: FixedEngineInput,
    symbol: str,
    action: str,
    reason_code: str,
    reason: str,
    conditions: list[FixedCondition],
    indicators: dict[str, float | str | bool | None],
    *,
    pair: str | None = None,
    execution_block_reason: str | None = None,
) -> FixedStrategySignal:
    return FixedStrategySignal(
        kind=_KIND,
        symbol=symbol,
        action=action,  # type: ignore[arg-type]
        reason_code=reason_code,
        reason=reason,
        conditions=conditions,
        indicators=indicators,
        observed_at=request.observed_at,
        pair=pair,
        execution_block_reason=execution_block_reason,
    )


class FixedDualMaEngine:
    """Evaluate supplied, already-closed 4h candles without side effects."""

    def evaluate(self, request: FixedEngineInput) -> FixedStrategySignal:
        candles = request.candles
        symbol = candles[-1].symbol if candles else "UNKNOWN"

        if not candles:
            return _signal(
                request,
                symbol,
                "blocked",
                "missing_candles",
                "At least one candle is required.",
                [
                    _condition(
                        "data.candles", "Closed candle count", "blocked",
                        "No candles were supplied.", actual=0,
                        threshold=_MIN_CANDLES, operator=">=",
                    )
                ],
                {"required_candles": float(_MIN_CANDLES), "candle_count": 0.0},
            )

        if any(candle.symbol != symbol for candle in candles):
            return _signal(
                request, symbol, "blocked", "mixed_symbols",
                "Candles contain more than one symbol; no signal can be inferred.",
                [_condition(
                    "data.symbols", "Candle symbols", "blocked",
                    "All supplied candles must belong to the same symbol.",
                    actual="mixed", threshold=symbol, operator="==",
                )],
                {"required_candles": float(_MIN_CANDLES), "candle_count": float(len(candles))},
            )

        if any(not candle.is_closed for candle in candles):
            unfinished = sum(not candle.is_closed for candle in candles)
            return _signal(
                request, symbol, "blocked", "unfinished_candles",
                "Every candle used by this 4h engine must be final (closed).",
                [_condition(
                    "data.closed", "All candles closed", "blocked",
                    "An unfinished candle was supplied; waiting for the 4h close.",
                    actual=False, threshold=True, operator="==",
                    timestamp_ms=candles[-1].timestamp_ms,
                )],
                {"required_candles": float(_MIN_CANDLES),
                 "candle_count": float(len(candles)), "unfinished_candles": float(unfinished)},
            )

        if any(
            current.timestamp_ms <= previous.timestamp_ms
            for previous, current in zip(candles, candles[1:])
        ):
            return _signal(
                request, symbol, "blocked", "invalid_candle_order",
                "Candle timestamps must be strictly increasing.",
                [_condition(
                    "data.timestamp_order", "Candle timestamp order", "blocked",
                    "Candles are not in strictly increasing chronological order.",
                    actual=False, threshold=True, operator="==",
                )],
                {"required_candles": float(_MIN_CANDLES), "candle_count": float(len(candles))},
            )

        if len(candles) < _MIN_CANDLES:
            return _signal(
                request, symbol, "blocked", "insufficient_candles",
                f"Need at least {_MIN_CANDLES} closed candles for SMA10/SMA20 cross detection.",
                [_condition(
                    "data.candles", "Closed candle count", "blocked",
                    "Insufficient final candles for the previous and current SMA10 values.",
                    actual=float(len(candles)), threshold=float(_MIN_CANDLES), operator=">=",
                    timestamp_ms=candles[-1].timestamp_ms,
                )],
                {"required_candles": float(_MIN_CANDLES), "candle_count": float(len(candles))},
            )

        closes = [candle.close for candle in candles]
        previous_sma10 = sum(closes[-_FAST_PERIOD - 1 : -1]) / _FAST_PERIOD
        current_sma10 = sum(closes[-_FAST_PERIOD:]) / _FAST_PERIOD
        current_sma20 = sum(closes[-_SLOW_PERIOD:]) / _SLOW_PERIOD
        previous_close = closes[-2]
        current_close = closes[-1]
        timestamp_ms = candles[-1].timestamp_ms
        bullish_cross = previous_close <= previous_sma10 and current_close > current_sma10
        bearish_cross = previous_close >= previous_sma10 and current_close < current_sma10
        bullish_trend = current_sma10 > current_sma20
        bearish_trend = current_sma10 < current_sma20
        indicators = {
            "sma10": current_sma10,
            "sma20": current_sma20,
            "previous_sma10": previous_sma10,
            "previous_close": previous_close,
            "close": current_close,
            "stop_loss_pct": _STOP_LOSS_PCT,
            "max_hold_hours": 168.0,
        }
        conditions = [
            _condition(
                "trend.sma10_vs_sma20", "SMA10 versus SMA20",
                "triggered" if bullish_trend or bearish_trend else "not_triggered",
                "SMA10 is above SMA20 (bullish trend)." if bullish_trend else
                "SMA10 is below SMA20 (bearish trend)." if bearish_trend else
                "SMA10 equals SMA20; trend is undefined.",
                actual=current_sma10, threshold=current_sma20, operator=">" if bullish_trend else "<" if bearish_trend else "==",
                timestamp_ms=timestamp_ms,
            ),
            _condition(
                "entry.price_cross_up", "Price crosses above SMA10",
                "triggered" if bullish_cross else "not_triggered",
                "Previous close is at or below previous SMA10 and current close is above current SMA10."
                if bullish_cross else "The bullish price-cross rule is not satisfied.",
                actual=current_close, threshold=current_sma10, operator=">",
                timestamp_ms=timestamp_ms,
            ),
            _condition(
                "entry.price_cross_down", "Price crosses below SMA10",
                "triggered" if bearish_cross else "not_triggered",
                "Previous close is at or above previous SMA10 and current close is below current SMA10."
                if bearish_cross else "The bearish price-cross rule is not satisfied.",
                actual=current_close, threshold=current_sma10, operator="<",
                timestamp_ms=timestamp_ms,
            ),
        ]

        position = request.position
        if position is not None:
            if position.symbol != symbol:
                return _signal(
                    request, symbol, "blocked", "position_symbol_mismatch",
                    "Position symbol does not match the supplied candle symbol.", conditions,
                    indicators, pair=position.pair,
                    execution_block_reason="Position and candle symbols must match.",
                )
            if position.entry_timestamp_ms > _observed_ms(request):
                return _signal(
                    request, symbol, "blocked", "invalid_position_time",
                    "Position entry time is later than the observation time.", conditions,
                    indicators, pair=position.pair,
                    execution_block_reason="Position timestamp is in the future.",
                )
            stop_price = position.entry_price * (1 - _STOP_LOSS_PCT) if position.side == "long" else position.entry_price * (1 + _STOP_LOSS_PCT)
            stop_triggered = current_close <= stop_price if position.side == "long" else current_close >= stop_price
            held_ms = _observed_ms(request) - position.entry_timestamp_ms
            timeout_triggered = held_ms >= _MAX_HOLD_MS
            opposite_cross = bearish_cross if position.side == "long" else bullish_cross
            conditions.extend([
                _condition(
                    "exit.stop_loss", "Adverse 5% stop loss",
                    "triggered" if stop_triggered else "not_triggered",
                    "Current close reached the adverse 5% stop threshold." if stop_triggered else "Current close has not reached the adverse 5% stop threshold.",
                    actual=current_close, threshold=stop_price,
                    operator="<=" if position.side == "long" else ">=", timestamp_ms=timestamp_ms,
                ),
                _condition(
                    "exit.timeout", "Maximum holding time",
                    "triggered" if timeout_triggered else "not_triggered",
                    "Position has been held for at least 168 hours." if timeout_triggered else "Position has not reached the 168-hour maximum holding time.",
                    actual=held_ms / 3_600_000, threshold=168.0, operator=">=",
                    timestamp_ms=timestamp_ms,
                ),
                _condition(
                    "exit.opposite_cross", "Opposite price/SMA10 cross",
                    "triggered" if opposite_cross else "not_triggered",
                    "Price crossed SMA10 against the position." if opposite_cross else "No opposite price/SMA10 cross occurred.",
                    actual=current_close, threshold=current_sma10,
                    operator="<" if position.side == "long" else ">", timestamp_ms=timestamp_ms,
                ),
            ])
            if stop_triggered:
                return _signal(request, symbol, "exit", "stop_loss", "Adverse 5% stop loss triggered.", conditions, indicators, pair=position.pair)
            if timeout_triggered:
                return _signal(request, symbol, "exit", "max_hold_timeout", "168-hour maximum holding time reached.", conditions, indicators, pair=position.pair)
            if opposite_cross:
                return _signal(request, symbol, "exit", "opposite_ma10_cross", "Opposite price/SMA10 cross triggered exit.", conditions, indicators, pair=position.pair)
            return _signal(request, symbol, "hold", "position_held", "Position remains open; no exit condition triggered.", conditions, indicators, pair=position.pair)

        if bullish_trend and bullish_cross:
            return _signal(request, symbol, "long_entry", "bullish_price_cross", "Bullish trend with price crossing above SMA10.", conditions, indicators)
        if bearish_trend and bearish_cross:
            return _signal(request, symbol, "short_entry", "bearish_price_cross", "Bearish trend with price crossing below SMA10.", conditions, indicators)
        return _signal(request, symbol, "no_signal", "no_entry_signal", "No valid trend-aligned price cross occurred.", conditions, indicators)


def evaluate_fixed_dual_ma(request: FixedEngineInput) -> FixedStrategySignal:
    """Functional entry point for the fixed dual-moving-average engine."""
    return FixedDualMaEngine().evaluate(request)


__all__ = ["FixedDualMaEngine", "evaluate_fixed_dual_ma"]
