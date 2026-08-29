"""Pure 4-hour leader-breakout signal engine.

This module intentionally has no market, execution, scheduler, or persistence
boundary.  It evaluates only final candles supplied by its caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean

from valuecell.server.api.schemas.fixed_strategy import (
    FixedCandle,
    FixedCondition,
    FixedEngineInput,
    FixedStrategySignal,
)

_KIND = "leader_breakout"
_BAR_MS = 4 * 60 * 60 * 1_000
_MIN_CANDLES = 60
_LIQUIDITY_MIN = 200_000.0
_RS_LOOKBACK = 6
_RS_MIN = 0.04
_BREAKOUT_LOOKBACK = 20
_BREAKOUT_VOLUME_MULTIPLIER = 1.5
_PATH_B_LOOKBACK = 15
_PATH_B_PULLBACK = 0.02


def evaluate_leader_candidate(
    candidate: FixedEngineInput,
    btc_candles: Sequence[FixedCandle],
) -> FixedStrategySignal:
    """Evaluate one candidate against final 4-hour BTC and coin candles.

    Quote liquidity is the sum of the latest six per-candle quote volumes, the
    strategy's 24-hour window.  A missing quote-volume fact is unavailable;
    the engine never substitutes base volume or a current market value.
    """

    candles = candidate.candles
    symbol = candles[-1].symbol
    indicators: dict[str, float | str | bool | None] = {
        "timeframe_minutes": 240.0,
        "minimum_candles": float(_MIN_CANDLES),
    }

    candidate_problem = _candle_problem(candles, expected_symbol=symbol)
    if candidate_problem is not None:
        return _signal(
            candidate,
            symbol,
            "blocked",
            "candidate_candles_invalid",
            candidate_problem,
            [_condition("candidate_candles", "Candidate 4h candles", "blocked", candidate_problem)],
            indicators,
        )
    if len(candles) < _MIN_CANDLES:
        indicators["candidate_candle_count"] = float(len(candles))
        return _signal(
            candidate,
            symbol,
            "blocked",
            "candidate_data_insufficient",
            f"{symbol} has {len(candles)} final 4h candles; {_MIN_CANDLES} are required.",
            [
                _condition(
                    "candidate_data_sufficient",
                    "Candidate data sufficiency",
                    "blocked",
                    "Need at least 60 final 4h candles.",
                    actual=float(len(candles)),
                    threshold=float(_MIN_CANDLES),
                    operator=">=",
                )
            ],
            indicators,
        )

    if candidate.position is not None:
        if candidate.position.symbol != symbol:
            return _signal(
                candidate,
                symbol,
                "blocked",
                "position_symbol_mismatch",
                "The supplied position belongs to a different symbol.",
                [
                    _condition(
                        "position_symbol_matches",
                        "Position symbol matches candidate",
                        "blocked",
                        "A position may only be evaluated with its own symbol.",
                        actual=candidate.position.symbol,
                        threshold=symbol,
                        operator="==",
                    )
                ],
                indicators,
            )
        indicators["position_present"] = True
        return _signal(
            candidate,
            symbol,
            "hold",
            "position_exit_facts_unprovided",
            "An existing position is unchanged because no explicit exit fact was supplied.",
            [
                _condition(
                    "existing_position",
                    "Existing position",
                    "triggered",
                    "Entry evaluation is skipped; this pure engine has no inferred exit rule.",
                    actual=True,
                    threshold=False,
                    operator="!=",
                )
            ],
            indicators,
        )

    quote_volumes = [candle.quote_volume for candle in candles[-_RS_LOOKBACK:]]
    if any(value is None for value in quote_volumes):
        return _signal(
            candidate,
            symbol,
            "blocked",
            "quote_volume_unavailable",
            "One or more of the latest six 4h quote-volume facts is unavailable.",
            [
                _condition(
                    "liquidity_quote_volume_available",
                    "24h quote-volume availability",
                    "unavailable",
                    "All six final 4h quote-volume values are required to calculate 24h liquidity.",
                )
            ],
            indicators,
        )
    liquidity = float(sum(value for value in quote_volumes if value is not None))
    indicators["quote_volume_24h"] = liquidity
    liquidity_passed = liquidity >= _LIQUIDITY_MIN
    liquidity_condition = _condition(
        "liquidity_minimum",
        "24h quote-volume liquidity",
        "triggered" if liquidity_passed else "not_triggered",
        "24h quote volume must meet the fixed liquidity minimum.",
        actual=liquidity,
        threshold=_LIQUIDITY_MIN,
        operator=">=",
    )
    if not liquidity_passed:
        return _signal(
            candidate,
            symbol,
            "no_signal",
            "liquidity_below_minimum",
            f"24h quote volume {liquidity:.6g} is below {_LIQUIDITY_MIN:.6g}.",
            [liquidity_condition],
            indicators,
        )

    close_now = candles[-1].close
    close_then = candles[-1 - _RS_LOOKBACK].close
    relative_strength = (close_now - close_then) / close_then
    indicators["relative_strength_24h"] = relative_strength
    rs_passed = relative_strength >= _RS_MIN
    rs_condition = _condition(
        "relative_strength_minimum",
        "24h relative strength",
        "triggered" if rs_passed else "not_triggered",
        "Latest close relative to the close six final 4h bars earlier.",
        actual=relative_strength,
        threshold=_RS_MIN,
        operator=">=",
    )
    if not rs_passed:
        return _signal(
            candidate,
            symbol,
            "no_signal",
            "relative_strength_below_minimum",
            f"24h relative strength {relative_strength:.4%} is below {_RS_MIN:.2%}.",
            [liquidity_condition, rs_condition],
            indicators,
        )

    btc_problem = _candle_problem(btc_candles)
    if btc_problem is not None or len(btc_candles) < _MIN_CANDLES:
        detail = btc_problem or (
            f"BTC has {len(btc_candles)} final 4h candles; {_MIN_CANDLES} are required."
        )
        indicators["btc_candle_count"] = float(len(btc_candles))
        return _signal(
            candidate,
            symbol,
            "blocked",
            "btc_data_unavailable",
            detail,
            [
                liquidity_condition,
                rs_condition,
                _condition(
                    "btc_data_sufficient",
                    "BTC 4h data sufficiency",
                    "blocked",
                    detail,
                    actual=float(len(btc_candles)),
                    threshold=float(_MIN_CANDLES),
                    operator=">=",
                ),
            ],
            indicators,
        )

    btc_ema21 = _ema([candle.close for candle in btc_candles], 21)
    btc_ema55 = _ema([candle.close for candle in btc_candles], 55)
    btc_trend_passed = btc_ema21[-1] > btc_ema55[-1]
    btc_slope_passed = btc_ema55[-1] > btc_ema55[-13]
    indicators.update(
        {
            "btc_ema21": btc_ema21[-1],
            "btc_ema55": btc_ema55[-1],
            "btc_ema55_12_bars_prior": btc_ema55[-13],
        }
    )
    btc_trend_condition = _condition(
        "btc_ema_bullish",
        "BTC EMA21 above EMA55",
        "triggered" if btc_trend_passed else "not_triggered",
        "BTC's current 4h EMA21 must exceed its EMA55.",
        actual=btc_ema21[-1],
        threshold=btc_ema55[-1],
        operator=">",
    )
    btc_slope_condition = _condition(
        "btc_ema55_rising",
        "BTC EMA55 above 12-bars-prior EMA55",
        "triggered" if btc_slope_passed else "not_triggered",
        "BTC's current EMA55 must exceed its EMA55 twelve final 4h bars earlier.",
        actual=btc_ema55[-1],
        threshold=btc_ema55[-13],
        operator=">",
    )
    if not btc_trend_passed or not btc_slope_passed:
        reason_code = "btc_ema_trend_blocked" if not btc_trend_passed else "btc_ema55_slope_blocked"
        return _signal(
            candidate,
            symbol,
            "no_signal",
            reason_code,
            "BTC gate did not pass for a new long entry.",
            [liquidity_condition, rs_condition, btc_trend_condition, btc_slope_condition],
            indicators,
        )

    closes = [candle.close for candle in candles]
    ema21 = _ema(closes, 21)
    ema55 = _ema(closes, 55)
    trend_passed = ema21[-1] > ema55[-1]
    volume_sma20 = fmean(candle.volume for candle in candles[-21:-1])
    volume_ratio = candles[-1].volume / volume_sma20 if volume_sma20 > 0 else None
    indicators.update(
        {
            "coin_ema21": ema21[-1],
            "coin_ema55": ema55[-1],
            "volume_sma20": volume_sma20,
            "volume_ratio_to_sma20": volume_ratio,
        }
    )

    path_a_breakout = close_now > max(candle.high for candle in candles[-21:-1])
    path_a_volume = volume_ratio is not None and volume_ratio > _BREAKOUT_VOLUME_MULTIPLIER
    path_a_passed = path_a_breakout and path_a_volume and trend_passed
    path_b_reference = _latest_path_a_reference(candles)
    path_b_floor = (
        path_b_reference.close * (1.0 - _PATH_B_PULLBACK)
        if path_b_reference is not None
        else None
    )
    path_b_passed = path_b_floor is not None and close_now >= path_b_floor
    indicators.update(
        {
            "path_a_prior_high_20": max(candle.high for candle in candles[-21:-1]),
            "path_b_breakout_close": path_b_reference.close if path_b_reference else None,
            "path_b_pullback_floor": path_b_floor,
        }
    )

    trend_condition = _condition(
        "coin_ema_bullish",
        "Coin EMA21 above EMA55",
        "triggered" if trend_passed else "not_triggered",
        "Current coin EMA21 and EMA55 from supplied final 4h closes.",
        actual=ema21[-1],
        threshold=ema55[-1],
        operator=">",
    )
    path_a_condition = _condition(
        "path_a_breakout",
        "Path A breakout confirmation",
        "triggered" if path_a_passed else "not_triggered",
        "Requires close above the prior 20 highs, volume above 1.5x SMA20, and bullish coin EMA alignment.",
        actual=close_now,
        threshold=indicators["path_a_prior_high_20"],
        operator=">",
    )
    path_b_condition = _condition(
        "path_b_pullback",
        "Path B pullback confirmation",
        "triggered" if path_b_passed else "not_triggered",
        "Requires the latest qualifying historical Path A in the prior 15 bars and close at or above its 2% pullback floor.",
        actual=close_now,
        threshold=path_b_floor,
        operator=">=" if path_b_floor is not None else None,
    )

    if not path_a_passed and not path_b_passed:
        return _signal(
            candidate,
            symbol,
            "no_signal",
            "entry_path_not_confirmed",
            "Neither fixed entry path is confirmed by the supplied final 4h candles.",
            [
                liquidity_condition,
                rs_condition,
                btc_trend_condition,
                btc_slope_condition,
                trend_condition,
                path_a_condition,
                path_b_condition,
            ],
            indicators,
        )

    entry_path = "path_a" if path_a_passed else "path_b"
    ecs_score = _ecs_score(
        liquidity=liquidity,
        relative_strength=relative_strength,
        entry_path=entry_path,
        volume_ratio=volume_ratio,
        trend_passed=trend_passed,
    )
    indicators.update({"entry_path": entry_path, "ecs_score": ecs_score})
    return _signal(
        candidate,
        symbol,
        "long_entry",
        f"{entry_path}_confirmed",
        f"{entry_path.replace('_', ' ').title()} is confirmed after liquidity, relative-strength, and BTC gates passed.",
        [
            liquidity_condition,
            rs_condition,
            btc_trend_condition,
            btc_slope_condition,
            trend_condition,
            path_a_condition,
            path_b_condition,
            _condition(
                "ecs_score",
                "Deterministic ECS-like score",
                "triggered",
                "Score is used only to rank confirmed candidates deterministically.",
                actual=ecs_score,
                threshold=0.0,
                operator=">=",
            ),
        ],
        indicators,
    )


def evaluate_leader_candidates(
    candidates: Sequence[FixedEngineInput],
    btc_candles: Sequence[FixedCandle],
) -> list[FixedStrategySignal]:
    """Evaluate every requested candidate and order by score then symbol.

    Signals without a confirmed entry have no ECS score and sort after entry
    candidates.  Symbol is the deterministic tie breaker.
    """

    signals = [evaluate_leader_candidate(candidate, btc_candles) for candidate in candidates]
    return sorted(
        signals,
        key=lambda signal: (
            signal.action != "long_entry",
            -float(signal.indicators.get("ecs_score", 0.0) or 0.0),
            signal.symbol,
        ),
    )


class FixedLeaderEngine:
    """Small object façade for callers that standardize engines as classes."""

    def evaluate(
        self,
        candidate: FixedEngineInput,
        btc_candles: Sequence[FixedCandle],
    ) -> FixedStrategySignal:
        return evaluate_leader_candidate(candidate, btc_candles)

    def evaluate_candidates(
        self,
        candidates: Sequence[FixedEngineInput],
        btc_candles: Sequence[FixedCandle],
    ) -> list[FixedStrategySignal]:
        return evaluate_leader_candidates(candidates, btc_candles)


def _latest_path_a_reference(candles: Sequence[FixedCandle]) -> FixedCandle | None:
    """Return the newest prior bar qualifying for Path B's frozen Path A fact."""

    current_index = len(candles) - 1
    first_index = max(_BREAKOUT_LOOKBACK, current_index - _PATH_B_LOOKBACK)
    for index in range(current_index - 1, first_index - 1, -1):
        prior = candles[index - _BREAKOUT_LOOKBACK:index]
        volume_sma = fmean(candle.volume for candle in prior)
        if (
            candles[index].close > max(candle.high for candle in prior)
            and volume_sma > 0
            and candles[index].volume > volume_sma * _BREAKOUT_VOLUME_MULTIPLIER
        ):
            return candles[index]
    return None


def _ecs_score(
    *,
    liquidity: float,
    relative_strength: float,
    entry_path: str,
    volume_ratio: float | None,
    trend_passed: bool,
) -> float:
    """Calculate the fixed ECS components used to rank confirmed entries."""

    liquidity_score = 1.0 if liquidity >= _LIQUIDITY_MIN else 0.0
    relative_strength_score = min(relative_strength * 100.0 * 5.0, 20.0)
    breakout_score = 5.0 if entry_path == "path_a" else 2.0
    volume_score = (
        10.0
        if volume_ratio is not None and volume_ratio >= _BREAKOUT_VOLUME_MULTIPLIER
        else 5.0
        if volume_ratio is not None and volume_ratio >= 1.0
        else 0.0
    )
    trend_score = 10.0 if trend_passed else 0.0
    return liquidity_score + relative_strength_score + breakout_score + volume_score + trend_score


def _ema(closes: Sequence[float], period: int) -> list[float]:
    multiplier = 2.0 / (period + 1.0)
    values = [float(closes[0])]
    for close in closes[1:]:
        values.append(values[-1] + (float(close) - values[-1]) * multiplier)
    return values


def _candle_problem(
    candles: Sequence[FixedCandle], expected_symbol: str | None = None) -> str | None:
    if not candles:
        return "No final 4h candles were supplied."
    symbol = expected_symbol or candles[0].symbol
    previous_timestamp: int | None = None
    for candle in candles:
        if candle.symbol != symbol:
            return "Candles must all belong to the same symbol."
        if not candle.is_closed:
            return "Every candle must be final before it can be evaluated."
        if candle.timestamp_ms % _BAR_MS != 0:
            return "Candle timestamps must align to UTC 4-hour boundaries."
        if previous_timestamp is not None and candle.timestamp_ms - previous_timestamp != _BAR_MS:
            return "Candles must be contiguous 4-hour bars in ascending timestamp order."
        if candle.low > min(candle.open, candle.close) or candle.high < max(candle.open, candle.close):
            return "Candle OHLC facts are internally inconsistent."
        previous_timestamp = candle.timestamp_ms
    return None


def _condition(
    code: str,
    label: str,
    state: str,
    detail: str,
    *,
    actual: float | str | bool | None = None,
    threshold: float | str | bool | None = None,
    operator: str | None = None,
) -> FixedCondition:
    return FixedCondition(
        code=code,
        label=label,
        state=state,  # type: ignore[arg-type]
        actual=actual,
        threshold=threshold,
        operator=operator,
        detail=detail,
    )


def _signal(
    source: FixedEngineInput,
    symbol: str,
    action: str,
    reason_code: str,
    reason: str,
    conditions: list[FixedCondition],
    indicators: dict[str, float | str | bool | None],
) -> FixedStrategySignal:
    return FixedStrategySignal(
        kind=_KIND,
        symbol=symbol,
        action=action,  # type: ignore[arg-type]
        reason_code=reason_code,
        reason=reason,
        conditions=conditions,
        indicators=indicators,
        observed_at=source.observed_at,
    )
