"""Pure paper signal engine for the fixed six-pair rotation strategy.

The engine deliberately stops at signal generation.  It consumes only final
candles and strategy-owned position state; market data, persistence, scheduling,
and order execution belong to the platform boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
from statistics import fmean
from typing import Sequence

from valuecell.server.api.schemas.fixed_strategy import (
    FixedCandle,
    FixedCondition,
    FixedEngineInput,
    FixedPosition,
    FixedStrategySignal,
)

RATIO_LOOKBACK = 240
Z_ENTRY = 2.0
Z_EXIT = 0.5
Z_STOP = 4.0
TIME_STOP = 180
_BAR_MS = 4 * 60 * 60 * 1_000


@dataclass(frozen=True, slots=True)
class PairDefinition:
    """The fixed A/B order is part of the strategy, not user configuration."""

    name: str
    a: str
    b: str


FIXED_PAIRS: tuple[PairDefinition, ...] = (
    PairDefinition("DOGE/PEPE", "DOGE-USDT", "PEPE-USDT"),
    PairDefinition("AAVE/LINK", "AAVE-USDT", "LINK-USDT"),
    PairDefinition("ADA/LTC", "ADA-USDT", "LTC-USDT"),
    PairDefinition("BNB/BTC", "BNB-USDT", "BTC-USDT"),
    PairDefinition("INJ/NEAR", "INJ-USDT", "NEAR-USDT"),
    PairDefinition("ETH/SOL", "ETH-USDT", "SOL-USDT"),
)

_PAIR_BY_NAME = {pair.name: pair for pair in FIXED_PAIRS}
_PAIR_BY_SYMBOL = {
    symbol: pair for pair in FIXED_PAIRS for symbol in (pair.a, pair.b)
}


def _utc_from_ms(timestamp_ms: int) -> datetime:
    return datetime.fromtimestamp(timestamp_ms / 1_000, tz=timezone.utc)


def _canonical_pair(value: str) -> str | None:
    """Accept the persisted slash form and symbol-qualified variants."""

    normalized = value.strip().upper().replace(" ", "").replace(":", "/")
    parts = normalized.split("/")
    if len(parts) != 2:
        return None
    symbols = [part if part.endswith("-USDT") else f"{part}-USDT" for part in parts]
    for pair in FIXED_PAIRS:
        if tuple(symbols) == (pair.a, pair.b):
            return pair.name
    return None


def _metric_conditions(
    *,
    pair: PairDefinition,
    timestamp_ms: int,
    ratio: float | None,
    mean: float | None,
    std: float | None,
    z_score: float | None,
    age_bars: int | None,
) -> list[FixedCondition]:
    def metric(
        code: str,
        label: str,
        actual: float | None,
        detail: str,
    ) -> FixedCondition:
        return FixedCondition(
            code=code,
            label=label,
            state="unavailable" if actual is None else "not_triggered",
            actual=actual,
            detail=detail,
            data_timestamp_ms=timestamp_ms,
        )

    conditions = [
        FixedCondition(
            code="pair_identity",
            label="固定币对",
            state="not_triggered",
            actual=pair.name,
            threshold=pair.name,
            operator="=",
            detail=f"固定 A/B 顺序为 {pair.a}/{pair.b}；不进行动态选对。",
            data_timestamp_ms=timestamp_ms,
        ),
        metric("ratio", "价格比 A/B", ratio, "当前同步收盘价比值 A/B。"),
        metric("rolling_mean", "240 根滚动均值", mean, "比值最近 240 根同步收盘的均值。"),
        metric("rolling_std", "240 根滚动标准差", std, "比值最近 240 根同步收盘的总体标准差。"),
        FixedCondition(
            code="z_entry_upper",
            label="Z 入场上界",
            state="unavailable" if z_score is None else ("triggered" if z_score > Z_ENTRY else "not_triggered"),
            actual=z_score,
            threshold=Z_ENTRY,
            operator=">",
            detail="Z 超过 2.0 时买入 B（A 相对高估）。",
            data_timestamp_ms=timestamp_ms,
        ),
        FixedCondition(
            code="z_entry_lower",
            label="Z 入场下界",
            state="unavailable" if z_score is None else ("triggered" if z_score < -Z_ENTRY else "not_triggered"),
            actual=z_score,
            threshold=-Z_ENTRY,
            operator="<",
            detail="Z 低于 -2.0 时买入 A（B 相对高估）。",
            data_timestamp_ms=timestamp_ms,
        ),
        FixedCondition(
            code="z_exit_band",
            label="Z 止盈回归带",
            state="unavailable" if z_score is None else ("triggered" if abs(z_score) <= Z_EXIT else "not_triggered"),
            actual=z_score,
            threshold=Z_EXIT,
            operator="abs≤",
            detail="持仓方向对应的 Z 回归至 ±0.5 时止盈。",
            data_timestamp_ms=timestamp_ms,
        ),
        FixedCondition(
            code="z_stop_band",
            label="Z 发散止损带",
            state="unavailable" if z_score is None else ("triggered" if abs(z_score) >= Z_STOP else "not_triggered"),
            actual=z_score,
            threshold=Z_STOP,
            operator="abs≥",
            detail="Z 达到 ±4.0 时关系发散止损。",
            data_timestamp_ms=timestamp_ms,
        ),
        FixedCondition(
            code="timeout_bars",
            label="持仓时间止损",
            state="unavailable" if age_bars is None else ("triggered" if age_bars > TIME_STOP else "not_triggered"),
            actual=age_bars,
            threshold=TIME_STOP,
            operator=">",
            detail="持仓超过 180 根 4h K 线时退出。",
            data_timestamp_ms=timestamp_ms,
        ),
    ]
    return conditions


def _signal(
    *,
    action: str,
    symbol: str,
    pair: PairDefinition | None,
    input_data: FixedEngineInput,
    reason_code: str,
    reason: str,
    conditions: list[FixedCondition],
    ratio: float | None = None,
    mean: float | None = None,
    std: float | None = None,
    z_score: float | None = None,
    age_bars: int | None = None,
    execution_block_reason: str | None = None,
) -> FixedStrategySignal:
    indicators: dict[str, float | str | bool | None] = {
        "ratio": ratio,
        "mean": mean,
        "std": std,
        "z_score": z_score,
        "z_entry": Z_ENTRY,
        "z_exit": Z_EXIT,
        "z_stop": Z_STOP,
        "timeout_bars": TIME_STOP,
        "history_bars": RATIO_LOOKBACK if z_score is not None else None,
        "position_age_bars": age_bars,
    }
    return FixedStrategySignal(
        kind="pair_rotation",
        symbol=symbol,
        action=action,  # type: ignore[arg-type]
        reason_code=reason_code,
        reason=reason,
        conditions=conditions,
        indicators=indicators,
        observed_at=input_data.observed_at,
        pair=pair.name if pair else None,
        execution_block_reason=execution_block_reason,
    )


def _coerce_input(
    candles_or_input: FixedEngineInput | Sequence[FixedCandle],
    position: FixedPosition | None,
    observed_at: datetime | None,
) -> FixedEngineInput:
    if isinstance(candles_or_input, FixedEngineInput):
        if position is not None or observed_at is not None:
            raise TypeError("position/observed_at must not be supplied with FixedEngineInput")
        return candles_or_input
    candles = list(candles_or_input)
    if not candles:
        raise ValueError("candles must not be empty")
    return FixedEngineInput(
        candles=candles,
        position=position,
        observed_at=observed_at or _utc_from_ms(max(candle.timestamp_ms for candle in candles)),
    )


def _select_pair(input_data: FixedEngineInput, requested_pair: str | None) -> PairDefinition | None:
    if requested_pair is not None:
        name = _canonical_pair(requested_pair)
        return _PAIR_BY_NAME.get(name) if name else None
    if input_data.position is not None and input_data.position.pair is not None:
        name = _canonical_pair(input_data.position.pair)
        return _PAIR_BY_NAME.get(name) if name else None
    if input_data.position is not None:
        return _PAIR_BY_SYMBOL.get(input_data.position.symbol.upper())
    symbols = {candle.symbol.upper() for candle in input_data.candles}
    complete = [pair for pair in FIXED_PAIRS if pair.a in symbols and pair.b in symbols]
    if complete:
        return complete[0]
    present = [pair for pair in FIXED_PAIRS if pair.a in symbols or pair.b in symbols]
    return present[0] if present else None


def evaluate_pair_rotation(
    candles_or_input: FixedEngineInput | Sequence[FixedCandle],
    position: FixedPosition | None = None,
    observed_at: datetime | None = None,
    *,
    pair: str | None = None,
) -> FixedStrategySignal:
    """Evaluate one deterministic pair-rotation decision from closed candles.

    When no pair is supplied, a position's pair (or its symbol) selects the
    pair.  For a flat account, a complete pair is selected in fixed order.
    """
    input_data = _coerce_input(candles_or_input, position, observed_at)
    selected = _select_pair(input_data, pair)
    requested_symbol = input_data.position.symbol if input_data.position is not None else input_data.candles[-1].symbol

    if selected is None:
        requested = pair or (input_data.position.pair if input_data.position else None)
        unsupported_condition = FixedCondition(
            code="pair_identity",
            label="固定币对",
            state="blocked",
            actual=requested,
            threshold="six fixed pairs",
            operator="in",
            detail="Requested pair is not one of the six fixed A/B pairs.",
        )
        return _signal(
            action="blocked",
            symbol=requested_symbol,
            pair=None,
            input_data=input_data,
            reason_code="unsupported_pair",
            reason="No fixed pair matches the requested pair or supplied position.",
            conditions=[unsupported_condition],
            execution_block_reason="unsupported_pair",
        )

    timestamp_ms = max(candle.timestamp_ms for candle in input_data.candles)
    base_conditions = _metric_conditions(
        pair=selected,
        timestamp_ms=timestamp_ms,
        ratio=None,
        mean=None,
        std=None,
        z_score=None,
        age_bars=None,
    )
    position_state = input_data.position
    if position_state is not None:
        if position_state.pair is not None and _canonical_pair(position_state.pair) != selected.name:
            return _signal(
                action="blocked",
                symbol=position_state.symbol,
                pair=selected,
                input_data=input_data,
                reason_code="position_pair_mismatch",
                reason="Position pair does not match the fixed pair selected for this evaluation.",
                conditions=base_conditions,
                execution_block_reason="position_pair_mismatch",
            )
        if position_state.symbol.upper() not in (selected.a, selected.b):
            reason_code = (
                "position_pair_mismatch"
                if position_state.pair is not None
                else "position_symbol_mismatch"
            )
            return _signal(
                action="blocked",
                symbol=position_state.symbol,
                pair=selected,
                input_data=input_data,
                reason_code=reason_code,
                reason="Position symbol is not one leg of the selected fixed pair.",
                conditions=base_conditions,
                execution_block_reason=reason_code,
            )
        if position_state.side != "long":
            return _signal(
                action="blocked",
                symbol=position_state.symbol,
                pair=selected,
                input_data=input_data,
                reason_code="unsupported_position_side",
                reason="Pair rotation is spot single-leg long only; short positions are unsupported.",
                conditions=base_conditions,
                execution_block_reason="unsupported_position_side",
            )

    pair_candles = [
        candle
        for candle in input_data.candles
        if candle.symbol.upper() in (selected.a, selected.b)
    ]
    unfinished = [candle for candle in pair_candles if not candle.is_closed]
    if unfinished:
        return _signal(
            action="blocked",
            symbol=requested_symbol,
            pair=selected,
            input_data=input_data,
            reason_code="unfinished_candles",
            reason="Every candle used for the pair ratio must be final and closed.",
            conditions=[
                *base_conditions,
                FixedCondition(
                    code="candles_closed",
                    label="配对 K 线已收盘",
                    state="blocked",
                    actual=False,
                    threshold=True,
                    operator="=",
                    detail="A pair leg includes an unfinished 4h candle; no ratio is inferred.",
                    data_timestamp_ms=max(candle.timestamp_ms for candle in unfinished),
                ),
            ],
            execution_block_reason="unfinished_candles",
        )

    candles_by_symbol: dict[str, dict[int, FixedCandle]] = {}
    invalid_duplicate = False
    for candle in input_data.candles:
        symbol = candle.symbol.upper()
        if not candle.is_closed:
            continue
        per_symbol = candles_by_symbol.setdefault(symbol, {})
        if candle.timestamp_ms in per_symbol:
            invalid_duplicate = True
        per_symbol[candle.timestamp_ms] = candle
    if invalid_duplicate:
        return _signal(
            action="blocked",
            symbol=requested_symbol,
            pair=selected,
            input_data=input_data,
            reason_code="duplicate_candle",
            reason="Duplicate timestamp facts for a pair leg cannot produce a deterministic ratio.",
            conditions=base_conditions,
            execution_block_reason="duplicate_candle",
        )
    a_candles = candles_by_symbol.get(selected.a, {})
    b_candles = candles_by_symbol.get(selected.b, {})
    if not a_candles or not b_candles:
        return _signal(
            action="blocked",
            symbol=requested_symbol,
            pair=selected,
            input_data=input_data,
            reason_code="missing_pair_candles",
            reason=f"Both closed candle legs are required for {selected.name}.",
            conditions=base_conditions,
            execution_block_reason="missing_pair_candles",
        )
    timestamps = sorted(set(a_candles).intersection(b_candles))
    if len(timestamps) < RATIO_LOOKBACK:
        return _signal(
            action="blocked",
            symbol=requested_symbol,
            pair=selected,
            input_data=input_data,
            reason_code="insufficient_history",
            reason=f"Only {len(timestamps)} synchronized closed bars are available; 240 are required.",
            conditions=base_conditions,
            execution_block_reason="insufficient_history",
        )

    timestamps = timestamps[-RATIO_LOOKBACK:]
    ratios = [a_candles[ts].close / b_candles[ts].close for ts in timestamps]
    ratio = ratios[-1]
    mean = fmean(ratios)
    variance = fmean([(value - mean) ** 2 for value in ratios])
    std = sqrt(variance)
    if std == 0.0:
        conditions = _metric_conditions(
            pair=selected,
            timestamp_ms=timestamps[-1],
            ratio=ratio,
            mean=mean,
            std=std,
            z_score=None,
            age_bars=None,
        )
        return _signal(
            action="blocked",
            symbol=position_state.symbol if position_state else "PAIR_ROTATION",
            pair=selected,
            input_data=input_data,
            reason_code="zero_std",
            reason="The 240-bar ratio standard deviation is zero; Z-score is undefined.",
            conditions=conditions,
            ratio=ratio,
            mean=mean,
            std=std,
            execution_block_reason="zero_std",
        )
    z_score = (ratio - mean) / std

    age_bars: int | None = None
    if position_state is not None:
        if position_state.entry_timestamp_ms > timestamps[-1]:
            conditions = _metric_conditions(
                pair=selected,
                timestamp_ms=timestamps[-1],
                ratio=ratio,
                mean=mean,
                std=std,
                z_score=z_score,
                age_bars=None,
            )
            return _signal(
                action="blocked",
                symbol=position_state.symbol,
                pair=selected,
                input_data=input_data,
                reason_code="invalid_position_timestamp",
                reason="Position entry timestamp is later than the latest closed pair bar.",
                conditions=conditions,
                ratio=ratio,
                mean=mean,
                std=std,
                z_score=z_score,
                execution_block_reason="invalid_position_timestamp",
            )
        age_bars = max(0, (timestamps[-1] - position_state.entry_timestamp_ms) // _BAR_MS)

    conditions = _metric_conditions(
        pair=selected,
        timestamp_ms=timestamps[-1],
        ratio=ratio,
        mean=mean,
        std=std,
        z_score=z_score,
        age_bars=age_bars,
    )

    if position_state is None:
        if z_score > Z_ENTRY:
            return _signal(
                action="long_entry",
                symbol=selected.b,
                pair=selected,
                input_data=input_data,
                reason_code="entry_buy_b",
                reason=f"Z-score {z_score:.6g} is above {Z_ENTRY}; buy the under-valued B leg.",
                conditions=conditions,
                ratio=ratio,
                mean=mean,
                std=std,
                z_score=z_score,
            )
        if z_score < -Z_ENTRY:
            return _signal(
                action="long_entry",
                symbol=selected.a,
                pair=selected,
                input_data=input_data,
                reason_code="entry_buy_a",
                reason=f"Z-score {z_score:.6g} is below {-Z_ENTRY}; buy the under-valued A leg.",
                conditions=conditions,
                ratio=ratio,
                mean=mean,
                std=std,
                z_score=z_score,
            )
        return _signal(
            action="no_signal",
            symbol="PAIR_ROTATION",
            pair=selected,
            input_data=input_data,
            reason_code="no_entry",
            reason=f"Z-score {z_score:.6g} is within the ±{Z_ENTRY} entry thresholds.",
            conditions=conditions,
            ratio=ratio,
            mean=mean,
            std=std,
            z_score=z_score,
        )

    held_symbol = position_state.symbol.upper()
    if abs(z_score) >= Z_STOP and ((held_symbol == selected.b and z_score >= Z_STOP) or (held_symbol == selected.a and z_score <= -Z_STOP)):
        return _signal(
            action="exit",
            symbol=position_state.symbol,
            pair=selected,
            input_data=input_data,
            reason_code="diverge_stop",
            reason=f"Z-score {z_score:.6g} reached the {Z_STOP} divergence stop for the held leg.",
            conditions=conditions,
            ratio=ratio,
            mean=mean,
            std=std,
            z_score=z_score,
            age_bars=age_bars,
        )
    if age_bars is not None and age_bars > TIME_STOP:
        return _signal(
            action="exit",
            symbol=position_state.symbol,
            pair=selected,
            input_data=input_data,
            reason_code="time_stop",
            reason=f"Position age {age_bars} bars exceeds the {TIME_STOP}-bar timeout.",
            conditions=conditions,
            ratio=ratio,
            mean=mean,
            std=std,
            z_score=z_score,
            age_bars=age_bars,
        )

    if held_symbol == selected.b and z_score <= -Z_ENTRY:
        return _signal(
            action="exit",
            symbol=position_state.symbol,
            pair=selected,
            input_data=input_data,
            reason_code="rotate_to_a",
            reason=f"Z-score {z_score:.6g} crossed {-Z_ENTRY}; exit B before rotating to A.",
            conditions=conditions,
            ratio=ratio,
            mean=mean,
            std=std,
            z_score=z_score,
            age_bars=age_bars,
        )
    if held_symbol == selected.a and z_score >= Z_ENTRY:
        return _signal(
            action="exit",
            symbol=position_state.symbol,
            pair=selected,
            input_data=input_data,
            reason_code="rotate_to_b",
            reason=f"Z-score {z_score:.6g} crossed {Z_ENTRY}; exit A before rotating to B.",
            conditions=conditions,
            ratio=ratio,
            mean=mean,
            std=std,
            z_score=z_score,
            age_bars=age_bars,
        )
    if (held_symbol == selected.b and z_score <= Z_EXIT) or (held_symbol == selected.a and z_score >= -Z_EXIT):
        return _signal(
            action="exit",
            symbol=position_state.symbol,
            pair=selected,
            input_data=input_data,
            reason_code="take_profit",
            reason=f"Z-score {z_score:.6g} returned to the held-leg ±{Z_EXIT} profit threshold.",
            conditions=conditions,
            ratio=ratio,
            mean=mean,
            std=std,
            z_score=z_score,
            age_bars=age_bars,
        )
    return _signal(
        action="hold",
        symbol=position_state.symbol,
        pair=selected,
        input_data=input_data,
        reason_code="hold_position",
        reason=f"Z-score {z_score:.6g} has not reached an exit, rotation, or stop threshold.",
        conditions=conditions,
        ratio=ratio,
        mean=mean,
        std=std,
        z_score=z_score,
        age_bars=age_bars,
    )


# Descriptive aliases keep the service easy to discover without introducing a
# second implementation or a configurable strategy path.
pair_rotation_signal = evaluate_pair_rotation
fixed_pair_rotation_signal = evaluate_pair_rotation


class PairRotationEngine:
    """Small callable facade for callers that prefer an engine object."""

    def evaluate(
        self,
        candles_or_input: FixedEngineInput | Sequence[FixedCandle],
        position: FixedPosition | None = None,
        observed_at: datetime | None = None,
        *,
        pair: str | None = None,
    ) -> FixedStrategySignal:
        return evaluate_pair_rotation(candles_or_input, position, observed_at, pair=pair)

    __call__ = evaluate


FixedPairRotationEngine = PairRotationEngine


__all__ = [
    "FIXED_PAIRS",
    "FixedPairRotationEngine",
    "PairDefinition",
    "PairRotationEngine",
    "RATIO_LOOKBACK",
    "TIME_STOP",
    "Z_ENTRY",
    "Z_EXIT",
    "Z_STOP",
    "evaluate_pair_rotation",
    "fixed_pair_rotation_signal",
    "pair_rotation_signal",
]
