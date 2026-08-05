"""Canonical, versioned technical-indicator calculations for Trend Resonance V2.1."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum, isfinite
from typing import Any

ROC_V2_1_PERIOD = 5
BRAR_V2_1_PERIOD = 26
MACD_V2_1_FAST_PERIOD = 12
MACD_V2_1_SLOW_PERIOD = 26
MACD_V2_1_SIGNAL_PERIOD = 9
MACD_V2_1_REQUIRED_CANDLES = (
    MACD_V2_1_SLOW_PERIOD + MACD_V2_1_SIGNAL_PERIOD - 1
)


@dataclass(frozen=True, slots=True)
class BrarV21:
    """The canonical BR and AR values from the most recent 26 candles."""

    br: float
    ar: float


@dataclass(frozen=True, slots=True)
class MacdV21:
    """The canonical DIF, SMA signal (DEA), and doubled histogram values."""

    dif: float
    dea: float
    histogram: float


def roc_v2_1(closes: Sequence[float]) -> float | None:
    """Return five-bar ROC, or ``None`` when the fixed V2.1 input is unavailable.

    The V2.1 definition is exactly ``((close_t - close_t_minus_5) /
    close_t_minus_5) * 100``. A zero historical close is unavailable rather
    than a division-by-zero fallback.
    """

    if len(closes) <= ROC_V2_1_PERIOD:
        return None
    current = _finite_float(closes[-1])
    historical = _finite_float(closes[-1 - ROC_V2_1_PERIOD])
    if current is None or historical is None or historical == 0:
        return None
    value = (current - historical) / historical * 100.0
    return value if isfinite(value) else None


def brar_v2_1(candles: Sequence[object]) -> BrarV21 | None:
    """Return V2.1 BR/AR from the most recent 26 candles, or ``None``.

    V2.1 deliberately uses the supplied formulas rather than the repository's
    legacy BRAR implementation:

    * ``BR = sum(high - close) / sum(close - low) * 100``
    * ``AR = sum(high) / sum(low) * 100``

    A zero denominator in either component makes the complete indicator
    unavailable. It is never represented as a fabricated value of 100.
    """

    if len(candles) < BRAR_V2_1_PERIOD:
        return None

    br_numerator = 0.0
    br_denominator = 0.0
    ar_numerator = 0.0
    ar_denominator = 0.0
    for candle in candles[-BRAR_V2_1_PERIOD:]:
        high = _candle_value(candle, "high")
        close = _candle_value(candle, "close")
        low = _candle_value(candle, "low")
        if high is None or close is None or low is None:
            return None
        br_numerator += high - close
        br_denominator += close - low
        ar_numerator += high
        ar_denominator += low

    if not all(
        isfinite(value)
        for value in (br_numerator, br_denominator, ar_numerator, ar_denominator)
    ):
        return None
    if br_denominator == 0 or ar_denominator == 0:
        return None

    br = br_numerator / br_denominator * 100.0
    ar = ar_numerator / ar_denominator * 100.0
    if not isfinite(br) or not isfinite(ar):
        return None
    return BrarV21(br=br, ar=ar)


def macd_v2_1(closes: Sequence[float]) -> MacdV21 | None:
    """Return canonical V2.1 MACD values, or ``None`` for incomplete input.

    DIF is EMA(12) minus EMA(26). DEA is the simple moving average of the most
    recent nine DIF values, not an EMA signal line. The histogram is exactly
    ``(DIF - DEA) * 2``. EMA values are seeded with the first supplied close,
    which makes the result deterministic for a persisted candle sequence.
    """

    if len(closes) < MACD_V2_1_REQUIRED_CANDLES:
        return None
    values = [_finite_float(value) for value in closes]
    if any(value is None for value in values):
        return None
    numeric_values = [float(value) for value in values if value is not None]

    fast_multiplier = 2.0 / (MACD_V2_1_FAST_PERIOD + 1.0)
    slow_multiplier = 2.0 / (MACD_V2_1_SLOW_PERIOD + 1.0)
    fast_ema = numeric_values[0]
    slow_ema = numeric_values[0]
    dif_values: deque[float] = deque(maxlen=MACD_V2_1_SIGNAL_PERIOD)
    for close in numeric_values:
        fast_ema += (close - fast_ema) * fast_multiplier
        slow_ema += (close - slow_ema) * slow_multiplier
        dif_values.append(fast_ema - slow_ema)

    if len(dif_values) != MACD_V2_1_SIGNAL_PERIOD:
        return None
    dif = dif_values[-1]
    dea = fsum(dif_values) / MACD_V2_1_SIGNAL_PERIOD
    histogram = (dif - dea) * 2.0
    if not all(isfinite(value) for value in (dif, dea, histogram)):
        return None
    return MacdV21(dif=dif, dea=dea, histogram=histogram)


def _candle_value(candle: object, field: str) -> float | None:
    if isinstance(candle, Mapping):
        return _finite_float(candle.get(field))
    return _finite_float(getattr(candle, field, None))


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None
