from datetime import datetime, timedelta, timezone

import pytest

from valuecell.server.api.schemas.fixed_strategy import FixedEngineInput, FixedPosition
from valuecell.server.services.fixed_dual_ma_engine import FixedDualMaEngine, evaluate_fixed_dual_ma

BASE_MS = 1_700_000_000_000
BASE_TIME = datetime.fromtimestamp(BASE_MS / 1000, tz=timezone.utc)


def _input(closes: list[float], *, position: FixedPosition | None = None, closed: bool = True, observed_at=None):
    candles = [
        {
            "symbol": "BTC-USDT",
            "timestamp_ms": BASE_MS + i * 4 * 60 * 60 * 1000,
            "open": close,
            "high": close + 1,
            "low": max(0.01, close - 1),
            "close": close,
            "volume": 1,
            "is_closed": closed,
        }
        for i, close in enumerate(closes)
    ]
    return FixedEngineInput.model_validate({
        "candles": candles,
        "position": position,
        "observed_at": observed_at or BASE_TIME + timedelta(hours=(len(closes) - 1) * 4),
    })


def _position(side: str, *, entry_price: float = 100, entry_timestamp_ms: int = BASE_MS):
    return FixedPosition.model_validate({
        "symbol": "BTC-USDT", "side": side, "quantity": 1,
        "entry_price": entry_price, "entry_timestamp_ms": entry_timestamp_ms,
    })


def test_bullish_trend_and_upward_price_cross_emits_long_entry():
    result = FixedDualMaEngine().evaluate(_input([100] * 21 + [101]))
    assert result.action == "long_entry"
    assert result.reason_code == "bullish_price_cross"
    assert result.indicators["sma10"] == pytest.approx(100.1)
    assert result.indicators["sma20"] == pytest.approx(100.05)
    assert all(condition.detail for condition in result.conditions)


def test_bearish_trend_and_downward_price_cross_emits_short_entry():
    result = evaluate_fixed_dual_ma(_input([100] * 21 + [99]))
    assert result.action == "short_entry"
    assert result.reason_code == "bearish_price_cross"
    assert result.indicators["sma10"] == pytest.approx(99.9)
    assert result.indicators["sma20"] == pytest.approx(99.95)


def test_long_position_stop_has_priority_over_timeout_and_cross():
    result = FixedDualMaEngine().evaluate(
        _input([100] * 21 + [95], position=_position("long"),
               observed_at=BASE_TIME + timedelta(hours=168))
    )
    assert result.action == "exit"
    assert result.reason_code == "stop_loss"


def test_position_timeout_exits_when_stop_and_signal_are_absent():
    result = FixedDualMaEngine().evaluate(
        _input([100] * 22, position=_position("long"),
               observed_at=BASE_TIME + timedelta(hours=168))
    )
    assert result.action == "exit"
    assert result.reason_code == "max_hold_timeout"
    timeout = next(c for c in result.conditions if c.code == "exit.timeout")
    assert timeout.actual == pytest.approx(168)
    assert timeout.threshold == 168
    assert timeout.operator == ">="


def test_opposite_cross_exits_existing_long_position():
    result = FixedDualMaEngine().evaluate(_input([100] * 21 + [99], position=_position("long")))
    assert result.action == "exit"
    assert result.reason_code == "opposite_ma10_cross"


def test_insufficient_candles_are_blocked_explicitly():
    result = FixedDualMaEngine().evaluate(_input([100] * 21))
    assert result.action == "blocked"
    assert result.reason_code == "insufficient_candles"
    condition = result.conditions[0]
    assert condition.actual == 21
    assert condition.threshold == 22
    assert condition.operator == ">="


def test_unfinished_candle_is_blocked_even_with_enough_history():
    result = FixedDualMaEngine().evaluate(_input([100] * 22, closed=False))
    assert result.action == "blocked"
    assert result.reason_code == "unfinished_candles"
    assert result.conditions[0].actual is False
