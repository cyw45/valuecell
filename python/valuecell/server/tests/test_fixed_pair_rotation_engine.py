from __future__ import annotations

from datetime import datetime, timezone

from valuecell.server.api.schemas.fixed_strategy import FixedCandle, FixedEngineInput, FixedPosition
from valuecell.server.services.fixed_pair_rotation_engine import evaluate_pair_rotation

UTC_NOW = datetime(2026, 8, 28, tzinfo=timezone.utc)
BAR_MS = 4 * 60 * 60 * 1_000


def make_candles(
    ratios: list[float], *, a: str = "DOGE-USDT", b: str = "PEPE-USDT"
) -> list[FixedCandle]:
    candles: list[FixedCandle] = []
    for index, ratio in enumerate(ratios):
        timestamp_ms = 1_700_000_000_000 + index * BAR_MS
        candles.extend(
            [
                FixedCandle(
                    symbol=a,
                    timestamp_ms=timestamp_ms,
                    open=ratio,
                    high=ratio,
                    low=ratio,
                    close=ratio,
                    volume=1,
                ),
                FixedCandle(
                    symbol=b,
                    timestamp_ms=timestamp_ms,
                    open=1,
                    high=1,
                    low=1,
                    close=1,
                    volume=1,
                ),
            ]
        )
    return candles


def engine(ratios: list[float], position: FixedPosition | None = None):
    return evaluate_pair_rotation(
        FixedEngineInput(candles=make_candles(ratios), position=position, observed_at=UTC_NOW)
    )


def varied_history(current: float) -> list[float]:
    return ([0.9, 1.1] * 120)[:-1] + [current]


def position(
    symbol: str, entry_index: int = 239, pair: str | None = "DOGE/PEPE"
) -> FixedPosition:
    return FixedPosition(
        symbol=symbol,
        side="long",
        quantity=1,
        entry_price=1,
        entry_timestamp_ms=1_700_000_000_000 + entry_index * BAR_MS,
        pair=pair,
    )


def test_entry_b_when_ratio_z_is_above_two() -> None:
    signal = engine(varied_history(1.5))
    assert signal.action == "long_entry"
    assert signal.symbol == "PEPE-USDT"
    assert signal.reason_code == "entry_buy_b"
    assert signal.indicators["z_score"] > 2
    assert {condition.code for condition in signal.conditions} >= {
        "ratio",
        "rolling_mean",
        "rolling_std",
        "z_entry_upper",
    }


def test_entry_a_when_ratio_z_is_below_negative_two() -> None:
    signal = engine(varied_history(0.5))
    assert signal.action == "long_entry"
    assert signal.symbol == "DOGE-USDT"
    assert signal.reason_code == "entry_buy_a"
    assert signal.indicators["z_score"] < -2


def test_position_profit_exit_uses_the_held_leg() -> None:
    signal_b = engine(varied_history(1.0), position(symbol="PEPE-USDT"))
    signal_a = engine(varied_history(1.0), position(symbol="DOGE-USDT"))
    assert signal_b.action == signal_a.action == "exit"
    assert signal_b.reason_code == signal_a.reason_code == "take_profit"
    assert signal_b.symbol == "PEPE-USDT"
    assert signal_a.symbol == "DOGE-USDT"


def test_position_divergence_stop_and_opposite_rotation_fact() -> None:
    stop = engine(varied_history(2.0), position(symbol="PEPE-USDT"))
    rotate = engine(varied_history(0.5), position(symbol="PEPE-USDT"))
    assert stop.action == "exit"
    assert stop.reason_code == "diverge_stop"
    assert rotate.reason_code == "rotate_to_a"
    assert rotate.symbol == "PEPE-USDT"


def test_position_timeout_exit() -> None:
    signal = engine(varied_history(1.1), position(symbol="PEPE-USDT", entry_index=0))
    assert signal.action == "exit"
    assert signal.reason_code == "time_stop"
    assert signal.indicators["position_age_bars"] > 180


def test_insufficient_history_and_zero_std_are_blocked() -> None:
    insufficient = engine([0.9, 1.1] * 100)
    zero_std = engine([1.0] * 240)
    assert insufficient.action == "blocked"
    assert insufficient.reason_code == "insufficient_history"
    assert zero_std.action == "blocked"
    assert zero_std.reason_code == "zero_std"
    assert zero_std.indicators["ratio"] == zero_std.indicators["mean"] == 1.0
    assert zero_std.indicators["std"] == 0.0


def test_position_pair_identity_is_enforced() -> None:
    signal = engine(varied_history(1.0), position(symbol="DOGE-USDT", pair="AAVE/LINK"))
    assert signal.action == "blocked"
    assert signal.reason_code == "position_pair_mismatch"
    assert signal.pair == "AAVE/LINK"


def test_missing_pair_leg_is_explicitly_blocked() -> None:
    candles = make_candles(varied_history(1.5))
    input_data = FixedEngineInput(
        candles=[candle for candle in candles if candle.symbol == "DOGE-USDT"],
        observed_at=UTC_NOW,
    )
    signal = evaluate_pair_rotation(input_data)
    assert signal.action == "blocked"
    assert signal.reason_code == "missing_pair_candles"
    assert signal.conditions
