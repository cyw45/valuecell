from datetime import UTC, datetime
import pytest

from valuecell.server.api.schemas.fixed_strategy import FixedCandle, FixedEngineInput
from valuecell.server.services.fixed_leader_engine import (
    evaluate_leader_candidate,
    evaluate_leader_candidates,
)


NOW = datetime(2026, 8, 28, 12, tzinfo=UTC)
START_MS = 1_728_000_000_000


def _candles(
    symbol: str,
    *,
    count: int = 60,
    start_price: float = 100.0,
    step: float = 2.0,
    quote_volume: float | None = 50_000.0,
    volumes: dict[int, float] | None = None,
    closes: list[float] | None = None,
) -> list[FixedCandle]:
    volume_by_index = volumes or {}
    prices = closes or [start_price + step * index for index in range(count)]
    return [
        FixedCandle(
            symbol=symbol,
            timestamp_ms=START_MS + index * 4 * 60 * 60 * 1_000,
            open=price,
            high=price + 0.5,
            low=price - 0.5,
            close=price,
            volume=volume_by_index.get(index, 10.0),
            quote_volume=quote_volume,
        )
        for index, price in enumerate(prices)
    ]


def _input(candles: list[FixedCandle]) -> FixedEngineInput:
    return FixedEngineInput(candles=candles, observed_at=NOW)


def _bullish_btc() -> list[FixedCandle]:
    return _candles("BTC-USDT", start_price=50_000, step=100)


def _condition(signal, code: str):
    return next(condition for condition in signal.conditions if condition.code == code)


def test_leader_engine_rejects_low_liquidity_and_relative_strength():
    low_liquidity = evaluate_leader_candidate(
        _input(_candles("LOW-USDT", quote_volume=1_000)), _bullish_btc()
    )
    assert low_liquidity.action == "no_signal"
    assert low_liquidity.reason_code == "liquidity_below_minimum"
    assert _condition(low_liquidity, "liquidity_minimum").state == "not_triggered"

    weak_rs = evaluate_leader_candidate(
        _input(_candles("WEAK-USDT", step=0.1)), _bullish_btc()
    )
    assert weak_rs.action == "no_signal"
    assert weak_rs.reason_code == "relative_strength_below_minimum"
    assert _condition(weak_rs, "relative_strength_minimum").state == "not_triggered"


def test_leader_engine_requires_both_btc_ema_gates():
    bearish_btc = _candles("BTC-USDT", start_price=50_000, step=-100)
    blocked = evaluate_leader_candidate(_input(_candles("ALT-USDT")), bearish_btc)

    assert blocked.action == "no_signal"
    assert blocked.reason_code == "btc_ema_trend_blocked"
    assert _condition(blocked, "btc_ema_bullish").state == "not_triggered"

    prices = [50_000 + index * 100 for index in range(48)] + [52_000] * 12
    slope_blocked = evaluate_leader_candidate(
        _input(_candles("SLOPE-USDT")),
        _candles("BTC-USDT", closes=prices),
    )
    assert slope_blocked.action == "no_signal"
    assert slope_blocked.reason_code == "btc_ema55_slope_blocked"
    assert _condition(slope_blocked, "btc_ema_bullish").state == "triggered"
    assert _condition(slope_blocked, "btc_ema55_rising").state == "not_triggered"


def test_leader_engine_confirms_path_a_and_explains_ecs_facts():
    signal = evaluate_leader_candidate(
        _input(_candles("PATHA-USDT", volumes={59: 30.0})), _bullish_btc()
    )

    assert signal.action == "long_entry"
    assert signal.reason_code == "path_a_confirmed"
    assert signal.indicators["entry_path"] == "path_a"
    assert signal.indicators["ecs_score"] == 46.0
    assert signal.indicators["quote_volume_24h"] == 300_000.0
    assert _condition(signal, "path_a_breakout").state == "triggered"
    assert _condition(signal, "ecs_score").actual == 46.0


def test_leader_engine_confirms_path_b_from_latest_prior_breakout():
    signal = evaluate_leader_candidate(
        _input(_candles("PATHB-USDT", volumes={55: 30.0})), _bullish_btc()
    )

    assert signal.action == "long_entry"
    assert signal.reason_code == "path_b_confirmed"
    assert signal.indicators["entry_path"] == "path_b"
    assert signal.indicators["path_b_breakout_close"] == 210.0
    assert signal.indicators["path_b_pullback_floor"] == pytest.approx(205.8)
    assert _condition(signal, "path_a_breakout").state == "not_triggered"
    assert _condition(signal, "path_b_pullback").state == "triggered"
    closes = [100.0 + index * 2.0 for index in range(60)]
    closes[53:60] = [195.0, 196.0, 210.0, 195.0, 195.0, 195.0, 203.0]
    latest_required = evaluate_leader_candidate(
        _input(
            _candles(
                "LATEST-USDT",
                closes=closes,
                volumes={50: 30.0, 55: 30.0},
            )
        ),
        _bullish_btc(),
    )
    assert latest_required.reason_code == "entry_path_not_confirmed"
    assert latest_required.indicators["path_b_breakout_close"] == 210.0
    assert _condition(latest_required, "path_b_pullback").state == "not_triggered"


def test_leader_engine_ranks_entries_by_ecs_then_symbol_tie_breaker():
    signals = evaluate_leader_candidates(
        [
            _input(_candles("ZETA-USDT", volumes={59: 30.0})),
            _input(_candles("BETA-USDT", volumes={55: 30.0})),
            _input(_candles("ALFA-USDT", volumes={59: 30.0})),
        ],
        _bullish_btc(),
    )

    assert [(signal.symbol, signal.indicators.get("ecs_score")) for signal in signals] == [
        ("ALFA-USDT", 46.0),
        ("ZETA-USDT", 46.0),
        ("BETA-USDT", 33.0),
    ]


def test_leader_engine_fails_closed_with_insufficient_or_nonfinal_data():
    insufficient = evaluate_leader_candidate(
        _input(_candles("SHORT-USDT", count=59)), _bullish_btc()
    )
    assert insufficient.action == "blocked"
    assert insufficient.reason_code == "candidate_data_insufficient"
    assert insufficient.indicators["candidate_candle_count"] == 59.0
    assert _condition(insufficient, "candidate_data_sufficient").state == "blocked"

    candles = _candles("OPEN-USDT")
    candles[-1] = candles[-1].model_copy(update={"is_closed": False})
    nonfinal = evaluate_leader_candidate(_input(candles), _bullish_btc())
    assert nonfinal.action == "blocked"
    assert nonfinal.reason_code == "candidate_candles_invalid"
    assert _condition(nonfinal, "candidate_candles").state == "blocked"
