from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from valuecell.server.api.schemas.leader_spot_v19_backtest import (
    LeaderSpotV19BacktestCandle,
    LeaderSpotV19BacktestRequest,
    LeaderSpotV19BacktestSignal,
)
from valuecell.server.services.leader_spot_v19_backtest_service import (
    LeaderSpotV19BacktestEngine,
)


START = datetime(2025, 1, 1, tzinfo=UTC)
DAY_MS = 24 * 60 * 60 * 1_000


def _candles(days: int = 366):
    return [
        LeaderSpotV19BacktestCandle(
            symbol="BTC-USDT",
            timestamp_ms=int((START + timedelta(days=index)).timestamp() * 1_000),
            open=100 + index,
            high=101 + index,
            low=99 + index,
            close=100 + index,
            volume=1_000,
        )
        for index in range(days)
    ]


def _request(signals):
    return LeaderSpotV19BacktestRequest(
        initial_equity_quote=1_000,
        candles=_candles(),
        signals=signals,
        config_snapshot={"position": {"order_amount_quote": 100}},
        data_source="frozen-synthetic",
    )


def test_backtest_requires_twelve_month_frozen_coverage():
    with pytest.raises(ValidationError, match="twelve months"):
        LeaderSpotV19BacktestRequest(
            initial_equity_quote=1_000,
            candles=_candles(30),
            config_snapshot={"position": {"order_amount_quote": 100}},
            data_source="frozen-synthetic",
        )


def test_v19_backtest_uses_next_bar_fee_slippage_and_fixed_order_amount():
    signals = [
        LeaderSpotV19BacktestSignal(
            symbol="BTC-USDT", timestamp_ms=_candles()[30].timestamp_ms,
            action="entry", reason_code="ENTRY",
        ),
        LeaderSpotV19BacktestSignal(
            symbol="BTC-USDT", timestamp_ms=_candles()[60].timestamp_ms,
            action="close", reason_code="TREND_EXIT",
        ),
    ]

    result = LeaderSpotV19BacktestEngine().run(_request(signals))

    assert len(result.fills) == 2
    buy, sell = result.fills
    assert buy.fill_timestamp_ms == _candles()[31].timestamp_ms
    assert buy.quote_amount == 100
    assert buy.fee_quote == pytest.approx(0.1)
    assert buy.fill_price == pytest.approx(131 * 1.005)
    assert sell.fill_price == pytest.approx(161 * 0.995)
    assert result.metrics["closed_trade_count"] == 1
    assert result.metrics["fill_count"] == 2
    assert result.metrics["final_equity_quote"] > 1_000


def test_v19_backtest_is_reproducible_and_emits_rolling_walk_forward_windows():
    signals = [
        LeaderSpotV19BacktestSignal(
            symbol="BTC-USDT", timestamp_ms=_candles()[30].timestamp_ms,
            action="entry", reason_code="ENTRY",
        ),
        LeaderSpotV19BacktestSignal(
            symbol="BTC-USDT", timestamp_ms=_candles()[60].timestamp_ms,
            action="close", reason_code="STOP_LOSS_8PCT",
        ),
    ]
    request = _request(signals)
    engine = LeaderSpotV19BacktestEngine()

    first = engine.run(request)
    second = engine.run(request)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert len(first.walk_forward) >= 1
    assert first.data_fingerprint
    assert first.config_fingerprint
    assert first.assumptions_fingerprint
