from datetime import datetime, timezone

import pytest

from valuecell.server.api.schemas.fixed_strategy import FixedCandle, FixedEngineInput
from valuecell.server.services.fixed_strategy_dispatcher import (
    FixedStrategyEngineUnavailableError,
    evaluate_fixed_strategy,
)


def test_dispatcher_rejects_leader_without_btc_facts() -> None:
    candle = FixedCandle(
        symbol="BTC-USDT",
        timestamp_ms=1_700_000_000_000,
        open=100,
        high=101,
        low=99,
        close=100,
        volume=1,
    )
    request = FixedEngineInput(
        candles=[candle],
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    with pytest.raises(FixedStrategyEngineUnavailableError):
        evaluate_fixed_strategy("leader_breakout", request)
