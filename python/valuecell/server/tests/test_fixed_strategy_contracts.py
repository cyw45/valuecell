from datetime import datetime, timezone

from valuecell.server.api.schemas.fixed_strategy import (
    FixedCandle,
    FixedEngineInput,
    FixedStrategySignal,
)
from valuecell.server.services.fixed_dual_ma_engine import evaluate_fixed_dual_ma


def test_fixed_signal_always_contains_strategy_identity_and_explanation() -> None:
    candles = [
        FixedCandle(
            symbol="BTC-USDT",
            timestamp_ms=(1_700_000_000_000 + index * 14_400_000),
            open=100,
            high=101,
            low=99,
            close=100,
            volume=10,
        )
        for index in range(22)
    ]
    signal = evaluate_fixed_dual_ma(
        FixedEngineInput(
            candles=candles,
            observed_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
        )
    )
    assert isinstance(signal, FixedStrategySignal)
    assert signal.kind == "dual_ma_trend"
    assert signal.reason_code
    assert signal.reason
