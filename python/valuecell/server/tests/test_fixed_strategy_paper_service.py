from decimal import Decimal

import pytest

from datetime import datetime, timezone


from valuecell.server.api.schemas.fixed_strategy import (
    FixedCandle,
    FixedEngineInput,
    FixedStrategySignal,
)
from valuecell.server.api.schemas.rule_strategy import RuleStrategyConfig
from valuecell.server.services.fixed_strategy_paper_service import (
    FixedDemoExecutionAdapter,
    FixedPaperEvaluationService,
)


class RecordingRepository:
    def __init__(self) -> None:
        self.journal = None

    def append_evaluation(self, journal):
        self.journal = journal
        journal.evaluation_id = "fixed-evaluation-1"
        return journal


def test_fixed_paper_service_persists_signal_conditions_and_batch() -> None:
    repository = RecordingRepository()
    candles = [
        FixedCandle(
            symbol="BTC-USDT",
            timestamp_ms=1_700_000_000_000 + index * 14_400_000,
            open=close,
            high=close + 1,
            low=close - 1,
            close=close,
            volume=1,
        )
        for index, close in enumerate([100] * 21 + [101])
    ]
    signal, evaluation_id = FixedPaperEvaluationService(repository).evaluate_and_record(
        strategy_id="strategy-a",
        tenant_id="tenant-a",
        strategy_kind="dual_ma_trend",
        batch_id="batch-a",
        request=FixedEngineInput(
            candles=candles,
            observed_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
        ),
    )
    assert signal.action == "long_entry"
    assert evaluation_id == "fixed-evaluation-1"
    assert repository.journal.batch_id == "batch-a"
    assert repository.journal.result["conditions"]
    assert repository.journal.result["execution_ledger"] == "paper_signal_only"


def test_fixed_demo_evaluation_never_labels_signal_as_paper() -> None:
    repository = RecordingRepository()
    candles = [
        FixedCandle(
            symbol="BTC-USDT",
            timestamp_ms=1_700_000_000_000 + index * 14_400_000,
            open=close,
            high=close + 1,
            low=close - 1,
            close=close,
            volume=1,
        )
        for index, close in enumerate([100] * 21 + [101])
    ]

    FixedPaperEvaluationService(repository).evaluate_and_record(
        strategy_id="strategy-a",
        tenant_id="tenant-a",
        strategy_kind="dual_ma_trend",
        batch_id="batch-a",
        request=FixedEngineInput(
            candles=candles,
            observed_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
        ),
        environment="okx_demo",
    )

    assert repository.journal.result["execution_ledger"] == "okx_demo"


def _demo_config() -> RuleStrategyConfig:
    return RuleStrategyConfig.model_validate(
        {
            "symbols": ["BTC-USDT"],
            "risk": {"order_quote_amount": 250},
            "execution": {
                "environment": "okx_demo",
                "sandbox_connection_id": "demo-connection",
            },
        }
    )


def _signal(kind: str, action: str) -> FixedStrategySignal:
    return FixedStrategySignal.model_validate(
        {
            "kind": kind,
            "symbol": "BTC-USDT",
            "action": action,
            "reason_code": "test_signal",
            "reason": "test signal",
            "observed_at": "2026-08-29T00:00:00Z",
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["dual_ma_trend", "pair_rotation", "leader_breakout"])
async def test_fixed_demo_adapter_routes_each_kind_long_entry_to_shared_boundary(kind: str) -> None:
    calls = []

    async def shared_boundary(*args):
        calls.append(args)
        return {"execution": "okx_demo_submitted"}

    result = await FixedDemoExecutionAdapter(shared_boundary).execute(
        tenant_id="tenant-a",
        strategy_id="strategy-a",
        config=_demo_config(),
        signal=_signal(kind, "long_entry"),
        price=Decimal("100"),
        candle_timestamp_ms=1_700_000_000_000,
        evaluation_id="evaluation-a",
    )

    assert result == {"execution": "okx_demo_submitted"}
    assert calls == [
        (
            "tenant-a", "strategy-a", _demo_config(), "BTC-USDT", "buy",
            Decimal("250"), Decimal("100"), 1_700_000_000_000, "evaluation-a",
        )
    ]


@pytest.mark.asyncio
async def test_fixed_demo_adapter_blocks_short_without_venue_submission() -> None:
    async def shared_boundary(*_args):
        raise AssertionError("spot venue must not receive a fixed short action")

    result = await FixedDemoExecutionAdapter(shared_boundary).execute(
        tenant_id="tenant-a",
        strategy_id="strategy-a",
        config=_demo_config(),
        signal=_signal("dual_ma_trend", "short_entry"),
        price=Decimal("100"),
        candle_timestamp_ms=1_700_000_000_000,
        evaluation_id="evaluation-a",
    )

    assert result is not None
    assert result["execution"] == "blocked_execution_environment"
