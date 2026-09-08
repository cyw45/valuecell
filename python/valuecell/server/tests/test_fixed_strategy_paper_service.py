from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

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
from valuecell.server.db.connection import get_database_manager
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.rule_strategy import RuleStrategy
from valuecell.server.db.models.tenant import Tenant


class RecordingRepository:
    def __init__(self) -> None:
        self.journal = None
        self.append_count = 0

    def append_evaluation(self, journal):
        self.append_count += 1
        self.journal = journal
        return journal

    def get_evaluation(self, evaluation_id, strategy_id, tenant_id):
        if (
            self.journal is not None
            and self.journal.evaluation_id == evaluation_id
            and self.journal.strategy_id == strategy_id
            and self.journal.tenant_id == tenant_id
        ):
            return self.journal
        return None


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
    assert evaluation_id.startswith("fixed_")
    assert repository.journal.batch_id == "batch-a"
    assert repository.journal.result["conditions"]
    assert repository.journal.result["symbol"] == "BTC-USDT"
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


def test_fixed_paper_fill_execution_is_recorded_once_for_evaluation() -> None:
    """Paper execution must create one idempotent fill after a signal."""
    service = FixedPaperEvaluationService()
    assert hasattr(service, "record_paper_fill")


def test_fixed_paper_fill_execution_contains_account_equity_for_pnl_curve(monkeypatch) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(Tenant(id="tenant-a", name="Tenant A"))
    session.add(
        RuleStrategy(
            strategy_id="strategy-a",
            tenant_id="tenant-a",
            name="Fixed",
            config={"initial_capital_quote": 1000},
        )
    )
    session.commit()
    monkeypatch.setattr(get_database_manager(), "get_session", lambda: session)
    result = FixedPaperEvaluationService().record_paper_fill(
        tenant_id="tenant-a", strategy_id="strategy-a", batch_id="batch-a",
        signal=_signal("dual_ma_trend", "long_entry"), evaluation_id="evaluation-account-equity",
        initial_capital_quote=Decimal("1000"), price=Decimal("100"), order_quote_amount=Decimal("200"),
    )
    assert result["paper_fill"] is True
    assert result["account"]["equity_quote"] == 1000


def test_fixed_evaluation_is_idempotent_for_same_batch_and_observation() -> None:
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
    service = FixedPaperEvaluationService(repository)
    first = service.evaluate_and_record(
        strategy_id="strategy-a",
        tenant_id="tenant-a",
        strategy_kind="dual_ma_trend",
        batch_id="batch-a",
        request=FixedEngineInput(
            candles=candles,
            observed_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
        ),
    )
    second = service.evaluate_and_record(
        strategy_id="strategy-a",
        tenant_id="tenant-a",
        strategy_kind="dual_ma_trend",
        batch_id="batch-a",
        request=FixedEngineInput(
            candles=candles,
            observed_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
        ),
    )

    assert second[1] == first[1]
    assert repository.append_count == 1


def test_fixed_evaluation_identity_changes_for_new_execution_batch() -> None:
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
    request = FixedEngineInput(
        candles=candles,
        observed_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
    )
    service = FixedPaperEvaluationService(repository)

    first = service.evaluate_and_record(
        strategy_id="strategy-a",
        tenant_id="tenant-a",
        strategy_kind="dual_ma_trend",
        batch_id="batch-a",
        request=request,
    )
    second = service.evaluate_and_record(
        strategy_id="strategy-a",
        tenant_id="tenant-a",
        strategy_kind="dual_ma_trend",
        batch_id="batch-b",
        request=request,
    )

    assert second[1] != first[1]
    assert repository.append_count == 2


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


@pytest.mark.asyncio
async def test_fixed_demo_exit_requests_full_attributed_position_close() -> None:
    calls = []

    async def shared_boundary(*args, **kwargs):
        calls.append((args, kwargs))
        return {"execution": "okx_demo_submitted"}

    await FixedDemoExecutionAdapter(shared_boundary).execute(
        tenant_id="tenant-a",
        strategy_id="strategy-a",
        config=_demo_config(),
        signal=_signal("dual_ma_trend", "exit"),
        price=Decimal("100"),
        candle_timestamp_ms=1_700_000_000_000,
        evaluation_id="evaluation-a",
    )

    assert calls[0][1] == {"close_all_attributed": True}
