from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from valuecell.server.api.schemas.rule_strategy import RuleStrategyConfig
from valuecell.server.services import strategy_scheduler


@pytest.mark.asyncio
async def test_sync_does_not_postpone_unchanged_running_strategy(monkeypatch):
    """A frequent sync must not reset a slower strategy job's next run time."""
    config = RuleStrategyConfig(interval="5m").model_dump(mode="json")
    strategy = SimpleNamespace(
        strategy_id="rule-demo",
        tenant_id="tenant-demo",
        config=config,
    )
    repository = SimpleNamespace(list_running=lambda: [strategy])
    monkeypatch.setattr(
        strategy_scheduler,
        "RuleStrategyRepository",
        lambda db_session: repository,
    )
    monkeypatch.setattr(
        strategy_scheduler.TenantAccessService,
        "access_for",
        lambda _db, _tenant_id: SimpleNamespace(active=True),
    )
    scheduler = strategy_scheduler.StrategyScheduler()
    await scheduler.start()
    scheduler._scheduler.pause()
    try:
        scheduler.sync_running_strategies(SimpleNamespace())
        first_job = scheduler._scheduler.get_job(strategy.strategy_id)
        assert first_job is not None
        assert first_job.next_run_time <= datetime.now(timezone.utc)

        first_next_run = first_job.next_run_time
        scheduler.sync_running_strategies(SimpleNamespace())
        synced_job = scheduler._scheduler.get_job(strategy.strategy_id)

        assert synced_job is not None
        assert synced_job.next_run_time == first_next_run
    finally:
        await scheduler.stop()



@pytest.mark.asyncio
async def test_paper_execution_never_routes_to_a_live_connection() -> None:
    config = RuleStrategyConfig(symbols=["BTC-USDT"])

    result = await strategy_scheduler.StrategyScheduler._execute_signal(
        "tenant-a",
        "rule-a",
        config,
        "BTC-USDT",
        "buy",
        Decimal("100"),
        Decimal("50000"),
        1234,
    )

    assert result == {"execution": "paper_filled", "sandbox": False}

@pytest.mark.asyncio
async def test_okx_demo_execution_uses_bound_sandbox_connection_and_deterministic_id(monkeypatch):
    calls = []

    class FakeQuery:
        def filter_by(self, **_kwargs):
            return self

        def all(self):
            return []

    class FakeSession:
        def query(self, _model):
            return FakeQuery()

        def rollback(self):
            pass

        def close(self):
            pass

    class FakeService:
        def __init__(self, session):
            assert isinstance(session, FakeSession)

        async def submit_order(self, tenant_id, credential_id, client_order_id, symbol, side, order_type, quote_amount, price):
            calls.append((tenant_id, credential_id, client_order_id, symbol, side, order_type, quote_amount, price))
            return {"id": "demo-order", "status": "open", "sandbox": True}

    monkeypatch.setattr(strategy_scheduler, "SandboxExchangeTradingService", FakeService)
    monkeypatch.setattr(
        strategy_scheduler, "get_database_manager", lambda: SimpleNamespace(get_session=FakeSession)
    )

    config = RuleStrategyConfig.model_validate({
        "symbols": ["BTC-USDT"],
        "execution": {"environment": "okx_demo", "sandbox_connection_id": "okx-demo-connection"},
    })
    first = await strategy_scheduler.StrategyScheduler._execute_okx_demo_signal(
        "tenant-a", "rule-a", config, "BTC-USDT", "buy", Decimal("100"), Decimal("50000"), 1234
    )
    second = await strategy_scheduler.StrategyScheduler._execute_okx_demo_signal(
        "tenant-a", "rule-a", config, "BTC-USDT", "buy", Decimal("100"), Decimal("50000"), 1234
    )

    assert first["execution"] == "okx_demo_submitted"
    assert second["execution"] == "okx_demo_submitted"
    assert len(calls) == 2
    assert calls[0][1] == "okx-demo-connection"
    assert calls[0][2] == calls[1][2]
    assert calls[0][2].startswith("vc-demo-")


@pytest.mark.asyncio
async def test_okx_demo_execution_blocks_when_strategy_total_limit_is_reached(monkeypatch):
    class FakeQuery:
        def filter_by(self, **_kwargs):
            return self

        def all(self):
            return [SimpleNamespace(requested_quote="100", status="open")]

    class FakeSession:
        def query(self, _model):
            return FakeQuery()

        def rollback(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        strategy_scheduler, "get_database_manager", lambda: SimpleNamespace(get_session=FakeSession)
    )
    config = RuleStrategyConfig.model_validate({
        "symbols": ["BTC-USDT"],
        "execution": {
            "environment": "okx_demo",
            "sandbox_connection_id": "okx-demo-connection",
            "max_order_quote_amount": 100,
            "max_total_quote_amount": 150,
        },
    })
    result = await strategy_scheduler.StrategyScheduler._execute_okx_demo_signal(
        "tenant-a", "rule-a", config, "BTC-USDT", "buy", Decimal("100"), Decimal("50000"), 1234
    )
    assert result["execution"] == "blocked"
    assert "total limit" in result["reason"]


@pytest.mark.asyncio
async def test_okx_demo_execution_blocks_when_daily_limit_is_reached(monkeypatch):
    class FakeQuery:
        def filter_by(self, **_kwargs):
            return self

        def all(self):
            return [
                SimpleNamespace(
                    requested_quote="100", status="open", created_at=datetime.now(timezone.utc)
                )
            ]

    class FakeSession:
        def query(self, _model):
            return FakeQuery()

        def rollback(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        strategy_scheduler, "get_database_manager", lambda: SimpleNamespace(get_session=FakeSession)
    )
    config = RuleStrategyConfig.model_validate({
        "symbols": ["BTC-USDT"],
        "execution": {
            "environment": "okx_demo",
            "sandbox_connection_id": "okx-demo-connection",
            "max_order_quote_amount": 100,
            "max_daily_quote_amount": 150,
            "max_total_quote_amount": 500,
        },
    })
    result = await strategy_scheduler.StrategyScheduler._execute_okx_demo_signal(
        "tenant-a", "rule-a", config, "BTC-USDT", "buy", Decimal("100"), Decimal("50000"), 1234
    )
    assert result["execution"] == "blocked"
    assert "daily limit" in result["reason"]
