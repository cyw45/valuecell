from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from valuecell.server.api.schemas.rule_strategy import RuleStrategyConfig
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.rule_strategy import (
    RuleStrategy,
    RuleStrategyEvaluationJournal,
)
from valuecell.server.services import strategy_scheduler


class DurableQuery:
    def __init__(self, session, model):
        self.session, self.model = session, model
        self.filters = {}

    def filter_by(self, **kwargs):
        self.filters.update(kwargs)
        return self

    def with_for_update(self):
        self.session.locked = True
        return self

    def first(self):
        if self.model is strategy_scheduler.RuleStrategy:
            return self.session.strategy
        return self.session.intent_by_evaluation.get(self.filters.get("evaluation_id"))

    def all(self):
        return self.session.intents


class DurableSession:
    def __init__(self, config, intents=()):
        self.strategy = SimpleNamespace(
            strategy_id="rule-a",
            tenant_id="tenant-a",
            status="running",
            execution_generation=1,
            config=config.model_dump(mode="json"),
        )
        self.intents = list(intents)
        self.intent_by_evaluation = {}
        self.locked = False

    def query(self, model):
        return DurableQuery(self, model)

    def add(self, item):
        evaluation_id = getattr(item, "evaluation_id", None)
        if evaluation_id:
            self.intent_by_evaluation[evaluation_id] = item
            self.intents.append(item)

    def flush(self):
        pass

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


class FakeJob:
    def __init__(self, job_id, args, next_run_time):
        self.id = job_id
        self.args = tuple(args)
        self.next_run_time = next_run_time


class FakeScheduler:
    """Minimal APScheduler-compatible fake for job synchronization tests."""

    def __init__(self):
        self.jobs = {}

    def get_jobs(self):
        return list(self.jobs.values())

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def remove_job(self, job_id):
        self.jobs.pop(job_id)

    def add_job(
        self,
        func,
        trigger=None,
        *,
        id,
        args=(),
        next_run_time=None,
        **kwargs,
    ):
        del func, trigger, kwargs
        self.jobs[id] = FakeJob(id, args, next_run_time)
        return self.jobs[id]


def test_sync_does_not_postpone_unchanged_running_strategy(monkeypatch):
    config = RuleStrategyConfig(interval="5m").model_dump(mode="json")
    strategy = SimpleNamespace(
        strategy_id="rule-demo", tenant_id="tenant-demo", config=config
    )
    monkeypatch.setattr(
        strategy_scheduler,
        "RuleStrategyRepository",
        lambda db_session: SimpleNamespace(list_running=lambda: [strategy]),
    )
    monkeypatch.setattr(
        strategy_scheduler.TenantAccessService,
        "access_for",
        lambda _db, _tenant_id: SimpleNamespace(active=True),
    )
    scheduler = strategy_scheduler.StrategyScheduler()
    fake_scheduler = FakeScheduler()
    monkeypatch.setattr(scheduler, "_scheduler", fake_scheduler)
    scheduler.sync_running_strategies(SimpleNamespace())
    first_job = fake_scheduler.get_job(strategy.strategy_id)
    assert first_job is not None
    first_next_run = first_job.next_run_time
    assert first_next_run is not None
    scheduler.sync_running_strategies(SimpleNamespace())
    second_job = fake_scheduler.get_job(strategy.strategy_id)
    assert second_job is not None
    assert second_job.next_run_time == first_next_run


@pytest.mark.asyncio
async def test_paper_execution_never_routes_to_a_live_connection():
    result = await strategy_scheduler.StrategyScheduler._execute_signal(
        "tenant-a",
        "rule-a",
        RuleStrategyConfig(symbols=["BTC-USDT"]),
        "BTC-USDT",
        "buy",
        Decimal("100"),
        Decimal("50000"),
        1234,
    )
    assert result == {
        "execution": "paper_filled",
        "execution_ledger": "paper",
        "paper_fill": True,
        "sandbox": False,
    }


def test_sync_does_not_remove_application_maintenance_jobs(monkeypatch):
    config = RuleStrategyConfig(interval="5m").model_dump(mode="json")
    strategy = SimpleNamespace(
        strategy_id="rule-demo", tenant_id="tenant-demo", config=config
    )
    monkeypatch.setattr(
        strategy_scheduler,
        "RuleStrategyRepository",
        lambda db_session: SimpleNamespace(list_running=lambda: [strategy]),
    )
    monkeypatch.setattr(
        strategy_scheduler.TenantAccessService,
        "access_for",
        lambda _db, _tenant_id: SimpleNamespace(active=True),
    )
    scheduler = strategy_scheduler.StrategyScheduler()
    scheduler._scheduler.add_job(
        lambda: None,
        trigger="interval",
        seconds=60,
        id="_scheduler_reconcile_demo_execution",
    )
    scheduler.sync_running_strategies(SimpleNamespace())
    assert (
        scheduler._scheduler.get_job("_scheduler_reconcile_demo_execution") is not None
    )


@pytest.mark.asyncio
async def test_okx_demo_submission_is_not_recorded_as_a_paper_fill(monkeypatch):
    class FakeService:
        def __init__(self, _session):
            pass

        async def submit_order(self, *_args, **_kwargs):
            return {"id": "demo-order", "status": "open", "sandbox": True}

    config = RuleStrategyConfig.model_validate(
        {
            "symbols": ["BTC-USDT"],
            "execution": {
                "environment": "okx_demo",
                "sandbox_connection_id": "okx-demo-connection",
            },
        }
    )
    session = DurableSession(config)
    monkeypatch.setattr(
        strategy_scheduler, "SandboxExchangeTradingService", FakeService
    )
    monkeypatch.setattr(
        strategy_scheduler,
        "get_database_manager",
        lambda: SimpleNamespace(get_session=lambda: session),
    )
    result = await strategy_scheduler.StrategyScheduler._execute_signal(
        "tenant-a",
        "rule-a",
        config,
        "BTC-USDT",
        "buy",
        Decimal("100"),
        Decimal("50000"),
        1234,
        "eval-a",
    )
    assert result["execution"] == "okx_demo_submitted"
    assert result["paper_fill"] is False
    assert result["execution_ledger"] == "okx_demo"
    assert session.locked is True


@pytest.mark.asyncio
async def test_okx_demo_execution_uses_bound_sandbox_connection_and_deterministic_id(
    monkeypatch,
):
    calls = []

    class FakeService:
        def __init__(self, session):
            assert isinstance(session, DurableSession)

        async def submit_order(
            self,
            tenant_id,
            credential_id,
            client_order_id,
            symbol,
            side,
            order_type,
            quote_amount,
            price,
            **kwargs,
        ):
            calls.append(
                (
                    tenant_id,
                    credential_id,
                    client_order_id,
                    symbol,
                    side,
                    order_type,
                    quote_amount,
                    price,
                )
            )
            assert kwargs["intent"].evaluation_id in {"eval-a", "eval-b"}
            assert kwargs["fenced"] is True
            return {"id": "demo-order", "status": "open", "sandbox": True}

    config = RuleStrategyConfig.model_validate(
        {
            "symbols": ["BTC-USDT"],
            "execution": {
                "environment": "okx_demo",
                "sandbox_connection_id": "okx-demo-connection",
            },
        }
    )
    session = DurableSession(config)
    monkeypatch.setattr(
        strategy_scheduler, "SandboxExchangeTradingService", FakeService
    )
    monkeypatch.setattr(
        strategy_scheduler,
        "get_database_manager",
        lambda: SimpleNamespace(get_session=lambda: session),
    )
    first = await strategy_scheduler.StrategyScheduler._execute_okx_demo_signal(
        "tenant-a",
        "rule-a",
        config,
        "BTC-USDT",
        "buy",
        Decimal("100"),
        Decimal("50000"),
        1234,
        "eval-a",
    )
    second = await strategy_scheduler.StrategyScheduler._execute_okx_demo_signal(
        "tenant-a",
        "rule-a",
        config,
        "BTC-USDT",
        "buy",
        Decimal("100"),
        Decimal("50000"),
        1234,
        "eval-b",
    )
    assert first["execution"] == second["execution"] == "okx_demo_submitted"
    assert len(calls) == 2
    assert calls[0][1] == "okx-demo-connection"
    assert calls[0][2] == calls[1][2]
    assert calls[0][2].startswith("vc-demo-")


@pytest.mark.asyncio
async def test_okx_demo_execution_blocks_when_strategy_total_limit_is_reached(
    monkeypatch,
):
    config = RuleStrategyConfig.model_validate(
        {
            "symbols": ["BTC-USDT"],
            "execution": {
                "environment": "okx_demo",
                "sandbox_connection_id": "okx-demo-connection",
                "max_order_quote_amount": 100,
                "max_total_quote_amount": 150,
            },
        }
    )
    session = DurableSession(
        config, [SimpleNamespace(requested_quote="100", status="open")]
    )
    monkeypatch.setattr(
        strategy_scheduler,
        "get_database_manager",
        lambda: SimpleNamespace(get_session=lambda: session),
    )
    result = await strategy_scheduler.StrategyScheduler._execute_okx_demo_signal(
        "tenant-a",
        "rule-a",
        config,
        "BTC-USDT",
        "buy",
        Decimal("100"),
        Decimal("50000"),
        1234,
        "eval-a",
    )
    assert result["execution"] == "blocked"
    assert "total limit" in result["reason"]


@pytest.mark.asyncio
async def test_okx_demo_execution_blocks_when_daily_limit_is_reached(monkeypatch):
    config = RuleStrategyConfig.model_validate(
        {
            "symbols": ["BTC-USDT"],
            "execution": {
                "environment": "okx_demo",
                "sandbox_connection_id": "okx-demo-connection",
                "max_order_quote_amount": 100,
                "max_daily_quote_amount": 150,
                "max_total_quote_amount": 500,
            },
        }
    )
    session = DurableSession(
        config,
        [
            SimpleNamespace(
                requested_quote="100",
                status="open",
                created_at=datetime.now(timezone.utc),
            )
        ],
    )
    monkeypatch.setattr(
        strategy_scheduler,
        "get_database_manager",
        lambda: SimpleNamespace(get_session=lambda: session),
    )
    result = await strategy_scheduler.StrategyScheduler._execute_okx_demo_signal(
        "tenant-a",
        "rule-a",
        config,
        "BTC-USDT",
        "buy",
        Decimal("100"),
        Decimal("50000"),
        1234,
        "eval-a",
    )
    assert result["execution"] == "blocked"
    assert "daily limit" in result["reason"]


def test_market_data_unavailable_reason_distinguishes_missing_and_stale_primary_candles():
    assert (
        strategy_scheduler._market_data_unavailable_reason(None)
        == "primary candles unavailable"
    )
    assert (
        strategy_scheduler._market_data_unavailable_reason(
            SimpleNamespace(freshness_status="stale", freshness_age_ms=121_000)
        )
        == "primary candles stale age_ms=121000"
    )


def test_record_execution_updates_durable_journal(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    session.add(
        RuleStrategy(
            strategy_id="rule-journal", tenant_id="tenant-a", name="journal", config={}
        )
    )
    session.add(
        RuleStrategyEvaluationJournal(
            evaluation_id="eval-journal",
            strategy_id="rule-journal",
            tenant_id="tenant-a",
            result={},
            trades=[{"action": "buy"}],
        )
    )
    session.commit()
    session.close()
    monkeypatch.setattr(
        strategy_scheduler,
        "get_database_manager",
        lambda: SimpleNamespace(get_session=session_factory),
    )

    strategy_scheduler.StrategyScheduler._record_execution(
        "tenant-a", "rule-journal", "eval-journal", {"execution": "paper_filled"}
    )

    verify = session_factory()
    try:
        journal = (
            verify.query(RuleStrategyEvaluationJournal)
            .filter_by(evaluation_id="eval-journal")
            .one()
        )
        assert journal.trades[-1]["execution"] == "paper_filled"
    finally:
        verify.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_record_execution_does_not_silence_database_errors(monkeypatch):
    class BrokenSession:
        def query(self, _model):
            raise RuntimeError("journal database unavailable")

        def rollback(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        strategy_scheduler,
        "get_database_manager",
        lambda: SimpleNamespace(get_session=BrokenSession),
    )
    with pytest.raises(RuntimeError, match="journal database unavailable"):
        strategy_scheduler.StrategyScheduler._record_execution(
            "tenant-a", "rule-a", "eval-a", {"execution": "paper_filled"}
        )


@pytest.mark.parametrize("create_journal", [False, True])
def test_record_execution_fails_when_journal_or_target_trade_is_missing(
    monkeypatch, create_journal
):
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    session.add(
        RuleStrategy(
            strategy_id="rule-missing",
            tenant_id="tenant-a",
            name="missing",
            config={},
        )
    )
    if create_journal:
        session.add(
            RuleStrategyEvaluationJournal(
                evaluation_id="eval-missing",
                strategy_id="rule-missing",
                tenant_id="tenant-a",
                result={},
                trades=[],
            )
        )
    session.commit()
    session.close()
    monkeypatch.setattr(
        strategy_scheduler,
        "get_database_manager",
        lambda: SimpleNamespace(get_session=session_factory),
    )

    try:
        with pytest.raises(RuntimeError, match="journal|trade"):
            strategy_scheduler.StrategyScheduler._record_execution(
                "tenant-a",
                "rule-missing",
                "eval-missing",
                {"execution": "paper_filled"},
            )
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()
