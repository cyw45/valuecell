from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from valuecell.server.api.schemas.rule_strategy import (
    RuleStrategyCandle,
    RuleStrategyConfig,
    RuleStrategyEngineMarketSnapshot,
    RuleStrategyPosition,
)
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


@pytest.mark.asyncio
async def test_okx_demo_limits_use_actual_order_cost_after_sell_preflight(monkeypatch):
    config = RuleStrategyConfig.model_validate(
        {
            "symbols": ["BTC-USDT"],
            "execution": {
                "environment": "okx_demo",
                "sandbox_connection_id": "okx-demo-connection",
                "max_order_quote_amount": 100,
                "max_daily_quote_amount": 150,
                "max_total_quote_amount": 150,
            },
        }
    )
    session = DurableSession(
        config,
        [
            SimpleNamespace(
                requested_quote="100",
                request_payload={"order_cost": "40"},
                status="open",
                created_at=datetime.now(timezone.utc),
            )
        ],
    )

    class FakeService:
        def __init__(self, _session):
            pass

        async def submit_order(self, *_args, **_kwargs):
            return {"id": "demo-order", "status": "open", "sandbox": True}

    monkeypatch.setattr(strategy_scheduler, "SandboxExchangeTradingService", FakeService)
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
        "sell",
        Decimal("100"),
        Decimal("50000"),
        1234,
        "eval-next",
    )

    assert result["execution"] == "okx_demo_submitted"
    # The new order retains its nominal risk reservation until preflight records
    # its factual order_cost; the prior order contributes only its factual 40.
    assert session.intent_by_evaluation["eval-next"].requested_quote == "100"


@pytest.mark.asyncio
async def test_tick_persists_safe_fetch_failure_diagnostic_for_every_symbol(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    config = RuleStrategyConfig(symbols=["BTC-USDT", "ETH-USDT"], interval="5m")
    session.add(
        RuleStrategy(
            strategy_id="rule-market-data",
            tenant_id="tenant-a",
            name="market data",
            status="running",
            config=config.model_dump(mode="json"),
        )
    )
    session.commit()
    session.close()

    class FailingMarketService:
        async def get_indicators(self, **_kwargs):
            raise RuntimeError("provider secret raw failure")

    monkeypatch.setattr(
        strategy_scheduler,
        "get_database_manager",
        lambda: SimpleNamespace(get_session=session_factory),
    )
    monkeypatch.setattr(
        strategy_scheduler.TenantAccessService,
        "access_for",
        lambda _session, _tenant_id: SimpleNamespace(active=True),
    )
    monkeypatch.setattr(
        strategy_scheduler, "get_crypto_market_service", FailingMarketService
    )
    await strategy_scheduler.StrategyScheduler()._tick(
        "rule-market-data", "tenant-a", config.model_dump(mode="json")
    )

    verify = session_factory()
    try:
        journals = verify.query(RuleStrategyEvaluationJournal).all()
        assert {journal.result["symbol"] for journal in journals} == {
            "BTC-USDT",
            "ETH-USDT",
        }
        for journal in journals:
            diagnostic = journal.result
            assert diagnostic["stage"] == "market_data"
            assert diagnostic["status"] == "blocked"
            assert diagnostic["reason_code"] == "fetch_failed"
            assert diagnostic["action"] == "no_op"
            assert diagnostic["checked_at"]
            assert diagnostic["next_check_at"]
            assert "行情" in diagnostic["reason"]
            assert "provider secret raw failure" not in str(diagnostic)
            assert journal.signals == []
            assert journal.trades == []
    finally:
        verify.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.mark.asyncio
async def test_tick_fetch_failure_log_never_exposes_exception_text(monkeypatch):
    secret = "provider-secret-response-body"
    messages = []

    class FailingMarketService:
        async def get_indicators(self, **_kwargs):
            raise RuntimeError(secret)

    class Session:
        def close(self):
            pass

    monkeypatch.setattr(
        strategy_scheduler,
        "get_database_manager",
        lambda: SimpleNamespace(get_session=Session),
    )
    monkeypatch.setattr(
        strategy_scheduler.TenantAccessService,
        "access_for",
        lambda *_args: SimpleNamespace(active=True),
    )
    monkeypatch.setattr(
        strategy_scheduler, "get_crypto_market_service", FailingMarketService
    )
    monkeypatch.setattr(
        strategy_scheduler.StrategyScheduler,
        "_record_market_data_diagnostics",
        staticmethod(lambda *_args, **_kwargs: None),
    )
    sink_id = strategy_scheduler.logger.add(
        lambda message: messages.append(str(message)), level="WARNING"
    )
    try:
        config = RuleStrategyConfig(symbols=["BTC-USDT", "ETH-USDT"], interval="5m")
        await strategy_scheduler.StrategyScheduler()._tick(
            "rule-log-redaction", "tenant-a", config.model_dump(mode="json")
        )
    finally:
        strategy_scheduler.logger.remove(sink_id)

    output = "".join(messages)
    assert secret not in output
    assert "err=" not in output
    assert "safe_code=SCHEDULER_MARKET_FETCH_FAILED" in output
    assert "err_type=RuntimeError" in output
    assert "symbol_count=2" in output


@pytest.mark.asyncio
async def test_tick_market_failure_log_never_exposes_failed_symbols_details(monkeypatch):
    secret = "raw-provider-failed-symbol-detail"
    messages = []

    class MarketService:
        async def get_indicators(self, **_kwargs):
            return SimpleNamespace(
                symbols=[],
                failed_symbols={"BTC-USDT": secret},
            )

    class Session:
        def close(self):
            pass

    diagnostics = []
    monkeypatch.setattr(
        strategy_scheduler,
        "get_database_manager",
        lambda: SimpleNamespace(get_session=Session),
    )
    monkeypatch.setattr(
        strategy_scheduler.TenantAccessService,
        "access_for",
        lambda *_args: SimpleNamespace(active=True),
    )
    monkeypatch.setattr(strategy_scheduler, "get_crypto_market_service", MarketService)
    monkeypatch.setattr(
        strategy_scheduler.StrategyScheduler,
        "_record_market_data_diagnostics",
        staticmethod(lambda *args, **_kwargs: diagnostics.append(args)),
    )
    sink_id = strategy_scheduler.logger.add(
        lambda message: messages.append(str(message)), level="WARNING"
    )
    try:
        config = RuleStrategyConfig(symbols=["BTC-USDT"], interval="5m")
        await strategy_scheduler.StrategyScheduler()._tick(
            "rule-failed-symbol-redaction", "tenant-a", config.model_dump(mode="json")
        )
    finally:
        strategy_scheduler.logger.remove(sink_id)

    output = "".join(messages)
    assert secret not in output
    assert "failed_symbols" not in output
    assert "safe_code=SCHEDULER_MARKET_DATA_BLOCKED" in output
    assert "err_type=None" in output
    assert "symbol_count=1" in output
    assert diagnostics[0][3] == "fetch_failed"


@pytest.mark.asyncio
async def test_tick_blocks_when_enabled_secondary_interval_is_stale(monkeypatch):
    config = RuleStrategyConfig.model_validate(
        {
            "symbols": ["BTC-USDT"],
            "interval": "5m",
            "advanced_rules": {
                "enabled": True,
                "rsi": {"enabled": True, "interval": "1h", "period": 2},
            },
        }
    )
    candle = SimpleNamespace(ts=1, open=100, high=101, low=99, close=100, volume=1)

    class MarketService:
        async def get_indicators(self, *, interval, **_kwargs):
            return SimpleNamespace(
                symbols=[
                    SimpleNamespace(
                        symbol="BTC-USDT",
                        candles=[candle],
                        latest_price=100,
                        freshness_status="fresh" if interval == "5m" else "stale",
                        freshness_age_ms=120_000,
                    )
                ],
                failed_symbols={},
            )

    class Session:
        def close(self):
            pass

    diagnostics = []
    monkeypatch.setattr(
        strategy_scheduler,
        "get_database_manager",
        lambda: SimpleNamespace(get_session=Session),
    )
    monkeypatch.setattr(
        strategy_scheduler.TenantAccessService,
        "access_for",
        lambda *_args: SimpleNamespace(active=True),
    )
    monkeypatch.setattr(strategy_scheduler, "get_crypto_market_service", MarketService)
    monkeypatch.setattr(
        strategy_scheduler.StrategyScheduler,
        "_record_market_data_diagnostics",
        staticmethod(lambda *args, **_kwargs: diagnostics.append(args)),
    )

    class MustNotEvaluate:
        def evaluate_batch(self, *_args, **_kwargs):
            raise AssertionError("stale secondary candles must block evaluation")

    monkeypatch.setattr(strategy_scheduler, "RuleStrategyService", MustNotEvaluate)
    await strategy_scheduler.StrategyScheduler()._tick(
        "rule-secondary-stale", "tenant-a", config.model_dump(mode="json")
    )

    assert len(diagnostics) == 1
    assert diagnostics[0][2] == ["BTC-USDT"]
    assert diagnostics[0][3] == "stale_candles"


def test_evaluations_exposes_market_data_diagnostic_without_breaking_success_shape():
    checked_at = datetime.now(timezone.utc).isoformat()
    diagnostic = SimpleNamespace(
        evaluation_id="diagnostic-a",
        created_at=datetime.now(timezone.utc),
        result={
            "stage": "market_data",
            "status": "blocked",
            "action": "no_op",
            "symbol": "BTC-USDT",
            "reason_code": "stale_candles",
            "reason": "行情数据已过期，已安全跳过本次评估。",
            "checked_at": checked_at,
            "next_check_at": checked_at,
        },
        trades=[],
    )
    strategy = SimpleNamespace(strategy_id="rule-a", tenant_id="tenant-a")

    class Repository:
        def get(self, strategy_id, tenant_id):
            return (
                strategy
                if (strategy_id, tenant_id) == ("rule-a", "tenant-a")
                else None
            )

        def get_evaluations(self, *_args, **_kwargs):
            return [diagnostic]

    entry = strategy_scheduler.RuleStrategyService(
        repository=Repository()
    ).evaluations("rule-a", "tenant-a", 10)[0]
    assert entry["stage"] == "market_data"
    assert entry["status"] == "blocked"
    assert entry["checked_at"] == checked_at
    assert entry["next_check_at"] == checked_at


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


def _advanced_exit_config():
    return RuleStrategyConfig.model_validate(
        {
            "symbols": ["BTC-USDT"],
            "interval": "5m",
            "advanced_rules": {
                "enabled": True,
                "rsi": {
                    "enabled": True,
                    "interval": "5m",
                    "period": 2,
                    "entry_comparator": "below",
                    "entry_threshold": 10,
                    "exit_enabled": True,
                    "exit_comparator": "above",
                    "exit_threshold": 70,
                },
            },
            "execution": {
                "environment": "okx_demo",
                "sandbox_connection_id": "okx-demo-connection",
            },
        }
    )


def _rising_candles():
    return [
        RuleStrategyCandle(
            timestamp_ms=index + 1,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=1,
        )
        for index, price in enumerate((100, 110, 120))
    ]


@pytest.mark.parametrize(
    ("quantity", "expected_action"), [("0.25", "sell"), ("0", "no_op")]
)
def test_demo_batch_uses_supplied_demo_position_for_advanced_exit_without_paper_ledger(
    quantity, expected_action
):
    config = _advanced_exit_config()
    strategy = SimpleNamespace(
        strategy_id="rule-demo-account",
        tenant_id="tenant-a",
        status="running",
        config=config.model_dump(mode="json"),
    )

    class Repository:
        def __init__(self):
            self.journals = []

        def get(self, *_args):
            return strategy

        def get_evaluations(self, *_args, **_kwargs):
            raise AssertionError("Demo evaluation must not read paper account history")

        def append_evaluation(self, journal):
            assert journal.trades == []
            self.journals.append(journal)
            return journal

    repository = Repository()
    result = strategy_scheduler.RuleStrategyService(repository=repository).evaluate_batch(
        "rule-demo-account",
        "tenant-a",
        [
            (
                {"5m": _rising_candles()},
                RuleStrategyEngineMarketSnapshot(
                    symbol="BTC-USDT",
                    price=120,
                    quote_balance=321,
                    equity_quote=351,
                    open_position_count=1 if Decimal(quantity) > 0 else 0,
                    position=RuleStrategyPosition(
                        quantity=float(quantity),
                        # Demo balance has no acquisition-cost basis. Missing is
                        # honest and must only disable P&L exits, not indicator exits.
                        entry_price=None,
                    ),
                ),
            )
        ],
    )[0]
    assert result["action"] == expected_action
    assert result["account"]["quote_balance"] == 321
    assert repository.journals[0].trades == []


@pytest.mark.asyncio
async def test_demo_account_sync_failure_blocks_tick_without_evaluation_or_raw_error(
    monkeypatch,
):
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    config = _advanced_exit_config()
    session = session_factory()
    session.add(
        RuleStrategy(
            strategy_id="rule-demo-account",
            tenant_id="tenant-a",
            name="demo account",
            status="running",
            config=config.model_dump(mode="json"),
        )
    )
    session.commit()
    session.close()

    class FailingDemoService:
        def __init__(self, _session):
            pass

        async def balance(self, tenant_id, credential_id):
            assert (tenant_id, credential_id) == (
                "tenant-a",
                "okx-demo-connection",
            )
            raise RuntimeError("secret exchange payload")

    class MarketService:
        async def get_indicators(self, **_kwargs):
            raise AssertionError("account sync must block before evaluation market I/O")

    monkeypatch.setattr(
        strategy_scheduler,
        "get_database_manager",
        lambda: SimpleNamespace(get_session=session_factory),
    )
    monkeypatch.setattr(
        strategy_scheduler.TenantAccessService,
        "access_for",
        lambda *_args: SimpleNamespace(active=True),
    )
    monkeypatch.setattr(
        strategy_scheduler, "SandboxExchangeTradingService", FailingDemoService
    )
    monkeypatch.setattr(strategy_scheduler, "get_crypto_market_service", MarketService)
    await strategy_scheduler.StrategyScheduler()._tick(
        "rule-demo-account", "tenant-a", config.model_dump(mode="json")
    )

    verify = session_factory()
    try:
        diagnostic = verify.query(RuleStrategyEvaluationJournal).one().result
        assert diagnostic["stage"] == "account_sync"
        assert diagnostic["status"] == "blocked"
        assert diagnostic["reason_code"] == "demo_account_unavailable"
        assert diagnostic["action"] == "no_op"
        assert "secret exchange payload" not in str(diagnostic)
    finally:
        verify.close()
        Base.metadata.drop_all(engine)
        engine.dispose()
