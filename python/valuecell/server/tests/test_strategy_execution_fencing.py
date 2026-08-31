from decimal import Decimal
from types import SimpleNamespace

from uuid import uuid4

import pytest

from valuecell.server.api.schemas.rule_strategy import RuleStrategyConfig
from valuecell.server.services import strategy_scheduler
from valuecell.server.services.sandbox_exchange_trading_service import (
    INTENT_SUBMISSION_UNKNOWN,
    SandboxExchangeTradingService,
)


def test_strategy_client_order_id_fits_okx_limit():
    key = strategy_scheduler._strategy_client_order_id(
        "rule-a", 1234, "BTC-USDT", "buy"
    )
    assert key.startswith("vcdemo")
    assert key.isalnum()
    assert len(key) <= 32


def test_strategy_position_ignores_shared_account_dust_after_confirmed_exit():
    inventory = {
        "CRV/USDT": (Decimal("0.361011"), Decimal("0.1001083503")),
        "TRX/USDT": (Decimal("250"), Decimal("75")),
    }

    quantity, cost, open_count = strategy_scheduler._strategy_position_from_inventory(
        inventory, "CRV-USDT", Decimal("0.00000017")
    )

    assert quantity == 0
    assert cost == 0
    assert open_count == 1


def test_strategy_position_uses_strategy_confirmed_fill_inventory():
    inventory = {"CRV/USDT": (Decimal("360.649819"), Decimal("100.1083"))}

    quantity, cost, open_count = strategy_scheduler._strategy_position_from_inventory(
        inventory, "crv-usdt"
    )

    assert quantity == Decimal("360.649819")
    assert cost == Decimal("100.1083")
    assert open_count == 1


def test_okx_parameter_rejection_is_deterministic_failure():
    assert SandboxExchangeTradingService._is_deterministic_order_rejection(
        'okx {"code":"1","data":[{"sCode":"51000","sMsg":"Parameter clOrdId error"}]}'
    )


@pytest.mark.asyncio
async def test_okx_demo_signal_without_durable_evaluation_is_blocked_without_submit(monkeypatch):
    submitted = False

    class FakeService:
        def __init__(self, _session):
            pass

        async def submit_order(self, *_args, **_kwargs):
            nonlocal submitted
            submitted = True
            return {"id": "demo-order", "status": "open", "sandbox": True}

    monkeypatch.setattr(strategy_scheduler, "SandboxExchangeTradingService", FakeService)
    config = RuleStrategyConfig.model_validate({
        "symbols": ["BTC-USDT"],
        "execution": {"environment": "okx_demo", "sandbox_connection_id": "okx-demo-connection"},
    })
    result = await strategy_scheduler.StrategyScheduler._execute_signal(
        "tenant-a", "rule-a", config, "BTC-USDT", "buy", Decimal("100"), Decimal("50000"), 1234
    )
    assert result["execution"] == "blocked"
    assert result["reason"] == "durable evaluation is required for strategy execution"
    assert submitted is False


def test_closed_exchange_lifecycle_is_normalized_to_filled():
    assert SandboxExchangeTradingService._normalise_status("closed") == "filled"


@pytest.mark.asyncio
async def test_unknown_intent_is_not_resubmitted_and_reconciliation_never_creates(monkeypatch):
    intent = SimpleNamespace(
        id="intent-a", tenant_id="tenant-a", credential_id="credential-a",
        idempotency_key="client-a", status=INTENT_SUBMISSION_UNKNOWN,
        attempt_count=1, error_code="sandbox_submission_unknown", strategy_id="rule-a", evaluation_id="eval-a", execution_generation=1, symbol="BTC/USDT",
    )
    service = object.__new__(SandboxExchangeTradingService)
    service.db = SimpleNamespace(query=lambda _model: SimpleNamespace(filter_by=lambda **_kwargs: SimpleNamespace(first=lambda: None)))
    result = await service.submit_order(
        "tenant-a", "credential-a", "ignored", "BTC/USDT", "buy", "market", Decimal("100"), None,
        intent=intent,
    )
    assert result["status"] == INTENT_SUBMISSION_UNKNOWN
    assert result["attempt_count"] == 1


    class ReconcileQuery:
        def filter_by(self, **_kwargs):
            return self
        def all(self):
            return [intent]
        def with_for_update(self):
            return self
        def first(self):
            return SimpleNamespace(status="running", execution_generation=1)

    class Exchange:
        def set_sandbox_mode(self, _enabled):
            pass

        async def close(self):
            pass

    db = SimpleNamespace(query=lambda _model: ReconcileQuery(), commit=lambda: None)
    reconcile = object.__new__(SandboxExchangeTradingService)
    reconcile.db = db
    monkeypatch.setattr(reconcile, "_active_sandbox_credential", lambda *_args: SimpleNamespace(provider="okx"))
    monkeypatch.setattr(reconcile, "_exchange_for", lambda *_args: Exchange())
    async def no_exchange_order(*_args):
        return None

    monkeypatch.setattr(reconcile, "_find_exchange_order_by_client_id", no_exchange_order)
    await reconcile.reconcile_nonterminal_intents("tenant-a")
    assert intent.status == INTENT_SUBMISSION_UNKNOWN
    assert intent.error_code == "reconciliation_required"

class _DemoExecutionQuery:
    def __init__(self, session, model):
        self.session = session
        self.model = model
        self.filters = {}

    def filter_by(self, **kwargs):
        self.filters.update(kwargs)
        return self

    def with_for_update(self):
        return self

    def first(self):
        if self.model is strategy_scheduler.RuleStrategy:
            return self.session.strategy
        if self.model is strategy_scheduler.StrategySharedAccount:
            return self.session.account
        return next(
            (
                intent
                for intent in self.session.intents
                if all(getattr(intent, key, None) == value for key, value in self.filters.items())
            ),
            None,
        )

    def all(self):
        return list(self.session.intents)


class _DemoExecutionSession:
    def __init__(self, strategy_id: str, config: RuleStrategyConfig):
        self.strategy = SimpleNamespace(
            strategy_id=strategy_id,
            tenant_id="tenant-a",
            status="running",
            execution_generation=1,
            current_batch_id=None,
            config=config.model_dump(mode="json"),
        )
        self.account = SimpleNamespace(
            id="shared-account",
            sync_status="healthy",
            active=True,
        )
        self.intents = []

    def query(self, model):
        return _DemoExecutionQuery(self, model)

    def add(self, intent):
        self.intents.append(intent)

    def flush(self):
        for intent in self.intents:
            if intent.id is None:
                intent.id = str(uuid4())

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


@pytest.mark.asyncio
async def test_okx_demo_shared_wallet_blocks_second_strategy_when_reservation_exhausts(monkeypatch):
    config = RuleStrategyConfig.model_validate(
        {
            "symbols": ["BTC-USDT"],
            "execution": {
                "environment": "okx_demo",
                "sandbox_connection_id": "credential-a",
                "max_order_quote_amount": 100,
                "max_daily_quote_amount": 1_000,
            },
        }
    )
    sessions = [
        _DemoExecutionSession("strategy-a", config),
        _DemoExecutionSession("strategy-b", config),
    ]
    submitted = []

    class Allocator:
        reserved = Decimal(0)

        def __init__(self, _session):
            pass

        def reserve(self, *, requested_quote, **_kwargs):
            if self.reserved + requested_quote > Decimal(100):
                raise strategy_scheduler.CapitalAllocationError("shared account has insufficient unreserved quote")
            self.__class__.reserved += requested_quote
            return SimpleNamespace(reservation_id=f"reservation-{self.reserved}")

        def bind_intent(self, *_args, **_kwargs):
            pass

        def settle(self, *_args, **_kwargs):
            pass

    class Service:
        def __init__(self, _session):
            pass

        async def submit_order(self, *_args, **_kwargs):
            submitted.append(True)
            return {"id": "order", "status": "open"}

    monkeypatch.setattr(strategy_scheduler, "SharedCapitalAllocator", Allocator)
    monkeypatch.setattr(strategy_scheduler, "SandboxExchangeTradingService", Service)
    monkeypatch.setattr(
        strategy_scheduler,
        "get_database_manager",
        lambda: SimpleNamespace(get_session=lambda: sessions.pop(0)),
    )

    first = await strategy_scheduler.StrategyScheduler._execute_okx_demo_signal(
        "tenant-a", "strategy-a", config, "BTC-USDT", "buy", Decimal(100), Decimal(50_000), 1, "eval-a"
    )
    second = await strategy_scheduler.StrategyScheduler._execute_okx_demo_signal(
        "tenant-a", "strategy-b", config, "BTC-USDT", "buy", Decimal(100), Decimal(50_000), 2, "eval-b"
    )

    assert first["execution"] == "okx_demo_submitted"
    assert second["execution"] == "blocked"
    assert "insufficient unreserved" in second["reason"]
    assert len(submitted) == 1


@pytest.mark.asyncio
async def test_okx_demo_sell_requires_confirmed_strategy_inventory(monkeypatch):
    config = RuleStrategyConfig.model_validate(
        {
            "symbols": ["BTC-USDT"],
            "execution": {"environment": "okx_demo", "sandbox_connection_id": "credential-a"},
        }
    )
    session = _DemoExecutionSession("strategy-a", config)
    submitted = False

    class Service:
        def __init__(self, _session):
            pass

        def list_orders(self, *_args):
            return []

        async def submit_order(self, *_args, **_kwargs):
            nonlocal submitted
            submitted = True
            return {"id": "order", "status": "open"}

    monkeypatch.setattr(strategy_scheduler, "SandboxExchangeTradingService", Service)
    monkeypatch.setattr(
        strategy_scheduler,
        "get_database_manager",
        lambda: SimpleNamespace(get_session=lambda: session),
    )

    result = await strategy_scheduler.StrategyScheduler._execute_okx_demo_signal(
        "tenant-a", "strategy-a", config, "BTC-USDT", "sell", Decimal(100), Decimal(50_000), 1, "eval-a"
    )

    assert result["execution"] == "blocked"
    assert "confirmed inventory" in result["reason"]
    assert submitted is False
