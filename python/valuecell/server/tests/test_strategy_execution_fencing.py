from decimal import Decimal
from types import SimpleNamespace

import pytest

from valuecell.server.api.schemas.rule_strategy import RuleStrategyConfig
from valuecell.server.services import strategy_scheduler
from valuecell.server.services.sandbox_exchange_trading_service import (
    INTENT_SUBMISSION_UNKNOWN,
    SandboxExchangeTradingService,
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
