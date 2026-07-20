"""Focused coverage for the periodic ambiguous-submission reconciliation job."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from valuecell.server.services import demo_execution_reconciliation as reconciliation
from valuecell.server.services import sandbox_exchange_trading_service as trading


@pytest.mark.asyncio
async def test_reconcile_active_tenant_intents_only_invokes_active_tenants(monkeypatch):
    session = object()

    class Access:
        def __init__(self, active):
            self.active = active

    monkeypatch.setattr(
        reconciliation.TenantAccessService,
        "access_for",
        lambda _db, tenant_id: Access(tenant_id == "active"),
    )
    reconcile = AsyncMock(return_value=[])

    class TradingService:
        def __init__(self, db):
            assert db is session

        reconcile_nonterminal_intents = reconcile

    await reconciliation.reconcile_active_tenant_intents(
        session,
        ["active", "inactive"],
        TradingService,
    )

    reconcile.assert_awaited_once_with("active")


@pytest.mark.asyncio
async def test_reconciliation_keeps_ambiguous_intent_after_strategy_stops(monkeypatch):
    """Stop fencing prevents new orders but cannot discard a prior ambiguous request."""
    intent = SimpleNamespace(
        id="intent-a", tenant_id="tenant-a", credential_id="conn-a",
        idempotency_key="client-a", symbol="BTC/USDT", status="submission_unknown",
        error_code=None, error_message=None, attempt_count=1,
    )

    class Query:
        def filter_by(self, **kwargs):
            self.kwargs = kwargs
            return self

        def all(self):
            return [intent]

    class Session:
        def __init__(self):
            self.commit_count = 0

        def query(self, _model):
            return Query()

        def commit(self):
            self.commit_count += 1

    session = Session()
    exchange = SimpleNamespace(set_sandbox_mode=lambda _enabled: None)
    monkeypatch.setattr(trading.SandboxExchangeTradingService, "_active_sandbox_credential", lambda *_: SimpleNamespace(provider="okx"))
    monkeypatch.setattr(trading.SandboxExchangeTradingService, "_exchange_for", lambda *_: exchange)
    monkeypatch.setattr(trading.SandboxExchangeTradingService, "_find_exchange_order_by_client_id", AsyncMock(return_value=None))
    monkeypatch.setattr(trading.SandboxExchangeTradingService, "_close", AsyncMock())

    service = object.__new__(trading.SandboxExchangeTradingService)
    service.db = session
    result = await service.reconcile_nonterminal_intents("tenant-a")

    assert result == [{"execution_intent_id": "intent-a", "id": None, "status": "submission_unknown", "error_code": "reconciliation_required", "attempt_count": 1}]
    assert intent.status == "submission_unknown"
    assert session.commit_count == 1
