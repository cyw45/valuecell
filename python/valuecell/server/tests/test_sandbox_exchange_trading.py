import base64
from collections.abc import Generator
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import valuecell.server.services.sandbox_exchange_service as validation_module
import valuecell.server.services.sandbox_exchange_trading_service as trading_module
import valuecell.server.services.tenant_credential_service as credential_module
from valuecell.server.api.auth import CurrentPrincipal, get_current_principal
from valuecell.server.api.routers.rule_strategy import create_rule_strategy_router
from valuecell.server.api.routers.sandbox_exchange import create_sandbox_exchange_router
from valuecell.server.db.connection import get_db
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.rule_strategy import (
    RuleStrategy,
    RuleStrategyEvaluationJournal,
    RuleStrategyExecutionIntent,
)
from valuecell.server.db.models.sandbox_exchange_order import SandboxExchangeOrder
from valuecell.server.db.models.tenant_credential import TenantCredential
from valuecell.server.db.repositories.rule_strategy_repository import RuleStrategyRepository
from valuecell.server.services.rule_strategy_service import RuleStrategyService

TEST_MASTER_KEY = base64.urlsafe_b64encode(b"0123456789abcdef0123456789abcdef").decode("ascii")


@dataclass(frozen=True)
class SettingsFixture:
    CREDENTIAL_MASTER_KEY: str = TEST_MASTER_KEY


def fake_exchange_class():
    class FakeExchange:
        instances: list["FakeExchange"] = []

        def __init__(self, config: dict):
            self.config = config
            self.calls: list[str] = []
            self.__class__.instances.append(self)

        def set_sandbox_mode(self, enabled: bool) -> None:
            assert enabled is True
            self.calls.append("sandbox")

        def _private(self, call: str) -> None:
            assert self.calls and self.calls[-1] == "sandbox"
            self.calls.append(call)

        async def fetch_balance(self) -> dict:
            self._private("fetch_balance")
            return {"total": {"USDT": 250, "BTC": 0}, "free": {"USDT": 200}, "used": {"USDT": 50}}

        async def load_markets(self) -> dict:
            return {
                "BTC/USDT": {
                    "spot": True,
                    "active": True,
                    "base": "BTC",
                    "quote": "USDT",
                    "precision": {"amount": 6},
                    "limits": {"amount": {"min": 0.0001}, "cost": {"min": 10}},
                }
            }

        def amount_to_precision(self, _symbol: str, amount: float) -> str:
            return f"{amount:.6f}"

        async def fetch_ticker(self, symbol: str) -> dict:
            assert symbol == "BTC/USDT"
            return {"last": 50000}

        async def fetch_order(self, order_id: str, symbol: str) -> dict:
            self._private("fetch_order")
            assert order_id == "test-order-1"
            assert symbol == "BTC/USDT"
            return {
                "id": order_id,
                "status": "closed",
                "symbol": symbol,
                "side": "buy",
                "type": "market",
                "amount": 0.002,
                "filled": 0.002,
                "remaining": 0,
                "cost": 100,
            }

        async def create_order(self, symbol: str, type_: str, side: str, amount: float, price, params: dict) -> dict:
            self._private("create_order")
            return {"id": "test-order-1", "status": "open", "symbol": symbol, "side": side, "type": type_, "amount": amount, "info": {"secret": "never-return"}}

        async def close(self) -> None:
            self.calls.append("close")

    return FakeExchange


@pytest.fixture
def sandbox_client(monkeypatch: pytest.MonkeyPatch) -> Generator[tuple[TestClient, list[CurrentPrincipal], type], None, None]:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    principal = [CurrentPrincipal(user_id="user-a", tenant_id="tenant-a")]
    fake_exchange = fake_exchange_class()
    monkeypatch.setattr(credential_module, "get_settings", lambda: SettingsFixture())
    monkeypatch.setattr(validation_module.ccxtpro, "binance", fake_exchange)
    monkeypatch.setattr(validation_module.ccxtpro, "okx", fake_exchange)
    monkeypatch.setattr(trading_module.ccxtpro, "binance", fake_exchange)
    monkeypatch.setattr(trading_module.ccxtpro, "okx", fake_exchange)

    def override_db() -> Generator[Session, None, None]:
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app = FastAPI()
    app.include_router(create_sandbox_exchange_router())
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_principal] = lambda: principal[0]
    try:
        with TestClient(app) as client:
            yield client, principal, fake_exchange
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def connection_request(**overrides: str) -> dict:
    request = {"provider": "binance", "api_key": "testnet-api-key", "api_secret": "testnet-api-secret", "label": "desk-a"}
    request.update(overrides)
    return request


def create_connection(client: TestClient) -> str:
    response = client.post("/saas/sandbox-exchanges/connections", json=connection_request())
    assert response.status_code == 201, response.text
    return response.json()["data"]["id"]


def test_validated_connection_is_encrypted_and_metadata_only(sandbox_client):
    client, _, _ = sandbox_client
    credential_id = create_connection(client)
    body = client.get("/saas/sandbox-exchanges/connections").json()
    assert body["data"] == [{"id": credential_id, "provider": "binance", "label": "desk-a", "metadata": {"sandbox": True, "provider": "binance", "market_type": "spot", "validated_at": body["data"][0]["metadata"]["validated_at"]}, "created_at": body["data"][0]["created_at"]}]
    assert "testnet-api-key" not in str(body)


def test_connection_failure_returns_only_safe_error_codes(sandbox_client, monkeypatch: pytest.MonkeyPatch):
    client, _, _ = sandbox_client

    async def invalid_credentials(self) -> dict:
        self._private("fetch_balance")
        raise RuntimeError("invalid sandbox credentials")

    monkeypatch.setattr(validation_module.ccxtpro.binance, "fetch_balance", invalid_credentials)
    request = connection_request(
        api_key="connection-failure-key",
        api_secret="connection-failure-secret",
    )

    response = client.post("/saas/sandbox-exchanges/connections", json=request)

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "sandbox_validation_failed",
        "error_code": "credential_or_permission_error",
    }
    assert "connection-failure-key" not in response.text
    assert "connection-failure-secret" not in response.text


def test_balance_is_sanitized_and_private_calls_use_sandbox(sandbox_client):
    client, _, fake_exchange = sandbox_client
    credential_id = create_connection(client)
    response = client.get(f"/saas/sandbox-exchanges/connections/{credential_id}/balance")
    assert response.status_code == 200
    assert response.json()["data"]["balances"] == [
        {
            "currency": "USDT",
            "total": 250,
            "free": 200,
            "used": 50,
            "frozen": 50,
            "mark_price_usdt": 1,
            "usdt_value": 250,
            "valuation_status": "priced",
        }
    ]
    assert fake_exchange.instances[-1].calls == ["sandbox", "fetch_balance", "close"]


def test_demo_balance_is_valued_and_exposes_spot_positions(sandbox_client, monkeypatch: pytest.MonkeyPatch):
    client, _, fake_exchange = sandbox_client
    response = client.post(
        "/saas/sandbox-exchanges/connections",
        json=connection_request(provider="okx", passphrase="test-passphrase", label="okx-demo"),
    )
    assert response.status_code == 201, response.text
    credential_id = response.json()["data"]["id"]

    async def btc_balance(self) -> dict:
        self._private("fetch_balance")
        return {
            "total": {"USDT": 25, "BTC": 0.01},
            "free": {"USDT": 20, "BTC": 0.008},
            "used": {"USDT": 5, "BTC": 0.002},
        }

    monkeypatch.setattr(fake_exchange, "fetch_balance", btc_balance)
    response = client.get(f"/saas/sandbox-exchanges/connections/{credential_id}/balance")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["source"] == "okx_demo"
    assert data["total_usdt_value"] == 525
    assert {item["currency"]: item for item in data["balances"]}["BTC"]["usdt_value"] == 500
    positions = client.get(
        f"/saas/sandbox-exchanges/connections/{credential_id}/positions"
    )
    assert positions.status_code == 200
    assert positions.json()["data"]["positions"] == [
        {
            "symbol": "BTC/USDT",
            "base_currency": "BTC",
            "quantity": 0.01,
            "available_quantity": 0.008,
            "frozen_quantity": 0.002,
            "mark_price": 50000,
            "notional_usdt": 500,
            "unrealized_pnl_usdt": None,
        }
    ]
    assert fake_exchange.instances[-1].calls == ["sandbox", "fetch_balance", "close"]


def test_order_is_idempotent_and_sanitizes_exchange_payload(sandbox_client):
    client, _, fake_exchange = sandbox_client
    credential_id = create_connection(client)
    request = {"credential_id": credential_id, "symbol": "BTC/USDT", "side": "buy", "type": "market", "quote_amount": "100", "idempotency_key": "order-key-0123456789", "sandbox": True}
    first = client.post("/saas/sandbox-exchanges/orders", json=request, headers={"Idempotency-Key": request["idempotency_key"]})
    second = client.post("/saas/sandbox-exchanges/orders", json=request)
    assert first.status_code == second.status_code == 201
    assert first.json()["data"]["exchange_order_id"] == "test-order-1"
    assert second.json()["data"]["id"] == first.json()["data"]["id"]
    assert "secret" not in str(first.json())
    assert fake_exchange.instances[-1].calls == [
        "sandbox",
        "fetch_balance",
        "sandbox",
        "create_order",
        "close",
    ]


@pytest.mark.parametrize(
    ("available_base", "expected_quantity"),
    [
        # A position smaller than the nominal request sells only what is really free.
        ("0.0012345", "0.001234"),
        # A larger shared position is capped by requested_quote / market price.
        ("0.0081234", "0.002000"),
    ],
)
def test_okx_demo_market_sell_is_bounded_by_available_base_and_requested_quote(
    sandbox_client, monkeypatch, available_base, expected_quantity
):
    client, _, fake_exchange = sandbox_client
    response = client.post(
        "/saas/sandbox-exchanges/connections",
        json=connection_request(provider="okx", passphrase="test-passphrase"),
    )
    credential_id = response.json()["data"]["id"]

    async def held_btc(self):
        self._private("fetch_balance")
        available = float(available_base)
        return {
            "total": {"USDT": 0, "BTC": available + 0.0004},
            "free": {"USDT": 0, "BTC": available},
            "used": {"BTC": 0.0004},
        }

    monkeypatch.setattr(fake_exchange, "fetch_balance", held_btc)
    request = {
        "credential_id": credential_id,
        "symbol": " btc-usdt ",
        "side": "sell",
        "type": "market",
        "quote_amount": "100",
        "idempotency_key": f"bounded-sell-{available_base}",
        "sandbox": True,
    }
    result = client.post("/saas/sandbox-exchanges/orders", json=request)

    assert result.status_code == 201, result.text
    assert result.json()["data"]["requested_quote"] == "100"
    assert result.json()["data"]["requested_quantity"] == expected_quantity


def test_list_orders_refreshes_non_terminal_exchange_statuses(sandbox_client):
    client, _, _ = sandbox_client
    credential_id = create_connection(client)
    request = {
        "credential_id": credential_id,
        "symbol": "BTC/USDT",
        "side": "buy",
        "type": "market",
        "quote_amount": "100",
        "idempotency_key": "poll-order-key-012345",
        "sandbox": True,
    }
    created = client.post("/saas/sandbox-exchanges/orders", json=request).json()["data"]
    assert created["status"] == "open"
    orders = client.get(
        f"/saas/sandbox-exchanges/orders?credential_id={credential_id}&refresh=true"
    )
    assert orders.status_code == 200
    assert orders.json()["data"][0]["status"] == "filled"


def test_rejects_unsafe_requests_and_tenant_cross_access(sandbox_client):
    client, principal, _ = sandbox_client
    assert client.post("/saas/sandbox-exchanges/connections", json=connection_request(provider="kraken")).status_code == 422
    assert client.post("/saas/sandbox-exchanges/connections", json=connection_request(provider="okx")).status_code == 422
    assert client.post("/saas/sandbox-exchanges/connections", json={**connection_request(), "private_key": "forbidden"}).status_code == 422
    credential_id = create_connection(client)
    assert client.post("/saas/sandbox-exchanges/orders", json={"credential_id": credential_id, "symbol": "BTC/USDT", "side": "buy", "type": "market", "quote_amount": "10", "sandbox": False, "idempotency_key": "order-key-0123456789"}).status_code == 422
    assert client.post("/saas/sandbox-exchanges/orders", json={"credential_id": credential_id, "symbol": "BTC/USDT", "side": "buy", "type": "market", "quote_amount": "10"}).status_code == 422
    principal[0] = CurrentPrincipal(user_id="user-b", tenant_id="tenant-b")
    assert client.get("/saas/sandbox-exchanges/connections").json()["data"] == []
    assert client.get(f"/saas/sandbox-exchanges/connections/{credential_id}/balance").status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("venue_status", "expected_status"),
    [("partially_filled", "partially_filled"), ("closed", "filled"), ("canceled", "canceled"), ("rejected", "rejected")],
)
async def test_fetch_order_status_syncs_attributed_execution_to_evaluation_api_shape(
    monkeypatch, venue_status, expected_status
):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    monkeypatch.setattr(credential_module, "get_settings", lambda: SettingsFixture())
    session = sessionmaker(bind=engine)()
    try:
        strategy = RuleStrategy(strategy_id="rule-sync", tenant_id="tenant-a", name="sync", status="running", config={})
        journal = RuleStrategyEvaluationJournal(
            evaluation_id="eval-sync", strategy_id="rule-sync", tenant_id="tenant-a",
            result={"action": "buy", "execution": {"execution": "okx_demo_submitted", "execution_ledger": "okx_demo", "paper_fill": False}},
        )
        credential = TenantCredential(
            id="credential-sync", tenant_id="tenant-a", created_by_user_id="user-a", kind="exchange",
            provider="okx", label="sync", encrypted_payload="ciphertext", nonce="nonce",
            metadata_json={"sandbox": True, "market_type": "spot"}, revoked=False,
        )
        session.add_all([strategy, journal, credential])
        session.flush()
        intent = RuleStrategyExecutionIntent(
            id="intent-sync", strategy_id="rule-sync", evaluation_id="eval-sync", execution_generation=1,
            execution_source="rule_strategy", tenant_id="tenant-a", credential_id="credential-sync",
            idempotency_key="sync-client-order", symbol="BTC/USDT", side="buy", order_type="market",
            requested_quote="100", status="open",
        )
        session.add(intent)
        session.flush()
        session.add(SandboxExchangeOrder(
            id="order-sync", tenant_id="tenant-a", credential_id="credential-sync", provider="okx",
            client_order_id="sync-client-order", symbol="BTC/USDT", side="buy", order_type="market",
            requested_quote="100", status="open", exchange_order_id="venue-order",
            strategy_id="rule-sync", evaluation_id="eval-sync", execution_generation=1,
            execution_source="rule_strategy", execution_intent_id="intent-sync",
        ))
        session.commit()

        class Exchange:
            def set_sandbox_mode(self, enabled):
                assert enabled is True

            async def fetch_order(self, order_id, symbol):
                return {"id": order_id, "symbol": symbol, "status": venue_status, "filled": "0.5"}

            async def close(self):
                pass

        service = trading_module.SandboxExchangeTradingService(session)
        monkeypatch.setattr(service, "_active_sandbox_credential", lambda *_args: credential)
        monkeypatch.setattr(service, "_exchange_for", lambda *_args: Exchange())
        await service.fetch_order_status("tenant-a", "order-sync")

        app = FastAPI()
        app.include_router(
            create_rule_strategy_router(
                service=RuleStrategyService(RuleStrategyRepository(session))
            )
        )
        app.dependency_overrides[get_current_principal] = lambda: CurrentPrincipal(
            user_id="user-a", tenant_id="tenant-a"
        )
        response = TestClient(app).get("/rule-strategies/rule-sync/evaluations")
        assert response.status_code == 200
        api_entry = response.json()["data"][0]
        assert api_entry["execution"]["status"] == expected_status
        assert api_entry["execution"]["order_id"] == "order-sync"
        assert api_entry["execution"]["execution_intent_id"] == "intent-sync"
        assert api_entry["execution"]["execution_ledger"] == "okx_demo"
        assert api_entry["execution"]["paper_fill"] is False
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.mark.asyncio
async def test_intent_reconcile_syncs_submission_unknown_without_cross_tenant_journal_write(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(credential_module, "get_settings", lambda: SettingsFixture())
    session = sessionmaker(bind=engine)()
    try:
        session.add_all([
            RuleStrategy(strategy_id="rule-a", tenant_id="tenant-a", name="a", config={}),
            RuleStrategy(strategy_id="rule-b", tenant_id="tenant-b", name="b", config={}),
            RuleStrategyEvaluationJournal(evaluation_id="eval-a", strategy_id="rule-a", tenant_id="tenant-a", result={"execution": {"status": "submitting"}}),
            RuleStrategyEvaluationJournal(evaluation_id="eval-b", strategy_id="rule-b", tenant_id="tenant-b", result={"execution": {"status": "untouched"}}),
            TenantCredential(
                id="credential-a", tenant_id="tenant-a", created_by_user_id="user-a", kind="exchange",
                provider="okx", label="a", encrypted_payload="ciphertext", nonce="nonce",
                metadata_json={"sandbox": True, "market_type": "spot"}, revoked=False,
            ),
        ])
        session.flush()
        session.add(RuleStrategyExecutionIntent(
            id="intent-a", strategy_id="rule-a", evaluation_id="eval-a", execution_generation=1,
            execution_source="rule_strategy", tenant_id="tenant-a", credential_id="credential-a",
            idempotency_key="intent-client-key", symbol="BTC/USDT", side="buy", order_type="market",
            requested_quote="100", status="submitting",
        ))
        session.commit()

        class Exchange:
            def set_sandbox_mode(self, enabled):
                assert enabled is True

            async def close(self):
                pass

        service = trading_module.SandboxExchangeTradingService(session)
        credential = session.query(TenantCredential).filter_by(id="credential-a").one()
        monkeypatch.setattr(service, "_active_sandbox_credential", lambda *_args: credential)
        monkeypatch.setattr(service, "_exchange_for", lambda *_args: Exchange())

        async def not_found(*_args):
            return None

        monkeypatch.setattr(service, "_find_exchange_order_by_client_id", not_found)
        await service.reconcile_nonterminal_intents("tenant-a")

        own = session.query(RuleStrategyEvaluationJournal).filter_by(evaluation_id="eval-a").one()
        other = session.query(RuleStrategyEvaluationJournal).filter_by(evaluation_id="eval-b").one()
        assert own.result["execution"]["status"] == "submission_unknown"
        assert own.result["execution"]["execution_intent_id"] == "intent-a"
        assert other.result["execution"]["status"] == "untouched"
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()
