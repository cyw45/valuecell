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
from valuecell.server.api.routers.sandbox_exchange import create_sandbox_exchange_router
from valuecell.server.db.connection import get_db
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.tenant_credential import TenantCredential

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
            return {"total": {"USDT": 25, "BTC": 0}, "free": {"USDT": 20}, "used": {"USDT": 5}}

        async def fetch_ticker(self, symbol: str) -> dict:
            assert symbol == "BTC/USDT"
            return {"last": 50000}

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


def test_balance_is_sanitized_and_private_calls_use_sandbox(sandbox_client):
    client, _, fake_exchange = sandbox_client
    credential_id = create_connection(client)
    response = client.get(f"/saas/sandbox-exchanges/connections/{credential_id}/balance")
    assert response.status_code == 200
    assert response.json()["data"]["balances"] == [{"currency": "USDT", "total": 25, "free": 20, "used": 5}]
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
    assert fake_exchange.instances[-1].calls == ["sandbox", "sandbox", "create_order", "close"]


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
