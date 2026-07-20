import base64
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import valuecell.server.api.routers.live_execution as live_router_module
import valuecell.server.services.live_execution_service as live_service_module
import valuecell.server.services.tenant_credential_service as credential_module
from valuecell.server.api.auth import CurrentPrincipal, get_current_principal
from valuecell.server.api.routers.live_execution import create_live_execution_router
from valuecell.server.api.routers.saas_admin import create_saas_admin_router
from valuecell.server.db.connection import get_db
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.rule_strategy import RuleStrategy
from valuecell.server.db.models.saas_control import ServicePlan, TenantSubscription
from valuecell.server.services.live_execution_authorization import (
    LiveAuthorizationManager,
)

TEST_MASTER_KEY = base64.urlsafe_b64encode(b"0123456789abcdef0123456789abcdef").decode(
    "ascii"
)
TEST_API_KEY = "live-test-api-key"
TEST_API_SECRET = "live-test-api-secret"


@dataclass(frozen=True)
class LiveSettingsFixture:
    CREDENTIAL_MASTER_KEY: str = TEST_MASTER_KEY
    LIVE_TRADING_ENABLED: bool = False
    LIVE_AUTHORIZATION_TTL_S: int = 900


def fake_exchange_class():
    class FakeExchange:
        instances: list["FakeExchange"] = []

        def __init__(self, config: dict) -> None:
            self.config = config
            self.calls: list[str] = []
            self.__class__.instances.append(self)

        async def fetch_balance(self) -> dict:
            self.calls.append("fetch_balance")
            return {"free": {"USDT": 1_000}}

        async def fetch_positions(self, symbols: list[str] | None = None) -> list[dict]:
            self.calls.append("fetch_positions")
            return []

        async def fetch_ticker(self, symbol: str) -> dict:
            assert symbol == "BTC/USDT"
            self.calls.append("fetch_ticker")
            return {"last": 50_000}

        async def create_order(
            self,
            symbol: str,
            order_type: str,
            side: str,
            amount: float,
            price: float | None,
            params: dict,
        ) -> dict:
            assert symbol == "BTC/USDT"
            assert order_type == "market"
            assert side == "buy"
            assert amount == 0.002
            assert price is None
            assert params["clientOrderId"]
            self.calls.append("create_order")
            return {
                "id": "exchange-order-1",
                "status": "open",
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "amount": amount,
                "info": {"apiKey": TEST_API_KEY, "secret": TEST_API_SECRET},
            }

        async def close(self) -> None:
            self.calls.append("close")

    return FakeExchange


@pytest.fixture
def live_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[TestClient, list[CurrentPrincipal], list[bool], type], None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    setup_session = sessionmaker(bind=engine)()
    try:
        plan = ServicePlan(
            code="live-test",
            name="Live test",
            commercial_model="subscription",
            duration_days=30,
            price_cents=0,
        )
        setup_session.add(plan)
        setup_session.flush()
        setup_session.add(
            TenantSubscription(
                tenant_id="tenant-a",
                plan_id=plan.id,
                starts_at=datetime.now(timezone.utc) - timedelta(minutes=1),
                ends_at=datetime.now(timezone.utc) + timedelta(days=1),
                granted_by_user_id="user-a",
            )
        )
        setup_session.commit()
    finally:
        setup_session.close()
    principal = [CurrentPrincipal(user_id="user-a", tenant_id="tenant-a")]
    live_enabled = [False]
    fake_exchange = fake_exchange_class()
    authorization_manager = LiveAuthorizationManager()

    def settings() -> LiveSettingsFixture:
        return LiveSettingsFixture(LIVE_TRADING_ENABLED=live_enabled[0])

    monkeypatch.setattr(credential_module, "get_settings", settings)
    monkeypatch.setattr(live_service_module, "get_settings", settings)
    monkeypatch.setattr(live_router_module, "get_settings", settings)
    monkeypatch.setattr(
        live_service_module, "live_authorization_manager", authorization_manager
    )
    monkeypatch.setattr(
        live_router_module, "live_authorization_manager", authorization_manager
    )
    monkeypatch.setattr(live_service_module.ccxtpro, "binance", fake_exchange)
    monkeypatch.setattr(live_service_module.ccxtpro, "okx", fake_exchange)

    def override_db() -> Generator[Session, None, None]:
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app = FastAPI()
    app.include_router(create_live_execution_router(), prefix="/api/v1")
    app.include_router(create_saas_admin_router(), prefix="/api/v1")
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_principal] = lambda: principal[0]

    try:
        with TestClient(app) as client:
            yield client, principal, live_enabled, fake_exchange
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def connection_request() -> dict:
    return {
        "label": "primary-live-desk",
        "provider": "binance",
        "market_type": "spot",
        "api_key": TEST_API_KEY,
        "api_secret": TEST_API_SECRET,
        "withdrawal_disabled_confirmed": True,
        "ip_allowlist_confirmed": True,
    }


def create_connection(client: TestClient, label: str = "primary-live-desk") -> str:
    response = client.post(
        "/api/v1/saas/live-execution/connections",
        json={**connection_request(), "label": label},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert TEST_API_KEY not in str(body)
    assert TEST_API_SECRET not in str(body)
    return body["data"]["id"]


def create_policy(client: TestClient) -> None:
    response = client.post(
        "/api/v1/saas/live-execution/risk-policies",
        json={
            "max_order_notional": "250",
            "max_open_positions": 2,
            "max_leverage": "1",
            "allowed_symbols": ["BTC-USDT"],
        },
    )
    assert response.status_code == 201, response.text


def create_binding(client: TestClient, connection_id: str) -> str:
    db = client.app.dependency_overrides[get_db]().__next__()
    try:
        if (
            db.query(RuleStrategy)
            .filter_by(strategy_id="strategy-a", tenant_id="tenant-a")
            .first()
            is None
        ):
            db.add(
                RuleStrategy(
                    strategy_id="strategy-a",
                    tenant_id="tenant-a",
                    name="strategy-a",
                    config={},
                )
            )
            db.commit()
    finally:
        db.close()
    response = client.post(
        "/api/v1/saas/live-execution/bindings",
        json={"strategy_id": "strategy-a", "connection_id": connection_id},
    )
    assert response.status_code == 201, response.text
    return response.json()["data"]["id"]


def authorize(client: TestClient, principal: list[CurrentPrincipal]) -> None:
    requester = principal[0]
    challenge = client.post(
        "/api/v1/saas/live-execution/startup-authorization/challenge"
    )
    assert challenge.status_code == 200, challenge.text
    principal[0] = CurrentPrincipal(
        user_id="qualified-approver",
        tenant_id=requester.tenant_id,
        role="admin",
    )
    try:
        confirmed = client.post(
            "/api/v1/saas/live-execution/startup-authorization/confirm",
            json={"challenge_code": challenge.json()["data"]["challenge_code"]},
        )
        assert confirmed.status_code == 200, confirmed.text
    finally:
        principal[0] = requester


def order_request(
    connection_id: str, idempotency_key: str = "live-order-key-0001"
) -> dict:
    return {
        "connection_id": connection_id,
        "symbol": "BTC/USDT",
        "side": "buy",
        "type": "market",
        "quote_amount": "100",
        "idempotency_key": idempotency_key,
    }


def test_global_live_disable_rejects_before_any_order_exchange_call(live_client):
    client, principal, _, fake_exchange = live_client
    connection_id = create_connection(client)
    create_policy(client)
    create_binding(client, connection_id)
    authorize(client, principal)

    response = client.post(
        "/api/v1/saas/live-execution/orders", json=order_request(connection_id)
    )

    assert response.status_code == 422
    assert [instance.calls for instance in fake_exchange.instances] == [
        ["fetch_balance", "close"]
    ]


def test_authorization_challenge_is_tenant_bound_and_revoke_disables_it(live_client):
    client, principal, _, _ = live_client
    challenge = client.post(
        "/api/v1/saas/live-execution/startup-authorization/challenge"
    )
    assert challenge.status_code == 200
    challenge_code = challenge.json()["data"]["challenge_code"]

    principal[0] = CurrentPrincipal(user_id="user-b", tenant_id="tenant-b")
    wrong_tenant_confirmation = client.post(
        "/api/v1/saas/live-execution/startup-authorization/confirm",
        json={"challenge_code": challenge_code},
    )
    assert wrong_tenant_confirmation.status_code == 422
    assert (
        client.get("/api/v1/saas/live-execution/status").json()["data"][
            "authorization_active"
        ]
        is False
    )

    principal[0] = CurrentPrincipal(
        user_id="user-c", tenant_id="tenant-a", role="owner"
    )
    confirmed = client.post(
        "/api/v1/saas/live-execution/startup-authorization/confirm",
        json={"challenge_code": challenge_code},
    )
    assert confirmed.status_code == 200
    assert (
        client.get("/api/v1/saas/live-execution/status").json()["data"][
            "authorization_active"
        ]
        is True
    )

    revoked = client.post("/api/v1/saas/live-execution/startup-authorization/revoke")
    assert revoked.status_code == 200
    assert revoked.json()["data"] == {"authorization_active": False}
    assert (
        client.get("/api/v1/saas/live-execution/status").json()["data"][
            "authorization_active"
        ]
        is False
    )


def test_binding_requires_tenant_policy_and_cross_tenant_connection_is_hidden(
    live_client,
):
    client, principal, _, _ = live_client
    connection_id = create_connection(client)

    missing_policy = client.post(
        "/api/v1/saas/live-execution/bindings",
        json={"strategy_id": "strategy-a", "connection_id": connection_id},
    )
    assert missing_policy.status_code == 422

    create_policy(client)
    binding_id = create_binding(client, connection_id)

    principal[0] = CurrentPrincipal(user_id="user-b", tenant_id="tenant-b")
    assert client.get("/api/v1/saas/live-execution/connections").json()["data"] == []
    assert (
        client.get("/api/v1/saas/live-execution/risk-policies").json()["data"] is None
    )
    cross_tenant_binding = client.post(
        "/api/v1/saas/live-execution/bindings",
        json={"strategy_id": "strategy-b", "connection_id": connection_id},
    )
    assert cross_tenant_binding.status_code == 422
    assert (
        client.post(
            f"/api/v1/saas/live-execution/bindings/{binding_id}/revoke"
        ).status_code
        == 404
    )


def test_binding_rejects_missing_and_cross_tenant_rule_strategy(live_client):
    client, _, _, _ = live_client
    connection_id = create_connection(client)
    create_policy(client)
    for strategy_id, tenant_id in (("missing", None), ("foreign", "tenant-b")):
        if tenant_id:
            db = client.app.dependency_overrides[get_db]().__next__()
            try:
                db.add(
                    RuleStrategy(
                        strategy_id=strategy_id,
                        tenant_id=tenant_id,
                        name=strategy_id,
                        config={},
                    )
                )
                db.commit()
            finally:
                db.close()
        response = client.post(
            "/api/v1/saas/live-execution/bindings",
            json={"strategy_id": strategy_id, "connection_id": connection_id},
        )
        assert response.status_code == 422


def test_authorization_rejects_self_approval_and_requires_owner_or_admin(live_client):
    client, principal, _, _ = live_client
    challenge = client.post(
        "/api/v1/saas/live-execution/startup-authorization/challenge"
    )
    code = challenge.json()["data"]["challenge_code"]
    assert (
        client.post(
            "/api/v1/saas/live-execution/startup-authorization/confirm",
            json={"challenge_code": code},
        ).status_code
        == 422
    )

    challenge = client.post(
        "/api/v1/saas/live-execution/startup-authorization/challenge"
    )
    # A trader has trade.execute, so this reaches the approver-role gate rather
    # than failing the endpoint's generic permission dependency first.
    principal[0] = CurrentPrincipal(
        user_id="user-b", tenant_id="tenant-a", role="trader"
    )
    assert (
        client.post(
            "/api/v1/saas/live-execution/startup-authorization/confirm",
            json={"challenge_code": challenge.json()["data"]["challenge_code"]},
        ).status_code
        == 403
    )


def test_idempotent_order_returns_original_audit_record_without_second_submission(
    live_client,
):
    client, principal, live_enabled, fake_exchange = live_client
    connection_id = create_connection(client)
    create_policy(client)
    create_binding(client, connection_id)
    live_enabled[0] = True
    authorize(client, principal)
    request = order_request(connection_id)

    first = client.post("/api/v1/saas/live-execution/orders", json=request)
    second = client.post("/api/v1/saas/live-execution/orders", json=request)

    assert first.status_code == second.status_code == 201
    assert first.json()["data"] == second.json()["data"]
    assert first.json()["data"]["exchange_order_id"] == "exchange-order-1"
    assert (
        sum(
            instance.calls.count("create_order") for instance in fake_exchange.instances
        )
        == 1
    )
    serialized_responses = str((first.json(), second.json()))
    assert TEST_API_KEY not in serialized_responses
    assert TEST_API_SECRET not in serialized_responses
    audit = client.get("/api/v1/saas/audit")
    assert audit.status_code == 200
    assert [event["action"] for event in audit.json()["data"]] == [
        "live.order.submitted",
        "live.binding.created",
        "live.connection.created",
    ]


def test_revoked_binding_blocks_execution_before_order_submission(live_client):
    client, principal, live_enabled, fake_exchange = live_client
    connection_id = create_connection(client)
    create_policy(client)
    binding_id = create_binding(client, connection_id)
    live_enabled[0] = True
    authorize(client, principal)
    revoked = client.post(f"/api/v1/saas/live-execution/bindings/{binding_id}/revoke")
    assert revoked.status_code == 200

    rejected = client.post(
        "/api/v1/saas/live-execution/orders", json=order_request(connection_id)
    )

    assert rejected.status_code == 422
    assert (
        sum(
            instance.calls.count("create_order") for instance in fake_exchange.instances
        )
        == 0
    )


@pytest.mark.asyncio
async def test_strategy_signal_without_binding_blocks_without_creating_order(
    live_client,
):
    client, principal, live_enabled, fake_exchange = live_client
    connection_id = create_connection(client)
    create_policy(client)
    live_enabled[0] = True
    authorize(client, principal)
    from decimal import Decimal

    from valuecell.server.services.live_execution_service import LiveExecutionService

    db = client.app.dependency_overrides[get_db]().__next__()
    try:
        result = await LiveExecutionService(db).execute_strategy_signal(
            "tenant-a",
            "strategy-a",
            "BTC/USDT",
            "buy",
            Decimal("100"),
            Decimal("50000"),
            1234,
        )
    finally:
        db.close()
    assert result["execution"] == "blocked"
    assert (
        sum(
            instance.calls.count("create_order") for instance in fake_exchange.instances
        )
        == 0
    )
    assert connection_id


@pytest.mark.asyncio
async def test_strategy_signal_is_idempotent_after_all_live_gates(live_client):
    client, principal, live_enabled, fake_exchange = live_client
    connection_id = create_connection(client)
    create_policy(client)
    create_binding(client, connection_id)
    live_enabled[0] = True
    authorize(client, principal)
    from decimal import Decimal

    from valuecell.server.services.live_execution_service import LiveExecutionService

    db = client.app.dependency_overrides[get_db]().__next__()
    try:
        service = LiveExecutionService(db)
        first = await service.execute_strategy_signal(
            "tenant-a",
            "strategy-a",
            "BTC/USDT",
            "buy",
            Decimal("100"),
            Decimal("50000"),
            1234,
        )
        second = await service.execute_strategy_signal(
            "tenant-a",
            "strategy-a",
            "BTC/USDT",
            "buy",
            Decimal("100"),
            Decimal("50000"),
            1234,
        )
    finally:
        db.close()
    assert first["execution"] == second["execution"] == "live_submitted", (
        first,
        second,
        [item.calls for item in fake_exchange.instances],
    )
    assert first["order_id"] == second["order_id"]
    assert (
        sum(
            instance.calls.count("create_order") for instance in fake_exchange.instances
        )
        == 1
    )


@pytest.mark.asyncio
async def test_strategy_signal_dispatches_to_each_bound_funding_account(live_client):
    client, principal, live_enabled, fake_exchange = live_client
    first_connection = create_connection(client, "primary-live-desk")
    second_connection = create_connection(client, "secondary-live-desk")
    create_policy(client)
    create_binding(client, first_connection)
    create_binding(client, second_connection)
    live_enabled[0] = True
    authorize(client, principal)
    from decimal import Decimal

    from valuecell.server.services.live_execution_service import LiveExecutionService

    db = client.app.dependency_overrides[get_db]().__next__()
    try:
        result = await LiveExecutionService(db).execute_strategy_signal(
            "tenant-a",
            "strategy-a",
            "BTC/USDT",
            "buy",
            Decimal("100"),
            Decimal("50000"),
            5678,
        )
    finally:
        db.close()

    assert result["execution"] == "live_submitted"
    assert {item["connection_id"] for item in result["orders"]} == {
        first_connection,
        second_connection,
    }
    assert (
        sum(item.calls.count("create_order") for item in fake_exchange.instances) == 2
    )
