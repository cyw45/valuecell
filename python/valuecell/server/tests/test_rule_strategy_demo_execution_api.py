from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from valuecell.server.api.auth import CurrentPrincipal, get_current_principal
from valuecell.server.api.routers import rule_strategy as router_module
from valuecell.server.api.routers.rule_strategy import create_rule_strategy_router


class StrategyService:
    def get(self, strategy_id, tenant_id):
        assert (strategy_id, tenant_id) == ("strategy-a", "tenant-a")
        return {
            "strategy_id": "strategy-a",
            "config": {"execution": {"environment": "okx_demo", "sandbox_connection_id": "conn-a"}},
        }


def _app(monkeypatch) -> FastAPI:
    monkeypatch.setattr(router_module, "get_official_test_baseline", lambda *_args, **_kwargs: None)
    app = FastAPI()
    app.include_router(create_rule_strategy_router(service=StrategyService()))
    app.dependency_overrides[get_current_principal] = lambda: CurrentPrincipal(
        user_id="user-a", tenant_id="tenant-a"
    )
    return app


def test_demo_execution_endpoint_reports_pending_snapshot_without_exchange_call(monkeypatch):
    app = _app(monkeypatch)
    monkeypatch.setattr(router_module, "get_latest_demo_account_snapshot", lambda *_args, **_kwargs: None)
    response = TestClient(app).get("/rule-strategies/strategy-a/demo-execution")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "demo_account_snapshot_pending"


def test_demo_execution_endpoint_reads_snapshot_and_local_orders_only(monkeypatch):
    app = _app(monkeypatch)
    snapshot = SimpleNamespace(
        id=1,
        source="okx_demo",
        total_usdt_value=1_000.0,
        balances=[],
        positions=[],
        observed_at=datetime(2026, 8, 7, tzinfo=timezone.utc),
    )
    orders = [
        {"id": f"order-{index}", "strategy_id": "strategy-a", "execution_source": "rule_strategy"}
        for index in range(23)
    ]
    monkeypatch.setattr(router_module, "get_latest_demo_account_snapshot", lambda *_args, **_kwargs: snapshot)
    monkeypatch.setattr(router_module, "get_official_test_baseline", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(router_module, "list_demo_account_snapshots", lambda *_args, **_kwargs: [snapshot])
    monkeypatch.setattr(router_module, "get_demo_account_sync_state", lambda *_args, **_kwargs: None)

    class LocalOrders:
        def __init__(self, _db):
            pass

        def list_orders(self, *_args, **_kwargs):
            return orders

    monkeypatch.setattr(router_module, "SandboxExchangeTradingService", LocalOrders)
    response = TestClient(app).get(
        "/rule-strategies/strategy-a/demo-execution?page=2&page_size=10"
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert [order["id"] for order in data["orders"]] == [
        f"order-{index}" for index in range(10, 20)
    ]
    assert data["pagination"] == {
        "page": 2,
        "page_size": 10,
        "total_items": 23,
        "total_pages": 3,
    }
    assert data["sync"]["observed_at"] == "2026-08-07T00:00:00+00:00"
