from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from valuecell.server.api.auth import CurrentPrincipal, get_current_principal
from valuecell.server.api.routers import rule_strategy as router_module
from valuecell.server.api.routers.rule_strategy import create_rule_strategy_router
from valuecell.server.services.rule_strategy_demo_execution_read_model import (
    DemoExecutionReadModelError,
)


class StrategyService:
    def get(self, strategy_id, tenant_id):
        assert (strategy_id, tenant_id) == ("strategy-a", "tenant-a")
        return {
            "strategy_id": "strategy-a",
            "config": {"execution": {"environment": "okx_demo", "sandbox_connection_id": "conn-a"}},
        }


def test_demo_execution_endpoint_is_not_a_paper_account_fallback(monkeypatch):
    app = FastAPI()
    app.include_router(create_rule_strategy_router(service=StrategyService()))
    app.dependency_overrides[get_current_principal] = lambda: CurrentPrincipal(
        user_id="user-a", tenant_id="tenant-a"
    )

    async def blocked(*_args, **_kwargs):
        raise DemoExecutionReadModelError("Strategy is not configured for OKX Demo execution")

    monkeypatch.setattr(router_module, "get_demo_execution_read_model", blocked)
    monkeypatch.setattr(router_module, "SandboxExchangeTradingService", lambda _db: object())
    response = TestClient(app).get("/rule-strategies/strategy-a/demo-execution")

    assert response.status_code == 409
    assert "not configured" in response.json()["detail"]


def test_demo_execution_endpoint_paginates_orders_without_changing_summary(monkeypatch):
    app = FastAPI()
    app.include_router(create_rule_strategy_router(service=StrategyService()))
    app.dependency_overrides[get_current_principal] = lambda: CurrentPrincipal(
        user_id="user-a", tenant_id="tenant-a"
    )

    async def read_model(*_args, **_kwargs):
        return {
            "connection_id": "conn-a",
            "account": {
                "data": {
                    "source": "okx_demo",
                    "total_usdt_value": 1_000.0,
                    "balances": [],
                }
            },
            "positions": {"data": {"source": "okx_demo", "positions": []}},
            "orders": [{"id": f"order-{index}"} for index in range(23)],
            "trade_summary": {"order_count": 23},
        }

    monkeypatch.setattr(router_module, "get_demo_execution_read_model", read_model)
    monkeypatch.setattr(router_module, "record_demo_account_snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        router_module,
        "list_demo_account_snapshots",
        lambda *_args, **_kwargs: [
            SimpleNamespace(
                observed_at=datetime(2026, 8, 6, tzinfo=timezone.utc),
                total_usdt_value=1_000.0,
            ),
            SimpleNamespace(
                observed_at=datetime(2026, 8, 7, tzinfo=timezone.utc),
                total_usdt_value=1_025.0,
            ),
        ],
    )
    monkeypatch.setattr(router_module, "SandboxExchangeTradingService", lambda _db: object())

    response = TestClient(app).get(
        "/rule-strategies/strategy-a/demo-execution?page=2&page_size=10"
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert [order["id"] for order in data["orders"]] == [
        f"order-{index}" for index in range(10, 20)
    ]
    assert data["trade_summary"]["order_count"] == 23
    assert data["pagination"] == {
        "page": 2,
        "page_size": 10,
        "total_items": 23,
        "total_pages": 3,
    }
    assert data["equity_curve"]["points"] == [
        {
            "ts": "2026-08-06T00:00:00Z",
            "cumulative_pnl": 0.0,
            "daily_pnl_quote": 0.0,
            "equity_quote": 1_000.0,
            "action": "wallet_snapshot",
        },
        {
            "ts": "2026-08-07T00:00:00Z",
            "cumulative_pnl": 25.0,
            "daily_pnl_quote": 25.0,
            "equity_quote": 1_025.0,
            "action": "wallet_snapshot",
        },
    ]
