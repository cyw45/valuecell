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
