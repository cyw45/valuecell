from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from valuecell.core.coordinate.orchestrator import AgentOrchestrator
from valuecell.server.api.routers.strategy_agent import create_strategy_agent_router
from valuecell.server.db.connection import get_db


def test_strategy_creation_rejects_live_mode_before_orchestration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator_calls: list[object] = []

    async def fail_if_orchestrated(*args: object, **kwargs: object):
        orchestrator_calls.append((args, kwargs))
        raise AssertionError("Live strategy creation must not reach the orchestrator")
        yield

    monkeypatch.setattr(AgentOrchestrator, "process_user_input", fail_if_orchestrated)

    app = FastAPI()
    app.include_router(create_strategy_agent_router())
    app.dependency_overrides[get_db] = lambda: None

    response = TestClient(app).post(
        "/strategies/create",
        json={
            "llm_model_config": {"provider": "test", "model_id": "test-model"},
            "exchange_config": {
                "exchange_id": "test-exchange",
                "trading_mode": "live",
                "market_type": "spot",
            },
            "trading_config": {"symbols": ["BTC-USDT"], "max_leverage": 1},
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "This deployment supports paper trading only."
    assert orchestrator_calls == []
