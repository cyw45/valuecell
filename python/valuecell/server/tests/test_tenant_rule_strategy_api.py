from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from valuecell.server.api.auth import CurrentPrincipal, get_current_principal
from valuecell.server.api.routers.rule_strategy import create_rule_strategy_router
from valuecell.server.services.rule_strategy_service import RuleStrategyService


CREATED_AT = datetime(2026, 7, 10, tzinfo=timezone.utc)


class TenantRuleStrategyRepository:
    """Deterministic in-memory persistence that enforces tenant scope at its boundary."""

    def __init__(self) -> None:
        self.strategies = {}
        self.evaluations = []
        self._strategy_sequence = 0
        self._evaluation_sequence = 0

    def create(self, strategy):
        self._strategy_sequence += 1
        strategy.strategy_id = f"rule_{self._strategy_sequence}"
        strategy.created_at = CREATED_AT
        strategy.updated_at = CREATED_AT
        self.strategies[(strategy.tenant_id, strategy.strategy_id)] = strategy
        return strategy

    def list(self, tenant_id: str):
        return [
            strategy
            for (stored_tenant_id, _), strategy in self.strategies.items()
            if stored_tenant_id == tenant_id
        ]

    def get(self, strategy_id: str, tenant_id: str):
        return self.strategies.get((tenant_id, strategy_id))

    def update(self, strategy):
        strategy.updated_at = CREATED_AT
        self.strategies[(strategy.tenant_id, strategy.strategy_id)] = strategy
        return strategy

    def append_evaluation(self, journal):
        self._evaluation_sequence += 1
        journal.evaluation_id = f"evaluation_{self._evaluation_sequence}"
        journal.created_at = CREATED_AT
        self.evaluations.append(journal)
        return journal

    def get_evaluations(self, strategy_id: str, tenant_id: str, limit: int = 100):
        matching = [
            journal
            for journal in self.evaluations
            if journal.strategy_id == strategy_id and journal.tenant_id == tenant_id
        ]
        return list(reversed(matching[-limit:]))


def _config() -> dict:
    return {
        "mode": "paper",
        "confirmation_mode": "all",
        "rsi": {"enabled": True, "period": 2, "oversold": 30, "overbought": 70},
        "risk": {
            "size_mode": "fixed_quote",
            "size_value": 100,
            "max_positions": 1,
            "leverage": 1,
        },
    }


def _evaluation_input() -> dict:
    return {
        "candles": [
            {
                "timestamp_ms": 1_000,
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100,
                "volume": 10,
            },
            {
                "timestamp_ms": 2_000,
                "open": 100,
                "high": 101,
                "low": 89,
                "close": 90,
                "volume": 10,
            },
            {
                "timestamp_ms": 3_000,
                "open": 90,
                "high": 91,
                "low": 79,
                "close": 80,
                "volume": 10,
            },
        ],
        "market": {
            "symbol": "BTC-USDT",
            "price": 80,
            "equity_quote": 1_000,
            "quote_balance": 1_000,
            "open_position_count": 0,
            "funding_rate": 0.001,
            "position": {"quantity": 0},
        },
    }


def _tenant_client() -> tuple[TestClient, list[CurrentPrincipal]]:
    principal = [CurrentPrincipal(user_id="user-a", tenant_id="tenant-a")]
    app = FastAPI()
    app.include_router(
        create_rule_strategy_router(
            service=RuleStrategyService(repository=TenantRuleStrategyRepository())
        )
    )
    app.dependency_overrides[get_current_principal] = lambda: principal[0]
    return TestClient(app), principal


def _create_strategy(client: TestClient, name: str) -> str:
    response = client.post(
        "/rule-strategies",
        json={"name": name, "config": _config()},
    )
    assert response.status_code == 201
    return response.json()["data"]["strategy_id"]


def test_rule_strategies_derive_tenant_scope_from_principal_and_isolate_records():
    client, principal = _tenant_client()

    rejected_tenant = client.post(
        "/rule-strategies",
        json={"name": "forbidden scope", "tenant_id": "tenant-b", "config": _config()},
    )
    rejected_user = client.post(
        "/rule-strategies",
        json={"name": "forbidden owner", "user_id": "user-b", "config": _config()},
    )
    assert rejected_tenant.status_code == 422
    assert rejected_user.status_code == 422

    strategy_id = _create_strategy(client, "Tenant A strategy")
    assert [item["strategy_id"] for item in client.get("/rule-strategies").json()["data"]] == [
        strategy_id
    ]

    client.post(f"/rule-strategies/{strategy_id}/start")
    exercised = client.post(f"/rule-strategies/{strategy_id}/evaluate", json=_evaluation_input())
    assert exercised.status_code == 200

    principal[0] = CurrentPrincipal(user_id="user-b", tenant_id="tenant-b")
    tenant_b_strategy_id = _create_strategy(client, "Tenant B strategy")
    tenant_b_list = client.get("/rule-strategies")
    assert tenant_b_list.status_code == 200
    assert [item["strategy_id"] for item in tenant_b_list.json()["data"]] == [
        tenant_b_strategy_id
    ]

    denied_responses = [
        client.get(f"/rule-strategies/{strategy_id}"),
        client.patch(f"/rule-strategies/{strategy_id}", json={"name": "not allowed"}),
        client.post(f"/rule-strategies/{strategy_id}/evaluate", json=_evaluation_input()),
        *[
            client.get(f"/rule-strategies/{strategy_id}/{log_type}")
            for log_type in ("signals", "trades", "funding")
        ],
    ]
    assert [response.status_code for response in denied_responses] == [404] * 6

    principal[0] = CurrentPrincipal(user_id="user-a", tenant_id="tenant-a")
    tenant_a_list = client.get("/rule-strategies")
    assert tenant_a_list.status_code == 200
    assert [item["strategy_id"] for item in tenant_a_list.json()["data"]] == [strategy_id]
