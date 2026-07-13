from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient
from valuecell.server.api.auth import CurrentPrincipal, get_current_principal

from valuecell.server.api.routers.rule_strategy import create_rule_strategy_router
from valuecell.server.services.rule_strategy_service import RuleStrategyService


STRATEGY_ID = "rule_deterministic"
EVALUATION_ID = "evaluation_deterministic"
CREATED_AT = datetime(2026, 7, 10, tzinfo=timezone.utc)
FIXED_PRINCIPAL = CurrentPrincipal(user_id="rule-test-user", tenant_id="rule-test-tenant")


class InMemoryRuleStrategyRepository:
    """Small deterministic persistence boundary for the public router contract."""

    def __init__(self) -> None:
        self.strategy = None
        self.evaluations = []

    def create(self, strategy):
        strategy.strategy_id = STRATEGY_ID
        strategy.created_at = CREATED_AT
        strategy.updated_at = CREATED_AT
        self.strategy = strategy
        return strategy

    def get(self, strategy_id: str, tenant_id: str):
        return self.strategy if self.strategy and strategy_id == STRATEGY_ID else None

    def list(self, tenant_id: str):
        return [self.strategy] if self.strategy else []

    def update(self, strategy):
        strategy.updated_at = CREATED_AT
        self.strategy = strategy
        return strategy

    def append_evaluation(self, journal):
        journal.evaluation_id = EVALUATION_ID
        journal.created_at = CREATED_AT
        self.evaluations.append(journal)
        return journal

    def get_evaluations(self, strategy_id: str, tenant_id: str, limit: int = 100):
        if strategy_id != STRATEGY_ID:
            return []
        return list(reversed(self.evaluations[-limit:]))


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
            "funding_rate": 0.001,
        },
    }


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(
        create_rule_strategy_router(
            service=RuleStrategyService(repository=InMemoryRuleStrategyRepository())
        )
    )
    app.dependency_overrides[get_current_principal] = lambda: FIXED_PRINCIPAL
    return TestClient(app)


def test_rule_strategy_api_persists_paper_only_config_and_refuses_live_fields():
    client = _client()

    response = client.post(
        "/rule-strategies",
        json={
            "name": "Oversold recovery",
            "description": "paper only",
            "initial_capital_quote": 1_000,
            "config": _config(),
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["code"] == 0
    data = body["data"]
    assert data["strategy_id"] == STRATEGY_ID
    assert data["name"] == "Oversold recovery"
    assert data["status"] == "stopped"
    assert data["mode"] == "paper"
    assert data["config"]["mode"] == "paper"
    assert data["config"]["rsi"] == {
        "enabled": True,
        "period": 2,
        "oversold": 30.0,
        "overbought": 70.0,
    }
    assert data["config"]["risk"] == {
        "size_mode": "fixed_quote",
        "size_value": 100.0,
        "take_profit_pct": None,
        "stop_loss_pct": None,
        "max_positions": 1,
        "leverage": 1.0,
    }
    assert not {"api_key", "secret_key", "live", "exchange_order_id"} & set(data)

    updated_config = {
        **_config(),
        "rsi": {"enabled": True, "period": 2, "oversold": 25, "overbought": 75},
        "risk": {
            "size_mode": "fixed_quote",
            "size_value": 250,
            "max_positions": 1,
            "leverage": 1,
        },
    }
    updated = client.patch(
        f"/rule-strategies/{STRATEGY_ID}",
        json={"name": "Deeper oversold recovery", "config": updated_config},
    )

    assert updated.status_code == 200
    assert updated.json()["data"]["name"] == "Deeper oversold recovery"
    assert updated.json()["data"]["mode"] == "paper"
    assert updated.json()["data"]["config"]["rsi"]["oversold"] == 25.0
    assert updated.json()["data"]["config"]["risk"]["size_value"] == 250.0

    detail = client.get(f"/rule-strategies/{STRATEGY_ID}")
    assert detail.status_code == 200
    assert detail.json()["data"] == updated.json()["data"]

    rejected = client.post(
        "/rule-strategies",
        json={
            "name": "Must not execute",
            "config": {**_config(), "api_key": "not-allowed"},
        },
    )

    assert rejected.status_code == 422


def test_rule_strategy_api_requires_start_then_explains_and_journals_paper_evaluation():
    client = _client()
    client.post(
        "/rule-strategies",
        json={
            "name": "Oversold recovery",
            "initial_capital_quote": 1_000,
            "config": _config(),
        },
    )

    stopped = client.post(
        f"/rule-strategies/{STRATEGY_ID}/evaluate", json=_evaluation_input()
    )
    assert stopped.status_code == 409
    assert stopped.json()["detail"] == "Rule strategy must be running before evaluation"

    started = client.post(f"/rule-strategies/{STRATEGY_ID}/start")
    assert started.status_code == 200
    assert started.json()["data"]["status"] == "running"

    evaluated = client.post(
        f"/rule-strategies/{STRATEGY_ID}/evaluate", json=_evaluation_input()
    )
    assert evaluated.status_code == 200
    result = evaluated.json()["data"]
    assert result["strategy_id"] == STRATEGY_ID
    assert result["evaluation_id"] == EVALUATION_ID
    assert result["mode"] == "paper"
    assert result["action"] == "buy"
    assert result["reason_code"] == "indicator_buy_confirmed"
    assert result["sizing"] == {
        "mode": "fixed_quote",
        "requested_quote": 100.0,
        "max_allowed_quote": 1_000.0,
        "affordable_quote": 1_000.0,
        "quantity": 1.25,
    }
    assert result["funding"] == {
        "funding_rate": 0.001,
        "current_notional_quote": 0.0,
        "projected_notional_quote": 100.0,
        "estimated_payment_quote": -0.1,
        "direction": "debit",
    }
    assert {
        (condition["code"], condition["state"]) for condition in result["conditions"]
    } >= {
        ("rsi", "triggered"),
        ("max_positions", "not_triggered"),
        ("available_collateral", "not_triggered"),
        ("leverage_limit", "not_triggered"),
    }

    signals = client.get(f"/rule-strategies/{STRATEGY_ID}/signals").json()["data"]
    trades = client.get(f"/rule-strategies/{STRATEGY_ID}/trades").json()["data"]
    funding = client.get(f"/rule-strategies/{STRATEGY_ID}/funding").json()["data"]

    assert signals["strategy_id"] == STRATEGY_ID
    assert signals["mode"] == "paper"
    assert {
        (entry["evaluation_id"], entry["code"], entry["state"])
        for entry in signals["entries"]
    } >= {
        (EVALUATION_ID, "rsi", "triggered"),
        (EVALUATION_ID, "max_positions", "not_triggered"),
        (EVALUATION_ID, "available_collateral", "not_triggered"),
        (EVALUATION_ID, "leverage_limit", "not_triggered"),
    }
    assert {entry["evaluated_at"] for entry in signals["entries"]} == {
        "2026-07-10T00:00:00Z"
    }
    assert trades["entries"] == [
        {
            "evaluation_id": EVALUATION_ID,
            "evaluated_at": "2026-07-10T00:00:00Z",
            "action": "buy",
            "reason_code": "indicator_buy_confirmed",
            "reason": "Buy recommendation: configured indicators confirm a buy signal.",
            "sizing": result["sizing"],
            "execution": "paper_filled",
            "symbol": "BTC-USDT",
            "price": 80.0,
            "quantity": 1.25,
            "quote_amount": 100.0,
            "realized_pnl_quote": 0.0,
        }
    ]
    assert funding["entries"] == [
        {
            "evaluation_id": EVALUATION_ID,
            "evaluated_at": "2026-07-10T00:00:00Z",
            **result["funding"],
        }
    ]

    stopped = client.post(f"/rule-strategies/{STRATEGY_ID}/stop")
    assert stopped.status_code == 200
    assert stopped.json()["data"]["status"] == "stopped"


def test_rule_strategy_api_returns_grouped_durable_evaluation_feedback() -> None:
    client = _client()
    assert (
        client.post(
            "/rule-strategies",
            json={
                "name": "Oversold recovery",
                "initial_capital_quote": 1_000,
                "config": _config(),
            },
        ).status_code
        == 201
    )
    assert client.post(f"/rule-strategies/{STRATEGY_ID}/start").status_code == 200
    evaluated = client.post(
        f"/rule-strategies/{STRATEGY_ID}/evaluate", json=_evaluation_input()
    )
    assert evaluated.status_code == 200
    result = evaluated.json()["data"]

    history = client.get(f"/rule-strategies/{STRATEGY_ID}/evaluations", params={"limit": 1})

    assert history.status_code == 200
    body = history.json()
    assert body["code"] == 0
    assert body["data"] == [
        {
            "strategy_id": STRATEGY_ID,
            "evaluation_id": EVALUATION_ID,
            "evaluated_at": "2026-07-10T00:00:00Z",
            "action": result["action"],
            "reason_code": result["reason_code"],
            "reason": result["reason"],
            "conditions": result["conditions"],
            "indicators": result["indicators"],
            "sizing": result["sizing"],
            "funding": result["funding"],
            "account": result["account"],
            "trades": [
                {
                    "action": result["action"],
                    "reason_code": result["reason_code"],
                    "reason": result["reason"],
                    "sizing": result["sizing"],
                    "execution": "paper_filled",
                    "symbol": "BTC-USDT",
                    "price": 80.0,
                    "quantity": 1.25,
                    "quote_amount": 100.0,
                    "realized_pnl_quote": 0.0,
                }
            ],
        }
    ]
    assert [condition["code"] for condition in body["data"][0]["conditions"]] == [
        condition["code"] for condition in result["conditions"]
    ]
