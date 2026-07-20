from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from valuecell.server.api.auth import CurrentPrincipal, get_current_principal
from valuecell.server.db.connection import get_db

from valuecell.server.api.routers.rule_strategy import create_rule_strategy_router
from valuecell.server.services.rule_strategy_service import RuleStrategyService


STRATEGY_ID = "rule_deterministic"
EVALUATION_ID = "evaluation_deterministic"
CREATED_AT = datetime(2026, 7, 10, tzinfo=timezone.utc)
FIXED_PRINCIPAL = CurrentPrincipal(
    user_id="rule-test-user", tenant_id="rule-test-tenant"
)


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

    def update_evaluation_execution(
        self, tenant_id, strategy_id, evaluation_id, execution
    ):
        if strategy_id != STRATEGY_ID or tenant_id != FIXED_PRINCIPAL.tenant_id:
            return None
        journal = next(
            (item for item in self.evaluations if item.evaluation_id == evaluation_id), None
        )
        if journal is not None:
            journal.result = {**journal.result, "execution": execution}
        return journal


def _config() -> dict:
    return {
        "mode": "paper",
        "confirmation_mode": "all",
        "rsi": {"enabled": True, "period": 2, "oversold": 30, "overbought": 70},
        "risk": {
            "order_quote_amount": 100,
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
    app.dependency_overrides[get_db] = lambda: SimpleNamespace(query=lambda _model: SimpleNamespace(filter_by=lambda **_kwargs: SimpleNamespace(first=lambda: None)))
    return TestClient(app)


def test_okx_demo_execution_config_requires_a_sandbox_connection_and_spot_risk_limits():
    from pydantic import ValidationError
    from valuecell.server.api.schemas.rule_strategy import RuleStrategyConfig

    with pytest.raises(ValidationError, match="sandbox_connection_id"):
        RuleStrategyConfig.model_validate(
            {**_config(), "execution": {"environment": "okx_demo"}}
        )

    with pytest.raises(ValidationError, match="leverage 1"):
        RuleStrategyConfig.model_validate(
            {
                **_config(),
                "risk": {**_config()["risk"], "leverage": 2},
                "execution": {
                    "environment": "okx_demo",
                    "sandbox_connection_id": "sandbox-okx-1",
                },
            }
        )

    with pytest.raises(ValidationError, match="max_daily_quote_amount"):
        RuleStrategyConfig.model_validate(
            {
                **_config(),
                "execution": {
                    "environment": "okx_demo",
                    "sandbox_connection_id": "sandbox-okx-1",
                    "max_order_quote_amount": 500,
                    "max_daily_quote_amount": 100,
                },
            }
        )

    config = RuleStrategyConfig.model_validate(
        {
            **_config(),
            "execution": {
                "environment": "okx_demo",
                "sandbox_connection_id": "sandbox-okx-1",
                "max_order_quote_amount": 250,
                "max_daily_quote_amount": 750,
            },
        }
    )
    assert config.execution.environment == "okx_demo"
    assert config.execution.sandbox_connection_id == "sandbox-okx-1"
    assert config.execution.max_order_quote_amount == 250
    assert config.execution.max_daily_quote_amount == 750



def test_rule_strategy_api_rejects_unverified_okx_demo_connection_without_leaking_details():
    client = _client()
    response = client.post(
        "/rule-strategies",
        json={
            "name": "Demo strategy",
            "config": {
                **_config(),
                "execution": {
                    "environment": "okx_demo",
                    "sandbox_connection_id": "not-owned-or-not-demo",
                },
            },
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "okx_demo_connection_invalid",
        "error_code": "credential_or_permission_error",
    }


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
        "order_quote_amount": 100.0,
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
            "order_quote_amount": 250,
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
    assert updated.json()["data"]["config"]["risk"]["order_quote_amount"] == 250.0

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

    target_change = client.patch(
        f"/rule-strategies/{STRATEGY_ID}",
        json={
            "config": {
                **_config(),
                "execution": {
                    "environment": "okx_demo",
                    "sandbox_connection_id": "demo-connection",
                },
            }
        },
    )
    assert target_change.status_code == 409
    assert target_change.json()["detail"] == (
        "Stop the strategy before changing its execution target"
    )

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

    history = client.get(
        f"/rule-strategies/{STRATEGY_ID}/evaluations", params={"limit": 1}
    )

    assert history.status_code == 200
    body = history.json()
    assert body["code"] == 0
    item = body["data"][0]
    assert {
        key: item[key]
        for key in (
            "strategy_id",
            "evaluation_id",
            "evaluated_at",
            "action",
            "reason_code",
            "reason",
            "conditions",
            "indicators",
            "sizing",
            "funding",
            "account",
        )
    } == {
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
    }
    assert item["trades"] == [
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
    ]
    assert [stage["code"] for stage in item["funnel"]] == [
        "strategy_run",
        "market_ready",
        "conditions",
        "risk",
        "order_submission",
        "fill",
    ]
    assert [stage["status"] for stage in item["funnel"]] == [
        "passed",
        "passed",
        "passed",
        "passed",
        "passed",
        "filled",
    ]
    assert item["blocked_stage"] is None
    assert item["condition_summary"] == {
        "matched": result["entry_confirmation"]["passed"],
        "total": result["entry_confirmation"]["enabled"],
        "required": result["entry_confirmation"]["required"],
        "available": result["entry_confirmation"]["available"],
    }
    assert [condition["code"] for condition in body["data"][0]["conditions"]] == [
        condition["code"] for condition in result["conditions"]
    ]


@pytest.mark.parametrize(
    ("order_status", "submission", "fill"),
    [
        ("open", "passed", "pending"),
        ("rejected", "rejected", "rejected"),
        ("partially_filled", "passed", "partial"),
        ("filled", "passed", "filled"),
    ],
)
def test_evaluations_api_reads_back_durable_demo_execution_mapping(
    order_status, submission, fill
):
    repository = InMemoryRuleStrategyRepository()
    service = RuleStrategyService(repository=repository)
    app = FastAPI()
    app.include_router(create_rule_strategy_router(service=service))
    app.dependency_overrides[get_current_principal] = lambda: FIXED_PRINCIPAL
    app.dependency_overrides[get_db] = lambda: SimpleNamespace()
    service.create(FIXED_PRINCIPAL.tenant_id, "demo", None, __import__(
        "valuecell.server.api.schemas.rule_strategy", fromlist=["RuleStrategyConfig"]
    ).RuleStrategyConfig())
    repository.append_evaluation(SimpleNamespace(
        evaluation_id=EVALUATION_ID,
        created_at=CREATED_AT,
        result={
            "action": "buy", "reason_code": "buy", "reason": "buy",
            "conditions": [
                {"code": "rsi", "category": "indicator", "state": "triggered", "detail": "yes"}
            ],
            "entry_confirmation": {"enabled": 1, "available": 1, "passed": 1, "required": 1, "mode": "all"},
            "execution": {"execution": "okx_demo_submitted", "status": order_status},
        },
        trades=[],
    ))

    response = TestClient(app).get(f"/rule-strategies/{STRATEGY_ID}/evaluations")
    assert response.status_code == 200
    item = response.json()["data"][0]
    assert item["execution"]["status"] == order_status
    assert item["funnel"][4]["status"] == submission
    assert item["funnel"][5]["status"] == fill
