from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from valuecell.server.api.auth import CurrentPrincipal, get_current_principal
from valuecell.server.api.routers.rule_strategy import create_rule_strategy_router
from valuecell.server.api.schemas.rule_strategy import (
    RuleStrategyConfig,
    RuleStrategyEngineMarketSnapshot,
    RuleStrategyMarketSnapshot,
    RuleStrategyPosition,
)
from valuecell.server.db.models.rule_strategy import RuleStrategyEvaluationJournal
from valuecell.server.services.rule_strategy_service import RuleStrategyService


CREATED_AT = datetime(2026, 7, 12, tzinfo=timezone.utc)


class PaperAccountRepository:
    """Deterministic repository preserving account journals by tenant."""

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
            "order_quote_amount": 100,
            "max_positions": 1,
            "leverage": 1,
        },
    }


def _market(price: float) -> dict:
    return {"symbol": "BTC-USDT", "price": price, "funding_rate": 0}


def _candles(*closes: float) -> list[dict]:
    return [
        {
            "timestamp_ms": index * 1_000,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 10,
        }
        for index, close in enumerate(closes, start=1)
    ]


def _client() -> tuple[TestClient, list[CurrentPrincipal]]:
    principal = [CurrentPrincipal(user_id="user-a", tenant_id="tenant-a")]
    app = FastAPI()
    app.include_router(
        create_rule_strategy_router(
            service=RuleStrategyService(repository=PaperAccountRepository())
        )
    )
    app.dependency_overrides[get_current_principal] = lambda: principal[0]
    return TestClient(app), principal


def _create_strategy(client: TestClient, name: str = "Paper ledger") -> str:
    response = client.post(
        "/rule-strategies",
        json={
            "name": name,
            "initial_capital_quote": 1_000,
            "config": _config(),
        },
    )
    assert response.status_code == 201
    return response.json()["data"]["strategy_id"]


def test_rule_strategy_defaults_and_persists_initial_paper_capital():
    client, _ = _client()

    default_capital = client.post(
        "/rule-strategies",
        json={"name": "Capital required", "config": _config()},
    )
    assert default_capital.status_code == 201
    assert (
        default_capital.json()["data"]["account"]["initial_capital_quote"] == 10_000.0
    )

    created = client.post(
        "/rule-strategies",
        json={
            "name": "Funded paper strategy",
            "initial_capital_quote": 1_000,
            "config": _config(),
        },
    )
    assert created.status_code == 201
    strategy_id = created.json()["data"]["strategy_id"]
    assert created.json()["data"]["config"]["initial_capital_quote"] == 1_000.0
    assert created.json()["data"]["account"] == {
        "initial_capital_quote": 1_000.0,
        "quote_balance": 1_000.0,
        "positions": {},
        "realized_pnl_quote": 0.0,
        "unrealized_pnl_quote": 0.0,
        "equity_quote": 1_000.0,
    }

    persisted = client.get(f"/rule-strategies/{strategy_id}")
    assert persisted.status_code == 200
    assert persisted.json()["data"]["account"] == created.json()["data"]["account"]


def test_paper_evaluations_use_server_account_and_record_buy_then_sell_pnl():
    client, _ = _client()
    strategy_id = _create_strategy(client)
    assert client.post(f"/rule-strategies/{strategy_id}/start").status_code == 200

    forged_account_facts = client.post(
        f"/rule-strategies/{strategy_id}/evaluate",
        json={
            "candles": _candles(100, 90, 80),
            "market": {
                **_market(80),
                "equity_quote": 9_999_999,
                "quote_balance": 9_999_999,
                "open_position_count": 0,
                "position": {"quantity": 0},
            },
        },
    )
    assert forged_account_facts.status_code == 422

    bought = client.post(
        f"/rule-strategies/{strategy_id}/evaluate",
        json={"candles": _candles(100, 90, 80), "market": _market(80)},
    )
    assert bought.status_code == 200
    buy_result = bought.json()["data"]
    assert buy_result["action"] == "buy"
    assert buy_result["account"] == {
        "initial_capital_quote": 1_000.0,
        "quote_balance": 900.0,
        "positions": {
            "BTC-USDT": {
                "quantity": 1.25,
                "entry_price": 80.0,
                "mark_price": 80.0,
            }
        },
        "realized_pnl_quote": 0.0,
        "unrealized_pnl_quote": 0.0,
        "equity_quote": 1_000.0,
    }

    sold = client.post(
        f"/rule-strategies/{strategy_id}/evaluate",
        json={"candles": _candles(80, 90, 100), "market": _market(100)},
    )
    assert sold.status_code == 200
    sell_result = sold.json()["data"]
    assert sell_result["action"] == "sell"
    assert sell_result["account"] == {
        "initial_capital_quote": 1_000.0,
        "quote_balance": 1_025.0,
        "positions": {},
        "realized_pnl_quote": 25.0,
        "unrealized_pnl_quote": 0.0,
        "equity_quote": 1_025.0,
    }

    trades = client.get(f"/rule-strategies/{strategy_id}/trades")
    assert trades.status_code == 200
    assert [entry["execution"] for entry in trades.json()["data"]["entries"]] == [
        "paper_filled",
        "paper_filled",
    ]
    assert [
        entry["realized_pnl_quote"] for entry in trades.json()["data"]["entries"]
    ] == [
        25.0,
        0.0,
    ]

    account = client.get(f"/rule-strategies/{strategy_id}/account")
    assert account.status_code == 200
    assert account.json()["data"] == sell_result["account"]


def test_paper_account_endpoint_is_tenant_scoped():
    client, principal = _client()
    strategy_id = _create_strategy(client, "Tenant A paper ledger")

    principal[0] = CurrentPrincipal(user_id="user-b", tenant_id="tenant-b")
    assert client.get(f"/rule-strategies/{strategy_id}/account").status_code == 404

    principal[0] = CurrentPrincipal(user_id="user-a", tenant_id="tenant-a")
    account = client.get(f"/rule-strategies/{strategy_id}/account")
    assert account.status_code == 200
    assert account.json()["data"]["initial_capital_quote"] == 1_000.0


def test_pnl_curve_skips_demo_and_incomplete_legacy_account_snapshots():
    repository = PaperAccountRepository()
    app = FastAPI()
    app.include_router(create_rule_strategy_router(RuleStrategyService(repository=repository)))
    app.dependency_overrides[get_current_principal] = lambda: CurrentPrincipal(
        user_id="user-a", tenant_id="tenant-a"
    )
    client = TestClient(app)
    strategy_id = _create_strategy(client, "PnL compatibility")
    repository.append_evaluation(
        RuleStrategyEvaluationJournal(
            evaluation_id="legacy-demo-snapshot",
            strategy_id=strategy_id,
            tenant_id="tenant-a",
            result={"account": {"source": "okx_demo", "equity_quote": 123.0}},
            signals=[],
            trades=[],
            funding=[],
        )
    )

    response = client.get(f"/rule-strategies/{strategy_id}/pnl-curve")

    assert response.status_code == 200
    assert response.json()["data"] == []


def test_batch_cycle_uses_fixed_amount_and_blocks_unaffordable_entries():
    repository = PaperAccountRepository()
    service = RuleStrategyService(repository=repository)
    config = RuleStrategyConfig.model_validate(
        {
            **_config(),
            "initial_capital_quote": 150,
            "symbols": ["BTC-USDT", "ETH-USDT", "SOL-USDT"],
            "risk": {
                "order_quote_amount": 100,
                "max_positions": 3,
                "leverage": 1,
            },
        }
    )
    created = service.create("tenant-a", "Fixed amount", None, config)
    service.start(created["strategy_id"], "tenant-a")

    results = service.evaluate_batch(
        created["strategy_id"],
        "tenant-a",
        [
            (_candles(100, 90, 80), RuleStrategyMarketSnapshot(symbol=symbol, price=80))
            for symbol in config.symbols
        ],
    )

    assert [result["action"] for result in results] == ["buy", "no_op", "no_op"]
    assert [result["sizing"]["requested_quote"] for result in results] == [
        100,
        100,
        100,
    ]
    assert [result["reason_code"] for result in results[1:]] == [
        "available_collateral",
        "available_collateral",
    ]
    account = results[-1]["account"]
    assert account["quote_balance"] == 50.0
    assert set(account["positions"]) == {"BTC-USDT"}


def test_okx_demo_batch_cycle_journals_signals_without_mutating_paper_account():
    repository = PaperAccountRepository()
    service = RuleStrategyService(repository=repository)
    config = RuleStrategyConfig.model_validate(
        {
            **_config(),
            "initial_capital_quote": 1_000,
            "symbols": ["BTC-USDT"],
            "execution": {
                "environment": "okx_demo",
                "sandbox_connection_id": "okx-demo-connection",
            },
        }
    )
    created = service.create("tenant-a", "OKX Demo ledger", None, config)
    service.start(created["strategy_id"], "tenant-a")

    results = service.evaluate_batch(
        created["strategy_id"],
        "tenant-a",
        [
            (
                _candles(100, 90, 80),
                RuleStrategyEngineMarketSnapshot(
                    symbol="BTC-USDT",
                    price=80,
                    equity_quote=1_000,
                    quote_balance=1_000,
                    open_position_count=0,
                    position=RuleStrategyPosition(),
                ),
            )
        ],
    )

    assert results[0]["action"] == "buy"
    assert results[0]["paper_fill"] is False
    assert results[0]["execution_ledger"] == "external"
    assert results[0]["account"]["quote_balance"] == 1_000.0
    assert results[0]["account"]["source"] == "okx_demo"
    assert results[0]["account"]["position"]["quantity"] == 0.0
    assert repository.evaluations[0].trades == []
