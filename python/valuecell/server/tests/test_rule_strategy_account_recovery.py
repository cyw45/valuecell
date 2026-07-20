from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from valuecell.server.api.auth import CurrentPrincipal, get_current_principal
from valuecell.server.api.routers.rule_strategy import create_rule_strategy_router
from valuecell.server.api.schemas.rule_strategy import RuleStrategyConfig
from valuecell.server.db.models.rule_strategy import RuleStrategy, RuleStrategyEvaluationJournal
from valuecell.server.services.rule_strategy_service import (
    RuleStrategyService,
    RuleStrategyUnsupportedEvaluationError,
)


TENANT_ID = "account-recovery-tenant"
STRATEGY_ID = "rule_account_recovery"
NOW = datetime(2026, 7, 20, tzinfo=timezone.utc)


def _config(environment: str = "paper") -> RuleStrategyConfig:
    execution = {"environment": environment}
    if environment == "okx_demo":
        execution["sandbox_connection_id"] = "okx-demo-credential"
    return RuleStrategyConfig.model_validate(
        {
            "mode": "paper",
            "initial_capital_quote": 1_000,
            "confirmation_mode": "all",
            "rsi": {"enabled": True, "period": 2, "oversold": 30, "overbought": 70},
            "risk": {"order_quote_amount": 100, "max_positions": 1, "leverage": 1},
            "execution": execution,
        }
    )


def _strategy(environment: str = "paper") -> RuleStrategy:
    strategy = RuleStrategy(
        strategy_id=STRATEGY_ID,
        tenant_id=TENANT_ID,
        name="account recovery",
        status="running",
        paper_mode=True,
        config=_config(environment).model_dump(mode="json"),
    )
    strategy.created_at = NOW
    strategy.updated_at = NOW
    strategy.execution_generation = 1
    return strategy


def _paper_account() -> dict:
    return {
        "initial_capital_quote": 1_000.0,
        "quote_balance": 750.0,
        "positions": {
            "BTC-USDT": {"quantity": 2.5, "entry_price": 100.0, "mark_price": 100.0}
        },
        "realized_pnl_quote": 0.0,
        "unrealized_pnl_quote": 0.0,
        "equity_quote": 1_000.0,
    }


class HistoryRepository:
    def __init__(self, strategy: RuleStrategy, journals: list[RuleStrategyEvaluationJournal]):
        self.strategy = strategy
        self.journals = journals
        self.account_query_calls = 0

    def get(self, strategy_id: str, tenant_id: str):
        return self.strategy if strategy_id == STRATEGY_ID and tenant_id == TENANT_ID else None

    def get_latest_account_evaluations(self, strategy_id: str, tenant_id: str):
        self.account_query_calls += 1
        return list(self.journals)


def _journal(index: int, account: dict) -> RuleStrategyEvaluationJournal:
    journal = RuleStrategyEvaluationJournal(
        evaluation_id=f"evaluation_{index}",
        strategy_id=STRATEGY_ID,
        tenant_id=TENANT_ID,
        result={"account": account},
        signals=[],
        trades=[],
        funding=[],
    )
    journal.created_at = NOW + timedelta(seconds=index)
    return journal


def test_account_recovery_skips_demo_diagnostics_and_malformed_snapshots():
    demo_diagnostic = {
        "quote_balance": 99.0,
        "equity_quote": 101.0,
        "open_position_count": 1,
        "position": {"quantity": 1.0, "entry_price": 100.0},
        "source": "okx_demo",
    }
    malformed_paper = {**_paper_account(), "positions": {"BTC-USDT": {"quantity": -1}}}
    repository = HistoryRepository(
        _strategy(),
        [_journal(3, demo_diagnostic), _journal(2, malformed_paper), _journal(1, _paper_account())],
    )

    account = RuleStrategyService(repository=repository)._account_from_history(
        repository.strategy, TENANT_ID, _config()
    )

    assert account.quote_balance == 750.0
    assert account.positions["BTC-USDT"].quantity == 2.5
    assert repository.account_query_calls == 1


def test_account_recovery_survives_more_than_one_hundred_newer_diagnostics():
    diagnostics = [
        _journal(
            index,
            {
                "quote_balance": float(index),
                "equity_quote": float(index),
                "open_position_count": 0,
                "position": {"quantity": 0},
                "source": "okx_demo",
            },
        )
        for index in range(101, 202)
    ]
    repository = HistoryRepository(_strategy(), [*reversed(diagnostics), _journal(1, _paper_account())])

    account = RuleStrategyService(repository=repository)._account_from_history(
        repository.strategy, TENANT_ID, _config()
    )

    assert account.quote_balance == 750.0
    assert account.positions["BTC-USDT"].quantity == 2.5


def test_manual_okx_demo_evaluation_is_rejected_before_using_paper_history():
    repository = HistoryRepository(_strategy("okx_demo"), [_journal(1, _paper_account())])
    service = RuleStrategyService(repository=repository)

    with pytest.raises(RuleStrategyUnsupportedEvaluationError, match="synchronized OKX Demo account"):
        service.evaluate(
            STRATEGY_ID,
            TENANT_ID,
            candles=[],
            market={"symbol": "BTC-USDT", "price": 100.0},
        )

    assert repository.account_query_calls == 0


def test_manual_okx_demo_evaluation_api_returns_understandable_business_error():
    repository = HistoryRepository(_strategy("okx_demo"), [_journal(1, _paper_account())])
    app = FastAPI()
    app.include_router(create_rule_strategy_router(RuleStrategyService(repository=repository)))
    app.dependency_overrides[get_current_principal] = lambda: CurrentPrincipal(
        user_id="user", tenant_id=TENANT_ID
    )
    response = TestClient(app).post(
        f"/rule-strategies/{STRATEGY_ID}/evaluate",
        json={
            "candles": [
                {"timestamp_ms": 1, "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1}
            ],
            "market": {"symbol": "BTC-USDT", "price": 100},
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "okx_demo_manual_evaluation_unsupported",
        "message": "Manual evaluation cannot reliably synchronize the bound OKX Demo account; use scheduled Demo evaluation instead.",
    }
