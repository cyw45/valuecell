from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from valuecell.agents.common.trading.constants import (
    FEATURE_GROUP_BY_KEY,
    FEATURE_GROUP_BY_MARKET_SNAPSHOT,
)
from valuecell.agents.common.trading.diagnostics import build_cycle_diagnostics
from valuecell.agents.common.trading.models import (
    DecisionCycleResult,
    ExchangeConfig,
    FeatureVector,
    HistoryRecord,
    InstrumentRef,
    LLMModelConfig,
    PortfolioView,
    StrategySummary,
    StrategyType as TradingStrategyType,
    TradeDecisionAction,
    TradeDigest,
    TradeInstruction,
    TradeSide,
    TradingConfig,
    UserRequest,
)
from valuecell.server.api.routers.strategy import create_strategy_router
from valuecell.server.services.strategy_service import StrategyService


STRATEGY_ID = "strategy-diagnostics-1"
LATEST_CREATED_AT = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)
PREVIOUS_CREATED_AT = datetime(2026, 7, 6, 11, 55, tzinfo=timezone.utc)


class DiagnosticsRepo:
    def __init__(self, *, strategy=None, diagnostics=None):
        self.strategy = strategy
        self.diagnostics = diagnostics or []

    def get_strategy_by_strategy_id(self, strategy_id: str):
        assert strategy_id == STRATEGY_ID
        return self.strategy

    def get_cycle_diagnostics(self, strategy_id: str, limit: int | None = None):
        assert strategy_id == STRATEGY_ID
        if limit is None:
            return list(self.diagnostics)
        return list(self.diagnostics[:limit])


def _strategy() -> SimpleNamespace:
    return SimpleNamespace(
        strategy_id=STRATEGY_ID,
        name="Transparent RSI",
        status="running",
        config={
            "llm_model_config": {
                "provider": "openrouter",
                "model_id": "deepseek/deepseek-chat",
            },
            "exchange_config": {
                "exchange_id": "okx",
                "trading_mode": "live",
                "market_type": "spot",
            },
            "trading_config": {
                "strategy_type": "LongTermSpotRsiStrategy",
                "symbols": ["BTC-USDT", "ETH-USDT", "SOL-USDT"],
                "decide_interval": 300,
                "max_leverage": 1.0,
                "cap_factor": 1.25,
                "initial_capital": 10_000,
            },
        },
        strategy_metadata={"runtime_health": {"ok": True, "state": "running"}},
    )


def _cycle(
    *,
    compose_id: str,
    cycle_index: int,
    created_at: datetime,
    rationale: str,
    symbol_decisions: list[dict],
    market_data_health: dict,
    explanation: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        strategy_id=STRATEGY_ID,
        compose_id=compose_id,
        created_at=created_at,
        payload={
            "compose_id": compose_id,
            "cycle_index": cycle_index,
            "created_at": created_at.isoformat(),
            "rationale": rationale,
            "instruction_count": sum(
                1 for item in symbol_decisions if item.get("action") != "noop"
            ),
            "order_count": sum(
                1 for item in symbol_decisions if item.get("action") != "noop"
            ),
            "no_order_count": sum(
                1 for item in symbol_decisions if item.get("action") == "noop"
            ),
            "market_data_health": market_data_health,
            "symbol_decisions": symbol_decisions,
            "explanation": explanation or {},
        },
    )


def _diagnostics_rows() -> list[SimpleNamespace]:
    latest_symbols = [
        {
            "symbol": "BTC-USDT",
            "intervals_seen": ["1m", "5m"],
            "has_market_snapshot": True,
            "latest_price": 63_000.5,
            "action": "open_long",
            "quantity": 0.01,
            "reason": "RSI recovered above ladder entry threshold",
        },
        {
            "symbol": "ETH-USDT",
            "intervals_seen": ["1m"],
            "has_market_snapshot": True,
            "latest_price": 3_100.0,
            "action": "noop",
            "quantity": 0.0,
            "reason": "No order: RSI is neutral and price is inside bands",
        },
        {
            "symbol": "SOL-USDT",
            "intervals_seen": [],
            "has_market_snapshot": False,
            "latest_price": None,
            "action": "noop",
            "quantity": 0.0,
            "reason": "No order: missing OKX market snapshot",
        },
    ]
    previous_symbols = [
        {
            "symbol": "BTC-USDT",
            "intervals_seen": ["1m"],
            "has_market_snapshot": True,
            "latest_price": 62_900.0,
            "action": "noop",
            "quantity": 0.0,
            "reason": "No order: cooldown active",
        }
    ]
    return [
        _cycle(
            compose_id="compose-latest",
            cycle_index=7,
            created_at=LATEST_CREATED_AT,
            rationale="Latest scan placed one order and skipped two symbols.",
            symbol_decisions=latest_symbols,
            market_data_health={
                "ok": False,
                "provider": "okx",
                "fetched_count": 2,
                "missing_count": 1,
                "missing_symbols": ["SOL-USDT"],
            },
            explanation={
                "action_reason": "RSI recovered above ladder entry threshold",
                "triggered_conditions": [
                    {
                        "label": "strategy_decision",
                        "status": "triggered",
                        "detail": "RSI recovered above ladder entry threshold",
                    }
                ],
                "blocked_conditions": [
                    {
                        "label": "strategy_decision",
                        "status": "not_triggered",
                        "detail": "No order: RSI is neutral and price is inside bands",
                    }
                ],
                "fund_impact": {
                    "portfolio_value_after_cycle": 10_000.0,
                    "estimated_notional": 630.005,
                },
            },
        ),
        _cycle(
            compose_id="compose-previous",
            cycle_index=6,
            created_at=PREVIOUS_CREATED_AT,
            rationale="Previous scan skipped BTC.",
            symbol_decisions=previous_symbols,
            market_data_health={
                "ok": True,
                "provider": "okx",
                "fetched_count": 1,
                "missing_count": 0,
                "missing_symbols": [],
            },
        ),
    ]


def test_cycle_diagnostics_explanation_adds_conditions_and_fund_impact_without_breaking_symbol_decisions():
    request = UserRequest(
        llm_model_config=LLMModelConfig(provider="openrouter", model_id="test-model"),
        exchange_config=ExchangeConfig(exchange_id="okx", market_type="spot"),
        trading_config=TradingConfig(
            strategy_name="Structured RSI",
            strategy_type=TradingStrategyType.LONG_TERM_SPOT_RSI,
            symbols=["BTC-USDT", "ETH-USDT"],
            decide_interval=300,
            initial_capital=10_000,
            initial_free_cash=8_000,
            max_leverage=1,
            max_positions=2,
            cap_factor=1.2,
        ),
    )
    btc = InstrumentRef(symbol="BTC-USDT", exchange_id="okx")
    eth = InstrumentRef(symbol="ETH-USDT", exchange_id="okx")
    features = [
        FeatureVector(
            ts=1_000,
            instrument=btc,
            values={"price.last": 63_000.0, "rsi": 28.5},
            meta={"interval": "5m"},
        ),
        FeatureVector(
            ts=1_100,
            instrument=btc,
            values={"price.last": 63_100.0},
            meta={FEATURE_GROUP_BY_KEY: FEATURE_GROUP_BY_MARKET_SNAPSHOT},
        ),
        FeatureVector(
            ts=1_000,
            instrument=eth,
            values={"price.last": 3_100.0, "rsi": 54.0},
            meta={"interval": "5m"},
        ),
    ]
    instruction = TradeInstruction(
        instruction_id="compose-1:BTC-USDT",
        compose_id="compose-1",
        instrument=btc,
        action=TradeDecisionAction.OPEN_LONG,
        side=TradeSide.BUY,
        quantity=0.02,
        meta={"rationale": "RSI crossed back above the entry threshold"},
    )
    result = DecisionCycleResult(
        compose_id="compose-1",
        timestamp_ms=1_200,
        cycle_index=3,
        rationale="BTC triggered while ETH remained neutral.",
        strategy_summary=StrategySummary(strategy_id=STRATEGY_ID),
        instructions=[instruction],
        trades=[],
        history_records=[
            HistoryRecord(
                ts=1_100,
                kind="features",
                reference_id="compose-1",
                payload={
                    "features": [feature.model_dump(mode="json") for feature in features]
                },
            )
        ],
        digest=TradeDigest(ts=1_200),
        portfolio_view=PortfolioView(
            strategy_id=STRATEGY_ID,
            ts=1_200,
            account_balance=8_000,
            total_value=10_000,
            free_cash=8_000,
            buying_power=8_000,
        ),
    )

    diagnostics = build_cycle_diagnostics(request=request, result=result)

    decisions = {item["symbol"]: item for item in diagnostics["symbol_decisions"]}
    assert decisions["BTC-USDT"]["action"] == "open_long"
    assert decisions["BTC-USDT"]["quantity"] == 0.02
    assert decisions["BTC-USDT"]["reason"] == "RSI crossed back above the entry threshold"
    assert decisions["ETH-USDT"]["action"] == "noop"
    assert decisions["ETH-USDT"]["reason"] == "BTC triggered while ETH remained neutral."

    explanation = diagnostics["explanation"]
    assert explanation["action_reason"] == "BTC triggered while ETH remained neutral."
    assert {
        (condition["label"], condition["status"], condition["detail"])
        for condition in explanation["triggered_conditions"]
    } >= {
        (
            "strategy_decision",
            "triggered",
            "RSI crossed back above the entry threshold",
        )
    }
    assert {
        (condition["label"], condition["status"], condition["detail"])
        for condition in explanation["blocked_conditions"]
    } >= {
        (
            "strategy_decision",
            "not_triggered",
            "BTC triggered while ETH remained neutral.",
        )
    }
    assert explanation["fund_impact"] == {
        "portfolio_value_after_cycle": 10_000.0,
        "cash_after_cycle": 8_000.0,
        "estimated_notional": 1_262.0,
        "executed_notional": 0.0,
        "fee_cost": 0.0,
        "realized_pnl": 0.0,
    }

    btc_conditions = decisions["BTC-USDT"]["conditions"]
    assert any(
        condition["status"] in {"passed", "triggered"} for condition in btc_conditions
    )
    assert any(condition["status"] == "triggered" for condition in btc_conditions)
    assert decisions["BTC-USDT"]["fund_impact"]["requested_quantity"] == 0.02
    assert decisions["BTC-USDT"]["fund_impact"]["estimated_notional"] == 1_262.0
    assert decisions["BTC-USDT"]["fund_impact"]["portfolio_value_after_cycle"] == 10_000.0


def test_cycle_diagnostics_exposes_stale_data_as_exposure_gate_blocked():
    request = UserRequest(
        llm_model_config=LLMModelConfig(provider="openrouter", model_id="test-model"),
        exchange_config=ExchangeConfig(exchange_id="okx", market_type="spot"),
        trading_config=TradingConfig(
            strategy_name="Structured RSI",
            strategy_type=TradingStrategyType.LONG_TERM_SPOT_RSI,
            symbols=["BTC-USDT"],
            decide_interval=300,
            initial_capital=10_000,
            initial_free_cash=8_000,
            max_leverage=1,
            max_positions=1,
        ),
    )
    feature = FeatureVector(
        ts=1,
        instrument=InstrumentRef(symbol="BTC-USDT", exchange_id="okx"),
        values={"price.last": 63_000.0},
        meta={
            FEATURE_GROUP_BY_KEY: FEATURE_GROUP_BY_MARKET_SNAPSHOT,
            "snapshot_ts_ms": 1,
            "freshness_age_ms": 90_000,
            "freshness_status": "stale",
            "coverage_status": "complete",
        },
    )
    result = DecisionCycleResult(
        compose_id="compose-stale",
        timestamp_ms=100_000,
        cycle_index=1,
        rationale=None,
        instructions=[],
        trades=[],
        strategy_summary=StrategySummary(strategy_id=STRATEGY_ID),
        history_records=[
            HistoryRecord(
                ts=1,
                kind="features",
                reference_id="compose-stale",
                payload={"features": [feature.model_dump(mode="json")]},
            )
        ],
        digest=TradeDigest(ts=100_000),
        portfolio_view=PortfolioView(
            strategy_id=STRATEGY_ID, ts=100_000, account_balance=8_000
        ),
    )

    diagnostics = build_cycle_diagnostics(request=request, result=result)

    health = diagnostics["market_data_health"]
    assert health["status"] == "degraded"
    assert health["freshness_status"] == "stale"
    assert health["coverage_status"] == "complete"
    assert health["stale_symbols"] == ["BTC-USDT"]
    assert health["exposure_increase_allowed"] is False
    decision = diagnostics["symbol_decisions"][0]
    assert decision["freshness_status"] == "stale"
    assert decision["exposure_increase_allowed"] is False
    assert {
        (condition["label"], condition["status"])
        for condition in decision["conditions"]
    } >= {("market_freshness", "blocked"), ("exposure_increase_gate", "blocked")}


@pytest.mark.asyncio
async def test_strategy_diagnostics_service_serializes_config_cycle_health_and_reasons(
    monkeypatch: pytest.MonkeyPatch,
):
    repo = DiagnosticsRepo(strategy=_strategy(), diagnostics=_diagnostics_rows())
    monkeypatch.setattr(
        "valuecell.server.services.strategy_service.get_strategy_repository",
        lambda: repo,
    )

    result = await StrategyService.get_strategy_diagnostics(STRATEGY_ID)

    assert result is not None
    data = result.model_dump(mode="json")
    assert data["strategy_id"] == STRATEGY_ID
    assert data["strategy_name"] == "Transparent RSI"
    assert data["status"] == "running"
    assert data["trading_mode"] == "live"
    assert data["exchange_id"] == "okx"
    assert data["strategy_type"] == "LongTermSpotRsiStrategy"
    assert data["runtime_health"] == {"ok": True, "state": "running"}

    assert data["expected_symbol_count"] == 3
    assert data["observed_symbol_count"] == 2
    assert data["config"] == {
        "llm_provider": "openrouter",
        "llm_model_id": "deepseek/deepseek-chat",
        "market_type": "spot",
        "symbols": ["BTC-USDT", "ETH-USDT", "SOL-USDT"],
        "decide_interval": 300,
        "max_leverage": 1.0,
        "cap_factor": 1.25,
        "initial_capital": 10_000,
        "strategy_params": {},
    }
    assert data["explanation"] == {
        "action_reason": "RSI recovered above ladder entry threshold",
        "triggered_conditions": [
            {
                "label": "strategy_decision",
                "status": "triggered",
                "detail": "RSI recovered above ladder entry threshold",
            }
        ],
        "blocked_conditions": [
            {
                "label": "strategy_decision",
                "status": "not_triggered",
                "detail": "No order: RSI is neutral and price is inside bands",
            }
        ],
        "fund_impact": {
            "portfolio_value_after_cycle": 10_000.0,
            "estimated_notional": 630.005,
        },
    }

    latest = data["latest_cycle"]
    assert latest["compose_id"] == "compose-latest"
    assert latest["cycle_index"] == 7
    assert latest["rationale"] == "Latest scan placed one order and skipped two symbols."
    assert latest["instruction_count"] == 1
    assert latest["order_count"] == 1
    assert latest["no_order_count"] == 2
    assert latest["market_data_health"] == {
        "ok": False,
        "provider": "okx",
        "fetched_count": 2,
        "missing_count": 1,
        "missing_symbols": ["SOL-USDT"],
        "status": "degraded",
        "freshness_status": "fresh",
        "coverage_status": "partial",
        "stale_count": 0,
        "stale_symbols": [],
        "exposure_increase_allowed": False,
    }

    decisions = {item["symbol"]: item for item in data["symbol_decisions"]}
    assert decisions["BTC-USDT"] | {
        "indicator_snapshot": {},
        "conditions": [],
        "decision_path": [],
        "fund_impact": {},
    } == {
        "symbol": "BTC-USDT",
        "intervals_seen": ["1m", "5m"],
        "has_market_snapshot": True,
        "latest_price": 63_000.5,
        "action": "open_long",
        "quantity": 0.01,
        "reason": "RSI recovered above ladder entry threshold",
        "indicator_snapshot": {},
        "snapshot_ts_ms": None,
        "freshness_age_ms": None,
        "freshness_status": "missing",
        "coverage_status": "missing",
        "exposure_increase_allowed": False,
        "conditions": [],
        "decision_path": [],
        "fund_impact": {},
    }
    assert decisions["ETH-USDT"]["action"] == "noop"
    assert decisions["ETH-USDT"]["reason"].startswith("No order:")
    assert decisions["SOL-USDT"]["has_market_snapshot"] is False
    assert decisions["SOL-USDT"]["reason"] == "No order: missing OKX market snapshot"

    assert [cycle["compose_id"] for cycle in data["recent_cycles"]] == [
        "compose-latest",
        "compose-previous",
    ]


@pytest.mark.asyncio
async def test_strategy_diagnostics_service_returns_none_for_missing_strategy(
    monkeypatch: pytest.MonkeyPatch,
):
    repo = DiagnosticsRepo(strategy=None, diagnostics=[])
    monkeypatch.setattr(
        "valuecell.server.services.strategy_service.get_strategy_repository",
        lambda: repo,
    )

    assert await StrategyService.get_strategy_diagnostics(STRATEGY_ID) is None


def test_strategy_diagnostics_api_preserves_structured_explanation(
    monkeypatch: pytest.MonkeyPatch,
):
    payload = {
        "strategy_id": STRATEGY_ID,
        "strategy_name": "Transparent RSI",
        "status": "running",
        "trading_mode": "live",
        "exchange_id": "okx",
        "strategy_type": "LongTermSpotRsiStrategy",
        "runtime_health": {"ok": True, "state": "running"},
        "config": {"symbols": ["BTC-USDT"]},
        "explanation": {
            "action_reason": "RSI recovered above the entry threshold",
            "triggered_conditions": [
                {
                    "label": "strategy_decision",
                    "status": "triggered",
                    "detail": "RSI recovered above the entry threshold",
                }
            ],
            "blocked_conditions": [
                {
                    "label": "market_snapshot",
                    "status": "blocked",
                    "detail": "Missing realtime market snapshot",
                }
            ],
            "fund_impact": {
                "portfolio_value_after_cycle": 10_000.0,
                "estimated_notional": 630.005,
            },
        },
        "observed_symbol_count": 1,
        "expected_symbol_count": 1,
        "latest_cycle": {
            "compose_id": "compose-latest",
            "cycle_index": 7,
            "created_at": LATEST_CREATED_AT.isoformat(),
            "rationale": "Order opened.",
            "instruction_count": 1,
            "order_count": 1,
            "no_order_count": 0,
            "market_data_health": {
                "ok": True,
                "provider": "okx",
                "fetched_count": 1,
                "missing_count": 0,
                "missing_symbols": [],
            },
        },
        "symbol_decisions": [
            {
                "symbol": "BTC-USDT",
                "intervals_seen": ["1m"],
                "has_market_snapshot": True,
                "latest_price": 63_000.5,
                "action": "open_long",
                "quantity": 0.01,
                "reason": "RSI recovered above the entry threshold",
            }
        ],
        "recent_cycles": [],
    }
    monkeypatch.setattr(
        StrategyService,
        "get_strategy_diagnostics",
        AsyncMock(return_value=payload),
        raising=False,
    )
    app = FastAPI()
    app.include_router(create_strategy_router())

    response = TestClient(app).get(f"/strategies/diagnostics?id={STRATEGY_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    data = body["data"]
    assert data["strategy_id"] == STRATEGY_ID
    assert data["latest_cycle"]["market_data_health"]["fetched_count"] == 1
    assert data["symbol_decisions"][0]["reason"] == (
        "RSI recovered above the entry threshold"
    )
    assert data["explanation"] == payload["explanation"]


def test_strategy_diagnostics_api_returns_404_for_missing_strategy(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        StrategyService,
        "get_strategy_diagnostics",
        AsyncMock(return_value=None),
        raising=False,
    )
    app = FastAPI()
    app.include_router(create_strategy_router())

    response = TestClient(app).get(f"/strategies/diagnostics?id={STRATEGY_ID}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Strategy not found"
