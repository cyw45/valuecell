from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

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
    }

    decisions = {item["symbol"]: item for item in data["symbol_decisions"]}
    assert decisions["BTC-USDT"] == {
        "symbol": "BTC-USDT",
        "intervals_seen": ["1m", "5m"],
        "has_market_snapshot": True,
        "latest_price": 63_000.5,
        "action": "open_long",
        "quantity": 0.01,
        "reason": "RSI recovered above ladder entry threshold",
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


def test_strategy_diagnostics_api_returns_success_response(monkeypatch: pytest.MonkeyPatch):
    payload = {
        "strategy_id": STRATEGY_ID,
        "strategy_name": "Transparent RSI",
        "status": "running",
        "trading_mode": "live",
        "exchange_id": "okx",
        "strategy_type": "LongTermSpotRsiStrategy",
        "runtime_health": {"ok": True, "state": "running"},
        "config": {"symbols": ["BTC-USDT"]},
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
                "reason": "RSI recovered above ladder entry threshold",
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
    assert body["data"]["strategy_id"] == STRATEGY_ID
    assert body["data"]["latest_cycle"]["market_data_health"]["fetched_count"] == 1
    assert body["data"]["symbol_decisions"][0]["reason"] == (
        "RSI recovered above ladder entry threshold"
    )


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
