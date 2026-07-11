import re

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from valuecell.server.api.routers.strategy_experiment import (
    create_strategy_experiment_router,
)
from valuecell.server.services.strategy_experiment_service import (
    StrategyExperimentService,
)


STRATEGY_TYPE = "LongTermSpotRsiStrategy"
VALID_PARAMETERS = {
    "entry_rsi_thresholds": [30.0, 25, 20.0],
    "sell_rsi_thresholds": [70.0, 80, 90.0],
    "bear_cap_ratio": 0.6,
    "daily_overbought_rsi": 82.0,
    "max_additions": 3.0,
}


def _dump(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()


def test_preview_service_canonicalizes_spot_rsi_input_and_is_repeatable():
    service = StrategyExperimentService()

    first = _dump(service.preview(STRATEGY_TYPE, VALID_PARAMETERS))
    second = _dump(
        service.preview(
            STRATEGY_TYPE,
            {
                "max_additions": 3,
                "daily_overbought_rsi": 82,
                "bear_cap_ratio": 0.60,
                "sell_rsi_thresholds": [70, 80.0, 90],
                "entry_rsi_thresholds": [30, 25.0, 20],
            },
        )
    )

    assert first["mode"] == "paper"
    assert first["strategy_type"] == STRATEGY_TYPE
    assert first["parameters"]["entry_rsi_thresholds"] == [30, 25, 20]
    assert first["parameters"]["sell_rsi_thresholds"] == [70, 80, 90]
    assert first["parameters"]["bear_cap_ratio"] == 0.6
    assert first["parameters"]["daily_overbought_rsi"] == 82.0
    assert first["parameters"]["max_additions"] == 3
    assert re.fullmatch(r"[0-9a-f]{64}", first["fingerprint"])
    assert first["fingerprint"] == second["fingerprint"]
    assert first["candidate_summary"] == second["candidate_summary"]
    assert isinstance(first["warnings"], list)
    assert isinstance(first["diagnostics"], list)
    assert {"entry_steps", "exit_steps", "total_entry_allocation", "max_exposure_ratio", "risk_level"} <= set(first["candidate_summary"])
    assert not {
        "backtest",
        "backtest_results",
        "historical_results",
        "historical_data",
        "persistence",
        "strategy_id",
    } & set(first)


@pytest.mark.parametrize(
    "parameters",
    [
        {"entry_rsi_thresholds": [20, 25, 30]},
        {"sell_rsi_thresholds": [90, 80, 70]},
        {"entry_rsi_thresholds": [30, "invalid", 20]},
        {"entry_rsi_thresholds": [30, 30]},
        {"entry_rsi_thresholds": [0, 101]},
        {"bear_cap_ratio": 1.1},
        {"daily_overbought_rsi": 101},
        {"max_additions": -1},
    ],
)
def test_preview_service_rejects_invalid_ladder_and_risk_values(parameters):
    with pytest.raises(ValueError):
        StrategyExperimentService().preview(STRATEGY_TYPE, parameters)


@pytest.mark.parametrize(
    "parameters",
    [
        {"entry_rsi_thresholds": [20, 25, 30]},
        {"entry_rsi_thresholds": [20, 30]},
        {"bear_cap_ratio": 0},
    ],
)
def test_preview_api_returns_422_for_invalid_dynamic_configuration(parameters):
    app = FastAPI()
    app.include_router(create_strategy_experiment_router())

    response = TestClient(app).post(
        "/strategies/experiments/preview",
        json={"strategy_type": STRATEGY_TYPE, "parameters": parameters},
    )

    assert response.status_code == 422


def test_preview_api_returns_paper_only_success_response():
    app = FastAPI()
    app.include_router(create_strategy_experiment_router())

    response = TestClient(app).post(
        "/strategies/experiments/preview",
        json={"strategy_type": STRATEGY_TYPE, "parameters": VALID_PARAMETERS},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert body["data"]["mode"] == "paper"
    assert re.fullmatch(r"[0-9a-f]{64}", body["data"]["fingerprint"])
    assert "candidate_summary" in body["data"]
    assert "historical_results" not in body["data"]
    assert "backtest" not in body["data"]
