import re

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from valuecell.server.api.routers.prediction_market import (
    create_prediction_market_router,
)
from valuecell.server.api.schemas.prediction_market_replay import (
    PredictionMarketReplayPreviewRequest,
)
from valuecell.server.services.prediction_market_replay_service import (
    PredictionMarketReplayService,
)


def _request(
    *,
    side: str = "buy",
    size: float = 6.0,
    max_levels: int = 2,
    latency_ms: int = 100,
    extra_slippage_bps: float = 0.0,
    snapshots: list[dict] | None = None,
) -> PredictionMarketReplayPreviewRequest:
    return PredictionMarketReplayPreviewRequest.model_validate(
        {
            "decision_time_ms": 1_000,
            "latency_ms": latency_ms,
            "order": {
                "side": side,
                "size": size,
                "max_levels": max_levels,
                "extra_slippage_bps": extra_slippage_bps,
            },
            "snapshots": snapshots
            or [
                {
                    "source_timestamp_ms": 1_090,
                    "observed_at_ms": 1_100,
                    "bids": [{"price": 0.49, "size": 5}],
                    "asks": [{"price": 0.51, "size": 5}],
                },
                {
                    "source_timestamp_ms": 1_100,
                    "observed_at_ms": 1_150,
                    "bids": [
                        {"price": 0.50, "size": 2},
                        {"price": 0.48, "size": 3},
                    ],
                    "asks": [
                        {"price": 0.52, "size": 2},
                        {"price": 0.54, "size": 3},
                    ],
                },
            ],
        }
    )


def _dump(value):
    return value.model_dump(mode="json")


def test_preview_uses_first_latency_eligible_snapshot_and_visible_ioc_liquidity():
    result = _dump(PredictionMarketReplayService().preview(_request()))

    assert result["mode"] == "paper"
    assert result["simulation_mode"] == "simulated"
    assert result["source"] == "polymarket-public"
    assert result["source_timestamp_ms"] == 1_100
    assert result["observed_at_ms"] == 1_150
    assert result["freshness_age_ms"] == 50
    assert result["freshness_status"] == "fresh"
    assert result["assumptions"] == {
        "eligible_time_ms": 1_100,
        "execution_snapshot_timestamp_ms": 1_100,
        "max_levels": 2,
        "extra_slippage_bps": 0.0,
        "remainder_policy": "cancel",
        "canceled_remainder": True,
        "liquidity_scope": "visible_frozen_levels",
    }
    assert result["fill"] == {
        "requested_size": 6.0,
        "filled_size": 5.0,
        "unfilled_size": 1.0,
        "vwap": pytest.approx(0.532),
        "levels_consumed": 2,
    }
    assert result["mark_to_book"] == {
        "mark_price": 0.51,
        "pnl": pytest.approx(-0.11),
        "currency": "quote",
    }
    assert re.fullmatch(r"[0-9a-f]{64}", result["fingerprint"])


@pytest.mark.parametrize(
    ("side", "expected_vwap", "expected_pnl"),
    [
        ("buy", 0.5266666666666667, -0.05),
        ("sell", 0.49333333333333335, -0.05),
    ],
)
def test_preview_walks_only_the_correct_side_of_the_visible_book(
    side: str, expected_vwap: float, expected_pnl: float
):
    result = _dump(PredictionMarketReplayService().preview(_request(side=side, size=3)))

    assert result["fill"] == {
        "requested_size": 3.0,
        "filled_size": 3.0,
        "unfilled_size": 0.0,
        "vwap": pytest.approx(expected_vwap),
        "levels_consumed": 2,
    }
    assert result["mark_to_book"]["mark_price"] == 0.51
    assert result["mark_to_book"]["pnl"] == pytest.approx(expected_pnl)


@pytest.mark.parametrize(
    ("side", "expected_vwap", "expected_pnl"),
    [
        ("buy", 0.5319333333333334, -0.0658),
        ("sell", 0.4884, -0.0648),
    ],
)
def test_preview_applies_extra_slippage_against_the_order_direction(
    side: str, expected_vwap: float, expected_pnl: float
):
    request = _request(
        side=side,
        size=3,
        extra_slippage_bps=100,
    )

    result = _dump(PredictionMarketReplayService().preview(request))

    assert result["assumptions"]["extra_slippage_bps"] == 100.0
    assert result["fill"]["vwap"] == pytest.approx(expected_vwap)
    assert result["mark_to_book"]["pnl"] == pytest.approx(expected_pnl)


def test_preview_cancels_entire_order_when_no_snapshot_is_latency_eligible():
    result = _dump(
        PredictionMarketReplayService().preview(
            _request(latency_ms=200, snapshots=_dump(_request()).get("snapshots"))
        )
    )

    assert result["assumptions"]["eligible_time_ms"] == 1_200
    assert result["assumptions"]["execution_snapshot_timestamp_ms"] is None
    assert result["assumptions"]["canceled_remainder"] is True
    assert result["freshness_status"] == "unavailable"
    assert result["fill"] == {
        "requested_size": 6.0,
        "filled_size": 0.0,
        "unfilled_size": 6.0,
        "vwap": None,
        "levels_consumed": 0,
    }
    assert result["mark_to_book"] == {
        "mark_price": None,
        "pnl": 0.0,
        "currency": "quote",
    }


def test_preview_is_repeatable_for_the_same_frozen_request():
    request = _request()
    service = PredictionMarketReplayService()

    assert _dump(service.preview(request)) == _dump(service.preview(request))


@pytest.mark.parametrize(
    "invalid_request",
    [
        _request(
            snapshots=[
                {
                    "source_timestamp_ms": 1_100,
                    "observed_at_ms": 1_150,
                    "bids": [{"price": 0.52, "size": 1}],
                    "asks": [{"price": 0.52, "size": 1}],
                }
            ]
        ),
        _request(max_levels=0),
    ],
)
def test_preview_rejects_crossed_or_invalid_books(invalid_request):
    with pytest.raises(ValueError):
        PredictionMarketReplayService().preview(invalid_request)


def test_preview_api_returns_structured_paper_replay_and_rejects_crossed_book():
    app = FastAPI()
    app.include_router(create_prediction_market_router())
    client = TestClient(app)
    request = _dump(_request())

    success = client.post("/prediction-markets/replay/preview", json=request)

    assert success.status_code == 200
    body = success.json()
    assert body["code"] == 0
    assert body["data"]["mode"] == "paper"
    assert body["data"]["fill"]["unfilled_size"] == 1.0
    assert body["data"]["assumptions"]["remainder_policy"] == "cancel"
    assert body["data"]["assumptions"]["canceled_remainder"] is True

    crossed = {
        **request,
        "snapshots": [
            {
                "source_timestamp_ms": 1_100,
                "observed_at_ms": 1_150,
                "bids": [{"price": 0.52, "size": 1}],
                "asks": [{"price": 0.52, "size": 1}],
            }
        ],
    }
    rejected = client.post("/prediction-markets/replay/preview", json=crossed)

    assert rejected.status_code == 422
    assert rejected.json()["detail"] == "snapshot best bid must be below best ask"
