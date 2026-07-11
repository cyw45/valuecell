from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from valuecell.server.api.routers.prediction_market import (
    create_prediction_market_router,
)
from valuecell.server.services.prediction_market_service import PredictionMarketService


MARKET_ID = "will-approval-pass"
YES_TOKEN_ID = "yes-token-id"
NO_TOKEN_ID = "no-token-id"


def _market(*, token_ids: str = '["yes-token-id", "no-token-id"]') -> dict[str, Any]:
    return {
        "id": MARKET_ID,
        "slug": "will-approval-pass",
        "question": "Will approval pass?",
        "active": True,
        "closed": False,
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": token_ids,
        "outcomePrices": '["0.62", "0.38"]',
    }


def _book(
    *,
    bids: list[dict[str, str]] | None = None,
    asks: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "market": MARKET_ID,
        "asset_id": YES_TOKEN_ID,
        "timestamp": "1",
        "hash": "official-clob-book-hash",
        "bids": bids
        if bids is not None
        else [
            {"price": "0.40", "size": "3"},
            {"price": "0.42", "size": "2"},
        ],
        "asks": asks
        if asks is not None
        else [
            {"price": "0.48", "size": "2"},
            {"price": "0.46", "size": "4"},
        ],
    }


class GammaClobTransport:
    def __init__(
        self,
        *,
        market: dict[str, Any] | None = None,
        book: dict[str, Any] | None = None,
    ) -> None:
        self.market = market or _market()
        self.book = book or _book()

    async def get(
        self, url: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if url.endswith("/markets/keyset"):
            return {"markets": [self.market], "next_cursor": "next-public-page"}
        if url.endswith(f"/markets/{MARKET_ID}"):
            return self.market
        if url.endswith("/book"):
            if params == {"token_id": YES_TOKEN_ID}:
                return self.book
            return {
                **self.book,
                "asset_id": NO_TOKEN_ID,
                "bids": [{"price": "0.30", "size": "1"}],
                "asks": [{"price": "0.70", "size": "1"}],
            }
        raise AssertionError(f"Unexpected public provider URL: {url}")


@pytest.mark.asyncio
async def test_catalog_pairs_official_gamma_outcomes_with_clob_tokens() -> None:
    catalog = await PredictionMarketService(GammaClobTransport()).catalog(limit=1)

    assert catalog.source == "polymarket-public"
    assert catalog.mode == "paper"
    assert catalog.freshness_status == "fresh"
    assert catalog.source_timestamp_ms == catalog.observed_at_ms
    assert catalog.freshness_age_ms == 0
    assert catalog.next_cursor == "next-public-page"
    assert [(item.outcome, item.token_id, item.price) for item in catalog.markets[0].outcomes] == [
        ("Yes", YES_TOKEN_ID, "0.62"),
        ("No", NO_TOKEN_ID, "0.38"),
    ]


@pytest.mark.asyncio
async def test_snapshot_normalizes_book_and_preserves_public_freshness() -> None:
    snapshot = await PredictionMarketService(GammaClobTransport()).snapshot(
        MARKET_ID, "Yes"
    )

    assert snapshot.source == "polymarket-public"
    assert snapshot.mode == "paper"
    assert snapshot.outcome == "Yes"
    assert snapshot.token_id == YES_TOKEN_ID
    assert [(level.price, level.size) for level in snapshot.book.bids] == [
        ("0.42", "2"),
        ("0.40", "3"),
    ]
    assert [(level.price, level.size) for level in snapshot.book.asks] == [
        ("0.46", "4"),
        ("0.48", "2"),
    ]
    assert snapshot.book.best_bid == "0.42"
    assert snapshot.book.best_ask == "0.46"
    assert snapshot.book.midpoint == "0.44"
    assert snapshot.book.microprice == "0.4333333333333333333333333333"
    assert snapshot.book.health.status == "valid"
    assert snapshot.source_timestamp_ms == 1
    assert snapshot.freshness_age_ms == snapshot.observed_at_ms - 1
    assert snapshot.freshness_status == "stale"


@pytest.mark.asyncio
async def test_signal_reports_microprice_and_insufficient_valid_history() -> None:
    signal_snapshot = await PredictionMarketService(GammaClobTransport()).signal(
        MARKET_ID,
        "Yes",
        ["0.40", "invalid", "0", "0.43"],
    )

    assert signal_snapshot.signal is not None
    assert signal_snapshot.signal.reference_price == signal_snapshot.book.microprice
    assert signal_snapshot.signal.reference_method == "microprice"
    assert signal_snapshot.signal.observation_count == 2
    assert signal_snapshot.signal.volatility is None
    assert signal_snapshot.signal.volatility_status == "insufficient_history"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("market", "book", "outcome", "message"),
    [
        (
            _market(token_ids='["yes-token-id"]'),
            None,
            "Yes",
            "equal non-zero length",
        ),
        (None, None, "Maybe", "outcome must match a market outcome"),
        (None, _book(bids=[{"price": "invalid", "size": "2"}]), "Yes", "invalid probability"),
        (
            None,
            _book(
                asks=[
                    {"price": "0.46", "size": "2"},
                    {"price": "0.46", "size": "3"},
                ]
            ),
            "Yes",
            "unique prices",
        ),
    ],
)
async def test_snapshot_rejects_malformed_markets_outcomes_and_book_levels(
    market: dict[str, Any] | None,
    book: dict[str, Any] | None,
    outcome: str,
    message: str,
) -> None:
    service = PredictionMarketService(GammaClobTransport(market=market, book=book))

    with pytest.raises(ValueError, match=message):
        await service.snapshot(MARKET_ID, outcome)


def _router_client() -> TestClient:
    app = FastAPI()
    app.include_router(create_prediction_market_router())
    return TestClient(app)


def test_prediction_market_router_returns_public_catalog_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = PredictionMarketService(GammaClobTransport())
    monkeypatch.setattr(
        "valuecell.server.api.routers.prediction_market.get_prediction_market_service",
        lambda: service,
    )

    response = _router_client().get("/prediction-markets/catalog", params={"limit": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    assert body["msg"] == "Public prediction markets retrieved"
    assert body["data"]["source"] == "polymarket-public"
    assert body["data"]["mode"] == "paper"
    assert body["data"]["markets"][0]["outcomes"] == [
        {"outcome": "Yes", "token_id": YES_TOKEN_ID, "price": "0.62"},
        {"outcome": "No", "token_id": NO_TOKEN_ID, "price": "0.38"},
    ]


def test_prediction_market_router_validates_requests_and_maps_upstream_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = PredictionMarketService(GammaClobTransport())
    monkeypatch.setattr(
        "valuecell.server.api.routers.prediction_market.get_prediction_market_service",
        lambda: service,
    )

    invalid = _router_client().get(f"/prediction-markets/markets/{MARKET_ID}")

    assert invalid.status_code == 422
    assert invalid.json()["detail"][0]["loc"] == ["query", "outcome"]
    monkeypatch.setattr(
        "valuecell.server.api.routers.prediction_market.get_prediction_market_service",
        lambda: PredictionMarketService(GammaClobTransport()),
    )

    invalid_outcome = _router_client().get(
        f"/prediction-markets/markets/{MARKET_ID}", params={"outcome": "Maybe"}
    )

    assert invalid_outcome.status_code == 400
    assert invalid_outcome.json() == {"detail": "outcome must match a market outcome"}

    class UnavailablePublicService:
        async def catalog(self, limit: int, after_cursor: str | None) -> None:
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        "valuecell.server.api.routers.prediction_market.get_prediction_market_service",
        lambda: UnavailablePublicService(),
    )

    unavailable = _router_client().get("/prediction-markets/catalog")

    assert unavailable.status_code == 502
    assert unavailable.json() == {
        "detail": "Public prediction-market data is unavailable."
    }
