"""Public Polymarket research routes with paper-only replay registration."""

from fastapi import APIRouter, HTTPException, Query

from valuecell.server.api.schemas.base import SuccessResponse
from valuecell.server.api.schemas.prediction_market import (
    PredictionMarketCatalogData,
    PredictionMarketSnapshotData,
)
from valuecell.server.services.prediction_market_service import (
    get_prediction_market_service,
)

from .prediction_market_replay import create_prediction_market_replay_router

def create_prediction_market_router() -> APIRouter:
    """Create public market observation and paper replay routes."""
    router = APIRouter(prefix="/prediction-markets", tags=["prediction-markets"])
    router.include_router(create_prediction_market_replay_router(prefix="/replay"))

    @router.get(
        "/catalog", response_model=SuccessResponse[PredictionMarketCatalogData]
    )
    async def catalog_prediction_markets(
        limit: int = Query(30, ge=1, le=100),
        after_cursor: str | None = Query(None),
    ) -> SuccessResponse[PredictionMarketCatalogData]:
        try:
            data = await get_prediction_market_service().catalog(limit, after_cursor)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            raise HTTPException(
                status_code=502, detail="Public prediction-market data is unavailable."
            ) from None
        return SuccessResponse.create(data=data, msg="Public prediction markets retrieved")

    @router.get(
        "/markets/{market_id}", response_model=SuccessResponse[PredictionMarketSnapshotData]
    )
    async def prediction_market_snapshot(
        market_id: str, outcome: str = Query(..., min_length=1)
    ) -> SuccessResponse[PredictionMarketSnapshotData]:
        try:
            data = await get_prediction_market_service().snapshot(market_id, outcome)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            raise HTTPException(
                status_code=502, detail="Public prediction-market order book is unavailable."
            ) from None
        return SuccessResponse.create(data=data, msg="Public order book retrieved")

    @router.get(
        "/markets/{market_id}/signal", response_model=SuccessResponse[PredictionMarketSnapshotData]
    )
    async def prediction_market_signal(
        market_id: str,
        outcome: str = Query(..., min_length=1),
        history: str = Query(""),
    ) -> SuccessResponse[PredictionMarketSnapshotData]:
        try:
            history_values = [item for item in history.split(",") if item]
            data = await get_prediction_market_service().signal(
                market_id, outcome, history_values
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            raise HTTPException(
                status_code=502, detail="Public prediction-market signal is unavailable."
            ) from None
        return SuccessResponse.create(data=data, msg="Public probability signal retrieved")

    return router
