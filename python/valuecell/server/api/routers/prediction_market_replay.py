"""Paper-only prediction-market deterministic replay API."""

from fastapi import APIRouter, HTTPException

from valuecell.server.api.schemas.base import SuccessResponse
from valuecell.server.api.schemas.prediction_market_replay import (
    PredictionMarketReplayPreviewData,
    PredictionMarketReplayPreviewRequest,
)
from valuecell.server.services.prediction_market_replay_service import (
    PredictionMarketReplayService,
)


def create_prediction_market_replay_router(prefix: str = "/prediction-markets/replay") -> APIRouter:
    """Create the isolated frozen-book paper replay router."""
    router = APIRouter(prefix=prefix, tags=["prediction-markets"])
    service = PredictionMarketReplayService()

    @router.post(
        "/preview",
        response_model=SuccessResponse[PredictionMarketReplayPreviewData],
        summary="Replay a frozen public prediction-market order book on paper",
    )
    async def preview_prediction_market_replay(
        request: PredictionMarketReplayPreviewRequest,
    ) -> SuccessResponse[PredictionMarketReplayPreviewData]:
        try:
            data = service.preview(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Paper replay preview ready")

    return router
