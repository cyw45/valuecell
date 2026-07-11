"""Paper-only strategy experiment preview API."""

from fastapi import APIRouter, HTTPException

from valuecell.server.api.schemas.base import SuccessResponse
from valuecell.server.api.schemas.strategy_experiment import (
    StrategyExperimentPreviewData,
    StrategyExperimentPreviewRequest,
)
from valuecell.server.services.strategy_experiment_service import StrategyExperimentService


def create_strategy_experiment_router() -> APIRouter:
    """Create the paper-only strategy experiment router."""
    router = APIRouter(prefix="/strategies/experiments", tags=["strategies"])
    service = StrategyExperimentService()

    @router.post(
        "/preview",
        response_model=SuccessResponse[StrategyExperimentPreviewData],
        summary="Validate a paper strategy parameter candidate",
    )
    async def preview_strategy_experiment(
        request: StrategyExperimentPreviewRequest,
    ) -> SuccessResponse[StrategyExperimentPreviewData]:
        try:
            data = service.preview(request.strategy_type, request.parameters)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Paper experiment preview ready")

    return router
