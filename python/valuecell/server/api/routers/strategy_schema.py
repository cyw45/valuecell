"""Strategy configuration schema API."""

from fastapi import APIRouter

from valuecell.server.api.schemas import SuccessResponse
from valuecell.server.api.schemas.strategy_schema import StrategyConfigSchemaCatalog
from valuecell.server.services.strategy_schema_service import get_strategy_schema_service


def create_strategy_schema_router() -> APIRouter:
    router = APIRouter(prefix="/strategies", tags=["strategies"])

    @router.get(
        "/schemas",
        response_model=SuccessResponse[StrategyConfigSchemaCatalog],
        summary="List dynamic strategy configuration schemas",
    )
    async def get_strategy_schemas():
        service = get_strategy_schema_service()
        return SuccessResponse.create(
            data=service.get_catalog(),
            msg="Strategy schemas retrieved successfully",
        )

    return router
