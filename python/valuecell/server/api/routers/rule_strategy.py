"""Standalone API for persisted, deterministic paper rule strategies."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator

from valuecell.server.api.auth import CurrentPrincipal, get_current_principal
from valuecell.server.api.schemas.base import SuccessResponse
from valuecell.server.api.schemas.rule_strategy import (
    RuleStrategyCandle,
    RuleStrategyConfig,
    RuleStrategyMarketSnapshot,
)
from valuecell.server.services.rule_strategy_service import (
    RuleStrategyNotFoundError,
    RuleStrategyNotRunningError,
    RuleStrategyService,
)
from valuecell.server.services.rule_strategy_advisory_service import (
    RuleStrategyAdvisoryService,
    RuleStrategyAdvisoryUnavailableError,
)
from valuecell.server.db.repositories.rule_strategy_repository import (
    RuleStrategyRepository,
)


class RuleStrategyCreateRequest(BaseModel):
    """Create a stored rule strategy that can only run in paper mode."""

    initial_capital_quote: float = Field(default=10_000.0, gt=0, le=100_000_000)
    config: RuleStrategyConfig
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)

class RuleStrategyUpdateRequest(BaseModel):
    """Update only explicitly provided strategy metadata or configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    config: RuleStrategyConfig | None = None

    @model_validator(mode="after")
    def validate_nonempty_update(self) -> RuleStrategyUpdateRequest:
        if not self.model_fields_set:
            raise ValueError("At least one field must be supplied")
        return self


class RuleStrategyEvaluateRequest(BaseModel):
    """Frozen candles and market price; paper-account facts are server-owned."""

    model_config = ConfigDict(extra="forbid")

    candles: list[RuleStrategyCandle] = Field(min_length=1, max_length=5_000)
    market: RuleStrategyMarketSnapshot

def create_rule_strategy_router(
    service: RuleStrategyService | None = None,
) -> APIRouter:
    """Create the independent paper-only rule strategy API router."""

    router = APIRouter(prefix="/rule-strategies", tags=["rule-strategies"])
    rule_service = service or RuleStrategyService()
    advisory_service = RuleStrategyAdvisoryService()

    @router.post("", response_model=SuccessResponse[dict[str, Any]], status_code=201)
    async def create_rule_strategy(
        request: RuleStrategyCreateRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict[str, Any]]:
        return SuccessResponse.create(
            data=rule_service.create(
                principal.tenant_id,
                request.name,
                request.description,
                request.config.model_copy(
                    update={"initial_capital_quote": request.initial_capital_quote}
                ),
            ),
            msg="Paper rule strategy created",
        )

    @router.get("", response_model=SuccessResponse[list[dict[str, Any]]])
    async def list_rule_strategies(
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[list[dict[str, Any]]]:
        return SuccessResponse.create(
            data=rule_service.list(principal.tenant_id),
            msg="Paper rule strategies retrieved",
        )

    @router.get("/{strategy_id}", response_model=SuccessResponse[dict[str, Any]])
    async def get_rule_strategy(
        strategy_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict[str, Any]]:
        try:
            data = rule_service.get(strategy_id, principal.tenant_id)
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Paper rule strategy retrieved")

    @router.patch("/{strategy_id}", response_model=SuccessResponse[dict[str, Any]])
    async def update_rule_strategy(
        strategy_id: str,
        request: RuleStrategyUpdateRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict[str, Any]]:
        try:
            data = rule_service.update(
                strategy_id,
                principal.tenant_id,
                request.name,
                request.description,
                request.config,
            )
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Paper rule strategy updated")

    @router.post(
        "/{strategy_id}/advisory-analysis",
        response_model=SuccessResponse[dict[str, Any]],
    )
    async def review_rule_strategy_configuration(
        strategy_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict[str, Any]]:
        try:
            strategy = rule_service.get(strategy_id, principal.tenant_id)
            evaluations = rule_service.evaluations(
                strategy_id, principal.tenant_id, limit=10
            )
            data = advisory_service.review_configuration(strategy, evaluations)
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuleStrategyAdvisoryUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="AI advisory generated")

    @router.post("/{strategy_id}/start", response_model=SuccessResponse[dict[str, Any]])
    async def start_rule_strategy(
        strategy_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict[str, Any]]:
        try:
            data = rule_service.start(strategy_id, principal.tenant_id)
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Paper rule strategy started")

    @router.post("/{strategy_id}/stop", response_model=SuccessResponse[dict[str, Any]])
    async def stop_rule_strategy(
        strategy_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict[str, Any]]:
        try:
            data = rule_service.stop(strategy_id, principal.tenant_id)
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Paper rule strategy stopped")

    @router.post(
        "/{strategy_id}/evaluate", response_model=SuccessResponse[dict[str, Any]]
    )
    async def evaluate_rule_strategy(
        strategy_id: str,
        request: RuleStrategyEvaluateRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict[str, Any]]:
        try:
            data = rule_service.evaluate(
                strategy_id, principal.tenant_id, request.candles, request.market
            )
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuleStrategyNotRunningError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Paper rule strategy evaluated")

    @router.get(
        "/{strategy_id}/evaluations",
        response_model=SuccessResponse[list[dict[str, Any]]],
    )
    async def get_rule_strategy_evaluations(
        strategy_id: str,
        limit: int = Query(default=100, ge=1, le=500),
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[list[dict[str, Any]]]:
        try:
            data = rule_service.evaluations(strategy_id, principal.tenant_id, limit)
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SuccessResponse.create(
            data=data, msg="Paper evaluation explanations retrieved"
        )

    @router.get(
        "/{strategy_id}/pnl-curve",
        response_model=SuccessResponse[list[dict[str, Any]]],
    )
    async def get_rule_strategy_pnl_curve(
        strategy_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[list[dict[str, Any]]]:
        # Verify strategy exists and is tenant-scoped
        try:
            rule_service.get(strategy_id, principal.tenant_id)
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        repository = RuleStrategyRepository()
        journals = list(
            reversed(
                repository.get_evaluations(strategy_id, principal.tenant_id, limit=500)
            )
        )
        points: list[dict[str, Any]] = []
        for journal in journals:
            result: dict[str, Any] = journal.result or {}
            raw_account = result.get("account")
            if raw_account is None:
                continue
            initial_capital = float(raw_account["initial_capital_quote"])
            equity = float(raw_account["equity_quote"])
            ts_val = journal.created_at
            ts_str = (
                ts_val.strftime("%Y-%m-%dT%H:%M:%SZ")
                if ts_val is not None
                else ""
            )
            points.append(
                {
                    "ts": ts_str,
                    "cumulative_pnl": equity - initial_capital,
                    "equity_quote": equity,
                    "action": result.get("action", "no_op"),
                }
            )
        return SuccessResponse.create(data=points, msg="PnL curve retrieved")


    @router.get(
        "/{strategy_id}/account", response_model=SuccessResponse[dict[str, Any]]
    )
    async def get_rule_strategy_account(
        strategy_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict[str, Any]]:
        try:
            data = rule_service.account(strategy_id, principal.tenant_id)
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Paper account retrieved")
    @router.get(
        "/{strategy_id}/{log_type}", response_model=SuccessResponse[dict[str, Any]]
    )
    async def get_rule_strategy_logs(
        strategy_id: str,
        log_type: str,
        limit: int = Query(default=100, ge=1, le=500),
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict[str, Any]]:
        if log_type not in {"signals", "trades", "funding"}:
            raise HTTPException(status_code=404, detail="Log type was not found")
        try:
            data = rule_service.logs(
                strategy_id, principal.tenant_id, log_type, limit
            )
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg=f"Paper {log_type} log retrieved")

    return router
