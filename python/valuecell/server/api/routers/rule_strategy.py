"""Standalone API for persisted, deterministic paper rule strategies."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator

from valuecell.server.api.auth import CurrentPrincipal, get_current_principal
from valuecell.server.api.schemas.base import SuccessResponse
from valuecell.server.db.connection import get_db
from valuecell.server.db.models.tenant_credential import TenantCredential
from valuecell.server.api.schemas.rule_strategy import (
    RuleStrategyCandle,
    RuleStrategyConfig,
    RuleStrategyMarketSnapshot,
    RuleStrategyTextImportProposal,
)
from valuecell.server.services.rule_strategy_service import (
    RuleStrategyNotFoundError,
    RuleStrategyNotRunningError,
    RuleStrategyService,
    RuleStrategyUnsupportedEvaluationError,
)
from valuecell.server.services.rule_strategy_advisory_service import (
    RuleStrategyAdvisoryService,
    RuleStrategyAdvisoryUnavailableError,
)
from valuecell.server.services.rule_strategy_text_import_service import (
    RuleStrategyTextImportService,
    RuleStrategyTextImportUnavailableError,
)
from valuecell.server.services.rule_strategy_demo_execution_read_model import (
    DemoExecutionReadModelError,
    get_demo_execution_read_model,
)
from valuecell.server.services.sandbox_exchange_trading_service import (
    SandboxExchangeTradingService,
    SandboxTradingError,
)
from valuecell.server.db.repositories.rule_strategy_repository import (
    RuleStrategyRepository,
)
from valuecell.server.services.saas_access_service import (
    require_active_tenant,
    require_tenant_permission,
)


class RuleStrategyCreateRequest(BaseModel):
    """Create a stored deterministic strategy with paper or validated OKX Demo execution."""

    model_config = ConfigDict(extra="forbid")

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


class RuleStrategyTextImportRequest(BaseModel):
    """Natural-language strategy description awaiting explicit user review."""

    model_config = ConfigDict(extra="forbid")

    strategy_text: str = Field(min_length=10, max_length=8_000)


def create_rule_strategy_router(
    service: RuleStrategyService | None = None,
) -> APIRouter:
    """Create the deterministic strategy API router with isolated Demo execution validation."""

    router = APIRouter(prefix="/rule-strategies", tags=["rule-strategies"])
    rule_service = service or RuleStrategyService()
    advisory_service = RuleStrategyAdvisoryService()
    text_import_service = RuleStrategyTextImportService()

    def require_strategy_read(principal: CurrentPrincipal) -> None:
        require_active_tenant(principal)
        require_tenant_permission(principal, "tenant.read")

    def require_strategy_manage(principal: CurrentPrincipal) -> None:
        require_active_tenant(principal)
        require_tenant_permission(principal, "strategy.manage")

    @router.post(
        "/parse-strategy-text",
        response_model=SuccessResponse[RuleStrategyTextImportProposal],
    )
    async def parse_strategy_text(
        request: RuleStrategyTextImportRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[RuleStrategyTextImportProposal]:
        require_strategy_manage(principal)
        try:
            data = await text_import_service.parse(request.strategy_text)
        except RuleStrategyTextImportUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Strategy text parsed for review")

    @router.post("", response_model=SuccessResponse[dict[str, Any]], status_code=201)
    async def create_rule_strategy(
        request: RuleStrategyCreateRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db=Depends(get_db),
    ) -> SuccessResponse[dict[str, Any]]:
        require_strategy_manage(principal)
        if request.config.execution.environment == "okx_demo":
            connection_id = request.config.execution.sandbox_connection_id or ""
            credential = db.query(TenantCredential).filter_by(
                id=connection_id, tenant_id=principal.tenant_id, revoked=False
            ).first()
            metadata = credential.metadata_json if credential is not None else {}
            if (
                credential is None
                or credential.kind != "exchange"
                or credential.provider != "okx"
                or metadata.get("sandbox") is not True
                or metadata.get("market_type") != "spot"
            ):
                raise HTTPException(
                    status_code=422,
                    detail={"code": "okx_demo_connection_invalid", "error_code": "credential_or_permission_error"},
                )
        return SuccessResponse.create(
            data=rule_service.create(
                principal.tenant_id,
                request.name,
                request.description,
                request.config.model_copy(
                    update={"initial_capital_quote": request.initial_capital_quote}
                ),
            ),
            msg=(
                "OKX Demo rule strategy created"
                if request.config.execution.environment == "okx_demo"
                else "Paper rule strategy created"
            ),
        )

    @router.get("", response_model=SuccessResponse[list[dict[str, Any]]])
    async def list_rule_strategies(
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[list[dict[str, Any]]]:
        require_strategy_read(principal)
        return SuccessResponse.create(
            data=rule_service.list(principal.tenant_id),
            msg="Paper rule strategies retrieved",
        )

    @router.get("/{strategy_id}", response_model=SuccessResponse[dict[str, Any]])
    async def get_rule_strategy(
        strategy_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict[str, Any]]:
        require_strategy_read(principal)
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
        db=Depends(get_db),
    ) -> SuccessResponse[dict[str, Any]]:
        require_strategy_manage(principal)
        if request.config is not None:
            try:
                current_strategy = rule_service.get(strategy_id, principal.tenant_id)
            except RuleStrategyNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            current_execution = current_strategy["config"].get("execution", {})
            requested_execution = request.config.execution
            if current_strategy["status"] == "running" and (
                current_execution.get("environment", "paper")
                != requested_execution.environment
                or current_execution.get("sandbox_connection_id")
                != requested_execution.sandbox_connection_id
            ):
                raise HTTPException(
                    status_code=409,
                    detail="Stop the strategy before changing its execution target",
                )
        if request.config is not None and request.config.execution.environment == "okx_demo":
            connection_id = request.config.execution.sandbox_connection_id or ""
            credential = db.query(TenantCredential).filter_by(
                id=connection_id, tenant_id=principal.tenant_id, revoked=False
            ).first()
            metadata = credential.metadata_json if credential is not None else {}
            if (
                credential is None
                or credential.kind != "exchange"
                or credential.provider != "okx"
                or metadata.get("sandbox") is not True
                or metadata.get("market_type") != "spot"
            ):
                raise HTTPException(
                    status_code=422,
                    detail={"code": "okx_demo_connection_invalid", "error_code": "credential_or_permission_error"},
                )
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
        require_strategy_read(principal)
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
        require_strategy_manage(principal)
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
        require_strategy_manage(principal)
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
        require_strategy_manage(principal)
        try:
            data = rule_service.evaluate(
                strategy_id, principal.tenant_id, request.candles, request.market
            )
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuleStrategyNotRunningError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuleStrategyUnsupportedEvaluationError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "okx_demo_manual_evaluation_unsupported",
                    "message": (
                        "Manual evaluation cannot reliably synchronize the bound "
                        "OKX Demo account; use scheduled Demo evaluation instead."
                    ),
                },
            ) from exc
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
        require_strategy_read(principal)
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
        require_strategy_read(principal)
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
            ts_str = ts_val.strftime("%Y-%m-%dT%H:%M:%SZ") if ts_val is not None else ""
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
        "/{strategy_id}/demo-execution", response_model=SuccessResponse[dict[str, Any]]
    )
    async def get_rule_strategy_demo_execution(
        strategy_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db=Depends(get_db),
    ) -> SuccessResponse[dict[str, Any]]:
        """Return explicit OKX Demo facts; never substitute the paper ledger."""
        require_strategy_read(principal)
        try:
            strategy = rule_service.get(strategy_id, principal.tenant_id)
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            data = await get_demo_execution_read_model(
                strategy,
                principal.tenant_id,
                SandboxExchangeTradingService(db),
            )
        except DemoExecutionReadModelError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except SandboxTradingError as exc:
            raise HTTPException(
                status_code=502,
                detail={"code": "okx_demo_read_unavailable", "detail": str(exc)},
            ) from exc
        return SuccessResponse.create(data=data, msg="OKX Demo strategy execution retrieved")

    @router.get(
        "/{strategy_id}/account", response_model=SuccessResponse[dict[str, Any]]
    )
    async def get_rule_strategy_account(
        strategy_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict[str, Any]]:
        require_strategy_read(principal)
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
        require_strategy_read(principal)
        if log_type not in {"signals", "trades", "funding"}:
            raise HTTPException(status_code=404, detail="Log type was not found")
        try:
            data = rule_service.logs(strategy_id, principal.tenant_id, log_type, limit)
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg=f"Paper {log_type} log retrieved")

    return router
