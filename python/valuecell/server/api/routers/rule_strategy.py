"""Standalone API for persisted, deterministic paper rule strategies."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
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
from valuecell.server.api.schemas.rule_strategy_validation import (
    RuleStrategyValidationCreateRequest,
)
from valuecell.server.services.rule_strategy_service import (
    RuleStrategyDeleteConflictError,
    RuleStrategyNotFoundError,
    RuleStrategyNotRunningError,
    RuleStrategyRunningUpdateError,
    RuleStrategyStartAdmissionError,
    RuleStrategyService,
    RuleStrategyUnsupportedEvaluationError,
)
from valuecell.server.services.rule_strategy_advisory_service import (
    RuleStrategyAdvisoryService,
    RuleStrategyAdvisoryUnavailableError,
)
from valuecell.server.services.rule_strategy_text_import_job_service import (
    RuleStrategyTextImportJob,
    RuleStrategyTextImportJobCapacityError,
    RuleStrategyTextImportJobConflictError,
    RuleStrategyTextImportJobNotFoundError,
    get_rule_strategy_text_import_job_service,
)
from valuecell.server.services.rule_strategy_demo_execution_read_model import (
    DemoExecutionReadModelError,
    get_demo_execution_read_model,
)
from valuecell.server.services.sandbox_exchange_trading_service import (
    SandboxExchangeTradingService,
    SandboxTradingError,
)
from valuecell.server.services.saas_access_service import (
    require_active_tenant,
    require_tenant_permission,
)
from valuecell.server.services.rule_strategy_export_service import (
    XLSX_MEDIA_TYPE,
    RuleStrategyExportService,
)
from valuecell.server.services.rule_strategy_templates import (
    get_rule_strategy_template,
    list_rule_strategy_templates,
)
from valuecell.server.services.rule_strategy_validation_service import (
    RuleStrategyValidationCoverageError,
    RuleStrategyValidationDataMaterializer,
    RuleStrategyValidationNotCompletedError,
    RuleStrategyValidationNotFoundError,
    RuleStrategyValidationService,
    RuleStrategyValidationWindowError,
)
from valuecell.server.services.rule_strategy_validation_export_service import (
    RuleStrategyValidationExportService,
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


class RuleStrategyTextImportJobRequest(RuleStrategyTextImportRequest):
    """Idempotent background strategy compilation request."""

    request_id: UUID



class RuleStrategyTemplateInstantiateRequest(BaseModel):
    """Explicit inputs for cloning a code-owned template into a tenant strategy."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    initial_capital_quote: float = Field(gt=0, le=100_000_000)
    symbol_candidates: list[str] = Field(min_length=1, max_length=100)
    execution_scope: Literal[
        "paper_virtual", "dedicated_credential", "dedicated_subaccount"
    ] = "paper_virtual"
    credential_id: str | None = Field(default=None, min_length=1, max_length=36)

def create_rule_strategy_router(
    service: RuleStrategyService | None = None,
    validation_materializer: RuleStrategyValidationDataMaterializer | None = None,
) -> APIRouter:
    """Create the deterministic strategy API router with isolated Demo execution validation."""

    router = APIRouter(prefix="/rule-strategies", tags=["rule-strategies"])
    rule_service = service or RuleStrategyService()
    advisory_service = RuleStrategyAdvisoryService()
    validation_service = RuleStrategyValidationService()
    validation_export_service = RuleStrategyValidationExportService(validation_service)
    text_import_jobs = get_rule_strategy_text_import_job_service()
    export_service = RuleStrategyExportService(rule_service)

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
        raise HTTPException(
            status_code=410,
            detail="Use /parse-strategy-text/jobs for background strategy parsing",
        )

    @router.post(
        "/parse-strategy-text/jobs",
        response_model=SuccessResponse[RuleStrategyTextImportJob],
        status_code=202,
    )
    async def submit_strategy_text_import(
        request: RuleStrategyTextImportJobRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[RuleStrategyTextImportJob]:
        require_strategy_manage(principal)
        try:
            data = await text_import_jobs.submit_async(
                request.strategy_text,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                request_id=str(request.request_id),
            )
        except RuleStrategyTextImportJobCapacityError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except RuleStrategyTextImportJobConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Strategy text parsing started")

    @router.get(
        "/parse-strategy-text/jobs/{job_id}",
        response_model=SuccessResponse[RuleStrategyTextImportJob],
    )
    async def get_strategy_text_import_job(
        job_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[RuleStrategyTextImportJob]:
        require_strategy_manage(principal)
        try:
            data = await text_import_jobs.get_async(
                job_id,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )
        except RuleStrategyTextImportJobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Strategy text parsing status")

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
        include_archived: bool = Query(default=False),
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[list[dict[str, Any]]]:
        require_strategy_read(principal)
        return SuccessResponse.create(
            data=rule_service.list(
                principal.tenant_id, include_archived=include_archived
            ),
            msg="Rule strategies retrieved",
        )

    @router.get("/summary", response_model=SuccessResponse[list[dict[str, Any]]])
    async def get_rule_strategy_summary(
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[list[dict[str, Any]]]:
        require_strategy_read(principal)
        summary_reader = getattr(rule_service.repository, "summaries", None)
        data = summary_reader(principal.tenant_id) if summary_reader is not None else []
        return SuccessResponse.create(data=data, msg="Rule strategy summaries retrieved")

    @router.post(
        "/{strategy_id}/validations",
        response_model=SuccessResponse[dict[str, Any]],
        status_code=202,
    )
    async def create_rule_strategy_validation(
        strategy_id: str,
        request: RuleStrategyValidationCreateRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict[str, Any]]:
        require_strategy_manage(principal)
        if validation_materializer is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "validation_materialized_source_required",
                    "detail": "服务器尚未配置可证明完整覆盖的数据物化器。",
                },
            )
        try:
            detail = validation_service.submit_materialized(
                strategy_id,
                principal.tenant_id,
                request,
                validation_materializer,
            )
        except RuleStrategyValidationNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuleStrategyValidationWindowError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuleStrategyValidationCoverageError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": exc.code, "detail": str(exc), "report": exc.report},
            ) from exc
        return SuccessResponse.create(
            data=detail.model_dump(mode="json"), msg="Strategy validation queued"
        )

    @router.get(
        "/{strategy_id}/validations",
        response_model=SuccessResponse[list[dict[str, Any]]],
    )
    async def list_rule_strategy_validations(
        strategy_id: str,
        limit: int = Query(default=100, ge=1, le=1_000),
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[list[dict[str, Any]]]:
        require_strategy_read(principal)
        try:
            runs = validation_service.list(strategy_id, principal.tenant_id, limit=limit)
        except RuleStrategyValidationNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SuccessResponse.create(
            data=[run.model_dump(mode="json") for run in runs],
            msg="Strategy validation runs retrieved",
        )

    @router.get(
        "/{strategy_id}/validations/{run_id}",
        response_model=SuccessResponse[dict[str, Any]],
    )
    async def get_rule_strategy_validation(
        strategy_id: str,
        run_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict[str, Any]]:
        require_strategy_read(principal)
        try:
            detail = validation_service.get(run_id, principal.tenant_id)
        except RuleStrategyValidationNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if detail.strategy_id != strategy_id:
            raise HTTPException(status_code=404, detail="Validation run was not found")
        return SuccessResponse.create(
            data=detail.model_dump(mode="json"), msg="Strategy validation retrieved"
        )

    @router.post(
        "/{strategy_id}/validations/{run_id}/cancel",
        response_model=SuccessResponse[dict[str, Any]],
    )
    async def cancel_rule_strategy_validation(
        strategy_id: str,
        run_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict[str, Any]]:
        require_strategy_manage(principal)
        try:
            detail = validation_service.cancel(run_id, principal.tenant_id)
        except RuleStrategyValidationNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if detail.strategy_id != strategy_id:
            raise HTTPException(status_code=404, detail="Validation run was not found")
        return SuccessResponse.create(
            data=detail.model_dump(mode="json"), msg="Strategy validation cancellation requested"
        )

    @router.get(
        "/{strategy_id}/validations/{run_id}/export",
        responses={200: {"content": {XLSX_MEDIA_TYPE: {}}}},
    )
    async def export_rule_strategy_validation(
        strategy_id: str,
        run_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> Response:
        require_strategy_read(principal)
        try:
            detail = validation_service.get(run_id, principal.tenant_id)
            if detail.strategy_id != strategy_id:
                raise RuleStrategyValidationNotFoundError("Validation run was not found")
            workbook, filename = validation_export_service.build(
                run_id, principal.tenant_id
            )
        except RuleStrategyValidationNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuleStrategyValidationNotCompletedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return Response(
            content=workbook,
            media_type=XLSX_MEDIA_TYPE,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
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


    @router.get(
        "/{strategy_id}/monitor-state",
        response_model=SuccessResponse[list[dict[str, Any]]],
    )
    async def get_rule_strategy_monitor_state(
        strategy_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[list[dict[str, Any]]]:
        require_strategy_read(principal)
        try:
            rule_service.get(strategy_id, principal.tenant_id)
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        rows = rule_service.repository.monitors(strategy_id, principal.tenant_id)
        return SuccessResponse.create(
            data=[
                {
                    "symbol": row.symbol,
                    "state": row.state,
                    "reason_code": row.reason_code,
                    "reason_detail": row.reason_detail,
                    "evaluated_at": row.evaluated_at,
                    "next_check_at": row.next_check_at,
                    "protected_held": row.protected_held,
                }
                for row in rows
            ],
            msg="Strategy monitor state retrieved",
        )

    @router.get(
        "/{strategy_id}/risk-state",
        response_model=SuccessResponse[dict[str, Any]],
    )
    async def get_rule_strategy_risk_state(
        strategy_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict[str, Any]]:
        require_strategy_read(principal)
        try:
            rule_service.get(strategy_id, principal.tenant_id)
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        current_state = rule_service.repository.get_account_state(
            strategy_id, principal.tenant_id
        )
        if current_state is None:
            raise HTTPException(status_code=404, detail="Strategy risk state was not found")
        _account, risk = current_state
        return SuccessResponse.create(
            data={
                "state": risk.state,
                "daily_equity_baseline": risk.daily_equity_baseline,
                "high_water_equity": risk.high_water_equity,
                "current_drawdown_pct": risk.current_drawdown_pct,
                "cooldown_until": risk.cooldown_until,
                "reason_code": risk.reason_code,
                "reason_detail": risk.reason_detail,
            },
            msg="Strategy risk state retrieved",
        )
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
        except RuleStrategyRunningUpdateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Paper rule strategy updated")

    @router.delete("/{strategy_id}", response_model=SuccessResponse[dict[str, Any]])
    async def delete_rule_strategy(
        strategy_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict[str, Any]]:
        require_strategy_manage(principal)
        try:
            archived = rule_service.delete(strategy_id, principal.tenant_id)
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuleStrategyDeleteConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return SuccessResponse.create(
            data={"strategy_id": strategy_id, "archived": archived},
            msg="策略已安全归档" if archived else "策略已删除",
        )

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
        except RuleStrategyStartAdmissionError as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": exc.reason_code, "detail": exc.detail},
            ) from exc
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
        "/{strategy_id}/export",
        responses={200: {"content": {XLSX_MEDIA_TYPE: {}}}},
    )
    async def export_rule_strategy(
        strategy_id: str,
        from_date: date | None = Query(default=None),
        to_date: date | None = Query(default=None),
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> Response:
        """Download complete, tenant-scoped strategy history as an XLSX workbook."""
        require_strategy_read(principal)
        if from_date is not None and to_date is not None and from_date > to_date:
            raise HTTPException(
                status_code=422,
                detail="from_date must be on or before to_date",
            )
        try:
            workbook, filename = export_service.build(
                strategy_id,
                principal.tenant_id,
                from_date,
                to_date,
            )
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(
            content=workbook,
            media_type=XLSX_MEDIA_TYPE,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
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
        # Verify strategy exists, is tenant-scoped, and supplies the immutable
        # capital and timestamp for the curve's baseline.
        try:
            strategy = rule_service.get(strategy_id, principal.tenant_id)
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        initial_capital = float(strategy["config"]["initial_capital_quote"])
        created_at = strategy["created_at"]
        created_at_str = (
            created_at.strftime("%Y-%m-%dT%H:%M:%SZ")
            if created_at is not None
            else ""
        )
        points: list[dict[str, Any]] = [
            {
                "ts": created_at_str,
                "cumulative_pnl": 0.0,
                "equity_quote": initial_capital,
                "action": "initial",
            }
        ]
        journals = list(
            reversed(
                rule_service.repository.get_evaluations(
                    strategy_id, principal.tenant_id, limit=500
                )
            )
        )
        for journal in journals:
            result: dict[str, Any] = journal.result or {}
            raw_account = result.get("account")
            if (
                not isinstance(raw_account, dict)
                or raw_account.get("source") == "okx_demo"
                or "initial_capital_quote" not in raw_account
                or "equity_quote" not in raw_account
            ):
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


def create_rule_strategy_template_router(
    service: RuleStrategyService | None = None,
) -> APIRouter:
    """Expose immutable code-owned templates outside the strategy CRUD prefix."""
    router = APIRouter(prefix="/rule-strategy-templates", tags=["rule-strategy-templates"])
    rule_service = service or RuleStrategyService()

    def require_template_read(principal: CurrentPrincipal) -> None:
        require_active_tenant(principal)
        require_tenant_permission(principal, "tenant.read")

    def require_template_manage(principal: CurrentPrincipal) -> None:
        require_active_tenant(principal)
        require_tenant_permission(principal, "strategy.manage")

    @router.get("", response_model=SuccessResponse[list[dict[str, Any]]])
    async def list_templates(
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[list[dict[str, Any]]]:
        require_template_read(principal)
        return SuccessResponse.create(
            data=[
                {
                    "template_id": template.template_id,
                    "template_version": template.template_version,
                    "display_name": template.display_name,
                    "execution_mode": template.execution_mode,
                    "config": template.config.model_dump(mode="json"),
                }
                for template in list_rule_strategy_templates()
            ],
            msg="Rule strategy templates retrieved",
        )

    @router.post(
        "/{template_id}/instantiate",
        response_model=SuccessResponse[dict[str, Any]],
        status_code=201,
    )
    async def instantiate_template(
        template_id: str,
        request: RuleStrategyTemplateInstantiateRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db=Depends(get_db),
    ) -> SuccessResponse[dict[str, Any]]:
        require_template_manage(principal)
        if get_rule_strategy_template(template_id) is None:
            raise HTTPException(status_code=404, detail="Unknown rule strategy template")
        if request.execution_scope != "paper_virtual":
            connection_id = request.credential_id or ""
            credential = db.query(TenantCredential).filter_by(
                id=connection_id,
                tenant_id=principal.tenant_id,
                revoked=False,
            ).first()
            metadata = credential.metadata_json if credential is not None else {}
            eligible_scope = metadata.get("strategy_account_scope") or metadata.get(
                "account_scope"
            )
            if (
                credential is None
                or credential.kind != "exchange"
                or credential.provider != "okx"
                or metadata.get("sandbox") is not True
                or metadata.get("market_type") != "spot"
                or eligible_scope != request.execution_scope
            ):
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": "dedicated_strategy_account_required",
                        "reason_code": "shared_exchange_account_requires_dedicated_scope",
                    },
                )
        try:
            data = rule_service.instantiate_template(
                principal.tenant_id,
                template_id=template_id,
                name=request.name,
                initial_capital_quote=request.initial_capital_quote,
                symbol_candidates=request.symbol_candidates,
                scope=request.execution_scope,
                credential_id=request.credential_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Rule strategy template instantiated")

    return router
