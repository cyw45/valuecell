"""Standalone API for persisted, deterministic paper rule strategies."""

from __future__ import annotations
from datetime import date, datetime, timezone

from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from valuecell.server.config.settings import get_settings

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
    build_demo_execution_read_model,
    build_strategy_daily_pnl_curve,
)
from valuecell.server.services.rule_strategy_pnl_service import (
    build_daily_pnl_points,
    observation_from_journal,
)
from valuecell.server.services.rule_strategy_demo_snapshot_service import (
    build_demo_daily_curve,
    get_demo_account_sync_state,
    get_latest_demo_account_snapshot,
    get_official_test_baseline,
    list_demo_account_snapshots,
)
from valuecell.server.services.rule_strategy_manual_close_service import (
    ManualCloseError,
    execute_manual_close,
)
from valuecell.server.services.sandbox_exchange_trading_service import (
    SandboxExchangeTradingService,
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
from valuecell.server.services.multi_strategy_registry import (
    fixed_strategy_definitions,
    strategy_code_fingerprint,
)
from valuecell.server.services.multi_strategy_account_summary import (
    SharedAccountSummaryUnavailable,
    shared_account_summary_dict,
)
from valuecell.server.services.rule_strategy_validation_service import (
    RuleStrategyValidationCoverageError,
    RuleStrategyValidationDataMaterializer,
    RuleStrategyValidationNotCompletedError,
    RuleStrategyValidationNotFoundError,
    RuleStrategyValidationService,
    RuleStrategyValidationWindowError,
)
from valuecell.server.services.multi_strategy_trade_facts import journal_trade_facts
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
class FixedStrategyCreateRequest(BaseModel):
    """Create a code-owned strategy without accepting algorithm parameters."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["dual_ma_trend", "pair_rotation", "leader_breakout"]
    name: str = Field(min_length=1, max_length=200)
    initial_capital_quote: float = Field(gt=0, le=100_000_000)
    environment: Literal["paper", "okx_demo"] = "paper"
    credential_id: str | None = Field(default=None, min_length=1, max_length=36)
class RuleStrategyManualCloseRequest(BaseModel):
    """Explicit, typed confirmation for one-symbol or all-position Demo close."""

    model_config = ConfigDict(extra="forbid")

    scope: Literal["symbol", "all"]
    symbol: str | None = Field(default=None, min_length=6, max_length=32)
    confirmation: str = Field(min_length=4, max_length=32)
    idempotency_key: str = Field(min_length=16, max_length=128)

    @field_validator("symbol", mode="before")
    @classmethod
    def canonicalize_symbol(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().upper().replace("-", "/")
        return value

    @model_validator(mode="after")
    def validate_confirmation(self) -> "RuleStrategyManualCloseRequest":
        expected = "确认平仓"
        if self.scope == "symbol" and self.symbol is None:
            raise ValueError("symbol is required for a symbol close")
        if self.scope == "all" and self.symbol is not None:
            raise ValueError("symbol is forbidden for an all-position close")
        if self.confirmation.strip() != expected:
            raise ValueError(f"confirmation must be exactly '{expected}'")
        return self


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

    @router.post("/fixed", response_model=SuccessResponse[dict[str, Any]], status_code=201)
    async def create_fixed_rule_strategy(
        request: FixedStrategyCreateRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict[str, Any]]:
        """Register a code-owned strategy instance without exposing its rules."""
        require_strategy_manage(principal)
        if request.environment == "okx_demo":
            credential = db.query(TenantCredential).filter_by(
                id=request.credential_id,
                tenant_id=principal.tenant_id,
                revoked=False,
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
                    detail={"code": "okx_demo_connection_invalid"},
                )
        try:
            data = rule_service.create_fixed(
                principal.tenant_id,
                kind=request.kind,
                name=request.name,
                initial_capital_quote=request.initial_capital_quote,
                environment=request.environment,
                credential_id=request.credential_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Fixed rule strategy created")

    @router.get("/shared-account-summary", response_model=SuccessResponse[dict[str, Any]])
    async def get_shared_account_summary(
        credential_id: str = Query(min_length=1, max_length=36),
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict[str, Any]]:
        """Return one OKX wallet summary plus strategy allocations."""
        require_strategy_read(principal)
        try:
            data = shared_account_summary_dict(
                db,
                tenant_id=principal.tenant_id,
                credential_id=credential_id,
            )
        except SharedAccountSummaryUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Shared account summary retrieved")

    @router.get("/all-trade-facts", response_model=SuccessResponse[list[dict[str, Any]]])
    async def get_all_trade_facts(
        limit: int = Query(default=100, ge=1, le=500),
        strategy_id: str | None = Query(default=None, max_length=100),
        batch_id: str | None = Query(default=None, max_length=36),
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[list[dict[str, Any]]]:
        """Return tenant-scoped attributed trade explanations across strategies."""
        require_strategy_read(principal)
        strategies = (
            [rule_service._require_strategy(strategy_id, principal.tenant_id)]
            if strategy_id
            else rule_service.repository.list(principal.tenant_id)
        )
        for strategy in strategies:
            effective_batch_id = batch_id or getattr(strategy, "current_batch_id", None)
            if effective_batch_id is None:
                continue
            journals = rule_service.repository.get_evaluations(
                strategy.strategy_id,
                principal.tenant_id,
                limit=limit,
                batch_id=effective_batch_id,
            )
            for journal in journals:
                facts.extend(journal_trade_facts(strategy, journal))
        facts.sort(key=lambda item: item.created_at, reverse=True)
        return SuccessResponse.create(
            data=[item.model_dump(mode="json") for item in facts[:limit]],
            msg="Unified trade facts retrieved",
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

    @router.get("/definitions", response_model=SuccessResponse[list[dict[str, Any]]])
    async def list_strategy_definitions(
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[list[dict[str, Any]]]:
        """List fixed strategy definitions without creating executable instances."""
        require_strategy_read(principal)
        data = [
            {
                **definition.model_dump(mode="json"),
                "code_fingerprint": strategy_code_fingerprint(definition.kind),
            }
            for definition in fixed_strategy_definitions()
        ]
        return SuccessResponse.create(data=data, msg="Fixed strategy definitions retrieved")

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

    @router.get(
        "/{strategy_id}/batches",
        response_model=SuccessResponse[dict[str, Any]],
    )
    async def list_rule_strategy_batches(
        strategy_id: str,
        status: Literal["all", "running", "stopped", "archived"] = Query(default="all"),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
        from_datetime: datetime | None = Query(default=None),
        to_datetime: datetime | None = Query(default=None),
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict[str, Any]]:
        require_strategy_read(principal)
        if from_datetime and to_datetime and from_datetime >= to_datetime:
            raise HTTPException(status_code=422, detail="from_datetime must be before to_datetime")
        try:
            data = rule_service.batches(
                strategy_id,
                principal.tenant_id,
                status=status,
                page=page,
                page_size=page_size,
                from_datetime=from_datetime,
                to_datetime=to_datetime,
            )
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Execution batches retrieved")

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
                    "metadata_provider": row.metadata_provider,
                    "listing_first_tradable_at": row.listing_first_tradable_at,
                    "listing_age_days": row.listing_age_days,
                    "average_quote_volume_30d": row.average_quote_volume_30d,
                    "price_quote": row.price_quote,
                    "price_observed_at": row.price_observed_at,
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
        batch_id: str | None = Query(default=None),
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[list[dict[str, Any]]]:
        require_strategy_read(principal)
        try:
            data = rule_service.evaluations(
                strategy_id, principal.tenant_id, limit, batch_id=batch_id
            )
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
        batch_id: str | None = Query(default=None),
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
                batch_id,
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
        batch_id: str | None = Query(default=None),
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[list[dict[str, Any]]]:
        require_strategy_read(principal)
        # Verify strategy exists, is tenant-scoped, and supplies the immutable
        # capital and timestamp for the curve's baseline.
        try:
            strategy = rule_service.get(strategy_id, principal.tenant_id)
            resolve_batch = getattr(rule_service, "resolve_batch", None)
            batch_capable = callable(resolve_batch) and hasattr(
                rule_service.repository, "get_batch"
            )
            batch = None
            if batch_capable:
                assert resolve_batch is not None
                batch = resolve_batch(strategy_id, principal.tenant_id, batch_id)
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if batch_capable and batch is None:
            return SuccessResponse.create(data=[], msg="Daily PnL curve retrieved")
        selected_batch_id = getattr(batch, "batch_id", None)
        baseline_config = getattr(batch, "config_snapshot", None) or strategy["config"]
        initial_capital = float(baseline_config["initial_capital_quote"])
        export_reader = getattr(
            rule_service.repository, "get_evaluations_for_export", None
        )
        journals = (
            [
                journal
                for journal in export_reader(strategy_id, principal.tenant_id)
                if selected_batch_id is None
                or getattr(journal, "batch_id", None) == selected_batch_id
            ]
            if export_reader is not None
            else reversed(
                rule_service.repository.get_evaluations(
                    strategy_id,
                    principal.tenant_id,
                    limit=500,
                    **({"batch_id": selected_batch_id} if selected_batch_id else {}),
                )
            )
        )
        observations = [
            observation
            for journal in journals
            if (observation := observation_from_journal(journal)) is not None
        ]
        points = build_daily_pnl_points(
            initial_capital,
            getattr(batch, "started_at", None) or strategy["created_at"],
            observations,
        )
        return SuccessResponse.create(data=points, msg="Daily PnL curve retrieved")

    @router.get(
        "/{strategy_id}/demo-execution", response_model=SuccessResponse[dict[str, Any]]
    )
    async def get_rule_strategy_demo_execution(
        strategy_id: str,
        batch_id: str | None = Query(default=None),
        all_history: bool = Query(default=False),
        principal: CurrentPrincipal = Depends(get_current_principal),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=10, ge=1, le=100),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict[str, Any]]:
        """Return only the latest persisted Demo snapshot and local orders."""
        require_strategy_read(principal)
        try:
            strategy = rule_service.get(strategy_id, principal.tenant_id)
            resolve_batch = getattr(rule_service, "resolve_batch", None)
            batch_capable = callable(resolve_batch)
            if (
                batch_capable
                and batch_id is None
                and not all_history
                and strategy.get("status") != "running"
            ):
                empty_checked_at = datetime.now(timezone.utc).isoformat()
                return SuccessResponse.create(
                    data={
                        "source": "okx_demo_spot",
                        "strategy_id": strategy.get("strategy_id"),
                        "connection_id": None,
                        "account": {"scope": "exchange_connection_shared_account", "data": {"source": "okx_demo", "balances": [], "total_usdt_value": None, "checked_at": empty_checked_at}},
                        "positions": {"scope": "exchange_connection_shared_spot_positions", "data": {"source": "okx_demo", "positions": [], "checked_at": empty_checked_at}},
                        "strategy_positions": [],
                        "orders": [],
                        "trade_summary": {"total_orders": 0, "filled_orders": 0, "open_orders": 0},
                        "pnl": {"status": "unavailable", "reason_code": "no_current_execution_batch"},
                        "equity_curve": {"status": "unavailable", "points": [], "reason_code": "no_current_execution_batch"},
                        "checked_at": empty_checked_at,
                        "batch": None,
                        "sync": {"status": "unavailable", "observed_at": None, "freshness_age_s": None},
                        "wallet_equity_curve": {"status": "unavailable", "points": [], "reason_code": "no_current_execution_batch"},
                        "pagination": {"page": page, "page_size": page_size, "total_items": 0, "total_pages": 1},
                    },
                    msg="OKX Demo strategy execution snapshot retrieved",
                )
            batch = (
                None
                if all_history
                else resolve_batch(strategy_id, principal.tenant_id, batch_id)
                if batch_capable
                else None
            )
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        baseline = get_official_test_baseline(
            db, tenant_id=principal.tenant_id, strategy_id=strategy_id
        )
        selected_config = getattr(batch, "config_snapshot", None) or strategy.get("config") or {}
        execution = selected_config.get("execution") or {}
        connection_id = execution.get("sandbox_connection_id")
        if not isinstance(connection_id, str) or not connection_id:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "okx_demo_connection_unavailable",
                    "detail": "OKX Demo connection is unavailable.",
                },
            )
        snapshot = get_latest_demo_account_snapshot(
            db,
            tenant_id=principal.tenant_id,
            strategy_id=strategy_id,
            credential_id=connection_id,
            started_at=getattr(batch, "started_at", None),
            stopped_at=getattr(batch, "stopped_at", None),
        )
        if snapshot is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "demo_account_snapshot_pending",
                    "detail": "后台尚未完成 OKX Demo 账户同步，请稍后重试。",
                },
            )

        observed_at = snapshot.observed_at
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        account = {
            "source": snapshot.source,
            "total_usdt_value": snapshot.total_usdt_value,
            "balances": list(snapshot.balances or []),
            "checked_at": observed_at.isoformat(),
        }
        positions = {
            "source": snapshot.source,
            "positions": list(snapshot.positions or []),
            "checked_at": observed_at.isoformat(),
        }
        # list_orders reads local rows only; it does not refresh the exchange.
        orders = (
            SandboxExchangeTradingService(db).list_orders(
                principal.tenant_id,
                connection_id,
                strategy_id=strategy_id,
                batch_id=getattr(batch, "batch_id", None),
            )
            if all_history or batch is not None or not batch_capable
            else []
        )
        data = build_demo_execution_read_model(
            strategy,
            account,
            positions,
            orders,
            started_at=baseline.started_at if baseline is not None else None,
        )
        data["batch"] = rule_service._batch_data(batch) if batch is not None else None
        snapshots = list_demo_account_snapshots(
            db,
            tenant_id=principal.tenant_id,
            strategy_id=strategy_id,
            credential_id=connection_id,
            started_at=getattr(batch, "started_at", None),
            stopped_at=getattr(batch, "stopped_at", None),
        )
        curve_points = build_demo_daily_curve(
            snapshots,
            started_at=baseline.started_at if baseline is not None else None,
        )
        sync_state = get_demo_account_sync_state(
            db, tenant_id=principal.tenant_id, strategy_id=strategy_id
        )
        freshness_age_s = max(0, int((datetime.now(timezone.utc) - observed_at).total_seconds()))
        data["sync"] = {
            "status": "stale" if freshness_age_s > get_settings().DEMO_ACCOUNT_SYNC_INTERVAL_S * 2 else "healthy",
            "observed_at": observed_at.isoformat(),
            "freshness_age_s": freshness_age_s,
            "last_attempt_at": sync_state.last_attempt_at.isoformat() if sync_state and sync_state.last_attempt_at else None,
            "last_success_at": sync_state.last_success_at.isoformat() if sync_state and sync_state.last_success_at else None,
            "consecutive_failures": sync_state.consecutive_failures if sync_state else 0,
            "last_error_code": sync_state.last_error_code if sync_state else None,
        }
        strategy_curve_points = build_strategy_daily_pnl_curve(
            snapshots,
            data["orders"],
            started_at=baseline.started_at if baseline is not None else None,
        )
        data["equity_curve"] = {
            "status": "available" if strategy_curve_points else "unavailable",
            "scope": "strategy_attributed_persisted_wallet_marks",
            "reason_code": None if strategy_curve_points else "strategy_pnl_history_unavailable",
            "points": strategy_curve_points,
        }
        data["wallet_equity_curve"] = {
            "status": "available" if curve_points else "unavailable",
            "scope": "persisted_exchange_account_wallet_snapshots",
            "reason_code": None if curve_points else "no_wallet_snapshots",
            "points": curve_points,
        }
        all_orders = list(data.get("orders") or [])
        total_items = len(all_orders)
        total_pages = max(1, (total_items + page_size - 1) // page_size)
        start = (page - 1) * page_size
        data["orders"] = all_orders[start : start + page_size]
        data["pagination"] = {
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
        }
        return SuccessResponse.create(
            data=data, msg="OKX Demo strategy execution snapshot retrieved"
        )

    @router.post(
        "/{strategy_id}/manual-close",
        response_model=SuccessResponse[dict[str, Any]],
        status_code=202,
    )
    async def manual_close_strategy(
        strategy_id: str,
        request: RuleStrategyManualCloseRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict[str, Any]]:
        require_strategy_manage(principal)
        require_tenant_permission(principal, "trade.execute")
        try:
            strategy = rule_service.get(strategy_id, principal.tenant_id)
            data = await execute_manual_close(
                db,
                tenant_id=principal.tenant_id,
                requested_by=principal.user_id,
                strategy=strategy,
                scope=request.scope,
                symbol=request.symbol,
                idempotency_key=request.idempotency_key,
            )
        except RuleStrategyNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ManualCloseError as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": "manual_close_blocked", "detail": str(exc)},
            ) from exc
        return SuccessResponse.create(data=data, msg="Manual close command submitted")

    @router.get(
        "/{strategy_id}/account", response_model=SuccessResponse[dict[str, Any]]
    )
    async def get_rule_strategy_account(
        strategy_id: str,
        batch_id: str | None = Query(default=None),
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict[str, Any]]:
        require_strategy_read(principal)
        try:
            data = rule_service.account(
                strategy_id, principal.tenant_id, batch_id=batch_id
            )
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
        batch_id: str | None = Query(default=None),
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict[str, Any]]:
        require_strategy_read(principal)
        if log_type not in {"signals", "trades", "funding"}:
            raise HTTPException(status_code=404, detail="Log type was not found")
        try:
            data = rule_service.logs(
                strategy_id, principal.tenant_id, log_type, limit, batch_id=batch_id
            )
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
