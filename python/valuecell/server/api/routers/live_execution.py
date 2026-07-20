"""Default-disabled, authenticated live execution configuration routes."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.orm import Session

from valuecell.server.api.auth import CurrentPrincipal, get_current_principal
from valuecell.server.api.schemas.base import SuccessResponse
from valuecell.server.config.settings import get_settings
from valuecell.server.db.connection import get_db
from valuecell.server.services.live_execution_authorization import (
    live_authorization_manager,
)
from valuecell.server.services.live_execution_service import (
    LiveExecutionError,
    LiveExecutionService,
)
from valuecell.server.services.saas_access_service import (
    require_active_tenant,
    require_tenant_permission,
)


class LiveConnectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1, max_length=200)
    provider: Literal["binance", "okx"]
    market_type: Literal["spot", "swap"]
    api_key: str = Field(min_length=1, max_length=512)
    api_secret: str = Field(min_length=1, max_length=512)
    passphrase: str | None = Field(default=None, min_length=1, max_length=512)
    withdrawal_disabled_confirmed: Literal[True]
    ip_allowlist_confirmed: Literal[True]

    @model_validator(mode="after")
    def check_okx_passphrase(self) -> "LiveConnectionRequest":
        if self.provider == "okx" and self.passphrase is None:
            raise ValueError("OKX 实盘连接需要 Passphrase")
        return self


class RiskPolicyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_order_notional: Decimal = Field(gt=0)
    max_open_positions: int = Field(ge=1, le=100)
    max_leverage: Decimal = Field(ge=1, le=125)
    max_total_notional: Decimal | None = Field(default=None, gt=0)
    max_daily_loss: Decimal | None = Field(default=None, gt=0)
    allowed_symbols: list[str] = Field(min_length=1, max_length=50)


class BindingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy_id: str = Field(min_length=1, max_length=100)
    connection_id: str = Field(min_length=1, max_length=36)


class ChallengeConfirmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    challenge_code: str = Field(min_length=16, max_length=128)


class LiveOrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    connection_id: str = Field(min_length=1, max_length=36)
    symbol: str = Field(min_length=6, max_length=32)
    side: Literal["buy", "sell"]
    type: Literal["market", "limit"]
    quote_amount: Decimal = Field(gt=0)
    price: Decimal | None = Field(default=None, gt=0)
    idempotency_key: str = Field(min_length=16, max_length=128)

    @model_validator(mode="after")
    def check_limit_price(self) -> "LiveOrderRequest":
        if self.type == "limit" and self.price is None:
            raise ValueError("限价单需要价格")
        return self


def create_live_execution_router() -> APIRouter:
    """Create live-execution routes. Strategy scheduler remains paper-only."""
    router = APIRouter(prefix="/saas/live-execution", tags=["live-execution"])

    def service(db: Session) -> LiveExecutionService:
        return LiveExecutionService(db)

    def require_live_read(principal: CurrentPrincipal) -> None:
        require_active_tenant(principal)
        require_tenant_permission(principal, "tenant.read")

    def require_live_manage(principal: CurrentPrincipal) -> None:
        require_active_tenant(principal)
        require_tenant_permission(principal, "trade.execute")

    def require_connection_manage(principal: CurrentPrincipal) -> None:
        require_active_tenant(principal)
        require_tenant_permission(principal, "connection.manage")

    @router.get("/status", response_model=SuccessResponse[dict])
    async def status(
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict]:
        require_live_read(principal)
        return SuccessResponse.create(
            data=service(db).status(principal.tenant_id),
            msg="Live execution status retrieved",
        )

    @router.get("/connections", response_model=SuccessResponse[list[dict]])
    async def connections(
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[list[dict]]:
        require_live_read(principal)
        return SuccessResponse.create(
            data=service(db).connections(principal.tenant_id),
            msg="Live connections retrieved",
        )

    @router.post("/connections", response_model=SuccessResponse[dict], status_code=201)
    async def create_connection(
        request: LiveConnectionRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict]:
        require_connection_manage(principal)
        try:
            data = await service(db).create_connection(
                principal.tenant_id,
                principal.user_id,
                request.provider,
                request.market_type,
                request.label,
                request.api_key,
                request.api_secret,
                request.passphrase,
                request.withdrawal_disabled_confirmed,
                request.ip_allowlist_confirmed,
            )
        except LiveExecutionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Live connection saved")

    @router.get("/risk-policies", response_model=SuccessResponse[dict | None])
    async def risk_policies(
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict | None]:
        require_live_read(principal)
        return SuccessResponse.create(
            data=service(db).policy_data_or_none(principal.tenant_id),
            msg="Live risk policy retrieved",
        )

    @router.post(
        "/risk-policies", response_model=SuccessResponse[dict], status_code=201
    )
    async def save_risk_policy(
        request: RiskPolicyRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict]:
        require_connection_manage(principal)
        try:
            data = service(db).save_policy(
                principal.tenant_id,
                request.max_order_notional,
                request.max_open_positions,
                request.max_leverage,
                request.allowed_symbols,
                request.max_total_notional,
                request.max_daily_loss,
            )
        except LiveExecutionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Live risk policy saved")

    @router.get("/bindings", response_model=SuccessResponse[list[dict]])
    async def bindings(
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[list[dict]]:
        require_live_read(principal)
        return SuccessResponse.create(
            data=service(db).bindings(principal.tenant_id),
            msg="Live strategy bindings retrieved",
        )

    @router.post("/bindings", response_model=SuccessResponse[dict], status_code=201)
    async def create_binding(
        request: BindingRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict]:
        require_connection_manage(principal)
        try:
            data = service(db).create_binding(
                principal.tenant_id, request.strategy_id, request.connection_id
            )
        except LiveExecutionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Live strategy binding saved")

    @router.post("/bindings/{binding_id}/revoke", response_model=SuccessResponse[dict])
    async def revoke_binding(
        binding_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict]:
        require_connection_manage(principal)
        try:
            service(db).revoke_binding(principal.tenant_id, binding_id)
        except LiveExecutionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SuccessResponse.create(
            data={"id": binding_id, "active": False},
            msg="Live strategy binding revoked",
        )

    @router.post(
        "/startup-authorization/challenge", response_model=SuccessResponse[dict]
    )
    async def challenge(
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict]:
        require_live_manage(principal)
        data = live_authorization_manager.issue_challenge(
            principal.tenant_id,
            principal.user_id,
            get_settings().LIVE_AUTHORIZATION_TTL_S,
        )
        return SuccessResponse.create(
            data=data, msg="Live authorization challenge issued"
        )

    @router.post("/startup-authorization/confirm", response_model=SuccessResponse[dict])
    async def confirm(
        request: ChallengeConfirmRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict]:
        require_live_manage(principal)
        if principal.role not in {"owner", "admin"} and not principal.is_platform_admin:
            raise HTTPException(status_code=403, detail="实盘授权确认仅限租户管理员")
        expires_at = live_authorization_manager.confirm(
            principal.tenant_id,
            principal.user_id,
            request.challenge_code,
            get_settings().LIVE_AUTHORIZATION_TTL_S,
        )
        if expires_at is None:
            raise HTTPException(status_code=422, detail="实盘授权挑战码无效或已过期")
        return SuccessResponse.create(
            data={"authorization_expires_at": expires_at.isoformat()},
            msg="Live authorization active",
        )

    @router.post("/startup-authorization/revoke", response_model=SuccessResponse[dict])
    async def revoke_authorization(
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict]:
        require_live_manage(principal)
        live_authorization_manager.revoke(principal.tenant_id)
        return SuccessResponse.create(
            data={"authorization_active": False}, msg="Live authorization revoked"
        )

    @router.post("/orders", response_model=SuccessResponse[dict], status_code=201)
    async def create_order(
        request: LiveOrderRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict]:
        require_live_manage(principal)
        try:
            data = await service(db).submit_order(
                principal.tenant_id,
                request.connection_id,
                request.idempotency_key,
                request.symbol,
                request.side,
                request.type,
                request.quote_amount,
                request.price,
            )
        except LiveExecutionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Live order recorded")

    @router.get("/orders", response_model=SuccessResponse[list[dict]])
    async def orders(
        connection_id: str | None = None,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[list[dict]]:
        require_live_read(principal)
        return SuccessResponse.create(
            data=service(db).orders(principal.tenant_id, connection_id),
            msg="Live orders retrieved",
        )

    @router.post("/orders/{order_id}/refresh", response_model=SuccessResponse[dict])
    async def refresh_order(
        order_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict]:
        require_live_manage(principal)
        try:
            data = await service(db).refresh_order(principal.tenant_id, order_id)
        except LiveExecutionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Live order refreshed")

    @router.get(
        "/connections/{connection_id}/positions",
        response_model=SuccessResponse[list[dict]],
    )
    async def positions(
        connection_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[list[dict]]:
        require_live_read(principal)
        try:
            data = await service(db).positions(principal.tenant_id, connection_id)
        except LiveExecutionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Live positions retrieved")

    return router
