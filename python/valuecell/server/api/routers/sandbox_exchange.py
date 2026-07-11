"""Tenant-scoped API for explicitly requested Binance Testnet and OKX Demo orders."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valuecell.server.api.auth import CurrentPrincipal, get_current_principal
from valuecell.server.api.schemas.base import SuccessResponse
from valuecell.server.db.connection import get_db
from valuecell.server.services.sandbox_exchange_service import SandboxExchangeService
from valuecell.server.services.sandbox_exchange_trading_service import (
    SandboxExchangeTradingService,
    SandboxTradingError,
)
from valuecell.server.services.tenant_credential_service import CredentialVaultError, TenantCredentialService


class SandboxConnectionRequest(BaseModel):
    """Ephemeral sandbox API credentials; fields are never echoed."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["binance", "okx"]
    api_key: str = Field(min_length=1, max_length=512)
    api_secret: str = Field(min_length=1, max_length=512)
    passphrase: str | None = Field(default=None, min_length=1, max_length=512)
    label: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def require_okx_passphrase(self) -> "SandboxConnectionRequest":
        if self.provider == "okx" and self.passphrase is None:
            raise ValueError("OKX sandbox connection requires a passphrase")
        return self


class SandboxOrderRequest(BaseModel):
    """Bounded, explicit sandbox spot order request."""

    model_config = ConfigDict(extra="forbid")

    credential_id: str = Field(min_length=1, max_length=36)
    symbol: str = Field(pattern=r"^[A-Z0-9]+/USDT$", min_length=6, max_length=32)
    side: Literal["buy", "sell"]
    type: Literal["market", "limit"]
    quote_amount: Decimal = Field(gt=0, le=Decimal("10000"))
    price: Decimal | None = Field(default=None, gt=0)
    idempotency_key: str | None = Field(default=None, min_length=16, max_length=128)
    sandbox: Literal[True]

    @model_validator(mode="after")
    def require_limit_price(self) -> "SandboxOrderRequest":
        if self.type == "limit" and self.price is None:
            raise ValueError("Limit orders require a price")
        return self


def create_sandbox_exchange_router() -> APIRouter:
    """Create routes deliberately isolated from the paper-only strategy scheduler."""
    router = APIRouter(prefix="/saas/sandbox-exchanges", tags=["sandbox-exchanges"])

    @router.post("/connections", response_model=SuccessResponse[dict], status_code=201)
    async def create_connection(
        request: SandboxConnectionRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict]:
        validation = await SandboxExchangeService().validate(request.provider, request.api_key, request.api_secret, request.passphrase)
        if not validation["validated"]:
            raise HTTPException(status_code=422, detail="Sandbox credentials could not be validated")
        secret = {"api_key": request.api_key, "api_secret": request.api_secret}
        if request.passphrase is not None:
            secret["passphrase"] = request.passphrase
        metadata = {"sandbox": True, "provider": request.provider, "market_type": "spot", "validated_at": datetime.now(timezone.utc).isoformat()}
        try:
            data = TenantCredentialService(db).create(principal.tenant_id, principal.user_id, "exchange", request.provider, request.label, secret, metadata)
        except (CredentialVaultError, IntegrityError) as exc:
            db.rollback()
            raise HTTPException(status_code=422, detail="Sandbox connection label already exists or is invalid") from exc
        return SuccessResponse.create(data=data, msg="Sandbox connection saved")

    @router.get("/connections", response_model=SuccessResponse[list[dict]])
    async def list_connections(
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[list[dict]]:
        return SuccessResponse.create(data=SandboxExchangeTradingService(db).connection_metadata(principal.tenant_id), msg="Sandbox connection metadata retrieved")

    @router.get("/connections/{credential_id}/balance", response_model=SuccessResponse[dict])
    async def connection_balance(
        credential_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict]:
        try:
            data = await SandboxExchangeTradingService(db).balance(principal.tenant_id, credential_id)
        except SandboxTradingError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Sandbox balance retrieved")

    @router.post("/orders", response_model=SuccessResponse[dict], status_code=201)
    async def create_order(
        request: SandboxOrderRequest,
        idempotency_header: str | None = Header(default=None, alias="Idempotency-Key", min_length=16, max_length=128),
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict]:
        if request.idempotency_key and idempotency_header and request.idempotency_key != idempotency_header:
            raise HTTPException(status_code=422, detail="Idempotency key header and body must match")
        client_order_id = idempotency_header or request.idempotency_key
        if client_order_id is None:
            raise HTTPException(status_code=422, detail="An idempotency key is required")
        try:
            data = await SandboxExchangeTradingService(db).submit_order(principal.tenant_id, request.credential_id, client_order_id, request.symbol, request.side, request.type, request.quote_amount, request.price)
        except SandboxTradingError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Sandbox order recorded")

    @router.get("/orders/{order_id}/status", response_model=SuccessResponse[dict])
    async def order_status(
        order_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict]:
        try:
            data = await SandboxExchangeTradingService(db).fetch_order_status(principal.tenant_id, order_id)
        except SandboxTradingError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Sandbox order status retrieved")

    @router.get("/orders", response_model=SuccessResponse[list[dict]])
    async def list_orders(
        credential_id: str | None = None,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[list[dict]]:
        data = SandboxExchangeTradingService(db).list_orders(principal.tenant_id, credential_id)
        return SuccessResponse.create(data=data, msg="Sandbox orders retrieved")

    return router
