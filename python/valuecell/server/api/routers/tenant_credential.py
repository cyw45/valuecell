"""Authenticated metadata-only tenant credential routes."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.orm import Session

from valuecell.server.api.auth import CurrentPrincipal, get_current_principal
from valuecell.server.api.schemas.base import SuccessResponse
from valuecell.server.db.connection import get_db
from valuecell.server.services.tenant_credential_service import (
    CredentialVaultError,
    TenantCredentialService,
)
from valuecell.server.services.sandbox_exchange_service import SandboxExchangeService
from valuecell.server.services.saas_access_service import (
    require_active_tenant,
    require_tenant_permission,
)


class CredentialCreateRequest(BaseModel):
    """Credential creation request; payload is never echoed in a response."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["exchange", "market_data"]
    provider: str = Field(min_length=1, max_length=100)
    label: str = Field(min_length=1, max_length=200)
    secret: dict[str, str] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SandboxCredentialValidationRequest(BaseModel):
    """Ephemeral sandbox exchange credentials used only for readiness validation."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["binance", "okx"]
    api_key: str = Field(min_length=1)
    api_secret: str = Field(min_length=1)
    passphrase: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_okx_passphrase(self) -> "SandboxCredentialValidationRequest":
        if self.provider == "okx" and self.passphrase is None:
            raise ValueError("OKX sandbox validation requires a passphrase")
        return self


def create_tenant_credential_router() -> APIRouter:
    """Create tenant-scoped vault metadata routes."""
    router = APIRouter(prefix="/saas/credentials", tags=["saas-credentials"])

    @router.post("", response_model=SuccessResponse[dict], status_code=201)
    async def create_credential(
        request: CredentialCreateRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict]:
        require_active_tenant(principal)
        require_tenant_permission(principal, "connection.manage")
        try:
            data = TenantCredentialService(db).create(
                principal.tenant_id,
                principal.user_id,
                request.kind,
                request.provider,
                request.label,
                request.secret,
                request.metadata,
            )
        except CredentialVaultError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Credential metadata created")

    @router.post("/sandbox/validate", response_model=SuccessResponse[dict])
    async def validate_sandbox_credential(
        request: SandboxCredentialValidationRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
    ) -> SuccessResponse[dict]:
        require_active_tenant(principal)
        require_tenant_permission(principal, "connection.manage")
        data = await SandboxExchangeService().validate(
            provider=request.provider,
            api_key=request.api_key,
            api_secret=request.api_secret,
            passphrase=request.passphrase,
        )
        return SuccessResponse.create(data=data, msg="Sandbox credentials checked")

    @router.get("", response_model=SuccessResponse[list[dict]])
    async def list_credentials(
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[list[dict]]:
        require_active_tenant(principal)
        require_tenant_permission(principal, "tenant.read")
        return SuccessResponse.create(
            data=TenantCredentialService(db).list(principal.tenant_id),
            msg="Credential metadata retrieved",
        )

    @router.post("/{credential_id}/revoke", response_model=SuccessResponse[dict])
    async def revoke_credential(
        credential_id: str,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict]:
        require_active_tenant(principal)
        require_tenant_permission(principal, "connection.manage")
        try:
            data = TenantCredentialService(db).revoke(
                principal.tenant_id, credential_id
            )
        except CredentialVaultError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SuccessResponse.create(data=data, msg="Credential revoked")

    return router
