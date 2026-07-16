"""Local-development SaaS registration and login routes."""

from __future__ import annotations

import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.orm import Session

from valuecell.server.api.auth import (
    CurrentPrincipal,
    create_access_token,
    get_current_principal,
    hash_password,
    verify_password,
)
from valuecell.server.api.schemas.base import SuccessResponse
from valuecell.server.db.connection import get_db
from valuecell.server.db.models.tenant import (
    SaaSUser,
    Tenant,
    TenantMembership,
    TenantProfile,
)


class RegisterRequest(BaseModel):
    """Registration input for a personal or enterprise tenant workspace."""

    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=12, max_length=256)
    tenant_type: Literal["personal", "enterprise"] = "personal"
    workspace_name: str = Field(min_length=1, max_length=200)
    organization_name: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def require_enterprise_organization(self) -> "RegisterRequest":
        if self.tenant_type == "enterprise" and not self.organization_name:
            raise ValueError("enterprise registration requires organization_name")
        return self


class LoginRequest(BaseModel):
    """Local MVP login input."""

    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=256)


class WorkspaceSwitchRequest(BaseModel):
    """Select one of the caller's tenant workspaces for a new token."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(min_length=1, max_length=36)


def create_saas_auth_router() -> APIRouter:
    """Create local MVP SaaS authentication endpoints."""
    router = APIRouter(prefix="/saas/auth", tags=["saas-auth"])

    @router.post("/register", response_model=SuccessResponse[dict], status_code=201)
    async def register(request: RegisterRequest, db: Session = Depends(get_db)):
        email = request.email.strip().lower()
        if db.query(SaaSUser).filter(SaaSUser.email == email).first() is not None:
            raise HTTPException(status_code=409, detail="Email is already registered")
        user = SaaSUser(
            id=str(uuid.uuid4()),
            email=email,
            password_hash=hash_password(request.password),
        )
        tenant = Tenant(id=str(uuid.uuid4()), name=request.workspace_name.strip())
        profile = TenantProfile(
            tenant_id=tenant.id,
            tenant_type=request.tenant_type,
            organization_name=request.organization_name.strip()
            if request.organization_name
            else None,
        )
        membership = TenantMembership(
            tenant_id=tenant.id, user_id=user.id, role="owner"
        )
        db.add_all([user, tenant, profile, membership])
        db.commit()
        token = create_access_token(
            CurrentPrincipal(user_id=user.id, tenant_id=tenant.id)
        )
        return SuccessResponse.create(
            data={
                "access_token": token,
                "token_type": "bearer",
                "tenant_id": tenant.id,
                "user_id": user.id,
                "email": user.email,
                "tenant_type": profile.tenant_type,
                "organization_name": profile.organization_name,
            },
            msg="SaaS workspace registered",
        )

    @router.post("/login", response_model=SuccessResponse[dict])
    async def login(request: LoginRequest, db: Session = Depends(get_db)):
        email = request.email.strip().lower()
        user = db.query(SaaSUser).filter(SaaSUser.email == email).first()
        if user is None or not verify_password(request.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        membership = (
            db.query(TenantMembership)
            .filter(TenantMembership.user_id == user.id)
            .order_by(TenantMembership.created_at.asc())
            .first()
        )
        if membership is None:
            raise HTTPException(
                status_code=403, detail="No active workspace membership"
            )
        token = create_access_token(
            CurrentPrincipal(user_id=user.id, tenant_id=membership.tenant_id)
        )
        return SuccessResponse.create(
            data={
                "access_token": token,
                "token_type": "bearer",
                "tenant_id": membership.tenant_id,
                "user_id": user.id,
                "email": user.email,
            },
            msg="SaaS login successful",
        )

    @router.get("/me", response_model=SuccessResponse[dict])
    async def current_user(
        principal: CurrentPrincipal = Depends(get_current_principal),
    ):
        return SuccessResponse.create(
            data={
                "user_id": principal.user_id,
                "tenant_id": principal.tenant_id,
                "role": principal.role,
                "is_platform_admin": principal.is_platform_admin,
                "access_status": principal.access_status,
                "commercial_model": principal.commercial_model,
                "access_expires_at": principal.access_expires_at,
            },
            msg="SaaS principal retrieved",
        )

    @router.get("/workspaces", response_model=SuccessResponse[list[dict]])
    async def workspaces(
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[list[dict]]:
        rows = (
            db.query(TenantMembership, Tenant, TenantProfile)
            .join(Tenant, Tenant.id == TenantMembership.tenant_id)
            .outerjoin(TenantProfile, TenantProfile.tenant_id == Tenant.id)
            .filter(TenantMembership.user_id == principal.user_id)
            .order_by(TenantMembership.created_at.asc())
            .all()
        )
        return SuccessResponse.create(
            data=[
                {
                    "tenant_id": membership.tenant_id,
                    "name": tenant.name,
                    "tenant_type": profile.tenant_type if profile else "personal",
                    "organization_name": profile.organization_name if profile else None,
                    "role": membership.role,
                    "selected": membership.tenant_id == principal.tenant_id,
                }
                for membership, tenant, profile in rows
            ],
            msg="SaaS workspaces retrieved",
        )

    @router.post("/switch", response_model=SuccessResponse[dict])
    async def switch_workspace(
        request: WorkspaceSwitchRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict]:
        membership = (
            db.query(TenantMembership)
            .filter(
                TenantMembership.user_id == principal.user_id,
                TenantMembership.tenant_id == request.tenant_id,
            )
            .first()
        )
        if membership is None:
            raise HTTPException(
                status_code=404, detail="Workspace membership was not found"
            )
        user = db.query(SaaSUser).filter(SaaSUser.id == principal.user_id).one()
        token = create_access_token(
            CurrentPrincipal(user_id=user.id, tenant_id=membership.tenant_id)
        )
        return SuccessResponse.create(
            data={
                "access_token": token,
                "token_type": "bearer",
                "tenant_id": membership.tenant_id,
                "user_id": user.id,
                "email": user.email,
            },
            msg="SaaS workspace switched",
        )

    return router
