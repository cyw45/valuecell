"""Local-development SaaS registration and login routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
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
from valuecell.server.db.models.tenant import SaaSUser, Tenant, TenantMembership


class RegisterRequest(BaseModel):
    """Local MVP account registration input."""

    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=12, max_length=256)
    workspace_name: str = Field(min_length=1, max_length=200)


class LoginRequest(BaseModel):
    """Local MVP login input."""

    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=256)


def create_saas_auth_router() -> APIRouter:
    """Create local MVP SaaS authentication endpoints."""
    router = APIRouter(prefix="/saas/auth", tags=["saas-auth"])

    @router.post("/register", response_model=SuccessResponse[dict], status_code=201)
    async def register(request: RegisterRequest, db: Session = Depends(get_db)):
        email = request.email.strip().lower()
        if db.query(SaaSUser).filter(SaaSUser.email == email).first() is not None:
            raise HTTPException(status_code=409, detail="Email is already registered")
        user = SaaSUser(id=str(uuid.uuid4()), email=email, password_hash=hash_password(request.password))
        tenant = Tenant(id=str(uuid.uuid4()), name=request.workspace_name.strip())
        membership = TenantMembership(tenant_id=tenant.id, user_id=user.id, role="owner")
        db.add_all([user, tenant, membership])
        db.commit()
        token = create_access_token(CurrentPrincipal(user_id=user.id, tenant_id=tenant.id))
        return SuccessResponse.create(
            data={
                "access_token": token,
                "token_type": "bearer",
                "tenant_id": tenant.id,
                "user_id": user.id,
                "email": user.email,
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
            raise HTTPException(status_code=403, detail="No active workspace membership")
        token = create_access_token(CurrentPrincipal(user_id=user.id, tenant_id=membership.tenant_id))
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
    async def current_user(principal: CurrentPrincipal = Depends(get_current_principal)):
        return SuccessResponse.create(
            data={"user_id": principal.user_id, "tenant_id": principal.tenant_id},
            msg="SaaS principal retrieved",
        )

    return router
