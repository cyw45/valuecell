"""Tenant administration and platform commercial-control API routes."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valuecell.server.api.auth import CurrentPrincipal, get_current_principal
from valuecell.server.api.schemas.base import SuccessResponse
from valuecell.server.db.connection import get_db
from valuecell.server.db.models.saas_control import (
    AuditEvent,
    EnterpriseAgreement,
    ProfitSettlement,
    ServicePlan,
    TenantSubscription,
)
from valuecell.server.db.models.tenant_credential import TenantCredential
from valuecell.server.db.models.tenant import (
    SaaSUser,
    Tenant,
    TenantMembership,
    TenantProfile,
)
from valuecell.server.services.audit_service import record_audit_event
from valuecell.server.services.profit_share_service import calculate_profit_share
from valuecell.server.services.saas_access_service import (
    TenantAccessService,
    require_active_tenant,
    require_platform_admin,
    require_tenant_permission,
)

TenantRole = Literal[
    "owner", "admin", "strategist", "trader", "viewer", "billing_manager"
]


class MemberCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=320)
    role: TenantRole


class TenantProfileUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_type: Literal["personal", "enterprise"]
    organization_name: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def require_enterprise_organization(self) -> "TenantProfileUpdateRequest":
        if self.tenant_type == "enterprise" and not self.organization_name:
            raise ValueError("enterprise tenant requires organization_name")
        return self


class ServicePlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,62}$")
    name: str = Field(min_length=1, max_length=160)
    duration_days: int = Field(gt=0, le=3_650)
    price_cents: int = Field(ge=0)
    currency: str = Field(default="CNY", min_length=3, max_length=8)
    entitlements: dict[str, int | bool] = Field(default_factory=dict)


class SubscriptionGrantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(min_length=1, max_length=36)
    plan_id: str = Field(min_length=1, max_length=36)
    ends_at: datetime
    note: str | None = Field(default=None, max_length=1_000)

    @model_validator(mode="after")
    def require_future_end(self) -> "SubscriptionGrantRequest":
        if self.ends_at <= datetime.now(timezone.utc):
            raise ValueError("ends_at must be in the future")
        return self


class EnterpriseAgreementRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(min_length=1, max_length=36)
    agreement_number: str = Field(min_length=1, max_length=100)
    revenue_share_rate: Decimal = Field(gt=0, le=1)
    settlement_cycle_days: int = Field(default=30, ge=1, le=366)
    high_water_mark_quote: Decimal = Field(default=Decimal("0"), ge=0)
    starts_at: datetime
    ends_at: datetime | None = None
    note: str | None = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def validate_period(self) -> "EnterpriseAgreementRequest":
        if self.ends_at is not None and self.ends_at <= self.starts_at:
            raise ValueError("ends_at must be later than starts_at")
        return self


class SettlementCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_id: str = Field(min_length=1, max_length=36)
    period_started_at: datetime
    period_ended_at: datetime
    ending_equity_quote: Decimal = Field(ge=0)
    net_external_cash_flow_quote: Decimal = Decimal("0")
    gross_realized_pnl_quote: Decimal = Decimal("0")
    fees_quote: Decimal = Field(default=Decimal("0"), ge=0)
    funding_quote: Decimal = Decimal("0")

    @model_validator(mode="after")
    def validate_period(self) -> "SettlementCreateRequest":
        if self.period_ended_at <= self.period_started_at:
            raise ValueError("period_ended_at must be later than period_started_at")
        return self


def create_saas_admin_router() -> APIRouter:
    """Create customer-workspace and platform-control SaaS routes."""

    router = APIRouter(prefix="/saas", tags=["saas-control"])

    @router.get("/access", response_model=SuccessResponse[dict[str, Any]])
    async def access(
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict[str, Any]]:
        profile = (
            db.query(TenantProfile)
            .filter(TenantProfile.tenant_id == principal.tenant_id)
            .first()
        )
        return SuccessResponse.create(
            data={
                "role": principal.role,
                "is_platform_admin": principal.is_platform_admin,
                "status": principal.access_status,
                "commercial_model": principal.commercial_model,
                "expires_at": principal.access_expires_at,
                "tenant_type": profile.tenant_type if profile else "personal",
                "organization_name": profile.organization_name if profile else None,
            },
            msg="Workspace access retrieved",
        )

    @router.get("/billing", response_model=SuccessResponse[dict[str, Any]])
    async def billing(
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict[str, Any]]:
        require_tenant_permission(principal, "billing.manage")
        subscriptions = (
            db.query(TenantSubscription)
            .filter(TenantSubscription.tenant_id == principal.tenant_id)
            .order_by(TenantSubscription.created_at.desc())
            .all()
        )
        agreement = (
            db.query(EnterpriseAgreement)
            .filter(EnterpriseAgreement.tenant_id == principal.tenant_id)
            .first()
        )
        settlements = []
        if agreement is not None:
            settlements = (
                db.query(ProfitSettlement)
                .filter(ProfitSettlement.agreement_id == agreement.id)
                .order_by(ProfitSettlement.period_ended_at.desc())
                .all()
            )
        return SuccessResponse.create(
            data={
                "access": _access_data(
                    TenantAccessService.access_for(db, principal.tenant_id)
                ),
                "subscriptions": [_subscription_data(item) for item in subscriptions],
                "agreement": _agreement_data(agreement)
                if agreement is not None
                else None,
                "settlements": [_settlement_data(item) for item in settlements],
            },
            msg="Tenant billing retrieved",
        )

    @router.get("/audit", response_model=SuccessResponse[list[dict[str, Any]]])
    async def tenant_audit_events(
        limit: int = Query(default=100, ge=1, le=500),
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[list[dict[str, Any]]]:
        require_active_tenant(principal)
        require_tenant_permission(principal, "tenant.read")
        events = (
            db.query(AuditEvent)
            .filter(AuditEvent.tenant_id == principal.tenant_id)
            .order_by(AuditEvent.created_at.desc())
            .limit(limit)
            .all()
        )
        return SuccessResponse.create(
            data=[_audit_data(event) for event in events],
            msg="Tenant audit events retrieved",
        )

    @router.get(
        "/workspace/members", response_model=SuccessResponse[list[dict[str, Any]]]
    )
    async def members(
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[list[dict[str, Any]]]:
        require_active_tenant(principal)
        require_tenant_permission(principal, "tenant.read")
        rows = (
            db.query(TenantMembership, SaaSUser)
            .join(SaaSUser, SaaSUser.id == TenantMembership.user_id)
            .filter(TenantMembership.tenant_id == principal.tenant_id)
            .order_by(TenantMembership.created_at.asc())
            .all()
        )
        return SuccessResponse.create(
            data=[
                {
                    "user_id": membership.user_id,
                    "email": user.email,
                    "role": membership.role,
                    "created_at": membership.created_at,
                }
                for membership, user in rows
            ],
            msg="Workspace members retrieved",
        )

    @router.post("/workspace/members", response_model=SuccessResponse[dict[str, Any]])
    async def add_member(
        request: MemberCreateRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict[str, Any]]:
        require_active_tenant(principal)
        require_tenant_permission(principal, "member.manage")
        user = (
            db.query(SaaSUser)
            .filter(SaaSUser.email == request.email.strip().lower())
            .first()
        )
        if user is None:
            raise HTTPException(status_code=404, detail="成员必须先完成平台注册")
        membership = (
            db.query(TenantMembership)
            .filter(
                TenantMembership.tenant_id == principal.tenant_id,
                TenantMembership.user_id == user.id,
            )
            .first()
        )
        if membership is None:
            membership = TenantMembership(
                tenant_id=principal.tenant_id,
                user_id=user.id,
                role=request.role,
            )
            db.add(membership)
        else:
            membership.role = request.role
        record_audit_event(
            db,
            action="tenant.member.upserted",
            target_type="tenant_membership",
            target_id=f"{principal.tenant_id}:{user.id}",
            outcome="success",
            tenant_id=principal.tenant_id,
            actor_user_id=principal.user_id,
            metadata={"role": request.role},
        )
        db.commit()
        return SuccessResponse.create(
            data={"user_id": user.id, "email": user.email, "role": membership.role},
            msg="Workspace member saved",
        )

    @router.get("/admin/tenants", response_model=SuccessResponse[list[dict[str, Any]]])
    async def tenants(
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[list[dict[str, Any]]]:
        require_platform_admin(principal)
        profiles = {
            profile.tenant_id: profile for profile in db.query(TenantProfile).all()
        }
        return SuccessResponse.create(
            data=[
                {
                    "id": tenant.id,
                    "name": tenant.name,
                    "tenant_type": profiles.get(tenant.id).tenant_type
                    if tenant.id in profiles
                    else "personal",
                    "organization_name": profiles.get(tenant.id).organization_name
                    if tenant.id in profiles
                    else None,
                    "created_at": tenant.created_at,
                    "access": _access_data(
                        TenantAccessService.access_for(db, tenant.id)
                    ),
                }
                for tenant in db.query(Tenant).order_by(Tenant.created_at.desc()).all()
            ],
            msg="Tenants retrieved",
        )

    @router.patch(
        "/admin/tenants/{tenant_id}/profile",
        response_model=SuccessResponse[dict[str, Any]],
    )
    async def update_tenant_profile(
        tenant_id: str,
        request: TenantProfileUpdateRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict[str, Any]]:
        require_platform_admin(principal)
        if db.query(Tenant).filter(Tenant.id == tenant_id).first() is None:
            raise HTTPException(status_code=404, detail="租户不存在")
        profile = (
            db.query(TenantProfile).filter(TenantProfile.tenant_id == tenant_id).first()
        )
        if profile is None:
            profile = TenantProfile(tenant_id=tenant_id)
            db.add(profile)
        profile.tenant_type = request.tenant_type
        profile.organization_name = (
            request.organization_name.strip() if request.organization_name else None
        )
        record_audit_event(
            db,
            action="platform.tenant.profile.updated",
            target_type="tenant_profile",
            target_id=tenant_id,
            outcome="success",
            tenant_id=tenant_id,
            actor_user_id=principal.user_id,
            metadata={"tenant_type": profile.tenant_type},
        )
        db.commit()
        return SuccessResponse.create(
            data={
                "tenant_id": tenant_id,
                "tenant_type": profile.tenant_type,
                "organization_name": profile.organization_name,
            },
            msg="Tenant profile updated",
        )

    @router.get("/admin/plans", response_model=SuccessResponse[list[dict[str, Any]]])
    async def plans(
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[list[dict[str, Any]]]:
        require_platform_admin(principal)
        return SuccessResponse.create(
            data=[
                _plan_data(plan)
                for plan in db.query(ServicePlan).order_by(ServicePlan.code).all()
            ],
            msg="Service plans retrieved",
        )

    @router.post(
        "/admin/plans", response_model=SuccessResponse[dict[str, Any]], status_code=201
    )
    async def create_plan(
        request: ServicePlanRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict[str, Any]]:
        require_platform_admin(principal)
        plan = ServicePlan(
            code=request.code,
            name=request.name,
            commercial_model="subscription",
            duration_days=request.duration_days,
            price_cents=request.price_cents,
            currency=request.currency.upper(),
            entitlements=request.entitlements,
        )
        db.add(plan)
        try:
            db.flush()
        except IntegrityError as exc:
            db.rollback()
            raise HTTPException(status_code=409, detail="套餐编码已存在") from exc
        record_audit_event(
            db,
            action="platform.plan.created",
            target_type="service_plan",
            target_id=plan.id,
            outcome="success",
            actor_user_id=principal.user_id,
            metadata={"code": plan.code},
        )
        db.commit()
        return SuccessResponse.create(data=_plan_data(plan), msg="Service plan created")

    @router.post(
        "/admin/subscriptions",
        response_model=SuccessResponse[dict[str, Any]],
        status_code=201,
    )
    async def grant_subscription(
        request: SubscriptionGrantRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict[str, Any]]:
        require_platform_admin(principal)
        if db.query(Tenant).filter(Tenant.id == request.tenant_id).first() is None:
            raise HTTPException(status_code=404, detail="租户不存在")
        if (
            db.query(ServicePlan).filter(ServicePlan.id == request.plan_id).first()
            is None
        ):
            raise HTTPException(status_code=404, detail="套餐不存在")
        now = datetime.now(timezone.utc)
        db.query(TenantSubscription).filter(
            TenantSubscription.tenant_id == request.tenant_id,
            TenantSubscription.status == "active",
        ).update({"status": "superseded"})
        subscription = TenantSubscription(
            tenant_id=request.tenant_id,
            plan_id=request.plan_id,
            starts_at=now,
            ends_at=request.ends_at,
            note=request.note,
            granted_by_user_id=principal.user_id,
        )
        db.add(subscription)
        db.flush()
        record_audit_event(
            db,
            action="platform.subscription.granted",
            target_type="tenant_subscription",
            target_id=subscription.id,
            outcome="success",
            tenant_id=request.tenant_id,
            actor_user_id=principal.user_id,
            metadata={
                "plan_id": request.plan_id,
                "ends_at": request.ends_at.isoformat(),
            },
        )
        db.commit()
        return SuccessResponse.create(
            data=_subscription_data(subscription), msg="Subscription activated"
        )

    @router.post(
        "/admin/agreements",
        response_model=SuccessResponse[dict[str, Any]],
        status_code=201,
    )
    async def create_agreement(
        request: EnterpriseAgreementRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict[str, Any]]:
        require_platform_admin(principal)
        if db.query(Tenant).filter(Tenant.id == request.tenant_id).first() is None:
            raise HTTPException(status_code=404, detail="租户不存在")
        if (
            db.query(EnterpriseAgreement)
            .filter(EnterpriseAgreement.tenant_id == request.tenant_id)
            .first()
            is not None
        ):
            raise HTTPException(status_code=409, detail="租户已有利润分成合同")
        agreement = EnterpriseAgreement(
            tenant_id=request.tenant_id,
            agreement_number=request.agreement_number,
            revenue_share_rate=str(request.revenue_share_rate),
            settlement_cycle_days=request.settlement_cycle_days,
            high_water_mark_quote=str(request.high_water_mark_quote),
            starts_at=request.starts_at,
            ends_at=request.ends_at,
            note=request.note,
            created_by_user_id=principal.user_id,
        )
        db.add(agreement)
        try:
            db.flush()
        except IntegrityError as exc:
            db.rollback()
            raise HTTPException(status_code=409, detail="合同编号已存在") from exc
        record_audit_event(
            db,
            action="platform.agreement.created",
            target_type="enterprise_agreement",
            target_id=agreement.id,
            outcome="success",
            tenant_id=request.tenant_id,
            actor_user_id=principal.user_id,
            metadata={"agreement_number": request.agreement_number},
        )
        db.commit()
        return SuccessResponse.create(
            data=_agreement_data(agreement), msg="Agreement activated"
        )

    @router.post(
        "/admin/agreements/{agreement_id}/settlements",
        response_model=SuccessResponse[dict[str, Any]],
        status_code=201,
    )
    async def create_settlement(
        agreement_id: str,
        request: SettlementCreateRequest,
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[dict[str, Any]]:
        require_platform_admin(principal)
        agreement = (
            db.query(EnterpriseAgreement)
            .filter(EnterpriseAgreement.id == agreement_id)
            .first()
        )
        if agreement is None or agreement.status != "active":
            raise HTTPException(status_code=404, detail="有效合同不存在")
        connection = (
            db.query(TenantCredential)
            .filter(
                TenantCredential.id == request.connection_id,
                TenantCredential.tenant_id == agreement.tenant_id,
                TenantCredential.kind == "exchange",
                TenantCredential.revoked.is_(False),
            )
            .first()
        )
        if connection is None:
            raise HTTPException(status_code=422, detail="结算资金账户不属于合同租户")
        rate = Decimal(agreement.revenue_share_rate)
        calculation = calculate_profit_share(
            high_water_mark_before=Decimal(agreement.high_water_mark_quote),
            ending_equity=request.ending_equity_quote,
            net_external_cash_flow=request.net_external_cash_flow_quote,
            revenue_share_rate=rate,
        )
        settlement = ProfitSettlement(
            tenant_id=agreement.tenant_id,
            agreement_id=agreement.id,
            connection_id=request.connection_id,
            period_started_at=request.period_started_at,
            period_ended_at=request.period_ended_at,
            ending_equity_quote=str(request.ending_equity_quote),
            net_external_cash_flow_quote=str(request.net_external_cash_flow_quote),
            gross_realized_pnl_quote=str(request.gross_realized_pnl_quote),
            fees_quote=str(request.fees_quote),
            funding_quote=str(request.funding_quote),
            high_water_mark_before_quote=str(calculation.high_water_mark_before),
            high_water_mark_after_quote=str(calculation.high_water_mark_after),
            eligible_profit_quote=str(calculation.eligible_profit),
            revenue_share_rate=str(rate),
            amount_due_quote=str(calculation.amount_due),
            reviewed_by_user_id=principal.user_id,
        )
        agreement.high_water_mark_quote = str(calculation.high_water_mark_after)
        db.add(settlement)
        try:
            db.flush()
        except IntegrityError as exc:
            db.rollback()
            raise HTTPException(
                status_code=409, detail="该账户结算周期已经存在"
            ) from exc
        record_audit_event(
            db,
            action="platform.settlement.created",
            target_type="profit_settlement",
            target_id=settlement.id,
            outcome="success",
            tenant_id=agreement.tenant_id,
            actor_user_id=principal.user_id,
            metadata={
                "agreement_id": agreement.id,
                "amount_due_quote": str(calculation.amount_due),
            },
        )
        db.commit()
        return SuccessResponse.create(
            data=_settlement_data(settlement), msg="Profit settlement created"
        )

    @router.get("/admin/audit", response_model=SuccessResponse[list[dict[str, Any]]])
    async def audit_events(
        tenant_id: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
        principal: CurrentPrincipal = Depends(get_current_principal),
        db: Session = Depends(get_db),
    ) -> SuccessResponse[list[dict[str, Any]]]:
        require_platform_admin(principal)
        query = db.query(AuditEvent)
        if tenant_id is not None:
            query = query.filter(AuditEvent.tenant_id == tenant_id)
        rows = query.order_by(AuditEvent.created_at.desc()).limit(limit).all()
        return SuccessResponse.create(
            data=[
                {
                    "id": event.id,
                    "tenant_id": event.tenant_id,
                    "actor_user_id": event.actor_user_id,
                    "action": event.action,
                    "target_type": event.target_type,
                    "target_id": event.target_id,
                    "outcome": event.outcome,
                    "metadata": event.metadata_json,
                    "created_at": event.created_at,
                }
                for event in rows
            ],
            msg="Audit events retrieved",
        )

    return router


def _access_data(access: Any) -> dict[str, Any]:
    return {
        "status": access.status,
        "commercial_model": access.commercial_model,
        "expires_at": access.expires_at,
        "entitlements": access.entitlements,
    }


def _plan_data(plan: ServicePlan) -> dict[str, Any]:
    return {
        "id": plan.id,
        "code": plan.code,
        "name": plan.name,
        "duration_days": plan.duration_days,
        "price_cents": plan.price_cents,
        "currency": plan.currency,
        "entitlements": plan.entitlements,
        "active": plan.active,
    }


def _subscription_data(subscription: TenantSubscription) -> dict[str, Any]:
    return {
        "id": subscription.id,
        "tenant_id": subscription.tenant_id,
        "plan_id": subscription.plan_id,
        "status": subscription.status,
        "starts_at": subscription.starts_at,
        "ends_at": subscription.ends_at,
        "note": subscription.note,
    }


def _agreement_data(agreement: EnterpriseAgreement) -> dict[str, Any]:
    return {
        "id": agreement.id,
        "tenant_id": agreement.tenant_id,
        "agreement_number": agreement.agreement_number,
        "status": agreement.status,
        "revenue_share_rate": agreement.revenue_share_rate,
        "settlement_cycle_days": agreement.settlement_cycle_days,
        "high_water_mark_quote": agreement.high_water_mark_quote,
        "starts_at": agreement.starts_at,
        "ends_at": agreement.ends_at,
    }


def _settlement_data(settlement: ProfitSettlement) -> dict[str, Any]:
    return {
        "id": settlement.id,
        "tenant_id": settlement.tenant_id,
        "agreement_id": settlement.agreement_id,
        "connection_id": settlement.connection_id,
        "period_started_at": settlement.period_started_at,
        "period_ended_at": settlement.period_ended_at,
        "ending_equity_quote": settlement.ending_equity_quote,
        "net_external_cash_flow_quote": settlement.net_external_cash_flow_quote,
        "gross_realized_pnl_quote": settlement.gross_realized_pnl_quote,
        "fees_quote": settlement.fees_quote,
        "funding_quote": settlement.funding_quote,
        "high_water_mark_before_quote": settlement.high_water_mark_before_quote,
        "high_water_mark_after_quote": settlement.high_water_mark_after_quote,
        "eligible_profit_quote": settlement.eligible_profit_quote,
        "revenue_share_rate": settlement.revenue_share_rate,
        "amount_due_quote": settlement.amount_due_quote,
        "status": settlement.status,
    }


def _audit_data(event: AuditEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "tenant_id": event.tenant_id,
        "actor_user_id": event.actor_user_id,
        "action": event.action,
        "target_type": event.target_type,
        "target_id": event.target_id,
        "outcome": event.outcome,
        "metadata": event.metadata_json,
        "created_at": event.created_at,
    }
