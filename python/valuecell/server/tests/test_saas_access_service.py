from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from valuecell.server.db.models.base import Base
from valuecell.server.db.models.saas_control import (
    EnterpriseAgreement,
    ServicePlan,
    TenantSubscription,
)
from valuecell.server.services.saas_access_service import TenantAccessService


def test_subscription_expiry_blocks_access_and_active_agreement_restores_it():
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        now = datetime.now(timezone.utc)
        plan = ServicePlan(
            code="individual-monthly",
            name="个人月度版",
            commercial_model="subscription",
            duration_days=30,
            price_cents=50_000,
            entitlements={"max_strategies": 3},
        )
        session.add(plan)
        session.flush()
        session.add(
            TenantSubscription(
                tenant_id="tenant-a",
                plan_id=plan.id,
                status="active",
                starts_at=now - timedelta(days=31),
                ends_at=now - timedelta(seconds=1),
                granted_by_user_id="admin-a",
            )
        )
        session.commit()

        assert (
            TenantAccessService.access_for(session, "tenant-a").status
            == "pending_activation"
        )

        agreement = EnterpriseAgreement(
            tenant_id="tenant-a",
            agreement_number="B-2026-001",
            status="active",
            revenue_share_rate="0.10",
            settlement_cycle_days=30,
            high_water_mark_quote="1000",
            starts_at=now - timedelta(days=1),
            created_by_user_id="admin-a",
        )
        session.add(agreement)
        session.commit()

        access = TenantAccessService.access_for(session, "tenant-a")
        assert access.active is True
        assert access.commercial_model == "revenue_share"
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()
