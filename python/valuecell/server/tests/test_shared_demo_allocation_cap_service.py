from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from valuecell.server.db.models.base import Base
from valuecell.server.db.models.multi_strategy import StrategySharedAccount
from valuecell.server.db.models.rule_strategy import RuleStrategy
from valuecell.server.db.models.shared_demo_execution import (
    SharedDemoStrategyAllocationCap,
)
from valuecell.server.db.models.tenant import SaaSUser, Tenant
from valuecell.server.db.models.tenant_credential import TenantCredential
from valuecell.server.services.shared_demo_allocation_cap_service import (
    ensure_initial_strategy_cap,
)


def _session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _seed_demo_scope(session) -> None:
    session.add(Tenant(id="tenant-a", name="Tenant A"))
    session.add(
        SaaSUser(
            id="user-a",
            email="owner@example.test",
            password_hash="not-used",
        )
    )
    session.add(
        TenantCredential(
            id="credential-a",
            tenant_id="tenant-a",
            created_by_user_id="user-a",
            kind="exchange",
            provider="okx",
            label="OKX Demo",
            encrypted_payload="not-used",
            nonce="not-used",
            metadata_json={"sandbox": True, "market_type": "spot"},
        )
    )
    session.add(
        RuleStrategy(
            strategy_id="strategy-a",
            tenant_id="tenant-a",
            name="Strategy A",
            strategy_kind="dual_ma_trend",
            strategy_version="v1",
            code_fingerprint="test",
            status="stopped",
            paper_mode=False,
            config={
                "initial_capital_quote": 500,
                "execution": {
                    "environment": "okx_demo",
                    "sandbox_connection_id": "credential-a",
                },
            },
        )
    )
    session.add(
        StrategySharedAccount(
            id="account-a",
            tenant_id="tenant-a",
            credential_id="credential-a",
            environment="okx_demo",
            active=True,
            sync_status="healthy",
            attribution_status="pending",
            observed_at=datetime.now(timezone.utc),
        )
    )
    session.commit()


def test_ensure_initial_strategy_cap_seeds_and_preserves_one_active_cap() -> None:
    session = _session()
    _seed_demo_scope(session)

    created = ensure_initial_strategy_cap(
        session,
        tenant_id="tenant-a",
        strategy_id="strategy-a",
        credential_id="credential-a",
        initial_capital_quote=500,
    )
    repeated = ensure_initial_strategy_cap(
        session,
        tenant_id="tenant-a",
        strategy_id="strategy-a",
        credential_id="credential-a",
        initial_capital_quote=900,
    )
    session.commit()

    assert created is not None
    assert repeated.cap_id == created.cap_id
    caps = session.query(SharedDemoStrategyAllocationCap).all()
    assert len(caps) == 1
    assert float(caps[0].max_reserved_quote) == 500
    assert float(caps[0].max_occupied_quote) == 500


def test_ensure_initial_strategy_cap_waits_for_shared_account_sync() -> None:
    session = _session()

    assert (
        ensure_initial_strategy_cap(
            session,
            tenant_id="tenant-a",
            strategy_id="strategy-a",
            credential_id="credential-a",
            initial_capital_quote=500,
        )
        is None
    )
    assert session.query(SharedDemoStrategyAllocationCap).count() == 0
