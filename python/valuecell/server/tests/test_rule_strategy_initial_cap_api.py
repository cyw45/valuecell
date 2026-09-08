from collections.abc import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from valuecell.server.api.auth import CurrentPrincipal, get_current_principal
from valuecell.server.api.routers.rule_strategy import create_rule_strategy_router
from valuecell.server.db.connection import get_db
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.multi_strategy import StrategySharedAccount
from valuecell.server.db.models.shared_demo_execution import (
    SharedDemoStrategyAllocationCap,
)
from valuecell.server.db.models.tenant import SaaSUser, Tenant
from valuecell.server.db.models.tenant_credential import TenantCredential
from valuecell.server.db.repositories.rule_strategy_repository import (
    RuleStrategyRepository,
)
from valuecell.server.services.rule_strategy_service import RuleStrategyService


@pytest.fixture
def client_and_session() -> Generator[tuple[TestClient, Session], None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
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
        StrategySharedAccount(
            id="account-a",
            tenant_id="tenant-a",
            credential_id="credential-a",
            environment="okx_demo",
            active=True,
        )
    )
    session.commit()

    app = FastAPI()
    app.include_router(
        create_rule_strategy_router(
            service=RuleStrategyService(
                repository=RuleStrategyRepository(db_session=session)
            )
        )
    )
    app.dependency_overrides[get_current_principal] = lambda: CurrentPrincipal(
        user_id="user-a", tenant_id="tenant-a"
    )
    app.dependency_overrides[get_db] = lambda: session
    try:
        with TestClient(app) as client:
            yield client, session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def _configurable_request(*, environment: str) -> dict:
    execution = {"environment": environment}
    if environment == "okx_demo":
        execution["sandbox_connection_id"] = "credential-a"
    return {
        "name": f"Configurable {environment}",
        "initial_capital_quote": 700,
        "config": {
            "symbols": ["BTC-USDT"],
            "risk": {"order_quote_amount": 100},
            "execution": execution,
        },
    }


def test_configurable_demo_creation_seeds_initial_allocation_cap(
    client_and_session: tuple[TestClient, Session],
) -> None:
    client, session = client_and_session

    response = client.post(
        "/rule-strategies", json=_configurable_request(environment="okx_demo")
    )

    assert response.status_code == 201
    strategy_id = response.json()["data"]["strategy_id"]
    cap = session.query(SharedDemoStrategyAllocationCap).one()
    assert cap.strategy_id == strategy_id
    assert float(cap.max_reserved_quote) == 700
    assert float(cap.max_occupied_quote) == 700


def test_fixed_demo_creation_seeds_initial_allocation_cap(
    client_and_session: tuple[TestClient, Session],
) -> None:
    client, session = client_and_session

    response = client.post(
        "/rule-strategies/fixed",
        json={
            "kind": "dual_ma_trend",
            "name": "Dual MA Demo",
            "initial_capital_quote": 900,
            "environment": "okx_demo",
            "credential_id": "credential-a",
        },
    )

    assert response.status_code == 201
    strategy_id = response.json()["data"]["strategy_id"]
    cap = session.query(SharedDemoStrategyAllocationCap).one()
    assert cap.strategy_id == strategy_id
    assert float(cap.max_reserved_quote) == 900
    assert float(cap.max_occupied_quote) == 900


def test_paper_creation_never_seeds_shared_demo_allocation_cap(
    client_and_session: tuple[TestClient, Session],
) -> None:
    client, session = client_and_session

    configurable = client.post(
        "/rule-strategies", json=_configurable_request(environment="paper")
    )
    fixed = client.post(
        "/rule-strategies/fixed",
        json={
            "kind": "dual_ma_trend",
            "name": "Dual MA Paper",
            "initial_capital_quote": 900,
            "environment": "paper",
        },
    )

    assert configurable.status_code == 201
    assert fixed.status_code == 201
    assert session.query(SharedDemoStrategyAllocationCap).count() == 0
