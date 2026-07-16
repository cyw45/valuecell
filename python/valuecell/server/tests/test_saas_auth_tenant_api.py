from datetime import datetime, timezone

from collections.abc import Generator
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import valuecell.server.api.auth as auth_module
from valuecell.server.api.routers.saas_auth import create_saas_auth_router
from valuecell.server.api.routers.rule_strategy import create_rule_strategy_router
from valuecell.server.db.connection import get_db
from valuecell.server.db.models.base import Base

from valuecell.server.services.rule_strategy_service import RuleStrategyService


@dataclass(frozen=True)
class AuthSettingsFixture:
    JWT_SECRET: str = "test-jwt-signing-secret"
    JWT_ISSUER: str = "valuecell-saas-test"
    JWT_ACCESS_TOKEN_TTL_S: int = 3_600
    PLATFORM_ADMIN_EMAILS: tuple[str, ...] = ("platform@example.test",)


class InMemoryTenantStrategyRepository:
    def __init__(self) -> None:
        self.strategies = {}
        self.sequence = 0

    def create(self, strategy):
        self.sequence += 1
        strategy.strategy_id = f"rule-{self.sequence}"
        strategy.created_at = datetime(2026, 7, 10, tzinfo=timezone.utc)
        strategy.updated_at = strategy.created_at
        self.strategies[(strategy.tenant_id, strategy.strategy_id)] = strategy
        return strategy

    def list(self, tenant_id: str):
        return [
            strategy
            for (stored_tenant_id, _), strategy in self.strategies.items()
            if stored_tenant_id == tenant_id
        ]

    def get(self, strategy_id: str, tenant_id: str):
        return self.strategies.get((tenant_id, strategy_id))

    def get_evaluations(self, strategy_id: str, tenant_id: str, limit: int = 100):
        return []


def _rule_config() -> dict:
    return {
        "mode": "paper",
        "confirmation_mode": "all",
        "rsi": {"enabled": True, "period": 2, "oversold": 30, "overbought": 70},
        "risk": {
            "order_quote_amount": 100,
            "max_positions": 1,
            "leverage": 1,
        },
    }


@pytest.fixture
def auth_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    monkeypatch.setattr(auth_module, "get_settings", lambda: AuthSettingsFixture())

    def override_db() -> Generator[Session, None, None]:
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app = FastAPI()
    app.include_router(create_saas_auth_router())
    app.include_router(
        create_rule_strategy_router(
            service=RuleStrategyService(repository=InMemoryTenantStrategyRepository())
        )
    )
    app.dependency_overrides[get_db] = override_db
    try:
        with TestClient(app) as client:
            yield client
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def _authorization(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_register_and_login_issue_signed_tokens_for_the_registered_tenant(
    auth_client: TestClient,
):
    registered = auth_client.post(
        "/saas/auth/register",
        json={
            "email": "owner@example.test",
            "password": "correct-horse-battery-staple",
            "workspace_name": "Research Desk",
        },
    )

    assert registered.status_code == 201
    registration = registered.json()["data"]
    assert registration["token_type"] == "bearer"

    registered_principal = auth_client.get(
        "/saas/auth/me", headers=_authorization(registration["access_token"])
    )
    assert registered_principal.status_code == 200
    assert (
        registered_principal.json()["data"].items()
        >= {
            "user_id": registration["user_id"],
            "tenant_id": registration["tenant_id"],
            "role": "owner",
            "access_status": "pending_activation",
        }.items()
    )

    logged_in = auth_client.post(
        "/saas/auth/login",
        json={
            "email": "OWNER@EXAMPLE.TEST",
            "password": "correct-horse-battery-staple",
        },
    )
    assert logged_in.status_code == 200
    login = logged_in.json()["data"]
    assert login["token_type"] == "bearer"

    logged_in_principal = auth_client.get(
        "/saas/auth/me", headers=_authorization(login["access_token"])
    )
    assert logged_in_principal.status_code == 200
    assert (
        logged_in_principal.json()["data"].items()
        >= {
            "user_id": registration["user_id"],
            "tenant_id": registration["tenant_id"],
            "role": "owner",
            "access_status": "pending_activation",
        }.items()
    )

    token_parts = registration["access_token"].split(".")
    token_parts[2] = "A" + token_parts[2][1:]
    tampered_token = ".".join(token_parts)
    rejected_tampered_token = auth_client.get(
        "/saas/auth/me", headers=_authorization(tampered_token)
    )
    assert rejected_tampered_token.status_code == 401


def test_auth_rejects_duplicate_canonical_email_and_wrong_password(
    auth_client: TestClient,
):
    payload = {
        "email": "owner@example.test",
        "password": "correct-horse-battery-staple",
        "workspace_name": "Research Desk",
    }
    assert auth_client.post("/saas/auth/register", json=payload).status_code == 201

    duplicate = auth_client.post(
        "/saas/auth/register",
        json={**payload, "email": " OWNER@EXAMPLE.TEST "},
    )
    rejected_login = auth_client.post(
        "/saas/auth/login",
        json={"email": payload["email"], "password": "incorrect-password-value"},
    )

    assert duplicate.status_code == 409
    assert rejected_login.status_code == 401


def test_registration_distinguishes_personal_and_enterprise_tenants(
    auth_client: TestClient,
):
    rejected = auth_client.post(
        "/saas/auth/register",
        json={
            "email": "missing-org@example.test",
            "password": "correct-horse-battery-staple",
            "tenant_type": "enterprise",
            "workspace_name": "Missing organization",
        },
    )
    assert rejected.status_code == 422

    enterprise = auth_client.post(
        "/saas/auth/register",
        json={
            "email": "enterprise-owner@example.test",
            "password": "correct-horse-battery-staple",
            "tenant_type": "enterprise",
            "workspace_name": "Quant Operations",
            "organization_name": "ValueCell Enterprise Test",
        },
    )
    assert enterprise.status_code == 201
    registration = enterprise.json()["data"]
    assert registration["tenant_type"] == "enterprise"
    assert registration["organization_name"] == "ValueCell Enterprise Test"

    workspaces = auth_client.get(
        "/saas/auth/workspaces",
        headers=_authorization(registration["access_token"]),
    )
    assert workspaces.status_code == 200
    assert workspaces.json()["data"] == [
        {
            "tenant_id": registration["tenant_id"],
            "name": "Quant Operations",
            "tenant_type": "enterprise",
            "organization_name": "ValueCell Enterprise Test",
            "role": "owner",
            "selected": True,
        }
    ]


def test_platform_administrator_is_distinct_from_tenant_owner(
    auth_client: TestClient,
):
    platform = auth_client.post(
        "/saas/auth/register",
        json={
            "email": "platform@example.test",
            "password": "correct-horse-battery-staple",
            "workspace_name": "Platform workspace",
        },
    ).json()["data"]
    tenant_owner = auth_client.post(
        "/saas/auth/register",
        json={
            "email": "cyw@example.test",
            "password": "correct-horse-battery-staple",
            "workspace_name": "CYW workspace",
        },
    ).json()["data"]

    platform_me = auth_client.get(
        "/saas/auth/me", headers=_authorization(platform["access_token"])
    ).json()["data"]
    tenant_owner_me = auth_client.get(
        "/saas/auth/me", headers=_authorization(tenant_owner["access_token"])
    ).json()["data"]

    assert platform_me["role"] == "owner"
    assert platform_me["is_platform_admin"] is True
    assert tenant_owner_me["role"] == "owner"
    assert tenant_owner_me["is_platform_admin"] is False
