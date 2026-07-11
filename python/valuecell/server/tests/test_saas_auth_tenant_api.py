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


def _rule_config() -> dict:
    return {
        "mode": "paper",
        "confirmation_mode": "all",
        "rsi": {"enabled": True, "period": 2, "oversold": 30, "overbought": 70},
        "risk": {
            "size_mode": "fixed_quote",
            "size_value": 100,
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
    assert registered_principal.json()["data"] == {
        "user_id": registration["user_id"],
        "tenant_id": registration["tenant_id"],
    }

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
    assert logged_in_principal.json()["data"] == {
        "user_id": registration["user_id"],
        "tenant_id": registration["tenant_id"],
    }

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


def test_rule_strategies_require_a_registered_token_and_derive_tenant_scope(
    auth_client: TestClient,
):
    unauthenticated = auth_client.get("/rule-strategies")
    assert unauthenticated.status_code == 401

    tenant_a = auth_client.post(
        "/saas/auth/register",
        json={
            "email": "owner-a@example.test",
            "password": "correct-horse-battery-staple",
            "workspace_name": "Tenant A",
        },
    ).json()["data"]
    created = auth_client.post(
        "/rule-strategies",
        headers=_authorization(tenant_a["access_token"]),
        json={"name": "Tenant A strategy", "config": _rule_config()},
    )
    assert created.status_code == 201
    strategy_id = created.json()["data"]["strategy_id"]

    tenant_b = auth_client.post(
        "/saas/auth/register",
        json={
            "email": "owner-b@example.test",
            "password": "correct-horse-battery-staple",
            "workspace_name": "Tenant B",
        },
    ).json()["data"]
    tenant_b_list = auth_client.get(
        "/rule-strategies", headers=_authorization(tenant_b["access_token"])
    )
    denied_lookup = auth_client.get(
        f"/rule-strategies/{strategy_id}", headers=_authorization(tenant_b["access_token"])
    )

    assert tenant_b_list.status_code == 200
    assert tenant_b_list.json()["data"] == []
    assert denied_lookup.status_code == 404
