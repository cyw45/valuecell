import base64
from collections.abc import Generator
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import valuecell.server.services.tenant_credential_service as credential_module
import valuecell.server.services.sandbox_exchange_service as sandbox_exchange_module
from valuecell.server.api.auth import CurrentPrincipal, get_current_principal
from valuecell.server.api.routers.tenant_credential import create_tenant_credential_router
from valuecell.server.db.connection import get_db
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.tenant_credential import TenantCredential
from valuecell.server.services.tenant_credential_service import (
    CredentialVaultError,
    TenantCredentialService,
)

TEST_MASTER_KEY = base64.urlsafe_b64encode(b"0123456789abcdef0123456789abcdef").decode("ascii")
SECRET = {"api_key": "test-api-key-not-for-production", "api_secret": "test-api-secret"}


@dataclass(frozen=True)
class CredentialSettingsFixture:
    CREDENTIAL_MASTER_KEY: str = TEST_MASTER_KEY


@pytest.fixture
def credential_database() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def credential_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[TestClient, list[CurrentPrincipal]], None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    principal = [CurrentPrincipal(user_id="user-a", tenant_id="tenant-a")]
    monkeypatch.setattr(credential_module, "get_settings", lambda: CredentialSettingsFixture())

    def override_db() -> Generator[Session, None, None]:
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app = FastAPI()
    app.include_router(create_tenant_credential_router())
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_principal] = lambda: principal[0]
    try:
        with TestClient(app) as client:
            yield client, principal
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def _credential_request(label: str = "primary") -> dict:
    return {
        "kind": "exchange",
        "provider": "paper-exchange",
        "label": label,
        "secret": SECRET,
        "metadata": {"environment": "paper"},
    }


def test_credential_api_returns_only_metadata_and_blocks_cross_tenant_access(
    credential_client: tuple[TestClient, list[CurrentPrincipal]],
):
    client, principal = credential_client

    unauthenticated_app = FastAPI()
    unauthenticated_app.include_router(create_tenant_credential_router())
    with TestClient(unauthenticated_app) as unauthenticated_client:
        assert unauthenticated_client.get("/saas/credentials").status_code == 401

    created = client.post("/saas/credentials", json=_credential_request())
    assert created.status_code == 201
    created_body = created.json()
    credential_id = created_body["data"]["id"]
    serialized_created = str(created_body)
    assert all(
        forbidden not in serialized_created
        for forbidden in (*SECRET.values(), "secret", "encrypted_payload", "nonce")
    )
    assert created_body["data"] == {
        "id": credential_id,
        "kind": "exchange",
        "provider": "paper-exchange",
        "label": "primary",
        "metadata": {"environment": "paper"},
        "revoked": False,
        "created_at": created_body["data"]["created_at"],
        "revoked_at": None,
    }

    principal[0] = CurrentPrincipal(user_id="user-b", tenant_id="tenant-b")
    tenant_b_list = client.get("/saas/credentials")
    denied_revoke = client.post(f"/saas/credentials/{credential_id}/revoke")
    assert tenant_b_list.status_code == 200
    assert tenant_b_list.json()["data"] == []
    assert denied_revoke.status_code == 404

    principal[0] = CurrentPrincipal(user_id="user-a", tenant_id="tenant-a")
    listed = client.get("/saas/credentials")
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["data"]] == [credential_id]


def test_vault_encrypts_at_rest_and_only_decrypts_for_owning_tenant(
    credential_database: Session,
):
    service = TenantCredentialService(credential_database, master_key=TEST_MASTER_KEY)
    metadata = service.create(
        tenant_id="tenant-a",
        user_id="user-a",
        kind="exchange",
        provider="paper-exchange",
        label="primary",
        secret=SECRET,
    )
    stored = credential_database.get(TenantCredential, metadata["id"])
    assert stored is not None
    assert stored.encrypted_payload != base64.urlsafe_b64encode(
        b'{"api_key":"test-api-key-not-for-production","api_secret":"test-api-secret"}'
    ).decode("ascii")
    assert all(value not in stored.encrypted_payload for value in SECRET.values())
    assert service.decrypt_for_internal_use("tenant-a", metadata["id"]) == SECRET

    with pytest.raises(CredentialVaultError, match="Active credential was not found"):
        service.decrypt_for_internal_use("tenant-b", metadata["id"])


def test_vault_rejects_missing_and_malformed_master_keys(
    credential_database: Session, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        credential_module, "get_settings", lambda: CredentialSettingsFixture(CREDENTIAL_MASTER_KEY=None)
    )
    with pytest.raises(CredentialVaultError, match="not configured"):
        TenantCredentialService(credential_database)

    for master_key, expected_message in (
        ("not-base64!", "malformed"),
        (base64.urlsafe_b64encode(b"too-short").decode("ascii"), "32 bytes"),
    ):
        with pytest.raises(CredentialVaultError, match=expected_message):
            TenantCredentialService(credential_database, master_key=master_key)


def _fake_exchange_class(*, fail_balance: bool = False):
    class FakeExchange:
        instances: list["FakeExchange"] = []

        def __init__(self, config: dict):
            self.config = config
            self.calls = ["init"]
            self.__class__.instances.append(self)

        def set_sandbox_mode(self, enabled: bool) -> None:
            assert enabled is True
            self.calls.append("sandbox")

        async def fetch_balance(self) -> dict:
            self.calls.append("fetch_balance")
            if fail_balance:
                raise RuntimeError("invalid sandbox credentials")
            return {"USDT": {"free": 1}}

        async def close(self) -> None:
            self.calls.append("close")

    return FakeExchange


def test_sandbox_binance_validation_uses_one_sandbox_balance_request_without_persisting(
    credential_client: tuple[TestClient, list[CurrentPrincipal]],
    monkeypatch: pytest.MonkeyPatch,
):
    client, _ = credential_client
    fake_exchange = _fake_exchange_class()
    monkeypatch.setattr(sandbox_exchange_module.ccxtpro, "binance", fake_exchange)
    request = {
        "provider": "binance",
        "api_key": "binance-test-key",
        "api_secret": "binance-test-secret",
    }

    response = client.post("/saas/credentials/sandbox/validate", json=request)

    assert response.status_code == 200
    body = response.json()
    assert body["data"] == {
        "provider": "binance",
        "sandbox": True,
        "validated": True,
        "checked_at": body["data"]["checked_at"],
    }
    assert "binance-test-key" not in str(body)
    assert "binance-test-secret" not in str(body)
    assert len(fake_exchange.instances) == 1
    instance = fake_exchange.instances[0]
    assert instance.config == {
        "apiKey": "binance-test-key",
        "secret": "binance-test-secret",
        "enableRateLimit": True,
    }
    assert instance.calls == ["init", "sandbox", "fetch_balance", "close"]
    assert client.get("/saas/credentials").json()["data"] == []


def test_sandbox_okx_requires_passphrase_and_maps_it_to_ccxt_password(
    credential_client: tuple[TestClient, list[CurrentPrincipal]],
    monkeypatch: pytest.MonkeyPatch,
):
    client, _ = credential_client
    fake_exchange = _fake_exchange_class()
    monkeypatch.setattr(sandbox_exchange_module.ccxtpro, "okx", fake_exchange)
    missing_passphrase = client.post(
        "/saas/credentials/sandbox/validate",
        json={
            "provider": "okx",
            "api_key": "okx-test-key",
            "api_secret": "okx-test-secret",
        },
    )

    assert missing_passphrase.status_code == 422
    assert fake_exchange.instances == []

    response = client.post(
        "/saas/credentials/sandbox/validate",
        json={
            "provider": "okx",
            "api_key": "okx-test-key",
            "api_secret": "okx-test-secret",
            "passphrase": "okx-test-passphrase",
        },
    )

    assert response.status_code == 200
    assert response.json()["data"]["validated"] is True
    instance = fake_exchange.instances[0]
    assert instance.config["password"] == "okx-test-passphrase"
    assert instance.calls == ["init", "sandbox", "fetch_balance", "close"]
    assert "okx-test-passphrase" not in str(response.json())
    assert client.get("/saas/credentials").json()["data"] == []


def test_sandbox_validation_sanitizes_balance_failures_and_closes_exchange(
    credential_client: tuple[TestClient, list[CurrentPrincipal]],
    monkeypatch: pytest.MonkeyPatch,
):
    client, _ = credential_client
    fake_exchange = _fake_exchange_class(fail_balance=True)
    monkeypatch.setattr(sandbox_exchange_module.ccxtpro, "binance", fake_exchange)
    request = {
        "provider": "binance",
        "api_key": "failed-binance-key",
        "api_secret": "failed-binance-secret",
    }

    response = client.post("/saas/credentials/sandbox/validate", json=request)

    assert response.status_code == 200
    assert response.json()["data"] == {
        "provider": "binance",
        "sandbox": True,
        "validated": False,
        "checked_at": response.json()["data"]["checked_at"],
    }
    assert "failed-binance-key" not in str(response.json())
    assert "failed-binance-secret" not in str(response.json())
    assert fake_exchange.instances[0].calls == ["init", "sandbox", "fetch_balance", "close"]
    assert client.get("/saas/credentials").json()["data"] == []
