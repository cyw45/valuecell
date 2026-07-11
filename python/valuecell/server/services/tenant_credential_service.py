"""Tenant-scoped encrypted credential vault for non-live SaaS configuration."""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy.orm import Session

from valuecell.server.config.settings import get_settings
from valuecell.server.db.models.tenant_credential import TenantCredential

_ALLOWED_KINDS = {"exchange", "market_data"}
_FORBIDDEN_FIELDS = {"private_key", "wallet_address", "signature", "seed_phrase"}


class CredentialVaultError(ValueError):
    """Raised when credential input or encryption configuration is unsafe."""


class TenantCredentialService:
    """Encrypts at rest and exposes metadata-only tenant credential operations."""

    def __init__(self, db: Session, master_key: str | None = None) -> None:
        self.db = db
        configured_key = master_key if master_key is not None else get_settings().CREDENTIAL_MASTER_KEY
        self._key = _decode_master_key(configured_key)

    def create(
        self,
        tenant_id: str,
        user_id: str,
        kind: str,
        provider: str,
        label: str,
        secret: dict[str, str],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if kind not in _ALLOWED_KINDS:
            raise CredentialVaultError("Credential kind is not allowed")
        if not provider.strip() or not label.strip():
            raise CredentialVaultError("Provider and label are required")
        if not secret or any(key in _FORBIDDEN_FIELDS for key in secret):
            raise CredentialVaultError("Credential payload contains unsupported secret fields")
        if any(not isinstance(value, str) or not value for value in secret.values()):
            raise CredentialVaultError("Credential secret values must be non-empty strings")
        nonce = os.urandom(12)
        payload = json.dumps(secret, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ciphertext = AESGCM(self._key).encrypt(nonce, payload, _associated_data(tenant_id, kind, provider))
        credential = TenantCredential(
            tenant_id=tenant_id,
            created_by_user_id=user_id,
            kind=kind,
            provider=provider.strip(),
            label=label.strip(),
            encrypted_payload=base64.urlsafe_b64encode(ciphertext).decode("ascii"),
            nonce=base64.urlsafe_b64encode(nonce).decode("ascii"),
            metadata_json=metadata or {},
        )
        self.db.add(credential)
        self.db.commit()
        self.db.refresh(credential)
        return _metadata(credential)

    def list(self, tenant_id: str) -> list[dict[str, Any]]:
        credentials = (
            self.db.query(TenantCredential)
            .filter(TenantCredential.tenant_id == tenant_id)
            .order_by(TenantCredential.created_at.desc())
            .all()
        )
        return [_metadata(item) for item in credentials]

    def revoke(self, tenant_id: str, credential_id: str) -> dict[str, Any]:
        credential = (
            self.db.query(TenantCredential)
            .filter(
                TenantCredential.id == credential_id,
                TenantCredential.tenant_id == tenant_id,
            )
            .first()
        )
        if credential is None:
            raise CredentialVaultError("Credential was not found")
        if not credential.revoked:
            credential.revoked = True
            credential.revoked_at = datetime.now(timezone.utc)
            self.db.commit()
            self.db.refresh(credential)
        return _metadata(credential)

    def decrypt_for_internal_use(self, tenant_id: str, credential_id: str) -> dict[str, str]:
        """Internal-only decrypt boundary; no current paper workflow calls this."""
        credential = (
            self.db.query(TenantCredential)
            .filter(
                TenantCredential.id == credential_id,
                TenantCredential.tenant_id == tenant_id,
                TenantCredential.revoked.is_(False),
            )
            .first()
        )
        if credential is None:
            raise CredentialVaultError("Active credential was not found")
        try:
            raw = AESGCM(self._key).decrypt(
                base64.urlsafe_b64decode(credential.nonce),
                base64.urlsafe_b64decode(credential.encrypted_payload),
                _associated_data(tenant_id, credential.kind, credential.provider),
            )
            payload = json.loads(raw)
        except Exception as exc:
            raise CredentialVaultError("Credential decryption failed") from exc
        if not isinstance(payload, dict) or not all(isinstance(value, str) for value in payload.values()):
            raise CredentialVaultError("Credential payload is invalid")
        return payload


def _decode_master_key(value: str | None) -> bytes:
    if not value:
        raise CredentialVaultError("Credential vault master key is not configured")
    try:
        key = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except ValueError as exc:
        raise CredentialVaultError("Credential vault master key is malformed") from exc
    if len(key) != 32:
        raise CredentialVaultError("Credential vault master key must decode to 32 bytes")
    return key


def _associated_data(tenant_id: str, kind: str, provider: str) -> bytes:
    return f"valuecell:credential:v1:{tenant_id}:{kind}:{provider}".encode("utf-8")


def _metadata(credential: TenantCredential) -> dict[str, Any]:
    return {
        "id": credential.id,
        "kind": credential.kind,
        "provider": credential.provider,
        "label": credential.label,
        "metadata": credential.metadata_json,
        "revoked": credential.revoked,
        "created_at": credential.created_at,
        "revoked_at": credential.revoked_at,
    }
