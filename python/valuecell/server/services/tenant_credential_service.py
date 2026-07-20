"""Tenant-scoped encrypted credential vault for non-live SaaS configuration."""

from __future__ import annotations

import base64
import json
import math
import os
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy.orm import Session

from valuecell.server.config.settings import get_settings
from valuecell.server.db.models.tenant_credential import TenantCredential

_ALLOWED_KINDS = {"exchange", "market_data"}
_FORBIDDEN_FIELDS = {"private_key", "wallet_address", "signature", "seed_phrase"}
_SENSITIVE_METADATA_KEYS = {
    "apikey",
    "apisecret",
    "secret",
    "clientsecret",
    "authorization",
    "credential",
    "credentials",
    "passphrase",
    "token",
    "accesstoken",
    "refreshtoken",
    "password",
    "privatekey",
    "seedphrase",
    "signature",
}
_ALLOWED_METADATA_KEYS = {
    "exchange": frozenset(
        {
            "active",
            "environment",
            "ip_allowlist_confirmed",
            "market_type",
            "provider",
            "sandbox",
            "validated_at",
            "verified",
            "withdrawal_disabled_confirmed",
        }
    ),
    # There are currently no persisted market-data-specific metadata fields.
    # Keep only the generic, non-secret descriptors until a provider schema is
    # introduced alongside a concrete use case.
    "market_data": frozenset({"environment", "provider", "validated_at"}),
}
_MAX_METADATA_DEPTH = 5
_MAX_METADATA_KEYS = 64
_MAX_METADATA_BYTES = 8192


class CredentialVaultError(ValueError):
    """Raised when credential input or encryption configuration is unsafe."""


class TenantCredentialService:
    """Encrypts at rest and exposes metadata-only tenant credential operations."""

    def __init__(self, db: Session, master_key: str | None = None) -> None:
        self.db = db
        configured_key = (
            master_key
            if master_key is not None
            else get_settings().CREDENTIAL_MASTER_KEY
        )
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
        *,
        allow_live_metadata: bool = False,
    ) -> dict[str, Any]:
        if kind not in _ALLOWED_KINDS:
            raise CredentialVaultError("Credential kind is not allowed")
        if not provider.strip() or not label.strip():
            raise CredentialVaultError("Provider and label are required")
        if not secret or any(key in _FORBIDDEN_FIELDS for key in secret):
            raise CredentialVaultError(
                "Credential payload contains unsupported secret fields"
            )
        if any(not isinstance(value, str) or not value for value in secret.values()):
            raise CredentialVaultError(
                "Credential secret values must be non-empty strings"
            )
        safe_metadata = _validate_metadata(kind, metadata or {})
        if (
            kind == "exchange"
            and not allow_live_metadata
            and (
                safe_metadata.get("environment") == "live"
                or safe_metadata.get("active") is True
            )
        ):
            safe_metadata = {
                **safe_metadata,
                "environment": "unverified",
                "active": False,
                "verified": False,
            }
        nonce = os.urandom(12)
        payload = json.dumps(secret, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        ciphertext = AESGCM(self._key).encrypt(
            nonce, payload, _associated_data(tenant_id, kind, provider)
        )
        credential = TenantCredential(
            tenant_id=tenant_id,
            created_by_user_id=user_id,
            kind=kind,
            provider=provider.strip(),
            label=label.strip(),
            encrypted_payload=base64.urlsafe_b64encode(ciphertext).decode("ascii"),
            nonce=base64.urlsafe_b64encode(nonce).decode("ascii"),
            metadata_json=safe_metadata,
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

    def decrypt_for_internal_use(
        self, tenant_id: str, credential_id: str
    ) -> dict[str, str]:
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
        if not isinstance(payload, dict) or not all(
            isinstance(value, str) for value in payload.values()
        ):
            raise CredentialVaultError("Credential payload is invalid")
        return payload


def _validate_metadata(kind: str, metadata: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = _ALLOWED_METADATA_KEYS[kind]
    key_count = 0

    def walk(value: Any, depth: int) -> Any:
        nonlocal key_count
        if depth > _MAX_METADATA_DEPTH:
            raise CredentialVaultError("Credential metadata is too deeply nested")
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, child in value.items():
                if not isinstance(key, str):
                    raise CredentialVaultError(
                        "Credential metadata keys must be strings"
                    )
                key_count += 1
                if key_count > _MAX_METADATA_KEYS:
                    raise CredentialVaultError(
                        "Credential metadata contains too many keys"
                    )
                normalized_key = unicodedata.normalize("NFKC", key).casefold()
                compact_key = re.sub(r"[^a-z0-9]", "", normalized_key)
                if compact_key in _SENSITIVE_METADATA_KEYS:
                    raise CredentialVaultError(
                        "Credential metadata contains a sensitive field"
                    )
                canonical_key = re.sub(r"[^a-z0-9]+", "_", normalized_key).strip("_")
                if canonical_key not in allowed_keys:
                    raise CredentialVaultError(
                        "Credential metadata field is not allowed"
                    )
                if canonical_key in result:
                    raise CredentialVaultError(
                        "Credential metadata contains duplicate normalized keys"
                    )
                result[canonical_key] = walk(child, depth + 1)
            return result
        if isinstance(value, list):
            return [walk(child, depth + 1) for child in value]
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise CredentialVaultError("Credential metadata numbers must be finite")
            return value
        raise CredentialVaultError("Credential metadata contains an unsupported value")

    safe_metadata = walk(metadata, 1)
    try:
        encoded = json.dumps(
            safe_metadata,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CredentialVaultError(
            "Credential metadata must be JSON serializable"
        ) from exc
    if len(encoded) > _MAX_METADATA_BYTES:
        raise CredentialVaultError("Credential metadata is too large")
    return safe_metadata


def _decode_master_key(value: str | None) -> bytes:
    if not value:
        raise CredentialVaultError("Credential vault master key is not configured")
    try:
        key = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except ValueError as exc:
        raise CredentialVaultError("Credential vault master key is malformed") from exc
    if len(key) != 32:
        raise CredentialVaultError(
            "Credential vault master key must decode to 32 bytes"
        )
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
