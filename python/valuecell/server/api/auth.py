"""JWT authentication dependencies for SaaS-owned endpoints."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from valuecell.server.db.connection import get_db
from valuecell.server.db.models.tenant import SaaSUser, TenantMembership
from valuecell.server.services.saas_access_service import TenantAccessService
from valuecell.server.config.settings import get_settings

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentPrincipal:
    """Verified identity, workspace role, and server-derived commercial access."""

    user_id: str
    tenant_id: str
    role: str = "owner"
    is_platform_admin: bool = False
    access_status: str = "active"
    commercial_model: str | None = None
    access_expires_at: str | None = None


def hash_password(password: str, salt: str | None = None) -> str:
    """Derive a password hash using scrypt and a per-user random salt."""
    if len(password) < 12:
        raise ValueError("Password must contain at least 12 characters")
    raw_salt = (
        secrets.token_bytes(16) if salt is None else base64.urlsafe_b64decode(salt)
    )
    derived = hashlib.scrypt(password.encode("utf-8"), salt=raw_salt, n=2**14, r=8, p=1)
    return f"{base64.urlsafe_b64encode(raw_salt).decode()}${base64.urlsafe_b64encode(derived).decode()}"


def verify_password(password: str, encoded: str) -> bool:
    """Verify a password against a stored scrypt representation."""
    try:
        salt, expected = encoded.split("$", 1)
        actual = hash_password(password, salt).split("$", 1)[1]
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def create_access_token(principal: CurrentPrincipal) -> str:
    """Create a short-lived HS256 JWT with explicit user/tenant scope."""
    settings = get_settings()
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": principal.user_id,
        "tenant_id": principal.tenant_id,
        "iss": settings.JWT_ISSUER,
        "iat": now,
        "exp": now + settings.JWT_ACCESS_TOKEN_TTL_S,
    }
    return _encode_jwt(header, payload, settings.JWT_SECRET)


def get_current_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> CurrentPrincipal:
    """Validate a bearer token and confirm its user/workspace scope remains active."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Bearer authentication is required")
    settings = get_settings()
    payload = _decode_jwt(credentials.credentials, settings.JWT_SECRET)
    if payload is None or payload.get("iss") != settings.JWT_ISSUER:
        raise HTTPException(status_code=401, detail="Invalid access token")
    if int(payload.get("exp", 0)) <= int(time.time()):
        raise HTTPException(status_code=401, detail="Access token has expired")
    user_id = payload.get("sub")
    tenant_id = payload.get("tenant_id")
    if not isinstance(user_id, str) or not isinstance(tenant_id, str):
        raise HTTPException(status_code=401, detail="Access token scope is invalid")
    membership = (
        db.query(TenantMembership)
        .filter(
            TenantMembership.user_id == user_id,
            TenantMembership.tenant_id == tenant_id,
        )
        .first()
    )
    user = db.query(SaaSUser).filter(SaaSUser.id == user_id).first()
    if membership is None or user is None:
        raise HTTPException(
            status_code=401, detail="Access token scope is no longer active"
        )
    access = TenantAccessService.access_for(db, tenant_id)
    admin_emails = set(getattr(settings, "PLATFORM_ADMIN_EMAILS", ()))
    is_platform_admin = user.email.lower() in admin_emails
    return CurrentPrincipal(
        user_id=user_id,
        tenant_id=tenant_id,
        role=membership.role,
        is_platform_admin=is_platform_admin,
        access_status="active" if is_platform_admin else access.status,
        commercial_model=access.commercial_model,
        access_expires_at=access.expires_at.isoformat() if access.expires_at else None,
    )


def _encode_jwt(header: dict, payload: dict, secret: str) -> str:
    header_part = _urlsafe_json(header)
    payload_part = _urlsafe_json(payload)
    signed = f"{header_part}.{payload_part}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).digest()
    return f"{header_part}.{payload_part}.{_urlsafe_b64encode(signature)}"


def _decode_jwt(token: str, secret: str) -> dict | None:
    try:
        header_part, payload_part, signature_part = token.split(".")
        if (
            _urlsafe_b64encode(_urlsafe_b64decode(header_part)) != header_part
            or _urlsafe_b64encode(_urlsafe_b64decode(payload_part)) != payload_part
            or _urlsafe_b64encode(_urlsafe_b64decode(signature_part)) != signature_part
        ):
            return None
        signed = f"{header_part}.{payload_part}".encode("ascii")
        expected = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _urlsafe_b64decode(signature_part)):
            return None
        header = json.loads(_urlsafe_b64decode(header_part))
        if header.get("alg") != "HS256":
            return None
        return json.loads(_urlsafe_b64decode(payload_part))
    except (ValueError, json.JSONDecodeError):
        return None


def _urlsafe_json(value: dict) -> str:
    return _urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode("utf-8"))


def _urlsafe_b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _urlsafe_b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
