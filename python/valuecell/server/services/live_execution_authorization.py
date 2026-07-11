"""Process-bound, explicit runtime authorization for live execution."""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone


class LiveAuthorizationManager:
    """Keeps live authorization in memory so it disappears on process restart."""

    def __init__(self) -> None:
        self._challenges: dict[str, tuple[str, datetime]] = {}
        self._authorizations: dict[str, datetime] = {}

    def issue_challenge(self, tenant_id: str, ttl_s: int) -> dict[str, str]:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_s)
        code = secrets.token_urlsafe(12)
        self._challenges[tenant_id] = (code, expires_at)
        return {"challenge_code": code, "expires_at": expires_at.isoformat()}

    def confirm(self, tenant_id: str, code: str, ttl_s: int) -> datetime | None:
        challenge = self._challenges.pop(tenant_id, None)
        if challenge is None:
            return None
        expected, expires_at = challenge
        if expires_at <= datetime.now(timezone.utc) or not secrets.compare_digest(expected, code):
            return None
        authorization_expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_s)
        self._authorizations[tenant_id] = authorization_expires_at
        return authorization_expires_at

    def active_until(self, tenant_id: str) -> datetime | None:
        expires_at = self._authorizations.get(tenant_id)
        if expires_at is None:
            return None
        if expires_at <= datetime.now(timezone.utc):
            self._authorizations.pop(tenant_id, None)
            return None
        return expires_at

    def revoke(self, tenant_id: str) -> None:
        self._challenges.pop(tenant_id, None)
        self._authorizations.pop(tenant_id, None)


live_authorization_manager = LiveAuthorizationManager()
