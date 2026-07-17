"""Sandbox-only exchange credential readiness validation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

import ccxt.pro as ccxtpro

SandboxProvider = Literal["binance", "okx"]


class SandboxExchangeService:
    """Validates testnet credentials without persisting or exposing them."""

    async def validate(
        self,
        provider: SandboxProvider,
        api_key: str,
        api_secret: str,
        passphrase: str | None = None,
    ) -> dict[str, Any]:
        """Check sandbox credentials with one authenticated, read-only request."""
        exchange_cls = getattr(ccxtpro, provider, None)
        checked_at = datetime.now(timezone.utc)
        if exchange_cls is None:
            return self._metadata(provider, False, checked_at)

        config: dict[str, Any] = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        }
        if provider == "okx":
            config["password"] = passphrase

        exchange = exchange_cls(config)
        try:
            # ccxt.pro must switch endpoints before its only network operation.
            exchange.set_sandbox_mode(True)
            await exchange.fetch_balance()
        except Exception as exc:
            return self._metadata(
                provider,
                False,
                checked_at,
                error_code=self._classify_error(exc),
            )
        finally:
            try:
                await exchange.close()
            except Exception:
                pass

        return self._metadata(provider, True, checked_at)

    @staticmethod
    def _metadata(
        provider: SandboxProvider,
        validated: bool,
        checked_at: datetime,
        error_code: str | None = None,
    ) -> dict[str, Any]:
        metadata = {
            "provider": provider,
            "sandbox": True,
            "validated": validated,
            "checked_at": checked_at.isoformat(),
        }
        if error_code is not None:
            metadata["error_code"] = error_code
        return metadata

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        """Return a safe, stable category without exposing exchange response details."""
        message = str(exc).lower()
        if "nonetype" in message and "supported between" in message:
            return "market_metadata_error"
        if "restricted location" in message or "http 451" in message:
            return "region_restricted"
        if "timed out" in message or "network" in message or "connection" in message:
            return "network_error"
        if "403" in message or "forbidden" in message or "ip" in message and "whitelist" in message:
            return "ip_whitelist_or_access_error"
        return "credential_or_permission_error"
