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
        except Exception:
            return self._metadata(provider, False, checked_at)
        finally:
            try:
                await exchange.close()
            except Exception:
                pass

        return self._metadata(provider, True, checked_at)

    @staticmethod
    def _metadata(
        provider: SandboxProvider, validated: bool, checked_at: datetime
    ) -> dict[str, Any]:
        return {
            "provider": provider,
            "sandbox": True,
            "validated": validated,
            "checked_at": checked_at.isoformat(),
        }
