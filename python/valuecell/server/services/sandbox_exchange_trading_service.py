"""Explicit, tenant-isolated sandbox exchange trading; strategy scheduling remains paper-only."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

import ccxt.pro as ccxtpro
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valuecell.server.db.models.sandbox_exchange_order import SandboxExchangeOrder
from valuecell.server.db.models.tenant_credential import TenantCredential
from valuecell.server.services.tenant_credential_service import (
    CredentialVaultError,
    TenantCredentialService,
)

SandboxProvider = Literal["binance", "okx"]


class SandboxTradingError(ValueError):
    """Raised when an explicit sandbox operation cannot be performed safely."""


class SandboxExchangeTradingService:
    """Uses encrypted tenant credentials exclusively against exchange sandbox endpoints."""

    _PROVIDERS = frozenset({"binance", "okx"})

    def __init__(self, db: Session) -> None:
        self.db = db
        self.credentials = TenantCredentialService(db)

    def connection_metadata(self, tenant_id: str) -> list[dict[str, Any]]:
        credentials = (
            self.db.query(TenantCredential)
            .filter(
                TenantCredential.tenant_id == tenant_id,
                TenantCredential.kind == "exchange",
                TenantCredential.revoked.is_(False),
            )
            .order_by(TenantCredential.created_at.desc())
            .all()
        )
        return [self._connection_metadata(item) for item in credentials if self._is_sandbox_spot(item)]

    async def balance(self, tenant_id: str, credential_id: str) -> dict[str, Any]:
        credential = self._active_sandbox_credential(tenant_id, credential_id)
        exchange = self._exchange_for(tenant_id, credential)
        try:
            exchange.set_sandbox_mode(True)
            raw = await exchange.fetch_balance()
        finally:
            await self._close(exchange)
        totals = raw.get("total", {}) if isinstance(raw, dict) else {}
        free = raw.get("free", {}) if isinstance(raw, dict) else {}
        used = raw.get("used", {}) if isinstance(raw, dict) else {}
        currencies = sorted(set(totals) | set(free) | set(used))
        balances = []
        for currency in currencies:
            total, available, held = totals.get(currency), free.get(currency), used.get(currency)
            if any(self._nonzero(value) for value in (total, available, held)):
                balances.append({"currency": currency, "total": total or 0, "free": available or 0, "used": held or 0})
        return {"balances": balances, "checked_at": datetime.now(timezone.utc).isoformat()}

    async def submit_order(
        self,
        tenant_id: str,
        credential_id: str,
        client_order_id: str,
        symbol: str,
        side: Literal["buy", "sell"],
        order_type: Literal["market", "limit"],
        quote_amount: Decimal,
        price: Decimal | None,
    ) -> dict[str, Any]:
        existing = self._order_by_client_id(tenant_id, client_order_id)
        if existing is not None:
            return self._order_metadata(existing)
        credential = self._active_sandbox_credential(tenant_id, credential_id)
        quantity = None
        if order_type == "limit":
            if price is None:
                raise SandboxTradingError("Limit orders require a price")
            quantity = quote_amount / price
        order = SandboxExchangeOrder(
            tenant_id=tenant_id,
            credential_id=credential.id,
            provider=credential.provider,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            requested_quote=str(quote_amount),
            requested_quantity=str(quantity) if quantity is not None else None,
            status="pending",
            sandbox=True,
        )
        self.db.add(order)
        try:
            self.db.commit()
        except IntegrityError:
            self.db.rollback()
            existing = self._order_by_client_id(tenant_id, client_order_id)
            if existing is not None:
                return self._order_metadata(existing)
            raise
        self.db.refresh(order)

        exchange = self._exchange_for(tenant_id, credential)
        try:
            exchange.set_sandbox_mode(True)
            if quantity is None:
                ticker = await exchange.fetch_ticker(symbol)
                last = self._decimal(ticker.get("last") if isinstance(ticker, dict) else None, "Ticker price unavailable")
                quantity = quote_amount / last
                order.requested_quantity = str(quantity)
            exchange.set_sandbox_mode(True)
            exchange_order = await exchange.create_order(symbol, order_type, side, float(quantity), float(price) if price is not None else None, {"clientOrderId": client_order_id})
            order.status = str(exchange_order.get("status") or "submitted")
            exchange_id = exchange_order.get("id")
            order.exchange_order_id = str(exchange_id) if exchange_id is not None else None
            order.response_metadata = self._safe_exchange_metadata(exchange_order)
        except Exception:
            order.status = "failed"
            order.error_code = "sandbox_order_rejected"
        finally:
            await self._close(exchange)
        self.db.commit()
        self.db.refresh(order)
        return self._order_metadata(order)

    async def fetch_order_status(self, tenant_id: str, order_id: str) -> dict[str, Any]:
        """Refresh one submitted sandbox order, always through testnet before private fetch."""
        order = self.db.query(SandboxExchangeOrder).filter_by(id=order_id, tenant_id=tenant_id).first()
        if order is None:
            raise SandboxTradingError("Sandbox order was not found")
        if not order.exchange_order_id:
            return self._order_metadata(order)
        credential = self._active_sandbox_credential(tenant_id, order.credential_id)
        exchange = self._exchange_for(tenant_id, credential)
        try:
            exchange.set_sandbox_mode(True)
            raw = await exchange.fetch_order(order.exchange_order_id, order.symbol)
            order.status = str(raw.get("status") or order.status)
            order.response_metadata = self._safe_exchange_metadata(raw)
            self.db.commit()
            self.db.refresh(order)
        finally:
            await self._close(exchange)
        return self._order_metadata(order)

    def list_orders(self, tenant_id: str, credential_id: str | None = None) -> list[dict[str, Any]]:
        query = self.db.query(SandboxExchangeOrder).filter_by(tenant_id=tenant_id)
        if credential_id:
            query = query.filter_by(credential_id=credential_id)
        return [self._order_metadata(order) for order in query.order_by(SandboxExchangeOrder.created_at.desc()).all()]

    def _active_sandbox_credential(self, tenant_id: str, credential_id: str) -> TenantCredential:
        credential = self.db.query(TenantCredential).filter_by(id=credential_id, tenant_id=tenant_id, revoked=False).first()
        if credential is None or not self._is_sandbox_spot(credential):
            raise SandboxTradingError("Active sandbox spot credential was not found")
        return credential

    def _exchange_for(self, tenant_id: str, credential: TenantCredential) -> Any:
        if credential.provider not in self._PROVIDERS:
            raise SandboxTradingError("Unsupported sandbox exchange provider")
        try:
            secret = self.credentials.decrypt_for_internal_use(tenant_id, credential.id)
        except CredentialVaultError as exc:
            raise SandboxTradingError("Sandbox credential could not be decrypted") from exc
        config: dict[str, Any] = {"apiKey": secret.get("api_key"), "secret": secret.get("api_secret"), "enableRateLimit": True}
        if credential.provider == "okx":
            config["password"] = secret.get("passphrase")
        if not config["apiKey"] or not config["secret"] or (credential.provider == "okx" and not config["password"]):
            raise SandboxTradingError("Sandbox credential payload is incomplete")
        exchange_cls = getattr(ccxtpro, credential.provider, None)
        if exchange_cls is None:
            raise SandboxTradingError("Sandbox exchange provider is unavailable")
        return exchange_cls(config)

    @staticmethod
    async def _close(exchange: Any) -> None:
        try:
            await exchange.close()
        except Exception:
            pass

    @classmethod
    def _is_sandbox_spot(cls, credential: TenantCredential) -> bool:
        metadata = credential.metadata_json or {}
        return credential.provider in cls._PROVIDERS and metadata.get("sandbox") is True and metadata.get("market_type") == "spot"

    @staticmethod
    def _connection_metadata(credential: TenantCredential) -> dict[str, Any]:
        return {"id": credential.id, "provider": credential.provider, "label": credential.label, "metadata": credential.metadata_json, "created_at": credential.created_at}

    @staticmethod
    def _order_metadata(order: SandboxExchangeOrder) -> dict[str, Any]:
        return {"id": order.id, "credential_id": order.credential_id, "provider": order.provider, "client_order_id": order.client_order_id, "symbol": order.symbol, "side": order.side, "type": order.order_type, "requested_quote": order.requested_quote, "requested_quantity": order.requested_quantity, "status": order.status, "exchange_order_id": order.exchange_order_id, "sandbox": order.sandbox, "error_code": order.error_code, "created_at": order.created_at, "updated_at": order.updated_at}

    @staticmethod
    def _safe_exchange_metadata(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        return {key: raw[key] for key in ("id", "status", "symbol", "side", "type", "amount", "filled", "remaining", "price", "cost", "timestamp") if key in raw}

    @staticmethod
    def _nonzero(value: Any) -> bool:
        try:
            return Decimal(str(value or 0)) != 0
        except (InvalidOperation, ValueError):
            return False

    @staticmethod
    def _decimal(value: Any, error: str) -> Decimal:
        try:
            result = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise SandboxTradingError(error) from exc
        if result <= 0:
            raise SandboxTradingError(error)
        return result

    def _order_by_client_id(self, tenant_id: str, client_order_id: str) -> SandboxExchangeOrder | None:
        return self.db.query(SandboxExchangeOrder).filter_by(tenant_id=tenant_id, client_order_id=client_order_id).first()
