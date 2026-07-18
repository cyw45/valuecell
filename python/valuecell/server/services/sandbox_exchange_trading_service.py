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
        """Return the exchange's Demo spot balance, with conservative USDT valuation.

        Assets without a direct active ``ASSET/USDT`` ticker are retained but marked
        unpriced; they are never silently treated as zero-value positions.
        """
        credential = self._active_sandbox_credential(tenant_id, credential_id)
        exchange = self._exchange_for(tenant_id, credential)
        try:
            exchange.set_sandbox_mode(True)
            raw = await exchange.fetch_balance()
            totals = raw.get("total", {}) if isinstance(raw, dict) else {}
            free = raw.get("free", {}) if isinstance(raw, dict) else {}
            used = raw.get("used", {}) if isinstance(raw, dict) else {}
            currencies = sorted(set(totals) | set(free) | set(used))
            balances: list[dict[str, Any]] = []
            total_value = Decimal("0")
            for currency in currencies:
                total = self._decimal_or_zero(totals.get(currency))
                available = self._decimal_or_zero(free.get(currency))
                held = self._decimal_or_zero(used.get(currency))
                if not any(value != 0 for value in (total, available, held)):
                    continue
                mark_price: Decimal | None = Decimal("1") if currency == "USDT" else None
                if currency != "USDT":
                    try:
                        ticker = await exchange.fetch_ticker(f"{currency}/USDT")
                        mark_price = self._decimal(
                            ticker.get("last") if isinstance(ticker, dict) else None,
                            "Ticker price unavailable",
                        )
                    except Exception:
                        mark_price = None
                value = total * mark_price if mark_price is not None else None
                if value is not None:
                    total_value += value
                balances.append(
                    {
                        "currency": currency,
                        "total": float(total),
                        "free": float(available),
                        "used": float(held),
                        "frozen": float(held),
                        "mark_price_usdt": float(mark_price) if mark_price is not None else None,
                        "usdt_value": float(value) if value is not None else None,
                        "valuation_status": "priced" if value is not None else "unpriced",
                    }
                )
            return {
                "source": f"{credential.provider}_demo",
                "balances": balances,
                "total_usdt_value": float(total_value),
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }
        finally:
            await self._close(exchange)

    async def positions(self, tenant_id: str, credential_id: str) -> dict[str, Any]:
        """Derive real spot positions from the Demo account balance, not local orders."""
        account = await self.balance(tenant_id, credential_id)
        positions = []
        for balance in account["balances"]:
            if balance["currency"] == "USDT" or not self._nonzero(balance["total"]):
                continue
            positions.append(
                {
                    "symbol": f"{balance['currency']}/USDT",
                    "base_currency": balance["currency"],
                    "quantity": balance["total"],
                    "available_quantity": balance["free"],
                    "frozen_quantity": balance["used"],
                    "mark_price": balance["mark_price_usdt"],
                    "notional_usdt": balance["usdt_value"],
                    "unrealized_pnl_usdt": None,
                }
            )
        return {"source": account["source"], "positions": positions, "checked_at": account["checked_at"]}

    async def tradable_symbols(self, tenant_id: str, credential_id: str) -> list[dict[str, Any]]:
        """Return the broad, exchange-authoritative active USDT spot universe."""
        credential = self.validate_strategy_connection(tenant_id, credential_id)
        exchange = self._exchange_for(tenant_id, credential)
        try:
            exchange.set_sandbox_mode(True)
            markets = await exchange.load_markets()
            result = []
            for symbol, market in markets.items():
                if not isinstance(market, dict) or not market.get("spot") or market.get("active") is False:
                    continue
                if market.get("quote") != "USDT" or not market.get("base"):
                    continue
                result.append({"symbol": symbol, "base": market["base"], "quote": "USDT"})
            return sorted(result, key=lambda item: item["symbol"])
        finally:
            await self._close(exchange)

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
            markets = await exchange.load_markets()
            market = markets.get(symbol) if isinstance(markets, dict) else None
            if (
                not isinstance(market, dict)
                or not market.get("spot")
                or market.get("active") is False
                or market.get("quote") != "USDT"
            ):
                raise SandboxTradingError("Symbol is not an active USDT spot market on OKX Demo")
            ticker = await exchange.fetch_ticker(symbol)
            market_price = self._decimal(
                ticker.get("last") if isinstance(ticker, dict) else None,
                "Ticker price unavailable",
            )
            effective_price = price or market_price
            quantity = quote_amount / effective_price
            precision = getattr(exchange, "amount_to_precision", None)
            if callable(precision):
                quantity = self._decimal(precision(symbol, float(quantity)), "Order amount unavailable")
            limits = market.get("limits") or {}
            min_amount = self._decimal_or_zero((limits.get("amount") or {}).get("min"))
            min_cost = self._decimal_or_zero((limits.get("cost") or {}).get("min"))
            if quantity <= 0 or (min_amount > 0 and quantity < min_amount) or (min_cost > 0 and quote_amount < min_cost):
                raise SandboxTradingError("Order does not satisfy OKX Demo minimum size")
            raw_balance = await exchange.fetch_balance()
            free = raw_balance.get("free", {}) if isinstance(raw_balance, dict) else {}
            required_currency = "USDT" if side == "buy" else str(market.get("base") or "")
            required_amount = quote_amount if side == "buy" else quantity
            if self._decimal_or_zero(free.get(required_currency)) < required_amount:
                raise SandboxTradingError("Insufficient available Demo balance")
            if order_type == "limit" and price is None:
                raise SandboxTradingError("Limit orders require a price")
            exchange.set_sandbox_mode(True)
            exchange_order = await exchange.create_order(
                symbol,
                order_type,
                side,
                float(quantity),
                float(price) if price is not None else None,
                {"clientOrderId": client_order_id},
            )
            order.requested_quantity = str(quantity)
            order.status = str(exchange_order.get("status") or "submitted")
            exchange_id = exchange_order.get("id")
            order.exchange_order_id = str(exchange_id) if exchange_id is not None else None
            order.response_metadata = self._safe_exchange_metadata(exchange_order)
        except SandboxTradingError:
            order.status = "failed"
            order.error_code = "sandbox_order_rejected"
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

    async def refresh_open_orders(
        self, tenant_id: str, credential_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Poll non-terminal orders so UI and risk checks do not rely on stale pending states."""
        query = self.db.query(SandboxExchangeOrder).filter_by(tenant_id=tenant_id)
        if credential_id:
            query = query.filter_by(credential_id=credential_id)
        order_ids = [
            order.id
            for order in query.all()
            if getattr(order, "exchange_order_id", None)
            and order.status not in {"closed", "canceled", "cancelled", "failed", "rejected"}
        ]
        refreshed = []
        for order_id in order_ids:
            try:
                refreshed.append(await self.fetch_order_status(tenant_id, order_id))
            except SandboxTradingError:
                continue
        return refreshed

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

    def _is_sandbox_spot(self, credential: TenantCredential) -> bool:
        metadata = credential.metadata_json or {}
        return credential.provider in self._PROVIDERS and metadata.get("sandbox") is True and metadata.get("market_type") == "spot"

    def validate_strategy_connection(self, tenant_id: str, credential_id: str) -> TenantCredential:
        credential = self._active_sandbox_credential(tenant_id, credential_id)
        if credential.provider != "okx":
            raise SandboxTradingError("Only an OKX Demo spot connection can execute a strategy")
        return credential

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
    def _decimal_or_zero(value: Any) -> Decimal:
        try:
            return Decimal(str(value or 0))
        except (InvalidOperation, ValueError, TypeError):
            return Decimal("0")

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
