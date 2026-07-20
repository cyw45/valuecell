"""Explicit, tenant-isolated sandbox exchange trading; strategy scheduling remains paper-only."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

import asyncio
import ccxt.pro as ccxtpro
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valuecell.server.db.models.rule_strategy import (
    RuleStrategy,
    RuleStrategyEvaluationJournal,
    RuleStrategyExecutionIntent,
)
from valuecell.server.db.models.sandbox_exchange_order import SandboxExchangeOrder
from valuecell.server.db.models.tenant_credential import TenantCredential
from valuecell.server.services.tenant_credential_service import (
    CredentialVaultError,
    TenantCredentialService,
)

# These are deliberately shared service semantics, rather than exchange-specific
# strings.  In particular submission_unknown means the remote request may have
# reached the venue and must only be reconciled, never automatically retried.
INTENT_PENDING = "pending"
INTENT_SUBMITTING = "submitting"
INTENT_SUBMISSION_UNKNOWN = "submission_unknown"
INTENT_SUBMITTED = "submitted"
INTENT_TERMINAL = frozenset({"closed", "filled", "canceled", "cancelled", "failed", "rejected", "stale"})
ORDER_TERMINAL = frozenset({"closed", "filled", "canceled", "cancelled", "failed", "rejected"})

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

    async def positions(
        self,
        tenant_id: str,
        credential_id: str,
        *,
        account: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Derive real spot positions from the Demo account balance, not local orders."""
        # Reuse a cycle's balance when supplied; otherwise preserve the public
        # endpoint's standalone behavior.
        account = account or await self.balance(tenant_id, credential_id)
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
        *,
        intent: RuleStrategyExecutionIntent | None = None,
        fenced: bool = False,
        submission_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Persist the order before I/O; an attributed intent is never paper-filled.

        ``submission_unknown`` is intentionally not retried here: a transport
        exception may have created an exchange order.
        """
        symbol = self._canonical_exchange_symbol(symbol)
        if intent is not None:
            if intent.tenant_id != tenant_id or intent.credential_id != credential_id:
                raise SandboxTradingError("Execution intent attribution does not match order")
            client_order_id = intent.idempotency_key
            existing_intent_order = self.db.query(SandboxExchangeOrder).filter_by(execution_intent_id=intent.id).first()
            if existing_intent_order is not None:
                return self._order_metadata(existing_intent_order)
            if intent.status == INTENT_SUBMISSION_UNKNOWN:
                return self._intent_metadata(intent, None)
            # Last fence immediately before routing. Stop/update after the intent
            # commit turns it stale without making any exchange request.
            strategy_query = self.db.query(RuleStrategy).filter_by(
                strategy_id=intent.strategy_id, tenant_id=tenant_id
            )
            # Scheduler already owns this lock. Other callers acquire it here.
            strategy = strategy_query.first() if fenced else strategy_query.with_for_update().first()
            if strategy is None or strategy.status != "running" or strategy.execution_generation != intent.execution_generation:
                intent.status = "stale"
                intent.terminal_at = datetime.now(timezone.utc)
                intent.error_code = "stale_generation"
                if fenced:
                    self.db.flush()
                else:
                    self.db.commit()
                return self._intent_metadata(intent, None)
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
            **(
                {
                    "strategy_id": intent.strategy_id,
                    "evaluation_id": intent.evaluation_id,
                    "execution_generation": intent.execution_generation,
                    "execution_source": intent.execution_source,
                    "execution_intent_id": intent.id,
                }
                if intent is not None
                else {}
            ),
        )
        self.db.add(order)
        try:
            if fenced:
                self.db.flush()
            else:
                self.db.commit()
        except IntegrityError:
            self.db.rollback()
            existing = self._order_by_client_id(tenant_id, client_order_id)
            if existing is not None:
                return self._order_metadata(existing)
            raise
        self.db.refresh(order)
        if intent is not None:
            intent.status = INTENT_SUBMITTING
            # Scheduler-owned intents enter ``submitting`` in a separate durable
            # transaction before this fenced remote-I/O section. Do not count the
            # same remote submission twice.
            if intent.submitted_at is None:
                intent.attempt_count = (intent.attempt_count or 0) + 1
                intent.submitted_at = datetime.now(timezone.utc)
            if fenced:
                self.db.flush()
            else:
                self.db.commit()

        exchange = self._exchange_for(tenant_id, credential)
        try:
            exchange.set_sandbox_mode(True)
            # Market discovery, ticker and balance are preflight operations: a
            # timeout there proves no create request was sent, so it is terminal
            # validation failure rather than ambiguous exchange submission.
            markets = await self._await_preflight(exchange.load_markets(), submission_timeout_s)
            market = markets.get(symbol) if isinstance(markets, dict) else None
            if (
                not isinstance(market, dict)
                or not market.get("spot")
                or market.get("active") is False
                or market.get("quote") != "USDT"
            ):
                raise SandboxTradingError("Symbol is not an active USDT spot market on OKX Demo")
            ticker = await self._await_preflight(exchange.fetch_ticker(symbol), submission_timeout_s)
            market_price = self._decimal(
                ticker.get("last") if isinstance(ticker, dict) else None,
                "Ticker price unavailable",
            )
            effective_price = price or market_price
            raw_balance = await self._await_preflight(exchange.fetch_balance(), submission_timeout_s)
            free = raw_balance.get("free", {}) if isinstance(raw_balance, dict) else {}
            required_currency = "USDT" if side == "buy" else str(market.get("base") or "")
            available = self._decimal_or_zero(free.get(required_currency))
            nominal_quantity = quote_amount / effective_price
            # A sell's quote amount is a sizing/risk ceiling, not an instruction
            # to liquidate the whole shared Demo balance.
            quantity = (
                nominal_quantity if side == "buy" else min(available, nominal_quantity)
            )
            precision = getattr(exchange, "amount_to_precision", None)
            if callable(precision):
                quantity = self._decimal(precision(symbol, float(quantity)), "Order amount unavailable")
            limits = market.get("limits") or {}
            min_amount = self._decimal_or_zero((limits.get("amount") or {}).get("min"))
            min_cost = self._decimal_or_zero((limits.get("cost") or {}).get("min"))
            order_cost = quantity * effective_price
            # Precision adapters are expected to truncate. Fail closed if an
            # adapter ever rounds above either the nominal or available ceiling.
            if quantity > nominal_quantity or (side == "sell" and quantity > available):
                raise SandboxTradingError("Order amount exceeds safe Demo sizing limit")
            if quantity <= 0 or (min_amount > 0 and quantity < min_amount) or (min_cost > 0 and order_cost < min_cost):
                raise SandboxTradingError("Order does not satisfy OKX Demo minimum size")
            required_amount = quote_amount if side == "buy" else quantity
            if available < required_amount:
                raise SandboxTradingError("Insufficient available Demo balance")
            if order_type == "limit" and price is None:
                raise SandboxTradingError("Limit orders require a price")
            exchange.set_sandbox_mode(True)
            order.requested_quantity = str(quantity)
            if intent is not None:
                intent.requested_quantity = str(quantity)
                intent.request_payload = {
                    **(intent.request_payload or {}),
                    "order_cost": str(order_cost),
                }
            create = exchange.create_order(
                symbol,
                order_type,
                side,
                float(quantity),
                float(price) if price is not None else None,
                {"clientOrderId": client_order_id},
            )
            exchange_order = (
                await asyncio.wait_for(create, timeout=submission_timeout_s)
                if submission_timeout_s is not None
                else await create
            )
            order.status = self._normalise_status(exchange_order.get("status") or "submitted")
            exchange_id = exchange_order.get("id")
            order.exchange_order_id = str(exchange_id) if exchange_id is not None else None
            order.response_metadata = self._safe_exchange_metadata(exchange_order)
            if intent is not None:
                intent.status = order.status if order.status not in ORDER_TERMINAL else order.status
                if order.status in ORDER_TERMINAL:
                    intent.terminal_at = datetime.now(timezone.utc)
        except SandboxTradingError as exc:
            order.status = "failed"
            order.error_code = "sandbox_order_rejected"
            if intent is not None:
                intent.status = "failed"
                intent.error_code = order.error_code
                intent.error_message = str(exc)
                intent.terminal_at = datetime.now(timezone.utc)
        except asyncio.CancelledError:
            # Cancellation can race a remote create. Persist ambiguity before
            # propagating it so a caller's rollback cannot erase the audit row.
            order.status = INTENT_SUBMISSION_UNKNOWN
            order.error_code = "sandbox_submission_unknown"
            if intent is not None:
                intent.status = INTENT_SUBMISSION_UNKNOWN
                intent.error_code = order.error_code
                intent.error_message = "submission cancelled"
            self.db.commit()
            raise
        except Exception as exc:
            # The request could have reached the exchange. Preserve that ambiguity
            # and force reconciliation rather than issuing a duplicate create.
            order.status = INTENT_SUBMISSION_UNKNOWN
            order.error_code = "sandbox_submission_unknown"
            if intent is not None:
                intent.status = INTENT_SUBMISSION_UNKNOWN
                intent.error_code = order.error_code
                intent.error_message = str(exc)
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
            order.status = self._normalise_status(raw.get("status") or order.status)
            order.response_metadata = self._safe_exchange_metadata(raw)
            self._sync_intent_from_order(order)
            self._sync_evaluation_execution(order=order)
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
            and order.status not in {"filled", "canceled", "cancelled", "failed", "rejected", "stale"}
        ]
        refreshed = []
        for order_id in order_ids:
            try:
                refreshed.append(await self.fetch_order_status(tenant_id, order_id))
            except SandboxTradingError:
                continue
        return refreshed

    async def reconcile_nonterminal_intents(self, tenant_id: str) -> list[dict[str, Any]]:
        """Lookup ambiguous submissions by idempotency key; never issue create_order."""
        intents = self.db.query(RuleStrategyExecutionIntent).filter_by(tenant_id=tenant_id).all()
        results: list[dict[str, Any]] = []
        for intent in intents:
            # A crash after the durable ``submitting`` transition but before (or
            # during) create_order is just as ambiguous as a transport timeout.
            # Neither state is ever resubmitted here; both are lookup-only.
            if intent.status not in {INTENT_SUBMITTING, INTENT_SUBMISSION_UNKNOWN}:
                continue
            # Generation fencing blocks new remote submissions, not recovery of
            # a request already durable in ``submitting``/``submission_unknown``.
            # Even after stop or a connection switch, reconciliation must retain
            # and query the original attributed connection so any venue order is
            # auditable rather than silently being marked stale.
            try:
                credential = self._active_sandbox_credential(tenant_id, intent.credential_id)
                exchange = self._exchange_for(tenant_id, credential)
            except SandboxTradingError as exc:
                intent.error_code = "reconciliation_deferred"
                intent.error_message = str(exc)
                if intent.status == INTENT_SUBMITTING:
                    intent.status = INTENT_SUBMISSION_UNKNOWN
                self._sync_evaluation_execution(intent=intent)
                self.db.commit()
                results.append(self._intent_metadata(intent, None))
                continue
            try:
                exchange.set_sandbox_mode(True)
                raw = await self._find_exchange_order_by_client_id(exchange, intent.symbol, intent.idempotency_key)
                if raw is None:
                    intent.error_code = "reconciliation_required"
                    # A durable in-flight submission was not located in this polling
                    # pass. Preserve its ambiguity and let the scheduled lookup retry;
                    # never turn this into another create_order request.
                    if intent.status == INTENT_SUBMITTING:
                        intent.status = INTENT_SUBMISSION_UNKNOWN
                    self._sync_evaluation_execution(intent=intent)
                    self.db.commit()
                    results.append(self._intent_metadata(intent, None))
                    continue
                order = self._order_by_client_id(tenant_id, intent.idempotency_key)
                if order is None:
                    order = SandboxExchangeOrder(tenant_id=tenant_id, credential_id=intent.credential_id, provider=credential.provider, client_order_id=intent.idempotency_key, symbol=intent.symbol, side=intent.side, order_type=intent.order_type, requested_quote=intent.requested_quote, status="submitted", sandbox=True, strategy_id=intent.strategy_id, evaluation_id=intent.evaluation_id, execution_generation=intent.execution_generation, execution_source=intent.execution_source, execution_intent_id=intent.id)
                    self.db.add(order)
                order.status = self._normalise_status(raw.get("status") or "submitted")
                order.exchange_order_id = str(raw["id"]) if raw.get("id") is not None else None
                order.response_metadata = self._safe_exchange_metadata(raw)
                self._sync_intent_from_order(order)
                self._sync_evaluation_execution(order=order, intent=intent)
                self.db.commit()
                results.append(self._order_metadata(order))
            except Exception as exc:
                # Recovery errors must not kill the whole tenant loop or permit a
                # retrying order submission. Keep the intent safely ambiguous.
                intent.error_code = "reconciliation_deferred"
                intent.error_message = str(exc)
                if intent.status == INTENT_SUBMITTING:
                    intent.status = INTENT_SUBMISSION_UNKNOWN
                self._sync_evaluation_execution(intent=intent)
                self.db.commit()
                results.append(self._intent_metadata(intent, None))
            finally:
                await self._close(exchange)
        return results

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
    async def _await_preflight(awaitable: Any, timeout_s: float | None) -> Any:
        """Bound an exchange operation known to run before create_order."""
        try:
            return (
                await asyncio.wait_for(awaitable, timeout=timeout_s)
                if timeout_s is not None
                else await awaitable
            )
        except asyncio.TimeoutError as exc:
            raise SandboxTradingError("OKX Demo preflight timed out before submission") from exc

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
    def _canonical_exchange_symbol(symbol: str) -> str:
        return symbol.strip().upper().replace("-", "/")

    @staticmethod
    def _connection_metadata(credential: TenantCredential) -> dict[str, Any]:
        return {"id": credential.id, "provider": credential.provider, "label": credential.label, "metadata": credential.metadata_json, "created_at": credential.created_at}

    @staticmethod
    def _intent_metadata(intent: RuleStrategyExecutionIntent, order: SandboxExchangeOrder | None) -> dict[str, Any]:
        return {
            "execution_intent_id": intent.id,
            "id": order.id if order is not None else None,
            "status": intent.status if order is None else order.status,
            "error_code": intent.error_code if order is None else order.error_code,
            "attempt_count": intent.attempt_count,
        }

    @staticmethod
    def _order_metadata(order: SandboxExchangeOrder) -> dict[str, Any]:
        return {
            "id": order.id,
            "credential_id": order.credential_id,
            "provider": order.provider,
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "side": order.side,
            "type": order.order_type,
            "requested_quote": order.requested_quote,
            "requested_quantity": order.requested_quantity,
            "status": order.status,
            "exchange_order_id": order.exchange_order_id,
            "sandbox": order.sandbox,
            "error_code": order.error_code,
            "strategy_id": order.strategy_id,
            "evaluation_id": order.evaluation_id,
            "execution_generation": order.execution_generation,
            "execution_source": order.execution_source,
            "execution_intent_id": order.execution_intent_id,
            "created_at": order.created_at,
            "updated_at": order.updated_at,
        }

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

    @staticmethod
    def _normalise_status(status: Any) -> str:
        """Map CCXT terminal vocabulary to durable execution vocabulary."""
        value = str(status).lower()
        return {"closed": "filled", "canceled": "canceled", "cancelled": "cancelled"}.get(value, value)

    def _sync_intent_from_order(self, order: SandboxExchangeOrder) -> None:
        """Persist order and attributed-intent lifecycle atomically."""
        if not order.execution_intent_id:
            return
        intent = self.db.query(RuleStrategyExecutionIntent).filter_by(
            id=order.execution_intent_id, tenant_id=order.tenant_id
        ).first()
        if intent is None:
            return
        intent.status = order.status
        if order.status in INTENT_TERMINAL:
            intent.terminal_at = datetime.now(timezone.utc)
        if order.error_code:
            intent.error_code = order.error_code

    def _sync_evaluation_execution(
        self,
        *,
        order: SandboxExchangeOrder | None = None,
        intent: RuleStrategyExecutionIntent | None = None,
    ) -> None:
        """Merge later venue facts into an exactly tenant-attributed journal."""
        source = order or intent
        if source is None or getattr(source, "execution_source", None) != "rule_strategy":
            return
        tenant_id = getattr(source, "tenant_id", None)
        strategy_id = getattr(source, "strategy_id", None)
        evaluation_id = getattr(source, "evaluation_id", None)
        if not all((tenant_id, strategy_id, evaluation_id)):
            return
        if order is not None and intent is not None and any(
            getattr(order, field, None) != getattr(intent, field, None)
            for field in ("tenant_id", "strategy_id", "evaluation_id", "execution_generation")
        ):
            return
        journal = self.db.query(RuleStrategyEvaluationJournal).filter_by(
            tenant_id=tenant_id,
            strategy_id=strategy_id,
            evaluation_id=evaluation_id,
        ).first()
        if journal is None:
            return
        previous = (journal.result or {}).get("execution")
        execution = dict(previous) if isinstance(previous, dict) else {}
        status = order.status if order is not None else getattr(intent, "status", None)
        intent_id = order.execution_intent_id if order is not None else getattr(intent, "id", None)
        error_code = order.error_code if order is not None else getattr(intent, "error_code", None)
        execution.update(
            {
                "execution_ledger": "okx_demo",
                "paper_fill": False,
                "sandbox": True,
                "status": status,
                "execution_intent_id": intent_id,
                "order_id": order.id if order is not None else execution.get("order_id"),
                "error_code": error_code,
            }
        )
        if order is not None and isinstance(order.response_metadata, dict):
            for field in ("filled", "remaining", "amount", "cost", "price"):
                if field in order.response_metadata:
                    execution[field] = order.response_metadata[field]
        journal.result = {**(journal.result or {}), "execution": execution}

    @staticmethod
    async def _find_exchange_order_by_client_id(exchange: Any, symbol: str, client_order_id: str) -> dict[str, Any] | None:
        """Use portable CCXT history fallbacks because venues differ."""
        candidates: list[Any] = []
        for method_name in ("fetch_orders", "fetch_open_orders", "fetch_closed_orders"):
            method = getattr(exchange, method_name, None)
            if not callable(method):
                continue
            try:
                candidates.extend(await method(symbol))
            except Exception:
                continue
        for raw in candidates:
            if not isinstance(raw, dict):
                continue
            info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
            if client_order_id in (raw.get("clientOrderId"), raw.get("client_order_id"), info.get("clientOrderId"), info.get("clOrdId")):
                return raw
        return None
