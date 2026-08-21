"""Venue-neutral strategy order lifecycle with fail-closed submission semantics."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol
from uuid import uuid4

from sqlalchemy.orm import Session

from valuecell.server.db.connection import get_database_manager
from valuecell.server.db.models.rule_strategy import (
    RuleStrategyExecutionIntent,
    RuleStrategyFill,
    RuleStrategyOrderAttempt,
)


class SpotExchangeAdapter(Protocol):
    """Validated tenant-credential spot adapter contract."""

    venue: str

    async def symbol_rules(self, symbol: str) -> dict[str, Decimal]: ...
    async def balance(self) -> dict[str, Decimal]: ...
    async def positions(self) -> dict[str, Decimal]: ...
    async def best_bid_ask(self, symbol: str) -> tuple[Decimal, Decimal]: ...
    async def create_order(self, request: dict[str, Any]) -> dict[str, Any]: ...
    async def cancel_order(self, client_order_id: str) -> dict[str, Any]: ...
    async def fetch_order(self, client_order_id: str) -> dict[str, Any] | None: ...


@dataclass(frozen=True, slots=True)
class RuleStrategyOrderRequest:
    """Immutable order request used before any venue submission."""

    symbol: str
    side: str
    leg_kind: str
    quantity: Decimal
    decision_price: Decimal
    execution_target: str
    client_order_id: str
    credential_id: str | None = None


@dataclass(frozen=True, slots=True)
class RuleStrategyOrderPolicy:
    """Design-book pricing and pre-submit slippage rules."""

    limit_offset: Decimal = Decimal("0.998")
    max_market_slippage_pct: Decimal = Decimal("0.005")
    limit_timeout_seconds: int = 30

    def limit_price(self, decision_price: Decimal) -> Decimal:
        return decision_price * self.limit_offset

    def projected_market_slippage_pct(
        self, decision_price: Decimal, best_price: Decimal
    ) -> Decimal:
        return abs(best_price - decision_price) / decision_price

    def allow_market_order(
        self, decision_price: Decimal, best_price: Decimal
    ) -> bool:
        return self.projected_market_slippage_pct(decision_price, best_price) <= self.max_market_slippage_pct

    def request_payload(
        self,
        request: RuleStrategyOrderRequest,
        *,
        order_type: str,
        best_price: Decimal | None = None,
    ) -> dict[str, Any]:
        if order_type == "market" and best_price is not None and not self.allow_market_order(
            request.decision_price, best_price
        ):
            raise ValueError("projected market slippage exceeds 0.5%")
        requested_price = (
            self.limit_price(request.decision_price) if order_type == "limit" else None
        )
        return {
            "symbol": request.symbol,
            "side": request.side,
            "type": order_type,
            "quantity": str(request.quantity),
            "price": str(requested_price) if requested_price is not None else None,
            "client_order_id": request.client_order_id,
        }


class RuleStrategySubmissionUnknownError(RuntimeError):
    """Raised after a remote submission outcome cannot be established."""


class RuleStrategyExecutionCoordinator:
    """Persist intent/attempt/fill facts and never retry across venues."""

    def __init__(
        self,
        session: Session | None = None,
        *,
        policy: RuleStrategyOrderPolicy | None = None,
    ) -> None:
        self.session = session or get_database_manager().get_session()
        self._owns_session = session is None
        self.policy = policy or RuleStrategyOrderPolicy()

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def create_intent(
        self,
        *,
        tenant_id: str,
        strategy_id: str,
        evaluation_id: str,
        execution_generation: int,
        batch_id: str | None = None,
        request: RuleStrategyOrderRequest,
        requested_quote: Decimal,
    ) -> RuleStrategyExecutionIntent:
        """Stage an intent before adapter I/O, including paper intents."""
        intent = RuleStrategyExecutionIntent(
            id=str(uuid4()),
            tenant_id=tenant_id,
            strategy_id=strategy_id,
            evaluation_id=evaluation_id,
            execution_generation=execution_generation,
            batch_id=batch_id,
            execution_source="rule_strategy",
            credential_id=request.credential_id,
            idempotency_key=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            order_type="limit",
            requested_quote=str(requested_quote),
            requested_quantity=str(request.quantity),
            decision_price=str(request.decision_price),
            execution_target=request.execution_target,
            leg_kind=request.leg_kind,
            lifecycle_state="pending",
            status="pending",
            request_payload={
                "symbol": request.symbol,
                "side": request.side,
                "leg_kind": request.leg_kind,
                "decision_price": str(request.decision_price),
            },
        )
        self.session.add(intent)
        self.session.commit()
        self.session.refresh(intent)
        return intent

    async def submit(
        self,
        intent: RuleStrategyExecutionIntent,
        adapter: SpotExchangeAdapter,
        *,
        order_type: str = "limit",
    ) -> dict[str, Any]:
        """Submit once; timeout/error becomes submission_unknown on that venue."""
        if intent.lifecycle_state == "submission_unknown":
            raise RuleStrategySubmissionUnknownError(
                "ambiguous intent must be reconciled on its original venue"
            )
        request = RuleStrategyOrderRequest(
            symbol=intent.symbol,
            side=intent.side,
            leg_kind=intent.leg_kind,
            quantity=Decimal(intent.requested_quantity or "0"),
            decision_price=Decimal(intent.decision_price or "0"),
            execution_target=intent.execution_target,
            client_order_id=intent.idempotency_key,
            credential_id=intent.credential_id,
        )
        best_price = None
        if order_type == "market":
            bid, ask = await adapter.best_bid_ask(request.symbol)
            best_price = ask if request.side == "buy" else bid
        payload = self.policy.request_payload(
            request, order_type=order_type, best_price=best_price
        )
        intent.lifecycle_state = "submitting"
        intent.status = "submitting"
        intent.attempt_count = (intent.attempt_count or 0) + 1
        self.session.commit()
        attempt = RuleStrategyOrderAttempt(
            intent_id=intent.id,
            tenant_id=intent.tenant_id,
            venue=adapter.venue,
            client_order_id=intent.idempotency_key,
            requested_price=payload.get("price"),
            requested_quantity=str(request.quantity),
            status="submitting",
        )
        self.session.add(attempt)
        self.session.commit()
        try:
            response = await adapter.create_order(payload)
        except Exception as exc:
            intent.lifecycle_state = "submission_unknown"
            intent.status = "submission_unknown"
            intent.error_code = "submission_unknown"
            intent.error_message = "venue response could not be established"
            attempt.status = "submission_unknown"
            attempt.error_code = "submission_unknown"
            self.session.commit()
            raise RuleStrategySubmissionUnknownError(str(exc)) from exc
        attempt.status = str(response.get("status", "submitted"))
        attempt.venue_order_id = response.get("order_id")
        intent.lifecycle_state = "submitted"
        intent.status = "submitted"
        intent.accepted_quantity = str(response.get("filled_quantity", request.quantity))
        intent.accepted_quote = str(response.get("filled_quote", "0"))
        self.session.commit()
        return response

    def record_fill(
        self,
        intent: RuleStrategyExecutionIntent,
        attempt: RuleStrategyOrderAttempt,
        *,
        average_price: Decimal,
        quantity: Decimal,
        fee_quote: Decimal,
        remaining_quantity: Decimal,
        decision_price: Decimal,
        reconciliation_source: str,
    ) -> RuleStrategyFill:
        """Append one normalized fill and its realized slippage observation."""
        slippage = (average_price - decision_price) / decision_price
        fill = RuleStrategyFill(
            intent_id=intent.id,
            attempt_id=attempt.id,
            tenant_id=intent.tenant_id,
            average_price=str(average_price),
            quantity=str(quantity),
            fee_quote=str(fee_quote),
            remaining_quantity=str(remaining_quantity),
            observed_slippage_pct=str(slippage),
            reconciliation_source=reconciliation_source,
        )
        self.session.add(fill)
        intent.lifecycle_state = "filled" if remaining_quantity == 0 else "partially_filled"
        intent.status = intent.lifecycle_state
        self.session.commit()
        self.session.refresh(fill)
        return fill

    async def reconcile_same_venue(
        self, intent: RuleStrategyExecutionIntent, adapter: SpotExchangeAdapter
    ) -> dict[str, Any] | None:
        """Resolve ambiguity by client ID on the original venue only."""
        return await adapter.fetch_order(intent.idempotency_key)


class OKXDemoSpotAdapter:
    """Marker adapter type for validated OKX Demo spot credentials."""

    venue = "okx_demo"

    def __init__(self, transport: SpotExchangeAdapter) -> None:
        self._transport = transport

    def __getattr__(self, name: str) -> Any:
        return getattr(self._transport, name)


class OKXLiveSpotAdapter(OKXDemoSpotAdapter):
    """Marker adapter type reserved for the existing live authorization gates."""

    venue = "okx_live"


class BinanceSpotAdapter(OKXDemoSpotAdapter):
    """Validated Binance spot adapter wrapper."""

    venue = "binance"


class BybitSpotAdapter(OKXDemoSpotAdapter):
    """Validated Bybit spot adapter wrapper."""

    venue = "bybit"
