"""Side-aware fixed-strategy Paper fills and position accounting."""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy.orm import Session

from valuecell.server.api.schemas.fixed_strategy import FixedStrategySignal
from valuecell.server.db.models.fixed_strategy_paper import (
    FixedPaperAccount,
    FixedPaperFill,
    FixedPaperPosition,
)


class FixedPaperLedgerError(RuntimeError):
    """Raised when a fixed-strategy Paper fill violates ledger invariants."""


class FixedPaperLedger:
    """Record deterministic Paper fills with independent long/short positions."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def account(
        self,
        *,
        tenant_id: str,
        strategy_id: str,
        batch_id: str,
        initial_capital_quote: Decimal,
    ) -> FixedPaperAccount:
        """Get or create one batch-scoped fixed strategy Paper account."""
        account = (
            self._session.query(FixedPaperAccount)
            .filter_by(
                tenant_id=tenant_id,
                strategy_id=strategy_id,
                batch_id=batch_id,
            )
            .with_for_update()
            .first()
        )
        if account is None:
            if initial_capital_quote <= 0:
                raise FixedPaperLedgerError("initial Paper capital must be positive")
            account = FixedPaperAccount(
                tenant_id=tenant_id,
                strategy_id=strategy_id,
                batch_id=batch_id,
                initial_capital_quote=float(initial_capital_quote),
                quote_balance=float(initial_capital_quote),
            )
            self._session.add(account)
            self._session.flush()
        return account

    def apply_signal(
        self,
        *,
        account: FixedPaperAccount,
        signal: FixedStrategySignal,
        evaluation_id: str,
        price: Decimal,
        quantity: Decimal,
        fee_quote: Decimal = Decimal("0"),
    ) -> FixedPaperFill | None:
        """Apply one executable signal as a confirmed full Paper fill."""
        if signal.action not in {"long_entry", "short_entry", "exit"}:
            return None
        if price <= 0 or quantity <= 0 or fee_quote < 0:
            raise FixedPaperLedgerError("Paper fill values must be positive")
        key = f"{evaluation_id}:{signal.action}:{signal.symbol}"
        duplicate = (
            self._session.query(FixedPaperFill)
            .filter_by(tenant_id=account.tenant_id, idempotency_key=key)
            .first()
        )
        if duplicate is not None:
            return duplicate
        position = (
            self._session.query(FixedPaperPosition)
            .filter_by(
                tenant_id=account.tenant_id,
                strategy_id=account.strategy_id,
                batch_id=account.batch_id,
                symbol=signal.symbol,
                status="open",
            )
            .with_for_update()
            .first()
        )
        quote_amount = price * quantity
        if signal.action in {"long_entry", "short_entry"}:
            if position is not None:
                raise FixedPaperLedgerError("one open position per symbol is required")
            if signal.action == "long_entry" and Decimal(str(account.quote_balance)) < quote_amount + fee_quote:
                raise FixedPaperLedgerError("Paper account has insufficient quote balance")
            side = "long" if signal.action == "long_entry" else "short"
            position = FixedPaperPosition(
                account_id=account.account_id,
                tenant_id=account.tenant_id,
                strategy_id=account.strategy_id,
                batch_id=account.batch_id,
                symbol=signal.symbol,
                pair=signal.pair,
                side=side,
                quantity=float(quantity),
                entry_price=float(price),
                entry_quote=float(quote_amount),
                entry_timestamp_ms=int(signal.observed_at.timestamp() * 1000),
            )
            self._session.add(position)
            account.quote_balance += float(quote_amount - fee_quote) if side == "short" else -float(quote_amount + fee_quote)
            account.occupied_quote += float(quote_amount)
            realized_pnl = Decimal("0")
        else:
            if position is None or quantity != Decimal(str(position.quantity)):
                raise FixedPaperLedgerError("Paper exit must close the matching full position")
            entry_price = Decimal(str(position.entry_price))
            realized_pnl = (price - entry_price) * quantity - fee_quote if position.side == "long" else (entry_price - price) * quantity - fee_quote
            account.quote_balance += float(quote_amount - fee_quote) if position.side == "long" else -float(quote_amount + fee_quote)
            account.occupied_quote = max(0.0, account.occupied_quote - float(position.entry_quote))
            position.status = "closed"
        fill = FixedPaperFill(
            account_id=account.account_id,
            position_id=position.position_id,
            tenant_id=account.tenant_id,
            strategy_id=account.strategy_id,
            batch_id=account.batch_id,
            evaluation_id=evaluation_id,
            idempotency_key=key,
            symbol=signal.symbol,
            pair=signal.pair,
            action=signal.action,
            side="buy" if signal.action in {"long_entry", "short_entry"} and position.side == "long" else "sell",
            quantity=float(quantity),
            price=float(price),
            quote_amount=float(quote_amount),
            fee_quote=float(fee_quote),
            realized_pnl_quote=float(realized_pnl),
        )
        account.realized_pnl_quote += float(realized_pnl)
        self._session.add(fill)
        self._session.flush()
        return fill
