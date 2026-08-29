"""Transactional shared-wallet capital reservations for concurrent strategies."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal
from uuid import uuid4

from sqlalchemy.orm import Session

from valuecell.server.db.models.multi_strategy import (
    StrategyCapitalReservation,
    StrategySharedAccount,
)


class CapitalAllocationError(RuntimeError):
    """Raised when one strategy cannot safely reserve shared account capital."""


class SharedCapitalAllocator:
    """Reserve, occupy, and release funds without assigning wallet assets to a strategy."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def reserve(
        self,
        *,
        account_id: str,
        tenant_id: str,
        strategy_id: str,
        batch_id: str | None,
        idempotency_key: str,
        symbol: str,
        side: str,
        requested_quote: Decimal,
    ) -> StrategyCapitalReservation:
        """Atomically reserve quote funds before one strategy submits an order."""

        if requested_quote <= 0:
            raise CapitalAllocationError("requested quote must be positive")
        existing = (
            self._session.query(StrategyCapitalReservation)
            .filter_by(tenant_id=tenant_id, idempotency_key=idempotency_key)
            .first()
        )
        if existing is not None:
            return existing
        account = (
            self._session.query(StrategySharedAccount)
            .filter_by(id=account_id, tenant_id=tenant_id, active=True)
            .with_for_update()
            .first()
        )
        if account is None:
            raise CapitalAllocationError("shared account is unavailable")
        if account.available_quote is None:
            raise CapitalAllocationError("shared account quote availability is unknown")
        available = Decimal(str(account.available_quote))
        reserved = Decimal(str(account.reserved_quote))
        if available - reserved < requested_quote:
            raise CapitalAllocationError("shared account has insufficient unreserved quote")
        reservation = StrategyCapitalReservation(
            reservation_id=str(uuid4()),
            account_id=account_id,
            tenant_id=tenant_id,
            strategy_id=strategy_id,
            batch_id=batch_id,
            idempotency_key=idempotency_key,
            symbol=symbol,
            side=side,
            requested_quote=float(requested_quote),
            reserved_quote=float(requested_quote),
            status="reserved",
        )
        account.reserved_quote = float(reserved + requested_quote)
        self._session.add(reservation)
        self._session.flush()
        return reservation

    def settle(
        self,
        reservation_id: str,
        *,
        consumed_quote: Decimal,
        outcome: Literal["occupied", "released", "partially_released"],
        reason: str | None = None,
    ) -> StrategyCapitalReservation:
        """Consume or release exactly one reservation after a confirmed outcome."""

        if consumed_quote < 0:
            raise CapitalAllocationError("consumed quote cannot be negative")
        reservation = (
            self._session.query(StrategyCapitalReservation)
            .filter_by(reservation_id=reservation_id)
            .with_for_update()
            .first()
        )
        if reservation is None:
            raise CapitalAllocationError("capital reservation was not found")
        if reservation.status not in {"reserved", "partially_released"}:
            raise CapitalAllocationError("capital reservation is already terminal")
        reserved_quote = Decimal(str(reservation.reserved_quote))
        if consumed_quote > reserved_quote:
            raise CapitalAllocationError("consumed quote exceeds reservation")
        account = (
            self._session.query(StrategySharedAccount)
            .filter_by(id=reservation.account_id, tenant_id=reservation.tenant_id)
            .with_for_update()
            .first()
        )
        if account is None:
            raise CapitalAllocationError("shared account is unavailable")
        released_quote = reserved_quote - consumed_quote
        account.reserved_quote = float(
            max(Decimal(0), Decimal(str(account.reserved_quote)) - reserved_quote)
        )
        account.occupied_notional_quote = float(
            Decimal(str(account.occupied_notional_quote)) + consumed_quote
        )
        reservation.consumed_quote = float(consumed_quote)
        reservation.released_quote = float(released_quote)
        reservation.status = outcome
        reservation.reason = reason
        self._session.flush()
        return reservation

    def release_occupied(
        self,
        *,
        account_id: str,
        tenant_id: str,
        released_quote: Decimal,
        reason: str,
    ) -> StrategySharedAccount:
        """Release confirmed sale proceeds for immediate reuse by another strategy."""

        if released_quote <= 0:
            raise CapitalAllocationError("released quote must be positive")
        account = (
            self._session.query(StrategySharedAccount)
            .filter_by(id=account_id, tenant_id=tenant_id, active=True)
            .with_for_update()
            .first()
        )
        if account is None:
            raise CapitalAllocationError("shared account is unavailable")
        occupied = Decimal(str(account.occupied_notional_quote))
        if released_quote > occupied:
            raise CapitalAllocationError("released quote exceeds occupied notional")
        account.occupied_notional_quote = float(occupied - released_quote)
        account.pending_settlement_quote = float(
            max(
                Decimal(0),
                Decimal(str(account.pending_settlement_quote)) - released_quote,
            )
        )
        reusable = (
            Decimal(str(account.reusable_quote))
            if account.reusable_quote is not None
            else Decimal(0)
        )
        account.reusable_quote = float(reusable + released_quote)
        self._session.flush()
        return account

    def bind_intent(
        self,
        reservation_id: str,
        *,
        tenant_id: str,
        intent_id: str,
    ) -> StrategyCapitalReservation:
        """Attach one staged execution intent to an existing reservation."""
        reservation = (
            self._session.query(StrategyCapitalReservation)
            .filter_by(reservation_id=reservation_id, tenant_id=tenant_id)
            .with_for_update()
            .first()
        )
        if reservation is None:
            raise CapitalAllocationError("capital reservation was not found")
        if reservation.status not in {"reserved", "partially_released"}:
            raise CapitalAllocationError("capital reservation is not bindable")
        if reservation.intent_id is not None and reservation.intent_id != intent_id:
            raise CapitalAllocationError("capital reservation is already bound")
        reservation.intent_id = intent_id
        self._session.flush()
        return reservation
