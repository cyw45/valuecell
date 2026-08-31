"""Transactional shared-wallet capital reservations for concurrent strategies."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Literal
from uuid import uuid4

from sqlalchemy.orm import Session

from valuecell.server.config.settings import get_settings
from valuecell.server.db.models.multi_strategy import (
    StrategyCapitalReservation,
    StrategySharedAccount,
)

from valuecell.server.db.models.shared_demo_execution import (
    SharedDemoStrategyAllocationCap,
)


_ZERO = Decimal("0")
_RECOVERY_OUTCOMES = frozenset({"submission_unknown", "recovery_required"})
_LIVE_RESERVATION_STATUSES = frozenset(
    {"reserved", "occupied", "partially_released", "submission_unknown", "recovery_required"}
)
_FINAL_SETTLEMENT_OUTCOMES = frozenset(
    {"occupied", "released", "partially_released", "failed", "cancelled", "rejected"}
)


class CapitalAllocationError(RuntimeError):
    """Raised when one strategy cannot safely reserve shared account capital."""


def _amount(value: object, label: str, *, allow_none: bool = False) -> Decimal | None:
    """Convert a persisted amount to a finite Decimal without accepting NaN."""
    if value is None and allow_none:
        return None
    try:
        converted = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise CapitalAllocationError(f"{label} is unavailable") from exc
    if not converted.is_finite():
        raise CapitalAllocationError(f"{label} is unavailable")
    return converted


def _utc(value: datetime) -> datetime:
    """Treat SQLite's naive timestamps as UTC, matching the persisted contract."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


class SharedCapitalAllocator:
    """Reserve, occupy, and release funds without assigning wallet assets to a strategy."""

    def __init__(self, session: Session) -> None:
        self._session = session

    @staticmethod
    def _same_request(
        reservation: StrategyCapitalReservation,
        *,
        account_id: str,
        tenant_id: str,
        strategy_id: str,
        batch_id: str | None,
        symbol: str,
        side: str,
        requested_quote: Decimal,
    ) -> bool:
        """Idempotency keys cannot be reused for a different immutable request."""
        persisted = _amount(reservation.requested_quote, "reservation requested quote")
        return (
            reservation.account_id == account_id
            and reservation.tenant_id == tenant_id
            and reservation.strategy_id == strategy_id
            and reservation.batch_id == batch_id
            and reservation.symbol == symbol
            and reservation.side == side
            and persisted == requested_quote
        )

    def _existing_reservation(
        self,
        *,
        tenant_id: str,
        idempotency_key: str,
    ) -> StrategyCapitalReservation | None:
        return (
            self._session.query(StrategyCapitalReservation)
            .filter_by(tenant_id=tenant_id, idempotency_key=idempotency_key)
            .with_for_update()
            .first()
        )

    def _account(
        self,
        *,
        account_id: str,
        tenant_id: str,
        lock: bool = True,
    ) -> StrategySharedAccount:
        query = self._session.query(StrategySharedAccount).filter_by(
            id=account_id,
            tenant_id=tenant_id,
            active=True,
        )
        if lock:
            query = query.with_for_update()
        account = query.first()
        if account is None or account.environment != "okx_demo":
            raise CapitalAllocationError("shared account is unavailable")
        return account

    def _require_fresh_account(self, account: StrategySharedAccount) -> None:
        """Opening a position is unsafe unless the persisted wallet fact is fresh."""
        if account.sync_status != "healthy" or account.observed_at is None:
            raise CapitalAllocationError("shared account facts are unavailable or stale")
        observed_at = _utc(account.observed_at)
        age = (datetime.now(timezone.utc) - observed_at).total_seconds()
        max_age = get_settings().DEMO_ACCOUNT_SYNC_INTERVAL_S * 2
        if age < -60 or age > max_age:
            raise CapitalAllocationError("shared account facts are unavailable or stale")

    @staticmethod
    def _account_capacity(account: StrategySharedAccount) -> Decimal:
        """Return free capacity without adding occupied and reusable balances.

        ``reusable_quote`` is the allocator's free-cash projection before the
        current live reservation hold. Settling an order reduces it by the
        occupied amount; releasing an occupied exit increases it. The account
        hold is then subtracted exactly once, while ``available_quote`` remains
        the exchange snapshot fallback until the allocator establishes its
        projection.
        """
        available = _amount(account.available_quote, "shared account quote availability")
        occupied = _amount(account.occupied_notional_quote, "occupied notional")
        reserved = _amount(account.reserved_quote, "reserved quote")
        reusable = _amount(account.reusable_quote, "reusable quote", allow_none=True)
        if available is None or available < _ZERO:
            raise CapitalAllocationError("shared account quote availability is unknown")
        if occupied is None or reserved is None or occupied < _ZERO or reserved < _ZERO:
            raise CapitalAllocationError("shared account capital facts are invalid")
        if reusable is not None:
            if reusable < _ZERO:
                raise CapitalAllocationError("shared account capital facts are invalid")
            return max(_ZERO, reusable - reserved)
        return max(_ZERO, available - reserved)
    def available_capacity(self, *, account_id: str, tenant_id: str) -> Decimal:
        """Return fresh account capacity available to all strategies."""
        account = self._account(account_id=account_id, tenant_id=tenant_id)
        self._require_fresh_account(account)
        return self._account_capacity(account)

    def _strategy_cap(
        self,
        account: StrategySharedAccount,
        *,
        tenant_id: str,
        strategy_id: str,
        strategy_cap_quote: Decimal | None,
    ) -> tuple[Decimal | None, Decimal | None]:
        """Return explicit/ persisted total-live and occupied cap limits."""
        explicit = None
        if strategy_cap_quote is not None:
            explicit = _amount(strategy_cap_quote, "strategy cap")
            if explicit is None or explicit < _ZERO:
                raise CapitalAllocationError("strategy cap cannot be negative")
        model = SharedDemoStrategyAllocationCap
        if model is None:
            return explicit, None
        try:
            rows = (
                self._session.query(model)
                .filter_by(
                    account_id=account.id,
                    tenant_id=tenant_id,
                    credential_id=account.credential_id,
                    environment="okx_demo",
                    strategy_id=strategy_id,
                    active=True,
                )
                .order_by(model.version.desc())
                .all()
            )
        except Exception as exc:
            raise CapitalAllocationError("strategy allocation cap is unavailable") from exc
        now = datetime.now(timezone.utc)
        for row in rows:
            effective_at = row.effective_at
            expires_at = row.expires_at
            if effective_at is not None and _utc(effective_at) > now:
                continue
            if expires_at is not None and _utc(expires_at) <= now:
                continue
            persisted_total = _amount(row.max_reserved_quote, "strategy reserve cap", allow_none=True)
            persisted_occupied = _amount(row.max_occupied_quote, "strategy occupied cap", allow_none=True)
            if persisted_total is not None and persisted_total < _ZERO:
                raise CapitalAllocationError("strategy reserve cap is invalid")
            if persisted_occupied is not None and persisted_occupied < _ZERO:
                raise CapitalAllocationError("strategy occupied cap is invalid")
            if explicit is None:
                explicit = persisted_total
            elif persisted_total is not None:
                explicit = min(explicit, persisted_total)
            return explicit, persisted_occupied
        return explicit, None

    def _strategy_live(self, *, account_id: str, tenant_id: str, strategy_id: str) -> tuple[Decimal, Decimal]:
        rows = (
            self._session.query(StrategyCapitalReservation)
            .filter(
                StrategyCapitalReservation.account_id == account_id,
                StrategyCapitalReservation.tenant_id == tenant_id,
                StrategyCapitalReservation.strategy_id == strategy_id,
                StrategyCapitalReservation.status.in_(_LIVE_RESERVATION_STATUSES),
            )
            .all()
        )
        live = _ZERO
        occupied = _ZERO
        for row in rows:
            row_reserved = _amount(row.reserved_quote, "reservation reserved quote")
            row_consumed = _amount(row.consumed_quote, "reservation occupied quote")
            if row_reserved is None or row_consumed is None or row_reserved < _ZERO or row_consumed < _ZERO:
                raise CapitalAllocationError("strategy reservation facts are invalid")
            live += row_reserved + row_consumed
            occupied += row_consumed
        return live, occupied

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
        strategy_cap_quote: Decimal | None = None,
    ) -> StrategyCapitalReservation:
        """Atomically reserve quote funds before one strategy submits an order."""
        requested = _amount(requested_quote, "requested quote")
        if requested is None or requested <= _ZERO:
            raise CapitalAllocationError("requested quote must be positive")
        if not idempotency_key:
            raise CapitalAllocationError("idempotency key is required")
        existing = self._existing_reservation(
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
        )
        if existing is not None:
            if not self._same_request(
                existing,
                account_id=account_id,
                tenant_id=tenant_id,
                strategy_id=strategy_id,
                batch_id=batch_id,
                symbol=symbol,
                side=side,
                requested_quote=requested,
            ):
                raise CapitalAllocationError("idempotency key is already used for another request")
            return existing

        # The account row is the serialization boundary for competing strategies.
        account = self._account(account_id=account_id, tenant_id=tenant_id)
        self._require_fresh_account(account)
        # Recheck after locking the account to close the same-account race window.
        existing = self._existing_reservation(
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
        )
        if existing is not None:
            if not self._same_request(
                existing,
                account_id=account_id,
                tenant_id=tenant_id,
                strategy_id=strategy_id,
                batch_id=batch_id,
                symbol=symbol,
                side=side,
                requested_quote=requested,
            ):
                raise CapitalAllocationError("idempotency key is already used for another request")
            return existing

        capacity = self._account_capacity(account)
        if capacity < requested:
            raise CapitalAllocationError("shared account has insufficient unreserved quote")
        live, occupied = self._strategy_live(
            account_id=account_id,
            tenant_id=tenant_id,
            strategy_id=strategy_id,
        )
        total_cap, occupied_cap = self._strategy_cap(
            account,
            tenant_id=tenant_id,
            strategy_id=strategy_id,
            strategy_cap_quote=strategy_cap_quote,
        )
        if total_cap is not None and live + requested > total_cap:
            raise CapitalAllocationError("strategy live capital cap exceeded")
        if occupied_cap is not None and occupied + requested > occupied_cap:
            raise CapitalAllocationError("strategy occupied capital cap exceeded")

        current_reserved = _amount(account.reserved_quote, "reserved quote")
        if current_reserved is None:
            raise CapitalAllocationError("reserved quote is unavailable")
        current_reusable = _amount(account.reusable_quote, "reusable quote", allow_none=True)
        if current_reusable is None:
            current_reusable = _amount(account.available_quote, "shared account quote availability")
        if current_reusable is None:
            raise CapitalAllocationError("shared account quote availability is unknown")
        account.reusable_quote = float(current_reusable)
        account.reserved_quote = float(current_reserved + requested)
        account.version = int(account.version) + 1
        reservation = StrategyCapitalReservation(
            reservation_id=str(uuid4()),
            account_id=account_id,
            tenant_id=tenant_id,
            strategy_id=strategy_id,
            batch_id=batch_id,
            idempotency_key=idempotency_key,
            symbol=symbol,
            side=side,
            requested_quote=float(requested),
            reserved_quote=float(requested),
            consumed_quote=0.0,
            released_quote=0.0,
            status="reserved",
        )
        self._session.add(reservation)
        self._session.flush()
        return reservation

    def settle(
        self,
        reservation_id: str,
        *,
        consumed_quote: Decimal,
        outcome: Literal[
            "occupied",
            "released",
            "partially_released",
            "failed",
            "cancelled",
            "rejected",
            "submission_unknown",
            "recovery_required",
        ],
        reason: str | None = None,
    ) -> StrategyCapitalReservation:
        """Settle one confirmed result, or lock it for venue reconciliation."""
        consumed = _amount(consumed_quote, "consumed quote")
        if consumed is None or consumed < _ZERO:
            raise CapitalAllocationError("consumed quote cannot be negative")
        if outcome not in _RECOVERY_OUTCOMES and outcome not in _FINAL_SETTLEMENT_OUTCOMES:
            raise CapitalAllocationError("unsupported reservation outcome")
        reservation = (
            self._session.query(StrategyCapitalReservation)
            .filter_by(reservation_id=reservation_id)
            .first()
        )
        if reservation is None:
            raise CapitalAllocationError("capital reservation was not found")
        account = self._account(
            account_id=reservation.account_id,
            tenant_id=reservation.tenant_id,
        )
        reservation = (
            self._session.query(StrategyCapitalReservation)
            .filter_by(reservation_id=reservation_id)
            .with_for_update()
            .first()
        )
        if reservation is None:
            raise CapitalAllocationError("capital reservation was not found")
        if reservation.status not in {"reserved", *_RECOVERY_OUTCOMES}:
            raise CapitalAllocationError(
                "capital reservation is already terminal"
            )
        live_reserved = _amount(reservation.reserved_quote, "reservation reserved quote")
        if live_reserved is None or live_reserved < _ZERO:
            raise CapitalAllocationError("reservation reserved quote is invalid")
        if outcome in _RECOVERY_OUTCOMES:
            if consumed != _ZERO:
                raise CapitalAllocationError("recovery outcome cannot consume quote")
            reservation.status = outcome
            reservation.reason = reason or "venue submission outcome requires reconciliation"
            account.version = int(account.version) + 1
            self._session.flush()
            return reservation
        if consumed > live_reserved:
            raise CapitalAllocationError("consumed quote exceeds live reservation")
        if outcome == "occupied" and consumed != live_reserved:
            raise CapitalAllocationError("occupied outcome must consume the full live reservation")
        if outcome == "released" and consumed != _ZERO:
            raise CapitalAllocationError("released outcome cannot consume quote")
        if outcome in {"failed", "cancelled", "rejected"} and consumed != _ZERO:
            raise CapitalAllocationError("failed outcome cannot consume quote")
        if outcome == "partially_released" and not (_ZERO < consumed < live_reserved):
            raise CapitalAllocationError("partial outcome must consume part of the reservation")

        released = live_reserved - consumed
        account_reserved = _amount(account.reserved_quote, "reserved quote")
        account_occupied = _amount(account.occupied_notional_quote, "occupied notional")
        account_pending = _amount(account.pending_settlement_quote, "pending settlement")
        reusable = _amount(account.reusable_quote, "reusable quote", allow_none=True)
        if account_reserved is None or account_occupied is None or account_pending is None:
            raise CapitalAllocationError("shared account capital facts are unavailable")
        if account_reserved < live_reserved:
            raise CapitalAllocationError("shared account reservation accounting is inconsistent")
        if reusable is None:
            reusable = _amount(account.available_quote, "shared account quote availability") or _ZERO
        if reusable < consumed:
            raise CapitalAllocationError("shared account reusable capital is inconsistent")
        account.reserved_quote = float(account_reserved - live_reserved)
        account.occupied_notional_quote = float(account_occupied + consumed)
        account.pending_settlement_quote = float(account_pending + consumed)
        account.reusable_quote = float(reusable - consumed)
        reservation.reserved_quote = 0.0
        prior_consumed = _amount(reservation.consumed_quote, "reservation occupied quote") or _ZERO
        prior_released = _amount(reservation.released_quote, "reservation released quote") or _ZERO
        reservation.consumed_quote = float(prior_consumed + consumed)
        reservation.released_quote = float(prior_released + released)
        reservation.status = outcome
        reservation.reason = reason
        account.version = int(account.version) + 1
        self._session.flush()
        return reservation

    def mark_recovery_required(
        self,
        reservation_id: str,
        *,
        tenant_id: str,
        reason: str | None = None,
        status: Literal["submission_unknown", "recovery_required"] = "recovery_required",
    ) -> StrategyCapitalReservation:
        """Lock an ambiguous submission without releasing its outstanding reserve."""
        reservation = (
            self._session.query(StrategyCapitalReservation)
            .filter_by(reservation_id=reservation_id, tenant_id=tenant_id)
            .first()
        )
        if reservation is None:
            raise CapitalAllocationError("capital reservation was not found")
        account = self._account(account_id=reservation.account_id, tenant_id=tenant_id)
        reservation = (
            self._session.query(StrategyCapitalReservation)
            .filter_by(reservation_id=reservation_id, tenant_id=tenant_id)
            .with_for_update()
            .first()
        )
        if reservation is None:
            raise CapitalAllocationError("capital reservation was not found")
        if reservation.status == status:
            return reservation
        if reservation.status not in {"reserved", *_RECOVERY_OUTCOMES}:
            raise CapitalAllocationError("capital reservation is already terminal")
        reservation.status = status
        reservation.reason = reason or "venue submission outcome requires reconciliation"
        account.version = int(account.version) + 1
        self._session.flush()
        return reservation

    def release_occupied(
        self,
        *,
        account_id: str,
        tenant_id: str,
        released_quote: Decimal,
        reason: str,
        reservation_id: str | None = None,
        strategy_id: str | None = None,
        batch_id: str | None = None,
        symbol: str | None = None,
    ) -> StrategySharedAccount:
        """Release only occupied capital proven to belong to the matching reservation."""
        released = _amount(released_quote, "released quote")
        if released is None or released <= _ZERO:
            raise CapitalAllocationError("released quote must be positive")
        if reservation_id is None and not all((strategy_id, batch_id, symbol)):
            raise CapitalAllocationError("occupied capital requires immutable reservation identity")
        account = self._account(account_id=account_id, tenant_id=tenant_id)
        if reservation_id is not None:
            reservation = (
                self._session.query(StrategyCapitalReservation)
                .filter_by(
                    reservation_id=reservation_id,
                    account_id=account_id,
                    tenant_id=tenant_id,
                )
                .with_for_update()
                .first()
            )
            candidates = [reservation] if reservation is not None else []
        else:
            query = (
                self._session.query(StrategyCapitalReservation)
                .filter(
                    StrategyCapitalReservation.account_id == account_id,
                    StrategyCapitalReservation.tenant_id == tenant_id,
                    StrategyCapitalReservation.status.in_({"occupied", "partially_released"}),
                    StrategyCapitalReservation.consumed_quote > 0,
                    StrategyCapitalReservation.strategy_id == strategy_id,
                    StrategyCapitalReservation.batch_id == batch_id,
                    StrategyCapitalReservation.symbol == symbol,
                )
                .with_for_update()
            )
            candidates = query.all()
        if len(candidates) != 1 or candidates[0] is None:
            raise CapitalAllocationError("occupied capital requires one matching reservation")
        reservation = candidates[0]
        if strategy_id is not None and reservation.strategy_id != strategy_id:
            raise CapitalAllocationError("occupied capital does not match strategy")
        if batch_id is not None and reservation.batch_id != batch_id:
            raise CapitalAllocationError("occupied capital does not match batch")
        if symbol is not None and reservation.symbol != symbol:
            raise CapitalAllocationError("occupied capital does not match symbol")
        occupied_for_reservation = _amount(reservation.consumed_quote, "reservation occupied quote")
        if occupied_for_reservation is None or released > occupied_for_reservation:
            raise CapitalAllocationError("released quote exceeds matching occupied capital")
        occupied = _amount(account.occupied_notional_quote, "occupied notional")
        pending = _amount(account.pending_settlement_quote, "pending settlement")
        reusable = _amount(account.reusable_quote, "reusable quote", allow_none=True) or _ZERO
        if occupied is None or pending is None or occupied < released:
            raise CapitalAllocationError("shared account occupied capital is inconsistent")
        reservation.consumed_quote = float(occupied_for_reservation - released)
        prior_released = _amount(reservation.released_quote, "reservation released quote") or _ZERO
        reservation.released_quote = float(prior_released + released)
        reservation.status = "released" if occupied_for_reservation == released else "partially_released"
        reservation.reason = reason
        account.occupied_notional_quote = float(occupied - released)
        account.pending_settlement_quote = float(max(_ZERO, pending - released))
        account.reusable_quote = float(reusable + released)
        account.version = int(account.version) + 1
        self._session.flush()
        return account

    def bind_intent(
        self,
        reservation_id: str,
        *,
        tenant_id: str,
        intent_id: str,
    ) -> StrategyCapitalReservation:
        """Attach one staged execution intent to an outstanding reservation."""
        if not intent_id:
            raise CapitalAllocationError("intent id is required")
        reservation = (
            self._session.query(StrategyCapitalReservation)
            .filter_by(reservation_id=reservation_id, tenant_id=tenant_id)
            .with_for_update()
            .first()
        )
        if reservation is None:
            raise CapitalAllocationError("capital reservation was not found")
        if reservation.status != "reserved":
            raise CapitalAllocationError("capital reservation is not bindable")
        if reservation.intent_id is not None and reservation.intent_id != intent_id:
            raise CapitalAllocationError("capital reservation is already bound")
        reservation.intent_id = intent_id
        self._session.flush()
        return reservation
