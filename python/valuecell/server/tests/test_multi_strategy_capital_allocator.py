from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from valuecell.server.db.models.base import Base
from valuecell.server.db.models.multi_strategy import (
    StrategyCapitalReservation,
    StrategySharedAccount,
)
from valuecell.server.db.models.rule_strategy import RuleStrategy  # noqa: F401
from valuecell.server.db.models.tenant import SaaSUser, Tenant  # noqa: F401
from valuecell.server.db.models.tenant_credential import TenantCredential  # noqa: F401
from valuecell.server.db.models.shared_demo_execution import SharedDemoStrategyAllocationCap
from valuecell.server.services.multi_strategy_capital_allocator import (
    CapitalAllocationError,
    SharedCapitalAllocator,
)


def _session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(Tenant(id="tenant-a", name="Tenant A"))
    account = StrategySharedAccount(
        id="account-a",
        tenant_id="tenant-a",
        credential_id="credential-a",
        available_quote=1_000,
        reserved_quote=0,
        occupied_notional_quote=0,
        pending_settlement_quote=0,
        reusable_quote=None,
        environment="okx_demo",
        sync_status="healthy",
        observed_at=datetime.now(timezone.utc),
    )
    session.add(account)
    session.commit()
    return session


def _reserve(
    allocator: SharedCapitalAllocator,
    *,
    strategy_id: str = "strategy-a",
    batch_id: str = "batch-a",
    idempotency_key: str = "reserve-a",
    requested_quote: str = "100",
    strategy_cap_quote: Decimal | None = None,
) -> StrategyCapitalReservation:
    return allocator.reserve(
        account_id="account-a",
        tenant_id="tenant-a",
        strategy_id=strategy_id,
        batch_id=batch_id,
        idempotency_key=idempotency_key,
        symbol="BTC-USDT",
        side="buy",
        requested_quote=Decimal(requested_quote),
        strategy_cap_quote=strategy_cap_quote,
    )


def test_reservation_prevents_two_strategies_using_the_same_quote() -> None:
    session = _session()
    allocator = SharedCapitalAllocator(session)
    _reserve(allocator, requested_quote="700")
    session.commit()

    with pytest.raises(CapitalAllocationError, match="insufficient unreserved"):
        _reserve(
            allocator,
            strategy_id="strategy-b",
            batch_id="batch-b",
            idempotency_key="reserve-b",
            requested_quote="400",
        )


def test_reservation_uses_persisted_strategy_cap_without_callsite_override() -> None:
    session = _session()
    session.add(
        RuleStrategy(
            strategy_id="strategy-a",
            tenant_id="tenant-a",
            name="Strategy A",
            strategy_kind="configurable_rule",
            config={"execution": {"environment": "okx_demo", "sandbox_connection_id": "credential-a"}},
        )
    )
    session.add(
        SharedDemoStrategyAllocationCap(
            account_id="account-a",
            tenant_id="tenant-a",
            credential_id="credential-a",
            environment="okx_demo",
            strategy_id="strategy-a",
            max_reserved_quote=Decimal("150"),
            max_occupied_quote=Decimal("150"),
            version=1,
            effective_at=datetime.now(timezone.utc),
        )
    )
    session.commit()
    allocator = SharedCapitalAllocator(session)
    _reserve(allocator, requested_quote="150")
    session.commit()

    with pytest.raises(CapitalAllocationError, match="strategy live capital cap exceeded"):
        _reserve(allocator, idempotency_key="reserve-b", requested_quote="1")


def test_strategy_live_cap_blocks_second_reservation() -> None:
    session = _session()
    allocator = SharedCapitalAllocator(session)
    _reserve(
        allocator,
        requested_quote="400",
        strategy_cap_quote=Decimal("500"),
    )
    with pytest.raises(CapitalAllocationError, match="strategy live capital cap exceeded"):
        _reserve(
            allocator,
            idempotency_key="reserve-b",
            requested_quote="200",
            strategy_cap_quote=Decimal("500"),
        )


def test_settling_reservation_releases_unfilled_quote() -> None:
    session = _session()
    allocator = SharedCapitalAllocator(session)
    reservation = _reserve(allocator, requested_quote="500")
    allocator.settle(
        reservation.reservation_id,
        consumed_quote=Decimal("320"),
        outcome="partially_released",
        reason="partial_fill",
    )
    session.commit()

    settled = session.query(StrategyCapitalReservation).one()
    account = session.query(StrategySharedAccount).one()
    assert settled.requested_quote == 500
    assert settled.reserved_quote == 0
    assert settled.consumed_quote == 320
    assert settled.released_quote == 180
    assert account.reserved_quote == 0
    assert account.occupied_notional_quote == 320
    assert account.pending_settlement_quote == 320


def test_confirmed_exit_releases_only_matching_occupied_capital() -> None:
    session = _session()
    allocator = SharedCapitalAllocator(session)
    first = _reserve(allocator, requested_quote="400")
    allocator.settle(first.reservation_id, consumed_quote=Decimal("400"), outcome="occupied")
    second = _reserve(
        allocator,
        strategy_id="strategy-b",
        batch_id="batch-b",
        idempotency_key="reserve-b",
        requested_quote="300",
    )
    allocator.settle(second.reservation_id, consumed_quote=Decimal("300"), outcome="occupied")

    released = allocator.release_occupied(
        account_id="account-a",
        tenant_id="tenant-a",
        reservation_id=first.reservation_id,
        released_quote=Decimal("100"),
        reason="confirmed_exit_fill",
    )
    session.flush()
    first_row = session.get(StrategyCapitalReservation, first.reservation_id)
    second_row = session.get(StrategyCapitalReservation, second.reservation_id)
    assert released.occupied_notional_quote == 600
    assert first_row is not None and first_row.consumed_quote == 300
    assert first_row.released_quote == 100
    assert second_row is not None and second_row.consumed_quote == 300

    with pytest.raises(CapitalAllocationError, match="immutable reservation identity"): 
        allocator.release_occupied(
            account_id="account-a",
            tenant_id="tenant-a",
            released_quote=Decimal("100"),
            reason="ambiguous_exit",
        )


def test_released_occupied_capital_can_be_reused_by_another_strategy() -> None:
    session = _session()
    allocator = SharedCapitalAllocator(session)
    first = _reserve(allocator, strategy_id="strategy-a", requested_quote="600")
    allocator.settle(
        first.reservation_id,
        consumed_quote=Decimal("600"),
        outcome="occupied",
        reason="filled",
    )
    allocator.release_occupied(
        account_id="account-a",
        tenant_id="tenant-a",
        reservation_id=first.reservation_id,
        released_quote=Decimal("600"),
        reason="confirmed_exit_fill",
    )

    second = _reserve(
        allocator,
        strategy_id="strategy-b",
        batch_id="batch-b",
        idempotency_key="reserve-b",
        requested_quote="600",
    )

    assert second.status == "reserved"
    account = session.query(StrategySharedAccount).one()
    assert account.reusable_quote == 1_000
    assert account.reserved_quote == 600
    assert allocator.available_capacity(account_id="account-a", tenant_id="tenant-a") == 400


def test_ambiguous_submission_keeps_live_reserve_locked() -> None:
    session = _session()
    allocator = SharedCapitalAllocator(session)
    reservation = _reserve(allocator, requested_quote="500")
    allocator.settle(
        reservation.reservation_id,
        consumed_quote=Decimal("0"),
        outcome="submission_unknown",
        reason="timeout_after_submit",
    )
    session.flush()

    locked = session.get(StrategyCapitalReservation, reservation.reservation_id)
    account = session.query(StrategySharedAccount).one()
    assert locked is not None and locked.status == "submission_unknown"
    assert locked.requested_quote == 500
    assert locked.reserved_quote == 500
    assert locked.consumed_quote == 0
    assert locked.released_quote == 0
    assert account.reserved_quote == 500
    with pytest.raises(CapitalAllocationError, match="matching occupied capital"):
        allocator.release_occupied(
            account_id="account-a",
            tenant_id="tenant-a",
            reservation_id=reservation.reservation_id,
            released_quote=Decimal("1"),
            reason="unreconciled_exit",
        )


def test_reconciled_ambiguous_submission_settles_once() -> None:
    session = _session()
    allocator = SharedCapitalAllocator(session)
    reservation = _reserve(allocator, requested_quote="500")
    allocator.settle(
        reservation.reservation_id,
        consumed_quote=Decimal("0"),
        outcome="submission_unknown",
        reason="timeout_after_submit",
    )

    settled = allocator.settle(
        reservation.reservation_id,
        consumed_quote=Decimal("320"),
        outcome="partially_released",
        reason="reconciled_terminal_partial_fill",
    )

    account = session.query(StrategySharedAccount).one()
    assert settled.status == "partially_released"
    assert settled.reserved_quote == 0
    assert settled.consumed_quote == 320
    assert settled.released_quote == 180
    assert account.reserved_quote == 0
    assert account.occupied_notional_quote == 320


def test_repeated_recovery_settlement_is_idempotent_after_restart() -> None:
    session = _session()
    allocator = SharedCapitalAllocator(session)
    reservation = _reserve(allocator, requested_quote="500")
    allocator.settle(
        reservation.reservation_id,
        consumed_quote=Decimal("0"),
        outcome="submission_unknown",
        reason="timeout_after_submit",
    )
    allocator.settle(
        reservation.reservation_id,
        consumed_quote=Decimal("320"),
        outcome="partially_released",
        reason="reconciled_terminal_partial_fill",
    )
    session.commit()

    restarted_allocator = SharedCapitalAllocator(session)
    repeated = restarted_allocator.settle(
        reservation.reservation_id,
        consumed_quote=Decimal("320"),
        outcome="partially_released",
        reason="reconciled_terminal_partial_fill",
    )

    account = session.query(StrategySharedAccount).one()
    assert repeated.status == "partially_released"
    assert repeated.consumed_quote == 320
    assert repeated.released_quote == 180
    assert account.reserved_quote == 0
    assert account.occupied_notional_quote == 320
    assert account.reusable_quote == 680


def test_stale_account_facts_fail_closed() -> None:
    session = _session()
    account = session.query(StrategySharedAccount).one()
    account.sync_status = "stale"
    session.commit()
    with pytest.raises(CapitalAllocationError, match="unavailable or stale"):
        _reserve(SharedCapitalAllocator(session), requested_quote="100")


def test_reservation_binds_only_its_own_intent() -> None:
    session = _session()
    allocator = SharedCapitalAllocator(session)
    reservation = _reserve(allocator)
    allocator.bind_intent(reservation.reservation_id, tenant_id="tenant-a", intent_id="intent-a")
    with pytest.raises(CapitalAllocationError, match="already bound"):
        allocator.bind_intent(reservation.reservation_id, tenant_id="tenant-a", intent_id="intent-b")
