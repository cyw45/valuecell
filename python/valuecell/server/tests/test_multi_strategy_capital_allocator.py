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
        environment="okx_demo",
    )
    session.add(account)
    session.commit()
    return session


def test_reservation_prevents_two_strategies_using_the_same_quote() -> None:
    session = _session()
    allocator = SharedCapitalAllocator(session)
    first = allocator.reserve(
        account_id="account-a",
        tenant_id="tenant-a",
        strategy_id="strategy-a",
        batch_id="batch-a",
        idempotency_key="reserve-a",
        symbol="BTC-USDT",
        side="buy",
        requested_quote=Decimal("700"),
    )
    session.commit()

    with pytest.raises(CapitalAllocationError, match="insufficient unreserved"):
        allocator.reserve(
            account_id="account-a",
            tenant_id="tenant-a",
            strategy_id="strategy-b",
            batch_id="batch-b",
            idempotency_key="reserve-b",
            symbol="ETH-USDT",
            side="buy",
            requested_quote=Decimal("400"),
        )
    assert first.status == "reserved"
    assert session.query(StrategySharedAccount).one().reserved_quote == 700


def test_settling_reservation_releases_unfilled_quote() -> None:
    session = _session()
    allocator = SharedCapitalAllocator(session)
    reservation = allocator.reserve(
        account_id="account-a",
        tenant_id="tenant-a",
        strategy_id="strategy-a",
        batch_id="batch-a",
        idempotency_key="reserve-a",
        symbol="BTC-USDT",
        side="buy",
        requested_quote=Decimal("500"),
    )
    allocator.settle(
        reservation.reservation_id,
        consumed_quote=Decimal("320"),
        outcome="partially_released",
        reason="partial_fill",
    )
    session.commit()

    settled = session.query(StrategyCapitalReservation).one()
    account = session.query(StrategySharedAccount).one()
    assert settled.consumed_quote == 320
    assert settled.released_quote == 180
    assert account.reserved_quote == 0
    assert account.occupied_notional_quote == 320

def test_confirmed_sale_releases_occupied_capital_for_reuse() -> None:
    session = _session()
    allocator = SharedCapitalAllocator(session)
    reservation = allocator.reserve(
        account_id="account-a",
        tenant_id="tenant-a",
        strategy_id="strategy-a",
        batch_id="batch-a",
        idempotency_key="reserve-a",
        symbol="BTC-USDT",
        side="buy",
        requested_quote=Decimal("400"),
    )
    allocator.settle(
        reservation.reservation_id,
        consumed_quote=Decimal("400"),
        outcome="occupied",
    )
    released = allocator.release_occupied(
        account_id="account-a",
        tenant_id="tenant-a",
        released_quote=Decimal("400"),
        reason="confirmed_exit_fill",
    )
    assert released.occupied_notional_quote == 0
    assert released.reusable_quote == 400

def test_reservation_binds_only_its_own_intent() -> None:
    session = _session()
    allocator = SharedCapitalAllocator(session)
    reservation = allocator.reserve(
        account_id="account-a",
        tenant_id="tenant-a",
        strategy_id="strategy-a",
        batch_id="batch-a",
        idempotency_key="reserve-a",
        symbol="BTC-USDT",
        side="buy",
        requested_quote=Decimal("100"),
    )
    allocator.bind_intent(reservation.reservation_id, tenant_id="tenant-a", intent_id="intent-a")
    with pytest.raises(CapitalAllocationError, match="already bound"):
        allocator.bind_intent(reservation.reservation_id, tenant_id="tenant-a", intent_id="intent-b")
