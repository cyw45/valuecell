from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from valuecell.server.api.schemas.multi_strategy import StrategyAllocation
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.multi_strategy import StrategyCapitalReservation, StrategySharedAccount
from valuecell.server.db.models.rule_strategy import RuleStrategy, RuleStrategyAccount
from valuecell.server.db.models.shared_demo_execution import (
    SharedDemoAccountSnapshot,
    SharedDemoAccountSyncState,
    SharedDemoFill,
    SharedDemoOrderProjection,
    SharedDemoStrategyAllocationCap,
    SharedDemoVenueOrder,
)
from valuecell.server.db.models.tenant import Tenant
from valuecell.server.services.multi_strategy_account_summary import (
    SharedAccountSummaryUnavailable,
    build_shared_account_overview,
)


def _session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(Tenant(id="tenant-a", name="Tenant A"))
    session.add(
        StrategySharedAccount(
            id="account-a",
            tenant_id="tenant-a",
            credential_id="credential-a",
            environment="okx_demo",
            wallet_equity_quote=1_000,
            available_quote=600,
            reserved_quote=400,
            occupied_notional_quote=300,
            pending_settlement_quote=0,
            reusable_quote=300,
            utilization_denominator_quote=1_000,
            sync_status="healthy",
            attribution_status="complete",
            observed_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
        )
    )
    session.add(
        RuleStrategy(
            strategy_id="strategy-a",
            tenant_id="tenant-a",
            name="Strategy A",
            strategy_kind="dual_ma_trend",
            strategy_version="v1",
            code_fingerprint="fingerprint-a",
            config={
                "initial_capital_quote": 600,
                "execution": {
                    "environment": "okx_demo",
                    "sandbox_connection_id": "credential-a",
                },
            }
        )
    )
    session.add(
        RuleStrategyAccount(
            tenant_id="tenant-a",
            strategy_id="strategy-a",
            allocation_quote=600,
            quote_balance=400,
            equity_quote=650,
            realized_pnl_quote=30,
            unrealized_pnl_quote=20,
        )
    )
    session.add(
        StrategyCapitalReservation(
            reservation_id="reservation-a",
            account_id="account-a",
            tenant_id="tenant-a",
            strategy_id="strategy-a",
            batch_id="batch-a",
            idempotency_key="key-a",
            symbol="BTC-USDT",
            side="buy",
            requested_quote=400,
            reserved_quote=400,
            consumed_quote=300,
            released_quote=100,
            status="partially_released",
        )
    )
    session.commit()
    return session


def test_summary_separates_wallet_and_strategy_allocation() -> None:
    session = _session()
    overview = build_shared_account_overview(
        session,
        tenant_id="tenant-a",
        credential_id="credential-a",
    )
    assert overview.wallet.total_equity_quote == 1_000
    assert overview.allocator.reserved_quote == 400
    assert overview.allocator.occupied_notional_quote == 300
    assert overview.allocator.allocations[0].net_pnl_quote is None
    assert overview.allocator.allocations[0].lifecycle_reason is not None
    assert overview.strategy_pnl_total_quote is None
    assert overview.allocator.available_for_strategies_quote == 0
    assert overview.execution_gate.status == "blocked"
    assert overview.execution_gate.can_open_positions is False
    assert "当前没有可分配的开仓资金" in overview.execution_gate.reasons


def test_summary_protects_new_entries_when_sync_or_reconciliation_is_not_healthy() -> None:
    session = _session()
    account = session.query(StrategySharedAccount).one()
    account.sync_status = "stale"
    session.add(
        SharedDemoAccountSyncState(
            account_id=account.id,
            tenant_id=account.tenant_id,
            credential_id=account.credential_id,
            environment="okx_demo",
            sync_status="stale",
            reconciliation_status="blocked",
            unresolved_submission_count=2,
        )
    )
    session.commit()

    overview = build_shared_account_overview(
        session, tenant_id="tenant-a", credential_id="credential-a"
    )

    assert overview.execution_gate.status == "blocked"
    assert overview.execution_gate.unresolved_submission_count == 2
    assert overview.execution_gate.can_open_positions is False
    assert "共享钱包同步状态不是 healthy" in overview.execution_gate.reasons
    assert "存在 2 个待远端对账订单" in overview.execution_gate.reasons


def test_summary_exposes_strategy_cap_and_actual_usage() -> None:
    """The matrix must distinguish configured caps from live reservations."""
    session = _session()
    session.add(
        SharedDemoStrategyAllocationCap(
            account_id="account-a",
            tenant_id="tenant-a",
            credential_id="credential-a",
            environment="okx_demo",
            strategy_id="strategy-a",
            max_reserved_quote=Decimal("250"),
            max_occupied_quote=Decimal("200"),
            active=1,
            version=1,
            effective_at=datetime.now(timezone.utc),
        )
    )
    session.commit()

    overview = build_shared_account_overview(
        session,
        tenant_id="tenant-a",
        credential_id="credential-a",
    )

    allocation = overview.allocator.allocations[0]
    assert allocation.max_reserved_quote == 250
    assert allocation.max_occupied_quote == 200


def test_summary_returns_strategy_runtime_and_aggregate_pnl() -> None:
    session = _session()
    strategy = session.query(RuleStrategy).first()
    strategy.status = "running"
    strategy.current_batch_id = "batch-a"
    session.commit()
    overview = build_shared_account_overview(
        session,
        tenant_id="tenant-a",
        credential_id="credential-a",
    )

    allocation = overview.allocator.allocations[0]
    assert allocation.status == "running"
    assert allocation.current_batch_id == "batch-a"
    assert allocation.utilization_ratio == 0.7
    assert overview.strategy_pnl_total_quote is None


def test_summary_keeps_unknown_submission_capital_in_live_reservation() -> None:
    session = _session()
    reservation = session.query(StrategyCapitalReservation).first()
    reservation.status = "submission_unknown"
    reservation.reason = "venue response timed out"
    session.commit()

    overview = build_shared_account_overview(
        session,
        tenant_id="tenant-a",
        credential_id="credential-a",
    )

    allocation = overview.allocator.allocations[0]
    assert allocation.reserved_quote == 400
    assert allocation.allocation_state == "submission_unknown"


def test_summary_accepts_settled_occupied_capital_with_no_outstanding_reserve() -> None:
    session = _session()
    reservation = session.query(StrategyCapitalReservation).first()
    reservation.status = "occupied"
    reservation.reserved_quote = 0
    reservation.consumed_quote = 300
    reservation.released_quote = 100
    session.commit()

    overview = build_shared_account_overview(
        session,
        tenant_id="tenant-a",
        credential_id="credential-a",
    )

    allocation = overview.allocator.allocations[0]
    assert allocation.reserved_quote == 0
    assert allocation.occupied_quote == 300
    assert allocation.allocation_state == "occupied"


def test_summary_derives_strategy_pnl_from_attributed_demo_fills() -> None:
    session = _session()
    session.add(
        SharedDemoAccountSnapshot(
            snapshot_id="snapshot-a",
            account_id="account-a",
            tenant_id="tenant-a",
            credential_id="credential-a",
            environment="okx_demo",
            observed_at=datetime(2026, 8, 28, 12, tzinfo=timezone.utc),
            wallet_equity_quote=1_020,
            available_quote=620,
            balances=[],
            positions=[],
            open_orders=[],
        )
    )
    session.add(
        SharedDemoVenueOrder(
            order_id="order-buy",
            intent_id="intent-buy",
            reservation_id="reservation-a",
            account_id="account-a",
            tenant_id="tenant-a",
            credential_id="credential-a",
            environment="okx_demo",
            strategy_id="strategy-a",
            batch_id="batch-a",
            client_order_id="client-buy",
            symbol="BTC-USDT",
            side="buy",
            order_type="market",
            leg_kind="entry",
            requested_quantity=1,
            requested_quote=100,
        )
    )
    session.add(
        SharedDemoVenueOrder(
            order_id="order-sell",
            intent_id="intent-sell",
            reservation_id="reservation-a",
            account_id="account-a",
            tenant_id="tenant-a",
            credential_id="credential-a",
            environment="okx_demo",
            strategy_id="strategy-a",
            batch_id="batch-a",
            client_order_id="client-sell",
            symbol="BTC-USDT",
            side="sell",
            order_type="market",
            leg_kind="exit",
            requested_quantity=1,
            requested_quote=120,
        )
    )
    session.add_all(
        [
            SharedDemoFill(
                fill_id="fill-buy",
                order_id="order-buy",
                venue="okx",
                venue_fill_id="venue-fill-buy",
                account_id="account-a",
                tenant_id="tenant-a",
                credential_id="credential-a",
                environment="okx_demo",
                strategy_id="strategy-a",
                batch_id="batch-a",
                price=100,
                quantity=1,
                quote_amount=100,
                occurred_at=datetime(2026, 8, 28, 10, tzinfo=timezone.utc),
                reconciliation_source="test",
            ),
            SharedDemoFill(
                fill_id="fill-sell",
                order_id="order-sell",
                venue="okx",
                venue_fill_id="venue-fill-sell",
                account_id="account-a",
                tenant_id="tenant-a",
                credential_id="credential-a",
                environment="okx_demo",
                strategy_id="strategy-a",
                batch_id="batch-a",
                price=120,
                quantity=1,
                quote_amount=120,
                fee_quote=2,
                occurred_at=datetime(2026, 8, 28, 11, tzinfo=timezone.utc),
                reconciliation_source="test",
            ),
        ]
    )
    session.add_all(
        [
            SharedDemoOrderProjection(
                order_id="order-buy",
                account_id="account-a",
                tenant_id="tenant-a",
                credential_id="credential-a",
                environment="okx_demo",
                strategy_id="strategy-a",
                batch_id="batch-a",
                status="filled",
                filled_quantity=1,
                filled_quote=100,
            ),
            SharedDemoOrderProjection(
                order_id="order-sell",
                account_id="account-a",
                tenant_id="tenant-a",
                credential_id="credential-a",
                environment="okx_demo",
                strategy_id="strategy-a",
                batch_id="batch-a",
                status="filled",
                filled_quantity=1,
                filled_quote=120,
                fee_quote=2,
            ),
        ]
    )
    session.commit()

    allocation = build_shared_account_overview(
        session,
        tenant_id="tenant-a",
        credential_id="credential-a",
    ).allocator.allocations[0]

    assert allocation.realized_pnl_quote == pytest.approx(20)
    assert allocation.unrealized_pnl_quote == pytest.approx(0)
    assert allocation.net_pnl_quote == pytest.approx(18)
    assert allocation.return_rate_pct == pytest.approx(18 / 600)


def test_summary_requires_authoritative_allocator_equity() -> None:
    session = _session()
    account = session.query(StrategySharedAccount).one()
    account.utilization_denominator_quote = None
    session.commit()
    with pytest.raises(SharedAccountSummaryUnavailable, match="equity is unavailable"):
        build_shared_account_overview(
            session,
            tenant_id="tenant-a",
            credential_id="credential-a",
        )


def test_allocation_contract_rejects_occupied_amount_above_reservation() -> None:
    with pytest.raises(ValueError):
        StrategyAllocation(
            strategy_id="strategy-a",
            kind="dual_ma_trend",
            reserved_quote=1,
            occupied_quote=2,
            released_quote=0,
            allocation_state="occupied",
            utilization_denominator_quote=100,
        )
