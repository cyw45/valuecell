from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from valuecell.server.api.schemas.multi_strategy import StrategyAllocation
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.multi_strategy import StrategyCapitalReservation, StrategySharedAccount
from valuecell.server.db.models.rule_strategy import RuleStrategy, RuleStrategyAccount
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
    assert overview.strategy_pnl_total_quote is None


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
