import asyncio
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from valuecell.server.db.models.base import Base

import valuecell.server.services.rule_strategy_demo_account_sync_service as sync_module
from valuecell.server.db.models.multi_strategy import StrategySharedAccount
from valuecell.server.db.models.rule_strategy import (
    RuleStrategy,
    RuleStrategyDemoAccountSnapshot,
    RuleStrategyDemoAccountSyncState,
    RuleStrategyExecutionIntent,
)
from valuecell.server.db.models.shared_demo_execution import (
    SharedDemoAccountSnapshot,
    SharedDemoAccountSyncState,
    SharedDemoExecutionIntent,
    SharedDemoExecutionReservation,
    SharedDemoStrategyAllocationCap,
)


class FakeExchange:
    balances = 0
    positions_calls = 0
    refreshes = 0

    def __init__(self, _session):
        pass

    async def balance(self, _tenant_id, _credential_id):
        type(self).balances += 1
        return {
            "source": "okx_demo",
            "total_usdt_value": 1_000.0,
            "balances": [],
            "checked_at": "2026-08-19T00:00:00+00:00",
        }

    async def positions(self, _tenant_id, _credential_id, *, account):
        type(self).positions_calls += 1
        return {
            "source": "okx_demo",
            "positions": [{"symbol": "BTC/USDT", "quantity": 1.0}],
            "checked_at": account["checked_at"],
        }

    async def refresh_open_orders(self, _tenant_id, _credential_id):
        type(self).refreshes += 1


def test_sync_fetches_shared_credential_once_and_deduplicates_snapshot(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        session.add_all(
            [
                RuleStrategy(
                    strategy_id="strategy-a",
                    tenant_id="tenant-a",
                    name="A",
                    status="running",
                    config={
                        "execution": {
                            "environment": "okx_demo",
                            "sandbox_connection_id": "credential-a",
                        },
                        "initial_capital_quote": 300,
                    },
                ),
                RuleStrategy(
                    strategy_id="strategy-b",
                    tenant_id="tenant-a",
                    name="B",
                    status="stopped",
                    config={
                        "execution": {
                            "environment": "okx_demo",
                            "sandbox_connection_id": "credential-a",
                        },
                        "initial_capital_quote": 200,
                    },
                ),
            ]
        )
        session.commit()
        monkeypatch.setattr(sync_module, "SandboxExchangeTradingService", FakeExchange)
        monkeypatch.setattr(
            sync_module,
            "get_settings",
            lambda: type(
                "Settings",
                (),
                {
                    "DEMO_ACCOUNT_SYNC_ATTEMPTS": 1,
                    "DEMO_ACCOUNT_SYNC_RETRY_DELAY_S": 0.0,
                    "DEMO_ACCOUNT_READ_TIMEOUT_S": 1.0,
                    "DEMO_ACCOUNT_SYNC_INTERVAL_S": 300,
                },
            )(),
        )

        first = asyncio.run(sync_module.sync_demo_account_snapshots(session))
        second = asyncio.run(sync_module.sync_demo_account_snapshots(session))

        assert first == {"accounts": 1, "synced": 2, "failed": 0}
        assert second == {"accounts": 1, "synced": 0, "failed": 0}
        assert FakeExchange.balances == 2
        assert FakeExchange.positions_calls == 2
        assert FakeExchange.refreshes == 2
        assert session.query(RuleStrategyDemoAccountSnapshot).count() == 2
        assert session.query(SharedDemoAccountSnapshot).count() == 1
        shared_state = session.query(SharedDemoAccountSyncState).one()
        assert shared_state.account_id == session.query(StrategySharedAccount).one().id
        assert shared_state.sync_status == "healthy"
        # A snapshot cycle with no unsettled intent or venue order is fully
        # attributed; the gate can open without a separate writer.
        assert shared_state.reconciliation_status == "complete"
        assert shared_state.unresolved_submission_count == 0
        assert shared_state.last_reconciled_at is not None
        assert session.query(StrategySharedAccount).one().attribution_status == "complete"
        assert session.query(StrategySharedAccount).count() == 1
        caps = session.query(SharedDemoStrategyAllocationCap).all()
        assert {cap.strategy_id for cap in caps} == {"strategy-a", "strategy-b"}
        assert {float(cap.max_reserved_quote) for cap in caps} == {200.0, 300.0}
        states = session.query(RuleStrategyDemoAccountSyncState).all()
        assert {state.strategy_id for state in states} == {"strategy-a", "strategy-b"}
        assert all(state.latest_snapshot_id is not None for state in states)
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_reconciliation_keeps_gate_closed_only_while_execution_is_unsettled(
    monkeypatch,
):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        session.add(
            RuleStrategy(
                strategy_id="strategy-a",
                tenant_id="tenant-a",
                name="A",
                status="running",
                config={
                    "execution": {
                        "environment": "okx_demo",
                        "sandbox_connection_id": "credential-a",
                    },
                    "initial_capital_quote": 300,
                },
            )
        )
        session.commit()
        monkeypatch.setattr(sync_module, "SandboxExchangeTradingService", FakeExchange)
        monkeypatch.setattr(
            sync_module,
            "get_settings",
            lambda: type(
                "Settings",
                (),
                {
                    "DEMO_ACCOUNT_SYNC_ATTEMPTS": 1,
                    "DEMO_ACCOUNT_SYNC_RETRY_DELAY_S": 0.0,
                    "DEMO_ACCOUNT_READ_TIMEOUT_S": 1.0,
                    "DEMO_ACCOUNT_SYNC_INTERVAL_S": 300,
                },
            )(),
        )

        asyncio.run(sync_module.sync_demo_account_snapshots(session))
        shared = session.query(StrategySharedAccount).one()
        assert shared.attribution_status == "complete"

        intent = RuleStrategyExecutionIntent(
            strategy_id="strategy-a",
            evaluation_id="evaluation-1",
            execution_generation=1,
            execution_source="rule_strategy",
            tenant_id="tenant-a",
            credential_id=shared.credential_id,
            idempotency_key="intent-key-1",
            symbol="BTC/USDT",
            side="sell",
            order_type="market",
            requested_quote="100",
            execution_target="okx_demo",
            status="submitting",
        )
        session.add(intent)
        session.commit()

        asyncio.run(sync_module.sync_demo_account_snapshots(session))
        state = session.query(SharedDemoAccountSyncState).one()
        assert state.reconciliation_status == "reconciling"
        assert state.unresolved_submission_count == 1
        assert session.query(StrategySharedAccount).one().attribution_status == "partial"

        intent.status = "submission_unknown"
        session.commit()
        asyncio.run(sync_module.sync_demo_account_snapshots(session))
        assert session.query(SharedDemoAccountSyncState).one().reconciliation_status == "reconciling"

        # Dust sells are audited no-ops: they must not hold the entry gate shut.
        intent.status = "ignored_dust"
        session.commit()
        asyncio.run(sync_module.sync_demo_account_snapshots(session))
        state = session.query(SharedDemoAccountSyncState).one()
        assert state.reconciliation_status == "complete"
        assert state.unresolved_submission_count == 0
        assert state.last_reconciled_at is not None
        assert session.query(StrategySharedAccount).one().attribution_status == "complete"
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_reconciliation_counts_only_work_the_shared_chain_can_advance(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        session.add(
            RuleStrategy(
                strategy_id="strategy-a",
                tenant_id="tenant-a",
                name="A",
                status="running",
                config={
                    "execution": {
                        "environment": "okx_demo",
                        "sandbox_connection_id": "credential-a",
                    },
                    "initial_capital_quote": 300,
                },
            )
        )
        session.commit()
        monkeypatch.setattr(sync_module, "SandboxExchangeTradingService", FakeExchange)
        monkeypatch.setattr(
            sync_module,
            "get_settings",
            lambda: type(
                "Settings",
                (),
                {
                    "DEMO_ACCOUNT_SYNC_ATTEMPTS": 1,
                    "DEMO_ACCOUNT_SYNC_RETRY_DELAY_S": 0.0,
                    "DEMO_ACCOUNT_READ_TIMEOUT_S": 1.0,
                    "DEMO_ACCOUNT_SYNC_INTERVAL_S": 300,
                },
            )(),
        )
        asyncio.run(sync_module.sync_demo_account_snapshots(session))
        shared = session.query(StrategySharedAccount).one()
        assert shared.attribution_status == "complete"

        stale_at = datetime.now(timezone.utc) - timedelta(hours=6)
        abandoned = RuleStrategyExecutionIntent(
            strategy_id="strategy-a",
            evaluation_id="evaluation-abandoned",
            execution_generation=1,
            execution_source="rule_strategy",
            tenant_id="tenant-a",
            batch_id="batch-a",
            credential_id="credential-a",
            idempotency_key="intent-key-abandoned",
            symbol="BTC/USDT",
            side="sell",
            order_type="market",
            requested_quote="100",
            execution_target="okx_demo",
            status="submission_unknown",
            created_at=stale_at,
            updated_at=stale_at,
        )
        session.add(abandoned)
        session.commit()

        asyncio.run(sync_module.sync_demo_account_snapshots(session))
        state = session.query(SharedDemoAccountSyncState).one()
        assert state.reconciliation_status == "complete"
        assert state.unresolved_submission_count == 0
        assert session.query(StrategySharedAccount).one().attribution_status == "complete"
        session.refresh(abandoned)
        assert abandoned.status == "stale"
        assert abandoned.error_code == "stale_unbound_submission"
        assert abandoned.terminal_at is not None

        # A chain-bound submission is real open work and must still gate.
        bound = RuleStrategyExecutionIntent(
            strategy_id="strategy-a",
            evaluation_id="evaluation-bound",
            execution_generation=1,
            execution_source="rule_strategy",
            tenant_id="tenant-a",
            batch_id="batch-a",
            credential_id="credential-a",
            idempotency_key="intent-key-bound",
            symbol="BTC/USDT",
            side="buy",
            order_type="market",
            requested_quote="100",
            execution_target="okx_demo",
            status="submitting",
            created_at=stale_at,
            updated_at=stale_at,
        )
        session.add(
            SharedDemoExecutionReservation(
                reservation_id="reservation-bound",
                account_id=shared.id,
                tenant_id="tenant-a",
                credential_id="credential-a",
                strategy_id="strategy-a",
                batch_id="batch-a",
                idempotency_key="intent-key-bound",
                symbol="BTC/USDT",
                side="buy",
                requested_quote=100,
                reserved_quote=100,
            )
        )
        session.add(bound)
        session.flush()
        session.add(
            SharedDemoExecutionIntent(
                intent_id=bound.id,
                reservation_id="reservation-bound",
                account_id=shared.id,
                tenant_id="tenant-a",
                credential_id="credential-a",
                strategy_id="strategy-a",
                batch_id="batch-a",
                client_order_id="intent-key-bound",
            )
        )
        session.commit()

        asyncio.run(sync_module.sync_demo_account_snapshots(session))
        state = session.query(SharedDemoAccountSyncState).one()
        assert state.reconciliation_status == "reconciling"
        assert state.unresolved_submission_count == 1
        assert session.query(StrategySharedAccount).one().attribution_status == "partial"
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()
