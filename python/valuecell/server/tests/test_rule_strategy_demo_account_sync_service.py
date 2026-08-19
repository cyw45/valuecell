import asyncio

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import valuecell.server.services.rule_strategy_demo_account_sync_service as sync_module
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.rule_strategy import (
    RuleStrategy,
    RuleStrategyDemoAccountSnapshot,
    RuleStrategyDemoAccountSyncState,
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
                        }
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
                        }
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
        states = session.query(RuleStrategyDemoAccountSyncState).all()
        assert {state.strategy_id for state in states} == {"strategy-a", "strategy-b"}
        assert all(state.latest_snapshot_id is not None for state in states)
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()
