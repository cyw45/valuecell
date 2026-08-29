from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from valuecell.server.api.schemas.leader_spot_v19_quality import (
    LeaderSpotV19DataQualityReport,
    LeaderSpotV19PriceObservation,
    LeaderSpotV19QualityInput,
    LeaderSpotV19RecoveryExit,
    LeaderSpotV19RecoveryObservation,
)
from valuecell.server.api.schemas.leader_spot_v19_snapshots import (
    LeaderSpotV19BookLevel,
    LeaderSpotV19MarketInput,
    LeaderSpotV19OrderBookSnapshot,
)
from valuecell.server.db import migrations
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.leader_spot_v19 import (
    LeaderSpotV19DataQualityReport as QualityRow,
    LeaderSpotV19Event,
    LeaderSpotV19ExecutionBatch,
    LeaderSpotV19Strategy,
)
from valuecell.server.db.models.tenant import Tenant
from valuecell.server.services.leader_spot_v19_quality_service import (
    LeaderSpotV19DataQualityGate,
)
from valuecell.server.services.leader_spot_v19_recovery_service import (
    LeaderSpotV19RecoveryCoordinator,
)


NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


def _market_inputs(now: datetime, *, gap: bool = False):
    book = LeaderSpotV19OrderBookSnapshot(
        symbol="BTC-USDT",
        bids=[LeaderSpotV19BookLevel(price=100, quantity=2)],
        asks=[LeaderSpotV19BookLevel(price=101, quantity=2)],
        observed_at=now,
        source="okx",
    )
    inputs = []
    for interval, step in (("1m", 60_000), ("5m", 300_000), ("15m", 900_000)):
        timestamps = [1, 1 + step * (2 if gap else 1)]
        inputs.append(
            LeaderSpotV19MarketInput(
                symbol="BTC-USDT",
                interval=interval,
                source="okx",
                candles=[
                    {
                        "timestamp_ms": timestamps[0],
                        "open": 99,
                        "high": 102,
                        "low": 98,
                        "close": 100,
                        "volume": 4,
                    },
                    {
                        "timestamp_ms": timestamps[1],
                        "open": 100,
                        "high": 103,
                        "low": 99,
                        "close": 101,
                        "volume": 5,
                    },
                ],
                latest_price=100,
                order_book=book,
                observed_at=now,
                expires_at=now + timedelta(minutes=2),
            )
        )
    return inputs


def _price(source: str, value: float, now: datetime = NOW):
    return LeaderSpotV19PriceObservation(
        symbol="BTC-USDT",
        source=source,
        price=value,
        observed_at=now,
    )


def _quality_input(*, gap: bool = False, secondary: bool = True, btc_secondary: bool = True):
    return LeaderSpotV19QualityInput(
        market_inputs=_market_inputs(NOW, gap=gap),
        primary_prices=[_price("okx", 100)],
        secondary_prices=[_price("binance", 100.5)] if secondary else [],
        btc_prices=[_price("okx", 100)],
        btc_secondary_prices=[_price("binance", 100.5)] if btc_secondary else [],
        required_symbols=["BTC-USDT"],
        observed_at=NOW,
    )


def _fixtures():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    tenant = Tenant(id="tenant-d", name="D")
    session.add(tenant)
    session.commit()
    strategy = LeaderSpotV19Strategy(
        strategy_id="leader-d",
        tenant_id=tenant.id,
        name="D",
        status="running",
        environment="paper",
        config={"module_id": "leader_spot_v19_0", "schema_version": 19},
    )
    session.add(strategy)
    session.commit()
    batch = LeaderSpotV19ExecutionBatch(
        batch_id="batch-d",
        tenant_id=tenant.id,
        strategy_id=strategy.strategy_id,
        strategy_name_snapshot=strategy.name,
        execution_generation=1,
        config_snapshot=strategy.config,
    )
    session.add(batch)
    session.commit()
    return session, batch


class FakeRecoveryVenue:
    def __init__(self, observation: LeaderSpotV19RecoveryObservation):
        self.observation = observation
        self.reconcile_calls = 0
        self.submitted: list[str] = []

    async def reconcile(self, tenant_id, strategy_id, batch_id):
        self.reconcile_calls += 1
        return self.observation

    async def submit_recovery_exit(self, tenant_id, strategy_id, batch_id, exit_request):
        self.submitted.append(exit_request.symbol)
        return f"venue-{exit_request.symbol}"


def test_quality_gate_accepts_complete_fresh_consistent_inputs():
    report = LeaderSpotV19DataQualityGate().evaluate(_quality_input(), now=NOW)

    assert isinstance(report, LeaderSpotV19DataQualityReport)
    assert report.data_state == "DATA_OK"
    assert report.accepted_for_entry is True
    assert report.fresh_input_count == 3
    assert report.issues == []
def test_quality_gate_rejects_gap_and_candidate_price_conflict():
    report = LeaderSpotV19DataQualityGate().evaluate(
        _quality_input(gap=True, secondary=False), now=NOW
    )

    assert report.data_state == "DATA_UNSAFE"
    assert report.accepted_for_entry is False
    assert {issue.code for issue in report.issues} >= {
        "candle_gap",
        "price_secondary_missing",
    }


def test_quality_gate_marks_missing_btc_secondary_degraded():
    report = LeaderSpotV19DataQualityGate().evaluate(
        _quality_input(btc_secondary=False), now=NOW
    )

    assert report.data_state == "DATA_DEGRADED"
    assert report.accepted_for_entry is False
    assert any(issue.code == "btc_price_secondary_missing" for issue in report.issues)


def test_quality_gate_persists_report_with_isolated_migration():
    session, batch = _fixtures()
    assert migrations.migrate_leader_spot_v19_quality(session) is True
    assert migrations.migrate_leader_spot_v19_quality(session) is False

    report = LeaderSpotV19DataQualityGate().evaluate_and_persist(
        session,
        tenant_id=batch.tenant_id,
        strategy_id=batch.strategy_id,
        batch_id=batch.batch_id,
        quality_input=_quality_input(),
        now=NOW,
    )

    assert report.data_state == "DATA_OK"
    row = session.query(QualityRow).one()
    assert row.batch_id == batch.batch_id
    assert row.accepted_for_entry is True
    assert migrations.LEADER_SPOT_V19_QUALITY_MIGRATION_VERSION in {
        item[0]
        for item in session.execute(text("SELECT version FROM schema_migrations")).all()
    }


@pytest.mark.asyncio
async def test_recovery_reconciles_before_submitting_only_missing_exits():
    session, batch = _fixtures()
    observation = LeaderSpotV19RecoveryObservation(
        positions=[{"symbol": "BTC-USDT", "quantity": 1}],
        orders=[{"symbol": "BTC-USDT", "status": "open"}],
        due_exits=[
            LeaderSpotV19RecoveryExit(
                symbol="BTC-USDT",
                quantity=1,
                reason_code="STOP_LOSS",
                local_triggered_at=NOW,
            ),
            LeaderSpotV19RecoveryExit(
                symbol="ETH-USDT",
                quantity=1,
                reason_code="LOSS_CIRCUIT_7D",
                local_triggered_at=NOW,
                venue_order_id="already-known",
            ),
        ],
        observed_at=NOW,
    )
    venue = FakeRecoveryVenue(observation)

    result = await LeaderSpotV19RecoveryCoordinator(session).recover(
        tenant_id=batch.tenant_id,
        strategy_id=batch.strategy_id,
        batch_id=batch.batch_id,
        venue=venue,
    )

    assert venue.reconcile_calls == 1
    assert venue.submitted == ["BTC-USDT"]
    assert result.positions_reconciled == 1
    assert result.orders_reconciled == 1
    assert result.exits_submitted == 1
    assert session.query(LeaderSpotV19Event).count() == 2
