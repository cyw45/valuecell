from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from valuecell.server.api.schemas.leader_spot_v19_snapshots import (
    LeaderSpotV19BookLevel,
    LeaderSpotV19MarketInput,
    LeaderSpotV19OrderBookSnapshot,
    LeaderSpotV19RankingItem,
    LeaderSpotV19RankingSnapshot,
)
from valuecell.server.db.models.base import Base

from valuecell.server.db.models.leader_spot_v19 import (
    LeaderSpotV19CandidateSnapshot,
    LeaderSpotV19ExecutionBatch,
    LeaderSpotV19MarketSnapshot,
    LeaderSpotV19Strategy,
)
from valuecell.server.db.models.tenant import Tenant
from valuecell.server.services.leader_spot_v19_snapshot_service import (
    LeaderSpotV19SnapshotCollector,
)


class FakeRankingProvider:
    def __init__(self, snapshot: LeaderSpotV19RankingSnapshot) -> None:
        self.snapshot = snapshot
        self.calls = 0

    async def fetch_ranking(self) -> LeaderSpotV19RankingSnapshot:
        self.calls += 1
        return self.snapshot


class FakeMarketProvider:
    def __init__(self, inputs: list[LeaderSpotV19MarketInput]) -> None:
        self.inputs = inputs
        self.calls = 0

    async def fetch_market_inputs(self, symbols, intervals):
        self.calls += 1
        assert tuple(intervals) == ("1m", "5m", "15m")
        assert symbols == ["BTC-USDT"]
        return self.inputs


def _fixtures():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    tenant = Tenant(id="tenant-c", name="C")
    strategy = LeaderSpotV19Strategy(
        strategy_id="leader-c",
        tenant_id=tenant.id,
        name="C",
        status="running",
        environment="paper",
        config={"module_id": "leader_spot_v19_0", "schema_version": 19},
    )
    session.add(tenant)
    session.commit()
    session.add(strategy)
    session.commit()
    batch = LeaderSpotV19ExecutionBatch(
        batch_id="batch-c",
        tenant_id=tenant.id,
        strategy_id=strategy.strategy_id,
        strategy_name_snapshot=strategy.name,
        execution_generation=1,
        config_snapshot=strategy.config,
    )
    session.add(batch)
    session.commit()
    return session, batch


def _ranking(now: datetime, *, expiry: datetime | None = None):
    return LeaderSpotV19RankingSnapshot(
        source="okx",
        observed_at=now - timedelta(seconds=5),
        expires_at=expiry or now + timedelta(seconds=55),
        source_snapshot_id="okx-ranking-c",
        items=[
            LeaderSpotV19RankingItem(
                symbol="BTC/USDT",
                rank=1,
                quote_volume_24h=1_000_000,
                listing_at=now - timedelta(days=100),
                spot_tradable=True,
                quote_asset="USDT",
            )
        ],
    )


def _market(now: datetime):
    book = LeaderSpotV19OrderBookSnapshot(
        symbol="BTC/USDT",
        bids=[LeaderSpotV19BookLevel(price=100, quantity=2)],
        asks=[LeaderSpotV19BookLevel(price=101, quantity=2)],
        observed_at=now,
        source="okx",
    )
    return [
        LeaderSpotV19MarketInput(
            symbol="BTC/USDT",
            interval=interval,
            source="okx",
            candles=[
                {"timestamp_ms": 1, "open": 99, "high": 102, "low": 98, "close": 100, "volume": 4}
            ],
            latest_price=100,
            order_book=book,
            observed_at=now,
            expires_at=now + timedelta(minutes=2),
        )
        for interval in ("1m", "5m", "15m")
    ]


@pytest.mark.asyncio
async def test_collector_persists_fresh_ranking_market_inputs_and_candidates():
    now = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    session, batch = _fixtures()
    ranking_provider = FakeRankingProvider(_ranking(now))
    market_provider = FakeMarketProvider(_market(now))

    result = await LeaderSpotV19SnapshotCollector(session).collect(
        tenant_id=batch.tenant_id,
        strategy_id=batch.strategy_id,
        batch_id=batch.batch_id,
        ranking_provider=ranking_provider,
        market_provider=market_provider,
        now=now,
    )

    assert result.data_state == "DATA_OK"
    assert result.ranking_fresh is True
    assert result.market_snapshot_count == 3
    assert result.accepted_candidate_count == 1
    assert ranking_provider.calls == 1
    assert session.query(LeaderSpotV19MarketSnapshot).count() == 4
    candidate = session.query(LeaderSpotV19CandidateSnapshot).first()
    assert candidate is not None
    assert candidate.accepted is True
    assert candidate.symbol == "BTC-USDT"


@pytest.mark.asyncio
async def test_expired_or_incomplete_ranking_fails_closed_without_market_fetch():
    now = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    session, batch = _fixtures()
    expired = _ranking(now, expiry=now - timedelta(seconds=1))
    expired = expired.model_copy(update={"completeness": "partial"})
    ranking_provider = FakeRankingProvider(expired)
    market_provider = FakeMarketProvider([])

    result = await LeaderSpotV19SnapshotCollector(session).collect(
        tenant_id=batch.tenant_id,
        strategy_id=batch.strategy_id,
        batch_id=batch.batch_id,
        ranking_provider=ranking_provider,
        market_provider=market_provider,
        now=now,
    )

    assert result.data_state == "DATA_UNSAFE"
    assert result.ranking_fresh is False
    assert result.market_snapshot_count == 0
    assert market_provider.calls == 0
    candidate = session.query(LeaderSpotV19CandidateSnapshot).first()
    assert candidate.accepted is False
    assert candidate.reason_code == "ranking_snapshot_expired"


def test_v19_snapshot_contract_rejects_invalid_book_order():
    with pytest.raises(ValueError):
        LeaderSpotV19OrderBookSnapshot(
            symbol="BTC-USDT",
            bids=[
                LeaderSpotV19BookLevel(price=99, quantity=1),
                LeaderSpotV19BookLevel(price=100, quantity=1),
            ],
            asks=[LeaderSpotV19BookLevel(price=101, quantity=1)],
            observed_at=datetime.now(UTC),
            source="okx",
        )
