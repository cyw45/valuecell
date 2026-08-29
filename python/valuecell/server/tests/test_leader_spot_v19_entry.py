from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from valuecell.server.api.schemas.leader_spot_v19 import LeaderSpotV19Config
from valuecell.server.api.schemas.leader_spot_v19_candidate import (
    LeaderSpotV19CandidateDecision,
)
from valuecell.server.api.schemas.leader_spot_v19_entry import (
    LeaderSpotV19EntryOrderResult,
    LeaderSpotV19EntryRequest,
)
from valuecell.server.api.schemas.leader_spot_v19_snapshots import (
    LeaderSpotV19BookLevel,
    LeaderSpotV19OrderBookSnapshot,
)
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.leader_spot_v19 import (
    LeaderSpotV19ExecutionBatch,
    LeaderSpotV19ExecutionIntent,
    LeaderSpotV19Fill,
    LeaderSpotV19OrderAttempt,
    LeaderSpotV19Position,
    LeaderSpotV19Strategy,
)
from valuecell.server.db.models.tenant import Tenant
from valuecell.server.services.leader_spot_v19_entry_service import (
    LeaderSpotV19EntryCoordinator,
)


NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


def _candidate(accepted: bool = True):
    return LeaderSpotV19CandidateDecision(
        symbol="BTC-USDT",
        source_rank=1,
        accepted=accepted,
        score=50 if accepted else None,
        reason_code=None if accepted else "score_below_threshold",
        steps=[],
        observed_at=NOW,
    )


def _request(**changes):
    values = {
        "signal_id": "signal-g",
        "candidate": _candidate(),
        "confirmation_price": 100,
        "open_position_count": 0,
        "observed_at": NOW,
    }
    values.update(changes)
    return LeaderSpotV19EntryRequest(**values)


def _book():
    return LeaderSpotV19OrderBookSnapshot(
        symbol="BTC-USDT",
        bids=[LeaderSpotV19BookLevel(price=100, quantity=2)],
        asks=[LeaderSpotV19BookLevel(price=100, quantity=2)],
        observed_at=NOW,
        source="okx",
    )


def _fixtures():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    tenant = Tenant(id="tenant-g", name="G")
    session.add(tenant)
    session.commit()
    strategy = LeaderSpotV19Strategy(
        strategy_id="leader-g",
        tenant_id=tenant.id,
        name="G",
        status="running",
        environment="paper",
        config={"module_id": "leader_spot_v19_0", "schema_version": 19},
    )
    session.add(strategy)
    session.commit()
    batch = LeaderSpotV19ExecutionBatch(
        batch_id="batch-g",
        tenant_id=tenant.id,
        strategy_id=strategy.strategy_id,
        strategy_name_snapshot=strategy.name,
        execution_generation=1,
        config_snapshot=strategy.config,
    )
    session.add(batch)
    session.commit()
    return session, batch


class FakeVenue:
    venue = "okx_demo"

    def __init__(self, submit_statuses, wait_statuses=()):
        self.submit_statuses = iter(submit_statuses)
        self.wait_statuses = iter(wait_statuses)
        self.submits = []
        self.books = 0

    async def current_order_book(self, symbol):
        self.books += 1
        return _book()

    async def submit_limit_buy(self, *, client_order_id, symbol, quote_amount, price):
        self.submits.append((client_order_id, quote_amount, price))
        status = next(self.submit_statuses)
        return LeaderSpotV19EntryOrderResult(
            client_order_id=client_order_id,
            venue_order_id=f"venue-{client_order_id}",
            status=status,
            filled_quantity=1 if status == "filled" else 0,
            average_price=price if status == "filled" else None,
            fee_quote=0.1 if status == "filled" else 0,
        )

    async def wait_for_order(self, client_order_id, timeout_seconds):
        status = next(self.wait_statuses)
        return LeaderSpotV19EntryOrderResult(
            client_order_id=client_order_id,
            venue_order_id=f"venue-{client_order_id}",
            status=status,
            filled_quantity=1 if status == "filled" else 0,
            average_price=100.5 if status == "filled" else None,
            fee_quote=0.1 if status == "filled" else 0,
        )

    async def cancel_order(self, client_order_id):
        return LeaderSpotV19EntryOrderResult(
            client_order_id=client_order_id,
            venue_order_id=f"venue-{client_order_id}",
            status="cancelled",
        )


@pytest.mark.asyncio
async def test_entry_uses_fixed_amount_and_advances_only_after_tier_timeout():
    session, batch = _fixtures()
    venue = FakeVenue(["open", "filled"], ["open"])

    decision = await LeaderSpotV19EntryCoordinator(session).execute(
        tenant_id=batch.tenant_id,
        strategy_id=batch.strategy_id,
        batch_id=batch.batch_id,
        execution_generation=1,
        config=LeaderSpotV19Config(),
        request=_request(),
        venue=venue,
        now=NOW,
    )

    assert decision.accepted is True
    assert decision.order_amount_quote == 100
    assert len(venue.submits) == 2
    assert [item[1] for item in venue.submits] == [100, 100]
    assert venue.submits[0][2] == pytest.approx(100.3)
    assert venue.submits[1][2] == pytest.approx(100.5)
    assert session.query(LeaderSpotV19ExecutionIntent).count() == 2
    assert session.query(LeaderSpotV19OrderAttempt).count() == 2
    assert session.query(LeaderSpotV19Fill).count() == 1
    position = session.query(LeaderSpotV19Position).one()
    assert position.protection_status == "PROTECTION_NONE"
    assert float(position.moving_stop_price) == pytest.approx(92.46)


@pytest.mark.asyncio
async def test_entry_refuses_candidate_capacity_held_and_cooldown_violations():
    session, batch = _fixtures()
    venue = FakeVenue([])
    coordinator = LeaderSpotV19EntryCoordinator(session)

    for request, reason in (
        (_request(candidate=_candidate(False)), "candidate_not_accepted"),
        (_request(open_position_count=6), "max_positions_reached"),
        (_request(held_symbols=["BTC-USDT"]), "symbol_already_held"),
        (_request(cooldown_symbols=["BTC-USDT"]), "symbol_cooldown_active"),
    ):
        decision = await coordinator.execute(
            tenant_id=batch.tenant_id,
            strategy_id=batch.strategy_id,
            batch_id=batch.batch_id,
            execution_generation=1,
            config=LeaderSpotV19Config(),
            request=request,
            venue=venue,
            now=NOW,
        )
        assert decision.reason_code == reason
    assert venue.submits == []
    assert session.query(LeaderSpotV19ExecutionIntent).count() == 0


@pytest.mark.asyncio
async def test_entry_never_retries_submission_unknown_with_same_signal():
    session, batch = _fixtures()
    venue = FakeVenue(["submission_unknown"])
    coordinator = LeaderSpotV19EntryCoordinator(session)

    first = await coordinator.execute(
        tenant_id=batch.tenant_id,
        strategy_id=batch.strategy_id,
        batch_id=batch.batch_id,
        execution_generation=1,
        config=LeaderSpotV19Config(),
        request=_request(),
        venue=venue,
        now=NOW,
    )
    second = await coordinator.execute(
        tenant_id=batch.tenant_id,
        strategy_id=batch.strategy_id,
        batch_id=batch.batch_id,
        execution_generation=1,
        config=LeaderSpotV19Config(),
        request=_request(),
        venue=venue,
        now=NOW,
    )

    assert first.reason_code == "entry_submission_unknown"
    assert second.reason_code == "entry_submission_unknown"
    assert len(venue.submits) == 1
    assert session.query(LeaderSpotV19ExecutionIntent).one().status == "submission_unknown"
