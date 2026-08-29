from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from valuecell.server.api.schemas.leader_spot_v19 import LeaderSpotV19Config
from valuecell.server.api.schemas.leader_spot_v19_candidate import (
    LeaderSpotV19BoxBreakoutEvidence,
    LeaderSpotV19CandidateInput,
    LeaderSpotV19ScoreEvidence,
)
from valuecell.server.api.schemas.leader_spot_v19_market_state import (
    LeaderSpotV19MarketStateDecision,
    LeaderSpotV19SignalStarvationPolicy,
)
from valuecell.server.api.schemas.leader_spot_v19_snapshots import (
    LeaderSpotV19BookLevel,
    LeaderSpotV19OrderBookSnapshot,
)
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.leader_spot_v19 import (
    LeaderSpotV19CandidateSnapshot,
    LeaderSpotV19ExecutionBatch,
    LeaderSpotV19Strategy,
)
from valuecell.server.db.models.tenant import Tenant
from valuecell.server.services.leader_spot_v19_candidate_service import (
    LeaderSpotV19CandidateFilter,
)


NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


def _market(state: str = "M3"):
    return LeaderSpotV19MarketStateDecision(
        market_state=state,
        entry_profile={"M0": "halt", "M1": "halt", "M2": "degraded", "M3": "standard", "M4": "strong_trend"}[state],
        can_open=state in {"M2", "M3", "M4"},
        reason_codes=[],
        starvation=LeaderSpotV19SignalStarvationPolicy(
            elapsed_hours=0,
            recovered=False,
            relative_strength_rank_pct=0.18,
            liquidity_quote=200_000,
            score_threshold=35,
        ),
        observed_at=NOW,
    )


def _book(depth: float = 2):
    return LeaderSpotV19OrderBookSnapshot(
        symbol="BTC-USDT",
        bids=[LeaderSpotV19BookLevel(price=100, quantity=depth)],
        asks=[LeaderSpotV19BookLevel(price=101, quantity=depth)],
        observed_at=NOW,
        source="okx",
    )


def _box(volume_multiplier: float = 1.0):
    return LeaderSpotV19BoxBreakoutEvidence(
        parameter_source="V16.1",
        parameter_fingerprint="v16-box-fingerprint",
        upper_bound=100,
        fifteen_minute_close_confirmed=True,
        five_minute_close_confirmations=2,
        second_five_minute_volume_confirmed=True,
        volume_multiplier=volume_multiplier,
        passed=True,
    )


def _score(total_score: float = 35):
    return LeaderSpotV19ScoreEvidence(
        formula_source="external_v19_score",
        formula_fingerprint="score-fingerprint",
        total_score=total_score,
        factors={"momentum": 20, "volume": 15},
    )


def _candidate(**changes):
    values = {
        "symbol": "BTC-USDT",
        "source_rank": 1,
        "market_state": "M3",
        "data_state": "DATA_OK",
        "quote_volume_24h": 250_000,
        "listing_at": NOW - timedelta(hours=100),
        "relative_strength_rank_pct": 0.10,
        "return_24h_pct": 0.10,
        "needle_detected": False,
        "br_value": 200,
        "order_book": _book(),
        "box": _box(),
        "score": _score(),
        "observed_at": NOW,
    }
    values.update(changes)
    return LeaderSpotV19CandidateInput(**values)


def _fixtures():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    tenant = Tenant(id="tenant-f", name="F")
    session.add(tenant)
    session.commit()
    strategy = LeaderSpotV19Strategy(
        strategy_id="leader-f",
        tenant_id=tenant.id,
        name="F",
        status="running",
        environment="paper",
        config={"module_id": "leader_spot_v19_0", "schema_version": 19},
    )
    session.add(strategy)
    session.commit()
    batch = LeaderSpotV19ExecutionBatch(
        batch_id="batch-f",
        tenant_id=tenant.id,
        strategy_id=strategy.strategy_id,
        strategy_name_snapshot=strategy.name,
        execution_generation=1,
        config_snapshot=strategy.config,
    )
    session.add(batch)
    session.commit()
    return session, batch


def test_candidate_filter_accepts_complete_v19_evidence():
    decision = LeaderSpotV19CandidateFilter().evaluate(
        LeaderSpotV19Config(), _market(), _candidate()
    )

    assert decision.accepted is True
    assert [step.stage for step in decision.steps] == [
        "entry_state", "liquidity", "new_coin", "relative_strength",
        "anomaly", "box_breakout", "score", "order_book",
    ]


def test_candidate_filter_fails_closed_without_box_or_score_formula():
    filter_service = LeaderSpotV19CandidateFilter()
    without_box = filter_service.evaluate(
        LeaderSpotV19Config(), _market(), _candidate(box=None)
    )
    assert without_box.accepted is False
    assert without_box.reason_code == "v16_1_box_parameters_unavailable"

    without_score = filter_service.evaluate(
        LeaderSpotV19Config(), _market(), _candidate(score=None)
    )
    assert without_score.accepted is False
    assert without_score.reason_code == "score_formula_unavailable"


def test_candidate_filter_applies_new_coin_and_high_pump_safety_gates():
    filter_service = LeaderSpotV19CandidateFilter()
    banned = filter_service.evaluate(
        LeaderSpotV19Config(), _market(), _candidate(listing_at=NOW - timedelta(hours=5))
    )
    assert banned.reason_code == "new_coin_banned"

    young = filter_service.evaluate(
        LeaderSpotV19Config(), _market(), _candidate(
            listing_at=NOW - timedelta(hours=12),
            strict_new_coin_requirements_met=False,
        )
    )
    assert young.reason_code == "new_coin_strict_requirements_unmet"

    high_pump = filter_service.evaluate(
        LeaderSpotV19Config(), _market("M2"), _candidate(return_24h_pct=0.41)
    )
    assert high_pump.reason_code == "high_pump_not_retest_confirmed"


def test_candidate_filter_applies_24_to_72_hour_volume_score_uplifts():
    candidate = _candidate(
        listing_at=NOW - timedelta(hours=48),
        box=_box(volume_multiplier=1.2),
        score=_score(41),
    )
    decision = LeaderSpotV19CandidateFilter().evaluate(
        LeaderSpotV19Config(), _market(), candidate
    )
    assert decision.reason_code == "score_below_threshold"
    assert decision.steps[-1].facts["threshold"] == 42

    accepted = LeaderSpotV19CandidateFilter().evaluate(
        LeaderSpotV19Config(), _market(), _candidate(
            listing_at=NOW - timedelta(hours=48),
            box=_box(volume_multiplier=1.2),
            score=_score(42),
        )
    )
    assert accepted.accepted is True


def test_candidate_filter_persists_full_funnel_evidence():
    session, batch = _fixtures()
    decision = LeaderSpotV19CandidateFilter().evaluate_and_persist(
        session,
        tenant_id=batch.tenant_id,
        strategy_id=batch.strategy_id,
        batch_id=batch.batch_id,
        snapshot_group_id="group-f",
        config=LeaderSpotV19Config(),
        market=_market(),
        candidate=_candidate(),
    )

    assert decision.accepted is True
    row = session.query(LeaderSpotV19CandidateSnapshot).one()
    assert row.accepted is True
    assert row.funnel_stage == "order_book"
    assert len(row.facts["steps"]) == 8
