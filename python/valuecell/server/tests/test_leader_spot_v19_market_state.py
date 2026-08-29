from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from valuecell.server.api.schemas.leader_spot_v19 import LeaderSpotV19Config
from valuecell.server.api.schemas.leader_spot_v19_market_state import (
    LeaderSpotV19MarketStateInput,
)
from valuecell.server.db import migrations
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.leader_spot_v19 import (
    LeaderSpotV19ExecutionBatch,
    LeaderSpotV19MarketStateDecision,
    LeaderSpotV19Strategy,
)
from valuecell.server.db.models.tenant import Tenant
from valuecell.server.services.leader_spot_v19_market_state_service import (
    LeaderSpotV19MarketStateEngine,
)


NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


def _inputs(**changes):
    values = {
        "data_state": "DATA_OK",
        "up_ratio": 0.50,
        "volume_ratio_to_5d_average": 1.2,
        "fear_greed_index": 40,
        "funding_rate": 0.0,
        "daily_loss_limit_reached": False,
        "valid_candidate_count": 0,
        "observed_at": NOW,
    }
    values.update(changes)
    return LeaderSpotV19MarketStateInput(**values)


def _fixtures():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    tenant = Tenant(id="tenant-e", name="E")
    session.add(tenant)
    session.commit()
    strategy = LeaderSpotV19Strategy(
        strategy_id="leader-e",
        tenant_id=tenant.id,
        name="E",
        status="running",
        environment="paper",
        config={"module_id": "leader_spot_v19_0", "schema_version": 19},
    )
    session.add(strategy)
    session.commit()
    batch = LeaderSpotV19ExecutionBatch(
        batch_id="batch-e",
        tenant_id=tenant.id,
        strategy_id=strategy.strategy_id,
        strategy_name_snapshot=strategy.name,
        execution_generation=1,
        config_snapshot=strategy.config,
    )
    session.add(batch)
    session.commit()
    return session, batch


def test_market_state_transitions_standard_degraded_strong_and_halt():
    engine = LeaderSpotV19MarketStateEngine()
    config = LeaderSpotV19Config()

    assert engine.decide(config, _inputs()).market_state == "M3"
    assert engine.decide(
        config,
        _inputs(up_ratio=0.60, volume_ratio_to_5d_average=1.4),
    ).market_state == "M4"
    degraded = engine.decide(config, _inputs(volume_ratio_to_5d_average=1.0))
    assert degraded.market_state == "M2"
    assert degraded.can_open is True
    assert engine.decide(
        config,
        _inputs(volume_ratio_to_5d_average=1.0, fear_greed_index=30),
    ).market_state == "M1"
    assert engine.decide(
        config,
        _inputs(daily_loss_limit_reached=True),
    ).market_state == "M1"


def test_market_state_data_rules_fail_closed_and_block_degraded_mode():
    engine = LeaderSpotV19MarketStateEngine()
    config = LeaderSpotV19Config()

    unsafe = engine.decide(config, _inputs(data_state="DATA_UNSAFE"))
    assert unsafe.market_state == "M0"
    assert unsafe.can_open is False
    degraded_data_standard_market = engine.decide(
        config,
        _inputs(data_state="DATA_DEGRADED"),
    )
    assert degraded_data_standard_market.market_state == "M3"
    degraded_data_one_failure = engine.decide(
        config,
        _inputs(data_state="DATA_DEGRADED", volume_ratio_to_5d_average=1.0),
    )
    assert degraded_data_one_failure.market_state == "M1"


def test_signal_starvation_only_relaxes_the_three_v19_candidate_thresholds():
    engine = LeaderSpotV19MarketStateEngine()
    config = LeaderSpotV19Config()

    at_48 = engine.decide(
        config,
        _inputs(no_valid_candidate_since=NOW - timedelta(hours=48)),
    ).starvation
    assert at_48.relative_strength_rank_pct == 0.23
    assert at_48.liquidity_quote == 200_000
    assert at_48.score_threshold == 35

    at_72 = engine.decide(
        config,
        _inputs(no_valid_candidate_since=NOW - timedelta(hours=72)),
    ).starvation
    assert at_72.relative_strength_rank_pct == 0.23
    assert at_72.liquidity_quote == 150_000
    assert at_72.score_threshold == 30

    recovered = engine.decide(
        config,
        _inputs(
            no_valid_candidate_since=NOW - timedelta(hours=100),
            valid_candidate_count=2,
        ),
    ).starvation
    assert recovered.recovered is True
    assert recovered.elapsed_hours == 0
    assert recovered.relative_strength_rank_pct == 0.18
    assert recovered.liquidity_quote == 200_000
    assert recovered.score_threshold == 35


def test_market_state_decision_persists_inputs_reasons_and_starvation_policy():
    session, batch = _fixtures()
    assert migrations.migrate_leader_spot_v19_market_state(session) is True
    assert migrations.migrate_leader_spot_v19_market_state(session) is False

    decision = LeaderSpotV19MarketStateEngine().decide_and_persist(
        session,
        tenant_id=batch.tenant_id,
        strategy_id=batch.strategy_id,
        batch_id=batch.batch_id,
        config=LeaderSpotV19Config(),
        inputs=_inputs(no_valid_candidate_since=NOW - timedelta(hours=72)),
    )

    row = session.query(LeaderSpotV19MarketStateDecision).one()
    assert row.market_state == "M3"
    assert row.input_facts["volume_ratio_to_5d_average"] == 1.2
    assert row.starvation["score_threshold"] == 30
    assert decision.reason_codes == ["standard_market"]
    assert migrations.LEADER_SPOT_V19_MARKET_STATE_MIGRATION_VERSION in {
        item[0]
        for item in session.execute(text("SELECT version FROM schema_migrations")).all()
    }
