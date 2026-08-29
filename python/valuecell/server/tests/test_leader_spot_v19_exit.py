from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from valuecell.server.api.schemas.leader_spot_v19 import LeaderSpotV19Config
from valuecell.server.api.schemas.leader_spot_v19_exit import LeaderSpotV19PositionExitInput
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.leader_spot_v19 import (
    LeaderSpotV19ExecutionBatch,
    LeaderSpotV19ExecutionIntent,
    LeaderSpotV19Position,
    LeaderSpotV19Strategy,
)
from valuecell.server.db.models.tenant import Tenant
from valuecell.server.services.leader_spot_v19_exit_service import LeaderSpotV19ExitEngine


NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


def _input(**changes):
    values = {
        "position_id": "position-h",
        "symbol": "BTC-USDT",
        "entry_price": 100,
        "quantity": 1,
        "entry_time": NOW - timedelta(hours=1),
        "protection_status": "PROTECTION_NONE",
        "peak_price": 100,
        "peak_profit_pct": 0,
        "moving_stop_price": 92,
        "loss_circuit_active": True,
        "trend_break_count": 0,
        "current_bid": 100,
        "market_state": "M3",
        "observed_at": NOW,
    }
    values.update(changes)
    return LeaderSpotV19PositionExitInput(**values)


def _fixtures():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    tenant = Tenant(id="tenant-h", name="H")
    session.add(tenant)
    session.commit()
    strategy = LeaderSpotV19Strategy(
        strategy_id="leader-h", tenant_id=tenant.id, name="H", status="running",
        environment="paper", config={"module_id": "leader_spot_v19_0", "schema_version": 19},
    )
    session.add(strategy)
    session.commit()
    batch = LeaderSpotV19ExecutionBatch(
        batch_id="batch-h", tenant_id=tenant.id, strategy_id=strategy.strategy_id,
        strategy_name_snapshot=strategy.name, execution_generation=1,
        config_snapshot=strategy.config,
    )
    session.add(batch)
    session.commit()
    position = LeaderSpotV19Position(
        position_id="position-h", tenant_id=tenant.id, strategy_id=strategy.strategy_id,
        batch_id=batch.batch_id, symbol="BTC-USDT", entry_price="100", entry_quantity="1",
        entry_time=NOW - timedelta(hours=1), protection_status="PROTECTION_NONE",
        peak_price="100", moving_stop_price="92", loss_circuit_started_at=NOW - timedelta(hours=1),
    )
    session.add(position)
    session.commit()
    return session, position


def test_exit_engine_prioritizes_hard_stop_then_loss_circuit_before_protection():
    engine = LeaderSpotV19ExitEngine()
    stopped = engine.decide(
        LeaderSpotV19Config(),
        _input(current_bid=92, hard_stop_two_source_confirmed=True),
    )
    assert stopped.exit_reason_code == "STOP_LOSS_8PCT"
    assert stopped.order_type == "market"

    circuit = engine.decide(
        LeaderSpotV19Config(),
        _input(entry_time=NOW - timedelta(hours=168), current_bid=100),
    )
    assert circuit.exit_reason_code == "LOSS_CIRCUIT_7D"


def test_exit_engine_activates_protection_after_60_seconds_and_disables_loss_paths():
    engine = LeaderSpotV19ExitEngine()
    pending = engine.decide(LeaderSpotV19Config(), _input(current_bid=105))
    assert pending.protection_status == "PROTECTION_PENDING"
    active = engine.decide(
        LeaderSpotV19Config(),
        _input(
            current_bid=106,
            protection_status="PROTECTION_PENDING",
            protection_started_at=NOW - timedelta(seconds=60),
            closed_one_minute_high=107,
        ),
    )
    assert active.protection_status == "PROTECTION_ACTIVE"
    assert active.loss_circuit_active is False
    assert active.peak_price == 107


def test_exit_engine_applies_moving_before_layered_and_stages_once():
    engine = LeaderSpotV19ExitEngine()
    direct = engine.decide(
        LeaderSpotV19Config(),
        _input(
            protection_status="PROTECTION_ACTIVE", current_bid=109,
            peak_price=115, peak_profit_pct=0.15, moving_stop_price=110,
            closed_one_minute_high=115,
        ),
    )
    assert direct.exit_reason_code == "MOVING_STOP"

    session, position = _fixtures()
    position.protection_status = "PROTECTION_ACTIVE"
    position.peak_price = "115"
    position.peak_profit_pct = 0.15
    position.moving_stop_price = "110"
    session.commit()
    first = engine.decide_and_stage_exit(
        session, config=LeaderSpotV19Config(), position=position, market_state="M3",
        current_bid=109, closed_one_minute_high=115, fifteen_minute_closes=[],
        execution_generation=1, now=NOW,
    )
    second = engine.decide_and_stage_exit(
        session, config=LeaderSpotV19Config(), position=position, market_state="M3",
        current_bid=109, closed_one_minute_high=115, fifteen_minute_closes=[],
        execution_generation=1, now=NOW,
    )
    assert first.exit_reason_code == "MOVING_STOP"
    assert second.exit_reason_code == "MOVING_STOP"
    intent = session.query(LeaderSpotV19ExecutionIntent).one()
    assert intent.side == "sell"
    assert intent.order_type == "market"


def test_exit_engine_trend_exit_requires_two_closes_and_falling_fast_ema():
    values = [100 + index for index in range(55)] + [140, 130]
    engine = LeaderSpotV19ExitEngine()
    first = engine.decide(
        LeaderSpotV19Config(),
        _input(
            protection_status="PROTECTION_ACTIVE",
            current_bid=125,
            peak_price=130,
            peak_profit_pct=0.3,
            moving_stop_price=110,
            trend_data_valid=True,
            fifteen_minute_closes=values,
        ),
    )
    assert first.exit_reason_code is None
    second = engine.decide(
        LeaderSpotV19Config(),
        _input(
            protection_status="PROTECTION_ACTIVE", current_bid=125, peak_price=130,
            peak_profit_pct=0.3, moving_stop_price=110, trend_break_count=1,
            trend_data_valid=True, fifteen_minute_closes=values,
        ),
    )
    assert second.exit_reason_code == "TREND_EXIT"
    assert second.order_type == "limit"
