from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from valuecell.server.db import migrations
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.leader_spot_v19 import (
    LeaderSpotV19ExecutionBatch,
    LeaderSpotV19ExecutionIntent,
    LeaderSpotV19Position,
    LeaderSpotV19Strategy,
)
from valuecell.server.db.models.tenant import Tenant

from valuecell.server.db.models.tenant_credential import TenantCredential

V19_TABLES = {
    "leader_spot_v19_strategies",
    "leader_spot_v19_execution_batches",
    "leader_spot_v19_accounts",
    "leader_spot_v19_risk_states",
    "leader_spot_v19_candidate_snapshots",
    "leader_spot_v19_market_snapshots",
    "leader_spot_v19_positions",
    "leader_spot_v19_execution_intents",
    "leader_spot_v19_order_attempts",
    "leader_spot_v19_fills",
    "leader_spot_v19_events",
    "leader_spot_v19_execution_leases",
}


def _session():
    engine = create_engine("sqlite://")
    with engine.connect() as connection:
        connection.execute(text("PRAGMA foreign_keys = ON"))
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine)()


def _legacy_session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(
        engine,
        tables=[Tenant.__table__, TenantCredential.__table__],
    )
    return engine, sessionmaker(bind=engine)()


def test_v19_storage_migration_is_idempotent_and_isolated() -> None:
    engine, session = _legacy_session()

    assert migrations.migrate_leader_spot_v19_storage(session) is True
    assert migrations.migrate_leader_spot_v19_storage(session) is False

    tables = set(inspect(engine).get_table_names())
    assert V19_TABLES <= tables
    assert not {name for name in V19_TABLES if name.startswith("rule_strategy_")}

    marker = session.execute(
        text("SELECT version FROM schema_migrations WHERE version = :version"),
        {"version": migrations.LEADER_SPOT_V19_STORAGE_MIGRATION_VERSION},
    ).scalar_one()
    assert marker == migrations.LEADER_SPOT_V19_STORAGE_MIGRATION_VERSION


def test_v19_models_enforce_batch_scoped_tenant_attribution() -> None:
    _engine, session = _session()
    session.add_all(
        [
            Tenant(id="tenant-v19-a", name="V19 A"),
            Tenant(id="tenant-v19-b", name="V19 B"),
        ]
    )
    session.commit()
    session.add(
        LeaderSpotV19Strategy(
            strategy_id="leader-v19-a",
            tenant_id="tenant-v19-a",
            name="V19 A",
            status="stopped",
            environment="paper",
            config={"module_id": "leader_spot_v19_0", "schema_version": 19},
        )
    )
    session.commit()
    batch = LeaderSpotV19ExecutionBatch(
        batch_id="batch-v19-a",
        tenant_id="tenant-v19-a",
        strategy_id="leader-v19-a",
        strategy_name_snapshot="V19 A",
        execution_generation=1,
        status="running",
        config_snapshot={"module_id": "leader_spot_v19_0", "schema_version": 19},
    )
    session.add(batch)
    session.commit()

    position = LeaderSpotV19Position(
        position_id="position-v19-a",
        tenant_id="tenant-v19-a",
        strategy_id="leader-v19-a",
        batch_id="batch-v19-a",
        symbol="BTC-USDT",
        entry_price="100",
        entry_quantity="1",
        entry_time=batch.started_at,
        peak_price="100",
        moving_stop_price="92",
        loss_circuit_started_at=batch.started_at,
    )
    session.add(position)
    session.commit()

    session.add(
        LeaderSpotV19ExecutionIntent(
            tenant_id="tenant-v19-b",
            strategy_id="leader-v19-a",
            batch_id="batch-v19-a",
            position_id=position.position_id,
            execution_generation=1,
            idempotency_key="cross-tenant-intent",
            symbol="BTC-USDT",
            side="sell",
            order_type="market",
            leg_kind="stop_loss",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_v19_persistence_contract_contains_protection_outbox_and_lease_facts() -> None:
    assert {
        "protection_status",
        "peak_price",
        "moving_stop_price",
        "layered_exit_price",
        "loss_circuit_active",
        "trend_break_count",
    } <= {column.name for column in LeaderSpotV19Position.__table__.columns}
    assert {
        "idempotency_key",
        "lifecycle_state",
        "status",
        "attempt_count",
        "request_payload",
    } <= {column.name for column in LeaderSpotV19ExecutionIntent.__table__.columns}
    assert "uq_leader_spot_v19_intent_idempotency" in {
        constraint.name for constraint in LeaderSpotV19ExecutionIntent.__table__.constraints
    }


def test_app_runs_v19_storage_migration_before_scheduler_starts() -> None:
    source = (
        Path(__file__).parents[1] / "api" / "app.py"
    ).read_text(encoding="utf-8")

    migration_call = source.index("migrate_leader_spot_v19_storage(session)")
    scheduler_start = source.index("await _scheduler.start()")
    assert migration_call < scheduler_start
