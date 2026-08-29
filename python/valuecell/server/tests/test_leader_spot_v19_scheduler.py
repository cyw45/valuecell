from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from valuecell.server.api.schemas.leader_spot_v19 import LeaderSpotV19Config
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.leader_spot_v19 import (
    LeaderSpotV19ExecutionBatch,
    LeaderSpotV19ExecutionLease,
    LeaderSpotV19Strategy,
)
from valuecell.server.db.models.tenant import Tenant
from valuecell.server.services.leader_spot_v19_scheduler import LeaderSpotV19Scheduler


def _session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    tenant = Tenant(id="tenant-j", name="J")
    session.add(tenant)
    session.commit()
    strategy = LeaderSpotV19Strategy(
        strategy_id="leader-j", tenant_id=tenant.id, name="J", status="stopped",
        environment="paper", execution_generation=1,
        config=LeaderSpotV19Config().model_dump(mode="json"),
    )
    session.add(strategy)
    session.commit()
    return session, strategy


def test_v19_scheduler_start_stop_creates_isolated_batches_account_risk():
    session, strategy = _session()
    scheduler = LeaderSpotV19Scheduler()

    started = scheduler.start_strategy(
        session, strategy_id=strategy.strategy_id, tenant_id=strategy.tenant_id
    )
    assert started.status == "running"
    assert started.execution_generation == 2
    assert session.query(LeaderSpotV19ExecutionBatch).count() == 1
    with pytest.raises(ValueError, match="already_running"):
        scheduler.start_strategy(session, strategy_id=strategy.strategy_id, tenant_id=strategy.tenant_id)

    stopped = scheduler.stop_strategy(
        session, strategy_id=strategy.strategy_id, tenant_id=strategy.tenant_id
    )
    assert stopped.status == "stopped"
    assert stopped.stopped_at is not None
    strategy = session.get(LeaderSpotV19Strategy, strategy.strategy_id)
    assert strategy.current_batch_id is None
    assert strategy.execution_generation == 3


def test_v19_scheduler_lease_is_generation_scoped_and_fenced():
    session, strategy = _session()
    first = LeaderSpotV19Scheduler()
    second = LeaderSpotV19Scheduler()

    assert first._claim_lease(session, strategy.strategy_id, 1) is True
    assert second._claim_lease(session, strategy.strategy_id, 1) is False
    assert second._claim_lease(session, strategy.strategy_id, 2) is True
    assert session.query(LeaderSpotV19ExecutionLease).count() == 2


def test_v19_scheduler_uses_its_own_job_id_namespace():
    scheduler = LeaderSpotV19Scheduler()
    assert scheduler._job_id("leader-j") == "leader_spot_v19:leader-j"
    assert scheduler._job_id("leader-j") != "leader-j"


def test_v19_scheduler_tick_fails_closed_before_execution_pipeline_wiring(monkeypatch):
    session, strategy = _session()
    batch = LeaderSpotV19ExecutionBatch(
        batch_id="batch-j", tenant_id=strategy.tenant_id, strategy_id=strategy.strategy_id,
        strategy_name_snapshot=strategy.name, execution_generation=1, status="running",
        config_snapshot=strategy.config,
    )
    strategy.status = "running"
    strategy.current_batch_id = batch.batch_id
    session.add(batch)
    session.commit()

    scheduler = LeaderSpotV19Scheduler()

    class DatabaseManager:
        def get_session(self):
            return session

    monkeypatch.setattr(
        "valuecell.server.services.leader_spot_v19_scheduler.get_database_manager",
        lambda: DatabaseManager(),
    )
    result = __import__("asyncio").run(
        scheduler._tick(strategy.strategy_id, strategy.tenant_id, 1)
    )
    assert result.status == "blocked"
    assert result.reason_code == "v19_execution_pipeline_not_wired"


def test_v19_scheduler_starts_and_stops_separately_from_legacy_scheduler():
    source = (
        __import__("pathlib").Path(__file__).parents[1] / "api" / "app.py"
    ).read_text(encoding="utf-8")

    assert "LeaderSpotV19Scheduler" in source
    assert "_leader_spot_v19_scheduler.install_sync_job()" in source
    assert source.index("await _leader_spot_v19_scheduler.stop()") < source.index(
        "await _scheduler.stop()"
    )