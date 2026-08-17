from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

from valuecell.server.db.models.base import Base
from valuecell.server.db.models.rule_strategy import (
    RuleStrategy,
    RuleStrategyEvaluationJournal,
    RuleStrategyExecutionLease,
    RuleStrategyMonitorSymbol,
)
from valuecell.server.db.models.tenant import Tenant
from valuecell.server.db.repositories.rule_strategy_repository import RuleStrategyRepository


def _session():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(
        engine,
        tables=[
            Tenant.__table__,
            RuleStrategy.__table__,
            RuleStrategyEvaluationJournal.__table__,
            RuleStrategyExecutionLease.__table__,
            RuleStrategyMonitorSymbol.__table__,
        ],
    )
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE rule_strategy_execution_intents ("
                "id VARCHAR(36) PRIMARY KEY, strategy_id VARCHAR(100) NOT NULL, "
                "tenant_id VARCHAR(36) NOT NULL)"
            )
        )
    return sessionmaker(bind=engine)()


def _strategy(strategy_id: str, status: str = "stopped") -> RuleStrategy:
    now = datetime.now(timezone.utc)
    return RuleStrategy(
        strategy_id=strategy_id,
        tenant_id="tenant-a",
        name=strategy_id,
        status=status,
        paper_mode=True,
        execution_generation=1,
        config={"mode": "paper"},
        created_at=now,
        updated_at=now,
    )


def test_repository_allows_independent_running_strategies_per_tenant():
    session = _session()
    repository = RuleStrategyRepository(db_session=session)
    repository.create(_strategy("first", status="running"))
    repository.create(_strategy("second", status="running"))

    running = repository.list_running()
    assert {item.strategy_id for item in running} == {"first", "second"}


def test_repository_generation_lease_allows_only_one_worker():
    session = _session()
    repository = RuleStrategyRepository(db_session=session)
    repository.create(_strategy("lease", status="running"))
    assert repository.claim_execution_lease("lease", 1, "worker-a") is True
    assert repository.claim_execution_lease("lease", 1, "worker-b") is False


def test_repository_monitor_lease_keeps_clock_and_duration_injectable():
    session = _session()
    repository = RuleStrategyRepository(db_session=session)
    session.add(Tenant(id="tenant-a", name="Tenant A"))
    session.commit()
    repository.create(_strategy("monitor", status="running"))
    now = datetime(2026, 8, 6, tzinfo=timezone.utc)
    session.add(
        RuleStrategyMonitorSymbol(
            strategy_id="monitor",
            tenant_id="tenant-a",
            symbol="BTC-USDT",
            state="candidate",
            next_check_at=now - timedelta(seconds=1),
        )
    )
    session.commit()

    rows = repository.claim_monitor_lease(
        "monitor",
        "tenant-a",
        "worker-a",
        now=now,
        lease_seconds=90,
    )

    assert [row.symbol for row in rows] == ["BTC-USDT"]
    assert rows[0].lease_owner == "worker-a"
    lease_until = rows[0].lease_until.replace(tzinfo=timezone.utc)
    assert lease_until == now + timedelta(seconds=90)


def test_repository_force_monitor_review_keeps_active_lease_protection():
    session = _session()
    repository = RuleStrategyRepository(db_session=session)
    session.add(Tenant(id="tenant-a", name="Tenant A"))
    session.commit()
    repository.create(_strategy("forced-monitor", status="running"))
    now = datetime(2026, 8, 6, tzinfo=timezone.utc)
    session.add(
        RuleStrategyMonitorSymbol(
            strategy_id="forced-monitor",
            tenant_id="tenant-a",
            symbol="BTC-USDT",
            state="admitted",
            next_check_at=now + timedelta(days=1),
            lease_owner="worker-a",
            lease_until=now + timedelta(seconds=60),
        )
    )
    session.commit()

    blocked = repository.claim_monitor_lease(
        "forced-monitor", "tenant-a", "worker-b", now=now, force=True
    )
    claimed = repository.claim_monitor_lease(
        "forced-monitor",
        "tenant-a",
        "worker-b",
        now=now + timedelta(seconds=61),
        force=True,
    )

    assert blocked == []
    assert [row.symbol for row in claimed] == ["BTC-USDT"]
    assert claimed[0].lease_owner == "worker-b"


def test_repository_rejects_stale_monitor_worker_writeback():
    session = _session()
    repository = RuleStrategyRepository(db_session=session)
    session.add(Tenant(id="tenant-a", name="Tenant A"))
    session.commit()
    repository.create(_strategy("fenced-monitor", status="running"))
    now = datetime(2026, 8, 6, tzinfo=timezone.utc)
    session.add(
        RuleStrategyMonitorSymbol(
            strategy_id="fenced-monitor",
            tenant_id="tenant-a",
            symbol="BTC-USDT",
            state="candidate",
            next_check_at=now - timedelta(seconds=1),
        )
    )
    session.commit()
    first = repository.claim_monitor_lease(
        "fenced-monitor", "tenant-a", "worker-a", now=now, lease_seconds=30
    )[0]
    repository.claim_monitor_lease(
        "fenced-monitor",
        "tenant-a",
        "worker-b",
        now=now + timedelta(seconds=31),
        lease_seconds=60,
    )

    stale = repository.update_monitor_state(
        first.id,
        "tenant-a",
        lease_owner="worker-a",
        state="admitted",
        reason_code="stale",
        reason_detail="stale worker",
        evaluated_at=now,
        next_check_at=now + timedelta(days=1),
        protected_held=False,
        consecutive_low_volume_days=0,
        metadata_provider="test",
        listing_first_tradable_at=now - timedelta(days=365),
        listing_age_days=365,
        average_quote_volume_30d=10_000_000.0,
        price_quote=100.0,
        price_observed_at=now,
    )

    current = session.get(RuleStrategyMonitorSymbol, first.id)
    session.refresh(current)
    assert stale is None
    assert current.state == "candidate"
    assert current.lease_owner == "worker-b"


def test_repository_persists_fresh_monitor_provider_evidence():
    session = _session()
    repository = RuleStrategyRepository(db_session=session)
    session.add(Tenant(id="tenant-a", name="Tenant A"))
    session.commit()
    repository.create(_strategy("monitor-evidence", status="running"))
    now = datetime(2026, 8, 6, tzinfo=timezone.utc)
    session.add(
        RuleStrategyMonitorSymbol(
            strategy_id="monitor-evidence",
            tenant_id="tenant-a",
            symbol="BTC-USDT",
            state="candidate",
        )
    )
    session.commit()
    claimed = repository.claim_monitor_lease(
        "monitor-evidence", "tenant-a", "worker-a", now=now, force=True
    )[0]

    saved = repository.update_monitor_state(
        claimed.id,
        "tenant-a",
        lease_owner="worker-a",
        state="admitted",
        reason_code="monitor_observation_enabled",
        reason_detail="完整交易所事实已记录。",
        evaluated_at=now,
        next_check_at=now + timedelta(days=1),
        protected_held=False,
        consecutive_low_volume_days=0,
        metadata_provider="okx",
        listing_first_tradable_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        listing_age_days=2_410,
        average_quote_volume_30d=12_500_000.0,
        price_quote=98_000.0,
        price_observed_at=now,
    )

    assert saved is not None
    assert saved.metadata_provider == "okx"
    assert saved.listing_age_days == 2_410
    assert saved.average_quote_volume_30d == 12_500_000.0
    assert saved.price_quote == 98_000.0
    assert saved.lease_owner is None


def test_repository_delete_cascades_journals_for_stopped_strategy():
    session = _session()
    repository = RuleStrategyRepository(db_session=session)
    repository.create(_strategy("deletable"))
    repository.append_evaluation(
        RuleStrategyEvaluationJournal(
            evaluation_id="evaluation-1",
            strategy_id="deletable",
            tenant_id="tenant-a",
            result={},
            signals=[],
            trades=[],
            funding=[],
        )
    )

    assert repository.delete_if_allowed("deletable", "tenant-a") == "deleted"
    assert session.query(RuleStrategyEvaluationJournal).count() == 0


def test_repository_rejects_running_delete():
    session = _session()
    repository = RuleStrategyRepository(db_session=session)
    repository.create(_strategy("running", status="running"))

    assert repository.delete_if_allowed("running", "tenant-a") == "running"
    assert repository.get("running", "tenant-a") is not None


def test_repository_archives_stopped_strategy_with_execution_audit():
    session = _session()
    repository = RuleStrategyRepository(db_session=session)
    repository.create(_strategy("audited"))
    session.execute(
        text(
            "INSERT INTO rule_strategy_execution_intents (id, strategy_id, tenant_id) "
            "VALUES ('intent-1', 'audited', 'tenant-a')"
        )
    )
    session.commit()

    assert repository.delete_if_allowed("audited", "tenant-a") == "archived"
    archived = repository.get("audited", "tenant-a")
    assert archived is not None
    assert str(archived.status) == "archived"
    assert repository.list("tenant-a") == []
