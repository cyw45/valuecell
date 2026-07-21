from datetime import datetime, timezone

from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from valuecell.server.db.migrations import ensure_single_running_rule_strategy_index
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.rule_strategy import (
    RuleStrategy,
    RuleStrategyEvaluationJournal,
)
from valuecell.server.db.repositories.rule_strategy_repository import RuleStrategyRepository


def _session():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(
        engine,
        tables=[
            RuleStrategy.__table__,
            RuleStrategyEvaluationJournal.__table__,
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


def test_repository_starts_only_one_strategy_per_tenant():
    session = _session()
    repository = RuleStrategyRepository(db_session=session)
    repository.create(_strategy("first"))
    repository.create(_strategy("second"))

    started, conflict = repository.start_exclusive("first", "tenant-a")
    rejected, second_conflict = repository.start_exclusive("second", "tenant-a")

    assert started is not None and started.status == "running"
    assert conflict is False
    assert rejected is not None and rejected.status == "stopped"
    assert second_conflict is True


def test_unique_index_rejects_a_second_running_strategy_per_tenant():
    session = _session()
    repository = RuleStrategyRepository(db_session=session)
    repository.create(_strategy("first", status="running"))
    repository.create(_strategy("second"))
    ensure_single_running_rule_strategy_index(session)

    second = repository.get("second", "tenant-a")
    assert second is not None
    second.status = "running"
    session.add(second)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
    else:
        raise AssertionError("second running strategy must violate the tenant index")


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
