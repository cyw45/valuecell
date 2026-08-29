from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from valuecell.server.db import migrations
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.multi_strategy import (  # noqa: F401
    StrategyCapitalReservation,
    StrategySharedAccount,
)
from valuecell.server.db.models.rule_strategy import (  # noqa: F401
    RuleStrategy,
    RuleStrategyExecutionBatch,
    RuleStrategyExecutionIntent,
)
from valuecell.server.db.models.tenant import SaaSUser, Tenant  # noqa: F401
from valuecell.server.db.models.tenant_credential import TenantCredential  # noqa: F401


def test_multi_strategy_migration_is_idempotent_and_adds_allocator_schema() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    assert migrations.migrate_multi_strategy_account(session) is True
    assert migrations.migrate_multi_strategy_account(session) is False

    inspector = inspect(engine)
    assert {"strategy_shared_accounts", "strategy_capital_reservations"} <= set(
        inspector.get_table_names()
    )
    strategy_columns = {
        column["name"] for column in inspector.get_columns("rule_strategies")
    }
    assert {"strategy_kind", "strategy_version", "code_fingerprint"} <= strategy_columns
    batch_columns = {
        column["name"]
        for column in inspector.get_columns("rule_strategy_execution_batches")
    }
    assert {"strategy_kind", "strategy_version", "code_fingerprint"} <= batch_columns
    intent_columns = {
        column["name"]
        for column in inspector.get_columns("rule_strategy_execution_intents")
    }
    assert "reservation_id" in intent_columns
