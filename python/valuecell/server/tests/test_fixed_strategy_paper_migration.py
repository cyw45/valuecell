from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from valuecell.server.db import migrations
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.fixed_strategy_paper import (  # noqa: F401
    FixedPaperAccount,
    FixedPaperFill,
    FixedPaperPosition,
)
from valuecell.server.db.models.rule_strategy import RuleStrategy  # noqa: F401
from valuecell.server.db.models.tenant import Tenant  # noqa: F401


def test_fixed_paper_ledger_migration_is_idempotent() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    assert migrations.migrate_fixed_strategy_paper_ledger(session) is True
    assert migrations.migrate_fixed_strategy_paper_ledger(session) is False
    inspector = inspect(engine)
    assert {"fixed_paper_accounts", "fixed_paper_positions", "fixed_paper_fills"} <= set(
        inspector.get_table_names()
    )
