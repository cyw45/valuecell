from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from valuecell.server.db.models.base import Base
from valuecell.server.db.models.fixed_strategy_paper import FixedPaperAccount, FixedPaperPosition
from valuecell.server.db.models.rule_strategy import RuleStrategy, RuleStrategyExecutionBatch
from valuecell.server.db.models.tenant import Tenant
from valuecell.server.db.repositories.rule_strategy_repository import RuleStrategyRepository


def test_fixed_paper_account_reads_persisted_marked_equity() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(Tenant(id="tenant-a", name="Tenant A"))
    session.add(RuleStrategy(strategy_id="strategy-a", tenant_id="tenant-a", name="A", config={}))
    session.add(RuleStrategyExecutionBatch(
        batch_id="batch-a",
        tenant_id="tenant-a",
        strategy_id="strategy-a",
        strategy_name_snapshot="A",
        execution_generation=1,
        status="paused",
        config_snapshot={"initial_capital_quote": 1000},
    ))
    session.flush()
    account = FixedPaperAccount(
        tenant_id="tenant-a",
        strategy_id="strategy-a",
        batch_id="batch-a",
        initial_capital_quote=1000,
        quote_balance=798,
        reserved_quote=12,
        occupied_quote=202,
        realized_pnl_quote=0,
        unrealized_pnl_quote=20,
    )
    session.add(account)
    session.flush()
    session.add(FixedPaperPosition(
        account_id=account.account_id,
        tenant_id="tenant-a",
        strategy_id="strategy-a",
        batch_id="batch-a",
        symbol="BTC-USDT",
        side="long",
        quantity=2,
        entry_price=101,
        entry_quote=202,
        entry_timestamp_ms=1,
        status="open",
    ))
    session.commit()

    data = RuleStrategyRepository(db_session=session).get_fixed_paper_account(
        "strategy-a", "tenant-a", batch_id="batch-a"
    )

    assert data is not None
    assert data["reserved_quote"] == 12
    assert data["occupied_quote"] == 202
    assert data["unrealized_pnl_quote"] == 20
    assert data["equity_quote"] == 1020
    assert data["batch_status"] == "paused"
