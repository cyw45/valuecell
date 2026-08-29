from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from valuecell.server.api.schemas.fixed_strategy import FixedStrategySignal
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.fixed_strategy_paper import FixedPaperFill, FixedPaperPosition
from valuecell.server.db.models.rule_strategy import RuleStrategy
from valuecell.server.db.models.tenant import Tenant
from valuecell.server.services.fixed_strategy_paper_ledger import FixedPaperLedger


def _session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(Tenant(id="tenant-a", name="Tenant A"))
    session.add(RuleStrategy(strategy_id="strategy-a", tenant_id="tenant-a", name="Fixed", config={"initial_capital_quote": 1_000}))
    session.commit()
    return session


def _signal(action: str) -> FixedStrategySignal:
    return FixedStrategySignal(
        kind="dual_ma_trend",
        symbol="BTC-USDT",
        action=action,
        reason_code="test",
        reason="测试策略事实",
        observed_at=datetime(2026, 8, 29, tzinfo=timezone.utc),
    )


def test_fixed_paper_ledger_records_long_fill_and_exit_pnl() -> None:
    session = _session()
    ledger = FixedPaperLedger(session)
    account = ledger.account(
        tenant_id="tenant-a",
        strategy_id="strategy-a",
        batch_id="batch-a",
        initial_capital_quote=Decimal("1000"),
    )
    entry = ledger.apply_signal(
        account=account,
        signal=_signal("long_entry"),
        evaluation_id="evaluation-entry",
        price=Decimal("100"),
        quantity=Decimal("2"),
    )
    ledger.apply_signal(
        account=account,
        signal=_signal("exit"),
        evaluation_id="evaluation-exit",
        price=Decimal("110"),
        quantity=Decimal("2"),
    )
    session.commit()
    assert entry is not None
    assert session.query(FixedPaperPosition).one().status == "closed"
    assert session.query(FixedPaperFill).count() == 2
    assert account.realized_pnl_quote == 20


def test_fixed_paper_ledger_records_short_profit() -> None:
    session = _session()
    ledger = FixedPaperLedger(session)
    account = ledger.account(
        tenant_id="tenant-a",
        strategy_id="strategy-a",
        batch_id="batch-a",
        initial_capital_quote=Decimal("1000"),
    )
    ledger.apply_signal(
        account=account,
        signal=_signal("short_entry"),
        evaluation_id="evaluation-short-entry",
        price=Decimal("100"),
        quantity=Decimal("2"),
    )
    ledger.apply_signal(
        account=account,
        signal=_signal("exit"),
        evaluation_id="evaluation-short-exit",
        price=Decimal("90"),
        quantity=Decimal("2"),
    )
    session.commit()
    assert account.realized_pnl_quote == 20
