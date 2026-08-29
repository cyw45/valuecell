from decimal import Decimal

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from valuecell.server.api.schemas.fixed_strategy import FixedStrategySignal
from valuecell.server.db.models.base import Base
from valuecell.server.db.models.rule_strategy import RuleStrategy
from valuecell.server.db.models.tenant import Tenant
from valuecell.server.services.fixed_strategy_paper_ledger import FixedPaperLedger


def _signal(symbol: str) -> FixedStrategySignal:
    return FixedStrategySignal.model_validate(
        {
            "kind": "dual_ma_trend",
            "symbol": symbol,
            "action": "long_entry",
            "reason_code": "test_entry",
            "reason": "测试入场",
            "observed_at": "2026-08-29T00:00:00Z",
        }
    )


def test_fixed_paper_accounts_remain_isolated_by_strategy_and_batch() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(Tenant(id="tenant-a", name="Tenant A"))
    session.add_all([
        RuleStrategy(strategy_id="strategy-a", tenant_id="tenant-a", name="A", config={}),
        RuleStrategy(strategy_id="strategy-b", tenant_id="tenant-a", name="B", config={}),
    ])
    session.commit()
    ledger = FixedPaperLedger(session)
    account_a = ledger.account(tenant_id="tenant-a", strategy_id="strategy-a", batch_id="batch-a", initial_capital_quote=Decimal("600"))
    account_b = ledger.account(tenant_id="tenant-a", strategy_id="strategy-b", batch_id="batch-b", initial_capital_quote=Decimal("600"))
    ledger.apply_signal(account=account_a, signal=_signal("BTC-USDT"), evaluation_id="eval-a", price=Decimal("100"), quantity=Decimal("2"))
    session.commit()
    assert account_a.quote_balance == 400
    assert account_b.quote_balance == 600
    assert account_a.account_id != account_b.account_id
