from datetime import datetime, timezone
from types import SimpleNamespace

from valuecell.server.services.multi_strategy_trade_facts import journal_trade_facts


def test_journal_trade_facts_preserve_strategy_identity_and_conditions() -> None:
    observed_at = datetime(2026, 8, 28, tzinfo=timezone.utc)
    strategy = SimpleNamespace(
        strategy_id="strategy-a",
        tenant_id="tenant-a",
        strategy_kind="dual_ma_trend",
        strategy_version="v1",
        code_fingerprint="fingerprint-a",
    )
    journal = SimpleNamespace(
        evaluation_id="evaluation-a",
        batch_id="batch-a",
        created_at=observed_at,
        result={
            "action": "buy",
            "reason": "SMA10 上穿 SMA20",
            "conditions": [
                {
                    "code": "ma_cross",
                    "label": "均线金叉",
                    "state": "triggered",
                    "detail": "价格上穿短期均线",
                    "values": {"left": 101, "right": 100, "comparator": "gt"},
                }
            ],
        },
        trades=[
            {
                "action": "buy",
                "symbol": "BTC-USDT",
                "price": 101,
                "quantity": 1,
                "quote_amount": 101,
                "execution": "paper_filled",
            }
        ],
    )
    facts = journal_trade_facts(strategy, journal)
    assert len(facts) == 1
    assert facts[0].identity.strategy_id == "strategy-a"
    assert facts[0].identity.kind == "dual_ma_trend"
    assert facts[0].batch_id == "batch-a"
    assert facts[0].explanation.conditions[0].actual == 101
    assert facts[0].status == "filled"
