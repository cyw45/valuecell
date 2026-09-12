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


def test_journal_trade_facts_include_shared_demo_fill_and_execution_ids() -> None:
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
            "action": "long_entry",
            "symbol": "BTC-USDT",
            "reason": "SMA10 上穿 SMA20",
            "conditions": [],
        },
        trades=[],
    )
    facts = journal_trade_facts(
        strategy,
        journal,
        shared_orders=[
            {
                "order_id": "order-a",
                "intent_id": "intent-a",
                "reservation_id": "reservation-a",
                "strategy_id": "strategy-a",
                "batch_id": "batch-a",
                "symbol": "BTC-USDT",
                "side": "buy",
                "requested_quote": "100",
                "requested_quantity": "1",
                "status": "filled",
                "created_at": observed_at,
            }
        ],
        shared_fills=[
            {
                "fill_id": "fill-a",
                "order_id": "order-a",
                "strategy_id": "strategy-a",
                "batch_id": "batch-a",
                "symbol": "BTC-USDT",
                "side": "buy",
                "quantity": "1",
                "quote_amount": "101",
                "price": "101",
                "fee_quote": "0.1",
                "occurred_at": observed_at,
            }
        ],
    )

    assert len(facts) == 1
    assert facts[0].order_id == "order-a"
    assert facts[0].fill_id == "fill-a"
    assert facts[0].intent_id == "intent-a"
    assert facts[0].reservation_id == "reservation-a"
    assert facts[0].filled_quantity == 1
    assert facts[0].average_fill_price == 101
    assert facts[0].fee_quote == 0.1
    assert facts[0].status == "filled"


def test_journal_trade_facts_preserve_unknown_submission_status() -> None:
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
        created_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
        result={"action": "long_entry", "symbol": "BTC-USDT", "reason": "entry", "conditions": []},
        trades=[],
    )
    facts = journal_trade_facts(
        strategy,
        journal,
        shared_orders=[
            {
                "order_id": "order-a",
                "intent_id": "intent-a",
                "reservation_id": "reservation-a",
                "strategy_id": "strategy-a",
                "batch_id": "batch-a",
                "symbol": "BTC-USDT",
                "side": "buy",
                "requested_quote": "100",
                "requested_quantity": "1",
                "status": "submission_unknown",
                "created_at": journal.created_at,
            }
        ],
    )
    assert facts[0].status == "submission_unknown"


def test_journal_trade_facts_restore_fixed_paper_fill_from_execution_result() -> None:
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
            "action": "exit",
            "symbol": "BTC-USDT",
            "reason": "SMA10 上穿 SMA20",
            "conditions": [],
            "execution": {
                "execution": "paper_filled",
                "execution_ledger": "paper",
                "paper_fill": True,
                "fill_id": "fill-a",
                "filled_side": "sell",
                "filled_quantity": 2,
                "filled_price": 101,
            },
        },
        trades=[],
    )

    facts = journal_trade_facts(strategy, journal)

    assert len(facts) == 1
    assert facts[0].status == "filled"
    assert facts[0].side == "sell"
    assert facts[0].fill_id == "fill-a"
    assert facts[0].filled_quantity == 2
    assert facts[0].average_fill_price == 101
    assert facts[0].filled_quote == 202

def test_journal_trade_facts_keep_fixed_engine_comparison_numbers() -> None:
    """Fixed engines persist actual/threshold/operator instead of values."""
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
            "action": "long_entry",
            "symbol": "BTC-USDT",
            "reason": "SMA10 上穿 SMA20",
            "conditions": [
                {
                    "code": "ma_trend",
                    "label": "长期趋势",
                    "state": "triggered",
                    "actual": 101.5,
                    "threshold": 100.25,
                    "operator": ">",
                    "detail": "收盘价在短期均线之上",
                    "data_timestamp_ms": 1_700_000_000_000,
                }
            ],
        },
        trades=[],
    )

    facts = journal_trade_facts(
        strategy,
        journal,
        shared_orders=[
            {
                "order_id": "order-a",
                "intent_id": "intent-a",
                "reservation_id": "reservation-a",
                "strategy_id": "strategy-a",
                "batch_id": "batch-a",
                "symbol": "BTC-USDT",
                "side": "buy",
                "requested_quote": "100",
                "status": "filled",
                "created_at": observed_at,
            }
        ],
        shared_fills=[
            {
                "fill_id": "fill-a",
                "order_id": "order-a",
                "symbol": "BTC-USDT",
                "side": "buy",
                "quantity": "0.001",
                "quote_amount": "101.5",
                "occurred_at": observed_at,
            }
        ],
    )

    condition = facts[0].explanation.conditions[0]
    assert condition.actual == 101.5
    assert condition.threshold == 100.25
    assert condition.operator == ">"
    assert condition.data_at == datetime.fromtimestamp(
        1_700_000_000_000 / 1000, tz=timezone.utc
    )
