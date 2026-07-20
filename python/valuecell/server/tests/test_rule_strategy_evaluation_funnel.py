from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from valuecell.server.services.rule_strategy_service import RuleStrategyService


STAGES = [
    "strategy_run",
    "market_ready",
    "conditions",
    "risk",
    "order_submission",
    "fill",
]


class Repository:
    def __init__(self, result, trades=None):
        self.strategy = SimpleNamespace(strategy_id="strategy-a")
        self.journal = SimpleNamespace(
            evaluation_id="evaluation-a",
            created_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
            result=result,
            trades=trades or [],
        )

    def get(self, strategy_id, tenant_id):
        return self.strategy

    def get_evaluations(self, strategy_id, tenant_id, limit=100):
        return [self.journal]


def evaluation(result, trades=None):
    return RuleStrategyService(repository=Repository(result, trades)).evaluations(
        "strategy-a", "tenant-a", 1
    )[0]


def assert_fixed_funnel(item):
    assert [stage["code"] for stage in item["funnel"]] == STAGES
    assert all(set(stage) == {"code", "label", "status", "detail"} for stage in item["funnel"])


def test_historical_no_signal_journal_has_safe_funnel_and_does_not_claim_risk_passed():
    item = evaluation(
        {
            "action": "no_op",
            "reason_code": "indicator_conditions_not_met",
            "reason": "No entry signal.",
            "conditions": [
                {"code": "rsi", "category": "indicator", "state": "not_triggered", "detail": "no"},
                {"code": "macd", "category": "indicator", "state": "triggered", "detail": "yes"},
            ],
        }
    )

    assert_fixed_funnel(item)
    assert [stage["status"] for stage in item["funnel"]] == [
        "passed", "passed", "blocked", "pending", "pending", "pending"
    ]
    assert item["blocked_stage"] == "conditions"
    assert item["condition_summary"] == {
        "matched": 1,
        "total": 2,
        "required": 2,
        "available": 2,
    }


@pytest.mark.parametrize(
    ("diagnostic_stage", "expected_code"),
    [("market_data", "market_ready"), ("account_sync", "risk")],
)
def test_sync_diagnostic_journal_degrades_to_an_explicit_safe_blocker(
    diagnostic_stage, expected_code
):
    item = evaluation(
        {
            "stage": diagnostic_stage,
            "status": "blocked",
            "action": "no_op",
            "reason_code": "temporarily_unavailable",
            "reason": "同步暂不可用，已安全跳过。",
        }
    )

    assert_fixed_funnel(item)
    stage = next(stage for stage in item["funnel"] if stage["code"] == expected_code)
    assert stage["status"] == "blocked"
    assert item["blocked_stage"] == expected_code
    assert item["condition_summary"] == {
        "matched": 0,
        "total": 0,
        "required": 0,
        "available": 0,
    }
    assert item["funnel"][3]["status"] != "passed"


@pytest.mark.parametrize(
    ("order_status", "submit_status", "fill_status", "blocked_stage"),
    [
        ("pending", "pending", "pending", None),
        ("open", "passed", "pending", None),
        ("partially_filled", "passed", "partial", None),
        ("filled", "passed", "filled", None),
        ("closed", "passed", "filled", None),
        ("cancelled", "passed", "rejected", "fill"),
        ("rejected", "rejected", "rejected", "order_submission"),
    ],
)
def test_demo_order_status_maps_to_submission_and_fill_stages(
    order_status, submit_status, fill_status, blocked_stage
):
    item = evaluation(
        {
            "action": "buy",
            "reason_code": "indicator_buy_confirmed",
            "reason": "buy",
            "conditions": [
                {"code": "rsi", "category": "indicator", "state": "triggered", "detail": "yes"},
                {"code": "available_collateral", "category": "risk", "state": "not_triggered", "detail": "ok"},
            ],
            "entry_confirmation": {
                "enabled": 1, "available": 1, "passed": 1, "required": 1, "mode": "all"
            },
            "execution": {"execution": "okx_demo_submitted", "status": order_status},
            "execution_ledger": "external",
            "paper_fill": False,
        }
    )

    assert_fixed_funnel(item)
    assert item["funnel"][4]["status"] == submit_status
    assert item["funnel"][5]["status"] == fill_status
    assert item["blocked_stage"] == blocked_stage
    assert item["condition_summary"] == {
        "matched": 1, "total": 1, "required": 1, "available": 1
    }


def test_paper_trade_is_reported_as_submitted_and_filled():
    item = evaluation(
        {
            "action": "buy",
            "reason_code": "indicator_buy_confirmed",
            "reason": "buy",
            "conditions": [
                {"code": "rsi", "category": "indicator", "state": "triggered", "detail": "yes"},
                {"code": "max_positions", "category": "risk", "state": "not_triggered", "detail": "ok"},
            ],
        },
        trades=[{"execution": "paper_filled"}],
    )

    assert [stage["status"] for stage in item["funnel"]] == [
        "passed", "passed", "passed", "passed", "passed", "filled"
    ]
    assert item["blocked_stage"] is None


def test_risk_blocker_takes_priority_over_no_op_condition_mapping():
    item = evaluation({
        "action": "no_op", "reason_code": "max_positions", "reason": "blocked",
        "conditions": [
            {"code": "rsi", "category": "indicator", "state": "triggered", "detail": "yes"},
            {"code": "max_positions", "category": "risk", "state": "blocked", "detail": "limit"},
        ],
        "entry_confirmation": {"enabled": 1, "available": 1, "passed": 1, "required": 1, "mode": "all"},
    })
    assert item["blocked_stage"] == "risk"
    assert [stage["status"] for stage in item["funnel"][:4]] == ["passed", "passed", "passed", "blocked"]


def test_account_sync_does_not_claim_market_ready():
    item = evaluation({
        "stage": "account_sync", "status": "blocked", "action": "no_op",
        "reason_code": "demo_account_unavailable", "reason": "sync failed",
    })
    assert item["blocked_stage"] == "risk"
    assert item["funnel"][1]["status"] == "pending"


def test_sell_summary_uses_exit_conditions_and_any_mode_not_entry_confirmation():
    item = evaluation({
        "action": "sell", "reason_code": "advanced_exit_confirmed", "reason": "sell",
        "conditions": [
            {"code": "rsi_entry", "category": "indicator", "state": "not_triggered", "detail": "no"},
            {"code": "rsi_exit", "category": "exit", "state": "triggered", "detail": "yes"},
            {"code": "momentum_exit", "category": "exit", "state": "not_triggered", "detail": "no"},
        ],
        "entry_confirmation": {"enabled": 1, "available": 1, "passed": 0, "required": 1, "mode": "all"},
        "exit_confirmation_mode": "any",
    })
    assert item["condition_summary"] == {"matched": 1, "total": 2, "required": 1, "available": 2}
