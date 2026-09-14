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


def test_fixed_entry_journal_reports_real_condition_numbers_instead_of_zero():
    item = evaluation({
        "action": "no_signal", "reason_code": "no_entry_signal",
        "reason": "No valid trend-aligned price cross occurred.",
        "conditions": [
            {"code": "trend.sma10_vs_sma20", "category": "indicator", "state": "triggered",
             "actual": 0.256090, "threshold": 0.257800, "operator": "<", "detail": "bearish"},
            {"code": "entry.price_cross_up", "category": "indicator", "state": "not_triggered",
             "actual": 100.0, "threshold": 101.0, "operator": ">", "detail": "no cross"},
            {"code": "entry.price_cross_down", "category": "indicator", "state": "not_triggered",
             "actual": 100.0, "threshold": 99.0, "operator": "<", "detail": "no cross"},
        ],
    })

    assert item["condition_summary"] == {
        "matched": 1, "total": 3, "required": 3, "available": 3
    }
    conditions_stage = next(stage for stage in item["funnel"] if stage["code"] == "conditions")
    assert conditions_stage["detail"] == "条件满足 1/3，需要 3 项。"
    assert item["blocked_stage"] == "conditions"


def test_fixed_hold_journal_explains_the_exit_rules_that_keep_the_position_open():
    item = evaluation({
        "action": "hold", "reason_code": "position_held",
        "reason": "Position remains open; no exit condition triggered.",
        "conditions": [
            {"code": "trend.sma10_vs_sma20", "category": "indicator", "state": "triggered", "detail": "bull"},
            {"code": "exit.stop_loss", "category": "exit", "state": "not_triggered",
             "actual": 100.0, "threshold": 95.0, "operator": "<=", "detail": "above stop"},
            {"code": "exit.timeout", "category": "exit", "state": "not_triggered",
             "actual": 12.0, "threshold": 168.0, "operator": ">=", "detail": "young"},
        ],
    })

    assert item["condition_summary"] == {
        "matched": 0, "total": 2, "required": 1, "available": 2
    }
    assert item["blocked_stage"] == "conditions"


def test_fixed_exit_journal_without_entry_confirmation_is_reported_as_submitted():
    item = evaluation({
        "action": "exit", "reason_code": "stop_loss",
        "reason": "Adverse 5% stop loss triggered.",
        "conditions": [
            {"code": "exit.stop_loss", "category": "exit", "state": "triggered",
             "actual": 94.0, "threshold": 95.0, "operator": "<=", "detail": "hit"},
            {"code": "exit.timeout", "category": "exit", "state": "not_triggered",
             "actual": 12.0, "threshold": 168.0, "operator": ">=", "detail": "young"},
        ],
        "execution": {"execution": "okx_demo_submitted", "status": "filled"},
    })

    assert item["condition_summary"] == {
        "matched": 1, "total": 2, "required": 1, "available": 2
    }
    assert item["funnel"][4]["status"] == "passed"
    assert item["funnel"][5]["status"] == "filled"
    assert item["blocked_stage"] is None


def test_legacy_fixed_journal_without_categories_still_counts_its_conditions():
    item = evaluation({
        "action": "no_signal", "reason_code": "no_entry_signal", "reason": "none",
        "conditions": [
            {"code": "trend.sma10_vs_sma20", "state": "not_triggered", "detail": "no"},
        ],
    })

    assert item["condition_summary"] == {
        "matched": 0, "total": 1, "required": 1, "available": 1
    }
    conditions_stage = next(stage for stage in item["funnel"] if stage["code"] == "conditions")
    assert conditions_stage["detail"] == "条件满足 0/1，需要 1 项。"

def test_fixed_blocked_action_is_reported_as_a_market_readiness_blocker():
    """A fixed engine that cannot read its market facts never ran its rules."""

    item = evaluation(
        {
            "action": "blocked",
            "reason_code": "quote_volume_unavailable",
            "reason": "One or more of the latest six 4h quote-volume facts is unavailable.",
            "conditions": [
                {
                    "code": "liquidity_quote_volume_available",
                    "category": "indicator",
                    "state": "unavailable",
                    "detail": "All six final 4h quote-volume values are required to calculate 24h liquidity.",
                }
            ],
        }
    )

    assert item["blocked_stage"] == "market_ready"
    market_stage = next(stage for stage in item["funnel"] if stage["code"] == "market_ready")
    assert market_stage["status"] == "blocked"
    assert market_stage["detail"].startswith("One or more of the latest six")
    conditions_stage = next(stage for stage in item["funnel"] if stage["code"] == "conditions")
    assert conditions_stage["status"] == "pending"
    assert item["condition_summary"] == {
        "matched": 0,
        "total": 1,
        "required": 1,
        "available": 0,
    }


def test_dispatch_blocked_entry_reports_the_recorded_execution_reason():
    """A refused dispatch keeps its own reason instead of a generic rejection."""

    item = evaluation(
        {
            "action": "long_entry",
            "reason_code": "bullish_price_cross",
            "reason": "Trend-aligned bullish price cross.",
            "conditions": [
                {
                    "code": "entry.price_cross_up",
                    "category": "indicator",
                    "state": "triggered",
                    "detail": "crossed",
                }
            ],
            "execution": {
                "execution": "blocked",
                "sandbox": True,
                "reason": "OKX Demo shared account is unavailable or stale",
            },
        }
    )

    assert item["blocked_stage"] == "order_submission"
    submission = next(stage for stage in item["funnel"] if stage["code"] == "order_submission")
    assert submission["status"] == "blocked"
    assert submission["detail"] == "OKX Demo shared account is unavailable or stale"
    fill = next(stage for stage in item["funnel"] if stage["code"] == "fill")
    assert fill["status"] == "pending"


def test_dust_ignored_exit_is_reported_as_a_skipped_no_op():
    """Dust stays out of the trade list, so the funnel must explain it here."""

    item = evaluation(
        {
            "action": "exit",
            "reason_code": "stop_loss",
            "reason": "Adverse 5% stop loss triggered.",
            "conditions": [
                {
                    "code": "exit.stop_loss",
                    "category": "exit",
                    "state": "triggered",
                    "detail": "hit",
                }
            ],
            "execution": {
                "execution": "ignored_dust",
                "sandbox": True,
                "status": "ignored_dust",
                "reason": "available balance is below the exchange minimum",
            },
        }
    )

    assert item["blocked_stage"] == "order_submission"
    submission = next(stage for stage in item["funnel"] if stage["code"] == "order_submission")
    assert submission["status"] == "blocked"
    assert "粉尘" in submission["detail"]
    assert [stage["status"] for stage in item["funnel"][:4]] == [
        "passed",
        "passed",
        "passed",
        "passed",
    ]
