"""Condition comparison facts must survive both persisted journal shapes."""

from valuecell.server.services.strategy_condition_facts import (
    comparison_facts,
    condition_data_timestamp_ms,
    with_comparison_values,
)

FIXED_CONDITION = {
    "code": "ma_trend",
    "label": "长期趋势",
    "state": "triggered",
    "actual": 101.5,
    "threshold": 100.25,
    "operator": ">",
    "detail": "收盘价在短期均线之上",
    "data_timestamp_ms": 1_700_000_000_000,
}

CONFIGURABLE_CONDITION = {
    "code": "program.entry.1",
    "label": "15m收盘价 > 15mMA20",
    "state": "triggered",
    "detail": "条件满足",
    "values": {"left": 84.125, "right": 80.0, "comparator": "gt"},
}


def test_fixed_strategy_conditions_expose_comparison_numbers() -> None:
    facts = comparison_facts(FIXED_CONDITION)
    assert facts.actual == 101.5
    assert facts.threshold == 100.25
    assert facts.operator == ">"


def test_configurable_conditions_keep_existing_comparison_numbers() -> None:
    facts = comparison_facts(CONFIGURABLE_CONDITION)
    assert facts.actual == 84.125
    assert facts.threshold == 80.0
    assert facts.operator == "gt"


def test_condition_data_timestamp_is_read_only_when_recorded() -> None:
    assert condition_data_timestamp_ms(FIXED_CONDITION) == 1_700_000_000_000
    assert condition_data_timestamp_ms(CONFIGURABLE_CONDITION) is None
    assert condition_data_timestamp_ms({**FIXED_CONDITION, "data_timestamp_ms": 0}) is None
    assert condition_data_timestamp_ms({**FIXED_CONDITION, "data_timestamp_ms": True}) is None


def test_fixed_conditions_gain_the_canonical_comparison_values() -> None:
    normalized = with_comparison_values(FIXED_CONDITION)
    assert normalized["values"] == {
        "left": 101.5,
        "right": 100.25,
        "comparator": ">",
    }
    assert normalized["actual"] == 101.5


def test_existing_comparison_values_are_never_rewritten() -> None:
    normalized = with_comparison_values(CONFIGURABLE_CONDITION)
    assert normalized["values"] == {
        "left": 84.125,
        "right": 80.0,
        "comparator": "gt",
    }


def test_conditions_without_comparison_numbers_stay_unchanged() -> None:
    raw = {"code": "risk.capacity", "label": "仓位容量", "state": "blocked", "detail": "无额度"}
    assert with_comparison_values(raw) == raw
