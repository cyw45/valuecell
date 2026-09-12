"""Canonical comparison facts for persisted strategy conditions.

Code-owned fixed strategies persist ``actual`` / ``threshold`` / ``operator``,
while the configurable rule engine persists ``values.{left,right,comparator}``.
Every read model and both clients need one comparable representation, so this
module extracts the persisted numbers instead of letting each surface guess.
It never derives, invents, or re-evaluates a strategy condition; values that
were not recorded stay missing.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple


class ComparisonFacts(NamedTuple):
    """The persisted comparison that justifies one strategy condition."""

    actual: Any = None
    threshold: Any = None
    operator: str | None = None


_COMPARISON_KEYS = ("left", "right", "comparator")


def comparison_facts(condition: Mapping[str, Any]) -> ComparisonFacts:
    """Return comparison numbers from either persisted condition shape.

    Persisted conditions are untyped JSON, so the checks here are the contract
    boundary between storage and the typed read models.
    """
    values = condition.get("values")
    if isinstance(values, Mapping) and any(key in values for key in _COMPARISON_KEYS):
        operator = values.get("comparator")
        return ComparisonFacts(
            actual=values.get("left"),
            threshold=values.get("right"),
            operator=operator if isinstance(operator, str) else None,
        )
    operator = condition.get("operator")
    return ComparisonFacts(
        actual=condition.get("actual"),
        threshold=condition.get("threshold"),
        operator=operator if isinstance(operator, str) else None,
    )


def condition_data_timestamp_ms(condition: Mapping[str, Any]) -> int | None:
    """Return the recorded observation time of one condition, when present."""
    value = condition.get("data_timestamp_ms")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value <= 0:
        return None
    return int(value)


def with_comparison_values(condition: Mapping[str, Any]) -> dict[str, Any]:
    """Return the condition carrying comparison values in the canonical shape.

    Conditions that already persist ``values`` are returned unchanged; fixed
    strategy conditions only gain the comparable alias.
    """
    values = condition.get("values")
    if isinstance(values, Mapping) and values:
        return dict(condition)
    facts = comparison_facts(condition)
    if facts.actual is None and facts.threshold is None and facts.operator is None:
        return dict(condition)
    return {
        **condition,
        "values": {
            "left": facts.actual,
            "right": facts.threshold,
            "comparator": facts.operator,
        },
    }
