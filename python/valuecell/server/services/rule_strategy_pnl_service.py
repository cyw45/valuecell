"""Daily strategy equity and profit/loss series from persisted account facts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class DailyPnlObservation:
    """One verified equity observation used to close a UTC calendar day."""

    occurred_at: datetime
    equity_quote: float
    action: str


def build_daily_pnl_points(
    initial_capital_quote: float,
    strategy_created_at: datetime | None,
    observations: Iterable[DailyPnlObservation],
) -> list[dict[str, float | str]]:
    """Return the final persisted equity fact for each UTC calendar day."""
    baseline = _finite_amount(initial_capital_quote)
    if baseline is None:
        raise ValueError("initial_capital_quote must be finite")
    baseline_day = _utc_day(strategy_created_at or datetime.now(timezone.utc))
    closes: dict[date, DailyPnlObservation] = {}
    for observation in observations:
        equity = _finite_amount(observation.equity_quote)
        if equity is None:
            continue
        observation_day = _utc_day(observation.occurred_at)
        if observation_day < baseline_day:
            continue
        previous = closes.get(observation_day)
        if previous is None or observation.occurred_at >= previous.occurred_at:
            closes[observation_day] = DailyPnlObservation(
                occurred_at=observation.occurred_at,
                equity_quote=equity,
                action=observation.action,
            )

    days = sorted({baseline_day, *closes})
    points: list[dict[str, float | str]] = []
    previous_equity = baseline
    for day in days:
        observation = closes.get(day)
        equity = observation.equity_quote if observation is not None else baseline
        action = observation.action if observation is not None else "initial"
        points.append(
            {
                "ts": _utc_midnight(day),
                "cumulative_pnl": equity - baseline,
                "daily_pnl_quote": equity - previous_equity,
                "equity_quote": equity,
                "action": action,
            }
        )
        previous_equity = equity
    return points


def observation_from_journal(journal: Any) -> DailyPnlObservation | None:
    """Extract an eligible paper-account close from one evaluation journal."""
    result = journal.result or {}
    account = result.get("account")
    if (
        not isinstance(account, dict)
        or account.get("source") == "okx_demo"
        or not isinstance(journal.created_at, datetime)
    ):
        return None
    equity = _finite_amount(account.get("equity_quote"))
    if equity is None:
        return None
    return DailyPnlObservation(
        occurred_at=journal.created_at,
        equity_quote=equity,
        action=str(result.get("action") or "no_op"),
    )


def _finite_amount(value: object) -> float | None:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    return amount if isfinite(amount) else None

def _utc_day(value: datetime) -> date:
    aware = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    return aware.astimezone(timezone.utc).date()


def _utc_midnight(value: date) -> str:
    return datetime(value.year, value.month, value.day, tzinfo=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
