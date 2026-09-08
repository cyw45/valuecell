from datetime import datetime, timezone
from types import SimpleNamespace

from valuecell.server.services.rule_strategy_pnl_service import (
    DailyPnlObservation,
    build_daily_pnl_points,
    observation_from_journal,
)


def test_daily_pnl_uses_final_fact_per_utc_day_without_gap_filling():
    points = build_daily_pnl_points(
        1_000.0,
        datetime(2026, 8, 1, 12, tzinfo=timezone.utc),
        [
            DailyPnlObservation(
                occurred_at=datetime(2026, 8, 2, 9, tzinfo=timezone.utc),
                equity_quote=1_010.0,
                action="buy",
            ),
            DailyPnlObservation(
                occurred_at=datetime(2026, 8, 2, 20, tzinfo=timezone.utc),
                equity_quote=1_025.0,
                action="sell",
            ),
            DailyPnlObservation(
                occurred_at=datetime(2026, 8, 4, 10, tzinfo=timezone.utc),
                equity_quote=1_015.0,
                action="close",
            ),
        ],
    )

    assert points == [
        {
            "ts": "2026-08-01T00:00:00Z",
            "cumulative_pnl": 0.0,
            "daily_pnl_quote": 0.0,
            "equity_quote": 1_000.0,
            "action": "initial",
        },
        {
            "ts": "2026-08-02T00:00:00Z",
            "cumulative_pnl": 25.0,
            "daily_pnl_quote": 25.0,
            "equity_quote": 1_025.0,
            "action": "sell",
        },
        {
            "ts": "2026-08-04T00:00:00Z",
            "cumulative_pnl": 15.0,
            "daily_pnl_quote": -10.0,
            "equity_quote": 1_015.0,
            "action": "close",
        },
    ]


def test_fixed_paper_execution_account_is_eligible_for_pnl_curve():
    journal = SimpleNamespace(
        created_at=datetime(2026, 9, 8, 4, tzinfo=timezone.utc),
        result={
            "action": "long_entry",
            "execution": {
                "execution_ledger": "paper",
                "paper_fill": True,
                "account": {
                    "source": "fixed_paper_ledger",
                    "equity_quote": 1_024,
                },
            },
        },
    )
    observation = observation_from_journal(journal)
    assert observation is not None
    assert observation.equity_quote == 1_024
