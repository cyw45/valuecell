from datetime import datetime, timezone
from types import SimpleNamespace

from valuecell.server.services.rule_strategy_demo_snapshot_service import (
    build_demo_daily_curve,
)


def test_demo_wallet_curve_uses_final_real_snapshot_per_utc_day():
    snapshots = [
        SimpleNamespace(
            observed_at=datetime(2026, 8, 1, 9, tzinfo=timezone.utc),
            total_usdt_value=1_000.0,
        ),
        SimpleNamespace(
            observed_at=datetime(2026, 8, 2, 9, tzinfo=timezone.utc),
            total_usdt_value=1_010.0,
        ),
        SimpleNamespace(
            observed_at=datetime(2026, 8, 2, 20, tzinfo=timezone.utc),
            total_usdt_value=1_025.0,
        ),
        SimpleNamespace(
            observed_at=datetime(2026, 8, 4, 9, tzinfo=timezone.utc),
            total_usdt_value=1_015.0,
        ),
    ]

    assert build_demo_daily_curve(snapshots) == [
        {
            "ts": "2026-08-01T00:00:00Z",
            "cumulative_pnl": 0.0,
            "daily_pnl_quote": 0.0,
            "equity_quote": 1_000.0,
            "action": "wallet_snapshot",
        },
        {
            "ts": "2026-08-02T00:00:00Z",
            "cumulative_pnl": 25.0,
            "daily_pnl_quote": 25.0,
            "equity_quote": 1_025.0,
            "action": "wallet_snapshot",
        },
        {
            "ts": "2026-08-04T00:00:00Z",
            "cumulative_pnl": 15.0,
            "daily_pnl_quote": -10.0,
            "equity_quote": 1_015.0,
            "action": "wallet_snapshot",
        },
    ]
