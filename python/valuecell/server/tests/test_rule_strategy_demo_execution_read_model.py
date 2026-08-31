from datetime import datetime, timezone
from decimal import Decimal

import pytest

from valuecell.server.services.rule_strategy_demo_execution_read_model import (
    DemoExecutionReadModelError,
    build_demo_execution_read_model,
    get_demo_execution_read_model,
    strategy_inventory_by_symbol,
)


def _demo_strategy():
    return {
        "id": "strategy-a",
        "config": {
            "execution": {
                "environment": "okx_demo",
                "sandbox_connection_id": "conn-a",
            }
        },
    }


def test_demo_read_model_excludes_manual_and_other_strategy_orders():
    result = build_demo_execution_read_model(
        _demo_strategy(),
        {"source": "okx_demo", "balances": [{"currency": "USDT"}]},
        {"source": "okx_demo", "positions": []},
        [
            {"id": "ours", "strategy_id": "strategy-a", "execution_source": "rule_strategy"},
            {"id": "manual", "strategy_id": None, "execution_source": "manual"},
            {"id": "other", "strategy_id": "strategy-b", "execution_source": "rule_strategy"},
        ],
    )

    assert result["source"] == "okx_demo_spot"
    assert result["strategy_positions"] == []
    assert result["orders"] == [
        {"id": "ours", "strategy_id": "strategy-a", "execution_source": "rule_strategy"}
    ]
    assert result["account"]["scope"] == "exchange_connection_shared_account"
    assert result["trade_summary"]["status"] == "not_bought"
    assert result["pnl"]["status"] == "unavailable"
    assert result["pnl"]["value"] is None
    assert result["pnl"]["reason_code"] == "no_filled_orders"
    assert result["equity_curve"]["reason_code"] == "no_filled_orders"


def test_demo_read_model_calculates_attributed_moving_average_pnl():
    result = build_demo_execution_read_model(
        _demo_strategy(),
        {"source": "okx_demo", "balances": [{"currency": "BTC", "total": 9}]},
        {"source": "okx_demo", "checked_at": "2024-01-04T00:00:00+00:00", "positions": [
            {"symbol": "BTC/USDT", "quantity": 9, "mark_price": 130},
        ]},
        [
            {"id": "b1", "strategy_id": "strategy-a", "execution_source": "rule_strategy", "symbol": "BTC/USDT", "side": "buy", "status": "filled", "filled_quantity": "2", "filled_quote": "200", "average_fill_price": "100", "filled_at": "2024-01-01T00:00:00+00:00"},
            {"id": "b2", "strategy_id": "strategy-a", "execution_source": "rule_strategy", "symbol": "BTC/USDT", "side": "buy", "status": "filled", "filled_quantity": "1", "filled_quote": "120", "average_fill_price": "120", "filled_at": "2024-01-02T00:00:00+00:00"},
            {"id": "s1", "strategy_id": "strategy-a", "execution_source": "rule_strategy", "symbol": "BTC/USDT", "side": "sell", "status": "filled", "filled_quantity": "1", "filled_quote": "140", "average_fill_price": "140", "filled_at": "2024-01-03T00:00:00+00:00"},
            {"id": "failed", "strategy_id": "strategy-a", "execution_source": "rule_strategy", "status": "failed"},
            {"id": "unknown", "strategy_id": "strategy-a", "execution_source": "rule_strategy", "status": "submission_unknown"},
        ],
    )

    assert result["trade_summary"]["status"] == "bought_and_sold"
    assert result["trade_summary"]["purchase_state"] == "bought"
    assert result["trade_summary"]["order_count"] == 5
    assert result["trade_summary"]["filled_order_count"] == 3
    assert result["trade_summary"]["partially_filled_order_count"] == 0
    assert result["trade_summary"]["failed_order_count"] == 1
    assert result["trade_summary"]["unknown_order_count"] == 1
    assert result["trade_summary"]["filled_buy_orders"] == 2
    assert result["trade_summary"]["filled_sell_orders"] == 1
    assert result["trade_summary"]["filled_buy_quantity"] == "3"
    assert result["trade_summary"]["filled_sell_quantity"] == "1"
    assert result["trade_summary"]["current_position_quantity"] == "2"
    assert result["trade_summary"]["failed_orders"] == 1
    assert result["trade_summary"]["submission_unknown_orders"] == 1
    assert result["trade_summary"]["latest_status"] == "submission_unknown"
    assert result["trade_summary"]["latest_order"]["id"] == "unknown"
    assert result["strategy_positions"] == [
        {
            "symbol": "BTC/USDT",
            "quantity": "2",
            "entry_price": "106.6666666666666666666666666",
            "mark_price": "130",
            "notional_usdt": "260",
            "unrealized_pnl_usdt": "46.6666666666666666666666667",
        }
    ]
    assert result["pnl"]["status"] == "available"
    assert result["pnl"]["position_quantity"] == "2"
    assert result["pnl"]["realized"] == "33.3333333333333333333333333"
    assert result["pnl"]["unrealized"] == "46.6666666666666666666666667"
    assert result["pnl"]["total"] == "80.0000000000000000000000000"
    assert result["equity_curve"]["status"] == "available"
    assert result["equity_curve"]["reason_code"] is None
    assert result["equity_curve"]["points"][-1] == {
        "ts": "2024-01-04T00:00:00+00:00",
        "cumulative_pnl": "80.0000000000000000000000000",
    }


def test_demo_read_model_replays_shared_append_only_fills() -> None:
    result = build_demo_execution_read_model(
        _demo_strategy(),
        {"source": "okx_demo", "balances": []},
        {
            "source": "okx_demo",
            "checked_at": "2026-08-29T00:00:00+00:00",
            "positions": [{"symbol": "BTC/USDT", "mark_price": 120}],
        },
        [],
        shared_venue_orders=[
            {
                "order_id": "shared-order-a",
                "intent_id": "intent-a",
                "reservation_id": "reservation-a",
                "strategy_id": "strategy-a",
                "batch_id": "batch-a",
                "symbol": "BTC/USDT",
                "side": "buy",
                "requested_quote": "100",
                "requested_quantity": "1",
                "created_at": "2026-08-29T00:00:00+00:00",
            }
        ],
        shared_order_projections=[
            {
                "order_id": "shared-order-a",
                "status": "filled",
                "filled_quantity": "1",
                "remaining_quantity": "0",
                "filled_quote": "100",
                "fee_quote": "0",
            }
        ],
        shared_fills=[
            {
                "order_id": "shared-order-a",
                "strategy_id": "strategy-a",
                "batch_id": "batch-a",
                "quantity": "1",
                "quote_amount": "100",
                "fee_quote": "0",
                "occurred_at": "2026-08-29T00:00:00+00:00",
            }
        ],
    )

    assert result["orders"][0]["execution_source"] == "shared_demo"
    assert result["strategy_positions"][0]["quantity"] == "1"
    assert result["pnl"]["status"] == "available"
    assert result["pnl"]["unrealized"] == "20"


def test_strategy_inventory_uses_only_confirmed_fills_and_tracks_average_cost():
    inventory = strategy_inventory_by_symbol([
        {"symbol": "BTC/USDT", "side": "buy", "status": "filled", "filled_quantity": "2", "filled_quote": "200"},
        {"symbol": "BTC/USDT", "side": "buy", "status": "filled", "filled_quantity": "1", "filled_quote": "120"},
        {"symbol": "BTC/USDT", "side": "sell", "status": "filled", "filled_quantity": "1", "filled_quote": "140"},
        {"symbol": "BTC/USDT", "side": "buy", "status": "submission_unknown", "filled_quantity": "99", "filled_quote": "1"},
    ])

    assert inventory["BTC/USDT"] == (
        Decimal("2"),
        Decimal("213.3333333333333333333333333"),
    )


def test_strategy_inventory_ignores_fills_before_official_test_baseline():
    inventory = strategy_inventory_by_symbol(
        [
            {
                "symbol": "BTC/USDT",
                "side": "buy",
                "status": "filled",
                "filled_quantity": "2",
                "filled_quote": "200",
                "filled_at": "2026-08-17T08:00:00+00:00",
            },
            {
                "symbol": "ETH/USDT",
                "side": "buy",
                "status": "filled",
                "filled_quantity": "1",
                "filled_quote": "100",
                "filled_at": "2026-08-17T09:00:00+00:00",
            },
        ],
        started_at=datetime(2026, 8, 17, 8, 30, tzinfo=timezone.utc),
    )

    assert inventory == {"ETH/USDT": (Decimal("1"), Decimal("100"))}


def test_demo_pnl_is_partial_for_sell_without_owned_cost_basis():
    result = build_demo_execution_read_model(
        _demo_strategy(), {}, {"checked_at": "now", "positions": [{"symbol": "BTC/USDT", "quantity": 50, "mark_price": None}]},
        [{"strategy_id": "strategy-a", "execution_source": "rule_strategy", "symbol": "BTC/USDT", "side": "sell", "status": "filled", "filled_quantity": "2", "filled_quote": "200", "filled_at": "then"}],
    )
    assert result["pnl"]["status"] == "partial"
    assert result["pnl"]["position_quantity"] == "0"
    assert result["pnl"]["reason_code"] == "insufficient_strategy_fill_history"
    assert result["pnl"]["value"] is None
    assert result["pnl"]["total"] is None
    assert result["equity_curve"]["points"] == []


def test_demo_read_model_marks_legacy_filled_orders_as_incomplete_history():
    result = build_demo_execution_read_model(
        _demo_strategy(), {}, {"checked_at": "now", "positions": []},
        [{"id": "legacy", "strategy_id": "strategy-a", "execution_source": "rule_strategy", "side": "buy", "status": "filled"}],
    )

    assert result["trade_summary"]["purchase_state"] == "unknown"
    assert result["pnl"]["status"] == "unavailable"
    assert result["pnl"]["reason_code"] == "legacy_fill_metadata_unavailable"
    assert result["equity_curve"]["points"] == []


def test_demo_purchase_state_and_counts_keep_partial_fills_distinct():
    result = build_demo_execution_read_model(
        _demo_strategy(), {}, {"checked_at": "now", "positions": [{"symbol": "BTC/USDT", "mark_price": 110}]},
        [{
            "id": "partial", "strategy_id": "strategy-a", "execution_source": "rule_strategy",
            "symbol": "BTC/USDT", "side": "buy", "status": "partially_filled",
            "filled_quantity": "0.5", "filled_quote": "50", "average_fill_price": "100",
        }],
    )

    assert result["trade_summary"]["purchase_state"] == "partially_filled"
    assert result["trade_summary"]["filled_order_count"] == 0
    assert result["trade_summary"]["partially_filled_order_count"] == 1
    assert result["strategy_positions"][0]["quantity"] == "0.5"


def test_demo_partial_pnl_never_exposes_a_complete_total_or_curve():
    result = build_demo_execution_read_model(
        _demo_strategy(), {}, {"checked_at": "now", "positions": [{"symbol": "BTC/USDT", "mark_price": 110}]},
        [
            {"id": "legacy", "strategy_id": "strategy-a", "execution_source": "rule_strategy", "symbol": "BTC/USDT", "side": "buy", "status": "filled"},
            {"id": "known", "strategy_id": "strategy-a", "execution_source": "rule_strategy", "symbol": "BTC/USDT", "side": "buy", "status": "filled", "filled_quantity": "1", "filled_quote": "100"},
        ],
    )

    assert result["pnl"]["status"] == "partial"
    assert result["pnl"]["value"] is None
    assert result["pnl"]["total"] is None
    assert result["equity_curve"]["points"] == []


def test_demo_pnl_sorts_mixed_fill_and_created_timestamps_chronologically():
    result = build_demo_execution_read_model(
        _demo_strategy(), {}, {"checked_at": "2024-01-01T11:00:00+00:00", "positions": []},
        [
            {
                "id": "buy", "strategy_id": "strategy-a", "execution_source": "rule_strategy",
                "symbol": "BTC/USDT", "side": "buy", "status": "filled",
                "filled_quantity": "1", "filled_quote": "100",
                "filled_at": "2024-01-01T09:00:00+00:00",
            },
            {
                "id": "sell", "strategy_id": "strategy-a", "execution_source": "rule_strategy",
                "symbol": "BTC/USDT", "side": "sell", "status": "filled",
                "filled_quantity": "1", "filled_quote": "120",
                "created_at": datetime(2024, 1, 1, 10, tzinfo=timezone.utc),
            },
        ],
    )

    assert result["pnl"]["status"] == "available"
    assert result["pnl"]["realized"] == "20"
    assert result["pnl"]["total"] == "20"


def test_demo_purchase_state_tracks_current_net_position_after_full_exit():
    result = build_demo_execution_read_model(
        _demo_strategy(), {}, {"checked_at": "now", "positions": []},
        [
            {"id": "buy", "strategy_id": "strategy-a", "execution_source": "rule_strategy", "symbol": "BTC/USDT", "side": "buy", "status": "filled", "filled_quantity": "1", "filled_quote": "100", "created_at": "2024-01-01"},
            {"id": "sell", "strategy_id": "strategy-a", "execution_source": "rule_strategy", "symbol": "BTC/USDT", "side": "sell", "status": "filled", "filled_quantity": "1", "filled_quote": "120", "created_at": "2024-01-02"},
        ],
    )

    assert result["trade_summary"]["purchase_state"] == "not_bought"
    assert result["trade_summary"]["current_position_quantity"] == "0"
    assert result["trade_summary"]["latest_order"]["id"] == "sell"


def test_demo_read_model_refuses_paper_strategy():
    with pytest.raises(DemoExecutionReadModelError, match="not configured"):
        build_demo_execution_read_model(
            {"id": "strategy-a", "config": {"execution": {"environment": "paper"}}},
            {},
            {},
            [],
        )


@pytest.mark.asyncio
async def test_demo_read_model_fetches_only_current_connection_and_refreshes_orders():
    calls = []

    class Service:
        async def balance(self, tenant_id, connection_id):
            calls.append(("balance", tenant_id, connection_id))
            return {"balances": []}

        async def positions(self, tenant_id, connection_id, *, account=None):
            calls.append(("positions", tenant_id, connection_id, account))
            return {"positions": []}

        async def refresh_open_orders(self, tenant_id, connection_id):
            calls.append(("refresh", tenant_id, connection_id))

        def list_orders(self, tenant_id, connection_id):
            calls.append(("orders", tenant_id, connection_id))
            return [{"strategy_id": "strategy-a", "execution_source": "rule_strategy"}]

    result = await get_demo_execution_read_model(_demo_strategy(), "tenant-a", Service())

    assert result["orders"]
    assert calls == [
        ("balance", "tenant-a", "conn-a"),
        ("positions", "tenant-a", "conn-a", {"balances": []}),
        ("refresh", "tenant-a", "conn-a"),
        ("orders", "tenant-a", "conn-a"),
    ]
