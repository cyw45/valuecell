import pytest

from valuecell.server.services.rule_strategy_demo_execution_read_model import (
    DemoExecutionReadModelError,
    build_demo_execution_read_model,
    get_demo_execution_read_model,
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
    assert result["orders"] == [
        {"id": "ours", "strategy_id": "strategy-a", "execution_source": "rule_strategy"}
    ]
    assert result["account"]["scope"] == "exchange_connection_shared_account"
    assert result["pnl"]["status"] == "unavailable"
    assert result["pnl"]["value"] is None


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

        async def positions(self, tenant_id, connection_id):
            calls.append(("positions", tenant_id, connection_id))
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
        ("positions", "tenant-a", "conn-a"),
        ("refresh", "tenant-a", "conn-a"),
        ("orders", "tenant-a", "conn-a"),
    ]
