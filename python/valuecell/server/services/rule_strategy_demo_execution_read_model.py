"""Strategy-scoped OKX Demo read model, deliberately independent of paper accounting."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from valuecell.server.services.sandbox_exchange_trading_service import (
    SandboxExchangeTradingService,
    SandboxTradingError,
)


class DemoExecutionReadModelError(ValueError):
    """A Demo strategy read cannot safely fall back to a paper representation."""


def build_demo_execution_read_model(
    strategy: dict[str, Any],
    account: dict[str, Any],
    positions: dict[str, Any],
    orders: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the explicit non-paper response after exchange reads finish."""
    execution = (strategy.get("config") or {}).get("execution") or {}
    if execution.get("environment") != "okx_demo":
        raise DemoExecutionReadModelError("Strategy is not configured for OKX Demo execution")
    strategy_id = strategy.get("strategy_id") or strategy.get("id")
    if not strategy_id:
        raise DemoExecutionReadModelError("Strategy identifier is unavailable")
    return {
        "source": "okx_demo_spot",
        "strategy_id": strategy_id,
        "connection_id": execution.get("sandbox_connection_id"),
        "account": {
            "scope": "exchange_connection_shared_account",
            "data": account,
        },
        "positions": {
            "scope": "exchange_connection_shared_spot_positions",
            "data": positions,
        },
        "orders": [
            order for order in orders
            if order.get("strategy_id") == strategy_id
            and order.get("execution_source") == "rule_strategy"
        ],
        "pnl": {
            "status": "unavailable",
            "value": None,
            "reason": "OKX Demo spot realized/unrealized PnL is not derived from paper ledger",
        },
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


async def get_demo_execution_read_model(
    strategy: dict[str, Any],
    tenant_id: str,
    service: Any,
) -> dict[str, Any]:
    """Fetch exchange-authoritative account facts and locally attributed order audit rows."""
    execution = (strategy.get("config") or {}).get("execution") or {}
    if execution.get("environment") != "okx_demo":
        raise DemoExecutionReadModelError("Strategy is not configured for OKX Demo execution")
    connection_id = execution.get("sandbox_connection_id")
    if not connection_id:
        raise DemoExecutionReadModelError("OKX Demo connection is unavailable")
    try:
        account = await service.balance(tenant_id, connection_id)
        positions = await service.positions(tenant_id, connection_id)
        await service.refresh_open_orders(tenant_id, connection_id)
        orders = service.list_orders(tenant_id, connection_id)
    except SandboxTradingError:
        raise
    return build_demo_execution_read_model(strategy, account, positions, orders)
