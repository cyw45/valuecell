"""Strategy-scoped OKX Demo read model, deliberately independent of paper accounting."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from valuecell.server.services.sandbox_exchange_trading_service import SandboxTradingError


class DemoExecutionReadModelError(ValueError):
    """A Demo strategy read cannot safely fall back to a paper representation."""


def _decimal(value: Any) -> Decimal | None:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return result if result.is_finite() else None


def _timestamp_sort_key(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(value / 1000 if value > 10_000_000_000 else value, tz=timezone.utc)
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return datetime.min.replace(tzinfo=timezone.utc)
    else:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _trade_summary(orders: list[dict[str, Any]]) -> dict[str, Any]:
    buys = [item for item in orders if item.get("side") == "buy" and (_decimal(item.get("filled_quantity")) or 0) > 0]
    sells = [item for item in orders if item.get("side") == "sell" and (_decimal(item.get("filled_quantity")) or 0) > 0]
    filled_orders = [item for item in orders if item.get("status") == "filled"]
    partially_filled_orders = [item for item in orders if item.get("status") in {"partial", "partially_filled"}]
    legacy_filled_orders = [item for item in filled_orders if (_decimal(item.get("filled_quantity")) or 0) <= 0]
    pending = [item for item in orders if item.get("status") in {"pending", "submitted", "open", "partial", "partially_filled"}]
    unknown = [item for item in orders if item.get("status") == "submission_unknown"]
    failed = [item for item in orders if item.get("status") in {"failed", "rejected"}]
    net_quantity_by_symbol: dict[str, Decimal] = {}
    for item in buys:
        symbol = str(item.get("symbol") or "")
        net_quantity_by_symbol[symbol] = net_quantity_by_symbol.get(symbol, Decimal(0)) + (_decimal(item.get("filled_quantity")) or Decimal(0))
    for item in sells:
        symbol = str(item.get("symbol") or "")
        net_quantity_by_symbol[symbol] = net_quantity_by_symbol.get(symbol, Decimal(0)) - (_decimal(item.get("filled_quantity")) or Decimal(0))
    current_position_quantity = sum((max(quantity, Decimal(0)) for quantity in net_quantity_by_symbol.values()), Decimal(0))
    if current_position_quantity > 0 and partially_filled_orders:
        purchase_state = "partially_filled"
    elif current_position_quantity > 0:
        purchase_state = "bought"
    elif legacy_filled_orders:
        purchase_state = "unknown"
    elif unknown:
        purchase_state = "unknown"
    elif pending:
        purchase_state = "partially_filled" if any(item.get("status") in {"partial", "partially_filled"} for item in pending) else "pending"
    elif failed and any(item.get("side") == "buy" for item in failed):
        purchase_state = "failed"
    else:
        purchase_state = "not_bought"
    status = "not_bought" if not buys else "bought_and_sold" if sells else "bought"
    latest = max(enumerate(orders), key=lambda pair: (_timestamp_sort_key(pair[1].get("created_at") or pair[1].get("updated_at")), pair[0]), default=(0, None))[1]
    return {
        "status": status,
        "purchase_state": purchase_state,
        "order_count": len(orders),
        "filled_order_count": len(filled_orders),
        "partially_filled_order_count": len(partially_filled_orders),
        "failed_order_count": len(failed),
        "unknown_order_count": len(unknown),
        "filled_buy_orders": len(buys),
        "filled_sell_orders": len(sells),
        "filled_buy_quantity": str(sum((_decimal(item.get("filled_quantity")) or Decimal(0) for item in buys), Decimal(0))),
        "filled_sell_quantity": str(sum((_decimal(item.get("filled_quantity")) or Decimal(0) for item in sells), Decimal(0))),
        "current_position_quantity": str(current_position_quantity),
        "failed_orders": sum(item.get("status") in {"failed", "rejected"} for item in orders),
        "submission_unknown_orders": sum(item.get("status") == "submission_unknown" for item in orders),
        "latest_status": latest.get("status") if latest else None,
        "latest_order": latest,
    }


def _pnl_and_curve(orders: list[dict[str, Any]], positions: dict[str, Any], checked_at: str) -> tuple[dict[str, Any], dict[str, Any]]:
    legacy_filled_orders = [item for item in orders if item.get("status") == "filled" and (_decimal(item.get("filled_quantity")) or 0) <= 0]
    fills = [item for item in orders if (_decimal(item.get("filled_quantity")) or 0) > 0]
    scope = "strategy_attributed_filled_orders_marked_with_shared_account_prices"
    if not fills:
        reason_code = "legacy_fill_metadata_unavailable" if legacy_filled_orders else "no_filled_orders"
        reason = "Filled orders exist but their historical fill metadata is unavailable" if legacy_filled_orders else "No strategy-attributed filled orders are available"
        return (
            {"status": "unavailable", "value": None, "scope": scope, "reason_code": reason_code, "reason": reason, "fees_included": False},
            {"status": "unavailable", "scope": scope, "reason_code": reason_code, "points": []},
        )

    # Shared positions provide marks only. Their quantities are never attributed
    # to this strategy; ownership comes exclusively from strategy fill metadata.
    marks = {item.get("symbol"): _decimal(item.get("mark_price")) for item in positions.get("positions", []) if isinstance(item, dict)}
    inventory: dict[str, tuple[Decimal, Decimal]] = {}
    realized = Decimal(0)
    incomplete_history = False
    missing_fill_facts = False
    for _, fill in sorted(
        enumerate(fills),
        key=lambda pair: (_timestamp_sort_key(pair[1].get("filled_at") or pair[1].get("created_at")), pair[0]),
    ):
        symbol = str(fill.get("symbol") or "")
        quantity = _decimal(fill.get("filled_quantity"))
        quote = _decimal(fill.get("filled_quote"))
        price = _decimal(fill.get("average_fill_price"))
        if quantity is None or quantity <= 0 or (quote is None and price is None):
            missing_fill_facts = True
            continue
        quote = quote if quote is not None else quantity * price  # type: ignore[operator]
        held, cost = inventory.get(symbol, (Decimal(0), Decimal(0)))
        if fill.get("side") == "buy":
            inventory[symbol] = (held + quantity, cost + quote)
        elif fill.get("side") == "sell":
            original_quantity = quantity
            if quantity > held:
                incomplete_history = True
                quantity = held
            if quantity > 0:
                average_cost = cost / held
                sale_price = price if price is not None else quote / original_quantity
                realized += quantity * (sale_price - average_cost)
                inventory[symbol] = (held - quantity, cost - quantity * average_cost)

    unrealized = Decimal(0)
    missing_mark = False
    position_quantity = Decimal(0)
    cost_basis = Decimal(0)
    for symbol, (quantity, cost) in inventory.items():
        position_quantity += quantity
        cost_basis += cost
        if quantity > 0:
            mark = marks.get(symbol)
            if mark is None:
                missing_mark = True
            else:
                unrealized += quantity * mark - cost
    partial = incomplete_history or missing_fill_facts or missing_mark or bool(legacy_filled_orders)
    reason_code = "legacy_fill_metadata_unavailable" if legacy_filled_orders else "insufficient_strategy_fill_history" if incomplete_history else "incomplete_fill_metadata" if missing_fill_facts else "mark_price_unavailable" if missing_mark else None
    total = realized + unrealized
    complete = not partial
    pnl = {
        "status": "partial" if partial else "available", "value": str(total) if complete else None, "scope": scope,
        "reason_code": reason_code, "position_quantity": str(position_quantity), "cost_basis": str(cost_basis),
        "realized": str(realized) if not (incomplete_history or missing_fill_facts or legacy_filled_orders) else None,
        "unrealized": str(unrealized) if complete else None,
        "total": str(total) if complete else None, "fees_included": False,
    }
    return pnl, {
        "status": "unavailable",
        "scope": scope,
        "reason_code": "strategy_equity_history_unavailable",
        "points": [],
    }


def build_demo_execution_read_model(strategy: dict[str, Any], account: dict[str, Any], positions: dict[str, Any], orders: list[dict[str, Any]]) -> dict[str, Any]:
    """Build an explainable, strategy-attributed, non-paper response."""
    execution = (strategy.get("config") or {}).get("execution") or {}
    if execution.get("environment") != "okx_demo":
        raise DemoExecutionReadModelError("Strategy is not configured for OKX Demo execution")
    strategy_id = strategy.get("strategy_id") or strategy.get("id")
    if not strategy_id:
        raise DemoExecutionReadModelError("Strategy identifier is unavailable")
    strategy_orders = [item for item in orders if item.get("strategy_id") == strategy_id and item.get("execution_source") == "rule_strategy"]
    checked_at = positions.get("checked_at") or account.get("checked_at") or datetime.now(timezone.utc).isoformat()
    pnl, equity_curve = _pnl_and_curve(strategy_orders, positions, checked_at)
    return {
        "source": "okx_demo_spot", "strategy_id": strategy_id,
        "connection_id": execution.get("sandbox_connection_id"),
        "account": {"scope": "exchange_connection_shared_account", "data": account},
        "positions": {"scope": "exchange_connection_shared_spot_positions", "data": positions},
        "orders": strategy_orders, "trade_summary": _trade_summary(strategy_orders),
        "pnl": pnl, "equity_curve": equity_curve, "checked_at": checked_at,
    }


async def get_demo_execution_read_model(strategy: dict[str, Any], tenant_id: str, service: Any) -> dict[str, Any]:
    """Fetch exchange-authoritative account facts and locally attributed order audit rows."""
    execution = (strategy.get("config") or {}).get("execution") or {}
    if execution.get("environment") != "okx_demo":
        raise DemoExecutionReadModelError("Strategy is not configured for OKX Demo execution")
    connection_id = execution.get("sandbox_connection_id")
    if not connection_id:
        raise DemoExecutionReadModelError("OKX Demo connection is unavailable")
    try:
        account = await service.balance(tenant_id, connection_id)
        positions = await service.positions(tenant_id, connection_id, account=account)
        await service.refresh_open_orders(tenant_id, connection_id)
        orders = service.list_orders(tenant_id, connection_id)
    except SandboxTradingError:
        raise
    return build_demo_execution_read_model(strategy, account, positions, orders)
