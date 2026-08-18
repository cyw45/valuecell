"""Fail-safe, idempotent manual close orchestration for OKX Demo strategies."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy.orm import Session

from valuecell.server.db.models.rule_strategy_manual_close import RuleStrategyManualCloseCommand
from valuecell.server.services.audit_service import record_audit_event
from valuecell.server.services.rule_strategy_demo_execution_read_model import strategy_inventory_by_symbol
from valuecell.server.services.sandbox_exchange_trading_service import (
    SandboxExchangeTradingService,
    SandboxTradingError,
)


class ManualCloseError(ValueError):
    """A manual close cannot be proven safe or completed."""


def canonical_symbol(value: str) -> str:
    return value.strip().upper().replace("-", "/")


def confirmation_text(scope: str, symbol: str | None) -> str:
    return "CLOSE ALL POSITIONS" if scope == "all" else f"CLOSE {canonical_symbol(symbol or '')}"


async def execute_manual_close(
    db: Session,
    *,
    tenant_id: str,
    requested_by: str,
    strategy: dict[str, Any],
    scope: str,
    symbol: str | None,
    idempotency_key: str,
) -> dict[str, Any]:
    """Persist a close command before remote I/O and return durable per-symbol facts."""
    strategy_id = str(strategy["strategy_id"])
    existing = (
        db.query(RuleStrategyManualCloseCommand)
        .filter_by(tenant_id=tenant_id, idempotency_key=idempotency_key)
        .first()
    )
    if existing is not None:
        return _command_data(existing)
    command = RuleStrategyManualCloseCommand(
        tenant_id=tenant_id,
        strategy_id=strategy_id,
        requested_by=requested_by,
        scope=scope,
        symbol=None if scope == "all" else canonical_symbol(symbol or ""),
        idempotency_key=idempotency_key,
        status="pending",
        results=[],
    )
    db.add(command)
    db.commit()
    db.refresh(command)
    try:
        execution = (strategy.get("config") or {}).get("execution") or {}
        if execution.get("environment") != "okx_demo":
            raise ManualCloseError("手动平仓当前只允许对 OKX Demo 策略执行。")
        credential_id = execution.get("sandbox_connection_id")
        if not isinstance(credential_id, str) or not credential_id:
            raise ManualCloseError("策略没有可用的 OKX Demo 连接。")
        service = SandboxExchangeTradingService(db)
        account = await service.balance(tenant_id, credential_id)
        positions = await service.positions(
            tenant_id, credential_id, account=account
        )
        orders = service.list_orders(tenant_id, credential_id)
        attributed_orders = [
            order
            for order in orders
            if order.get("strategy_id") == strategy_id
            and order.get("execution_source") in {"rule_strategy", "manual_close"}
        ]
        inventory = strategy_inventory_by_symbol(attributed_orders)
        _subtract_prior_manual_closes(db, tenant_id, strategy_id, inventory)
        position_by_symbol = {
            canonical_symbol(str(item.get("symbol") or "")): item
            for item in positions.get("positions", [])
            if isinstance(item, dict)
        }
        symbols = (
            [canonical_symbol(symbol or "")]
            if scope == "symbol"
            else sorted(
                target
                for target, (quantity, _cost) in inventory.items()
                if quantity > 0
            )
        )
        if not symbols or any(not item for item in symbols):
            raise ManualCloseError("没有可验证的策略持仓。")
        plans: list[tuple[str, Decimal, Decimal]] = []
        for target in symbols:
            owned, _cost = inventory.get(target, (Decimal(0), Decimal(0)))
            exchange_position = position_by_symbol.get(target)
            available = _decimal(exchange_position.get("available_quantity")) if exchange_position else None
            mark_price = _decimal(exchange_position.get("mark_price")) if exchange_position else None
            if owned <= 0:
                raise ManualCloseError(f"{target} 没有可验证的策略归属持仓。")
            if available is None or mark_price is None or available < owned:
                raise ManualCloseError(f"{target} 的策略归属数量与交易所可用数量不一致，已拒绝平仓。")
            if mark_price <= 0:
                raise ManualCloseError(f"{target} 没有有效的交易所标记价格。")
            plans.append((target, owned, mark_price))
        results: list[dict[str, Any]] = []
        for target, quantity, mark_price in plans:
            order = await service.submit_order(
                tenant_id,
                credential_id,
                f"manualclose-{command.id}-{target.replace('/', '')}",
                target,
                "sell",
                "market",
                quantity * mark_price,
                mark_price,
            )
            results.append(
                {
                    "symbol": target,
                    "requested_quantity": str(quantity),
                    "requested_quote": str(quantity * mark_price),
                    "status": str(order.get("status") or "unknown"),
                    "order_id": order.get("id"),
                    "exchange_order_id": order.get("exchange_order_id"),
                }
            )
        command.results = results
        command.status = "completed" if all(item["status"] in {"filled", "closed"} for item in results) else "submitted"
        record_audit_event(
            db,
            action="strategy.manual_close.submitted",
            target_type="rule_strategy",
            target_id=strategy_id,
            outcome=command.status,
            tenant_id=tenant_id,
            actor_user_id=requested_by,
            metadata={"command_id": command.id, "scope": scope, "symbol": symbol},
        )
        db.commit()
        db.refresh(command)
        return _command_data(command)
    except (ManualCloseError, SandboxTradingError) as exc:
        command.status = "blocked"
        command.results = [{"status": "blocked", "reason": str(exc)}]
        record_audit_event(
            db,
            action="strategy.manual_close.blocked",
            target_type="rule_strategy",
            target_id=strategy_id,
            outcome="blocked",
            tenant_id=tenant_id,
            actor_user_id=requested_by,
            metadata={"command_id": command.id, "scope": scope, "symbol": symbol},
        )
        db.commit()
        raise ManualCloseError(str(exc)) from exc


def _subtract_prior_manual_closes(
    db: Session,
    tenant_id: str,
    strategy_id: str,
    inventory: dict[str, tuple[Decimal, Decimal]],
) -> None:
    commands = (
        db.query(RuleStrategyManualCloseCommand)
        .filter_by(tenant_id=tenant_id, strategy_id=strategy_id)
        .all()
    )
    for command in commands:
        if command.status in {"blocked", "failed"}:
            continue
        for result in command.results or []:
            if result.get("status") not in {"submitted", "filled", "closed", "partially_filled"}:
                continue
            target = canonical_symbol(str(result.get("symbol") or ""))
            quantity = _decimal(result.get("requested_quantity")) or Decimal(0)
            held, cost = inventory.get(target, (Decimal(0), Decimal(0)))
            sold = min(held, quantity)
            inventory[target] = (held - sold, cost - (cost / held * sold if held > 0 else 0))


def _decimal(value: object) -> Decimal | None:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return number if number.is_finite() else None


def _command_data(command: RuleStrategyManualCloseCommand) -> dict[str, Any]:
    return {
        "command_id": command.id,
        "strategy_id": command.strategy_id,
        "scope": command.scope,
        "symbol": command.symbol,
        "status": command.status,
        "results": command.results or [],
        "created_at": command.created_at,
        "updated_at": command.updated_at,
    }
