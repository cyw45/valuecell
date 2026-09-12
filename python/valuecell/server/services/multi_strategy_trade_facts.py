"""Normalize persisted strategy journal entries into shared trade facts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from valuecell.server.api.schemas.multi_strategy import (
    ExplanationCondition,
    StrategyIdentity,
    TradeExplanation,
    UnifiedTradeFact,
)
from valuecell.server.services.multi_strategy_registry import strategy_code_fingerprint
from valuecell.server.services.strategy_condition_facts import (
    comparison_facts,
    condition_data_timestamp_ms,
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _condition(value: Any, observed_at: datetime) -> ExplanationCondition | None:
    if not isinstance(value, dict):
        return None
    code = value.get("code")
    if not isinstance(code, str) or not code:
        return None
    state = value.get("state")
    if state not in {"triggered", "not_triggered", "blocked", "unavailable"}:
        state = "unavailable"
    label = value.get("label") or code
    detail = value.get("detail") or "服务端已记录该策略条件。"
    facts = comparison_facts(value)
    data_timestamp_ms = condition_data_timestamp_ms(value)
    data_at = (
        datetime.fromtimestamp(data_timestamp_ms / 1000, tz=timezone.utc)
        if data_timestamp_ms is not None
        else observed_at
    )
    return ExplanationCondition(
        code=code,
        label=str(label),
        state=state,
        actual=facts.actual,
        threshold=facts.threshold,
        operator=facts.operator,
        detail=str(detail),
        data_at=data_at,
    )


def _shared_demo_facts(
    identity: StrategyIdentity,
    journal: Any,
    conditions: list[ExplanationCondition],
    reason: str,
    shared_orders: list[Any],
    shared_fills: list[Any],
) -> list[UnifiedTradeFact]:
    """Build one fact per shared Demo order without consulting Paper rows."""
    fills_by_order: dict[str, list[Any]] = {}
    for fill in shared_fills:
        order_id = _field(fill, "order_id")
        if order_id:
            fills_by_order.setdefault(str(order_id), []).append(fill)
    facts: list[UnifiedTradeFact] = []
    for order in shared_orders:
        order_id = str(_field(order, "order_id") or "")
        if not order_id:
            continue
        side_value = str(_field(order, "side") or "")
        side = side_value if side_value in {"buy", "sell", "short", "cover"} else None
        symbol = _field(order, "symbol") or journal.result.get("symbol")
        if side is None or not isinstance(symbol, str) or not symbol:
            continue
        fills = fills_by_order.get(order_id, [])
        quantity = sum((_number(_field(item, "quantity")) or 0 for item in fills), 0.0)
        quote = sum((_number(_field(item, "quote_amount")) or 0 for item in fills), 0.0)
        fee = sum((_number(_field(item, "fee_quote")) or 0 for item in fills), 0.0)
        price = quote / quantity if quantity > 0 and quote > 0 else None
        status = str(_field(order, "status") or "pending")
        status_map = {
            "open": "submitted",
            "rejected": "failed",
            "partial": "partially_filled",
        }
        normalized_status = status_map.get(status, status)
        if normalized_status not in {"signal", "blocked", "pending", "submitted", "submission_unknown", "recovery_required", "partially_filled", "filled", "cancelled", "failed"}:
            normalized_status = "pending"
        facts.append(
            UnifiedTradeFact(
                identity=identity,
                batch_id=_field(order, "batch_id") or getattr(journal, "batch_id", None),
                evaluation_id=getattr(journal, "evaluation_id", None),
                intent_id=_field(order, "intent_id"),
                reservation_id=_field(order, "reservation_id"),
                order_id=order_id,
                fill_id=str(_field(fills[0], "fill_id")) if fills else None,
                symbol=symbol,
                side=side,
                status=normalized_status,
                requested_quote=_number(_field(order, "requested_quote")),
                filled_quote=quote or None,
                requested_quantity=_number(_field(order, "requested_quantity")),
                filled_quantity=quantity or None,
                average_fill_price=price,
                fee_quote=fee or None,
                created_at=getattr(journal, "created_at"),
                filled_at=_field(fills[-1], "occurred_at") if fills else None,
                explanation=TradeExplanation(
                    decision=str(journal.result.get("action") or side),
                    decision_reason=reason,
                    conditions=conditions,
                    execution_path="okx_demo",
                    final_result=normalized_status,
                ),
            )
        )
    return facts


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def journal_trade_facts(
    strategy: Any,
    journal: Any,
    *,
    shared_fills: list[Any] | None = None,
    shared_orders: list[Any] | None = None,
) -> list[UnifiedTradeFact]:
    strategy_kind = getattr(strategy, "strategy_kind", "configurable_rule")
    strategy_version = getattr(strategy, "strategy_version", "existing")
    code_fingerprint = getattr(strategy, "code_fingerprint", "legacy-configurable")
    observed_at = journal.created_at
    identity = StrategyIdentity(
        strategy_id=str(strategy.strategy_id),
        tenant_id=str(strategy.tenant_id),
        kind=strategy_kind,
        strategy_version=strategy_version,
        code_fingerprint=code_fingerprint
        if code_fingerprint
        else strategy_code_fingerprint(strategy_kind),
    )
    result = journal.result if isinstance(journal.result, dict) else {}
    raw_conditions = result.get("conditions") or []
    conditions = [
        condition
        for raw in raw_conditions
        if (condition := _condition(raw, observed_at)) is not None
    ]
    reason = str(result.get("reason") or "服务端未记录策略决策原因。")
    facts: list[UnifiedTradeFact] = []
    for index, trade in enumerate(journal.trades or []):
        if not isinstance(trade, dict):
            continue
        action = str(trade.get("action") or result.get("action") or "")
        side = "buy" if action in {"buy", "entry", "add"} else "sell" if action in {"sell", "reduce", "close"} else None
        symbol = trade.get("symbol") or result.get("symbol")
        if side is None or not isinstance(symbol, str) or not symbol:
            continue
        quantity = _number(trade.get("quantity"))
        quote_amount = _number(trade.get("quote_amount"))
        price = _number(trade.get("price"))
        facts.append(
            UnifiedTradeFact(
                identity=identity,
                batch_id=getattr(journal, "batch_id", None),
                evaluation_id=getattr(journal, "evaluation_id", None),
                reservation_id=None,
                symbol=symbol,
                side=side,
                status="filled" if trade.get("execution") == "paper_filled" else "signal",
                requested_quote=quote_amount,
                filled_quote=quote_amount,
                filled_quantity=quantity,
                average_fill_price=price,
                created_at=observed_at,
                filled_at=observed_at if trade.get("execution") == "paper_filled" else None,
                explanation=TradeExplanation(
                    decision=action,
                    decision_reason=reason,
                    conditions=conditions,
                    execution_path=str(trade.get("execution")) if trade.get("execution") else None,
                    final_result="paper_filled" if trade.get("execution") == "paper_filled" else None,
                ),
            )
        )
    execution = result.get("execution")
    if (
        not facts
        and isinstance(execution, dict)
        and execution.get("execution") == "paper_filled"
        and execution.get("paper_fill") is True
    ):
        action = str(result.get("action") or "")
        recorded_side = execution.get("filled_side")
        side = recorded_side if recorded_side in {"buy", "sell", "short", "cover"} else (
            "buy"
            if action in {"long_entry", "buy", "entry", "add"}
            else "sell"
            if action in {"exit", "short_entry", "sell", "reduce", "close"}
            else None
        )
        symbol = result.get("symbol")
        quantity = _number(execution.get("filled_quantity"))
        price = _number(execution.get("filled_price"))
        if side is not None and isinstance(symbol, str) and symbol and quantity and price:
            facts.append(
                UnifiedTradeFact(
                    identity=identity,
                    batch_id=getattr(journal, "batch_id", None),
                    evaluation_id=getattr(journal, "evaluation_id", None),
                    fill_id=str(execution.get("fill_id")) if execution.get("fill_id") else None,
                    symbol=symbol,
                    side=side,
                    status="filled",
                    requested_quote=quantity * price,
                    filled_quote=quantity * price,
                    filled_quantity=quantity,
                    average_fill_price=price,
                    created_at=observed_at,
                    filled_at=observed_at,
                    explanation=TradeExplanation(
                        decision=action,
                        decision_reason=reason,
                        conditions=conditions,
                        execution_path="paper",
                        final_result="paper_filled",
                    ),
                )
            )
    if shared_orders:
        facts.extend(
            _shared_demo_facts(
                identity,
                journal,
                conditions,
                reason,
                shared_orders,
                shared_fills or [],
            )
        )
    return facts
