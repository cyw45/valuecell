"""Normalize persisted strategy journal entries into shared trade facts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from valuecell.server.api.schemas.multi_strategy import (
    ExplanationCondition,
    StrategyIdentity,
    TradeExplanation,
    UnifiedTradeFact,
)
from valuecell.server.services.multi_strategy_registry import strategy_code_fingerprint


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
    values = value.get("values") if isinstance(value.get("values"), dict) else {}
    return ExplanationCondition(
        code=code,
        label=str(label),
        state=state,
        actual=values.get("left"),
        threshold=values.get("right"),
        operator=values.get("comparator"),
        detail=str(detail),
        data_at=observed_at,
    )


def journal_trade_facts(strategy: Any, journal: Any) -> list[UnifiedTradeFact]:
    """Read only durable journal data; never infer facts from current config."""
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
    return facts
