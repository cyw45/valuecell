"""Build UI-safe diagnostics for strategy decision cycles."""

from __future__ import annotations

from typing import Any

from valuecell.agents.common.trading.constants import (
    FEATURE_GROUP_BY_KEY,
    FEATURE_GROUP_BY_MARKET_SNAPSHOT,
)
from valuecell.agents.common.trading.models import (
    DecisionCycleResult,
    FeatureVector,
    TradeHistoryEntry,
    TradeInstruction,
    UserRequest,
)


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _enum_value(value: object) -> str | None:
    raw = getattr(value, "value", value)
    return str(raw) if raw is not None else None


def _market_snapshot_health(feature: FeatureVector) -> dict[str, Any] | None:
    meta = feature.meta or {}
    if meta.get(FEATURE_GROUP_BY_KEY) != FEATURE_GROUP_BY_MARKET_SNAPSHOT:
        return None
    snapshot_ts = _safe_int(meta.get("snapshot_ts_ms")) or _safe_int(feature.ts)
    freshness_age = _safe_int(meta.get("freshness_age_ms"))
    freshness_status = str(meta.get("freshness_status") or "unknown")
    coverage_status = str(meta.get("coverage_status") or "complete")
    return {
        "snapshot_ts_ms": snapshot_ts,
        "freshness_age_ms": freshness_age,
        "freshness_status": freshness_status,
        "coverage_status": coverage_status,
        "exposure_increase_allowed": freshness_status == "fresh" and coverage_status == "complete",
    }


def _feature_interval(feature: FeatureVector) -> str | None:
    meta = feature.meta or {}
    if meta.get(FEATURE_GROUP_BY_KEY) == FEATURE_GROUP_BY_MARKET_SNAPSHOT:
        return "market"
    interval = meta.get("interval")
    return str(interval) if interval is not None else None


def _feature_price(feature: FeatureVector) -> float | None:
    values = feature.values or {}
    for key in ("price.last", "price.close", "close"):
        price = _safe_float(values.get(key))
        if price is not None:
            return price
    return None


def _feature_indicators(feature: FeatureVector) -> dict[str, float | bool | None]:
    values = feature.values or {}
    return {
        "rsi": _safe_float(values.get("rsi")),
        "sma20": _safe_float(values.get("sma20")),
        "sma60": _safe_float(values.get("sma60")),
        "bb_upper": _safe_float(values.get("bb_upper")),
        "bb_middle": _safe_float(values.get("bb_middle")),
        "bb_lower": _safe_float(values.get("bb_lower")),
        "momentum": _safe_float(values.get("mtm14") or values.get("momentum")),
        "rsi_turn_up": bool(values.get("rsi_turn_up")),
        "mtm_turn_up": bool(values.get("mtm_turn_up")),
        "bb_squeeze": bool(values.get("bb_squeeze")),
    }


def _build_symbol_feature_summary(
    features: list[FeatureVector],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for feature in features:
        symbol = feature.instrument.symbol
        summary = summaries.setdefault(
            symbol,
            {
                "symbol": symbol,
                "intervals_seen": [],
                "has_market_snapshot": False,
                "latest_price": None,
                "latest_feature_ts": None,
                "indicator_snapshot": {},
                "snapshot_ts_ms": None,
                "freshness_age_ms": None,
                "freshness_status": "missing",
                "coverage_status": "missing",
                "exposure_increase_allowed": False,
            },
        )
        interval = _feature_interval(feature)
        if interval and interval not in summary["intervals_seen"]:
            summary["intervals_seen"].append(interval)
        if interval == "market":
            summary["has_market_snapshot"] = True
            snapshot_health = _market_snapshot_health(feature)
            if snapshot_health:
                summary.update(snapshot_health)
        price = _feature_price(feature)
        if price is not None:
            summary["latest_price"] = price
        if interval and interval != "market":
            summary["indicator_snapshot"][interval] = _feature_indicators(feature)
        ts = _safe_int(feature.ts)
        if ts is not None:
            current_ts = summary.get("latest_feature_ts")
            summary["latest_feature_ts"] = max(ts, current_ts or ts)
    return summaries


def _instruction_by_symbol(
    instructions: list[TradeInstruction],
) -> dict[str, TradeInstruction]:
    return {instruction.instrument.symbol: instruction for instruction in instructions}


def _trade_impact_by_symbol(trades: list[TradeHistoryEntry]) -> dict[str, dict[str, Any]]:
    impacts: dict[str, dict[str, Any]] = {}
    for trade in trades:
        symbol = trade.instrument.symbol
        impact = impacts.setdefault(
            symbol,
            {
                "filled_quantity": 0.0,
                "notional": 0.0,
                "fee_cost": 0.0,
                "realized_pnl": 0.0,
                "trade_count": 0,
            },
        )
        impact["filled_quantity"] += abs(float(trade.quantity or 0.0))
        impact["notional"] += abs(
            float(trade.notional_entry or trade.notional_exit or 0.0)
        )
        impact["fee_cost"] += float(trade.fee_cost or 0.0)
        impact["realized_pnl"] += float(trade.realized_pnl or 0.0)
        impact["trade_count"] += 1
    return impacts


def _diagnostic_config(request: UserRequest) -> dict[str, Any]:
    trading = request.trading_config
    exchange = request.exchange_config
    return {
        "strategy_name": trading.strategy_name,
        "strategy_type": _enum_value(trading.strategy_type),
        "symbols": list(trading.symbols or []),
        "decide_interval": trading.decide_interval,
        "initial_capital": trading.initial_capital,
        "initial_free_cash": trading.initial_free_cash,
        "max_leverage": trading.max_leverage,
        "max_positions": trading.max_positions,
        "cap_factor": trading.cap_factor,
        "strategy_params": dict(trading.strategy_params or {}),
        "exchange_id": exchange.exchange_id,
        "trading_mode": _enum_value(exchange.trading_mode),
        "market_type": _enum_value(exchange.market_type),
        "margin_mode": _enum_value(exchange.margin_mode),
        "fee_bps": exchange.fee_bps,
    }


def _conditions_for_symbol(
    *,
    summary: dict[str, Any],
    instruction: TradeInstruction | None,
    reason: str,
) -> list[dict[str, Any]]:
    has_market = bool(summary.get("has_market_snapshot"))
    latest_price = summary.get("latest_price")
    intervals = summary.get("intervals_seen") or []
    triggered = instruction is not None
    freshness_status = str(summary.get("freshness_status") or "missing")
    exposure_increase_allowed = bool(summary.get("exposure_increase_allowed"))
    return [
        {
            "label": "market_snapshot",
            "status": "passed" if has_market else "blocked",
            "detail": "Realtime market snapshot available" if has_market else "Missing realtime market snapshot",
        },
        {
            "label": "latest_price",
            "status": "passed" if latest_price is not None else "blocked",
            "detail": f"latest_price={latest_price}" if latest_price is not None else "No latest price in features",
        },
        {
            "label": "market_freshness",
            "status": "passed" if freshness_status == "fresh" else "blocked",
            "detail": (
                f"Realtime snapshot is fresh (age_ms={summary.get('freshness_age_ms')})"
                if freshness_status == "fresh"
                else f"Realtime snapshot freshness is {freshness_status}"
            ),
        },
        {
            "label": "exposure_increase_gate",
            "status": "passed" if exposure_increase_allowed else "blocked",
            "detail": (
                "Realtime data permits exposure increases"
                if exposure_increase_allowed
                else "Realtime data blocks new or increased exposure; reduce-only exits remain permitted"
            ),
        },
        {
            "label": "indicator_coverage",
            "status": "passed" if intervals else "blocked",
            "detail": ", ".join(intervals) if intervals else "No interval indicators observed",
        },
        {
            "label": "strategy_decision",
            "status": "triggered" if triggered else "not_triggered",
            "detail": reason,
        },
    ]


def _fund_impact(
    *,
    instruction: TradeInstruction | None,
    summary: dict[str, Any],
    trade_impact: dict[str, Any] | None,
    portfolio_value: float | None,
) -> dict[str, Any]:
    latest_price = _safe_float(summary.get("latest_price"))
    requested_quantity = _safe_float(getattr(instruction, "quantity", None))
    estimated_notional = None
    if latest_price is not None and requested_quantity is not None:
        estimated_notional = abs(latest_price * requested_quantity)
    return {
        "portfolio_value_after_cycle": portfolio_value,
        "requested_quantity": requested_quantity,
        "estimated_notional": estimated_notional,
        "filled_quantity": (trade_impact or {}).get("filled_quantity", 0.0),
        "executed_notional": (trade_impact or {}).get("notional", 0.0),
        "fee_cost": (trade_impact or {}).get("fee_cost", 0.0),
        "realized_pnl": (trade_impact or {}).get("realized_pnl", 0.0),
        "trade_count": (trade_impact or {}).get("trade_count", 0),
    }


def _decision_path(
    *,
    action: str,
    reason: str,
    has_instruction: bool,
) -> list[str]:
    path = ["Fetched market data", "Computed indicators", "Evaluated strategy rules"]
    if has_instruction:
        path.append(f"Emitted action: {action}")
    else:
        path.append("No order emitted")
    path.append(reason)
    return path


def build_cycle_diagnostics(
    *,
    request: UserRequest,
    result: DecisionCycleResult,
) -> dict[str, Any]:
    """Build a compact diagnostics payload for persistence and frontend display."""
    features = []
    for record in result.history_records:
        if record.kind == "features":
            raw_features = record.payload.get("features", [])
            if isinstance(raw_features, list):
                features = [FeatureVector.model_validate(item) for item in raw_features]
            break

    feature_summary = _build_symbol_feature_summary(features)
    instructions_by_symbol = _instruction_by_symbol(result.instructions or [])
    trade_impacts = _trade_impact_by_symbol(result.trades or [])
    expected_symbols = list(request.trading_config.symbols or [])
    symbol_decisions: list[dict[str, Any]] = []
    portfolio_value = _safe_float(result.portfolio_view.total_value)

    for symbol in expected_symbols:
        summary = feature_summary.get(
            symbol,
            {
                "symbol": symbol,
                "intervals_seen": [],
                "has_market_snapshot": False,
                "latest_price": None,
                "latest_feature_ts": None,
                "indicator_snapshot": {},
                "snapshot_ts_ms": None,
                "freshness_age_ms": None,
                "freshness_status": "missing",
                "coverage_status": "missing",
                "exposure_increase_allowed": False,
            },
        )
        instruction = instructions_by_symbol.get(symbol)
        if instruction is None:
            action = "noop"
            quantity = None
            reason = result.rationale or "No actionable signals"
        else:
            action = _enum_value(instruction.action) or _enum_value(instruction.side)
            quantity = instruction.quantity
            reason = None
            if instruction.meta:
                reason_value = instruction.meta.get("rationale")
                reason = str(reason_value) if reason_value is not None else None
            reason = reason or result.rationale or "Order emitted"
        conditions = _conditions_for_symbol(
            summary=summary,
            instruction=instruction,
            reason=reason,
        )
        symbol_decisions.append(
            {
                **summary,
                "action": action,
                "quantity": quantity,
                "reason": reason,
                "conditions": conditions,
                "decision_path": _decision_path(
                    action=action,
                    reason=reason,
                    has_instruction=instruction is not None,
                ),
                "fund_impact": _fund_impact(
                    instruction=instruction,
                    summary=summary,
                    trade_impact=trade_impacts.get(symbol),
                    portfolio_value=portfolio_value,
                ),
            }
        )

    observed_symbols = sorted(feature_summary.keys())
    missing_symbols = [symbol for symbol in expected_symbols if symbol not in feature_summary]
    market_symbols = [
        symbol
        for symbol, summary in feature_summary.items()
        if bool(summary.get("has_market_snapshot"))
    ]
    stale_symbols = [
        symbol
        for symbol in expected_symbols
        if feature_summary.get(symbol, {}).get("freshness_status") == "stale"
    ]
    fetched_count = len(market_symbols)
    coverage_status = (
        "complete"
        if fetched_count == len(expected_symbols)
        else "partial" if fetched_count else "missing"
    )
    freshness_status = "stale" if stale_symbols else "fresh" if fetched_count else "missing"
    exposure_increase_allowed = (
        coverage_status == "complete" and freshness_status == "fresh"
    )
    instruction_count = len(result.instructions or [])
    order_count = sum(
        1 for item in result.instructions or [] if _enum_value(item.action) != "noop"
    )

    triggered_conditions = [
        condition
        for decision in symbol_decisions
        for condition in decision.get("conditions", [])
        if condition.get("status") in {"passed", "triggered"}
    ]
    blocked_conditions = [
        condition
        for decision in symbol_decisions
        for condition in decision.get("conditions", [])
        if condition.get("status") in {"blocked", "not_triggered"}
    ]
    fund_impact = {
        "portfolio_value_after_cycle": portfolio_value,
        "cash_after_cycle": _safe_float(result.portfolio_view.free_cash),
        "estimated_notional": sum(
            float((decision.get("fund_impact") or {}).get("estimated_notional") or 0.0)
            for decision in symbol_decisions
        ),
        "executed_notional": sum(
            float((decision.get("fund_impact") or {}).get("executed_notional") or 0.0)
            for decision in symbol_decisions
        ),
        "fee_cost": sum(
            float((decision.get("fund_impact") or {}).get("fee_cost") or 0.0)
            for decision in symbol_decisions
        ),
        "realized_pnl": sum(
            float((decision.get("fund_impact") or {}).get("realized_pnl") or 0.0)
            for decision in symbol_decisions
        ),
    }

    return {
        "strategy_id": result.strategy_summary.strategy_id,
        "compose_id": result.compose_id,
        "cycle_index": result.cycle_index,
        "created_at_ms": result.timestamp_ms,
        "rationale": result.rationale,
        "config": _diagnostic_config(request),
        "expected_symbol_count": len(expected_symbols),
        "observed_symbol_count": len(observed_symbols),
        "observed_symbols": observed_symbols,
        "missing_symbols": missing_symbols,
        "instruction_count": instruction_count,
        "order_count": order_count,
        "no_order_count": max(0, len(expected_symbols) - order_count),
        "market_data_health": {
            "ok": exposure_increase_allowed,
            "provider": request.exchange_config.exchange_id,
            "fetched_count": fetched_count,
            "missing_count": len(missing_symbols),
            "missing_symbols": missing_symbols,
            "status": "healthy" if exposure_increase_allowed else "degraded" if fetched_count else "missing",
            "freshness_status": freshness_status,
            "coverage_status": coverage_status,
            "stale_count": len(stale_symbols),
            "stale_symbols": stale_symbols,
            "exposure_increase_allowed": exposure_increase_allowed,
        },
        "explanation": {
            "summary": result.rationale or "No cycle rationale recorded",
            "action_reason": result.rationale or "No cycle rationale recorded",
            "triggered_conditions": triggered_conditions,
            "blocked_conditions": blocked_conditions,
            "fund_impact": fund_impact,
            "orders": order_count,
            "no_orders": max(0, len(expected_symbols) - order_count),
            "symbols_evaluated": len(expected_symbols),
        },
        "symbol_decisions": symbol_decisions,
    }
