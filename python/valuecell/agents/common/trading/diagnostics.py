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
            },
        )
        interval = _feature_interval(feature)
        if interval and interval not in summary["intervals_seen"]:
            summary["intervals_seen"].append(interval)
        if interval == "market":
            summary["has_market_snapshot"] = True
        price = _feature_price(feature)
        if price is not None:
            summary["latest_price"] = price
        ts = _safe_int(feature.ts)
        if ts is not None:
            current_ts = summary.get("latest_feature_ts")
            summary["latest_feature_ts"] = max(ts, current_ts or ts)
    return summaries


def _instruction_by_symbol(
    instructions: list[TradeInstruction],
) -> dict[str, TradeInstruction]:
    return {instruction.instrument.symbol: instruction for instruction in instructions}


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
        "exchange_id": exchange.exchange_id,
        "trading_mode": _enum_value(exchange.trading_mode),
        "market_type": _enum_value(exchange.market_type),
        "margin_mode": _enum_value(exchange.margin_mode),
        "fee_bps": exchange.fee_bps,
    }


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
    expected_symbols = list(request.trading_config.symbols or [])
    symbol_decisions: list[dict[str, Any]] = []

    for symbol in expected_symbols:
        summary = feature_summary.get(
            symbol,
            {
                "symbol": symbol,
                "intervals_seen": [],
                "has_market_snapshot": False,
                "latest_price": None,
                "latest_feature_ts": None,
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
        symbol_decisions.append(
            {
                **summary,
                "action": action,
                "quantity": quantity,
                "reason": reason,
            }
        )

    observed_symbols = sorted(feature_summary.keys())
    missing_symbols = [symbol for symbol in expected_symbols if symbol not in feature_summary]
    market_symbols = [
        symbol
        for symbol, summary in feature_summary.items()
        if bool(summary.get("has_market_snapshot"))
    ]
    instruction_count = len(result.instructions or [])
    order_count = sum(1 for item in result.instructions or [] if _enum_value(item.action) != "noop")

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
            "ok": not missing_symbols and bool(features),
            "provider": request.exchange_config.exchange_id,
            "fetched_count": len(market_symbols),
            "missing_count": len(missing_symbols),
            "missing_symbols": missing_symbols,
        },
        "symbol_decisions": symbol_decisions,
    }
