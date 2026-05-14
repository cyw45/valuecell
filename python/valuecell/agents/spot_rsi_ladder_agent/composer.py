"""Rule-based spot RSI ladder strategy composer."""

from __future__ import annotations

from typing import Dict, List

from valuecell.agents.common.trading.decision.interfaces import BaseComposer
from valuecell.agents.common.trading.models import (
    ComposeContext,
    ComposeResult,
    InstrumentRef,
    TradeDecisionAction,
    TradeDecisionItem,
    TradePlanProposal,
    UserRequest,
)
from valuecell.agents.common.trading.utils import extract_price_map

from .config import (
    ADD_BUY_RATIO,
    DAILY_CIRCUIT_BREAKER_RATIO,
    SpotRsiStrategyProfile,
    TAIL_DRAWDOWN_RATIO,
)
from .state import SymbolStrategyState


def _value_to_float(mapping: Dict[str, object], key: str) -> float | None:
    value = mapping.get(key)
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def _value_to_bool(mapping: Dict[str, object], key: str) -> bool:
    return bool(mapping.get(key))


class SpotRsiLadderComposer(BaseComposer):
    """Executes the user's fixed RSI ladder rules without any LLM."""

    def __init__(
        self,
        request: UserRequest,
        profile: SpotRsiStrategyProfile,
    ) -> None:
        super().__init__(request, default_slippage_bps=15)
        self._profile = profile
        self._state: Dict[str, SymbolStrategyState] = {}

    async def compose(self, context: ComposeContext) -> ComposeResult:
        feature_map = self._build_feature_map(context)
        price_map = self._build_price_map(context, feature_map)
        bear_market = self._is_bear_market(feature_map)
        if bear_market:
            return ComposeResult(
                instructions=[],
                rationale="Bear market silent mode: no orders generated",
            )

        items: List[TradeDecisionItem] = []
        rationales: List[str] = []
        reserved_cash = 0.0

        for symbol in self._request.trading_config.symbols:
            item = self._build_symbol_decision(
                context=context,
                feature_map=feature_map,
                price_map=price_map,
                symbol=symbol,
                available_cash=self._available_cash(context, reserved_cash),
            )
            if item is None:
                continue
            items.append(item)
            if item.action == TradeDecisionAction.OPEN_LONG:
                reserved_cash += item.target_qty * price_map.get(symbol, 0.0)
            if item.rationale:
                rationales.append(item.rationale)

        plan = TradePlanProposal(
            items=items,
            rationale=" | ".join(rationales) if rationales else "No actionable signals",
        )
        instructions = self._normalize_plan(context, plan)
        return ComposeResult(instructions=instructions, rationale=plan.rationale)

    def _build_feature_map(
        self,
        context: ComposeContext,
    ) -> Dict[str, Dict[str, Dict[str, object]]]:
        feature_map: Dict[str, Dict[str, Dict[str, object]]] = {}
        for feature in context.features:
            interval = str((feature.meta or {}).get("interval") or "")
            if not interval:
                continue
            symbol = feature.instrument.symbol
            feature_map.setdefault(symbol, {})[interval] = feature.values
        return feature_map

    def _build_price_map(
        self,
        context: ComposeContext,
        feature_map: Dict[str, Dict[str, Dict[str, object]]],
    ) -> Dict[str, float]:
        price_map = extract_price_map(context.features)
        for symbol, by_interval in feature_map.items():
            if symbol in price_map:
                continue
            primary = by_interval.get(self._profile.primary_interval, {})
            close_price = _value_to_float(primary, "close")
            if close_price is not None:
                price_map[symbol] = close_price
        return price_map

    def _portfolio_equity(self, context: ComposeContext) -> float:
        if context.portfolio.total_value is not None:
            return max(0.0, float(context.portfolio.total_value))
        return max(0.0, float(context.portfolio.account_balance or 0.0))

    def _strategy_bucket_total(
        self,
        context: ComposeContext,
    ) -> float:
        return self._portfolio_equity(context)

    def _available_cash(self, context: ComposeContext, reserved_cash: float) -> float:
        free_cash = context.portfolio.free_cash
        if free_cash is None:
            free_cash = context.portfolio.account_balance
        return max(0.0, float(free_cash or 0.0) - reserved_cash)

    def _current_quantity(self, context: ComposeContext, symbol: str) -> float:
        position = context.portfolio.positions.get(symbol)
        if position is None:
            return 0.0
        return max(0.0, float(position.quantity))

    def _is_bear_market(
        self,
        feature_map: Dict[str, Dict[str, Dict[str, object]]],
    ) -> bool:
        btc = feature_map.get("BTC-USDT") or feature_map.get(
            self._request.trading_config.symbols[0]
        )
        if not btc:
            return False
        day_values = btc.get("1d")
        trend_values = btc.get("4h")
        if not day_values or not trend_values:
            return False
        close_price = _value_to_float(day_values, "close")
        sma60 = _value_to_float(day_values, "sma60")
        sma60_slope = _value_to_float(day_values, "sma60_slope")
        trend_rsi = _value_to_float(trend_values, "rsi")
        if close_price is None or sma60 is None or sma60_slope is None:
            return False
        return (
            close_price < sma60
            and sma60_slope < 0
            and trend_rsi is not None
            and trend_rsi < 45
        )

    def _daily_circuit_breaker(
        self,
        symbol_features: Dict[str, Dict[str, object]],
    ) -> bool:
        day_values = symbol_features.get("1d")
        if not day_values:
            return False
        day_change = _value_to_float(day_values, "change_pct")
        return day_change is not None and abs(day_change) >= DAILY_CIRCUIT_BREAKER_RATIO

    def _ma_is_flat_or_up(self, values: Dict[str, object], ma_key: str) -> bool:
        ma_value = _value_to_float(values, ma_key)
        slope = _value_to_float(values, f"{ma_key}_slope")
        if ma_value is None or slope is None:
            return False
        flat_band = abs(ma_value) * 0.0005
        return slope >= -flat_band

    def _entry_confirmed(
        self,
        symbol_features: Dict[str, Dict[str, object]],
    ) -> bool:
        for interval in self._profile.entry_confirm_intervals:
            values = symbol_features.get(interval)
            if not values:
                return False
            if not _value_to_bool(values, "close_turn_up"):
                return False
            if not _value_to_bool(values, "rsi_turn_up"):
                return False
        return True

    def _trend_confirmed(
        self,
        symbol_features: Dict[str, Dict[str, object]],
    ) -> bool:
        for interval in self._profile.trend_confirm_intervals:
            values = symbol_features.get(interval)
            if not values:
                return False
            if not (
                _value_to_bool(values, "rsi_turn_up")
                or self._ma_is_flat_or_up(values, self._profile.ma_field)
            ):
                return False
        return True

    def _momentum_confirmed(self, primary: Dict[str, object]) -> bool:
        if self._profile.require_mtm_turn_up and not _value_to_bool(
            primary, "mtm_turn_up"
        ):
            return False
        if self._profile.require_mtm_below_zero and not _value_to_bool(
            primary, "mtm_below_zero"
        ):
            return False
        return True

    def _volatility_confirmed(self, primary: Dict[str, object]) -> bool:
        if self._profile.require_bollinger_squeeze and not _value_to_bool(
            primary, "bb_squeeze"
        ):
            return False
        if self._profile.require_bollinger_lower_touch and not _value_to_bool(
            primary, "bb_near_lower"
        ):
            return False
        return True

    def _ma_is_up(self, values: Dict[str, object], ma_key: str) -> bool:
        slope = _value_to_float(values, f"{ma_key}_slope")
        return slope is not None and slope > 0

    def _has_clear_bearish_trend(
        self,
        values: Dict[str, object] | None,
    ) -> bool:
        if not values:
            return False

        close_price = _value_to_float(values, "close")
        sma20 = _value_to_float(values, "sma20")
        sma20_slope = _value_to_float(values, "sma20_slope")
        current_rsi = _value_to_float(values, "rsi")
        current_mtm = _value_to_float(values, "mtm14")
        if (
            close_price is None
            or sma20 is None
            or sma20_slope is None
            or current_rsi is None
            or current_mtm is None
        ):
            return False

        return (
            close_price < sma20
            and sma20_slope < 0
            and current_rsi < 45
            and current_mtm < 0
        )

    def _build_symbol_decision(
        self,
        *,
        context: ComposeContext,
        feature_map: Dict[str, Dict[str, Dict[str, object]]],
        price_map: Dict[str, float],
        symbol: str,
        available_cash: float,
    ) -> TradeDecisionItem | None:
        symbol_features = feature_map.get(symbol)
        if not symbol_features:
            return None
        primary = symbol_features.get(self._profile.primary_interval)
        if not primary:
            return None

        current_price = price_map.get(symbol)
        primary_close = _value_to_float(primary, "close")
        if current_price is None and primary_close is not None:
            current_price = primary_close
        if current_price is None or current_price <= 0:
            return None

        state = self._state.setdefault(symbol, SymbolStrategyState())
        current_qty = self._current_quantity(context, symbol)
        if current_qty <= self._quantity_precision:
            state.reset_on_flat()

        if self._daily_circuit_breaker(symbol_features):
            return None

        current_rsi = _value_to_float(primary, "rsi")
        if current_rsi is not None and current_rsi < self._profile.reset_exit_rsi:
            state.reset_exit_ladder()

        sell_item = self._build_sell_item(
            symbol=symbol,
            state=state,
            primary=primary,
            current_qty=current_qty,
            current_price=current_price,
        )
        if sell_item is not None:
            state.entry_thresholds_hit.clear()
            state.add_count = 0
            return sell_item

        add_item = self._build_add_item(
            context=context,
            symbol=symbol,
            state=state,
            symbol_features=symbol_features,
            primary=primary,
            current_qty=current_qty,
            current_price=current_price,
            available_cash=available_cash,
        )
        if add_item is not None:
            return add_item

        return self._build_entry_item(
            context=context,
            symbol=symbol,
            state=state,
            symbol_features=symbol_features,
            primary=primary,
            current_qty=current_qty,
            current_price=current_price,
            available_cash=available_cash,
        )

    def _build_entry_item(
        self,
        *,
        context: ComposeContext,
        symbol: str,
        state: SymbolStrategyState,
        symbol_features: Dict[str, Dict[str, object]],
        primary: Dict[str, object],
        current_qty: float,
        current_price: float,
        available_cash: float,
    ) -> TradeDecisionItem | None:
        close_price = _value_to_float(primary, "close")
        ma_value = _value_to_float(primary, self._profile.ma_field)
        current_rsi = _value_to_float(primary, "rsi")
        if close_price is None or ma_value is None or current_rsi is None:
            return None
        if close_price >= ma_value:
            return None
        if self._profile.ma_field == "sma60" and not self._ma_is_flat_or_up(primary, "sma60"):
            return None
        if not self._entry_confirmed(symbol_features):
            return None
        if not self._trend_confirmed(symbol_features):
            return None
        if not self._volatility_confirmed(primary):
            return None
        if not self._momentum_confirmed(primary):
            return None

        if available_cash <= 0:
            return None

        thresholds = self._profile.entry_rsi_thresholds
        entry_buy_ratios = self._profile.entry_buy_ratios
        strategy_bucket_total = self._strategy_bucket_total(context)
        if strategy_bucket_total <= 0:
            return None

        buy_notional = 0.0
        triggered: List[int] = []
        for threshold in thresholds:
            if current_rsi > threshold or threshold in state.entry_thresholds_hit:
                continue
            tranche = strategy_bucket_total * entry_buy_ratios.get(threshold, 0.0)
            tranche = min(tranche, max(0.0, available_cash - buy_notional))
            if tranche <= 0:
                continue
            buy_notional += tranche
            triggered.append(threshold)

        if buy_notional <= 0:
            return None

        for threshold in triggered:
            state.entry_thresholds_hit.add(threshold)

        return TradeDecisionItem(
            instrument=InstrumentRef(symbol=symbol),
            action=TradeDecisionAction.OPEN_LONG,
            target_qty=buy_notional / current_price,
            leverage=1.0,
            confidence=0.90,
            rationale=(
                f"{self._profile.display_name}: RSI ladder entry {triggered} "
                f"with {buy_notional:.2f} quote currency allocation"
            ),
        )

    def _build_add_item(
        self,
        *,
        context: ComposeContext,
        symbol: str,
        state: SymbolStrategyState,
        symbol_features: Dict[str, Dict[str, object]],
        primary: Dict[str, object],
        current_qty: float,
        current_price: float,
        available_cash: float,
    ) -> TradeDecisionItem | None:
        if current_qty <= self._quantity_precision:
            return None
        if state.add_count >= self._profile.max_additions:
            return None
        if not _value_to_bool(primary, "bb_mid_cross_up"):
            return None
        if self._profile.add_requires_trend_up and not self._ma_is_up(
            primary, self._profile.ma_field
        ):
            return None
        if self._profile.add_requires_no_bear_trend_4h and self._has_clear_bearish_trend(
            symbol_features.get("4h")
        ):
            return None

        add_notional = max(0.0, available_cash) * ADD_BUY_RATIO
        if add_notional <= 0:
            return None

        state.add_count += 1
        return TradeDecisionItem(
            instrument=InstrumentRef(symbol=symbol),
            action=TradeDecisionAction.OPEN_LONG,
            target_qty=add_notional / current_price,
            leverage=1.0,
            confidence=0.75,
            rationale=(
                f"{self._profile.display_name}: Bollinger mid-line add #{state.add_count}"
            ),
        )

    def _build_sell_item(
        self,
        *,
        symbol: str,
        state: SymbolStrategyState,
        primary: Dict[str, object],
        current_qty: float,
        current_price: float,
    ) -> TradeDecisionItem | None:
        if current_qty <= self._quantity_precision:
            return None

        close_price = _value_to_float(primary, "close")
        ma_value = _value_to_float(primary, self._profile.ma_field)
        current_rsi = _value_to_float(primary, "rsi")
        if close_price is None or ma_value is None or current_rsi is None:
            return None

        final_sell_threshold = self._profile.sell_rsi_thresholds[-1]
        penultimate_sell_threshold = self._profile.sell_rsi_thresholds[-2]

        if state.tail_stop_armed and state.tail_peak_price is not None:
            state.tail_peak_price = max(state.tail_peak_price, current_price)
            if (
                current_rsi < final_sell_threshold
                and current_price <= state.tail_peak_price * (1.0 - TAIL_DRAWDOWN_RATIO)
            ):
                state.reset_on_flat()
                return TradeDecisionItem(
                    instrument=InstrumentRef(symbol=symbol),
                    action=TradeDecisionAction.CLOSE_LONG,
                    target_qty=current_qty,
                    leverage=1.0,
                    confidence=0.95,
                    rationale=(
                        f"{self._profile.display_name}: tail position drawdown stop"
                    ),
                )

        sell_thresholds = self._profile.sell_rsi_thresholds
        sell_cumulative_ratios = self._profile.sell_cumulative_ratios
        if close_price <= ma_value or current_rsi < sell_thresholds[0]:
            return None

        if state.exit_basis_qty is None or current_qty > state.exit_basis_qty:
            state.exit_basis_qty = current_qty

        reached_thresholds = [
            threshold
            for threshold in sell_thresholds
            if current_rsi >= threshold and threshold not in state.exit_thresholds_hit
        ]
        if not reached_thresholds:
            return None

        highest = max(reached_thresholds)
        target_remaining_ratio = 1.0 - sell_cumulative_ratios[highest]
        target_remaining_qty = max(0.0, state.exit_basis_qty * target_remaining_ratio)
        sell_qty = max(0.0, current_qty - target_remaining_qty)
        if sell_qty <= self._quantity_precision:
            return None

        for threshold in sell_thresholds:
            if threshold <= highest:
                state.exit_thresholds_hit.add(threshold)

        if (
            highest >= penultimate_sell_threshold
            and highest < final_sell_threshold
            and (current_qty - sell_qty) > self._quantity_precision
        ):
            state.tail_stop_armed = True
            state.tail_peak_price = current_price
        else:
            state.tail_stop_armed = False
            state.tail_peak_price = None

        return TradeDecisionItem(
            instrument=InstrumentRef(symbol=symbol),
            action=TradeDecisionAction.CLOSE_LONG,
            target_qty=sell_qty,
            leverage=1.0,
            confidence=0.92,
            rationale=(
                f"{self._profile.display_name}: RSI ladder exit to stage {highest}"
            ),
        )
