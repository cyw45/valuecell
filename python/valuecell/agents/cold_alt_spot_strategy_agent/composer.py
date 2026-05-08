"""Rule-based cold alt spot strategy composer."""

from __future__ import annotations

from typing import Dict, Literal

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
    AR_WARM_THRESHOLD,
    BR_WARM_THRESHOLD,
    CAPITAL_FRACTION,
    CONFIRM_INTERVAL,
    FINAL_EXIT_CUMULATIVE_EXIT_RATIO,
    ICE_RSI_MAX,
    ICE_RSI_MIN,
    MAX_CONCURRENT_POSITIONS,
    MAX_SINGLE_DAY_VOLATILITY_RATIO,
    NEW_ADD_ALLOC_RATIO,
    NEW_COIN_MAX_DAYS,
    NEW_COIN_MIN_4H_BARS,
    NEW_INITIAL_ALLOC_RATIO,
    NEW_MAX_EXPOSURE_RATIO,
    NEW_SIDEWAYS_RANGE_MAX_RATIO,
    OLD_ADD_ALLOC_RATIO,
    OLD_COIN_MIN_DAYS,
    OLD_INITIAL_ALLOC_RATIO,
    OLD_MAX_EXPOSURE_RATIO,
    OLD_SIDEWAYS_RANGE_MAX_RATIO,
    PRIMARY_INTERVAL,
    RISK_FILTER_INTERVAL,
    SCREEN_INTERVAL,
    STAGE1_CUMULATIVE_EXIT_RATIO,
    STAGE2_CUMULATIVE_EXIT_RATIO,
    TAKE_PROFIT_RSI,
    TIME_STOP_DAYS,
    VOLUME_CONTRACTION_MAX_RATIO,
)
from .state import ColdAltSymbolState

CoinClass = Literal["old", "new"]


def _value_to_float(mapping: Dict[str, object], key: str) -> float | None:
    value = mapping.get(key)
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def _value_to_bool(mapping: Dict[str, object], key: str) -> bool:
    return bool(mapping.get(key))


class ColdAltSpotComposer(BaseComposer):
    """Executes the cold alt spot strategy using fixed rules only."""

    def __init__(self, request: UserRequest) -> None:
        super().__init__(request, default_slippage_bps=15)
        self._state: Dict[str, ColdAltSymbolState] = {}

    async def compose(self, context: ComposeContext) -> ComposeResult:
        feature_map = self._build_feature_map(context)
        price_map = self._build_price_map(context, feature_map)
        plan = TradePlanProposal(items=[], rationale="No actionable signals")
        rationales: list[str] = []

        active_symbols = sum(
            1
            for snap in context.portfolio.positions.values()
            if abs(float(snap.quantity)) > self._quantity_precision
        )

        for symbol in self._request.trading_config.symbols:
            symbol_features = feature_map.get(symbol)
            if not symbol_features:
                continue

            item = self._build_symbol_decision(
                context=context,
                symbol=symbol,
                symbol_features=symbol_features,
                price_map=price_map,
                active_symbols=active_symbols,
            )
            if item is None:
                continue
            plan.items.append(item)
            if item.rationale:
                rationales.append(item.rationale)
            if item.action == TradeDecisionAction.OPEN_LONG:
                active_symbols += 1

        plan.rationale = " | ".join(rationales) if rationales else "No actionable signals"
        return ComposeResult(
            instructions=self._normalize_plan(context, plan),
            rationale=plan.rationale,
        )

    def _build_feature_map(
        self,
        context: ComposeContext,
    ) -> Dict[str, Dict[str, Dict[str, object]]]:
        feature_map: Dict[str, Dict[str, Dict[str, object]]] = {}
        for feature in context.features:
            interval = str((feature.meta or {}).get("interval") or "")
            if not interval:
                continue
            feature_map.setdefault(feature.instrument.symbol, {})[interval] = feature.values
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
            primary = by_interval.get(PRIMARY_INTERVAL, {})
            close_price = _value_to_float(primary, "close")
            if close_price is not None:
                price_map[symbol] = close_price
        return price_map

    def _portfolio_equity(self, context: ComposeContext) -> float:
        if context.portfolio.total_value is not None:
            return max(0.0, float(context.portfolio.total_value))
        return max(0.0, float(context.portfolio.account_balance or 0.0))

    def _current_quantity(self, context: ComposeContext, symbol: str) -> float:
        position = context.portfolio.positions.get(symbol)
        if position is None:
            return 0.0
        return max(0.0, float(position.quantity))

    def _coin_class(
        self,
        symbol_features: Dict[str, Dict[str, object]],
    ) -> CoinClass | None:
        day_values = symbol_features.get(SCREEN_INTERVAL)
        four_hour = symbol_features.get(PRIMARY_INTERVAL)
        if not day_values or not four_hour:
            return None

        day_count = int(day_values.get("count") or 0)
        four_hour_count = int(four_hour.get("count") or 0)
        if day_count >= OLD_COIN_MIN_DAYS:
            return "old"
        if 3 <= day_count <= NEW_COIN_MAX_DAYS and four_hour_count >= NEW_COIN_MIN_4H_BARS:
            return "new"
        return None

    def _passes_old_screen(self, screen_values: Dict[str, object]) -> bool:
        sideways = _value_to_float(screen_values, "sideways_range_ratio_14")
        contraction = _value_to_float(screen_values, "volume_contraction_ratio_14v90")
        change_pct = _value_to_float(screen_values, "change_pct")
        if sideways is None or contraction is None or change_pct is None:
            return False
        return (
            sideways <= OLD_SIDEWAYS_RANGE_MAX_RATIO
            and contraction <= VOLUME_CONTRACTION_MAX_RATIO
            and abs(change_pct) < MAX_SINGLE_DAY_VOLATILITY_RATIO
        )

    def _passes_new_screen(
        self,
        primary: Dict[str, object],
        risk_filter: Dict[str, object] | None,
    ) -> bool:
        sideways = _value_to_float(primary, "sideways_range_ratio_14")
        contraction = _value_to_float(primary, "volume_contraction_ratio_14v90")
        if sideways is None or contraction is None:
            return False
        if sideways > NEW_SIDEWAYS_RANGE_MAX_RATIO or contraction > VOLUME_CONTRACTION_MAX_RATIO:
            return False
        if risk_filter and (
            _value_to_bool(risk_filter, "launch_rocket_15m")
            or int(risk_filter.get("early_upper_shadow_traps") or 0) >= 2
        ):
            return False
        return True

    def _entry_signal(
        self,
        primary: Dict[str, object],
        confirm: Dict[str, object] | None,
    ) -> bool:
        current_rsi = _value_to_float(primary, "rsi")
        current_ar = _value_to_float(primary, "ar26")
        current_br = _value_to_float(primary, "br26")
        if current_rsi is None or current_ar is None or current_br is None:
            return False
        if not (ICE_RSI_MIN <= current_rsi <= ICE_RSI_MAX):
            return False
        if int(primary.get("rsi_ice_zone_bars") or 0) < 2:
            return False
        if not _value_to_bool(primary, "rsi_turn_up"):
            return False
        if not _value_to_bool(primary, "rsi_bottom_divergence"):
            return False
        if not (
            _value_to_bool(primary, "price_in_lower_half")
            and _value_to_bool(primary, "price_near_lower_band")
            and _value_to_bool(primary, "bb_squeeze_3bar")
        ):
            return False
        if not (
            (
                _value_to_bool(primary, "mtm_cross_up_zero")
                or (
                    _value_to_float(primary, "mtm14") is not None
                    and _value_to_float(primary, "mtm14") > 0
                    and int(primary.get("mtm_positive_bars") or 0) >= 1
                )
            )
            and _value_to_bool(primary, "mtm_turn_up")
        ):
            return False
        if not (
            current_ar >= AR_WARM_THRESHOLD
            and current_br >= BR_WARM_THRESHOLD
            and _value_to_bool(primary, "arbr_warm_up")
        ):
            return False
        if confirm is None:
            return False
        return (
            _value_to_bool(confirm, "rsi_turn_up")
            and _value_to_bool(confirm, "mtm_turn_up")
            and _value_to_bool(confirm, "close_turn_up")
        )

    def _can_add(
        self,
        primary: Dict[str, object],
        confirm: Dict[str, object] | None,
        current_price: float,
        avg_price: float | None,
    ) -> bool:
        if avg_price is not None and current_price < avg_price:
            return False
        if not _value_to_bool(primary, "close_turn_up"):
            return False
        if not _value_to_bool(primary, "mtm_turn_up"):
            return False
        if not _value_to_bool(primary, "arbr_warm_up"):
            return False
        if confirm is None:
            return False
        return _value_to_bool(confirm, "rsi_turn_up") and _value_to_bool(
            confirm, "mtm_turn_up"
        )

    def _stage_target_remaining_ratio(self, stage: int) -> float:
        if stage <= 1:
            return 1.0 - STAGE1_CUMULATIVE_EXIT_RATIO
        if stage == 2:
            return 1.0 - STAGE2_CUMULATIVE_EXIT_RATIO
        return 1.0 - FINAL_EXIT_CUMULATIVE_EXIT_RATIO

    def _stop_signal(
        self,
        context: ComposeContext,
        position,
        primary: Dict[str, object],
    ) -> bool:
        current_rsi = _value_to_float(primary, "rsi")
        if _value_to_bool(primary, "below_lower_band_2bar"):
            return True
        if current_rsi is not None and current_rsi < ICE_RSI_MIN and not _value_to_bool(
            primary, "rsi_turn_up"
        ):
            return True
        entry_ts = getattr(position, "entry_ts", None)
        if entry_ts:
            elapsed_days = max(0.0, (context.ts - int(entry_ts)) / 86_400_000)
            if elapsed_days >= TIME_STOP_DAYS:
                mtm = _value_to_float(primary, "mtm14")
                if (
                    (mtm is None or mtm <= 0)
                    and not _value_to_bool(primary, "close_turn_up")
                ):
                    return True
        return False

    def _build_symbol_decision(
        self,
        *,
        context: ComposeContext,
        symbol: str,
        symbol_features: Dict[str, Dict[str, object]],
        price_map: Dict[str, float],
        active_symbols: int,
    ) -> TradeDecisionItem | None:
        coin_class = self._coin_class(symbol_features)
        if coin_class is None:
            return None

        state = self._state.setdefault(symbol, ColdAltSymbolState())
        if state.blacklisted:
            return None

        screen_values = symbol_features.get(SCREEN_INTERVAL)
        primary = symbol_features.get(PRIMARY_INTERVAL)
        confirm = symbol_features.get(CONFIRM_INTERVAL)
        risk_filter = symbol_features.get(RISK_FILTER_INTERVAL)
        if not screen_values or not primary:
            return None

        current_price = price_map.get(symbol) or _value_to_float(primary, "close")
        if current_price is None or current_price <= 0:
            return None

        if coin_class == "old":
            if not self._passes_old_screen(screen_values):
                return None
        else:
            if risk_filter and (
                _value_to_bool(risk_filter, "launch_rocket_15m")
                or int(risk_filter.get("early_upper_shadow_traps") or 0) >= 2
            ):
                state.blacklisted = True
                return None
            if not self._passes_new_screen(primary, risk_filter):
                return None

        current_qty = self._current_quantity(context, symbol)
        position = context.portfolio.positions.get(symbol)
        if current_qty <= self._quantity_precision:
            state.reset_on_flat()

        if current_qty > self._quantity_precision and position is not None:
            stop_item = self._build_stop_or_exit_item(
                context=context,
                symbol=symbol,
                state=state,
                primary=primary,
                current_qty=current_qty,
                current_price=current_price,
                position=position,
            )
            if stop_item is not None:
                return stop_item

            add_item = self._build_add_item(
                context=context,
                symbol=symbol,
                coin_class=coin_class,
                state=state,
                primary=primary,
                confirm=confirm,
                current_qty=current_qty,
                current_price=current_price,
                avg_price=getattr(position, "avg_price", None),
            )
            if add_item is not None:
                return add_item
            return None

        if active_symbols >= MAX_CONCURRENT_POSITIONS:
            return None
        if not self._entry_signal(primary, confirm):
            return None

        equity = self._portfolio_equity(context)
        initial_ratio = (
            OLD_INITIAL_ALLOC_RATIO if coin_class == "old" else NEW_INITIAL_ALLOC_RATIO
        )
        buy_notional = equity * initial_ratio
        if buy_notional <= 0:
            return None

        return TradeDecisionItem(
            instrument=InstrumentRef(symbol=symbol),
            action=TradeDecisionAction.OPEN_LONG,
            target_qty=buy_notional / current_price,
            leverage=1.0,
            confidence=0.92,
            rationale=(
                f"Cold alt spot {coin_class} entry: Bollinger/RSI/MTM/ARBR resonance"
            ),
        )

    def _build_add_item(
        self,
        *,
        context: ComposeContext,
        symbol: str,
        coin_class: CoinClass,
        state: ColdAltSymbolState,
        primary: Dict[str, object],
        confirm: Dict[str, object] | None,
        current_qty: float,
        current_price: float,
        avg_price: float | None,
    ) -> TradeDecisionItem | None:
        if state.add_completed:
            return None
        if not self._can_add(primary, confirm, current_price, avg_price):
            return None

        equity = self._portfolio_equity(context)
        max_exposure_ratio = (
            OLD_MAX_EXPOSURE_RATIO if coin_class == "old" else NEW_MAX_EXPOSURE_RATIO
        )
        add_ratio = OLD_ADD_ALLOC_RATIO if coin_class == "old" else NEW_ADD_ALLOC_RATIO
        current_notional = current_qty * current_price
        remaining_capacity = max(0.0, equity * max_exposure_ratio - current_notional)
        add_notional = min(equity * add_ratio, remaining_capacity)
        if add_notional <= 0:
            return None

        state.add_completed = True
        return TradeDecisionItem(
            instrument=InstrumentRef(symbol=symbol),
            action=TradeDecisionAction.OPEN_LONG,
            target_qty=add_notional / current_price,
            leverage=1.0,
            confidence=0.84,
            rationale=f"Cold alt spot {coin_class} add: secondary resonance confirmed",
        )

    def _build_stop_or_exit_item(
        self,
        *,
        context: ComposeContext,
        symbol: str,
        state: ColdAltSymbolState,
        primary: Dict[str, object],
        current_qty: float,
        current_price: float,
        position,
    ) -> TradeDecisionItem | None:
        if self._stop_signal(context, position, primary):
            state.reset_on_flat()
            return TradeDecisionItem(
                instrument=InstrumentRef(symbol=symbol),
                action=TradeDecisionAction.CLOSE_LONG,
                target_qty=current_qty,
                leverage=1.0,
                confidence=0.95,
                rationale="Cold alt spot stop: technical breakdown or time stop",
            )

        if state.exit_basis_qty is None or current_qty > state.exit_basis_qty:
            state.exit_basis_qty = current_qty

        current_rsi = _value_to_float(primary, "rsi")
        stage = 0
        if _value_to_bool(primary, "arbr_cooling_off") or _value_to_bool(
            primary, "below_middle_band"
        ) or _value_to_bool(primary, "bb_opening_down"):
            stage = 3
        elif _value_to_bool(primary, "mtm_weak_2bar") or _value_to_bool(
            primary, "rsi_top_divergence"
        ):
            stage = 2
        elif _value_to_bool(primary, "upper_band_rejection") or (
            current_rsi is not None and current_rsi >= TAKE_PROFIT_RSI
        ):
            stage = 1

        if stage <= state.exit_stage or stage == 0:
            return None

        state.exit_stage = stage
        if stage >= 3:
            state.reset_on_flat()
            return TradeDecisionItem(
                instrument=InstrumentRef(symbol=symbol),
                action=TradeDecisionAction.CLOSE_LONG,
                target_qty=current_qty,
                leverage=1.0,
                confidence=0.93,
                rationale="Cold alt spot final exit: sentiment and trend rolled over",
            )

        target_remaining_ratio = self._stage_target_remaining_ratio(stage)
        target_remaining_qty = max(0.0, state.exit_basis_qty * target_remaining_ratio)
        sell_qty = max(0.0, current_qty - target_remaining_qty)
        if sell_qty <= self._quantity_precision:
            return None

        return TradeDecisionItem(
            instrument=InstrumentRef(symbol=symbol),
            action=TradeDecisionAction.CLOSE_LONG,
            target_qty=sell_qty,
            leverage=1.0,
            confidence=0.88,
            rationale=f"Cold alt spot staged exit: stage {stage}",
        )
