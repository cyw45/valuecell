"""Pure, deterministic paper-only crypto rule evaluator with no market-data access."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from valuecell.server.api.schemas.rule_strategy import (
    RuleStrategyConditionCheck,
    RuleStrategyEvaluationRequest,
    RuleStrategyEvaluationResult,
    RuleStrategyEntryConfirmation,
    RuleStrategyFundingImpact,
    RuleStrategyIndicatorValues,
    RuleStrategySizing,
)

Signal = Literal["buy", "sell", "neutral", "unavailable"]


@dataclass(frozen=True)
class _SignalAssessment:
    code: str
    signal: Signal
    check: RuleStrategyConditionCheck


class RuleEngine:
    """Evaluate supplied OHLCV and account snapshots without side effects."""

    def evaluate(
        self, request: RuleStrategyEvaluationRequest
    ) -> RuleStrategyEvaluationResult:
        if request.config.advanced_rules.enabled:
            return self._evaluate_advanced(request)

        closes = [candle.close for candle in request.candles]
        config = request.config
        market = request.market
        indicators = RuleStrategyIndicatorValues()
        assessments: list[_SignalAssessment] = []

        if config.moving_average.enabled:
            assessment, values = self._moving_average_assessment(
                closes,
                config.moving_average.short_window,
                config.moving_average.long_window,
            )
            indicators = indicators.model_copy(update=values)
            assessments.append(assessment)
        if config.rsi.enabled:
            assessment, values = self._rsi_assessment(
                closes, config.rsi.period, config.rsi.oversold, config.rsi.overbought
            )
            indicators = indicators.model_copy(update=values)
            assessments.append(assessment)
        if config.bollinger.enabled:
            assessment, values = self._bollinger_assessment(
                closes, config.bollinger.period, config.bollinger.standard_deviations
            )
            indicators = indicators.model_copy(update=values)
            assessments.append(assessment)
        if config.momentum_macd.enabled:
            assessment, values = self._momentum_macd_assessment(
                closes,
                config.momentum_macd.momentum_period,
                config.momentum_macd.macd_fast_window,
                config.momentum_macd.macd_slow_window,
                config.momentum_macd.macd_signal_window,
            )
            indicators = indicators.model_copy(update=values)
            assessments.append(assessment)

        conditions = [assessment.check for assessment in assessments]
        sizing = self._sizing(request)
        entry_side = self._confirmed_side(assessments, config.confirmation_mode)
        _, entry_confirmation = self._entry_confirmation(
            assessments, config.confirmation_mode
        )
        exit_conditions, exit_reason = self._exit_conditions(request)
        conditions.extend(exit_conditions)

        if market.position.quantity > 0:
            action, reason_code, reason = self._position_action(
                entry_side, exit_reason, config.confirmation_mode, assessments
            )
        else:
            action, reason_code, reason = self._flat_action(
                entry_side, assessments, config.confirmation_mode
            )

        risk_conditions = self._risk_conditions(request, action, sizing)
        conditions.extend(risk_conditions)
        if action == "buy":
            block = next(
                (check for check in risk_conditions if check.state == "blocked"), None
            )
            if block is not None:
                action = "no_op"
                reason_code = block.code
                reason = block.detail

        funding = self._funding(request, action, sizing)
        return RuleStrategyEvaluationResult(
            action=action,
            reason_code=reason_code,
            reason=reason,
            conditions=conditions,
            indicators=indicators,
            sizing=sizing,
            funding=funding,
            entry_confirmation=entry_confirmation,
        )

    def _evaluate_advanced(
        self, request: RuleStrategyEvaluationRequest
    ) -> RuleStrategyEvaluationResult:
        """Evaluate independently configured multi-timeframe indicator rules."""
        rules = request.config.advanced_rules
        indicators = RuleStrategyIndicatorValues()
        entry_assessments: list[_SignalAssessment] = []
        exit_assessments: list[_SignalAssessment] = []

        if rules.moving_average.enabled:
            assessment, values = self._advanced_ma_assessment(
                self._candles_for(request, rules.moving_average.interval),
                rules.moving_average.period,
                rules.moving_average.entry_comparator,
                rules.moving_average.interval,
            )
            entry_assessments.append(assessment)
            indicators = indicators.model_copy(update=values)
        if rules.macd.enabled:
            assessment, values = self._advanced_macd_assessment(
                self._candles_for(request, rules.macd.interval),
                rules.macd.fast_window,
                rules.macd.slow_window,
                rules.macd.signal_window,
                rules.macd.entry_cross,
                rules.macd.interval,
            )
            entry_assessments.append(assessment)
            indicators = indicators.model_copy(update=values)
        if rules.bollinger.enabled:
            assessment, values = self._advanced_bollinger_assessment(
                self._candles_for(request, rules.bollinger.interval),
                rules.bollinger.period,
                rules.bollinger.standard_deviations,
                rules.bollinger.entry_reference,
                rules.bollinger.entry_comparator,
                rules.bollinger.interval,
            )
            entry_assessments.append(assessment)
            indicators = indicators.model_copy(update=values)

        rsi_entries, rsi_exits, rsi_values = self._advanced_rsi_assessments(
            request, rules.rsi
        )
        entry_assessments.extend(rsi_entries)
        exit_assessments.extend(rsi_exits)
        indicators = indicators.model_copy(update=rsi_values)

        momentum_entries, momentum_exits, momentum_values = (
            self._advanced_momentum_assessments(request, rules.momentum)
        )
        entry_assessments.extend(momentum_entries)
        exit_assessments.extend(momentum_exits)
        indicators = indicators.model_copy(update=momentum_values)

        brar_entries, brar_exits, brar_values = self._advanced_brar_assessments(
            request, rules.brar
        )
        entry_assessments.extend(brar_entries)
        exit_assessments.extend(brar_exits)
        indicators = indicators.model_copy(update=brar_values)

        conditions = [assessment.check for assessment in entry_assessments]
        conditions.extend(assessment.check for assessment in exit_assessments)
        sizing = self._sizing(request)
        entry_side, entry_confirmation = self._entry_confirmation(
            entry_assessments,
            rules.entry_confirmation_mode,
            rules.entry_confirmation_count,
            rules.entry_confirmation_ratio,
        )
        exit_side = self._confirmed_side(exit_assessments, rules.exit_confirmation_mode)
        risk_exit_conditions, risk_exit_reason = self._exit_conditions(request)
        conditions.extend(risk_exit_conditions)

        if request.market.position.quantity > 0:
            if risk_exit_reason is not None:
                action, reason_code, reason = self._position_action(
                    "neutral", risk_exit_reason, rules.exit_confirmation_mode, []
                )
            elif exit_side == "sell":
                action = "sell"
                reason_code = "advanced_exit_confirmed"
                reason = "Sell recommendation: configured exit rules are confirmed."
            elif exit_side == "unavailable":
                action = "no_op"
                reason_code = "insufficient_candle_history"
                reason = "No action: supplied candle history is insufficient."
            else:
                action = "no_op"
                reason_code = "no_exit_signal"
                reason = "No action: configured exit rules are not confirmed."
        elif entry_side == "buy":
            action = "buy"
            reason_code = "advanced_entry_confirmed"
            reason = (
                "Buy recommendation: configured multi-timeframe rules are confirmed."
            )
        elif entry_side == "unavailable":
            action = "no_op"
            reason_code = "insufficient_candle_history"
            reason = "No action: supplied candle history is insufficient."
        else:
            action = "no_op"
            reason_code = "advanced_entry_not_confirmed"
            reason = (
                "No action: configured multi-timeframe entry rules are not confirmed."
            )

        risk_conditions = self._risk_conditions(request, action, sizing)
        conditions.extend(risk_conditions)
        if action == "buy":
            block = next(
                (check for check in risk_conditions if check.state == "blocked"), None
            )
            if block is not None:
                action = "no_op"
                reason_code = block.code
                reason = block.detail
        funding = self._funding(request, action, sizing)
        return RuleStrategyEvaluationResult(
            action=action,
            reason_code=reason_code,
            reason=reason,
            conditions=conditions,
            indicators=indicators,
            sizing=sizing,
            funding=funding,
            entry_confirmation=entry_confirmation,
        )

    @staticmethod
    def _candles_for(request: RuleStrategyEvaluationRequest, interval: str):
        # The generic candles belong only to the configured primary interval.
        # Never evaluate another timeframe against unrelated primary candles.
        if interval == request.config.interval:
            return request.candles
        return request.candle_sets.get(interval, [])

    def _advanced_ma_assessment(
        self, candles, period: int, comparator: str, interval: str
    ) -> tuple[_SignalAssessment, dict[str, float]]:
        if len(candles) < period:
            return self._unavailable("price_ma", "indicator", period, len(candles)), {}
        average = self._sma([candle.close for candle in candles[-period:]])
        price = candles[-1].close
        matched = price >= average if comparator == "above" else price <= average
        detail = (
            f"price is {comparator} the {period}-period moving average on {interval}"
        )
        return self._assessment(
            "price_ma",
            "buy" if matched else "neutral",
            detail,
            {"price": price, "moving_average_long": average, "interval": interval},
        ), {"moving_average_long": average}

    def _advanced_macd_assessment(
        self,
        candles,
        fast_window: int,
        slow_window: int,
        signal_window: int,
        cross: str,
        interval: str,
    ) -> tuple[_SignalAssessment, dict[str, float]]:
        required = slow_window + signal_window
        if len(candles) < required:
            return self._unavailable(
                "macd_cross", "indicator", required, len(candles)
            ), {}
        closes = [candle.close for candle in candles]
        fast = self._ema_series(closes, fast_window)
        slow = self._ema_series(closes, slow_window)
        macd_series = [left - right for left, right in zip(fast, slow)]
        signal_series = self._ema_series(macd_series, signal_window)
        previous_macd, macd = macd_series[-2], macd_series[-1]
        previous_signal, signal = signal_series[-2], signal_series[-1]
        golden = previous_macd <= previous_signal and macd > signal
        death = previous_macd >= previous_signal and macd < signal
        matched = golden if cross == "golden" else death
        values = {
            "macd": macd,
            "macd_signal": signal,
            "previous_macd": previous_macd,
            "previous_macd_signal": previous_signal,
            "interval": interval,
        }
        return self._assessment(
            "macd_cross",
            "buy" if matched else "neutral",
            f"{cross} MACD crossover on {interval}",
            values,
        ), values

    def _advanced_bollinger_assessment(
        self,
        candles,
        period: int,
        multiplier: float,
        reference: str,
        comparator: str,
        interval: str,
    ) -> tuple[_SignalAssessment, dict[str, float]]:
        if len(candles) < period:
            return self._unavailable(
                "bollinger_price", "indicator", period, len(candles)
            ), {}
        closes = [candle.close for candle in candles[-period:]]
        middle = self._sma(closes)
        deviation = math.sqrt(sum((close - middle) ** 2 for close in closes) / period)
        references = {
            "upper": middle + multiplier * deviation,
            "middle": middle,
            "lower": middle - multiplier * deviation,
        }
        price = candles[-1].close
        target = references[reference]
        matched = price >= target if comparator == "above" else price <= target
        values = {
            "bollinger_upper": references["upper"],
            "bollinger_middle": middle,
            "bollinger_lower": references["lower"],
            "price": price,
            "interval": interval,
        }
        return self._assessment(
            "bollinger_price",
            "buy" if matched else "neutral",
            f"price is {comparator} Bollinger {reference} band on {interval}",
            values,
        ), values

    def _advanced_rsi_assessments(self, request, rule):
        if not rule.enabled:
            return [], [], {}
        candles = self._candles_for(request, rule.interval)
        required = rule.period + 1
        if len(candles) < required:
            assessment = self._unavailable(
                "rsi_entry", "indicator", required, len(candles)
            )
            return [assessment], [assessment] if rule.exit_enabled else [], {}
        closes = [candle.close for candle in candles]
        rsi = self._rsi_value(closes, rule.period)
        return self._threshold_assessments(
            "rsi", rsi, rule, {"rsi": rsi, "interval": rule.interval}
        )

    def _advanced_momentum_assessments(self, request, rule):
        if not rule.enabled:
            return [], [], {}
        candles = self._candles_for(request, rule.interval)
        required = rule.period + 1
        if len(candles) < required:
            assessment = self._unavailable(
                "momentum_entry", "indicator", required, len(candles)
            )
            return [assessment], [assessment] if rule.exit_enabled else [], {}
        momentum = candles[-1].close - candles[-rule.period - 1].close
        return self._threshold_assessments(
            "momentum",
            momentum,
            rule,
            {"momentum": momentum, "interval": rule.interval},
        )

    def _advanced_brar_assessments(self, request, rule):
        if not rule.enabled:
            return [], [], {}
        candles = self._candles_for(request, rule.interval)
        required = rule.period + 1
        if len(candles) < required:
            assessment = self._unavailable(
                "brar_entry", "indicator", required, len(candles)
            )
            return [assessment], [assessment] if rule.exit_enabled else [], {}
        window = candles[-rule.period :]
        previous = candles[-rule.period - 1 : -1]
        ar_numerator = sum(candle.high - candle.open for candle in window)
        ar_denominator = sum(candle.open - candle.low for candle in window)
        br_numerator = sum(
            max(candle.high - prior.close, 0.0)
            for candle, prior in zip(window, previous)
        )
        br_denominator = sum(
            max(prior.close - candle.low, 0.0)
            for candle, prior in zip(window, previous)
        )
        ar = 100.0 if ar_denominator == 0 else ar_numerator / ar_denominator * 100
        br = 100.0 if br_denominator == 0 else br_numerator / br_denominator * 100
        value = ar if rule.component == "ar" else br
        return self._threshold_assessments(
            "brar",
            value,
            rule,
            {"brar_ar": ar, "brar_br": br, "interval": rule.interval},
        )

    def _threshold_assessments(self, code, value, rule, values):
        entry_matched = self._matches_threshold(
            value, rule.entry_comparator, rule.entry_threshold
        )
        entry = self._assessment(
            f"{code}_entry",
            "buy" if entry_matched else "neutral",
            f"{code} is {rule.entry_comparator} the entry threshold",
            values,
        )
        if not rule.exit_enabled:
            return [entry], [], values
        exit_matched = self._matches_threshold(
            value, rule.exit_comparator, rule.exit_threshold
        )
        exit_assessment = self._assessment(
            f"{code}_exit",
            "sell" if exit_matched else "neutral",
            f"{code} is {rule.exit_comparator} the exit threshold",
            values,
        )
        return [entry], [exit_assessment], values

    @staticmethod
    def _matches_threshold(value: float, comparator: str, threshold: float) -> bool:
        return value >= threshold if comparator == "above" else value <= threshold

    @staticmethod
    def _rsi_value(closes: list[float], period: int) -> float:
        changes = [
            current - previous
            for previous, current in zip(closes[-period - 1 :], closes[-period:])
        ]
        average_gain = sum(max(change, 0.0) for change in changes) / period
        average_loss = sum(max(-change, 0.0) for change in changes) / period
        if average_loss == 0:
            return 100.0 if average_gain > 0 else 50.0
        return 100.0 - 100.0 / (1.0 + average_gain / average_loss)

    def _moving_average_assessment(
        self, closes: list[float], short_window: int, long_window: int
    ) -> tuple[_SignalAssessment, dict[str, float]]:
        required = long_window + 1
        if len(closes) < required:
            return self._unavailable(
                "ma_crossover", "indicator", required, len(closes)
            ), {}
        previous_short = self._sma(closes[-short_window - 1 : -1])
        previous_long = self._sma(closes[-long_window - 1 : -1])
        short = self._sma(closes[-short_window:])
        long = self._sma(closes[-long_window:])
        values = {
            "moving_average_short": short,
            "moving_average_long": long,
            "previous_moving_average_short": previous_short,
            "previous_moving_average_long": previous_long,
        }
        if previous_short <= previous_long and short > long:
            return self._assessment(
                "ma_crossover", "buy", "moving averages crossed upward", values
            ), values
        if previous_short >= previous_long and short < long:
            return self._assessment(
                "ma_crossover", "sell", "moving averages crossed downward", values
            ), values
        return self._assessment(
            "ma_crossover", "neutral", "no moving-average crossover", values
        ), values

    def _rsi_assessment(
        self, closes: list[float], period: int, oversold: float, overbought: float
    ) -> tuple[_SignalAssessment, dict[str, float]]:
        required = period + 1
        if len(closes) < required:
            return self._unavailable("rsi", "indicator", required, len(closes)), {}
        changes = [
            current - previous
            for previous, current in zip(closes[-required:], closes[-period:])
        ]
        average_gain = sum(max(change, 0.0) for change in changes) / period
        average_loss = sum(max(-change, 0.0) for change in changes) / period
        rsi = (
            100.0
            if average_loss == 0 and average_gain > 0
            else 50.0
            if average_loss == 0
            else 100.0 - 100.0 / (1.0 + average_gain / average_loss)
        )
        values = {"rsi": rsi}
        if rsi <= oversold:
            return self._assessment(
                "rsi", "buy", "RSI is at or below oversold threshold", values
            ), values
        if rsi >= overbought:
            return self._assessment(
                "rsi", "sell", "RSI is at or above overbought threshold", values
            ), values
        return self._assessment(
            "rsi", "neutral", "RSI is between configured thresholds", values
        ), values

    def _bollinger_assessment(
        self, closes: list[float], period: int, multiplier: float
    ) -> tuple[_SignalAssessment, dict[str, float]]:
        if len(closes) < period:
            return self._unavailable("bollinger", "indicator", period, len(closes)), {}
        window = closes[-period:]
        middle = self._sma(window)
        standard_deviation = math.sqrt(
            sum((close - middle) ** 2 for close in window) / period
        )
        upper = middle + multiplier * standard_deviation
        lower = middle - multiplier * standard_deviation
        values = {
            "bollinger_upper": upper,
            "bollinger_middle": middle,
            "bollinger_lower": lower,
        }
        if standard_deviation == 0:
            return self._assessment(
                "bollinger", "neutral", "Bollinger bands have zero width", values
            ), values
        if closes[-1] <= lower:
            return self._assessment(
                "bollinger", "buy", "close is at or below lower Bollinger band", values
            ), values
        if closes[-1] >= upper:
            return self._assessment(
                "bollinger", "sell", "close is at or above upper Bollinger band", values
            ), values
        return self._assessment(
            "bollinger", "neutral", "close is inside Bollinger bands", values
        ), values

    def _momentum_macd_assessment(
        self,
        closes: list[float],
        momentum_period: int,
        fast_window: int,
        slow_window: int,
        signal_window: int,
    ) -> tuple[_SignalAssessment, dict[str, float]]:
        required = max(momentum_period + 1, slow_window + signal_window)
        if len(closes) < required:
            return self._unavailable(
                "momentum_macd", "indicator", required, len(closes)
            ), {}
        fast = self._ema_series(closes, fast_window)
        slow = self._ema_series(closes, slow_window)
        macd_series = [
            fast_value - slow_value for fast_value, slow_value in zip(fast, slow)
        ]
        signal_series = self._ema_series(macd_series, signal_window)
        momentum = closes[-1] - closes[-momentum_period - 1]
        macd, previous_macd = macd_series[-1], macd_series[-2]
        signal, previous_signal = signal_series[-1], signal_series[-2]
        values = {
            "momentum": momentum,
            "macd": macd,
            "macd_signal": signal,
            "previous_macd": previous_macd,
            "previous_macd_signal": previous_signal,
        }
        if momentum > 0 and previous_macd <= previous_signal and macd > signal:
            return self._assessment(
                "momentum_macd",
                "buy",
                "positive momentum with upward MACD crossover",
                values,
            ), values
        if momentum < 0 and previous_macd >= previous_signal and macd < signal:
            return self._assessment(
                "momentum_macd",
                "sell",
                "negative momentum with downward MACD crossover",
                values,
            ), values
        return self._assessment(
            "momentum_macd", "neutral", "momentum/MACD entry conditions not met", values
        ), values

    def _exit_conditions(
        self, request: RuleStrategyEvaluationRequest
    ) -> tuple[list[RuleStrategyConditionCheck], str | None]:
        position = request.market.position
        risk = request.config.risk
        if position.quantity == 0:
            return [
                RuleStrategyConditionCheck(
                    code="take_profit",
                    category="exit",
                    state="not_triggered",
                    detail="no open position",
                ),
                RuleStrategyConditionCheck(
                    code="stop_loss",
                    category="exit",
                    state="not_triggered",
                    detail="no open position",
                ),
            ], None
        entry_price = position.entry_price
        if entry_price is None:
            # Exchange spot balances expose quantity but no trustworthy cost
            # basis. Keep indicator exits available while explicitly disabling
            # price-relative exits rather than inventing an entry price.
            return [
                RuleStrategyConditionCheck(
                    code="take_profit",
                    category="exit",
                    state="unavailable",
                    detail="position entry price is unavailable",
                ),
                RuleStrategyConditionCheck(
                    code="stop_loss",
                    category="exit",
                    state="unavailable",
                    detail="position entry price is unavailable",
                ),
            ], None
        return_pct = request.market.price / entry_price - 1.0
        take_profit_hit = (
            risk.take_profit_pct is not None and return_pct >= risk.take_profit_pct
        )
        stop_loss_hit = (
            risk.stop_loss_pct is not None and return_pct <= -risk.stop_loss_pct
        )
        conditions = [
            RuleStrategyConditionCheck(
                code="take_profit",
                category="exit",
                state="triggered" if take_profit_hit else "not_triggered",
                detail="take-profit threshold reached"
                if take_profit_hit
                else "take-profit threshold not reached",
                values={
                    "return_pct": return_pct,
                    "threshold_pct": risk.take_profit_pct,
                },
            ),
            RuleStrategyConditionCheck(
                code="stop_loss",
                category="exit",
                state="triggered" if stop_loss_hit else "not_triggered",
                detail="stop-loss threshold reached"
                if stop_loss_hit
                else "stop-loss threshold not reached",
                values={"return_pct": return_pct, "threshold_pct": risk.stop_loss_pct},
            ),
        ]
        return (
            conditions,
            "take_profit"
            if take_profit_hit
            else "stop_loss"
            if stop_loss_hit
            else None,
        )

    def _position_action(
        self,
        side: Signal,
        exit_reason: str | None,
        confirmation_mode: str,
        assessments: list[_SignalAssessment],
    ) -> tuple[Literal["sell", "no_op"], str, str]:
        if exit_reason == "take_profit":
            return (
                "sell",
                "take_profit_triggered",
                "Sell recommendation: take-profit threshold reached.",
            )
        if exit_reason == "stop_loss":
            return (
                "sell",
                "stop_loss_triggered",
                "Sell recommendation: stop-loss threshold reached.",
            )
        if side == "sell":
            return (
                "sell",
                "indicator_sell_confirmed",
                "Sell recommendation: configured indicators confirm a sell signal.",
            )
        if not assessments:
            return (
                "no_op",
                "no_enabled_indicators",
                "No action: no indicator rule is enabled.",
            )
        if side == "unavailable":
            return (
                "no_op",
                "insufficient_candle_history",
                "No action: supplied candle history is insufficient for configured indicators.",
            )
        return (
            "no_op",
            "no_exit_signal",
            "No action: no configured exit signal is confirmed.",
        )

    def _flat_action(
        self, side: Signal, assessments: list[_SignalAssessment], confirmation_mode: str
    ) -> tuple[Literal["buy", "no_op"], str, str]:
        if not assessments:
            return (
                "no_op",
                "no_enabled_indicators",
                "No action: no indicator rule is enabled.",
            )
        if side == "buy":
            return (
                "buy",
                "indicator_buy_confirmed",
                "Buy recommendation: configured indicators confirm a buy signal.",
            )
        if side == "sell":
            return (
                "no_op",
                "sell_signal_without_position",
                "No action: sell signal cannot open a paper long position.",
            )
        if side == "unavailable":
            return (
                "no_op",
                "insufficient_candle_history",
                "No action: supplied candle history is insufficient for configured indicators.",
            )
        return (
            "no_op",
            "indicators_not_confirmed",
            "No action: configured indicators do not confirm an entry signal.",
        )

    def _risk_conditions(
        self,
        request: RuleStrategyEvaluationRequest,
        action: str,
        sizing: RuleStrategySizing,
    ) -> list[RuleStrategyConditionCheck]:
        market, risk = request.market, request.config.risk
        position_limit_blocked = (
            action == "buy" and market.open_position_count >= risk.max_positions
        )
        capital_blocked = (
            action == "buy" and sizing.requested_quote > sizing.affordable_quote
        )
        leverage_blocked = (
            action == "buy" and sizing.requested_quote > sizing.max_allowed_quote
        )
        return [
            RuleStrategyConditionCheck(
                code="max_positions",
                category="risk",
                state="blocked" if position_limit_blocked else "not_triggered",
                detail="maximum open positions reached"
                if position_limit_blocked
                else "position limit permits an entry",
                values={
                    "open_position_count": market.open_position_count,
                    "max_positions": risk.max_positions,
                },
            ),
            RuleStrategyConditionCheck(
                code="available_collateral",
                category="risk",
                state="blocked" if capital_blocked else "not_triggered",
                detail="insufficient quote balance for configured leveraged size"
                if capital_blocked
                else "quote balance covers configured leveraged size",
                values={
                    "requested_quote": sizing.requested_quote,
                    "affordable_quote": sizing.affordable_quote,
                },
            ),
            RuleStrategyConditionCheck(
                code="leverage_limit",
                category="risk",
                state="blocked" if leverage_blocked else "not_triggered",
                detail="configured size exceeds equity-based leverage limit"
                if leverage_blocked
                else "configured size is within equity-based leverage limit",
                values={
                    "requested_quote": sizing.requested_quote,
                    "max_allowed_quote": sizing.max_allowed_quote,
                },
            ),
        ]

    def _sizing(self, request: RuleStrategyEvaluationRequest) -> RuleStrategySizing:
        market, risk = request.market, request.config.risk
        requested = risk.order_quote_amount
        return RuleStrategySizing(
            mode="fixed_quote",
            requested_quote=requested,
            max_allowed_quote=market.equity_quote * risk.leverage,
            affordable_quote=market.quote_balance * risk.leverage,
            quantity=requested / market.price,
        )

    def _funding(
        self,
        request: RuleStrategyEvaluationRequest,
        action: str,
        sizing: RuleStrategySizing,
    ) -> RuleStrategyFundingImpact:
        position = request.market.position
        current_notional = position.quantity * request.market.price
        projected_notional = (
            sizing.requested_quote
            if action == "buy"
            else 0.0
            if action == "sell"
            else current_notional
        )
        payment = -projected_notional * request.market.funding_rate
        return RuleStrategyFundingImpact(
            funding_rate=request.market.funding_rate,
            current_notional_quote=current_notional,
            projected_notional_quote=projected_notional,
            estimated_payment_quote=payment,
            direction="credit" if payment > 0 else "debit" if payment < 0 else "none",
        )

    @staticmethod
    def _confirmed_side(assessments: list[_SignalAssessment], mode: str) -> Signal:
        if not assessments:
            return "neutral"
        signals = [assessment.signal for assessment in assessments]
        if mode == "all":
            if any(signal == "unavailable" for signal in signals):
                return "unavailable"
            if all(signal == "buy" for signal in signals):
                return "buy"
            if all(signal == "sell" for signal in signals):
                return "sell"
            return "neutral"
        if "buy" in signals and "sell" not in signals:
            return "buy"
        if "sell" in signals and "buy" not in signals:
            return "sell"
        if all(signal == "unavailable" for signal in signals):
            return "unavailable"
        return "neutral"

    @staticmethod
    def _entry_confirmation(
        assessments: list[_SignalAssessment],
        mode: Literal["all", "any", "at_least", "ratio"],
        count: int = 1,
        ratio: float = 1.0,
    ) -> tuple[Signal, RuleStrategyEntryConfirmation]:
        enabled = len(assessments)
        available_assessments = [
            assessment for assessment in assessments if assessment.signal != "unavailable"
        ]
        available = len(available_assessments)
        passed = sum(assessment.signal == "buy" for assessment in available_assessments)
        required = (
            1 if mode == "any" else enabled if mode == "all" else count
            if mode == "at_least" else math.ceil(enabled * ratio)
        )
        summary = RuleStrategyEntryConfirmation(
            enabled=enabled,
            available=available,
            passed=passed,
            required=required,
            mode=mode,
        )
        if not assessments:
            return "neutral", summary
        if available < required:
            return "unavailable", summary
        if passed >= required:
            return "buy", summary
        return "neutral", summary

    @staticmethod
    def _assessment(
        code: str, signal: Signal, detail: str, values: dict[str, float]
    ) -> _SignalAssessment:
        return _SignalAssessment(
            code=code,
            signal=signal,
            check=RuleStrategyConditionCheck(
                code=code,
                category="indicator",
                state="triggered" if signal in {"buy", "sell"} else "not_triggered",
                detail=detail,
                values=values,
            ),
        )

    @staticmethod
    def _unavailable(
        code: str,
        category: Literal["indicator", "exit", "risk"],
        required: int,
        supplied: int,
    ) -> _SignalAssessment:
        return _SignalAssessment(
            code=code,
            signal="unavailable",
            check=RuleStrategyConditionCheck(
                code=code,
                category=category,
                state="unavailable",
                detail="insufficient supplied candle history",
                values={"required_candles": required, "supplied_candles": supplied},
            ),
        )

    @staticmethod
    def _sma(values: list[float]) -> float:
        return sum(values) / len(values)

    @staticmethod
    def _ema_series(values: list[float], window: int) -> list[float]:
        multiplier = 2.0 / (window + 1.0)
        output = [values[0]]
        for value in values[1:]:
            output.append((value - output[-1]) * multiplier + output[-1])
        return output
