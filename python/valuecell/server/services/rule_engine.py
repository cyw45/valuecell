"""Pure, deterministic paper-only crypto rule evaluator with no market-data access."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from valuecell.server.api.schemas.rule_strategy import (
    RuleStrategyConditionCheck,
    RuleStrategyEvaluationRequest,
    RuleStrategyEvaluationResult,
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
        closes = [candle.close for candle in request.candles]
        config = request.config
        market = request.market
        indicators = RuleStrategyIndicatorValues()
        assessments: list[_SignalAssessment] = []

        if config.moving_average.enabled:
            assessment, values = self._moving_average_assessment(
                closes, config.moving_average.short_window, config.moving_average.long_window
            )
            indicators = indicators.model_copy(update=values)
            assessments.append(assessment)
        if config.rsi.enabled:
            assessment, values = self._rsi_assessment(closes, config.rsi.period, config.rsi.oversold, config.rsi.overbought)
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
            block = next((check for check in risk_conditions if check.state == "blocked"), None)
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
        )

    def _moving_average_assessment(
        self, closes: list[float], short_window: int, long_window: int
    ) -> tuple[_SignalAssessment, dict[str, float]]:
        required = long_window + 1
        if len(closes) < required:
            return self._unavailable("ma_crossover", "indicator", required, len(closes)), {}
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
            return self._assessment("ma_crossover", "buy", "moving averages crossed upward", values), values
        if previous_short >= previous_long and short < long:
            return self._assessment("ma_crossover", "sell", "moving averages crossed downward", values), values
        return self._assessment("ma_crossover", "neutral", "no moving-average crossover", values), values

    def _rsi_assessment(
        self, closes: list[float], period: int, oversold: float, overbought: float
    ) -> tuple[_SignalAssessment, dict[str, float]]:
        required = period + 1
        if len(closes) < required:
            return self._unavailable("rsi", "indicator", required, len(closes)), {}
        changes = [current - previous for previous, current in zip(closes[-required:], closes[-period:])]
        average_gain = sum(max(change, 0.0) for change in changes) / period
        average_loss = sum(max(-change, 0.0) for change in changes) / period
        rsi = 100.0 if average_loss == 0 and average_gain > 0 else 50.0 if average_loss == 0 else 100.0 - 100.0 / (1.0 + average_gain / average_loss)
        values = {"rsi": rsi}
        if rsi <= oversold:
            return self._assessment("rsi", "buy", "RSI is at or below oversold threshold", values), values
        if rsi >= overbought:
            return self._assessment("rsi", "sell", "RSI is at or above overbought threshold", values), values
        return self._assessment("rsi", "neutral", "RSI is between configured thresholds", values), values

    def _bollinger_assessment(
        self, closes: list[float], period: int, multiplier: float
    ) -> tuple[_SignalAssessment, dict[str, float]]:
        if len(closes) < period:
            return self._unavailable("bollinger", "indicator", period, len(closes)), {}
        window = closes[-period:]
        middle = self._sma(window)
        standard_deviation = math.sqrt(sum((close - middle) ** 2 for close in window) / period)
        upper = middle + multiplier * standard_deviation
        lower = middle - multiplier * standard_deviation
        values = {"bollinger_upper": upper, "bollinger_middle": middle, "bollinger_lower": lower}
        if standard_deviation == 0:
            return self._assessment("bollinger", "neutral", "Bollinger bands have zero width", values), values
        if closes[-1] <= lower:
            return self._assessment("bollinger", "buy", "close is at or below lower Bollinger band", values), values
        if closes[-1] >= upper:
            return self._assessment("bollinger", "sell", "close is at or above upper Bollinger band", values), values
        return self._assessment("bollinger", "neutral", "close is inside Bollinger bands", values), values

    def _momentum_macd_assessment(
        self, closes: list[float], momentum_period: int, fast_window: int, slow_window: int, signal_window: int
    ) -> tuple[_SignalAssessment, dict[str, float]]:
        required = max(momentum_period + 1, slow_window + signal_window)
        if len(closes) < required:
            return self._unavailable("momentum_macd", "indicator", required, len(closes)), {}
        fast = self._ema_series(closes, fast_window)
        slow = self._ema_series(closes, slow_window)
        macd_series = [fast_value - slow_value for fast_value, slow_value in zip(fast, slow)]
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
            return self._assessment("momentum_macd", "buy", "positive momentum with upward MACD crossover", values), values
        if momentum < 0 and previous_macd >= previous_signal and macd < signal:
            return self._assessment("momentum_macd", "sell", "negative momentum with downward MACD crossover", values), values
        return self._assessment("momentum_macd", "neutral", "momentum/MACD entry conditions not met", values), values

    def _exit_conditions(
        self, request: RuleStrategyEvaluationRequest
    ) -> tuple[list[RuleStrategyConditionCheck], str | None]:
        position = request.market.position
        risk = request.config.risk
        if position.quantity == 0:
            return [
                RuleStrategyConditionCheck(code="take_profit", category="exit", state="not_triggered", detail="no open position"),
                RuleStrategyConditionCheck(code="stop_loss", category="exit", state="not_triggered", detail="no open position"),
            ], None
        entry_price = position.entry_price
        assert entry_price is not None
        return_pct = request.market.price / entry_price - 1.0
        take_profit_hit = risk.take_profit_pct is not None and return_pct >= risk.take_profit_pct
        stop_loss_hit = risk.stop_loss_pct is not None and return_pct <= -risk.stop_loss_pct
        conditions = [
            RuleStrategyConditionCheck(
                code="take_profit", category="exit", state="triggered" if take_profit_hit else "not_triggered",
                detail="take-profit threshold reached" if take_profit_hit else "take-profit threshold not reached",
                values={"return_pct": return_pct, "threshold_pct": risk.take_profit_pct},
            ),
            RuleStrategyConditionCheck(
                code="stop_loss", category="exit", state="triggered" if stop_loss_hit else "not_triggered",
                detail="stop-loss threshold reached" if stop_loss_hit else "stop-loss threshold not reached",
                values={"return_pct": return_pct, "threshold_pct": risk.stop_loss_pct},
            ),
        ]
        return conditions, "take_profit" if take_profit_hit else "stop_loss" if stop_loss_hit else None

    def _position_action(
        self, side: Signal, exit_reason: str | None, confirmation_mode: str, assessments: list[_SignalAssessment]
    ) -> tuple[Literal["sell", "no_op"], str, str]:
        if exit_reason == "take_profit":
            return "sell", "take_profit_triggered", "Sell recommendation: take-profit threshold reached."
        if exit_reason == "stop_loss":
            return "sell", "stop_loss_triggered", "Sell recommendation: stop-loss threshold reached."
        if side == "sell":
            return "sell", "indicator_sell_confirmed", "Sell recommendation: configured indicators confirm a sell signal."
        if not assessments:
            return "no_op", "no_enabled_indicators", "No action: no indicator rule is enabled."
        if side == "unavailable":
            return "no_op", "insufficient_candle_history", "No action: supplied candle history is insufficient for configured indicators."
        return "no_op", "no_exit_signal", "No action: no configured exit signal is confirmed."

    def _flat_action(
        self, side: Signal, assessments: list[_SignalAssessment], confirmation_mode: str
    ) -> tuple[Literal["buy", "no_op"], str, str]:
        if not assessments:
            return "no_op", "no_enabled_indicators", "No action: no indicator rule is enabled."
        if side == "buy":
            return "buy", "indicator_buy_confirmed", "Buy recommendation: configured indicators confirm a buy signal."
        if side == "sell":
            return "no_op", "sell_signal_without_position", "No action: sell signal cannot open a paper long position."
        if side == "unavailable":
            return "no_op", "insufficient_candle_history", "No action: supplied candle history is insufficient for configured indicators."
        return "no_op", "indicators_not_confirmed", "No action: configured indicators do not confirm an entry signal."

    def _risk_conditions(self, request: RuleStrategyEvaluationRequest, action: str, sizing: RuleStrategySizing) -> list[RuleStrategyConditionCheck]:
        market, risk = request.market, request.config.risk
        position_limit_blocked = action == "buy" and market.open_position_count >= risk.max_positions
        capital_blocked = action == "buy" and sizing.requested_quote > sizing.affordable_quote
        leverage_blocked = action == "buy" and sizing.requested_quote > sizing.max_allowed_quote
        return [
            RuleStrategyConditionCheck(
                code="max_positions", category="risk", state="blocked" if position_limit_blocked else "not_triggered",
                detail="maximum open positions reached" if position_limit_blocked else "position limit permits an entry",
                values={"open_position_count": market.open_position_count, "max_positions": risk.max_positions},
            ),
            RuleStrategyConditionCheck(
                code="available_collateral", category="risk", state="blocked" if capital_blocked else "not_triggered",
                detail="insufficient quote balance for configured leveraged size" if capital_blocked else "quote balance covers configured leveraged size",
                values={"requested_quote": sizing.requested_quote, "affordable_quote": sizing.affordable_quote},
            ),
            RuleStrategyConditionCheck(
                code="leverage_limit", category="risk", state="blocked" if leverage_blocked else "not_triggered",
                detail="configured size exceeds equity-based leverage limit" if leverage_blocked else "configured size is within equity-based leverage limit",
                values={"requested_quote": sizing.requested_quote, "max_allowed_quote": sizing.max_allowed_quote},
            ),
        ]

    def _sizing(self, request: RuleStrategyEvaluationRequest) -> RuleStrategySizing:
        market, risk = request.market, request.config.risk
        requested = risk.size_value if risk.size_mode == "fixed_quote" else market.equity_quote * risk.size_value
        return RuleStrategySizing(
            mode=risk.size_mode,
            requested_quote=requested,
            max_allowed_quote=market.equity_quote * risk.leverage,
            affordable_quote=market.quote_balance * risk.leverage,
            quantity=requested / market.price,
        )

    def _funding(self, request: RuleStrategyEvaluationRequest, action: str, sizing: RuleStrategySizing) -> RuleStrategyFundingImpact:
        position = request.market.position
        current_notional = position.quantity * request.market.price
        projected_notional = sizing.requested_quote if action == "buy" else 0.0 if action == "sell" else current_notional
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
    def _assessment(code: str, signal: Signal, detail: str, values: dict[str, float]) -> _SignalAssessment:
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
    def _unavailable(code: str, category: Literal["indicator", "exit", "risk"], required: int, supplied: int) -> _SignalAssessment:
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
