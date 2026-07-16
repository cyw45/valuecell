"""Typed contracts for deterministic, paper-only crypto rule evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuleStrategyModel(BaseModel):
    """Strict base contract that rejects unknown and non-finite inputs."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RuleStrategyCandle(RuleStrategyModel):
    """One explicitly supplied OHLCV candle; the engine never fetches candles."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    timestamp_ms: int = Field(gt=0)
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_price_range(self) -> RuleStrategyCandle:
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("candle high must not be below open, close, or low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("candle low must not exceed open, close, or high")
        return self


class RuleStrategyPosition(RuleStrategyModel):
    """Current paper long position available to the deterministic evaluator."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    quantity: float = Field(default=0.0, ge=0)
    entry_price: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_entry_price(self) -> RuleStrategyPosition:
        if self.quantity > 0 and self.entry_price is None:
            raise ValueError("entry_price is required when position quantity is positive")
        if self.quantity == 0 and self.entry_price is not None:
            raise ValueError("entry_price must be omitted when position quantity is zero")
        return self


class RuleStrategyMarketSnapshot(RuleStrategyModel):
    """Market facts supplied by the caller for one deterministic paper tick."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    symbol: str = Field(min_length=1, max_length=64)
    price: float = Field(gt=0)
    funding_rate: float = Field(default=0.0, ge=-1, le=1)


class RuleStrategyEngineMarketSnapshot(RuleStrategyMarketSnapshot):
    """Server-derived account facts consumed by the pure rule engine."""

    equity_quote: float = Field(ge=0)
    quote_balance: float = Field(ge=0)
    open_position_count: int = Field(default=0, ge=0)
    position: RuleStrategyPosition = Field(default_factory=RuleStrategyPosition)


class RuleStrategyPaperPosition(RuleStrategyModel):
    """One server-owned long position in the paper account."""

    quantity: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    mark_price: float = Field(gt=0)


class RuleStrategyPaperAccount(RuleStrategyModel):
    """Immutable paper-account balance snapshot persisted with each evaluation."""

    initial_capital_quote: float = Field(gt=0)
    quote_balance: float = Field(ge=0)
    positions: dict[str, RuleStrategyPaperPosition] = Field(default_factory=dict)
    realized_pnl_quote: float = 0.0
    unrealized_pnl_quote: float = 0.0
    equity_quote: float = Field(ge=0)


class MovingAverageRuleConfig(RuleStrategyModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    enabled: bool = False
    short_window: int = Field(default=9, ge=1, le=500)
    long_window: int = Field(default=21, ge=2, le=500)

    @model_validator(mode="after")
    def validate_windows(self) -> MovingAverageRuleConfig:
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be smaller than long_window")
        return self


class RsiRuleConfig(RuleStrategyModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    enabled: bool = False
    period: int = Field(default=14, ge=2, le=500)
    oversold: float = Field(default=30.0, ge=0, le=100)
    overbought: float = Field(default=70.0, ge=0, le=100)

    @model_validator(mode="after")
    def validate_thresholds(self) -> RsiRuleConfig:
        if self.oversold >= self.overbought:
            raise ValueError("oversold must be lower than overbought")
        return self


class BollingerRuleConfig(RuleStrategyModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    enabled: bool = False
    period: int = Field(default=20, ge=2, le=500)
    standard_deviations: float = Field(default=2.0, gt=0, le=10)


class MomentumMacdRuleConfig(RuleStrategyModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    enabled: bool = False
    momentum_period: int = Field(default=14, ge=1, le=500)
    macd_fast_window: int = Field(default=12, ge=1, le=500)
    macd_slow_window: int = Field(default=26, ge=2, le=500)
    macd_signal_window: int = Field(default=9, ge=1, le=500)

    @model_validator(mode="after")
    def validate_windows(self) -> MomentumMacdRuleConfig:
        if self.macd_fast_window >= self.macd_slow_window:
            raise ValueError("macd_fast_window must be smaller than macd_slow_window")
        return self


class RuleStrategyRiskConfig(RuleStrategyModel):
    """Risk limits and one fixed quote amount for each new position."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    order_quote_amount: float = Field(default=100.0, gt=0, le=100_000_000)
    take_profit_pct: float | None = Field(default=None, gt=0, le=1)
    stop_loss_pct: float | None = Field(default=None, gt=0, le=1)
    max_positions: int = Field(default=1, ge=1, le=1_000)
    leverage: float = Field(default=1.0, ge=1, le=100)


class AdvancedMovingAverageRuleConfig(RuleStrategyModel):
    """Price-to-moving-average comparison on an independently selected interval."""

    enabled: bool = False
    interval: Literal["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"] = "1d"
    period: int = Field(default=20, ge=2, le=500)
    entry_comparator: Literal["above", "below"] = "above"


class AdvancedMacdRuleConfig(RuleStrategyModel):
    """MACD crossover condition with an independently selected interval."""

    enabled: bool = False
    interval: Literal["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"] = "5m"
    fast_window: int = Field(default=12, ge=1, le=500)
    slow_window: int = Field(default=26, ge=2, le=500)
    signal_window: int = Field(default=9, ge=1, le=500)
    entry_cross: Literal["golden", "death"] = "golden"

    @model_validator(mode="after")
    def validate_windows(self) -> AdvancedMacdRuleConfig:
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be smaller than slow_window")
        return self


class AdvancedBollingerRuleConfig(RuleStrategyModel):
    """Price-to-Bollinger-band comparison on an independently selected interval."""

    enabled: bool = False
    interval: Literal["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"] = "15m"
    period: int = Field(default=20, ge=2, le=500)
    standard_deviations: float = Field(default=2.0, gt=0, le=10)
    entry_reference: Literal["upper", "middle", "lower"] = "middle"
    entry_comparator: Literal["above", "below"] = "above"


class AdvancedThresholdRuleConfig(RuleStrategyModel):
    """Threshold entry and exit configuration for a scalar technical indicator."""

    enabled: bool = False
    interval: Literal["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"] = "15m"
    period: int = Field(default=14, ge=1, le=500)
    entry_comparator: Literal["above", "below"] = "below"
    entry_threshold: float = 20.0
    exit_enabled: bool = True
    exit_comparator: Literal["above", "below"] = "above"
    exit_threshold: float = 85.0


class AdvancedBrarRuleConfig(AdvancedThresholdRuleConfig):
    """BRAR threshold configuration. Operators may use the AR or BR component."""

    period: int = Field(default=26, ge=2, le=500)
    component: Literal["ar", "br"] = "br"
    entry_threshold: float = 30.0
    exit_enabled: bool = False


class AdvancedRuleSetConfig(RuleStrategyModel):
    """Fully configurable multi-timeframe indicator rule set."""

    enabled: bool = False
    entry_confirmation_mode: Literal["all", "any"] = "all"
    exit_confirmation_mode: Literal["all", "any"] = "any"
    moving_average: AdvancedMovingAverageRuleConfig = Field(
        default_factory=AdvancedMovingAverageRuleConfig
    )
    macd: AdvancedMacdRuleConfig = Field(default_factory=AdvancedMacdRuleConfig)
    bollinger: AdvancedBollingerRuleConfig = Field(
        default_factory=AdvancedBollingerRuleConfig
    )
    rsi: AdvancedThresholdRuleConfig = Field(
        default_factory=lambda: AdvancedThresholdRuleConfig(period=14)
    )
    momentum: AdvancedThresholdRuleConfig = Field(
        default_factory=lambda: AdvancedThresholdRuleConfig(period=14)
    )
    brar: AdvancedBrarRuleConfig = Field(default_factory=AdvancedBrarRuleConfig)


class RuleStrategyConfig(RuleStrategyModel):
    """Paper-only rules and their normalized public-market evaluation scope."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    mode: Literal["paper"] = "paper"
    initial_capital_quote: float = Field(default=10_000.0, gt=0, le=100_000_000)
    symbols: list[str] = Field(
        default_factory=lambda: ["BTC-USDT"], min_length=1, max_length=100
    )
    interval: Literal["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"] = "1h"
    decide_interval_s: int | None = Field(default=None, ge=60)
    confirmation_mode: Literal["all", "any"] = "all"
    moving_average: MovingAverageRuleConfig = Field(default_factory=MovingAverageRuleConfig)
    rsi: RsiRuleConfig = Field(default_factory=RsiRuleConfig)
    bollinger: BollingerRuleConfig = Field(default_factory=BollingerRuleConfig)
    momentum_macd: MomentumMacdRuleConfig = Field(default_factory=MomentumMacdRuleConfig)
    advanced_rules: AdvancedRuleSetConfig = Field(default_factory=AdvancedRuleSetConfig)
    risk: RuleStrategyRiskConfig = Field(default_factory=RuleStrategyRiskConfig)

    @model_validator(mode="after")
    def normalize_symbols(self) -> RuleStrategyConfig:
        normalized: list[str] = []
        for raw_symbol in self.symbols:
            symbol = raw_symbol.strip().upper().replace("/", "-")
            if not symbol.endswith("-USDT"):
                raise ValueError("Only USDT crypto symbols are supported")
            if symbol not in normalized:
                normalized.append(symbol)
        self.symbols = normalized
        return self


class RuleStrategyEvaluationRequest(BaseModel):
    """Frozen inputs for one deterministic paper rule-engine evaluation."""

    model_config = ConfigDict(extra="forbid")

    config: RuleStrategyConfig
    candles: list[RuleStrategyCandle] = Field(min_length=1, max_length=5_000)
    candle_sets: dict[str, list[RuleStrategyCandle]] = Field(default_factory=dict)
    market: RuleStrategyEngineMarketSnapshot

    @model_validator(mode="after")
    def validate_candle_order(self) -> RuleStrategyEvaluationRequest:
        timestamps = [candle.timestamp_ms for candle in self.candles]
        if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ValueError("candle timestamps must be strictly increasing")
        for interval, candle_set in self.candle_sets.items():
            if not candle_set:
                raise ValueError(f"candle set for {interval} must not be empty")
            set_timestamps = [candle.timestamp_ms for candle in candle_set]
            if any(
                current <= previous
                for previous, current in zip(set_timestamps, set_timestamps[1:])
            ):
                raise ValueError(
                    f"candle timestamps for {interval} must be strictly increasing"
                )
        return self


class RuleStrategyConditionCheck(BaseModel):
    """A single evaluation fact, including unavailable and risk-blocked conditions."""

    model_config = ConfigDict(extra="forbid")

    code: str
    category: Literal["indicator", "exit", "risk"]
    state: Literal["triggered", "not_triggered", "blocked", "unavailable"]
    detail: str
    values: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


class RuleStrategyIndicatorValues(BaseModel):
    model_config = ConfigDict(extra="forbid")

    moving_average_short: float | None = None
    moving_average_long: float | None = None
    previous_moving_average_short: float | None = None
    previous_moving_average_long: float | None = None
    rsi: float | None = None
    bollinger_upper: float | None = None
    bollinger_middle: float | None = None
    bollinger_lower: float | None = None
    momentum: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    previous_macd: float | None = None
    previous_macd_signal: float | None = None
    brar_ar: float | None = None
    brar_br: float | None = None


class RuleStrategySizing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["fixed_quote"] = "fixed_quote"
    requested_quote: float
    max_allowed_quote: float
    affordable_quote: float
    quantity: float


class RuleStrategyFundingImpact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    funding_rate: float
    current_notional_quote: float
    projected_notional_quote: float
    estimated_payment_quote: float
    direction: Literal["credit", "debit", "none"]


class RuleStrategyEvaluationResult(BaseModel):
    """Explainable recommendation only; it never creates or submits an order."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["paper"] = "paper"
    action: Literal["buy", "sell", "no_op"]
    reason_code: str
    reason: str
    conditions: list[RuleStrategyConditionCheck]
    indicators: RuleStrategyIndicatorValues
    sizing: RuleStrategySizing
    funding: RuleStrategyFundingImpact


class RuleStrategyTextImportConfig(RuleStrategyModel):
    """AI-proposed fields that can be safely copied into a strategy draft."""

    interval: Literal["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"] = "15m"
    advanced_rules: AdvancedRuleSetConfig
    risk: RuleStrategyRiskConfig


class RuleStrategyTextImportProposal(RuleStrategyModel):
    """Validated, review-only structured result from a natural-language strategy."""

    strategy_name: str | None = Field(default=None, max_length=200)
    config: RuleStrategyTextImportConfig
    summary: str = Field(min_length=1, max_length=2_000)
    unresolved_items: list[str] = Field(default_factory=list, max_length=30)
