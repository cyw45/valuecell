"""Typed contracts for deterministic, paper-only crypto rule evaluation."""

from __future__ import annotations

from typing import Annotated, Literal, Union

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
    highest_price: float | None = Field(default=None, gt=0)
    addition_count: int = Field(default=0, ge=0, le=100)

    @model_validator(mode="after")
    def validate_entry_price(self) -> RuleStrategyPosition:
        if self.quantity == 0 and (
            self.entry_price is not None or self.highest_price is not None
        ):
            raise ValueError("position prices must be omitted when quantity is zero")
        if self.quantity == 0 and self.addition_count:
            raise ValueError("addition_count must be zero when quantity is zero")
        if self.highest_price is not None and self.entry_price is None:
            raise ValueError("highest_price requires entry_price")
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
    total_position_quote: float = Field(default=0.0, ge=0)
    position: RuleStrategyPosition = Field(default_factory=RuleStrategyPosition)


class RuleStrategyPaperPosition(RuleStrategyModel):
    """One server-owned long position in the paper account."""

    quantity: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    mark_price: float = Field(gt=0)
    highest_price: float | None = Field(default=None, gt=0)
    addition_count: int = Field(default=0, ge=0, le=100)


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
    trailing_take_profit_pct: float | None = Field(default=None, gt=0, le=1)
    max_total_position_pct: float = Field(default=1.0, gt=0, le=1)
    max_symbol_position_pct: float = Field(default=1.0, gt=0, le=1)
    add_to_winners: bool = False
    max_additions: int = Field(default=0, ge=0, le=20)
    max_positions: int = Field(default=1, ge=1, le=1_000)
    leverage: float = Field(default=1.0, ge=1, le=100)

    @model_validator(mode="after")
    def validate_additions(self) -> "RuleStrategyRiskConfig":
        if self.add_to_winners and self.max_additions == 0:
            raise ValueError("max_additions must be positive when add_to_winners is enabled")
        if not self.add_to_winners and self.max_additions != 0:
            raise ValueError("max_additions requires add_to_winners")
        return self


RuleInterval = Literal["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]


class ProgramConstantRef(RuleStrategyModel):
    kind: Literal["constant"]
    value: float


class ProgramPriceRef(RuleStrategyModel):
    kind: Literal["price"]
    interval: RuleInterval
    source: Literal["open", "high", "low", "close"] = "close"


class ProgramVolumeRef(RuleStrategyModel):
    kind: Literal["volume"]
    interval: RuleInterval


class ProgramIndicatorRef(RuleStrategyModel):
    kind: Literal["indicator"]
    name: Literal[
        "ma", "ema", "slope", "rsi", "atr", "adx", "volume_ma",
        "bollinger", "macd",
    ]
    interval: RuleInterval
    period: int = Field(default=14, ge=1, le=500)
    lookback: int = Field(default=1, ge=1, le=100)
    multiplier: float = Field(default=1.0, gt=0, le=100)
    component: Literal[
        "middle", "upper", "lower", "line", "signal", "histogram"
    ] = "line"
    fast_period: int = Field(default=12, ge=1, le=500)
    slow_period: int = Field(default=26, ge=2, le=500)
    signal_period: int = Field(default=9, ge=1, le=500)

    @model_validator(mode="after")
    def validate_indicator_parameters(self) -> "ProgramIndicatorRef":
        if self.name == "macd" and self.fast_period >= self.slow_period:
            raise ValueError("macd fast_period must be smaller than slow_period")
        if self.name == "bollinger" and self.component not in {
            "middle",
            "upper",
            "lower",
        }:
            raise ValueError("bollinger component must be middle, upper, or lower")
        if self.name == "macd" and self.component not in {
            "line",
            "signal",
            "histogram",
        }:
            raise ValueError("macd component must be line, signal, or histogram")
        if self.name not in {"bollinger", "macd"} and self.component != "line":
            raise ValueError("component is only configurable for bollinger and macd")
        return self


ProgramValueRef = Annotated[
    Union[ProgramConstantRef, ProgramPriceRef, ProgramVolumeRef, ProgramIndicatorRef],
    Field(discriminator="kind"),
]


class ProgramAllCondition(RuleStrategyModel):
    op: Literal["all"]
    args: list["ProgramCondition"] = Field(min_length=1, max_length=64)


class ProgramAnyCondition(RuleStrategyModel):
    op: Literal["any"]
    args: list["ProgramCondition"] = Field(min_length=1, max_length=64)


class ProgramAtLeastCondition(RuleStrategyModel):
    op: Literal["at_least"]
    count: int = Field(ge=1, le=64)
    args: list["ProgramCondition"] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_count(self) -> "ProgramAtLeastCondition":
        if self.count > len(self.args):
            raise ValueError("at_least count cannot exceed condition count")
        return self


class ProgramNotCondition(RuleStrategyModel):
    op: Literal["not"]
    arg: "ProgramCondition"


class ProgramCompareCondition(RuleStrategyModel):
    op: Literal["compare"]
    left: ProgramValueRef
    comparator: Literal["gt", "gte", "lt", "lte", "eq", "neq"]
    right: ProgramValueRef


class ProgramCrossCondition(RuleStrategyModel):
    op: Literal["cross"]
    left: ProgramValueRef
    direction: Literal["above", "below"]
    right: ProgramValueRef


class ProgramOrderedCondition(RuleStrategyModel):
    op: Literal["ordered"]
    direction: Literal["ascending", "descending"]
    values: list[ProgramValueRef] = Field(min_length=2, max_length=32)


ProgramCondition = Annotated[
    Union[
        ProgramAllCondition,
        ProgramAnyCondition,
        ProgramAtLeastCondition,
        ProgramNotCondition,
        ProgramCompareCondition,
        ProgramCrossCondition,
        ProgramOrderedCondition,
    ],
    Field(discriminator="op"),
]


class RuleStrategyProgramV2(RuleStrategyModel):
    """Closed, resource-bounded declarative strategy program."""

    schema_version: Literal[2]
    entry: ProgramCondition
    exit: ProgramCondition | None = None

    @model_validator(mode="after")
    def validate_resource_limits(self) -> "RuleStrategyProgramV2":
        node_count = 0
        intervals: set[str] = set()

        def visit_value(value: ProgramValueRef) -> None:
            nonlocal node_count
            node_count += 1
            if not isinstance(value, ProgramConstantRef):
                intervals.add(value.interval)
            if isinstance(value, ProgramIndicatorRef):
                required = program_ref_lookback(value)
                if required > 500:
                    raise ValueError("program indicator lookback cannot exceed 500 candles")

        def visit(condition: ProgramCondition, depth: int) -> None:
            nonlocal node_count
            if depth > 12:
                raise ValueError("program condition depth cannot exceed 12")
            node_count += 1
            if isinstance(condition, (ProgramAllCondition, ProgramAnyCondition, ProgramAtLeastCondition)):
                for child in condition.args:
                    visit(child, depth + 1)
            elif isinstance(condition, ProgramNotCondition):
                visit(condition.arg, depth + 1)
            elif isinstance(condition, (ProgramCompareCondition, ProgramCrossCondition)):
                visit_value(condition.left)
                visit_value(condition.right)
            else:
                for value in condition.values:
                    visit_value(value)

        visit(self.entry, 1)
        if self.exit is not None:
            visit(self.exit, 1)
        if node_count > 128:
            raise ValueError("program cannot exceed 128 nodes")
        if len(intervals) > 8:
            raise ValueError("program cannot reference more than 8 intervals")
        return self


def program_ref_lookback(ref: ProgramIndicatorRef) -> int:
    if ref.name == "macd":
        return ref.slow_period + ref.signal_period + 1
    if ref.name == "adx":
        return 2 * ref.period + 1
    if ref.name in {"rsi", "atr"}:
        return ref.period + 1
    if ref.name == "slope":
        return ref.period + ref.lookback
    return ref.period


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
    entry_confirmation_mode: Literal["all", "any", "at_least", "ratio"] = "all"
    entry_confirmation_count: int = Field(default=1, ge=1, le=6)
    entry_confirmation_ratio: float = Field(default=1.0, gt=0, le=1)
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

    @model_validator(mode="after")
    def validate_entry_confirmation_count(self) -> "AdvancedRuleSetConfig":
        enabled_count = sum(
            rule.enabled
            for rule in (
                self.moving_average, self.macd, self.bollinger,
                self.rsi, self.momentum, self.brar,
            )
        )
        if (self.enabled and self.entry_confirmation_mode == "at_least"
                and self.entry_confirmation_count > enabled_count):
            raise ValueError(
                "entry_confirmation_count cannot exceed enabled entry conditions"
            )
        return self


class RuleStrategyExecutionConfig(RuleStrategyModel):
    """Explicit execution target; only OKX Demo spot can leave the paper ledger."""

    environment: Literal["paper", "okx_demo"] = "paper"
    sandbox_connection_id: str | None = Field(default=None, min_length=1, max_length=36)
    max_order_quote_amount: float = Field(default=10_000.0, gt=0, le=10_000)
    max_daily_quote_amount: float = Field(default=10_000.0, gt=0, le=100_000)
    max_total_quote_amount: float = Field(default=10_000.0, gt=0, le=100_000)

    @model_validator(mode="after")
    def validate_target(self) -> "RuleStrategyExecutionConfig":
        if self.environment == "okx_demo" and not self.sandbox_connection_id:
            raise ValueError("sandbox_connection_id is required for okx_demo execution")
        if self.environment == "paper" and self.sandbox_connection_id is not None:
            raise ValueError("sandbox_connection_id is only allowed for okx_demo execution")
        if self.max_order_quote_amount > self.max_daily_quote_amount:
            raise ValueError("max_order_quote_amount cannot exceed max_daily_quote_amount")
        if self.max_order_quote_amount > self.max_total_quote_amount:
            raise ValueError("max_order_quote_amount cannot exceed max_total_quote_amount")
        return self


class RuleStrategyConfig(RuleStrategyModel):
    """Rule evaluation configuration with a paper or explicitly bound OKX Demo target."""

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
    program: RuleStrategyProgramV2 | None = None
    risk: RuleStrategyRiskConfig = Field(default_factory=RuleStrategyRiskConfig)
    execution: RuleStrategyExecutionConfig = Field(default_factory=RuleStrategyExecutionConfig)

    @model_validator(mode="after")
    def normalize_symbols(self) -> RuleStrategyConfig:
        if self.execution.environment == "okx_demo" and self.risk.leverage != 1:
            raise ValueError("OKX Demo spot execution requires leverage 1")
        if self.execution.environment == "okx_demo" and (
            self.risk.trailing_take_profit_pct is not None
            or self.risk.add_to_winners
        ):
            raise ValueError(
                "OKX Demo does not expose durable strategy entry/highest-price state; "
                "use paper execution for trailing take profit or winner pyramiding"
            )
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


class RuleStrategyEntryConfirmation(BaseModel):
    """Aggregate entry evidence; ratio requirements are rounded up."""

    model_config = ConfigDict(extra="forbid")

    enabled: int = Field(ge=0)
    available: int = Field(ge=0)
    passed: int = Field(ge=0)
    required: int = Field(ge=0)
    mode: Literal["all", "any", "at_least", "ratio"]


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
    entry_confirmation: RuleStrategyEntryConfirmation


class RuleStrategyTextImportConfig(RuleStrategyModel):
    """AI-proposed fields that can be safely copied into a strategy draft."""

    interval: Literal["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"] = "15m"
    advanced_rules: AdvancedRuleSetConfig = Field(default_factory=AdvancedRuleSetConfig)
    program: RuleStrategyProgramV2 | None = None
    risk: RuleStrategyRiskConfig


class RuleStrategyTextImportProposal(RuleStrategyModel):
    """Validated semantic compilation result from natural-language rules."""

    strategy_name: str | None = Field(default=None, max_length=200)
    executable: bool = True
    config: RuleStrategyTextImportConfig | None
    summary: str = Field(min_length=1, max_length=2_000)
    unresolved_items: list[str] = Field(default_factory=list, max_length=30)
    corrections: list[str] = Field(default_factory=list, max_length=30)
    rejection_reasons: list[str] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def validate_executable_contract(self) -> "RuleStrategyTextImportProposal":
        if self.executable and self.config is None:
            raise ValueError("config is required when executable is true")
        if not self.executable and not self.rejection_reasons:
            raise ValueError("rejection_reasons are required when executable is false")
        return self
