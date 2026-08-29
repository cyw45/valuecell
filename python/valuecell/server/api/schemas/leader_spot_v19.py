"""Strict configuration and wire contracts for the isolated V19 leader strategy."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


LEADER_SPOT_V19_MODULE_ID = "leader_spot_v19_0"
LEADER_SPOT_V19_SCHEMA_VERSION = 19

LeaderSpotV19Environment = Literal["paper", "okx_demo"]
LeaderSpotV19StrategyStatus = Literal["running", "stopped", "paused", "archived"]
LeaderSpotV19DataState = Literal["DATA_OK", "DATA_DEGRADED", "DATA_UNSAFE"]
LeaderSpotV19MarketState = Literal["M0", "M1", "M2", "M3", "M4"]
LeaderSpotV19ProtectionStatus = Literal[
    "PROTECTION_NONE",
    "PROTECTION_PENDING",
    "PROTECTION_ACTIVE",
]
LeaderSpotV19ExecutionStatus = Literal[
    "idle",
    "pending",
    "submitted",
    "partially_filled",
    "filled",
    "cancelled",
    "failed",
    "submission_unknown",
    "manual_alert",
]


class LeaderSpotV19Model(BaseModel):
    """Reject unknown and non-finite values at the module boundary."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        frozen=True,
    )


class LeaderSpotV19PositionConfig(LeaderSpotV19Model):
    """Fixed-money spot sizing; only the order amount is configurable."""

    order_amount_quote: float = Field(default=100.0, gt=0)
    max_positions: Literal[6] = 6
    position_cooldown_hours: Literal[24] = 24
    leverage_enabled: Literal[False] = False
    contract_enabled: Literal[False] = False
    max_order_equity_pct: Literal[0.2] = 0.2


class LeaderSpotV19LossConfig(LeaderSpotV19Model):
    """The two loss exits that precede confirmed profit protection."""

    stop_loss_pct: Literal[0.08] = 0.08
    loss_circuit_days: Literal[7] = 7
    loss_circuit_hours: Literal[168] = 168
    loss_circuit_profit_pct: Literal[0.05] = 0.05
    loss_circuit_use_kline: Literal[True] = True


class LeaderSpotV19MovingStopTier(LeaderSpotV19Model):
    """One immutable peak-profit moving-stop tier."""

    profit_pct: float = Field(gt=0)
    stop_multiplier: float = Field(gt=0)


DEFAULT_MOVING_STOP_TIERS: tuple[LeaderSpotV19MovingStopTier, ...] = (
    LeaderSpotV19MovingStopTier(profit_pct=0.05, stop_multiplier=1.02),
    LeaderSpotV19MovingStopTier(profit_pct=0.08, stop_multiplier=1.04),
    LeaderSpotV19MovingStopTier(profit_pct=0.12, stop_multiplier=1.07),
    LeaderSpotV19MovingStopTier(profit_pct=0.15, stop_multiplier=1.10),
    LeaderSpotV19MovingStopTier(profit_pct=0.25, stop_multiplier=1.15),
    LeaderSpotV19MovingStopTier(profit_pct=0.40, stop_multiplier=1.25),
    LeaderSpotV19MovingStopTier(profit_pct=0.60, stop_multiplier=1.40),
    LeaderSpotV19MovingStopTier(profit_pct=1.00, stop_multiplier=1.60),
    LeaderSpotV19MovingStopTier(profit_pct=2.00, stop_multiplier=2.50),
    LeaderSpotV19MovingStopTier(profit_pct=3.00, stop_multiplier=3.50),
)


class LeaderSpotV19LayeredExitTier(LeaderSpotV19Model):
    """One immutable current-profit layered retracement tier."""

    minimum_profit_pct: float = Field(ge=0)
    maximum_profit_pct: float | None = Field(default=None, gt=0)
    retracement_pct: float = Field(gt=0, lt=1)
    floor_multiplier: float = Field(gt=0)


DEFAULT_LAYERED_EXIT_TIERS: tuple[LeaderSpotV19LayeredExitTier, ...] = (
    LeaderSpotV19LayeredExitTier(
        minimum_profit_pct=0.05,
        maximum_profit_pct=0.10,
        retracement_pct=0.04,
        floor_multiplier=1.01,
    ),
    LeaderSpotV19LayeredExitTier(
        minimum_profit_pct=0.10,
        maximum_profit_pct=0.20,
        retracement_pct=0.05,
        floor_multiplier=1.05,
    ),
    LeaderSpotV19LayeredExitTier(
        minimum_profit_pct=0.20,
        maximum_profit_pct=0.40,
        retracement_pct=0.06,
        floor_multiplier=1.10,
    ),
    LeaderSpotV19LayeredExitTier(
        minimum_profit_pct=0.40,
        maximum_profit_pct=0.80,
        retracement_pct=0.07,
        floor_multiplier=1.20,
    ),
    LeaderSpotV19LayeredExitTier(
        minimum_profit_pct=0.80,
        maximum_profit_pct=1.50,
        retracement_pct=0.06,
        floor_multiplier=1.40,
    ),
    LeaderSpotV19LayeredExitTier(
        minimum_profit_pct=1.50,
        maximum_profit_pct=3.00,
        retracement_pct=0.05,
        floor_multiplier=1.80,
    ),
    LeaderSpotV19LayeredExitTier(
        minimum_profit_pct=3.00,
        maximum_profit_pct=None,
        retracement_pct=0.04,
        floor_multiplier=3.00,
    ),
)


class LeaderSpotV19TrendConfig(LeaderSpotV19Model):
    """15-minute EMA trend-exit parameters."""

    standard_profit_threshold: float = Field(default=0.15, ge=0)
    degraded_profit_threshold: float = Field(default=0.10, ge=0)
    ema_fast_period: int = Field(default=21, ge=2)
    ema_slow_period: int = Field(default=55, ge=2)
    confirm_bars: Literal[2] = 2
    limit_wait_seconds: Literal[60] = 60
    limit_offset: Literal[0.002] = 0.002

    @model_validator(mode="after")
    def validate_ema_order(self) -> "LeaderSpotV19TrendConfig":
        if self.ema_fast_period >= self.ema_slow_period:
            raise ValueError("ema_fast_period must be smaller than ema_slow_period")
        return self


class LeaderSpotV19ProfitConfig(LeaderSpotV19Model):
    """Confirmed-profit protection settings and immutable exit ladders."""

    protection_profit_pct: Literal[0.05] = 0.05
    protection_hold_seconds: Literal[60] = 60
    moving_stop_track_mode: Literal["peak_profit"] = "peak_profit"
    moving_stop_initial_multiplier: Literal[1.02] = 1.02
    moving_stop_tiers: tuple[LeaderSpotV19MovingStopTier, ...] = DEFAULT_MOVING_STOP_TIERS
    layered_use_current: Literal[True] = True
    peak_update_source: Literal["1min_kline"] = "1min_kline"
    peak_not_retreat: Literal[True] = True
    layered_exit_tiers: tuple[LeaderSpotV19LayeredExitTier, ...] = DEFAULT_LAYERED_EXIT_TIERS
    trend: LeaderSpotV19TrendConfig = Field(default_factory=LeaderSpotV19TrendConfig)

    @model_validator(mode="after")
    def validate_immutable_ladders(self) -> "LeaderSpotV19ProfitConfig":
        if self.moving_stop_tiers != DEFAULT_MOVING_STOP_TIERS:
            raise ValueError("V19 moving stop tiers are immutable")
        if self.layered_exit_tiers != DEFAULT_LAYERED_EXIT_TIERS:
            raise ValueError("V19 layered exit tiers are immutable")
        return self


class LeaderSpotV19MarketConfig(LeaderSpotV19Model):
    """Market-state thresholds and strong-trend promotion thresholds."""

    up_ratio_standard: float = Field(default=0.42, ge=0, le=1)
    volume_ratio_standard: float = Field(default=1.1, gt=0)
    fear_greed_standard: int = Field(default=30, ge=0, le=100)
    fear_greed_degraded: int = Field(default=25, ge=0, le=100)
    funding_rate_standard_min: float = -0.0015
    funding_rate_standard_max: float = 0.0015
    funding_rate_degraded_min: float = -0.0020
    funding_rate_degraded_max: float = 0.0020
    up_ratio_halt: float = Field(default=0.35, ge=0, le=1)
    standard_allow_fail: Literal[1] = 1
    strong_trend_up_ratio: float = Field(default=0.55, ge=0, le=1)
    strong_trend_volume_ratio: float = Field(default=1.3, gt=0)

    @model_validator(mode="after")
    def validate_funding_ranges(self) -> "LeaderSpotV19MarketConfig":
        if self.funding_rate_standard_min >= self.funding_rate_standard_max:
            raise ValueError("standard funding-rate bounds are invalid")
        if self.funding_rate_degraded_min >= self.funding_rate_degraded_max:
            raise ValueError("degraded funding-rate bounds are invalid")
        if self.fear_greed_degraded > self.fear_greed_standard:
            raise ValueError("degraded fear-greed threshold cannot exceed standard")
        if self.up_ratio_halt >= self.up_ratio_standard:
            raise ValueError("halt up-ratio threshold must be below standard")
        return self


class LeaderSpotV19CandidateConfig(LeaderSpotV19Model):
    """Candidate funnel thresholds; risk floors are not relaxable."""

    relative_strength_rank_pct_standard: float = Field(default=0.18, gt=0, le=1)
    relative_strength_rank_pct_starved: float = Field(default=0.23, gt=0, le=1)
    liquidity_quote_standard: float = Field(default=200_000, gt=0)
    liquidity_quote_starved: float = Field(default=150_000, gt=0)
    score_standard: int = Field(default=35, ge=0)
    score_degraded: int = Field(default=42, ge=0)
    score_starved: int = Field(default=30, ge=0)
    signal_starve_48h_enabled: Literal[True] = True
    signal_starve_72h_enabled: Literal[True] = True
    signal_recover_count: Literal[2] = 2
    high_pump_24h_pct: Literal[0.40] = 0.40


class LeaderSpotV19BreakoutConfig(LeaderSpotV19Model):
    """V16.1 box inputs and V19 dual-close breakout contract."""

    box_params_source: Literal["V16.1"] = "V16.1"
    box_volume_degraded_add: Literal[0.2] = 0.2
    confirm_a: Literal["15min_close"] = "15min_close"
    confirm_b: Literal["5min_x2"] = "5min_x2"
    second_confirmation_volume_required: Literal[True] = True
    needle_amplitude_pct: Literal[0.12] = 0.12
    needle_br_threshold: Literal[280.0] = 280.0


class LeaderSpotV19EntryConfig(LeaderSpotV19Model):
    """Three bounded limit-entry tiers; market entry is not representable."""

    mode: Literal["limit"] = "limit"
    tier1_offset: Literal[0.003] = 0.003
    tier1_wait_seconds: Literal[300] = 300
    tier2_offset: Literal[0.005] = 0.005
    tier2_wait_seconds: Literal[180] = 180
    tier3_offset: Literal[0.008] = 0.008
    tier3_wait_seconds: Literal[120] = 120
    tier3_failure: Literal["cancel"] = "cancel"
    slippage_check: Literal[True] = True

    @model_validator(mode="after")
    def validate_tiers(self) -> "LeaderSpotV19EntryConfig":
        if not self.tier1_offset < self.tier2_offset < self.tier3_offset:
            raise ValueError("entry offsets must increase through tier three")
        return self


class LeaderSpotV19NewCoinConfig(LeaderSpotV19Model):
    """Listing-age safety gates."""

    ban_hours: Literal[6] = 6
    strict_hours: Literal[24] = 24
    degraded_hours: Literal[72] = 72
    degraded_volume_add: Literal[0.2] = 0.2
    degraded_score_add: Literal[7] = 7


class LeaderSpotV19AccountRiskConfig(LeaderSpotV19Model):
    """The two V19 account-level circuit breakers."""

    daily_loss_reset: Literal["00:00"] = "00:00"
    equity_drawdown_halt_pct: Literal[0.15] = 0.15
    equity_halt_hours: Literal[24] = 24
    equity_calculation_interval_seconds: Literal[60] = 60


class LeaderSpotV19DataConfig(LeaderSpotV19Model):
    """Data freshness, retry, and cross-source tolerance parameters."""

    check_interval_seconds: Literal[60] = 60
    api_retry_max: Literal[3] = 3
    api_retry_interval_seconds: Literal[1] = 1
    api_reconnect_interval_seconds: Literal[5] = 5
    timestamp_tolerance_seconds: Literal[30] = 30
    price_source_deviation_pct: Literal[0.05] = 0.05
    btc_cross_source_deviation_pct: Literal[0.02] = 0.02
    log_retention_days: Literal[30] = 30


class LeaderSpotV19BacktestConfig(LeaderSpotV19Model):
    """Reproducibility assumptions for V19 validation runs."""

    period_months: int = Field(default=12, ge=12)
    fee_pct: Literal[0.001] = 0.001
    slippage_pct: Literal[0.005] = 0.005
    data_type: Literal["1min_or_tick"] = "1min_or_tick"
    walk_forward_train_ratio: Literal[0.7] = 0.7
    walk_forward_window_months: Literal[3] = 3


class LeaderSpotV19Config(LeaderSpotV19Model):
    """Complete versioned configuration for the isolated V19 module."""

    module_id: Literal["leader_spot_v19_0"] = LEADER_SPOT_V19_MODULE_ID
    schema_version: Literal[19] = LEADER_SPOT_V19_SCHEMA_VERSION
    environment: LeaderSpotV19Environment = "paper"
    sandbox_connection_id: str | None = Field(default=None, min_length=1, max_length=36)
    position: LeaderSpotV19PositionConfig = Field(default_factory=LeaderSpotV19PositionConfig)
    loss: LeaderSpotV19LossConfig = Field(default_factory=LeaderSpotV19LossConfig)
    profit: LeaderSpotV19ProfitConfig = Field(default_factory=LeaderSpotV19ProfitConfig)
    market: LeaderSpotV19MarketConfig = Field(default_factory=LeaderSpotV19MarketConfig)
    candidate: LeaderSpotV19CandidateConfig = Field(default_factory=LeaderSpotV19CandidateConfig)
    breakout: LeaderSpotV19BreakoutConfig = Field(default_factory=LeaderSpotV19BreakoutConfig)
    entry: LeaderSpotV19EntryConfig = Field(default_factory=LeaderSpotV19EntryConfig)
    new_coin: LeaderSpotV19NewCoinConfig = Field(default_factory=LeaderSpotV19NewCoinConfig)
    account_risk: LeaderSpotV19AccountRiskConfig = Field(default_factory=LeaderSpotV19AccountRiskConfig)
    data: LeaderSpotV19DataConfig = Field(default_factory=LeaderSpotV19DataConfig)
    backtest: LeaderSpotV19BacktestConfig = Field(default_factory=LeaderSpotV19BacktestConfig)

    @computed_field
    @property
    def daily_loss_limit_quote(self) -> float:
        """Calculate the daily realized-loss ceiling from fixed strategy limits."""

        return (
            self.position.order_amount_quote
            * self.position.max_positions
            * self.loss.stop_loss_pct
        )

    @model_validator(mode="after")
    def validate_execution_target(self) -> "LeaderSpotV19Config":
        if self.environment == "okx_demo" and self.sandbox_connection_id is None:
            raise ValueError("sandbox_connection_id is required for okx_demo")
        if self.environment == "paper" and self.sandbox_connection_id is not None:
            raise ValueError("sandbox_connection_id is only allowed for okx_demo")
        return self


class LeaderSpotV19CreateRequest(LeaderSpotV19Model):
    """Tenant request contract for creating a stopped V19 strategy."""

    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2_000)
    config: LeaderSpotV19Config = Field(default_factory=LeaderSpotV19Config)


class LeaderSpotV19StateSnapshot(LeaderSpotV19Model):
    """Shared Web/Mobile state view; it never contains secrets."""

    data_state: LeaderSpotV19DataState
    market_state: LeaderSpotV19MarketState
    can_open: bool
    positions_count: int = Field(ge=0, le=6)
    max_positions: Literal[6] = 6
    daily_realized_loss_quote: float
    daily_loss_limit_quote: float = Field(ge=0)
    equity_drawdown_pct: float | None = None
    account_halt_until: str | None = None


class LeaderSpotV19Summary(LeaderSpotV19Model):
    """Shared list/detail summary for both clients."""

    strategy_id: str
    name: str
    description: str | None
    status: LeaderSpotV19StrategyStatus
    environment: LeaderSpotV19Environment
    module_id: Literal["leader_spot_v19_0"] = LEADER_SPOT_V19_MODULE_ID
    schema_version: Literal[19] = LEADER_SPOT_V19_SCHEMA_VERSION
    current_batch_id: str | None
    state: LeaderSpotV19StateSnapshot | None = None
