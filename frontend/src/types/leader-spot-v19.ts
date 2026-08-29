export const LEADER_SPOT_V19_MODULE_ID = "leader_spot_v19_0" as const;
export const LEADER_SPOT_V19_SCHEMA_VERSION = 19 as const;

export type LeaderSpotV19Environment = "paper" | "okx_demo";
export type LeaderSpotV19StrategyStatus =
  | "running"
  | "stopped"
  | "paused"
  | "archived";
export type LeaderSpotV19DataState = "DATA_OK" | "DATA_DEGRADED" | "DATA_UNSAFE";
export type LeaderSpotV19MarketState = "M0" | "M1" | "M2" | "M3" | "M4";
export type LeaderSpotV19ProtectionStatus =
  | "PROTECTION_NONE"
  | "PROTECTION_PENDING"
  | "PROTECTION_ACTIVE";
export type LeaderSpotV19ExecutionStatus =
  | "idle"
  | "pending"
  | "submitted"
  | "partially_filled"
  | "filled"
  | "cancelled"
  | "failed"
  | "submission_unknown"
  | "manual_alert";

export interface LeaderSpotV19PositionConfig {
  order_amount_quote: number;
  max_positions: 6;
  position_cooldown_hours: 24;
  leverage_enabled: false;
  contract_enabled: false;
  max_order_equity_pct: 0.2;
}

export interface LeaderSpotV19LossConfig {
  stop_loss_pct: 0.08;
  loss_circuit_days: 7;
  loss_circuit_hours: 168;
  loss_circuit_profit_pct: 0.05;
  loss_circuit_use_kline: true;
}

export interface LeaderSpotV19MovingStopTier {
  profit_pct: number;
  stop_multiplier: number;
}

export interface LeaderSpotV19LayeredExitTier {
  minimum_profit_pct: number;
  maximum_profit_pct: number | null;
  retracement_pct: number;
  floor_multiplier: number;
}

export interface LeaderSpotV19TrendConfig {
  standard_profit_threshold: number;
  degraded_profit_threshold: number;
  ema_fast_period: number;
  ema_slow_period: number;
  confirm_bars: 2;
  limit_wait_seconds: 60;
  limit_offset: 0.002;
}

export interface LeaderSpotV19ProfitConfig {
  protection_profit_pct: 0.05;
  protection_hold_seconds: 60;
  moving_stop_track_mode: "peak_profit";
  moving_stop_initial_multiplier: 1.02;
  moving_stop_tiers: LeaderSpotV19MovingStopTier[];
  layered_use_current: true;
  peak_update_source: "1min_kline";
  peak_not_retreat: true;
  layered_exit_tiers: LeaderSpotV19LayeredExitTier[];
  trend: LeaderSpotV19TrendConfig;
}

export interface LeaderSpotV19MarketConfig {
  up_ratio_standard: number;
  volume_ratio_standard: number;
  fear_greed_standard: number;
  fear_greed_degraded: number;
  funding_rate_standard_min: number;
  funding_rate_standard_max: number;
  funding_rate_degraded_min: number;
  funding_rate_degraded_max: number;
  up_ratio_halt: number;
  standard_allow_fail: 1;
  strong_trend_up_ratio: number;
  strong_trend_volume_ratio: number;
}

export interface LeaderSpotV19CandidateConfig {
  relative_strength_rank_pct_standard: number;
  relative_strength_rank_pct_starved: number;
  liquidity_quote_standard: number;
  liquidity_quote_starved: number;
  score_standard: number;
  score_degraded: number;
  score_starved: number;
  signal_starve_48h_enabled: true;
  signal_starve_72h_enabled: true;
  signal_recover_count: 2;
  high_pump_24h_pct: 0.4;
}

export interface LeaderSpotV19BreakoutConfig {
  box_params_source: "V16.1";
  box_volume_degraded_add: 0.2;
  confirm_a: "15min_close";
  confirm_b: "5min_x2";
  second_confirmation_volume_required: true;
  needle_amplitude_pct: 0.12;
  needle_br_threshold: 280;
}

export interface LeaderSpotV19EntryConfig {
  mode: "limit";
  tier1_offset: 0.003;
  tier1_wait_seconds: 300;
  tier2_offset: 0.005;
  tier2_wait_seconds: 180;
  tier3_offset: 0.008;
  tier3_wait_seconds: 120;
  tier3_failure: "cancel";
  slippage_check: true;
}

export interface LeaderSpotV19NewCoinConfig {
  ban_hours: 6;
  strict_hours: 24;
  degraded_hours: 72;
  degraded_volume_add: 0.2;
  degraded_score_add: 7;
}

export interface LeaderSpotV19AccountRiskConfig {
  daily_loss_reset: "00:00";
  equity_drawdown_halt_pct: 0.15;
  equity_halt_hours: 24;
  equity_calculation_interval_seconds: 60;
}

export interface LeaderSpotV19DataConfig {
  check_interval_seconds: 60;
  api_retry_max: 3;
  api_retry_interval_seconds: 1;
  api_reconnect_interval_seconds: 5;
  timestamp_tolerance_seconds: 30;
  price_source_deviation_pct: 0.05;
  btc_cross_source_deviation_pct: 0.02;
  log_retention_days: 30;
}

export interface LeaderSpotV19BacktestConfig {
  period_months: number;
  fee_pct: 0.001;
  slippage_pct: 0.005;
  data_type: "1min_or_tick";
  walk_forward_train_ratio: 0.7;
  walk_forward_window_months: 3;
}

export interface LeaderSpotV19Config {
  module_id: typeof LEADER_SPOT_V19_MODULE_ID;
  schema_version: typeof LEADER_SPOT_V19_SCHEMA_VERSION;
  environment: LeaderSpotV19Environment;
  sandbox_connection_id: string | null;
  position: LeaderSpotV19PositionConfig;
  loss: LeaderSpotV19LossConfig;
  profit: LeaderSpotV19ProfitConfig;
  market: LeaderSpotV19MarketConfig;
  candidate: LeaderSpotV19CandidateConfig;
  breakout: LeaderSpotV19BreakoutConfig;
  entry: LeaderSpotV19EntryConfig;
  new_coin: LeaderSpotV19NewCoinConfig;
  account_risk: LeaderSpotV19AccountRiskConfig;
  data: LeaderSpotV19DataConfig;
  backtest: LeaderSpotV19BacktestConfig;
  daily_loss_limit_quote: number;
}

export interface LeaderSpotV19CreateRequest {
  name: string;
  description?: string | null;
  config: LeaderSpotV19Config;
}

export interface LeaderSpotV19StateSnapshot {
  data_state: LeaderSpotV19DataState;
  market_state: LeaderSpotV19MarketState;
  can_open: boolean;
  positions_count: number;
  max_positions: 6;
  daily_realized_loss_quote: number;
  daily_loss_limit_quote: number;
  equity_drawdown_pct: number | null;
  account_halt_until: string | null;
}

export interface LeaderSpotV19Summary {
  strategy_id: string;
  name: string;
  description: string | null;
  status: LeaderSpotV19StrategyStatus;
  environment: LeaderSpotV19Environment;
  module_id: typeof LEADER_SPOT_V19_MODULE_ID;
  schema_version: typeof LEADER_SPOT_V19_SCHEMA_VERSION;
  current_batch_id: string | null;
  state: LeaderSpotV19StateSnapshot | null;
}
