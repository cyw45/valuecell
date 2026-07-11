export type RuleStrategyStatus = "running" | "stopped";
export type RuleStrategyAction = "buy" | "sell" | "no_op";
export type RuleConditionState =
  | "triggered"
  | "not_triggered"
  | "blocked"
  | "unavailable";

export interface RuleStrategyCandle {
  timestamp_ms: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface RuleStrategyPosition {
  quantity: number;
  entry_price?: number;
}

export interface RuleStrategyMarketSnapshot {
  symbol: string;
  price: number;
  equity_quote: number;
  quote_balance: number;
  open_position_count?: number;
  funding_rate?: number;
  position?: RuleStrategyPosition;
}

export interface MovingAverageRuleConfig {
  enabled: boolean;
  short_window: number;
  long_window: number;
}

export interface RsiRuleConfig {
  enabled: boolean;
  period: number;
  oversold: number;
  overbought: number;
}

export interface BollingerRuleConfig {
  enabled: boolean;
  period: number;
  standard_deviations: number;
}

export interface MomentumMacdRuleConfig {
  enabled: boolean;
  momentum_period: number;
  macd_fast_window: number;
  macd_slow_window: number;
  macd_signal_window: number;
}

export interface RuleStrategyRiskConfig {
  size_mode: "fixed_quote" | "equity_fraction";
  size_value: number;
  take_profit_pct?: number;
  stop_loss_pct?: number;
  max_positions: number;
  leverage: number;
}

export type RuleStrategyInterval = "5m" | "15m" | "1h" | "4h" | "1d";

export interface RuleStrategyConfig {
  mode: "paper";
  confirmation_mode: "all" | "any";
  symbols: string[];
  interval: RuleStrategyInterval;
  decide_interval_s?: number | null;
  moving_average: MovingAverageRuleConfig;
  rsi: RsiRuleConfig;
  bollinger: BollingerRuleConfig;
  momentum_macd: MomentumMacdRuleConfig;
  risk: RuleStrategyRiskConfig;
}

export interface RuleStrategy {
  strategy_id: string;
  name: string;
  description: string | null;
  status: RuleStrategyStatus;
  mode: "paper";
  config: RuleStrategyConfig;
  created_at?: string;
  updated_at?: string;
}

export interface RuleStrategyCondition {
  code: string;
  category: "indicator" | "exit" | "risk";
  state: RuleConditionState;
  detail: string;
  values: Record<string, number | string | boolean | null>;
}

export interface RuleStrategyIndicators {
  moving_average_short: number | null;
  moving_average_long: number | null;
  previous_moving_average_short: number | null;
  previous_moving_average_long: number | null;
  rsi: number | null;
  bollinger_upper: number | null;
  bollinger_middle: number | null;
  bollinger_lower: number | null;
  momentum: number | null;
  macd: number | null;
  macd_signal: number | null;
  previous_macd: number | null;
  previous_macd_signal: number | null;
}

export interface RuleStrategySizing {
  mode: "fixed_quote" | "equity_fraction";
  requested_quote: number;
  max_allowed_quote: number;
  affordable_quote: number;
  quantity: number;
}

export interface RuleStrategyFundingImpact {
  funding_rate: number;
  current_notional_quote: number;
  projected_notional_quote: number;
  estimated_payment_quote: number;
  direction: "credit" | "debit" | "none";
}

export interface RuleStrategyEvaluation {
  strategy_id: string;
  evaluation_id: string;
  mode: "paper";
  action: RuleStrategyAction;
  reason_code: string;
  reason: string;
  conditions: RuleStrategyCondition[];
  indicators: RuleStrategyIndicators;
  sizing: RuleStrategySizing;
  funding: RuleStrategyFundingImpact;
}

export interface RuleStrategyLogEntry extends RuleStrategyCondition {
  evaluation_id: string;
  evaluated_at: string;
}

export interface RuleStrategyTradeLogEntry {
  evaluation_id: string;
  evaluated_at: string;
  action: Exclude<RuleStrategyAction, "no_op">;
  reason_code: string;
  reason: string;
  sizing: RuleStrategySizing;
  execution: "not_submitted";
}

export interface RuleStrategyFundingLogEntry extends RuleStrategyFundingImpact {
  evaluation_id: string;
  evaluated_at: string;
}

export interface RuleStrategyLog<T> {
  strategy_id: string;
  mode: "paper";
  entries: T[];
}

export interface CreateRuleStrategyRequest {
  name: string;
  description?: string;
  config: RuleStrategyConfig;
}

export interface UpdateRuleStrategyRequest {
  name?: string;
  description?: string;
  config?: RuleStrategyConfig;
}

export interface EvaluateRuleStrategyRequest {
  candles: RuleStrategyCandle[];
  market: RuleStrategyMarketSnapshot;
}

export interface RuleStrategyPnlPoint { ts: string; cumulative_pnl: number; action: string; }
