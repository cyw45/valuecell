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
  funding_rate?: number;
}

export interface RuleStrategyPaperPosition {
  quantity: number;
  entry_price: number;
  mark_price: number;
}

export interface RuleStrategyPaperAccount {
  initial_capital_quote: number;
  quote_balance: number;
  positions: Record<string, RuleStrategyPaperPosition>;
  realized_pnl_quote: number;
  unrealized_pnl_quote: number;
  equity_quote: number;
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

export type RuleStrategyCandleInterval =
  | "1m"
  | "3m"
  | "5m"
  | "15m"
  | "30m"
  | "1h"
  | "4h"
  | "1d";

export interface AdvancedMovingAverageRuleConfig {
  enabled: boolean;
  interval: RuleStrategyCandleInterval;
  period: number;
  entry_comparator: "above" | "below";
}

export interface AdvancedMacdRuleConfig {
  enabled: boolean;
  interval: RuleStrategyCandleInterval;
  fast_window: number;
  slow_window: number;
  signal_window: number;
  entry_cross: "golden" | "death";
}

export interface AdvancedBollingerRuleConfig {
  enabled: boolean;
  interval: RuleStrategyCandleInterval;
  period: number;
  standard_deviations: number;
  entry_reference: "upper" | "middle" | "lower";
  entry_comparator: "above" | "below";
}

export interface AdvancedThresholdRuleConfig {
  enabled: boolean;
  interval: RuleStrategyCandleInterval;
  period: number;
  entry_comparator: "above" | "below";
  entry_threshold: number;
  exit_enabled: boolean;
  exit_comparator: "above" | "below";
  exit_threshold: number;
}

export interface AdvancedBrarRuleConfig extends AdvancedThresholdRuleConfig {
  component: "ar" | "br";
}

export interface AdvancedRuleSetConfig {
  enabled: boolean;
  entry_confirmation_mode: "all" | "any";
  exit_confirmation_mode: "all" | "any";
  moving_average: AdvancedMovingAverageRuleConfig;
  macd: AdvancedMacdRuleConfig;
  bollinger: AdvancedBollingerRuleConfig;
  rsi: AdvancedThresholdRuleConfig;
  momentum: AdvancedThresholdRuleConfig;
  brar: AdvancedBrarRuleConfig;
}

export interface RuleStrategyRiskConfig {
  order_quote_amount: number;
  take_profit_pct?: number;
  stop_loss_pct?: number;
  max_positions: number;
  leverage: number;
}

export type RuleStrategyInterval = RuleStrategyCandleInterval;

export interface RuleStrategyExecutionConfig {
  environment: "paper" | "okx_demo";
  sandbox_connection_id?: string;
  max_order_quote_amount: number;
  max_daily_quote_amount: number;
  max_total_quote_amount: number;
}

export interface RuleStrategyConfig {
  mode: "paper";
  initial_capital_quote: number;
  confirmation_mode: "all" | "any";
  symbols: string[];
  interval: RuleStrategyInterval;
  decide_interval_s?: number | null;
  moving_average: MovingAverageRuleConfig;
  rsi: RsiRuleConfig;
  bollinger: BollingerRuleConfig;
  momentum_macd: MomentumMacdRuleConfig;
  advanced_rules: AdvancedRuleSetConfig;
  execution: RuleStrategyExecutionConfig;
  risk: RuleStrategyRiskConfig;
}

export interface RuleStrategy {
  strategy_id: string;
  name: string;
  description: string | null;
  status: RuleStrategyStatus;
  mode: "paper";
  config: RuleStrategyConfig;
  account: RuleStrategyPaperAccount;
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
  mode: "fixed_quote";
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
  account: RuleStrategyPaperAccount;
}

export interface RuleStrategyTextImportConfig {
  interval: RuleStrategyCandleInterval;
  advanced_rules: AdvancedRuleSetConfig;
  risk: RuleStrategyRiskConfig;
}

export interface RuleStrategyTextImportProposal {
  strategy_name: string | null;
  config: RuleStrategyTextImportConfig;
  summary: string;
  unresolved_items: string[];
}

export interface RuleStrategyEvaluationHistoryEntry
  extends RuleStrategyEvaluation {
  symbol?: string;
  evaluated_at: string;
  trades: RuleStrategyTradeLogEntry[];
}

export interface RuleStrategyAdvisory {
  kind: "configuration_review";
  authority: "advisory_only";
  provider: string;
  model_id: string;
  content: string;
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
  execution: "paper_filled";
  symbol: string;
  price: number;
  quantity: number;
  quote_amount: number;
  realized_pnl_quote: number;
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
  initial_capital_quote: number;
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

export interface RuleStrategyPnlPoint {
  ts: string;
  cumulative_pnl: number;
  action: string;
}
