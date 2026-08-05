export type TenantRole =
  | "owner"
  | "admin"
  | "strategist"
  | "trader"
  | "viewer"
  | "billing_manager";

export type TenantType = "personal" | "enterprise";
export type ExecutionEnvironment = "paper" | "okx_demo";
export type RuleStrategyStatus = "running" | "stopped" | "archived";
export type RuleStrategyAction = "entry" | "add" | "reduce" | "close" | "buy" | "sell" | "no_op";
export type RuleStrategyConditionState =
  | "triggered"
  | "not_triggered"
  | "blocked"
  | "unavailable";
export type RuleConditionState = RuleStrategyConditionState;
export type RuleStrategyCandleInterval =
  | "1m"
  | "3m"
  | "5m"
  | "15m"
  | "30m"
  | "1h"
  | "4h"
  | "1d";
export type RuleStrategyInterval = RuleStrategyCandleInterval;

/** Device-local session data; it is not a server wire payload. */
export interface Session {
  accessToken: string;
  userId: string;
  tenantId: string;
  email: string;
}

export interface ApiEnvelope<T> {
  code: number;
  data: T;
  msg: string;
}

// SaaS authentication, access, and workspace contracts.
export interface SaaSRegisterRequest {
  email: string;
  password: string;
  tenant_type: TenantType;
  workspace_name: string;
  organization_name?: string;
}

export interface SaaSLoginRequest {
  email: string;
  password: string;
}

export interface SaaSAuthResponse {
  access_token: string;
  token_type?: "bearer";
  user_id: string;
  tenant_id: string;
  email: string;
  tenant_type?: TenantType;
  organization_name?: string | null;
}

export interface SaaSMeResponse {
  user_id: string;
  tenant_id: string;
  role: TenantRole;
  is_platform_admin: boolean;
  access_status: "active" | "pending_activation";
  commercial_model: "subscription" | "revenue_share" | null;
  access_expires_at: string | null;
}

export interface SaaSAccess {
  role: TenantRole;
  tenant_type: TenantType;
  organization_name: string | null;
  is_platform_admin: boolean;
  status: "active" | "pending_activation";
  commercial_model: "subscription" | "revenue_share" | null;
  expires_at: string | null;
}

export interface Workspace {
  tenant_id: string;
  name: string;
  tenant_type: TenantType;
  organization_name: string | null;
  role: TenantRole;
  selected: boolean;
}

export interface WorkspaceMember {
  user_id: string;
  email: string;
  role: TenantRole;
  created_at: string;
}

export interface SaveWorkspaceMemberRequest {
  email: string;
  role: TenantRole;
}

export interface AuditEvent {
  id: string;
  tenant_id: string | null;
  actor_user_id: string | null;
  action: string;
  target_type: string;
  target_id: string;
  outcome: string;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface Subscription {
  id: string;
  tenant_id: string;
  plan_id: string;
  status: string;
  starts_at: string;
  ends_at: string;
  note: string | null;
}

export interface EnterpriseAgreement {
  id: string;
  tenant_id: string;
  agreement_number: string;
  status: string;
  revenue_share_rate: string;
  settlement_cycle_days: number;
  high_water_mark_quote: string;
  starts_at: string;
  ends_at: string | null;
}

export interface ProfitSettlement {
  id: string;
  connection_id: string;
  period_started_at: string;
  period_ended_at: string;
  ending_equity_quote: string;
  eligible_profit_quote: string;
  revenue_share_rate: string;
  amount_due_quote: string;
  status: string;
}

export interface SaaSAccessWithEntitlements extends SaaSAccess {
  entitlements: Record<string, unknown>;
}

export interface TenantBilling {
  access: SaaSAccessWithEntitlements;
  subscriptions: Subscription[];
  agreement: EnterpriseAgreement | null;
  settlements: ProfitSettlement[];
}

export interface ServicePlan {
  id: string;
  code: string;
  name: string;
  duration_days: number;
  price_cents: number;
  currency: string;
  entitlements: Record<string, number | boolean>;
  active: string;
}

export interface AdminTenant {
  id: string;
  name: string;
  tenant_type: TenantType;
  organization_name: string | null;
  created_at: string;
  access: SaaSAccessWithEntitlements;
}

export interface CreatePlanRequest {
  code: string;
  name: string;
  duration_days: number;
  price_cents: number;
  currency?: string;
}

export interface GrantSubscriptionRequest {
  tenant_id: string;
  plan_id: string;
  ends_at: string;
  note?: string;
}

export interface UpdateTenantProfileRequest {
  tenant_id: string;
  tenant_type: TenantType;
  organization_name?: string;
}

export interface TenantProfile {
  tenant_id: string;
  tenant_type: TenantType;
  organization_name: string | null;
}

export interface CreateEnterpriseAgreementRequest {
  tenant_id: string;
  agreement_number: string;
  revenue_share_rate: string;
  settlement_cycle_days: number;
  starts_at: string;
}

// Rule-strategy contracts.
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
  entry_price?: number | null;
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
  take_profit_pct?: number | null;
  stop_loss_pct?: number | null;
  max_positions: number;
  leverage: number;
}

export interface RuleStrategyExecutionConfig {
  environment: ExecutionEnvironment;
  sandbox_connection_id?: string | null;
  max_order_quote_amount: number;
  max_daily_quote_amount: number;
  max_total_quote_amount: number;
}

export interface RuleStrategyConfig {
  mode: "paper";
  initial_capital_quote: number;
  confirmation_mode: "all" | "any";
  symbols: string[];
  interval: RuleStrategyCandleInterval;
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
  mode: ExecutionEnvironment;
  config: RuleStrategyConfig;
  account: RuleStrategyPaperAccount;
  execution_generation?: number;
  archived_at: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface RuleStrategyCondition {
  code: string;
  category: "indicator" | "exit" | "risk";
  state: RuleStrategyConditionState;
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
  brar_ar?: number | null;
  brar_br?: number | null;
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
  config?: RuleStrategyConfig;
  execution_ledger?: "external";
  paper_fill?: boolean;
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
  description?: string | null;
  config?: RuleStrategyConfig;
}

export interface EvaluateRuleStrategyRequest {
  candles: RuleStrategyCandle[];
  market: RuleStrategyMarketSnapshot;
}

export interface RuleStrategyPnlPoint {
  ts: string;
  cumulative_pnl: number;
  equity_quote?: number;
  action: string;
}

// Sandbox exchange contracts. Raw credential fields exist only in create requests.
export type SandboxExchangeProvider = "binance" | "okx";
export type SandboxOrderSide = "buy" | "sell";
export type SandboxOrderType = "market" | "limit";

export interface SandboxConnectionMetadata {
  sandbox: true;
  provider: SandboxExchangeProvider;
  market_type: "spot";
  validated_at: string;
}

export interface SandboxConnection {
  id: string;
  label: string;
  provider: SandboxExchangeProvider;
  metadata: SandboxConnectionMetadata;
  created_at: string;
}

export interface SavedSandboxConnection extends SandboxConnection {
  kind: "exchange";
  revoked: boolean;
  revoked_at: string | null;
}

export interface CreateSandboxConnectionRequest {
  provider: SandboxExchangeProvider;
  label: string;
  api_key: string;
  api_secret: string;
  passphrase?: string;
}

export interface SandboxBalance {
  currency: string;
  free: string | number;
  used: string | number;
  frozen: string | number;
  total: string | number;
  mark_price_usdt: number | null;
  usdt_value: number | null;
  valuation_status: "priced" | "unpriced";
}

export interface SandboxConnectionBalance {
  source: "okx_demo" | "binance_demo";
  balances: SandboxBalance[];
  total_usdt_value: number;
  checked_at: string;
}

export interface SandboxPosition {
  symbol: string;
  base_currency: string;
  quantity: number;
  available_quantity: number;
  frozen_quantity: number;
  mark_price: number | null;
  notional_usdt: number | null;
  unrealized_pnl_usdt: null;
}

export interface SandboxPositions {
  source: "okx_demo" | "binance_demo";
  positions: SandboxPosition[];
  checked_at: string;
}

export interface SandboxSymbol {
  symbol: string;
  base: string;
  quote: "USDT";
}

export interface CreateSandboxOrderRequest {
  credential_id: string;
  symbol: string;
  side: SandboxOrderSide;
  type: SandboxOrderType;
  quote_amount: number;
  price?: number;
  idempotency_key: string;
  sandbox: true;
}

export type CreateSandboxOrderSubmission = Omit<
  CreateSandboxOrderRequest,
  "idempotency_key"
>;

export interface SandboxOrder {
  id: string;
  credential_id: string;
  provider: SandboxExchangeProvider;
  client_order_id: string;
  symbol: string;
  side: SandboxOrderSide;
  type: SandboxOrderType;
  requested_quote: string | number;
  requested_quantity?: string | number | null;
  status: string;
  exchange_order_id?: string | null;
  sandbox: true;
  error_code?: string | null;
  strategy_id?: string | null;
  evaluation_id?: string | null;
  execution_generation?: number | null;
  execution_source?: string | null;
  execution_intent_id?: string | null;
  created_at: string;
  updated_at: string;
}

// Live execution contracts. Raw credential fields exist only in create requests.
export type LiveExchangeProvider = "binance" | "okx";
export type LiveMarketType = "spot" | "swap";
export type LiveOrderSide = "buy" | "sell";
export type LiveOrderType = "market" | "limit";

export interface LiveExecutionStatus {
  live_trading_enabled: boolean;
  authorization_active: boolean;
  authorization_expires_at: string | null;
  gate_reasons: string[];
}

export interface LiveConnection {
  id: string;
  label: string;
  provider: LiveExchangeProvider;
  market_type: LiveMarketType;
  active: boolean;
  created_at: string;
}

export interface CreateLiveConnectionRequest {
  label: string;
  provider: LiveExchangeProvider;
  market_type: LiveMarketType;
  api_key: string;
  api_secret: string;
  passphrase?: string;
  withdrawal_disabled_confirmed: boolean;
  ip_allowlist_confirmed: boolean;
}

export interface LiveRiskPolicy {
  id: string;
  max_order_notional: number;
  max_total_notional: number;
  max_daily_loss: number;
  max_open_positions: number;
  max_leverage: number;
  allowed_symbols: string[];
  active: boolean;
}

export interface SaveLiveRiskPolicyRequest {
  max_order_notional: number;
  max_open_positions: number;
  max_leverage: number;
  allowed_symbols: string[];
  max_total_notional?: number;
  max_daily_loss?: number;
}

export interface LiveStrategyBinding {
  id: string;
  strategy_id: string;
  connection_id: string;
  active: boolean;
  revoked_at: string | null;
  created_at: string;
}

export interface CreateLiveStrategyBindingRequest {
  strategy_id: string;
  connection_id: string;
}

export interface StartupAuthorizationChallenge {
  challenge_code: string;
  expires_at: string;
}

export interface ConfirmStartupAuthorizationRequest {
  challenge_code: string;
}

export interface StartupAuthorizationConfirmation {
  authorization_expires_at: string;
}

export interface StartupAuthorizationRevocation {
  authorization_active: false;
}

export interface CreateLiveOrderRequest {
  connection_id: string;
  symbol: string;
  side: LiveOrderSide;
  type: LiveOrderType;
  quote_amount: number;
  price?: number;
  idempotency_key: string;
}

export type CreateLiveOrderSubmission = Omit<
  CreateLiveOrderRequest,
  "idempotency_key"
>;

export interface LiveOrder {
  id: string;
  status: string;
  exchange_order_id: string | null;
  created_at: string;
}

export interface LivePosition {
  symbol: string | null;
  contracts: string | number | null;
  notional: string | number | null;
  entry_price: string | number | null;
  mark_price: string | number | null;
  side: string | null;
}

// Crypto market contracts.
export interface CryptoCandle {
  ts: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface CryptoBollingerBand {
  upper?: number | null;
  middle?: number | null;
  lower?: number | null;
}

export interface CryptoIndicatorPoint {
  ts: number;
  ma: Record<string, number | null | undefined>;
  rsi?: number | null;
  bollinger: CryptoBollingerBand;
  momentum?: number | null;
  macd?: number | null;
  macd_signal?: number | null;
  macd_histogram?: number | null;
}

export interface CryptoSymbolIndicators {
  symbol: string;
  exchange_symbol: string;
  provider: string;
  interval: string;
  candles: CryptoCandle[];
  indicators: CryptoIndicatorPoint[];
  latest_price?: number | null;
  warning?: string | null;
  snapshot_ts_ms?: number | null;
  freshness_age_ms?: number | null;
  freshness_status: "fresh" | "stale" | "unknown";
  coverage_status: "complete" | "partial";
}

export interface CryptoMarketIndicators {
  interval: string;
  lookback: number;
  providers: string[];
  symbols: CryptoSymbolIndicators[];
  snapshot_fetched_at?: string | null;
  failed_symbols: Record<string, string>;
}

export interface CryptoSymbolCatalog {
  quote_asset: string;
  symbols: string[];
}

export interface CryptoMarketQueryOptions {
  providers?: string[];
  from_ts_ms?: number;
  to_ts_ms?: number;
}

// Public prediction-market research contracts.
export type PredictionMarketMode = "paper";
export type PredictionMarketFreshnessStatus =
  | "fresh"
  | "delayed"
  | "stale"
  | "unavailable";

export interface PredictionMarketOutcome {
  outcome: string;
  token_id: string;
  price?: string | null;
}

export interface PredictionMarketSummary {
  market_id: string;
  slug: string;
  question: string;
  active: boolean;
  closed: boolean;
  outcomes: PredictionMarketOutcome[];
}

export interface PredictionMarketCatalog {
  source: "polymarket-public";
  mode: PredictionMarketMode;
  source_timestamp_ms: number;
  observed_at_ms: number;
  freshness_age_ms: number;
  freshness_status: PredictionMarketFreshnessStatus;
  markets: PredictionMarketSummary[];
  next_cursor?: string | null;
  warnings?: string[];
}

export interface PredictionMarketBookLevel {
  price: string;
  size: string;
}

export interface PredictionMarketBookHealth {
  status?: "valid" | "crossed" | "one_sided" | "empty" | "invalid" | "stale";
  reason?: string | null;
  crossed?: boolean;
  one_sided?: boolean;
  bid_levels?: number;
  ask_levels?: number;
}

export interface PredictionMarketOrderBook {
  bids: PredictionMarketBookLevel[];
  asks: PredictionMarketBookLevel[];
  best_bid?: string | null;
  best_ask?: string | null;
  midpoint?: string | null;
  microprice?: string | null;
  health?: PredictionMarketBookHealth;
}

export interface PredictionMarketSignal {
  reference_price?: string | null;
  reference_method?: "microprice" | "midpoint" | "unavailable";
  volatility?: string | null;
  observation_count?: number;
  volatility_status?: "available" | "insufficient_history" | "invalid_history";
}

export interface PredictionMarketSnapshot {
  source: "polymarket-public";
  mode: PredictionMarketMode;
  source_timestamp_ms: number;
  observed_at_ms: number;
  freshness_age_ms: number;
  freshness_status: PredictionMarketFreshnessStatus;
  market_id: string;
  question: string;
  outcome: string;
  token_id: string;
  book: PredictionMarketOrderBook;
  signal?: PredictionMarketSignal | null;
  warnings?: string[];
}

export interface PredictionReplayRequest {
  decision_time_ms: number;
  latency_ms: number;
  order: {
    side: "buy" | "sell";
    size: number;
    max_levels: number;
    extra_slippage_bps: number;
  };
  snapshots: Array<{
    source_timestamp_ms: number;
    observed_at_ms: number;
    bids: PredictionMarketBookLevel[];
    asks: PredictionMarketBookLevel[];
  }>;
}

export interface PredictionReplayResult {
  source: "polymarket-public";
  mode: "paper";
  simulation_mode: "simulated";
  source_timestamp_ms: number | null;
  observed_at_ms: number | null;
  freshness_age_ms: number | null;
  freshness_status: "fresh" | "stale" | "unavailable";
  fingerprint: string;
  assumptions: {
    eligible_time_ms: number;
    execution_snapshot_timestamp_ms: number | null;
    max_levels: number;
    extra_slippage_bps: number;
    canceled_remainder?: boolean;
    remainder_policy: "cancel";
    liquidity_scope: "visible_frozen_levels";
  };
  fill: {
    requested_size: number;
    filled_size: number;
    unfilled_size: number;
    vwap: number | null;
    levels_consumed: number;
  };
  mark_to_book: {
    mark_price: number | null;
    pnl: number;
    currency: "quote";
  };
}

// WorldMonitor stored-evidence contracts.
export interface WorldIntelligenceFeedStatus {
  feed: string;
  latest_snapshot_at: string | null;
}

export interface WorldIntelligenceStatus {
  enabled: boolean;
  feeds: WorldIntelligenceFeedStatus[];
}

export interface WorldIntelligenceSnapshot {
  id: number;
  feed: string;
  payload: unknown;
  captured_at: string;
}

export interface WorldIntelligenceSnapshotList {
  snapshots: WorldIntelligenceSnapshot[];
}

export interface WorldIntelligenceSnapshotsRequest {
  feed?: string;
  limit?: number;
}

// Exchange-authoritative execution facts for an OKX Demo rule strategy.
export type DemoExecutionAccountScope = "exchange_connection_shared_account";
export type DemoExecutionPositionsScope =
  "exchange_connection_shared_spot_positions";

export interface RuleStrategyDemoExecutionAccount {
  scope: DemoExecutionAccountScope;
  data: SandboxConnectionBalance;
}

export interface RuleStrategyDemoExecutionPositions {
  scope: DemoExecutionPositionsScope;
  data: SandboxPositions;
}

export interface RuleStrategyDemoExecutionPnl {
  status: "unavailable";
  value: null;
  reason: string;
}

export interface RuleStrategyDemoExecution {
  source: "okx_demo_spot";
  strategy_id: string;
  connection_id: string | null;
  account: RuleStrategyDemoExecutionAccount;
  positions: RuleStrategyDemoExecutionPositions;
  orders: SandboxOrder[];
  pnl: RuleStrategyDemoExecutionPnl;
  checked_at: string;
}

// Existing mobile names remain the wire-contract names consumed by current screens.
export type Strategy = RuleStrategy;
export type StrategyConfig = RuleStrategyConfig;
export type StrategyExecution = RuleStrategyExecutionConfig;
export type StrategyRisk = RuleStrategyRiskConfig;
export type StrategyAccount = RuleStrategyPaperAccount;
export type Candle = CryptoCandle;
export type IndicatorPoint = CryptoIndicatorPoint;
export type MarketSymbol = CryptoSymbolIndicators;
export type MarketResponse = CryptoMarketIndicators;
export interface DemoConnection extends SandboxConnection {
  revoked?: boolean;
  revoked_at?: string | null;
}

export interface RuleStrategyMonitorState {
  symbol: string;
  state: "candidate" | "admitted" | "held" | "removed";
  reason_code: string | null;
  reason_detail: string | null;
  evaluated_at: string | null;
  next_check_at: string | null;
  protected_held: boolean;
}

export interface RuleStrategyRiskState {
  state: "normal" | "warn" | "only_reduce" | "halted";
  daily_equity_baseline: number;
  high_water_equity: number;
  current_drawdown_pct: number;
  cooldown_until: string | null;
  reason_code: string | null;
  reason_detail: string | null;
}
