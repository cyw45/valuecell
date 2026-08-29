export type StrategyKind =
  | "configurable_rule"
  | "dual_ma_trend"
  | "pair_rotation"
  | "leader_breakout";

export type StrategyParameterSource = "configurable" | "code";
export type StrategyExecutionEnvironment = "paper" | "okx_demo";
export type StrategyStatus = "running" | "stopped" | "archived" | "paused";
export type AllocationState =
  | "available"
  | "reserved"
  | "occupied"
  | "partially_released"
  | "released"
  | "blocked";

export type StrategyDefinition = {
  kind: StrategyKind;
  display_name: string;
  description: string;
  rule_source: string;
  strategy_version: string;
  parameter_source: StrategyParameterSource;
  editable: boolean;
  execution_environments: StrategyExecutionEnvironment[];
};

export type StrategyIdentity = {
  strategy_id: string;
  tenant_id: string;
  kind: StrategyKind;
  strategy_version: string;
  code_fingerprint: string;
};

export type StrategyAllocation = {
  strategy_id: string;
  kind: StrategyKind;
  reserved_quote: number;
  occupied_quote: number;
  released_quote: number;
  realized_pnl_quote: number | null;
  unrealized_pnl_quote: number | null;
  net_pnl_quote: number | null;
  allocation_state: AllocationState;
  utilization_denominator_quote: number;
};

export type SharedWalletSummary = {
  tenant_id: string;
  credential_id: string;
  environment: StrategyExecutionEnvironment;
  total_equity_quote: number | null;
  available_quote: number | null;
  currency_balances: Record<string, number | null>;
  observed_at: string;
  sync_status: "healthy" | "stale" | "unavailable";
  attribution_status: "complete" | "partial" | "unavailable";
  unassigned_equity_quote: number | null;
};

export type CapitalAllocatorSummary = {
  wallet_equity_quote: number | null;
  available_for_strategies_quote: number | null;
  reserved_quote: number;
  occupied_notional_quote: number;
  pending_settlement_quote: number;
  reusable_quote: number | null;
  utilization_denominator_quote: number;
  account_utilization_ratio: number;
  allocations: StrategyAllocation[];
  observed_at: string;
};

export type AccountStrategyOverview = {
  wallet: SharedWalletSummary;
  allocator: CapitalAllocatorSummary;
  strategy_pnl_total_quote: number | null;
  wallet_strategy_reconciliation_delta_quote: number | null;
  data_complete: boolean;
  incomplete_reason: string | null;
};

export type ExplanationCondition = {
  code: string;
  label: string;
  state: "triggered" | "not_triggered" | "blocked" | "unavailable";
  actual: number | string | boolean | null;
  threshold: number | string | boolean | null;
  operator: string | null;
  detail: string;
  data_at: string | null;
};

export type TradeExplanation = {
  decision: string;
  decision_reason: string;
  conditions: ExplanationCondition[];
  execution_path: string | null;
  risk_check: string | null;
  block_reason: string | null;
  final_result: string | null;
};

export type UnifiedTradeFact = {
  identity: StrategyIdentity;
  batch_id: string | null;
  evaluation_id: string | null;
  intent_id: string | null;
  order_id: string | null;
  fill_id: string | null;
  symbol: string;
  pair: string | null;
  side: "buy" | "sell" | "short" | "cover";
  status:
    | "signal"
    | "blocked"
    | "pending"
    | "submitted"
    | "partially_filled"
    | "filled"
    | "cancelled"
    | "failed";
  requested_quote: number | null;
  filled_quote: number | null;
  requested_quantity: number | null;
  filled_quantity: number | null;
  average_fill_price: number | null;
  fee_quote: number | null;
  execution_cost_quote: number | null;
  borrow_cost_quote: number | null;
  created_at: string;
  filled_at: string | null;
  failure_code: string | null;
  failure_reason: string | null;
  explanation: TradeExplanation;
};
