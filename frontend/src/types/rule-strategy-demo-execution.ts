import type {
  SandboxConnectionBalance,
  SandboxOrder,
  SandboxPositions,
} from "@/types/sandbox-exchange";

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

export type DemoPurchaseState =
  | "not_bought"
  | "pending"
  | "partially_filled"
  | "bought"
  | "failed"
  | "unknown";

export interface RuleStrategyDemoExecutionPnl {
  status: "available" | "partial" | "unavailable";
  scope?: string | null;
  reason_code?: string | null;
  value?: number | string | null;
  total?: number | string | null;
  realized?: number | string | null;
  unrealized?: number | string | null;
  total_pnl?: number | string | null;
  realized_pnl?: number | string | null;
  unrealized_pnl?: number | string | null;
  return_pct?: number | null;
  reason?: string | null;
  fees_included?: boolean;
}

export interface DemoTradeSummary {
  purchase_state?: DemoPurchaseState | null;
  order_count?: number;
  filled_order_count?: number;
  partially_filled_order_count?: number;
  failed_order_count?: number;
  latest_order?: SandboxOrder | null;
  filled_buy_orders?: number;
  filled_sell_orders?: number;
  failed_orders?: number;
  submission_unknown_orders?: number;
  current_position_quantity?: string | number;
}

export interface DemoEquityCurvePoint {
  ts?: string;
  timestamp?: string;
  equity_quote?: number | null;
  equity?: number | null;
  value?: number | string | null;
  cumulative_pnl?: number | string | null;
  daily_pnl_quote?: number | string | null;
  total_pnl?: number | string | null;
  pnl?: number | string | null;
  action?: string | null;
}

export interface DemoEquityCurve {
  points?: DemoEquityCurvePoint[] | null;
}

export interface RuleStrategyDemoExecution {
  source: "okx_demo_spot";
  strategy_id: string;
  connection_id: string | null;
  account: RuleStrategyDemoExecutionAccount;
  positions: RuleStrategyDemoExecutionPositions;
  orders: SandboxOrder[];
  pagination: {
    page: number;
    page_size: number;
    total_items: number;
    total_pages: number;
  };
  trade_summary?: DemoTradeSummary | null;
  pnl: RuleStrategyDemoExecutionPnl;
  equity_curve?: DemoEquityCurve | null;
  checked_at: string;
}

type DemoExecutionTimestampSnapshot = {
  checked_at?: string;
  account: { data: { checked_at: string } };
};

type DemoExecutionValuationSnapshot = {
  account: {
    data: { balances: Array<{ valuation_status: "priced" | "unpriced" }> };
  };
};

export function demoExecutionCheckedAtLabel(
  snapshot: DemoExecutionTimestampSnapshot,
): string {
  return snapshot.checked_at || snapshot.account.data.checked_at;
}

export function demoExecutionUnvaluedAssetCount(
  snapshot: DemoExecutionValuationSnapshot,
): number {
  return snapshot.account.data.balances.filter(
    (balance) => balance.valuation_status === "unpriced",
  ).length;
}
