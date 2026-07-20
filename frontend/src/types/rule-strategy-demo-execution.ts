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
