import type { NavigatorScreenParams } from "@react-navigation/native";

export type PostAuthTab = "工作台" | "我的";

export type WorkbenchStackParamList = {
  StrategyOverview: { strategyId?: string } | undefined;
  StrategySymbols: { strategyId: string };
  ExecutionFacts: { strategyId: string; kind: "positions" | "balances" | "orders" };
  TradeLedger: { strategyId?: string } | undefined;
  FundingPnl: { strategyId?: string } | undefined;
};

export type StrategyStackParamList = {
  StrategyList: undefined;
  StrategyDetail: { strategyId: string };
  StrategyEditor: { strategyId?: string } | undefined;
  StrategyAdvisory: { strategyId: string };
};

export type MarketStackParamList = {
  Market: { strategyId?: string; symbol?: string } | undefined;
  WorldMonitor: undefined;
  Polymarket: { marketId?: string; outcome?: string } | undefined;
};

export type AccountStackParamList = {
  Account: undefined;
  Preferences: undefined;
  SandboxConnections: undefined;
  SandboxConnectionEditor: { connectionId?: string } | undefined;
  SandboxConnectionDetail: { connectionId: string };
  LiveExecution: undefined;
  WorkspaceMembers: undefined;
  WorkspaceBilling: undefined;
  WorkspaceAudit: undefined;
  PlatformAdmin: undefined;
};

export type WorkspaceTabParamList = {
  工作台: NavigatorScreenParams<WorkbenchStackParamList> | undefined;
  策略: NavigatorScreenParams<StrategyStackParamList> | undefined;
  行情: NavigatorScreenParams<MarketStackParamList> | undefined;
  我的: NavigatorScreenParams<AccountStackParamList> | undefined;
};
