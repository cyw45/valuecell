export type TenantRole =
  | "owner"
  | "admin"
  | "strategist"
  | "trader"
  | "viewer"
  | "billing_manager";

export type ExecutionEnvironment = "paper" | "okx_demo";

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

export interface SaaSAccess {
  role: TenantRole;
  tenant_type: "personal" | "enterprise";
  organization_name: string | null;
  is_platform_admin: boolean;
  status: "active" | "pending_activation";
  commercial_model: "subscription" | "revenue_share" | null;
  expires_at: string | null;
}

export interface Workspace {
  tenant_id: string;
  name: string;
  tenant_type: "personal" | "enterprise";
  organization_name: string | null;
  role: TenantRole;
  selected: boolean;
}

export interface StrategyExecution {
  environment: ExecutionEnvironment;
  sandbox_connection_id?: string;
  max_order_quote_amount: number;
  max_daily_quote_amount: number;
  max_total_quote_amount: number;
}

export interface StrategyRisk {
  order_quote_amount: number;
  max_positions: number;
  leverage: number;
  take_profit_pct?: number | null;
  stop_loss_pct?: number | null;
}

export interface StrategyConfig {
  symbols: string[];
  interval: "1m" | "3m" | "5m" | "15m" | "30m" | "1h" | "4h" | "1d";
  initial_capital_quote: number;
  execution: StrategyExecution;
  risk: StrategyRisk;
  [key: string]: unknown;
}

export interface StrategyAccount {
  quote_balance: number;
  equity_quote: number;
  realized_pnl_quote: number;
  unrealized_pnl_quote: number;
  positions: Record<string, { quantity: number; entry_price: number; mark_price: number }>;
}

export interface Strategy {
  strategy_id: string;
  name: string;
  description: string | null;
  status: "running" | "stopped";
  config: StrategyConfig;
  account: StrategyAccount;
  created_at?: string;
  updated_at?: string;
}

export interface Candle {
  ts: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface IndicatorPoint {
  ts: number;
  ma: Record<string, number | null | undefined>;
  rsi?: number | null;
}

export interface MarketSymbol {
  symbol: string;
  exchange_symbol: string;
  provider: string;
  interval: string;
  candles: Candle[];
  indicators: IndicatorPoint[];
  latest_price?: number | null;
  freshness_status: "fresh" | "stale" | "unknown";
}

export interface MarketResponse {
  interval: string;
  symbols: MarketSymbol[];
  failed_symbols: Record<string, string>;
}

export interface DemoConnection {
  id: string;
  label: string;
  provider: "binance" | "okx";
  metadata: {
    sandbox?: boolean;
    market_type?: "spot" | "swap";
    validated_at?: string;
  };
  revoked: boolean;
}
