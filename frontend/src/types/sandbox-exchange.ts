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
  entry_price?: number | null;
  unrealized_pnl_usdt: number | null;
}

export interface SandboxPositions {
  source: "okx_demo" | "binance_demo";
  positions: SandboxPosition[];
  checked_at: string;
}

export interface CreateSandboxOrderRequest {
  credential_id: string;
  symbol: string;
  side: SandboxOrderSide;
  type: SandboxOrderType;
  quote_amount: number;
  price?: number;
  idempotency_key: string;
}

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
  filled_quantity?: string | number | null;
  average_fill_price?: string | number | null;
  filled_quote?: string | number | null;
  remaining_quantity?: string | number | null;
  status: string;
  exchange_order_id?: string | null;
  sandbox: true;
  error_code?: string | null;
  error_message?: string | null;
  strategy_id?: string | null;
  evaluation_id?: string | null;
  execution_generation?: number | null;
  execution_source?: string | null;
  execution_intent_id?: string | null;
  decision_reason_code?: string | null;
  decision_reason?: string | null;
  decision_conditions?: Array<{
    code?: string;
    label?: string | null;
    category?: string;
    state?: string;
    detail?: string;
    values?: Record<string, unknown>;
  }>;
  filled_at?: string | null;
  created_at: string;
  updated_at: string;
}
